"""
Geo-Tag Master
===============
Bulk geo-tag + keyword/title/description tagging for image batches.

- Upload a ZIP (50-100 images), tag them, download the ZIP back.
- Pixel data is NEVER touched (metadata-only writes via exiftool) -> no quality/size loss.
- Optional AI auto-tagging (Google Gemini, free tier) per image, with
  rate-limit throttling + 429 backoff so large batches don't lose rows.
- AI guidance: give a draft title / seed keywords and AI returns the best
  title, an expanded top-relevance keyword set (cap configurable up to 120),
  and a natural human-style description.
- AI keyword research is current-year aware: titles & tags are generated from
  modern (e.g. 2026) search trends/terminology automatically.
- Star rating: writes a configurable XMP + EXIF rating (default 5 ⭐) to every
  image, visible in Windows Explorer, Lightroom, Bridge, macOS, etc.
- GPS geo-tagging is fully optional (enter both lat & lng, or leave blank).
- Reverse geocoding: GPS coordinates -> city/region/country keywords.
- Supports JPEG, PNG, WebP, TIFF.
- Full Unicode / Bengali keyword support (IPTC CodedCharacterSet=UTF8).
- Abandoned-session temp dirs are swept automatically on app load.

Author: Zaman Arif
"""

import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
APP_TITLE = "🗺️ Geo-Tag Master"
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}

# Safety limits (tune for your Streamlit Cloud plan's RAM/disk)
MAX_FILES = 150                       # reject zips with more images than this
MAX_TOTAL_UNCOMPRESSED = 2 * 1024**3  # 2 GB uncompressed -> zip-bomb guard
MAX_SINGLE_FILE = 200 * 1024**2       # 200 MB per image guard

# Temp-dir handling: every batch lives in a TEMP_PREFIX dir. Abandoned sessions
# (user closes tab without "Start over") are swept on next app load.
TEMP_PREFIX = "imgmeta_"
TEMP_MAX_AGE_HOURS = 2

# Gemini free-tier rate limiting. Flash is ~15 RPM (some models 5 RPM) on the
# free tier, so we throttle proactively AND back off on 429s. The user can
# lower this in the sidebar if they hit limits.
DEFAULT_RPM = 12                      # stay safely under the 15 RPM Flash cap
RATE_LIMIT_MAX_RETRIES = 4
RATE_LIMIT_BASE_DELAY = 5             # seconds; doubles each retry (5,10,20,40)

# Reverse geocoding (OpenStreetMap Nominatim — free, no key). Used ONCE per batch
# since GPS is a common field, so we stay well within their 1 req/sec policy.
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
GEOCODE_USER_AGENT = "Geo-Tag-Master/1.0 (metadata tagging tool)"

# Gemini models that currently expose a free tier. Change the default here if a
# model is deprecated; the user can also pick another from the sidebar.
# gemini-2.5-flash is the default (listed first).
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

# AI keyword cap. The user can dial this in the sidebar (10..MAX_KEYWORDS_CAP).
# Note: most stock platforms cap keywords (~25-50). The higher range is meant for
# your own SEO / owned-site use where a richer keyword set helps.
DEFAULT_MAX_KEYWORDS = 25
MAX_KEYWORDS_CAP = 120


def build_ai_prompt(seed_title: str = "", master_keywords: str = "",
                    location: str = "", keyword_style: str = "",
                    max_keywords: int = DEFAULT_MAX_KEYWORDS) -> str:
    """
    Build the Gemini prompt. The user can steer the output with:

    - seed_title      : a draft/"demo" title (intent/context for the topic).
    - master_keywords : the core services/skills to rank for, e.g.
                        "Wordpress Developer, Shopify Expert, SEO writer".
    - location        : a targeted country/city, e.g. "Houston, TX".
    - keyword_style   : example keyword phrasings the user wants the AI to
                        imitate, e.g. "Best Local SEO Experts in Houston TX,
                        Master Local SEO Experts in USA".

    When master_keywords + location + keyword_style are supplied, the model is
    instructed to behave like an SEO strategist who studied the top-ranking
    Google SERP for those services in that location, and to produce localized,
    search-intent keyword phrases that mirror the demo style (e.g.
    "Best <service> in <city>", "Top <service> near <city>"), blended with the
    image's own visual context.
    """
    seed_title = (seed_title or "").strip()
    master_keywords = (master_keywords or "").strip()
    location = (location or "").strip()
    keyword_style = (keyword_style or "").strip()
    max_keywords = max(1, int(max_keywords))

    guidance = ""
    serp_directive = ""
    if seed_title or master_keywords or location or keyword_style:
        parts = ["\nThe user provided this guidance — treat it as intent/context, "
                 "keep their topic, but improve on it:"]
        if seed_title:
            parts.append(f'- Draft title: "{seed_title}"')
        if master_keywords:
            parts.append(f"- Master keywords (core services/skills to rank for): {master_keywords}")
        if location:
            parts.append(f"- Targeted country/city: {location}")
        if keyword_style:
            parts.append(f"- Demo keyword style (imitate this phrasing pattern): {keyword_style}")
        parts.append("Stay on this topic. Correct anything the image contradicts.")
        guidance = "\n".join(parts) + "\n"

        # Only switch into SERP/localized-keyword mode when there is something
        # concrete to localize or a style to imitate.
        if master_keywords or location or keyword_style:
            serp_directive = (
                "\nKEYWORD STRATEGY: Act as an expert local-SEO strategist who has "
                "studied the top-ranking Google search results (SERP) for the master "
                "keywords above"
                + (f" in {location}" if location else "")
                + ". Generate high-intent, commercial keyword PHRASES that real "
                "top-ranking pages target — not single generic words. "
                + (f"Weave the location ({location}) and its country/region naturally "
                   "into many phrases (e.g. city, 'near me', state, country variants). "
                   if location else "")
                + (f"Closely imitate the user's demo keyword style — produce phrases in "
                   f"the SAME pattern as: {keyword_style}. "
                   if keyword_style else "")
                + "Mix in a few of the exact master keywords and some broader "
                "variations so the set covers the full search funnel. Keep every "
                "keyword relevant to what the image actually shows.\n"
            )

    # Always-on directive: make the AI research and use CURRENT-year search
    # behaviour so titles/keywords reflect how people actually search today,
    # not dated terminology. The year is derived at runtime so this stays
    # future-proof without code changes.
    current_year = time.localtime().tm_year
    modern_directive = (
        f"\nMODERN KEYWORD RESEARCH ({current_year}): Act as if you have studied "
        f"the latest {current_year} search trends for this subject. Generate a "
        "fresh, up-to-date title and keyword set that mirror how real users search "
        f"RIGHT NOW in {current_year} — current trending terminology, natural-language "
        "and voice-search phrasing, and 'near me' / high-intent queries where they "
        "fit. Avoid dated, deprecated, or obsolete tags. The title and keywords must "
        f"feel modern and relevant for {current_year}.\n"
    )

    return (
        "You are a professional stock-photography & SEO metadata expert. "
        "Analyze the image carefully (subject, setting, action, mood, colors, "
        "and any visible context) together with the user's guidance below."
        f"{guidance}{serp_directive}{modern_directive}"
        "Then respond with ONLY a raw JSON object (no markdown, no code fences, "
        "no commentary). Schema:\n"
        '{"title": "<the BEST concise, search-friendly title, max 70 chars>",\n'
        ' "description": "<a natural, human-written description of 1-2 sentences '
        '(max 200 chars). Write like a person, not a keyword list. No stuffing.>",\n'
        f' "keywords": ["<up to {max_keywords} highly relevant, top-ranked SEO '
        "keywords/phrases, ordered most-relevant first; follow the keyword strategy "
        'above when given; no duplicates, no hashtags, no numbering>"]}'
    )

st.set_page_config(page_title="Geo-Tag Master", page_icon="🏷️", layout="wide")


# --------------------------------------------------------------------------- #
# exiftool helpers  (lossless metadata writing)
# --------------------------------------------------------------------------- #
def exiftool_available() -> bool:
    return shutil.which("exiftool") is not None


def build_exiftool_args(meta: dict, ext: str) -> list[str]:
    """
    Build exiftool tag-assignment arguments for one image.

    `meta` keys (all optional): title, description, author, copyright,
    keywords (list[str]), lat (float), lng (float).

    We write to EXIF + IPTC + XMP so the tags are readable by stock sites,
    OS file explorers, and the web. IPTC is skipped for WebP (not a valid
    IPTC container); XMP + EXIF cover it instead. `-m` ignores minor warnings.
    """
    is_webp = ext.lower() == ".webp"
    args: list[str] = []

    title = (meta.get("title") or "").strip()
    desc = (meta.get("description") or "").strip()
    author = (meta.get("author") or "").strip()
    rights = (meta.get("copyright") or "").strip()
    keywords = [k.strip() for k in meta.get("keywords", []) if k and k.strip()]

    if title:
        args += [f"-XMP-dc:Title={title}"]
        if not is_webp:
            args += [f"-IPTC:ObjectName={title}"]

    if desc:
        args += [f"-XMP-dc:Description={desc}", f"-EXIF:ImageDescription={desc}"]
        if not is_webp:
            args += [f"-IPTC:Caption-Abstract={desc}"]

    if author:
        args += [f"-XMP-dc:Creator={author}", f"-EXIF:Artist={author}"]
        if not is_webp:
            args += [f"-IPTC:By-line={author}"]

    if rights:
        args += [f"-XMP-dc:Rights={rights}", f"-EXIF:Copyright={rights}"]
        if not is_webp:
            args += [f"-IPTC:CopyrightNotice={rights}"]

    if keywords:
        # Clear existing list values first, then append (prevents duplicates on re-runs)
        args += ["-XMP-dc:Subject="]
        if not is_webp:
            args += ["-IPTC:Keywords="]
        for kw in keywords:
            args += [f"-XMP-dc:Subject={kw}"]
            if not is_webp:
                args += [f"-IPTC:Keywords={kw}"]

    # Star rating (0-5). Written to BOTH the XMP and EXIF rating fields so it
    # shows up everywhere: Windows Explorer & Photos read the EXIF Rating /
    # RatingPercent (MS) tags, while Lightroom / Bridge / Mac read XMP:Rating.
    # 0 (or missing) means "don't write a rating" — we leave the field untouched.
    rating = meta.get("rating")
    if rating is not None:
        try:
            rating = int(rating)
        except (TypeError, ValueError):
            rating = 0
        if 1 <= rating <= 5:
            rating_percent = {1: 1, 2: 25, 3: 50, 4: 75, 5: 99}[rating]
            args += [
                f"-XMP-xmp:Rating={rating}",
                f"-EXIF:Rating={rating}",
                f"-EXIF:RatingPercent={rating_percent}",
            ]

    lat, lng = meta.get("lat"), meta.get("lng")
    if lat is not None and lng is not None:
        lat_ref = "N" if lat >= 0 else "S"
        lng_ref = "E" if lng >= 0 else "W"
        args += [
            f"-GPSLatitude={abs(lat)}", f"-GPSLatitudeRef={lat_ref}",
            f"-GPSLongitude={abs(lng)}", f"-GPSLongitudeRef={lng_ref}",
        ]

    return args


def write_metadata(path: str, meta: dict) -> tuple[bool, str]:
    """Write metadata in-place, losslessly. Returns (ok, message)."""
    ext = Path(path).suffix
    tag_args = build_exiftool_args(meta, ext)
    if not tag_args:
        return True, "nothing to write"

    cmd = [
        "exiftool",
        "-charset", "iptc=UTF8",          # Bengali / Unicode keywords
        "-codedcharacterset=UTF8",
        "-overwrite_original",            # no *_original backup clutter
        "-m",                             # ignore minor warnings
        *tag_args,
        path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout or "exiftool failed").strip()
        return True, (proc.stdout or "ok").strip()
    except subprocess.TimeoutExpired:
        return False, "exiftool timed out"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


# --------------------------------------------------------------------------- #
# ZIP helpers  (with security guards)
# --------------------------------------------------------------------------- #
def _is_within(directory: str, target: str) -> bool:
    abs_dir = os.path.abspath(directory)
    abs_target = os.path.abspath(target)
    return os.path.commonpath([abs_dir]) == os.path.commonpath([abs_dir, abs_target])


def safe_extract(zip_bytes: bytes, dest: str) -> list[str]:
    """
    Extract image files from a zip with protection against:
      - Zip-slip (path traversal via ../ entries)
      - Zip-bombs (huge uncompressed size)
      - Too many files
    Returns the list of extracted image paths.
    """
    extracted: list[str] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]

        # zip-bomb guard
        total = sum(i.file_size for i in infos)
        if total > MAX_TOTAL_UNCOMPRESSED:
            raise ValueError(
                f"Uncompressed size {total/1024**2:.0f} MB exceeds the "
                f"{MAX_TOTAL_UNCOMPRESSED/1024**2:.0f} MB limit."
            )

        image_infos = [
            i for i in infos
            if Path(i.filename).suffix.lower() in SUPPORTED_EXT
            and not Path(i.filename).name.startswith(".")  # skip __MACOSX etc.
            and "__MACOSX" not in i.filename
        ]
        if not image_infos:
            raise ValueError("No supported images (JPEG/PNG/WebP/TIFF) found in the zip.")
        if len(image_infos) > MAX_FILES:
            raise ValueError(f"Zip has {len(image_infos)} images; limit is {MAX_FILES}.")

        for info in image_infos:
            if info.file_size > MAX_SINGLE_FILE:
                continue  # skip oversized single file
            # Flatten to basename to neutralise any path traversal entirely
            safe_name = os.path.basename(info.filename)
            out_path = os.path.join(dest, safe_name)
            # collision handling
            stem, suf = os.path.splitext(safe_name)
            n = 1
            while os.path.exists(out_path):
                out_path = os.path.join(dest, f"{stem}_{n}{suf}")
                n += 1
            if not _is_within(dest, out_path):
                continue  # paranoia guard
            with zf.open(info) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(out_path)

    extracted.sort(key=lambda p: os.path.basename(p).lower())
    return extracted


def repack(image_dir: str) -> bytes:
    """Zip up all images in image_dir (stored, no recompression of pixels)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(os.listdir(image_dir)):
            fp = os.path.join(image_dir, name)
            if os.path.isfile(fp) and Path(fp).suffix.lower() in SUPPORTED_EXT:
                zf.write(fp, arcname=name)
    buf.seek(0)
    return buf.read()


# --------------------------------------------------------------------------- #
# AI auto-tagging  (Google Gemini) with rate-limit handling
# --------------------------------------------------------------------------- #
def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect Gemini free-tier 429 / quota-exhausted errors across SDK versions."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(s in text for s in ("429", "resourceexhausted", "rate limit",
                                   "quota", "exceeded", "too many requests"))


def analyze_image(path: str, model_name: str, api_key: str,
                  seed_title: str = "", master_keywords: str = "",
                  location: str = "", keyword_style: str = "",
                  max_keywords: int = DEFAULT_MAX_KEYWORDS) -> dict:
    """
    Return {'title', 'description', 'keywords'} from Gemini.

    Optional seed_title / master_keywords / location / keyword_style steer the
    output toward the user's intended topic and localized SEO phrasing;
    max_keywords caps how many keywords are returned.

    Retries with exponential backoff on rate-limit (429) errors so a 50-100
    image batch doesn't lose rows the moment it crosses the per-minute cap.
    Raises on non-rate-limit failures (caller counts those as row failures).
    """
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    img = Image.open(path)
    img.load()  # force read so the file handle can close

    prompt = build_ai_prompt(seed_title, master_keywords, location,
                             keyword_style, max_keywords)

    last_exc = None
    for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
        try:
            resp = model.generate_content([prompt, img])
            raw = (resp.text or "").strip()
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
            data = json.loads(raw)
            # De-dupe (case-insensitive) and hard-cap the keyword list so the
            # model can never blow past the user's chosen maximum.
            seen, kws = set(), []
            for k in data.get("keywords", []):
                k = str(k).strip().lstrip("#").strip()
                if k and k.lower() not in seen:
                    seen.add(k.lower())
                    kws.append(k)
            return {
                "title": str(data.get("title", "")).strip(),
                "description": str(data.get("description", "")).strip(),
                "keywords": kws[:max(1, int(max_keywords))],
            }
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_rate_limit_error(exc) and attempt < RATE_LIMIT_MAX_RETRIES:
                time.sleep(RATE_LIMIT_BASE_DELAY * (2 ** attempt))
                continue
            raise
    raise last_exc  # pragma: no cover


# --------------------------------------------------------------------------- #
# Reverse geocoding  (lat/lng -> place-name keywords, via OpenStreetMap)
# --------------------------------------------------------------------------- #
def reverse_geocode(lat: float, lng: float, timeout: int = 10) -> list[str]:
    """
    Turn coordinates into location keywords (city, region, country, etc.).
    One HTTP call per batch. Returns [] on any failure — geocoding is a
    nice-to-have and must never block tagging.
    """
    params = urllib.parse.urlencode(
        {"lat": lat, "lon": lng, "format": "jsonv2", "zoom": 12,
         "addressdetails": 1, "accept-language": "en"}
    )
    req = urllib.request.Request(
        f"{NOMINATIM_URL}?{params}", headers={"User-Agent": GEOCODE_USER_AGENT}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return []

    addr = data.get("address", {}) or {}
    # Pull the useful, human-meaningful fields in priority order
    ordered = [
        addr.get("city") or addr.get("town") or addr.get("village")
        or addr.get("municipality"),
        addr.get("suburb") or addr.get("neighbourhood"),
        addr.get("county"),
        addr.get("state") or addr.get("region"),
        addr.get("country"),
    ]
    seen, keywords = set(), []
    for kw in ordered:
        if kw and kw.lower() not in seen:
            seen.add(kw.lower())
            keywords.append(kw)
    return keywords


# --------------------------------------------------------------------------- #
# Session helpers
# --------------------------------------------------------------------------- #
def sweep_old_temp_dirs(max_age_hours: int = TEMP_MAX_AGE_HOURS) -> int:
    """
    Remove batch temp dirs left behind by abandoned sessions (user closed the
    tab without clicking 'Start over'). Runs once per session on app load.
    Returns the count removed.
    """
    root = tempfile.gettempdir()
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    try:
        for name in os.listdir(root):
            if not name.startswith(TEMP_PREFIX):
                continue
            path = os.path.join(root, name)
            try:
                if os.path.isdir(path) and os.path.getmtime(path) < cutoff:
                    shutil.rmtree(path, ignore_errors=True)
                    removed += 1
            except OSError:
                continue
    except OSError:
        pass
    return removed


def reset_session():
    old = st.session_state.get("work_dir")
    if old and os.path.isdir(old):
        shutil.rmtree(old, ignore_errors=True)
    for k in ("work_dir", "df", "image_paths", "result_zip", "report",
              "geo_keywords", "geo_resolved_for"):
        st.session_state.pop(k, None)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def main():
    # One-time per session: clean up temp dirs from abandoned sessions.
    if not st.session_state.get("_swept"):
        sweep_old_temp_dirs()
        st.session_state._swept = True

    st.title(APP_TITLE)
    st.caption(
        "Bulk geo-tag + keyword/title tagging for image batches. "
        "Pixel data is never re-encoded — original quality and size are preserved."
    )

    if not exiftool_available():
        st.error(
            "**exiftool is not installed.** On Streamlit Cloud, add a file named "
            "`packages.txt` containing `libimage-exiftool-perl`. Locally, run "
            "`sudo apt install libimage-exiftool-perl` (or `brew install exiftool`)."
        )
        st.stop()

    # ---- Sidebar -------------------------------------------------------- #
    with st.sidebar:
        st.header("⚙️ Settings")
        use_ai = st.toggle("Enable AI auto-tagging", value=True)
        api_key = ""
        model_name = GEMINI_MODELS[0]
        max_keywords = DEFAULT_MAX_KEYWORDS
        if use_ai:
            # API key is read ONLY from Streamlit secrets (Settings → Secrets):
            #   GEMINI_API_KEY = "your-key"
            # No key input is shown in the UI.
            try:
                api_key = st.secrets.get("GEMINI_API_KEY", "")
            except Exception:  # noqa: BLE001 — secrets file may be absent locally
                api_key = ""
            if not api_key:
                st.warning(
                    "No Gemini key found. Add `GEMINI_API_KEY` in **Settings → "
                    "Secrets** (Streamlit Cloud) or in `.streamlit/secrets.toml` "
                    "locally to enable AI tagging."
                )
            model_name = st.selectbox("Gemini model", GEMINI_MODELS)
            max_keywords = st.slider(
                "Max keywords per image", min_value=10, max_value=MAX_KEYWORDS_CAP,
                value=DEFAULT_MAX_KEYWORDS,
                help="How many keywords AI generates per image. Tip: most stock "
                     "sites cap at ~25-50; go higher only for your own SEO use.",
            )
            rpm = st.slider(
                "Max requests / minute", min_value=3, max_value=60,
                value=DEFAULT_RPM,
                help="Free Gemini Flash allows ~15 RPM. Lower this if you see "
                     "rate-limit errors; raise it on a paid key.",
            )
        st.divider()
        # Star rating written to every image (default 5 ⭐). Shows up in Windows
        # Explorer, Lightroom, Bridge, macOS, etc. 0 = don't write a rating.
        star_rating = st.select_slider(
            "⭐ Star rating (every image)",
            options=[0, 1, 2, 3, 4, 5],
            value=5,
            help="Writes an XMP + EXIF star rating to every image. 5 = ⭐⭐⭐⭐⭐. "
                 "Set 0 to skip writing a rating.",
        )
        st.divider()
        if st.button("🔄 Start over", use_container_width=True):
            reset_session()
            st.rerun()
        st.caption(f"Limits: ≤ {MAX_FILES} images, "
                   f"≤ {MAX_TOTAL_UNCOMPRESSED // 1024**2} MB uncompressed.")

    # ---- Step 1: Upload ------------------------------------------------- #
    st.subheader("1. Upload image ZIP")
    upload = st.file_uploader("Drop a .zip of images", type=["zip"], key="uploader")

    if upload and "image_paths" not in st.session_state:
        with st.spinner("Extracting & validating…"):
            try:
                work_dir = tempfile.mkdtemp(prefix="imgmeta_")
                paths = safe_extract(upload.getvalue(), work_dir)
            except Exception as exc:  # noqa: BLE001
                st.error(f"❌ {exc}")
                st.stop()
            st.session_state.work_dir = work_dir
            st.session_state.image_paths = paths
            st.session_state.df = pd.DataFrame(
                {
                    "file": [os.path.basename(p) for p in paths],
                    "title": "",
                    "keywords": "",       # comma-separated in the editor
                    "description": "",
                }
            )
        st.rerun()

    if "image_paths" not in st.session_state:
        st.info("Upload a zip to begin.")
        return

    paths = st.session_state.image_paths
    st.success(f"✅ {len(paths)} images ready.")

    with st.expander("Preview thumbnails", expanded=False):
        cols = st.columns(6)
        for i, p in enumerate(paths[:18]):
            with cols[i % 6]:
                try:
                    st.image(p, caption=os.path.basename(p), use_container_width=True)
                except Exception:
                    st.text(os.path.basename(p))
        if len(paths) > 18:
            st.caption(f"…and {len(paths) - 18} more")

    # ---- Step 2: Common metadata --------------------------------------- #
    st.subheader("2. Common metadata (applied to every image)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        author = st.text_input("Author / Creator", placeholder="Your name")
    with c2:
        copyright_ = st.text_input("Copyright", placeholder="© 2026 Digital Zeon")
    with c3:
        lat_str = st.text_input("GPS Latitude (optional)", placeholder="e.g. 23.8103")
    with c4:
        lng_str = st.text_input("GPS Longitude (optional)", placeholder="e.g. 90.4125")

    # GPS is fully optional. It's only written when BOTH fields hold valid
    # numbers. A blank or half-filled pair simply skips geo-tagging — it never
    # blocks the batch.
    lat = lng = None
    lat_in, lng_in = lat_str.strip(), lng_str.strip()
    if lat_in or lng_in:
        if lat_in and lng_in:
            try:
                _lat, _lng = float(lat_in), float(lng_in)
                if -90 <= _lat <= 90 and -180 <= _lng <= 180:
                    lat, lng = _lat, _lng
                else:
                    st.caption("⚠️ GPS out of range (lat −90..90, lng −180..180) — "
                               "skipping geo-tag.")
            except ValueError:
                st.caption("⚠️ GPS values must be decimal numbers — skipping geo-tag.")
        else:
            st.caption("ℹ️ Enter *both* latitude and longitude to geo-tag — "
                       "skipping GPS for now.")

    # --- Reverse geocoding: GPS -> location keywords (added to every image) --- #
    geo_keywords = st.session_state.get("geo_keywords", [])
    if lat is not None and lng is not None:
        add_loc = st.checkbox(
            "🌍 Add location keywords from GPS (city, region, country)", value=False
        )
        if add_loc:
            coord_key = f"{lat:.5f},{lng:.5f}"
            if st.session_state.get("geo_resolved_for") != coord_key:
                with st.spinner("Looking up location…"):
                    geo_keywords = reverse_geocode(lat, lng)
                st.session_state.geo_keywords = geo_keywords
                st.session_state.geo_resolved_for = coord_key
            if geo_keywords:
                st.success("Location keywords: " + ", ".join(geo_keywords))
            else:
                st.warning("Couldn't resolve a location for those coordinates.")
        else:
            geo_keywords = []
            st.session_state.pop("geo_keywords", None)
            st.session_state.pop("geo_resolved_for", None)

    # ---- Step 3: Per-image metadata + AI ------------------------------- #
    st.subheader("3. Per-image metadata")
    seed_title = master_keywords = location = keyword_style = ""
    if use_ai:
        with st.expander("🎯 AI guidance (optional) — steer titles, keywords & descriptions"):
            st.caption(
                "Steer the AI toward localized, SERP-style SEO keywords. Give your "
                "core services as **Master Keywords**, a **Targeted Country/City**, "
                "and a **Demo Keyword Style** to imitate. The AI studies the intent "
                "of top-ranking results and writes the best title, localized keyword "
                "phrases in your style, and a natural description per image. Leave "
                "blank to let AI work from the image alone."
            )
            seed_title = st.text_input(
                "Draft / demo title",
                placeholder="e.g. Professional WordPress development services",
            )
            master_keywords = st.text_input(
                "Master Keywords (comma-separated)",
                placeholder="e.g. Wordpress Developer, Shopify Expert, Semantic Content Writer",
                help="Core services/skills you want to rank for. The AI keeps these "
                     "as the topic and expands them into localized SEO phrases.",
            )
            location = st.text_input(
                "Targeted Country/City",
                placeholder="e.g. Houston, TX",
                help="Where you want to rank. The AI weaves this city/state/country "
                     "into the keyword phrases (e.g. 'in Houston TX', 'near me', 'USA').",
            )
            keyword_style = st.text_input(
                "Demo Keyword Style (comma-separated examples)",
                placeholder="e.g. Best Local SEO Experts in Houston TX, Master Local SEO Experts in USA",
                help="Example phrasings to imitate. The AI generates new keywords in "
                     "this same pattern, localized to your services and city.",
            )
        if st.button("🤖 Generate tags with AI", type="secondary"):
            if not api_key:
                st.error("No Gemini API key configured. Add `GEMINI_API_KEY` to "
                         "Streamlit secrets (Settings → Secrets) to enable AI tagging.")
            else:
                df = st.session_state.df
                delay = 60.0 / max(rpm, 1)   # spacing to respect RPM
                progress = st.progress(0.0, text="Analyzing images…")
                failures = 0
                for idx, p in enumerate(paths):
                    if idx > 0:
                        time.sleep(delay)   # proactive throttle (backoff handles 429s)
                    try:
                        out = analyze_image(
                            p, model_name, api_key,
                            seed_title=seed_title,
                            master_keywords=master_keywords,
                            location=location,
                            keyword_style=keyword_style,
                            max_keywords=max_keywords,
                        )
                        df.at[idx, "title"] = out["title"]
                        df.at[idx, "keywords"] = ", ".join(out["keywords"])
                        df.at[idx, "description"] = out["description"]
                    except Exception:  # noqa: BLE001
                        failures += 1
                    progress.progress((idx + 1) / len(paths),
                                      text=f"Analyzing {idx+1}/{len(paths)}…")
                st.session_state.df = df
                progress.empty()
                if failures:
                    st.warning(f"AI finished with {failures} failure(s) — "
                               "edit those rows manually below.")
                else:
                    st.success("AI tagging complete. Review & edit below.")

    st.caption("Edit any cell. Keywords are comma-separated.")
    edited = st.data_editor(
        st.session_state.df,
        use_container_width=True,
        num_rows="fixed",
        disabled=["file"],
        column_config={
            "file": st.column_config.TextColumn("File", width="medium"),
            "title": st.column_config.TextColumn("Title"),
            "keywords": st.column_config.TextColumn("Keywords (comma-separated)", width="large"),
            "description": st.column_config.TextColumn("Description", width="large"),
        },
        key="editor",
    )
    st.session_state.df = edited

    # ---- Step 4: Process & download ------------------------------------ #
    st.subheader("4. Apply & download")
    if st.button("💾 Write metadata & build ZIP", type="primary"):
        df = st.session_state.df
        loc_keywords = st.session_state.get("geo_keywords", [])
        progress = st.progress(0.0, text="Writing metadata…")
        report = []
        for idx, p in enumerate(paths):
            row = df.iloc[idx]
            per_image = [k.strip() for k in str(row["keywords"]).split(",") if k.strip()]
            # merge location keywords without duplicates (case-insensitive)
            merged, seen = [], set()
            for k in per_image + loc_keywords:
                if k.lower() not in seen:
                    seen.add(k.lower())
                    merged.append(k)
            meta = {
                "title": row["title"],
                "description": row["description"],
                "keywords": merged,
                "author": author,
                "copyright": copyright_,
                "lat": lat,
                "lng": lng,
                "rating": star_rating,
            }
            ok, msg = write_metadata(p, meta)
            report.append({"file": row["file"], "status": "✅" if ok else "❌",
                           "detail": "" if ok else msg})
            progress.progress((idx + 1) / len(paths), text=f"Writing {idx+1}/{len(paths)}…")
        progress.empty()

        st.session_state.result_zip = repack(st.session_state.work_dir)
        st.session_state.report = pd.DataFrame(report)

    if "result_zip" in st.session_state:
        rep = st.session_state.report
        ok_n = (rep["status"] == "✅").sum()
        st.success(f"Done — {ok_n}/{len(rep)} images tagged successfully.")
        st.download_button(
            "⬇️ Download tagged ZIP",
            data=st.session_state.result_zip,
            file_name="tagged_images.zip",
            mime="application/zip",
            type="primary",
        )
        if (rep["status"] == "❌").any():
            with st.expander("Show errors"):
                st.dataframe(rep[rep["status"] == "❌"], use_container_width=True)


if __name__ == "__main__":
    main()
