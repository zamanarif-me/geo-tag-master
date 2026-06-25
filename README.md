# 🗺️ Geo-Tag Master

Bulk **geo-tag + keyword / title / description** tagging for image batches.
Upload a ZIP of 50–100 images, tag them (manually or with AI), download the ZIP back.

**Pixel data is never re-encoded** — original image quality and resolution are preserved exactly. Only the metadata segments are written (via `exiftool`).

---

## Features

- 📦 ZIP in → ZIP out (50–100 images per batch)
- 🗺️ GPS geo-tagging — **fully optional** (enter both lat & lng, or leave blank)
- 🌍 Reverse geocoding — GPS → city / region / country keywords (one OSM lookup per batch)
- 🏷️ Keywords, Title, Description, Author, Copyright
- 🤖 Optional AI auto-tagging per image (Google Gemini free tier)
- 🎯 AI guidance — feed a draft title, **Master Keywords**, a **Targeted
  Country/City**, and a **Demo Keyword Style** to imitate; AI returns the best
  title, localized SERP-style keyword phrases (cap **up to 120**), and a
  human-style description
- 🚦 Rate-limit safe — proactive RPM throttle + exponential backoff on 429s
- ✏️ Editable table — review/override every AI suggestion
- 🌐 Full Unicode / **Bengali** keyword support (IPTC `CodedCharacterSet=UTF8`)
- 🖼️ Formats: **JPEG, PNG, WebP, TIFF**
- 🔒 Hardened: zip-slip, zip-bomb, file-count & size guards
- 🧹 Abandoned-session temp dirs swept automatically on app load
- 💾 Lossless — verified pixel-identical output

Metadata is written to **EXIF + IPTC + XMP** so it's readable by stock sites
(Adobe Stock, Shutterstock, Getty), OS file explorers, and the web.
WebP gets XMP + EXIF only (it has no valid IPTC container).

---

## Deploy on Streamlit Cloud

1. Push these files to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select the repo, `app.py`.
3. For AI: **Settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your-google-ai-studio-key"
   ```
   Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

`packages.txt` installs `exiftool` automatically. Nothing else to configure.

> The Gemini API key is read **only** from Streamlit secrets — there is no key
> field in the UI. Without the secret, AI tagging is disabled; you can still
> tag manually.

---

## Run locally

```bash
# 1. exiftool
sudo apt install libimage-exiftool-perl      # Debian/Ubuntu
# brew install exiftool                        # macOS

# 2. Python deps
pip install -r requirements.txt

# 3. (optional) AI key
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then edit it

# 4. run
streamlit run app.py
```

---

## How it works

```
ZIP upload
   └─ safe_extract()   zip-slip / zip-bomb / count guards, basename flatten
        └─ thumbnails + editable table
             ├─ Common fields (GPS, Author, Copyright) → all images
             └─ AI / manual (Title, Keywords, Description) → per image
                  └─ write_metadata()   exiftool, lossless, UTF-8
                       └─ repack() → download
```

---

## Configuration

Edit the constants at the top of `app.py`:

| Constant | Meaning | Default |
|---|---|---|
| `MAX_FILES` | max images per zip | 150 |
| `MAX_TOTAL_UNCOMPRESSED` | zip-bomb ceiling | 2 GB |
| `MAX_SINGLE_FILE` | per-image ceiling | 200 MB |
| `DEFAULT_RPM` | AI requests/min throttle | 12 |
| `RATE_LIMIT_MAX_RETRIES` | 429 backoff retries | 4 |
| `TEMP_MAX_AGE_HOURS` | abandoned-dir sweep age | 2 |
| `GEMINI_MODELS` | selectable AI models (first = default) | `gemini-2.5-flash`, … |

`.streamlit/config.toml` raises the upload limit to 1 GB.

### Rate limiting

Free Gemini Flash allows ~15 requests/minute. The app throttles to `DEFAULT_RPM`
(12) between calls and retries with exponential backoff (5 → 10 → 20 → 40 s) on
any 429 / quota error, so a 100-image batch completes instead of failing mid-way.
A 100-image AI run therefore takes roughly 8–9 minutes on the free tier — raise
the RPM slider (and `DEFAULT_RPM`) on a paid key to go faster.

### Reverse geocoding

When you enter GPS coordinates, tick **"Add location keywords from GPS"** to
resolve city / region / country via OpenStreetMap Nominatim (free, no key) and
append them to every image's keywords. One lookup per batch keeps it well within
Nominatim's usage policy.

---

## Notes & caveats

- **Streamlit Cloud RAM (free tier ≈ 1 GB).** 100 very large images decoded for
  thumbnails/AI can hit the limit — thumbnails are capped at 18 to help. For huge
  batches, raise the plan or process in smaller zips.
- **Stock sites & WebP.** Most stock platforms want JPEG/TIFF, not WebP. WebP
  tagging works fine for web/client delivery.
- **AI model names change.** If `gemini-2.5-flash` is deprecated, pick another
  from the sidebar dropdown or edit `GEMINI_MODELS`.
- **Speed.** Metadata is written one image per `exiftool` call. For very large
  batches you can switch to `exiftool -stay_open` batch mode for a big speedup.

---

Created by **Zaman Arif** · © 2026 Digital Zeon
