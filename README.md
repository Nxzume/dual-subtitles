# Dual Subtitles

Build dual-language subtitle files for films and TV — translate with an [NVIDIA NIM](https://build.nvidia.com) LLM, or merge two existing tracks by time.

| File | Role |
|---|---|
| `dual_subs.py` | App (CLI + desktop UI) |
| `setup.bat` | Creates `.venv` and installs dependencies (Windows) |
| `Dual Subs UI.bat` | Launches the UI from `.venv` |
| `requirements.txt` | Python packages |
| `.env.example` | Template for `NVIDIA_API_KEY` |
| `test_dual_subs.py` | Unit tests |

**Input:** `.srt` / `.vtt` / `.ass` / `.ssa`  
**Output** (next to the source file):

| File | Contents |
|---|---|
| `movie.dual.srt` | Both languages (default: stacked — two lines per cue) |
| `movie.en.srt` | Source language only (translate mode) |
| `movie.zh-CN.srt` | Target language only (translate mode) |

## Requirements

- Python 3.10+
- An NVIDIA API key ([build.nvidia.com](https://build.nvidia.com) — free tier, no credit card) — **only for Translate (AI)**
- Tkinter (usually included with Python) — **only for the desktop UI**

## Setup

### Windows (recommended)

1. Double-click `setup.bat` — creates `.venv`, installs packages, copies `.env.example` → `.env` if needed.
2. Edit `.env` and set your key:

```
NVIDIA_API_KEY=nvapi-...
```

3. Double-click `Dual Subs UI.bat` to open the app.

### Manual

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # Windows: copy .env.example .env
```

Optional env overrides in `.env`:

```
NVIDIA_MODEL=qwen/qwen2.5-72b-instruct
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
```

## Desktop UI

```bash
python dual_subs.py
python dual_subs.py --ui
python dual_subs.py --ui movie.srt          # prefill Subtitle
python dual_subs.py --ui en.srt zh.srt      # prefill Merge (Subtitle 1 + 2)
```

On Windows, prefer `Dual Subs UI.bat` after `setup.bat` so the venv is used.

### Modes

| Mode | What it does | API key |
|---|---|---|
| **Translate (AI)** | Translate one subtitle file into a dual file | Required |
| **Merge two files** | Fuse Subtitle 1 + Subtitle 2 by time overlap | Not needed |
| **Edit** | Full cue editor (text, times, add/delete) → Save / Save as | Not needed |

### Options

- **Source lang** — `auto` (detect from text) or a language code
- **Target lang** — dropdown (zh-CN, zh-TW, en, ja, ko, …)
- **Line order** — `source-top` or `target-top`
- **Dual format** — `srt` or `ass`
- **Dual layout** — `stacked` (two lines per cue, default) or `single-line` (`ZH \| EN`)
- **Model** — NIM model picker (default `qwen/qwen2.5-72b-instruct`)
- **Context** — optional show/movie notes for the translator

**Merge sync** (Merge mode only):

- **Auto-shift** — estimate a global sync offset when tracks are misaligned
- **Shift ms (Subtitle 2)** — manual offset applied to Subtitle 2 before spine selection
- **Drop unmatched** — omit Subtitle 2 cues that don’t overlap anything

### Edit mode

Load any `.srt` / `.vtt` / `.ass` / `.ssa` and edit cues in place:

- Cue list with start, end, and text
- Detail panel to change times and multiline text (including dual/stacked lines)
- **Add cue** / **Delete cue**
- **Save** (overwrite) or **Save as…**
- **Reload** from disk (warns if you have unsaved changes)

After a successful Translate or Merge, **Edit result** opens the dual output file in Edit mode so you can fix lines by hand before shipping.

### Preview & run

- Preview always shows cue text/timings for the loaded file(s) (Translate / Merge).
- **Live AI sample preview** (off by default) — translates ~8 cues with the selected model so you can check quality before a full run. This is the only preview action that calls the API.
- **Show video preview** (off by default) — compact on-screen mockup of dual lines; use Prev/Next cue to step through.
- **Run** starts Translate/Merge; **Cancel** stops an in-progress translation; the progress bar tracks batches.
- **Edit result** opens the last dual output in the editor.
- **Open output folder** opens the input file’s folder (Windows / macOS / Linux).
- Progress details also appear on the **Log** tab.

## CLI

```bash
# Translate
python dual_subs.py movie.srt
python dual_subs.py movie.srt --target-lang zh-TW --order target-top
python dual_subs.py movie.srt --context "The Amazing Spider-Man (2012), casual teen dialogue"
python dual_subs.py movie.srt --model qwen/qwen2.5-7b-instruct

# Merge (no API key)
python dual_subs.py --merge movie.en.srt movie.zh.srt
python dual_subs.py --merge movie.en.srt movie.zh.srt -o movie.dual.srt
python dual_subs.py --merge movie.en.srt movie.zh.srt --order target-top
python dual_subs.py --merge a.srt b.srt --shift-ms -400 --auto-shift
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--source-lang` | `auto` | Source language, or `auto` to detect from text |
| `--target-lang` | `zh-CN` | Target (`zh-CN`, `zh-TW`, …) |
| `--order` | `source-top` | Line order: `source-top` or `target-top` |
| `--format` | `srt` | Dual output: `srt` or `ass` |
| `--layout` | `stacked` | `stacked` or `single-line` |
| `--model` | `qwen/qwen2.5-72b-instruct` | NIM model id ([catalog](https://build.nvidia.com/models)) |
| `--batch-size` | `20` | Cues per API request |
| `--workers` | `6` | Parallel API requests |
| `--merge FILE_1 FILE_2` | — | Fuse two tracks into dual (time-sync, no API) |
| `-o` / `--output` | auto | Output path for `--merge` |
| `--auto-shift` | off | Estimate a global sync offset if tracks are misaligned |
| `--shift-ms` | `0` | Manual offset (ms) for **FILE_2 / Subtitle 2** (before spine selection) |
| `--min-overlap-ms` | `80` | Minimum overlap to pair cues when merging |
| `--drop-unmatched` | off | Drop FILE_2 cues that don’t overlap anything |
| `--context` | _(none)_ | Movie/show notes for the translator |
| `--ui` | off | Launch the desktop UI |

CLI exits with status `1` if any input fails.

## How it works

**Translate**
1. Loads cues from a subtitle file.
2. Detects source language when set to `auto`.
3. Skips empty cues and deduplicates identical lines to save API calls.
4. Sends numbered batches to NVIDIA NIM in parallel (with retries / rate-limit backoff).
5. Restores multi-line cues and writes dual + single-language files.

**Merge**
1. Applies manual `--shift-ms` to FILE_2, then picks a timing spine (prefers Latin when pairing with CJK).
2. Optionally auto-shifts the secondary track.
3. Pairs cues by time overlap and writes a dual file.

Default model: `qwen/qwen2.5-72b-instruct`. The older `qwen/qwen3.5-397b-a17b` remains selectable in the UI / via `--model`. Override the default globally with `NVIDIA_MODEL` in `.env`.

## Tests

```bash
python -m unittest test_dual_subs.py -v
```

## Notes

- Timing and cue structure are preserved; inline ASS styling is not carried into dual output.
- A ~90 minute movie (~1000–1500 cues) typically finishes in a few minutes with the default parallel settings.
- Never commit your `.env` file — it is gitignored.

## Jellyfin Web (Chinese as ☐☐☐ boxes)

Dual files use **stacked** layout by default (two lines in one cue). If Chinese shows as empty boxes while English is fine, that is **not** a bad `.srt` — Jellyfin Web is missing a CJK font. Jellyfin Web may also only show one of the two stacked lines; if so, use burn-in or two separate tracks.

### Fix 1 — Fallback fonts (best)

1. Download a light CJK web font, e.g. [Noto Sans SC (woff2)](https://github.com/CodePlayer/webfont-noto) or from [Google Fonts](https://fonts.google.com/noto/specimen/Noto+Sans+SC).
2. Jellyfin: **Dashboard → Playback → Fallback fonts** — enable and point at a folder of `.woff2` / `.ttf` (folder size limit ~20 MB).
3. Restart Jellyfin / hard-refresh the web client, then play the `.dual.srt`.

Docs: [Fallback fonts](https://jellyfin.org/docs/general/administration/configuration#fallback-fonts) · [Text not rendering](https://jellyfin.org/docs/general/administration/troubleshooting#text-not-rendering-properly)

### Fix 2 — Burn subtitles

User settings → Subtitles → **Burn subtitles** = All (or complex formats). Uses server fonts (install CJK fonts on the host). Forces transcoding.

### Fix 3 — Two tracks

Keep `movie.en.srt` + `movie.zh.srt` and use Jellyfin’s primary + secondary subtitle controls.

### Recommended output

```bash
python dual_subs.py --merge en.srt zh.srt --format srt --layout stacked
```

Or in the UI: **Merge two files**, format `srt`, layout `stacked`. Name the file like the video (e.g. `Movie Name (2021).srt`), place it next to the media, then scan the library.
