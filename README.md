# Dual Subtitles

Build dual-language subtitle files for films and TV — translate with an [NVIDIA NIM](https://build.nvidia.com) LLM, merge two existing tracks by time, or extract soft subs from video.

**Input:** `.srt` / `.vtt` / `.ass` / `.ssa`, or a video with soft subtitle tracks (`.mkv`, `.mp4`, …)  
**Output** (next to the source file, names depend on mode):

| File | Contents |
|---|---|
| `movie.dual.srt` | Both languages (default: overlap layout for Jellyfin Web) |
| `movie.en.srt` | Source language only (translate mode) |
| `movie.zh-CN.srt` | Target language only (translate mode) |

## Requirements

- Python 3.10+
- An NVIDIA API key ([build.nvidia.com](https://build.nvidia.com) — free tier, no credit card) — **only for Translate (AI)**
- Optional: [ffmpeg](https://ffmpeg.org) on PATH (extract soft tracks from video)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # on Windows: copy .env.example .env
```

Edit `.env` and set your key:

```
NVIDIA_API_KEY=nvapi-...
```

## Desktop UI

Double-click `Dual Subs UI.bat`, or run:

```bash
python ui.py
```

### Modes

| Mode | What it does | API key |
|---|---|---|
| **Translate (AI)** | Translate a subtitle (or soft track from video) into a dual file | Required |
| **Merge two files** | Fuse two existing language files by time overlap | Not needed |
| **Extract from video** | Dump a soft text track to `.srt` | Not needed |

### Options

- **Source lang** — `auto` (detect from text) or a language code
- **Target lang** — dropdown (zh-CN, zh-TW, en, ja, ko, …)
- **Line order** — `source-top` or `target-top`
- **Dual format** — `srt` (recommended for Jellyfin Web) or `ass`
- **Dual layout** — `overlap` (best for Jellyfin Web), `stacked`, or `single-line`
- **Context** — optional show/movie notes for the translator

### Preview

The Preview tab lists cues and timings after you pick a file. In Translate mode it also runs a short live sample (~8 cues) so you can check quality before a full run.

Check **Show video preview** if you want a compact on-screen mockup of how dual lines look (off by default). Use **Prev cue** / **Next cue** to step through the sample.

**Refresh preview** reloads the list/sample; **Open output folder** opens the folder of the input file; progress appears on the **Log** tab.

## Drag & drop

Drop a subtitle or video onto `Drag Subtitles Here.bat` for a quick translate run (same as the CLI defaults).

## CLI

```bash
# Subtitle file → dual subs (AI translate)
python dual_subs.py movie.srt

# Video with soft tracks → extract preferred track → dual subs
python dual_subs.py movie.mkv

# Extract only (no translation)
python dual_subs.py movie.mkv --extract-only

# Pick a specific soft track (0-based)
python dual_subs.py movie.mkv --sub-stream 1

# Fuse two existing language files (no API key) — pair cues by time overlap
python dual_subs.py --merge movie.en.srt movie.zh.srt
python dual_subs.py --merge movie.en.srt movie.zh.srt -o movie.dual.srt
python dual_subs.py --merge movie.en.srt movie.zh.srt --order target-top

# Traditional Chinese, Chinese line on top
python dual_subs.py movie.srt --target-lang zh-TW --order target-top

# Extra context for better tone / names
python dual_subs.py movie.srt --context "The Amazing Spider-Man (2012), casual teen dialogue"
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--source-lang` | `auto` | Source language, or `auto` to detect from text |
| `--target-lang` | `zh-CN` | Target (`zh-CN` Simplified, `zh-TW` Traditional, …) |
| `--order` | `source-top` | Line order: `source-top` or `target-top` |
| `--format` | `srt` | Dual output: `srt` (use this for Jellyfin Web) or `ass` |
| `--layout` | `overlap` | `overlap` (two cues, same time — best for Jellyfin Web), `stacked`, or `single-line` |
| `--model` | `qwen/qwen3.5-397b-a17b` | NIM model id ([catalog](https://build.nvidia.com/models)) |
| `--batch-size` | `20` | Cues per API request |
| `--workers` | `6` | Parallel API requests |
| `--sub-stream` | auto | Soft track index when input is a video |
| `--extract-only` | off | Extract soft subs only |
| `--merge A B` | — | Fuse two existing tracks into dual (time-sync, no API) |
| `-o` / `--output` | auto | Output path for `--merge` |
| `--auto-shift` | off | Only if tracks are globally misaligned: estimate a sync offset |
| `--shift-ms` | `0` | Manual offset (ms) for the second merge track |
| `--min-overlap-ms` | `80` | Minimum overlap to pair cues when merging |
| `--drop-unmatched` | off | Drop second-file cues that don't overlap anything |
| `--context` | _(none)_ | Movie/show notes for the translator |

## How it works

**Translate mode**
1. Loads cues (or extracts a soft text track from video via `ffmpeg`).
2. Detects source language when set to `auto`.
3. Sends numbered batches of lines to NVIDIA NIM in parallel.
4. Reattaches translations to the original timings.
5. Writes dual + single-language files.

**Merge mode** (`--merge` / UI “Merge two files”)
1. Detects script family (Latin vs CJK) to pick a timing spine.
2. Pairs cues by time overlap and writes a dual file.
3. Assumes both files are already timed to the same video (optional `--auto-shift` / `--shift-ms` only if not).

**Extract mode**
1. Uses `ffmpeg` to dump a soft text track to `.srt`.

Defaults to a large Qwen model for strong bilingual quality. Smaller/faster models can be set with `--model`.

Text soft tracks (`srt`, `ass`, `mov_text`, …) extract cleanly. Image-based tracks (`PGS`, `VobSub`) need OCR and are not supported.

## Notes

- Timing and cue structure are preserved; inline styling on translated lines is not.
- A ~90 minute movie (~1000–1500 cues) typically finishes in a few minutes with the default parallel settings.
- Never commit your `.env` file — it is gitignored.

## Jellyfin Web (Chinese as ☐☐☐ boxes)

Dual **timing** can work (`--format srt --layout overlap`). If Chinese shows as empty boxes while English is fine, that is **not** a bad `.srt` — Jellyfin Web is missing a CJK font for subtitle rendering.

### Fix 1 — Fallback fonts (official, best)

1. Download a light CJK web font, e.g. [Noto Sans SC (woff2)](https://github.com/CodePlayer/webfont-noto) or from [Google Fonts](https://fonts.google.com/noto/specimen/Noto+Sans+SC).
2. On the Jellyfin server: **Dashboard → Playback → Fallback fonts**
   - Enable fallback fonts
   - Point at a folder that contains the `.woff2` / `.ttf` (total folder size limit ~20 MB)
3. Restart Jellyfin / hard-refresh the web client
4. Play again with your `.dual.srt` (`overlap` layout)

Docs: [Fallback fonts](https://jellyfin.org/docs/general/administration/configuration#fallback-fonts) · [Text not rendering](https://jellyfin.org/docs/general/administration/troubleshooting#text-not-rendering-properly)

### Fix 2 — Burn subtitles

User settings → Subtitles → **Burn subtitles** = All (or complex formats).  
Uses server fonts (install CJK fonts on the host/Docker image). Works, but forces transcoding.

### Fix 3 — Two tracks (no dual file)

Keep `movie.en.srt` + `movie.zh.srt` and use Jellyfin’s **primary + secondary** subtitle controls. Same font issue can still apply until fallback fonts are set.

### What this tool should output for Jellyfin

```bash
python dual_subs.py --merge en.srt zh.srt --format srt --layout overlap
```

Or in the UI: **Merge two files**, Dual format `srt`, Dual layout `overlap`.

Name the file like the video, e.g. `Movie Name (2021).srt`, next to the media file, then scan the library.
