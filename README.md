# Dual Subs

Turn English subtitles into dual-language subtitles (e.g. English + Chinese) using an [NVIDIA NIM](https://build.nvidia.com) LLM.

**Input:** `.srt` / `.vtt` / `.ass` / `.ssa`, or a video with soft subtitle tracks (`.mkv`, `.mp4`, …)  
**Output** (next to the source file):

| File | Contents |
|---|---|
| `movie.dual.srt` | English + Chinese on each cue |
| `movie.en.srt` | Original English |
| `movie.zh-CN.srt` | Chinese only |

## Requirements

- Python 3.10+
- An NVIDIA API key ([build.nvidia.com](https://build.nvidia.com) — free tier, no credit card)
- Optional: [ffmpeg](https://ffmpeg.org) on PATH (only needed to extract soft tracks from video)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # on Windows: copy .env.example .env
```

Edit `.env` and set your key:

```
NVIDIA_API_KEY=nvapi-...
```

## Usage

**Windows:** drag a subtitle or video file onto `Drag Subtitles Here.bat`.

**CLI:**

```bash
# Subtitle file → dual subs
python dual_subs.py movie.srt

# Video with soft tracks → extract English track → dual subs
python dual_subs.py movie.mkv

# Extract only (no translation)
python dual_subs.py movie.mkv --extract-only

# Pick a specific soft track (0-based)
python dual_subs.py movie.mkv --sub-stream 1

# Traditional Chinese, Chinese line on top
python dual_subs.py movie.srt --target-lang zh-TW --order target-top

# Extra context for better tone / names
python dual_subs.py movie.srt --context "The Amazing Spider-Man (2012), casual teen dialogue"
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--source-lang` | `en` | Source language |
| `--target-lang` | `zh-CN` | Target (`zh-CN` Simplified, `zh-TW` Traditional) |
| `--order` | `source-top` | Dual layout: `source-top` or `target-top` |
| `--model` | `qwen/qwen3.5-397b-a17b` | NIM model id ([catalog](https://build.nvidia.com/models)) |
| `--batch-size` | `20` | Cues per API request |
| `--workers` | `6` | Parallel API requests |
| `--sub-stream` | auto | Soft track index when input is a video |
| `--extract-only` | off | Extract soft subs only |
| `--context` | _(none)_ | Movie/show notes for the translator |

## How it works

1. Loads cues (or extracts a soft text track from video via `ffmpeg`).
2. Sends numbered batches of lines to NVIDIA NIM in parallel.
3. Reattaches translations to the original timings.
4. Writes dual + single-language files.

Defaults to a large Qwen model for strong English↔Chinese quality. Smaller/faster models can be set with `--model`.

Text soft tracks (`srt`, `ass`, `mov_text`, …) extract cleanly. Image-based tracks (`PGS`, `VobSub`) need OCR and are not supported.

## Notes

- Timing and cue structure are preserved; inline styling on translated lines is not.
- A ~90 minute movie (~1000–1500 cues) typically finishes in a few minutes with the default parallel settings.
- Never commit your `.env` file — it is gitignored.
