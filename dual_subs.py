"""
dual_subs.py - Turn a movie/TV subtitle file into dual-language (e.g. English + Chinese)
subtitles, translated via an NVIDIA NIM hosted LLM.

Usage:
    python dual_subs.py movie.srt
    python dual_subs.py movie.mkv                    # extract soft track, then translate
    python dual_subs.py movie.mkv --extract-only     # just dump the soft track to .srt
    python dual_subs.py movie.mkv --sub-stream 1     # pick a specific soft track
    python dual_subs.py movie.srt --target-lang zh-TW --order target-top
    (or drag & drop .srt/.vtt/.ass/.ssa/.mkv/.mp4 onto "Drag Subtitles Here.bat")

For each input "movie.srt" this produces, next to the original file:
    movie.dual.srt   - combined two-line-per-cue dual subtitle
    movie.en.srt     - exact copy of the original (source language only)
    movie.zh-CN.srt  - translation only, in the target language
"""

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pysubs2
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

for _enc in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le"):
    try:
        if load_dotenv(encoding=_enc) and os.environ.get("NVIDIA_API_KEY"):
            break
    except Exception:
        continue

# Windows consoles default to a legacy codepage (e.g. cp1252) that cannot print
# Chinese/Japanese/etc. text and would crash mid-run. Force UTF-8 output if possible.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

SUPPORTED_EXTS = {".srt", ".vtt", ".ass", ".ssa"}
VIDEO_EXTS = {".mkv", ".mp4", ".m4v", ".avi", ".mov", ".webm", ".m2ts", ".ts"}

DEFAULT_MODEL = os.environ.get("NVIDIA_MODEL", "qwen/qwen3.5-397b-a17b")
DEFAULT_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

LANG_NAMES = {
    "en": "English",
    "zh-CN": "Simplified Chinese",
    "zh-TW": "Traditional Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
}

ENCODINGS_TO_TRY = ["utf-8-sig", "utf-8", "cp1252", "gb18030", "latin-1"]

SYSTEM_PROMPT_TMPL = """You are a professional subtitle translator for films and TV shows.
Translate subtitle lines from {src} to {tgt}.

Rules:
- Prioritize natural, idiomatic phrasing over literal word-for-word translation.
- Preserve tone/register. Keep names consistent. Keep lines short enough to read as subtitles.
- Sound effects / music cues: short equivalent, not a long explanation.
- Do not add notes, romanization, or commentary.
- Do not merge, split, or reorder lines.

Input format: one cue per line as NNN|text (NNN is a 3+ digit index).
Output format: the SAME number of lines, SAME indices, same order:
NNN|translated text
No blank lines, no markdown, no extra text.
{context}"""

LINE_RE = re.compile(r"^(\d{1,4})\s*\|\s*(.*)$")


def lang_name(code: str) -> str:
    return LANG_NAMES.get(code, code)


def get_client() -> OpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        sys.exit(
            "ERROR: NVIDIA_API_KEY is not set.\n"
            "1. Get a free key at https://build.nvidia.com (sign in -> any model page -> Get API Key)\n"
            "2. Copy .env.example to .env and paste your key into it.\n"
        )
    return OpenAI(base_url=DEFAULT_BASE_URL, api_key=api_key)


def load_subs(path: Path) -> pysubs2.SSAFile:
    last_err = None
    for enc in ENCODINGS_TO_TRY:
        try:
            return pysubs2.load(str(path), encoding=enc)
        except (UnicodeDecodeError, UnicodeError) as e:
            last_err = e
            continue
    raise last_err


def _run_cmd(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def list_subtitle_streams(video_path: Path):
    """Return soft subtitle streams via ffprobe: [{index, codec, language, title}, ...]."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "s",
        "-show_entries",
        "stream=index,codec_name:stream_tags=language,title",
        "-of",
        "json",
        str(video_path),
    ]
    result = _run_cmd(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            "ffprobe failed. Is ffmpeg/ffprobe installed and on PATH?\n" + (result.stderr or result.stdout)
        )
    data = json.loads(result.stdout or "{}")
    streams = []
    for i, stream in enumerate(data.get("streams") or []):
        tags = stream.get("tags") or {}
        streams.append(
            {
                "sub_index": i,  # 0-based among subtitle streams (for -map 0:s:N)
                "stream_index": stream.get("index"),
                "codec": stream.get("codec_name") or "unknown",
                "language": (tags.get("language") or "und").lower(),
                "title": tags.get("title") or "",
            }
        )
    return streams


def pick_subtitle_stream(streams, prefer_lang: str | None = None):
    if not streams:
        return None
    if prefer_lang:
        prefer = prefer_lang.lower().split("-")[0]
        for s in streams:
            if s["language"].startswith(prefer):
                return s
    # Prefer text-based codecs over image/bitmap (PGS/VobSub need OCR).
    text_codecs = {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "text", "subtitle"}
    for s in streams:
        if s["codec"].lower() in text_codecs:
            return s
    return streams[0]


def extract_soft_subs(video_path: Path, prefer_lang: str | None = None, stream_index: int | None = None) -> Path:
    """
    Extract a soft subtitle track from a video into a sibling .srt/.ass file.
    Returns the path of the extracted subtitle file.
    """
    streams = list_subtitle_streams(video_path)
    if not streams:
        raise RuntimeError(f"No soft subtitle tracks found in {video_path.name}")

    print(f"  Found {len(streams)} soft subtitle track(s):")
    for s in streams:
        title = f' "{s["title"]}"' if s["title"] else ""
        print(f"    [{s['sub_index']}] lang={s['language']} codec={s['codec']}{title}")

    if stream_index is not None:
        chosen = next((s for s in streams if s["sub_index"] == stream_index), None)
        if chosen is None:
            raise RuntimeError(f"Subtitle stream index {stream_index} not found (valid: 0-{len(streams)-1})")
    else:
        chosen = pick_subtitle_stream(streams, prefer_lang)

    # Image-based tracks (PGS/dvd_subtitle) can't be dumped to plain .srt by ffmpeg alone.
    image_codecs = {"hdmv_pgs_subtitle", "dvd_subtitle", "dvdsub", "pgssub", "xsub"}
    if chosen["codec"].lower() in image_codecs:
        raise RuntimeError(
            f"Track [{chosen['sub_index']}] is image-based ({chosen['codec']}), not plain text. "
            "Those need OCR — pick a text track with --sub-stream if one exists."
        )

    out_ext = ".ass" if chosen["codec"].lower() in {"ass", "ssa"} else ".srt"
    out_path = video_path.with_name(f"{video_path.stem}.{chosen['language']}{out_ext}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-map",
        f"0:s:{chosen['sub_index']}",
        str(out_path),
    ]
    print(f"  Extracting track [{chosen['sub_index']}] ({chosen['language']}/{chosen['codec']}) -> {out_path.name}")
    result = _run_cmd(cmd)
    if result.returncode != 0 or not out_path.exists():
        raise RuntimeError("ffmpeg extract failed:\n" + (result.stderr or result.stdout or "(no output)"))
    return out_path


def resolve_input(path: Path, args) -> Path:
    """If path is a video, extract soft subs first; otherwise return as-is."""
    if path.suffix.lower() in VIDEO_EXTS:
        return extract_soft_subs(
            path,
            prefer_lang=args.source_lang,
            stream_index=args.sub_stream,
        )
    return path


def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _format_numbered(lines, start_index: int = 0) -> str:
    # Collapse internal newlines so each cue stays on one transport line.
    return "\n".join(f"{start_index + i:03d}|{line.replace(chr(10), ' / ')}" for i, line in enumerate(lines))


def _parse_numbered(text: str, expected_indices: list):
    text = _strip_think_tags(text.strip())
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```$", "", text).strip()

    found = {}
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        m = LINE_RE.match(raw)
        if m:
            found[int(m.group(1))] = m.group(2)

    if all(i in found for i in expected_indices):
        return [found[i] for i in expected_indices]

    # Fallback: plain lines in order if count matches.
    plain = [ln.strip() for ln in text.splitlines() if ln.strip()]
    stripped = []
    for ln in plain:
        m = LINE_RE.match(ln)
        stripped.append(m.group(2) if m else ln)
    if len(stripped) == len(expected_indices):
        return stripped

    raise ValueError(
        f"Could not align numbered output (got {len(found)}/{len(expected_indices)} indices; "
        f"content: {text[:160]!r})"
    )


def _message_text(choice) -> str:
    msg = choice.message
    content = getattr(msg, "content", None) or ""
    if content:
        return content
    dump = msg.model_dump() if hasattr(msg, "model_dump") else {}
    return dump.get("reasoning_content") or dump.get("refusal") or ""


def translate_batch(client, model, lines, src, tgt, context, start_index=0, max_retries=3):
    if not lines:
        return []

    indices = list(range(start_index, start_index + len(lines)))
    system_prompt = SYSTEM_PROMPT_TMPL.format(
        src=lang_name(src),
        tgt=lang_name(tgt),
        context=f"Context about this content: {context}\n" if context else "",
    )
    user_payload = _format_numbered(lines, start_index)
    max_tokens = min(8192, 300 + 120 * len(lines))

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.2,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_payload},
                ],
            )
            choice = response.choices[0]
            content = _message_text(choice)
            if choice.finish_reason == "length":
                print(f"  [warn] batch@{start_index} truncated; splitting...", file=sys.stderr)
                break
            if not content:
                raise ValueError(f"empty content (finish_reason={choice.finish_reason!r})")
            return _parse_numbered(content, indices)
        except Exception as e:
            msg = str(e)
            is_empty = "empty content" in msg
            is_align = "Could not align" in msg
            if is_align or (is_empty and attempt + 1 >= max_retries):
                print(f"  [warn] batch@{start_index} n={len(lines)} failed ({e}); splitting...", file=sys.stderr)
                break
            wait = 1.5 if is_empty else 2 ** attempt
            print(f"  [warn] batch@{start_index} n={len(lines)} failed ({e}); retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)

    if len(lines) == 1:
        print(f"  [warn] giving up on line {start_index}: {lines[0]!r}", file=sys.stderr)
        return lines
    mid = len(lines) // 2
    return translate_batch(
        client, model, lines[:mid], src, tgt, context, start_index, max_retries
    ) + translate_batch(
        client, model, lines[mid:], src, tgt, context, start_index + mid, max_retries
    )


def translate_all(client, model, lines, src, tgt, context, batch_size, workers=6):
    """Extract cues -> translate numbered batches in parallel -> reassemble in order."""
    batches = []
    for start in range(0, len(lines), batch_size):
        batches.append((start, lines[start : start + batch_size]))

    results = [None] * len(batches)

    def _work(item):
        start, chunk = item
        return start, translate_batch(client, model, chunk, src, tgt, context, start_index=start)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_work, item): i for i, item in enumerate(batches)}
        with tqdm(total=len(batches), desc="Translating", unit="batch") as bar:
            for fut in as_completed(futures):
                i = futures[fut]
                _start, translated = fut.result()
                results[i] = translated
                bar.update(1)

    out = []
    for part in results:
        out.extend(part)
    if len(out) != len(lines):
        raise RuntimeError(f"internal error: got {len(out)} translations for {len(lines)} cues")
    return out


def build_dual_and_target(subs: pysubs2.SSAFile, translations, order: str):
    dual = copy.deepcopy(subs)
    target_only = copy.deepcopy(subs)

    for event, translation in zip(dual, translations):
        original = event.plaintext
        # Restore any collapsed multi-line cues from " / " separators if needed.
        if order == "target-top":
            event.plaintext = f"{translation}\n{original}"
        else:
            event.plaintext = f"{original}\n{translation}"

    for event, translation in zip(target_only, translations):
        event.plaintext = translation

    return dual, target_only


def process_file(client, args, path: Path):
    print(f"\n=== {path.name} ===")

    try:
        sub_path = resolve_input(path, args)
    except Exception as e:
        print(f"  [error] {e}", file=sys.stderr)
        return

    if args.extract_only:
        print(f"  -> extracted only: {sub_path.name}")
        return

    if sub_path.suffix.lower() not in SUPPORTED_EXTS:
        print(f"  [skip] unsupported extension: {sub_path.suffix}")
        return

    if sub_path != path:
        print(f"  Using extracted subtitle: {sub_path.name}")

    subs = load_subs(sub_path)
    lines = [event.plaintext for event in subs]
    print(f"  {len(lines)} cues loaded")
    print(f"  batch_size={args.batch_size}, workers={args.workers}")

    translations = translate_all(
        client,
        args.model,
        lines,
        args.source_lang,
        args.target_lang,
        args.context,
        args.batch_size,
        workers=args.workers,
    )

    dual_subs, target_subs = build_dual_and_target(subs, translations, args.order)

    stem, ext = sub_path.stem, sub_path.suffix
    dual_path = sub_path.with_name(f"{stem}.dual{ext}")
    target_path = sub_path.with_name(f"{stem}.{args.target_lang}{ext}")
    source_path = sub_path.with_name(f"{stem}.{args.source_lang}{ext}")

    dual_subs.save(str(dual_path))
    target_subs.save(str(target_path))
    if sub_path.resolve() != source_path.resolve():
        shutil.copy2(str(sub_path), str(source_path))

    print(f"  -> {dual_path.name}")
    print(f"  -> {target_path.name}")
    print(f"  -> {source_path.name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create dual-language subtitles using an NVIDIA NIM model.")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Subtitle file(s) and/or video file(s) with soft subtitle tracks",
    )
    parser.add_argument("--source-lang", default="en", help="Source language code (default: en)")
    parser.add_argument("--target-lang", default="zh-CN", help="Target language code (default: zh-CN)")
    parser.add_argument(
        "--order",
        choices=["source-top", "target-top"],
        default="source-top",
        help="Line order in the combined dual file (default: source-top)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"NVIDIA NIM model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--batch-size", type=int, default=20, help="Subtitle lines per API request (default: 20)")
    parser.add_argument("--workers", type=int, default=6, help="Parallel API requests (default: 6)")
    parser.add_argument(
        "--sub-stream",
        type=int,
        default=None,
        help="Which soft subtitle track to extract from a video (0-based). Default: prefer --source-lang, else first text track",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract soft subs from video(s); do not translate",
    )
    parser.add_argument(
        "--context",
        default=os.environ.get("SUBS_CONTEXT", ""),
        help="Optional short description of the movie/show to improve translation accuracy",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.paths:
        print(__doc__)
        raw = input("Drag a subtitle/video file here (or type its path) and press Enter: ").strip().strip('"')
        if raw:
            args.paths = [raw]

    if not args.paths:
        print("No files given. Exiting.")
        return

    client = None if args.extract_only else get_client()

    for raw_path in args.paths:
        path = Path(raw_path)
        if not path.exists():
            print(f"\n=== {raw_path} ===\n  [error] file not found")
            continue
        try:
            process_file(client, args, path)
        except Exception as e:
            print(f"  [error] failed to process {path.name}: {e}", file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    finally:
        default_pause = "1" if sys.stdin.isatty() else "0"
        if os.environ.get("PROMPT_ON_EXIT", default_pause) == "1":
            try:
                input("\nPress Enter to exit...")
            except EOFError:
                pass
