"""
dual_subs.py - Turn a movie/TV subtitle file into dual-language (e.g. English + Chinese)
subtitles, translated via an NVIDIA NIM hosted LLM.

Usage:
    python dual_subs.py movie.srt
    python dual_subs.py movie.mkv                    # extract soft track, then translate
    python dual_subs.py movie.mkv --extract-only     # just dump the soft track to .srt
    python dual_subs.py movie.mkv --sub-stream 1     # pick a specific soft track
    python dual_subs.py movie.srt --target-lang zh-TW --order target-top
    python dual_subs.py --merge en.srt zh.srt        # fuse two existing language tracks
    (or drag & drop .srt/.vtt/.ass/.ssa/.mkv/.mp4 onto "Drag Subtitles Here.bat")

For each input "movie.srt" this produces, next to the original file:
    movie.dual.srt   - combined two-line-per-cue dual subtitle
    movie.en.srt     - exact copy of the original (source language only)
    movie.zh-CN.srt  - translation only, in the target language

With --merge, only a dual file is written (timings synced by overlap).
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
    "auto": "Auto-detect",
    "en": "English",
    "zh-CN": "Simplified Chinese",
    "zh-TW": "Traditional Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "th": "Thai",
    "vi": "Vietnamese",
    "it": "Italian",
    "nl": "Dutch",
}

# Common translate targets for UI pickers (code, label).
TARGET_LANG_CHOICES = [
    ("zh-CN", "Simplified Chinese (zh-CN)"),
    ("zh-TW", "Traditional Chinese (zh-TW)"),
    ("en", "English (en)"),
    ("ja", "Japanese (ja)"),
    ("ko", "Korean (ko)"),
    ("es", "Spanish (es)"),
    ("fr", "French (fr)"),
    ("de", "German (de)"),
    ("pt", "Portuguese (pt)"),
    ("ru", "Russian (ru)"),
    ("ar", "Arabic (ar)"),
    ("hi", "Hindi (hi)"),
    ("th", "Thai (th)"),
    ("vi", "Vietnamese (vi)"),
    ("it", "Italian (it)"),
    ("nl", "Dutch (nl)"),
]

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
        prefer = getattr(args, "source_lang", None)
        if prefer and str(prefer).lower() in {"", "auto", "detect"}:
            prefer = "en"  # best-effort track preference before we can read text
        return extract_soft_subs(
            path,
            prefer_lang=prefer,
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


def _ass_escape(text: str) -> str:
    return (text or "").replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


# BOM / bidi / zero-width marks that leak out of some Chinese subtitle files and
# confuse players (and can make mixed CN+EN lines look like "only one language").
_INVISIBLE_RE = re.compile("[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")


def clean_sub_text(text: str) -> str:
    return _INVISIBLE_RE.sub("", text or "").strip()


def prepare_dual_output(subs: pysubs2.SSAFile, fmt: str, layout: str = "stacked") -> pysubs2.SSAFile:
    """
    Prepare dual subs for saving.

    layout:
      - stacked: one cue, two lines (\\N / newline). Default; works in most players.
      - single-line: one cue, "ZH  |  EN" on one line. Some players truncate long lines.
    """
    fmt = (fmt or "srt").lower().lstrip(".")
    layout = (layout or "stacked").lower()
    if layout == "overlap":
        # Removed: concurrent same-time cues interleaved badly with multi-line source.
        layout = "stacked"

    # Normalize text and strip invisible marks first.
    base = copy.deepcopy(subs)
    for event in base:
        parts = [clean_sub_text(p) for p in (event.plaintext or "").split("\n")]
        parts = [p for p in parts if p]
        event.plaintext = "\n".join(parts)

    prepared = base
    for event in prepared:
        parts = [p for p in (event.plaintext or "").split("\n") if p]
        if not parts:
            event.plaintext = ""
        elif layout == "stacked" and len(parts) >= 2:
            event.plaintext = "\n".join(parts)
        else:
            event.plaintext = "  |  ".join(parts)

    if fmt != "ass":
        return prepared

    style = pysubs2.SSAStyle()
    style.fontname = "Noto Sans SC"
    style.fontsize = 16
    style.primarycolor = pysubs2.Color(255, 255, 255, 0)
    style.secondarycolor = pysubs2.Color(255, 255, 255, 0)
    style.outlinecolor = pysubs2.Color(0, 0, 0, 0)
    style.backcolor = pysubs2.Color(0, 0, 0, 0)
    style.bold = False
    style.outline = 1
    style.shadow = 0
    style.borderstyle = 1
    style.alignment = pysubs2.Alignment.BOTTOM_CENTER
    style.marginl = 10
    style.marginr = 10
    style.marginv = 20
    prepared.styles.clear()
    prepared.styles["Default"] = style

    for event in prepared:
        event.style = "Default"
        text = clean_sub_text(event.plaintext)
        if "\n" in (event.plaintext or ""):
            parts = [p for p in event.plaintext.split("\n") if p]
            event.text = "\\N".join(_ass_escape(p) for p in parts)
        else:
            event.text = _ass_escape(text)
    return prepared


def dual_output_path(base: Path, fmt: str) -> Path:
    fmt = (fmt or "srt").lower().lstrip(".")
    if fmt == "ass":
        return base.with_name(f"{base.stem}.dual.ass")
    return base.with_name(f"{base.stem}.dual.srt")


def save_dual(subs: pysubs2.SSAFile, path: Path, fmt: str, layout: str = "stacked") -> Path:
    fmt = (fmt or "srt").lower().lstrip(".")
    prepared = prepare_dual_output(subs, fmt, layout=layout)
    out_path = path
    if fmt == "ass" and out_path.suffix.lower() not in {".ass", ".ssa"}:
        out_path = out_path.with_suffix(".ass")
    if fmt == "srt" and out_path.suffix.lower() != ".srt":
        out_path = out_path.with_suffix(".srt")
    if fmt == "ass":
        prepared.save(str(out_path), format_="ass")
    else:
        prepared.save(str(out_path), format_="srt")
    return out_path


# --- Merge / fuse two existing subtitle files ---------------------------------

CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")
LATIN_RE = re.compile(r"[A-Za-z]")


def detect_script(subs: pysubs2.SSAFile) -> str:
    """Rough script family from cue text: 'cjk', 'latin', or 'unknown'."""
    cjk = latin = 0
    for event in subs:
        text = event.plaintext or ""
        cjk += len(CJK_RE.findall(text))
        latin += len(LATIN_RE.findall(text))
    if cjk == 0 and latin == 0:
        return "unknown"
    if cjk >= latin * 0.35:
        return "cjk"
    if latin > cjk:
        return "latin"
    return "unknown"


_HANGUL_RE = re.compile(r"[\uac00-\ud7af]")
_HIRAGANA_RE = re.compile(r"[\u3040-\u309f]")
_KATAKANA_RE = re.compile(r"[\u30a0-\u30ff]")
_CJK_UNIFIED_RE = re.compile(r"[\u4e00-\u9fff]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
_THAI_RE = re.compile(r"[\u0e00-\u0e7f]")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")
# Traditional-leaning characters that rarely appear in Simplified-only text.
_TRAD_HINT_RE = re.compile(r"[繁體體後發國學門開關東車風頭過還來時會對點說]")


def detect_language(subs: pysubs2.SSAFile) -> str:
    """
    Best-effort language code from subtitle text (no extra dependencies).
    Returns codes like en, zh-CN, zh-TW, ja, ko, es is NOT distinguished from
    other Latin languages — Latin defaults to en.
    """
    hangul = hiragana = katakana = cjk = cyrillic = arabic = thai = deva = latin = trad_hints = 0
    sample = []
    for i, event in enumerate(subs):
        if i > 400:
            break
        sample.append(event.plaintext or "")
    blob = "\n".join(sample)

    hangul = len(_HANGUL_RE.findall(blob))
    hiragana = len(_HIRAGANA_RE.findall(blob))
    katakana = len(_KATAKANA_RE.findall(blob))
    cjk = len(_CJK_UNIFIED_RE.findall(blob))
    cyrillic = len(_CYRILLIC_RE.findall(blob))
    arabic = len(_ARABIC_RE.findall(blob))
    thai = len(_THAI_RE.findall(blob))
    deva = len(_DEVANAGARI_RE.findall(blob))
    latin = len(LATIN_RE.findall(blob))
    trad_hints = len(_TRAD_HINT_RE.findall(blob))

    scores = {
        "ko": hangul,
        "ja": hiragana + katakana,
        "zh": cjk,
        "ru": cyrillic,
        "ar": arabic,
        "th": thai,
        "hi": deva,
        "en": latin,
    }
    # Japanese often mixes kanji + kana; boost ja when kana present with CJK.
    if hiragana + katakana > 20 and cjk > 0:
        scores["ja"] = hiragana + katakana + cjk * 0.5
        scores["zh"] = cjk * 0.3

    best = max(scores, key=scores.get)
    if scores[best] < 10:
        return "en"
    if best == "zh":
        # Rough Simplified vs Traditional guess.
        return "zh-TW" if trad_hints >= max(8, cjk * 0.02) else "zh-CN"
    return best


def resolve_source_lang(subs: pysubs2.SSAFile, source_lang: str) -> str:
    """Return concrete language code; auto-detect when source_lang is auto/empty."""
    code = (source_lang or "auto").strip()
    if code.lower() in {"", "auto", "detect"}:
        detected = detect_language(subs)
        print(f"  Auto-detected source language: {detected} ({lang_name(detected)})")
        return detected
    return code


def _overlap_ms(a_start, a_end, b_start, b_end) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _shift_subs(subs: pysubs2.SSAFile, shift_ms: int) -> pysubs2.SSAFile:
    out = copy.deepcopy(subs)
    if shift_ms:
        out.shift(ms=shift_ms)
    return out


def estimate_time_shift(
    primary: pysubs2.SSAFile,
    secondary: pysubs2.SSAFile,
    search_ms: int = 15000,
    coarse_step_ms: int = 200,
    fine_step_ms: int = 40,
) -> int:
    """
    Estimate a constant millisecond offset to apply to secondary so it lines up
    with primary, by maximizing total time-overlap across a search window.
    """
    if not primary or not secondary:
        return 0

    # Sample primary cues for speed on long movies.
    step = max(1, len(primary) // 120)
    sample = [(e.start, e.end) for i, e in enumerate(primary) if i % step == 0]
    sec = [(e.start, e.end) for e in secondary]

    def total_overlap(shift: int) -> int:
        total = 0
        j = 0
        # Secondary is sorted by start; advance a pointer as we scan sample.
        for a0, a1 in sample:
            # Move j to first secondary that might overlap a after shift.
            while j < len(sec) and sec[j][1] + shift < a0:
                j += 1
            k = j
            while k < len(sec) and sec[k][0] + shift <= a1:
                b0 = sec[k][0] + shift
                b1 = sec[k][1] + shift
                total += _overlap_ms(a0, a1, b0, b1)
                k += 1
        return total

    # Coarse search for best shift (secondary relative to primary: try negative
    # of offsets so applying +shift to secondary increases overlap).
    # If secondary is late, we need negative shift to pull it back → search both signs.
    best_shift, best_score = 0, total_overlap(0)
    for shift in range(-search_ms, search_ms + 1, coarse_step_ms):
        score = total_overlap(shift)
        if score > best_score:
            best_score, best_shift = score, shift

    # Fine search around the coarse winner.
    lo = best_shift - coarse_step_ms
    hi = best_shift + coarse_step_ms
    for shift in range(lo, hi + 1, fine_step_ms):
        score = total_overlap(shift)
        if score > best_score:
            best_score, best_shift = score, shift

    # Ignore tiny/noisy adjustments.
    if abs(best_shift) < 60:
        return 0
    return int(best_shift)


def _join_texts(texts):
    parts = []
    seen = set()
    for t in texts:
        t = clean_sub_text(t)
        if not t or t in seen:
            continue
        seen.add(t)
        parts.append(t)
    return "\n".join(parts)


def merge_subs(
    primary: pysubs2.SSAFile,
    secondary: pysubs2.SSAFile,
    order: str = "source-top",
    min_overlap_ms: int = 80,
    include_unmatched: bool = True,
) -> pysubs2.SSAFile:
    """
    Fuse two subtitle files into dual-line cues using time overlap.

    Primary supplies the timing spine. For each primary cue, overlapping
    secondary cue text is attached. Optionally append unmatched secondary cues.
    """
    dual = pysubs2.SSAFile()
    # Preserve styles from primary when present (ASS).
    dual.styles = copy.deepcopy(primary.styles)
    dual.info = copy.deepcopy(primary.info)

    used_secondary = set()

    for p in primary:
        matches = []
        for si, s in enumerate(secondary):
            ov = _overlap_ms(p.start, p.end, s.start, s.end)
            if ov >= min_overlap_ms:
                matches.append((ov, si, s))
        matches.sort(key=lambda x: (-x[0], x[2].start))
        other_text = _join_texts(s.plaintext for _, _, s in matches)
        for _, si, _ in matches:
            used_secondary.add(si)

        p_text = (p.plaintext or "").strip()
        if order == "target-top":
            top, bottom = other_text, p_text
        else:
            top, bottom = p_text, other_text

        if not top and not bottom:
            continue
        if top and bottom:
            text = f"{top}\n{bottom}"
        else:
            text = top or bottom

        event = copy.deepcopy(p)
        event.plaintext = text
        dual.append(event)

    if include_unmatched:
        for si, s in enumerate(secondary):
            if si in used_secondary:
                continue
            text = (s.plaintext or "").strip()
            if not text:
                continue
            event = copy.deepcopy(s)
            event.plaintext = text
            dual.append(event)

    dual.sort()
    return dual


def process_merge(args, path_a: Path, path_b: Path):
    print(f"\n=== merge: {path_a.name} + {path_b.name} ===")

    # Allow video inputs: extract soft tracks first.
    a = resolve_input(path_a, args) if path_a.suffix.lower() in VIDEO_EXTS else path_a
    b = resolve_input(path_b, args) if path_b.suffix.lower() in VIDEO_EXTS else path_b
    if a.suffix.lower() not in SUPPORTED_EXTS or b.suffix.lower() not in SUPPORTED_EXTS:
        raise RuntimeError("Both merge inputs must be subtitle files (or videos with soft text tracks).")

    subs_a = load_subs(a)
    subs_b = load_subs(b)
    script_a, script_b = detect_script(subs_a), detect_script(subs_b)
    print(f"  {a.name}: {len(subs_a)} cues ({script_a})")
    print(f"  {b.name}: {len(subs_b)} cues ({script_b})")

    # Prefer Latin (usually English) as the timing spine / "source" line.
    if script_a == "cjk" and script_b == "latin":
        primary, secondary = subs_b, subs_a
        primary_path, secondary_path = b, a
        print("  Detected: Latin timing spine + CJK partner (swapped order)")
    else:
        primary, secondary = subs_a, subs_b
        primary_path, secondary_path = a, b
        if script_a == "latin" and script_b == "cjk":
            print("  Detected: Latin timing spine + CJK partner")
        else:
            print("  Using first file as timing spine (could not confidently detect scripts)")

    shift = args.shift_ms
    if args.auto_shift:
        estimated = estimate_time_shift(primary, secondary)
        shift += estimated
        print(f"  Auto sync offset for second track: {estimated:+d} ms")
    if shift:
        print(f"  Applying shift to second track: {shift:+d} ms")
        secondary = _shift_subs(secondary, shift)

    dual = merge_subs(
        primary,
        secondary,
        order=args.order,
        min_overlap_ms=args.min_overlap_ms,
        include_unmatched=not args.drop_unmatched,
    )

    fmt = getattr(args, "format", "srt") or "srt"
    layout = getattr(args, "layout", "stacked") or "stacked"
    out = args.output
    if out:
        out_path = Path(out)
        if fmt == "ass" and out_path.suffix.lower() not in {".ass", ".ssa"}:
            out_path = out_path.with_suffix(".ass")
        if fmt == "srt" and out_path.suffix.lower() != ".srt":
            out_path = out_path.with_suffix(".srt")
    else:
        out_path = dual_output_path(primary_path, fmt)

    out_path = save_dual(dual, out_path, fmt, layout=layout)
    matched = sum(1 for e in dual if "\n" in (e.plaintext or ""))
    print(f"  -> {out_path.name}  ({len(dual)} cues, {matched} stacked-source cues, format={fmt}, layout={layout})")


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
    source_lang = resolve_source_lang(subs, args.source_lang)
    print(f"  batch_size={args.batch_size}, workers={args.workers}")

    translations = translate_all(
        client,
        args.model,
        lines,
        source_lang,
        args.target_lang,
        args.context,
        args.batch_size,
        workers=args.workers,
    )

    dual_subs, target_subs = build_dual_and_target(subs, translations, args.order)

    stem, ext = sub_path.stem, sub_path.suffix
    fmt = getattr(args, "format", "srt") or "srt"
    layout = getattr(args, "layout", "stacked") or "stacked"
    dual_path = dual_output_path(sub_path, fmt)
    target_path = sub_path.with_name(f"{stem}.{args.target_lang}{ext}")
    source_path = sub_path.with_name(f"{stem}.{source_lang}{ext}")

    dual_path = save_dual(dual_subs, dual_path, fmt, layout=layout)
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
    parser.add_argument(
        "--merge",
        nargs=2,
        metavar=("FILE_A", "FILE_B"),
        help="Fuse two existing subtitle (or video soft-track) files into one dual file by time sync",
    )
    parser.add_argument("-o", "--output", default=None, help="Output path for --merge (default: next to first/spine file)")
    parser.add_argument(
        "--shift-ms",
        type=int,
        default=0,
        help="With --merge: shift the second track by this many milliseconds (after auto-shift if enabled)",
    )
    parser.add_argument(
        "--auto-shift",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="With --merge: auto-estimate a global sync offset if tracks are misaligned (default: off).",
    )
    parser.add_argument(
        "--min-overlap-ms",
        type=int,
        default=80,
        help="With --merge: minimum overlap in ms to pair cues (default: 80)",
    )
    parser.add_argument(
        "--drop-unmatched",
        action="store_true",
        help="With --merge: drop cues from the second file that don't overlap anything",
    )
    parser.add_argument(
        "--source-lang",
        default="auto",
        help="Source language code, or 'auto' to detect from subtitle text (default: auto)",
    )
    parser.add_argument("--target-lang", default="zh-CN", help="Target language code (default: zh-CN)")
    parser.add_argument(
        "--order",
        choices=["source-top", "target-top"],
        default="source-top",
        help="Line order in the combined dual file (default: source-top). For --merge, source = timing spine.",
    )
    parser.add_argument(
        "--format",
        choices=["srt", "ass"],
        default="srt",
        help="Dual subtitle output format (default: srt). Soft ASS cannot render CJK reliably in Jellyfin Web.",
    )
    parser.add_argument(
        "--layout",
        choices=["stacked", "single-line"],
        default="stacked",
        help="Dual layout: stacked (two lines per cue) or single-line (ZH | EN)",
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

    # --- Merge mode: fuse two existing language files (no API key needed) ---
    if args.merge:
        a, b = Path(args.merge[0]), Path(args.merge[1])
        for p in (a, b):
            if not p.exists():
                sys.exit(f"ERROR: file not found: {p}")
        try:
            process_merge(args, a, b)
        except Exception as e:
            print(f"  [error] merge failed: {e}", file=sys.stderr)
            sys.exit(1)
        print("\nDone.")
        return

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
