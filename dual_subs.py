"""
dual_subs.py - Turn a movie/TV subtitle file into dual-language (e.g. English + Chinese)
subtitles, translated via an NVIDIA NIM hosted LLM.

CLI usage:
    python dual_subs.py movie.srt
    python dual_subs.py movie.srt --target-lang zh-TW --order target-top
    python dual_subs.py --merge en.srt zh.srt        # fuse two existing language tracks

Desktop UI (no arguments):
    Double-click Dual Subs UI.bat
    python dual_subs.py
    python dual_subs.py --ui

For each input "movie.srt" this produces, next to the original file:
    movie.dual.srt   - combined two-line-per-cue dual subtitle
    movie.en.srt     - exact copy of the original (source language only)
    movie.zh-CN.srt  - translation only, in the target language

With --merge, only a dual file is written (timings synced by overlap).
"""

from __future__ import annotations

import argparse
import copy
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pysubs2
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

try:
    import tkinter as tk
    from tkinter import filedialog, font as tkfont, messagebox, ttk

    _HAS_TK = True
except ImportError:  # headless environments (CLI must still work)
    tk = None  # type: ignore[assignment]
    filedialog = messagebox = ttk = tkfont = None  # type: ignore[assignment]
    _HAS_TK = False

# Load .env from the directory of this script explicitly (not just CWD), trying a
# few encodings, then fall back to CWD discovery.
_ENV_PATH = Path(__file__).resolve().parent / ".env"
for _enc in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le"):
    try:
        if _ENV_PATH.exists() and load_dotenv(dotenv_path=_ENV_PATH, encoding=_enc):
            break
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
DEFAULT_MODEL = os.environ.get("NVIDIA_MODEL", "qwen/qwen2.5-72b-instruct")
DEFAULT_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

# Model ids offered in the UI picker. The env var NVIDIA_MODEL / --model still
# overrides. The former default (qwen3.5-397b-a17b) is kept for continuity.
MODEL_CHOICES = [
    "qwen/qwen2.5-72b-instruct",
    "qwen/qwen2.5-7b-instruct",
    "qwen/qwen3.5-397b-a17b",
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.1-8b-instruct",
]

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

ENCODINGS_TO_TRY = ["utf-8-sig", "utf-8", "utf-16", "utf-16-le", "utf-16-be", "cp1252", "gb18030", "latin-1"]

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

LINE_RE = re.compile(r"^(\d+)\s*\|\s*(.*)$")


def lang_name(code: str) -> str:
    return LANG_NAMES.get(code, code)


def get_client() -> OpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set.\n"
            "1. Get a free key at https://build.nvidia.com (sign in -> any model page -> Get API Key)\n"
            "2. Copy .env.example to .env and paste your key into it.\n"
        )
    return OpenAI(base_url=DEFAULT_BASE_URL, api_key=api_key)


def load_subs(path: Path) -> pysubs2.SSAFile:
    last_err = None
    for enc in ENCODINGS_TO_TRY:
        try:
            subs = pysubs2.load(str(path), encoding=enc)
            if enc == "latin-1":
                print(
                    f"  [warn] {path.name}: decoded as latin-1 fallback; "
                    "characters may be garbled if the file is not really latin-1.",
                    file=sys.stderr,
                )
            return subs
        except (UnicodeDecodeError, UnicodeError) as e:
            last_err = e
            continue
    raise last_err


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
            found[int(m.group(1))] = m.group(2).replace(" / ", "\n")

    if all(i in found for i in expected_indices):
        return [found[i] for i in expected_indices]

    # Fallback: plain lines in order if count matches.
    plain = [ln.strip() for ln in text.splitlines() if ln.strip()]
    stripped = []
    for ln in plain:
        m = LINE_RE.match(ln)
        stripped.append((m.group(2) if m else ln).replace(" / ", "\n"))
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


def _retry_after_seconds(msg: str):
    """Best-effort parse of a Retry-After hint from an error message, else None."""
    m = re.search(r"retry[-\s]?after[\"':\s]+(\d+(?:\.\d+)?)", msg, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


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
            is_rate = "429" in msg or "rate" in msg.lower()
            jitter = random.uniform(0, 0.5)
            if is_rate:
                wait = _retry_after_seconds(msg)
                if wait is None:
                    wait = random.uniform(5, 20)
            elif is_empty:
                wait = 1.5
            else:
                wait = 2 ** attempt
            wait += jitter
            print(
                f"  [warn] batch@{start_index} n={len(lines)} failed ({e}); retrying in {wait:.1f}s...",
                file=sys.stderr,
            )
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


def translate_all(
    client,
    model,
    lines,
    src,
    tgt,
    context,
    batch_size,
    workers=6,
    cancel_event: "threading.Event | None" = None,
    progress_cb=None,
):
    """Extract cues -> translate numbered batches in parallel -> reassemble in order.

    Empty/whitespace-only cues pass through as "". Identical non-empty cues are
    translated once and mapped back. Supports cooperative cancellation via
    cancel_event and progress reporting via progress_cb(done_batches, total_batches).
    """

    def _cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    # Collect the unique non-empty cue texts, preserving first-seen order.
    unique_texts = []
    seen = {}
    for line in lines:
        if line and line.strip() and line not in seen:
            seen[line] = None
            unique_texts.append(line)

    batches = []
    for start in range(0, len(unique_texts), batch_size):
        batches.append((start, unique_texts[start : start + batch_size]))

    results = [None] * len(batches)
    total_batches = len(batches)
    done_batches = 0
    if progress_cb is not None:
        progress_cb(0, total_batches)

    if _cancelled():
        raise RuntimeError("Cancelled")

    def _work(item):
        start, chunk = item
        return start, translate_batch(client, model, chunk, src, tgt, context, start_index=start)

    disable_bar = not (hasattr(sys.stderr, "isatty") and sys.stderr.isatty())

    if total_batches:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(_work, item): i for i, item in enumerate(batches)}
            with tqdm(total=total_batches, desc="Translating", unit="batch", disable=disable_bar) as bar:
                for fut in as_completed(futures):
                    i = futures[fut]
                    _start, translated = fut.result()
                    results[i] = translated
                    bar.update(1)
                    done_batches += 1
                    if progress_cb is not None:
                        progress_cb(done_batches, total_batches)
                    if _cancelled():
                        for f in futures:
                            f.cancel()
                        raise RuntimeError("Cancelled")

    translated_unique = []
    for part in results:
        translated_unique.extend(part or [])
    if len(translated_unique) != len(unique_texts):
        raise RuntimeError(
            f"internal error: got {len(translated_unique)} translations for {len(unique_texts)} unique cues"
        )

    mapping = dict(zip(unique_texts, translated_unique))
    out = []
    for line in lines:
        if not line or not line.strip():
            out.append("")
        else:
            out.append(mapping.get(line, line))
    return out


def build_dual_and_target(subs: pysubs2.SSAFile, translations, order: str):
    dual = copy.deepcopy(subs)
    target_only = copy.deepcopy(subs)

    for event, translation in zip(dual, translations, strict=True):
        original = event.plaintext
        if order == "target-top":
            event.plaintext = f"{translation}\n{original}"
        else:
            event.plaintext = f"{original}\n{translation}"

    for event, translation in zip(target_only, translations, strict=True):
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
    sec.sort(key=lambda x: x[0])

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

    a, b = path_a, path_b
    if a.suffix.lower() not in SUPPORTED_EXTS or b.suffix.lower() not in SUPPORTED_EXTS:
        raise RuntimeError("Both merge inputs must be subtitle files (.srt/.vtt/.ass/.ssa).")

    subs_a = load_subs(a)
    subs_b = load_subs(b)
    script_a, script_b = detect_script(subs_a), detect_script(subs_b)
    print(f"  {a.name}: {len(subs_a)} cues ({script_a})")
    print(f"  {b.name}: {len(subs_b)} cues ({script_b})")

    # Manual --shift-ms always targets Subtitle 2 / FILE_2 (subs_b), applied
    # before spine selection so the semantics are stable regardless of swaps.
    if args.shift_ms:
        print(f"  Applying manual shift to {b.name} (Subtitle 2): {args.shift_ms:+d} ms")
        subs_b = _shift_subs(subs_b, args.shift_ms)

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

    if args.auto_shift:
        estimated = estimate_time_shift(primary, secondary)
        print(f"  Auto sync offset for second track: {estimated:+d} ms")
        if estimated:
            secondary = _shift_subs(secondary, estimated)

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


def process_file(client, args, path: Path, cancel_event=None, progress_cb=None) -> bool:
    """Translate one subtitle file. Returns True on success, False on skip/failure."""
    print(f"\n=== {path.name} ===")

    sub_path = path
    if sub_path.suffix.lower() not in SUPPORTED_EXTS:
        print(f"  [skip] unsupported extension: {sub_path.suffix}")
        return False

    subs = load_subs(sub_path)
    lines = [event.plaintext for event in subs]
    print(f"  {len(lines)} cues loaded")
    source_lang = resolve_source_lang(subs, args.source_lang)

    if source_lang == args.target_lang:
        print(
            f"  [error] source language ({source_lang}) equals target language "
            f"({args.target_lang}); nothing to translate. Skipping.",
            file=sys.stderr,
        )
        return False

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
        cancel_event=cancel_event,
        progress_cb=progress_cb,
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
    if sub_path.resolve() != source_path.resolve():
        print(f"  -> {source_path.name}")
    return True



# --- Desktop UI (Tkinter) -----------------------------------------------------





SUB_TYPES = [
    ("Subtitles", "*.srt *.vtt *.ass *.ssa"),
    ("All files", "*.*"),
]

PREVIEW_CUES = 40
TRANSLATE_PREVIEW_CUES = 8


class QueueWriter:
    """Redirect print() into a thread-safe queue for the log pane."""

    def __init__(self, q: queue.Queue):
        self.q = q

    def write(self, text: str):
        if text:
            self.q.put(text)

    def flush(self):
        pass


def _fmt_ts(ms: int) -> str:
    h = ms // 3_600_000
    m = (ms % 3_600_000) // 60_000
    s = (ms % 60_000) // 1000
    milli = ms % 1000
    if h:
        return f"{h:d}:{m:02d}:{s:02d}.{milli:03d}"
    return f"{m:02d}:{s:02d}.{milli:03d}"


def _format_subs_preview(subs, title: str, limit: int = PREVIEW_CUES) -> str:
    lines = [f"{title}  —  {len(subs)} cues", "-" * 56]
    for i, event in enumerate(subs):
        if i >= limit:
            lines.append(f"… ({len(subs) - limit} more cues)")
            break
        text = (event.plaintext or "").replace("\n", " ↵ ")
        lines.append(f"{_fmt_ts(event.start)} → {_fmt_ts(event.end)}")
        lines.append(f"  {text}")
        lines.append("")
    return "\n".join(lines)


_TkBase = tk.Tk if _HAS_TK else object


class DualSubsApp(_TkBase):
    def __init__(self):
        super().__init__()
        self.title("Dual Subtitles")
        self.geometry("880x740")
        self.minsize(720, 580)

        self.log_q: queue.Queue = queue.Queue()
        self.worker: "threading.Thread | None" = None
        self._preview_job = None
        self._preview_token = 0
        self._translate_preview_worker: "threading.Thread | None" = None
        self._cancel_event = threading.Event()
        self._build()
        self.after(100, self._drain_log)

    def _build(self):
        pad = {"padx": 10, "pady": 6}
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        # Mode
        mode_row = ttk.Frame(root)
        mode_row.pack(fill=tk.X, **pad)
        ttk.Label(mode_row, text="Mode").pack(side=tk.LEFT)
        self.mode = tk.StringVar(value="translate")
        for value, label in (
            ("translate", "Translate (AI)"),
            ("merge", "Merge two files"),
        ):
            ttk.Radiobutton(
                mode_row,
                text=label,
                value=value,
                variable=self.mode,
                command=self._on_mode_change,
            ).pack(side=tk.LEFT, padx=(12, 0))

        # Files
        files = ttk.LabelFrame(root, text="Files", padding=10)
        files.pack(fill=tk.X, **pad)

        self.file_a_var = tk.StringVar()
        self.file_b_var = tk.StringVar()
        self.file_a_label = ttk.Label(files, text="Subtitle")
        self.file_b_label = ttk.Label(files, text="Subtitle 2")

        self.file_a_label.grid(row=0, column=0, sticky=tk.W, pady=4)
        ttk.Entry(files, textvariable=self.file_a_var).grid(row=0, column=1, sticky=tk.EW, padx=8)
        ttk.Button(files, text="Browse…", command=lambda: self._browse(self.file_a_var)).grid(row=0, column=2)

        self.file_b_label.grid(row=1, column=0, sticky=tk.W, pady=4)
        self.file_b_entry = ttk.Entry(files, textvariable=self.file_b_var)
        self.file_b_entry.grid(row=1, column=1, sticky=tk.EW, padx=8)
        self.file_b_btn = ttk.Button(files, text="Browse…", command=lambda: self._browse(self.file_b_var))
        self.file_b_btn.grid(row=1, column=2)
        files.columnconfigure(1, weight=1)

        self.file_a_var.trace_add("write", lambda *_: self._schedule_preview())
        self.file_b_var.trace_add("write", lambda *_: self._schedule_preview())

        # Options
        opts = ttk.LabelFrame(root, text="Options", padding=10)
        opts.pack(fill=tk.X, **pad)

        ttk.Label(opts, text="Source lang").grid(row=0, column=0, sticky=tk.W, pady=4)
        self.source_lang = tk.StringVar(value="auto")
        self.source_lang_box = ttk.Combobox(
            opts,
            textvariable=self.source_lang,
            values=["auto"] + [code for code, _ in TARGET_LANG_CHOICES],
            width=14,
        )
        self.source_lang_box.grid(row=0, column=1, sticky=tk.W, padx=8)
        self.source_lang_box.bind("<<ComboboxSelected>>", lambda *_: self._schedule_preview())
        self.source_lang.trace_add("write", lambda *_: self._schedule_preview())

        ttk.Label(opts, text="Target lang").grid(row=0, column=2, sticky=tk.W, padx=(16, 0))
        self.target_lang = tk.StringVar(value="zh-CN")
        self.target_lang_labels = {label: code for code, label in TARGET_LANG_CHOICES}
        self.target_lang_by_code = {code: label for code, label in TARGET_LANG_CHOICES}
        self.target_lang_box = ttk.Combobox(
            opts,
            values=[label for _, label in TARGET_LANG_CHOICES],
            state="readonly",
            width=28,
        )
        self.target_lang_box.set(self.target_lang_by_code["zh-CN"])
        self.target_lang_box.grid(row=0, column=3, sticky=tk.W, padx=8)
        self.target_lang_box.bind("<<ComboboxSelected>>", self._on_target_picked)

        self.source_detect_label = ttk.Label(opts, text="Source: auto-detect when a file is loaded")
        self.source_detect_label.grid(row=4, column=0, columnspan=4, sticky=tk.W, pady=(4, 0))

        ttk.Label(opts, text="Line order").grid(row=1, column=0, sticky=tk.W, pady=4)
        self.order = tk.StringVar(value="source-top")
        order_box = ttk.Combobox(
            opts,
            textvariable=self.order,
            values=["source-top", "target-top"],
            state="readonly",
            width=14,
        )
        order_box.grid(row=1, column=1, sticky=tk.W, padx=8)
        order_box.bind("<<ComboboxSelected>>", lambda *_: self._schedule_preview())

        ttk.Label(opts, text="Dual format").grid(row=1, column=2, sticky=tk.W, padx=(16, 0))
        self.dual_format = tk.StringVar(value="srt")
        ttk.Combobox(
            opts,
            textvariable=self.dual_format,
            values=["srt", "ass"],
            state="readonly",
            width=8,
        ).grid(row=1, column=3, sticky=tk.W, padx=8)

        ttk.Label(opts, text="Dual layout").grid(row=2, column=0, sticky=tk.W, pady=4)
        self.dual_layout = tk.StringVar(value="stacked")
        layout_box = ttk.Combobox(
            opts,
            textvariable=self.dual_layout,
            values=["stacked", "single-line"],
            state="readonly",
            width=14,
        )
        layout_box.grid(row=2, column=1, sticky=tk.W, padx=8)
        layout_box.bind("<<ComboboxSelected>>", lambda *_: self._schedule_preview())

        ttk.Label(opts, text="Model").grid(row=2, column=2, sticky=tk.W, padx=(16, 0))
        self.model = tk.StringVar(value=DEFAULT_MODEL)
        model_values = list(MODEL_CHOICES)
        if DEFAULT_MODEL not in model_values:
            model_values.insert(0, DEFAULT_MODEL)
        self.model_box = ttk.Combobox(
            opts,
            textvariable=self.model,
            values=model_values,
            width=28,
        )
        self.model_box.grid(row=2, column=3, sticky=tk.W, padx=8)

        ttk.Label(opts, text="Context").grid(row=3, column=0, sticky=tk.NW, pady=4)
        self.context = tk.Text(opts, height=2, width=50, wrap=tk.WORD)
        self.context.grid(row=3, column=1, columnspan=3, sticky=tk.EW, padx=8, pady=4)
        opts.columnconfigure(3, weight=1)

        # Merge sync options (only meaningful in Merge mode).
        self.sync_frame = ttk.LabelFrame(root, text="Merge sync", padding=10)
        self.auto_shift = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.sync_frame,
            text="Auto-shift (estimate global sync offset)",
            variable=self.auto_shift,
            command=lambda: self._schedule_preview(delay_ms=100),
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W)

        ttk.Label(self.sync_frame, text="Shift ms (Subtitle 2)").grid(row=1, column=0, sticky=tk.W, pady=4)
        self.shift_ms = tk.IntVar(value=0)
        self.shift_spin = ttk.Spinbox(
            self.sync_frame,
            from_=-600000,
            to=600000,
            increment=100,
            textvariable=self.shift_ms,
            width=12,
            command=lambda: self._schedule_preview(delay_ms=100),
        )
        self.shift_spin.grid(row=1, column=1, sticky=tk.W, padx=8)
        self.shift_ms.trace_add("write", lambda *_: self._schedule_preview(delay_ms=400))

        self.drop_unmatched = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.sync_frame,
            text="Drop unmatched Subtitle 2 cues",
            variable=self.drop_unmatched,
            command=lambda: self._schedule_preview(delay_ms=100),
        ).grid(row=2, column=0, columnspan=2, sticky=tk.W)

        # Actions
        actions = ttk.Frame(root)
        self.actions = actions
        actions.pack(fill=tk.X, **pad)
        self.run_btn = ttk.Button(actions, text="Run", command=self._run)
        self.run_btn.pack(side=tk.LEFT)
        self.cancel_btn = ttk.Button(actions, text="Cancel", command=self._cancel, state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(actions, text="Refresh preview", command=self._refresh_preview).pack(side=tk.LEFT, padx=8)
        ttk.Button(actions, text="Open output folder", command=self._open_folder).pack(side=tk.LEFT, padx=8)
        self.status = ttk.Label(actions, text="Ready")
        self.status.pack(side=tk.RIGHT)
        self.progress = ttk.Progressbar(actions, mode="determinate", length=180)
        self.progress.pack(side=tk.RIGHT, padx=10)

        # Preview (default) + Log tabs
        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True, **pad)

        preview_frame = ttk.Frame(notebook, padding=6)
        log_frame = ttk.Frame(notebook, padding=6)
        notebook.add(preview_frame, text="Preview")
        notebook.add(log_frame, text="Log")
        self.notebook = notebook

        self.preview_meta = ttk.Label(preview_frame, text="Load a subtitle file to preview cues and timing.")
        self.preview_meta.pack(fill=tk.X, pady=(0, 4))

        player_toggle = ttk.Frame(preview_frame)
        player_toggle.pack(fill=tk.X, pady=(0, 4))
        self.show_player_preview = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            player_toggle,
            text="Show video preview",
            variable=self.show_player_preview,
            command=self._on_player_preview_toggle,
        ).pack(side=tk.LEFT)
        self.live_preview = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            player_toggle,
            text="Live AI sample preview",
            variable=self.live_preview,
            command=lambda: self._schedule_preview(delay_ms=100),
        ).pack(side=tk.LEFT, padx=(16, 0))

        # Fake player stage — how dual lines would look over video (opt-in).
        self.player_stage = ttk.LabelFrame(preview_frame, text="Player look", padding=4)
        self.player_canvas = tk.Canvas(self.player_stage, height=96, bg="#141414", highlightthickness=0)
        self.player_canvas.pack(fill=tk.X)
        self.player_canvas.bind("<Configure>", lambda e: self._redraw_player())

        nav = ttk.Frame(self.player_stage)
        nav.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(nav, text="◀ Prev cue", command=self._player_prev).pack(side=tk.LEFT)
        ttk.Button(nav, text="Next cue ▶", command=self._player_next).pack(side=tk.LEFT, padx=6)
        self.player_cue_label = ttk.Label(nav, text="No cue")
        self.player_cue_label.pack(side=tk.LEFT, padx=10)
        self._player_cues: list[tuple[str, str, str]] = []  # (top, bottom, time_label)
        self._player_index = 0

        list_wrap = ttk.Frame(preview_frame)
        self._preview_list_wrap = list_wrap
        list_wrap.pack(fill=tk.BOTH, expand=True)
        self.preview = tk.Text(list_wrap, wrap=tk.WORD, state=tk.DISABLED, font=("Consolas", 10), height=10)
        preview_scroll = ttk.Scrollbar(list_wrap, command=self.preview.yview)
        self.preview.configure(yscrollcommand=preview_scroll.set)
        self.preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        preview_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.preview.tag_configure("time", foreground="#555555")
        self.preview.tag_configure("ok", foreground="#0a7a32")
        self.preview.tag_configure("warn", foreground="#a15c00")
        self.preview.tag_configure("miss", foreground="#a10000")
        self.preview.tag_configure("header", font=("Consolas", 10, "bold"))

        self.log = tk.Text(log_frame, wrap=tk.WORD, state=tk.DISABLED)
        log_scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=log_scroll.set)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._on_mode_change()

    def _on_mode_change(self):
        merge = self.mode.get() == "merge"
        state = tk.NORMAL if merge else tk.DISABLED
        self.file_b_entry.configure(state=state)
        self.file_b_btn.configure(state=state)
        if merge:
            self.file_a_label.configure(text="Subtitle 1")
            self.file_b_label.configure(text="Subtitle 2")
            self.sync_frame.pack(fill=tk.X, padx=10, pady=6, before=self.actions)
        else:
            self.file_a_label.configure(text="Subtitle")
            self.file_b_label.configure(text="Subtitle 2")
            self.sync_frame.pack_forget()
        self._schedule_preview()

    def _on_target_picked(self, *_):
        label = self.target_lang_box.get()
        code = self.target_lang_labels.get(label)
        if code:
            self.target_lang.set(code)
        # Re-run translate sample preview for the newly chosen target.
        if self.mode.get() == "translate":
            self._schedule_preview(delay_ms=400)

    def _resolved_source_lang(self, subs) -> str:
        selected = (self.source_lang.get() or "auto").strip()
        if selected.lower() in {"", "auto", "detect"}:
            return detect_language(subs)
        return selected

    def _update_source_detect_label(self, code: str | None, auto: bool):
        if not code:
            self.source_detect_label.configure(text="Source: auto-detect when a file is loaded")
            return
        name = lang_name(code)
        if auto:
            self.source_detect_label.configure(text=f"Detected source: {name} ({code})")
        else:
            self.source_detect_label.configure(text=f"Source: {name} ({code})")

    def _player_font(self, size: int = 16, bold: bool = False):
        weight = "bold" if bold else "normal"
        # Prefer a CJK-capable UI font that is actually installed.
        preferred = ("Microsoft YaHei UI", "Microsoft YaHei", "Noto Sans SC", "PingFang SC", "Segoe UI", "Arial")
        try:
            available = set(tkfont.families())
        except Exception:
            available = set()
        for family in preferred:
            if family in available:
                return (family, size, weight)
        return ("Segoe UI" if "Segoe UI" in available else "Arial", size, weight)

    def _on_player_preview_toggle(self):
        if self.show_player_preview.get():
            self.player_stage.pack(fill=tk.X, pady=(0, 6), before=self._preview_list_wrap)
            self.after_idle(self._redraw_player)
        else:
            self.player_stage.pack_forget()

    def _set_player_cues(self, cues: list[tuple[str, str, str]], index: int = 0):
        """cues: list of (top_line, bottom_line, time_label). bottom may be ''."""
        self._player_cues = cues or []
        self._player_index = max(0, min(index, len(self._player_cues) - 1)) if self._player_cues else 0
        if self.show_player_preview.get():
            self._redraw_player()

    def _player_prev(self):
        if not self._player_cues:
            return
        self._player_index = (self._player_index - 1) % len(self._player_cues)
        self._redraw_player()

    def _player_next(self):
        if not self._player_cues:
            return
        self._player_index = (self._player_index + 1) % len(self._player_cues)
        self._redraw_player()

    def _draw_outlined_text(self, canvas, x, y, text, font, fill="#ffffff"):
        if not text:
            return
        # Soft black outline like a typical video player.
        for dx, dy in (
            (-2, 0),
            (2, 0),
            (0, -2),
            (0, 2),
            (-1, -1),
            (1, -1),
            (-1, 1),
            (1, 1),
        ):
            canvas.create_text(x + dx, y + dy, text=text, fill="#000000", font=font, anchor="s")
        canvas.create_text(x, y, text=text, fill=fill, font=font, anchor="s")

    def _redraw_player(self):
        if not self.show_player_preview.get():
            return
        canvas = self.player_canvas
        canvas.delete("all")
        w = max(canvas.winfo_width(), 320)
        h = max(canvas.winfo_height(), 90)

        # Compact strip — just enough room for the subtitle lines.
        canvas.create_rectangle(0, 0, w, h, fill="#1a1a1a", outline="")
        canvas.create_rectangle(4, 4, w - 4, h - 4, fill="#222222", outline="#333333")

        if not self._player_cues:
            canvas.create_text(
                w // 2,
                h // 2,
                text="Load a file to preview how subs look on screen",
                fill="#777777",
                font=("Segoe UI", 10),
            )
            self.player_cue_label.configure(text="No cue")
            return

        top, bottom, time_label = self._player_cues[self._player_index]
        layout = self.dual_layout.get() or "stacked"
        self.player_cue_label.configure(
            text=f"Cue {self._player_index + 1}/{len(self._player_cues)}  ·  {time_label}"
        )

        cx = w // 2
        if layout == "single-line":
            line = f"{top}  |  {bottom}" if bottom else top
            self._draw_outlined_text(canvas, cx, h - 18, line, self._player_font(13), fill="#ffffff")
        else:
            if bottom:
                self._draw_outlined_text(canvas, cx, h - 40, top, self._player_font(13), fill="#ffffff")
                self._draw_outlined_text(canvas, cx, h - 16, bottom, self._player_font(13), fill="#ffffff")
            else:
                self._draw_outlined_text(canvas, cx, h - 18, top, self._player_font(13), fill="#ffffff")

    def _browse(self, var: tk.StringVar):
        path = filedialog.askopenfilename(title="Choose file", filetypes=SUB_TYPES)
        if path:
            var.set(path)

    def _set_text(self, widget: tk.Text, content: str, tagged_chunks=None):
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        if tagged_chunks:
            for text, tag in tagged_chunks:
                widget.insert(tk.END, text, tag if tag else ())
        else:
            widget.insert(tk.END, content)
        widget.configure(state=tk.DISABLED)

    def _append_log(self, text: str):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, text)
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _drain_log(self):
        try:
            while True:
                self._append_log(self.log_q.get_nowait())
        except queue.Empty:
            pass
        self.after(100, self._drain_log)

    def _set_busy(self, busy: bool):
        self.run_btn.configure(state=tk.DISABLED if busy else tk.NORMAL)
        self.cancel_btn.configure(state=tk.NORMAL if busy else tk.DISABLED)
        self.status.configure(text="Working…" if busy else "Ready")
        self.progress.configure(value=0, maximum=100)

    def _cancel(self):
        self._cancel_event.set()
        self.status.configure(text="Cancelling…")

    def _progress_cb(self, done: int, total: int):
        def apply():
            self.progress.configure(maximum=max(1, total), value=done)
            if total:
                self.status.configure(text=f"Working… {done}/{total} batches")

        self.after(0, apply)

    def _open_folder(self):
        path = self.file_a_var.get().strip().strip('"') or self.file_b_var.get().strip().strip('"')
        folder = Path(path).parent if path else Path.cwd()
        if not folder.exists():
            messagebox.showinfo("Open folder", "Pick a file first.")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(folder))  # noqa: S606 — Windows explorer open
            elif sys.platform == "darwin":
                subprocess.run(["open", str(folder)], check=False)
            else:
                subprocess.run(["xdg-open", str(folder)], check=False)
        except Exception as e:
            messagebox.showerror("Open folder", f"Could not open folder:\n{e}")

    def _schedule_preview(self, delay_ms: int = 250):
        if self._preview_job is not None:
            self.after_cancel(self._preview_job)
        self._preview_job = self.after(delay_ms, self._refresh_preview)

    def _load_sub_preview(self, path: Path):
        if path.suffix.lower() not in SUPPORTED_EXTS:
            return None, f"Unsupported file type: {path.suffix}"
        return load_subs(path), None

    def _refresh_preview(self):
        self._preview_job = None
        # Invalidate any in-flight sample-preview worker so stale results are dropped.
        self._preview_token += 1
        path_a = self.file_a_var.get().strip().strip('"')
        path_b = self.file_b_var.get().strip().strip('"')
        mode = self.mode.get()

        if not path_a:
            self.preview_meta.configure(text="Load a subtitle file to preview cues and timing.")
            self._set_text(self.preview, "")
            self._set_player_cues([])
            return

        a = Path(path_a)
        if not a.exists():
            self.preview_meta.configure(text=f"Not found: {a.name}")
            self._set_text(self.preview, "")
            self._set_player_cues([])
            return

        try:
            if mode == "merge" and path_b:
                self._preview_merge(a, Path(path_b))
            else:
                self._preview_single(a)
        except Exception as e:
            self.preview_meta.configure(text=f"Preview error: {e}")
            self._set_text(self.preview, str(e))

    def _preview_single(self, path: Path):
        subs, note = self._load_sub_preview(path)
        if note:
            self.preview_meta.configure(text=note)
            self._set_text(self.preview, note)
            self._update_source_detect_label(None, False)
            return
        script = detect_script(subs)
        auto = (self.source_lang.get() or "auto").strip().lower() in {"", "auto", "detect"}
        detected = self._resolved_source_lang(subs)
        self._update_source_detect_label(detected, auto=auto)

        if self.mode.get() != "translate":
            self.preview_meta.configure(
                text=f"{path.name}  ·  {len(subs)} cues  ·  script={script}  ·  lang={detected}"
            )
            self._set_text(self.preview, _format_subs_preview(subs, path.name))
            cues = []
            for event in list(subs)[:PREVIEW_CUES]:
                text = (event.plaintext or "").replace("\n", " ")
                cues.append((text, "", f"{_fmt_ts(event.start)} → {_fmt_ts(event.end)}"))
            self._set_player_cues(cues, 0)
            return

        # Translate mode: always show the source cue list. Only call the API for a
        # live sample when the user opts in (Live AI sample preview checkbox).
        label = self.target_lang_box.get()
        if label in self.target_lang_labels:
            self.target_lang.set(self.target_lang_labels[label])
        target = self.target_lang.get().strip() or "zh-CN"
        order = self.order.get() or "source-top"

        source_text = _format_subs_preview(subs, f"SOURCE ({detected})", limit=PREVIEW_CUES)
        source_cues = []
        for event in list(subs)[:PREVIEW_CUES]:
            source_cues.append(
                ((event.plaintext or "").replace("\n", " "), "", f"{_fmt_ts(event.start)} → {_fmt_ts(event.end)}")
            )

        if not self.live_preview.get():
            self.preview_meta.configure(
                text=f"{path.name}  ·  {len(subs)} cues  ·  {detected} → {target}  ·  (enable 'Live AI sample preview' to sample translations)"
            )
            self._set_text(self.preview, source_text)
            self._set_player_cues(source_cues, 0)
            return

        self.preview_meta.configure(
            text=(
                f"{path.name}  ·  {len(subs)} cues  ·  {detected} → {target}  ·  "
                f"translating sample preview…"
            )
        )
        self._set_text(
            self.preview,
            _format_subs_preview(subs, f"SOURCE ({detected})", limit=TRANSLATE_PREVIEW_CUES)
            + "\n\n(Waiting for translation sample…)\n",
        )
        self._set_player_cues(source_cues, 0)

        if not os.environ.get("NVIDIA_API_KEY"):
            self.preview_meta.configure(
                text=f"{path.name}  ·  {detected} → {target}  ·  set NVIDIA_API_KEY in .env for live translate preview"
            )
            self._set_text(self.preview, source_text)
            return

        token = self._preview_token
        model = self.model.get().strip() or DEFAULT_MODEL
        sample_events = list(subs)[:TRANSLATE_PREVIEW_CUES]
        sample_lines = [e.plaintext for e in sample_events]
        context = self.context.get("1.0", tk.END).strip()

        def work():
            try:
                client = get_client()
                translations = translate_batch(
                    client,
                    model,
                    sample_lines,
                    detected,
                    target,
                    context,
                    start_index=0,
                    max_retries=2,
                )
                self.after(
                    0,
                    lambda t=translations: self._apply_translate_preview(
                        token, path, sample_events, sample_lines, t, detected, target, order
                    ),
                )
            except Exception as e:
                err = e
                self.after(
                    0,
                    lambda err=err: self._apply_translate_preview_error(token, path, detected, target, err),
                )

        self._translate_preview_worker = threading.Thread(target=work, daemon=True)
        self._translate_preview_worker.start()

    def _apply_translate_preview_error(self, token: int, path: Path, src: str, tgt: str, err: Exception):
        if token != self._preview_token:
            return
        self.preview_meta.configure(text=f"{path.name}  ·  {src} → {tgt}  ·  preview failed: {err}")

    def _apply_translate_preview(
        self,
        token: int,
        path: Path,
        events,
        source_lines,
        translations,
        src: str,
        tgt: str,
        order: str,
    ):
        if token != self._preview_token:
            return

        layout = self.dual_layout.get() or "stacked"
        self.preview_meta.configure(
            text=f"{path.name}  ·  sample dual preview  ·  {src} → {tgt}  ·  layout={layout}"
        )

        player_cues = []
        chunks = []
        chunks.append((f"TRANSLATE PREVIEW ({len(translations)} cues → {lang_name(tgt)})\n", "header"))
        chunks.append(("Player stage above shows how a cue looks. Use Prev/Next to flip samples.\n\n", "time"))

        for event, original, translated in zip(events, source_lines, translations):
            if order == "target-top":
                top, bottom = translated, original
            else:
                top, bottom = original, translated
            time_label = f"{_fmt_ts(event.start)} → {_fmt_ts(event.end)}"
            player_cues.append((top, bottom, time_label))
            chunks.append((f"{time_label}\n", "time"))
            if layout == "single-line":
                chunks.append((f"  {top}  |  {bottom}\n\n", "ok"))
            else:
                chunks.append((f"  {top}\n", "ok"))
                chunks.append((f"  {bottom}\n\n", "ok"))

        chunks.append(("=" * 56 + "\n", "time"))
        chunks.append(("SOURCE FILE (same sample)\n", "header"))
        for event, original in zip(events, source_lines):
            chunks.append((f"{_fmt_ts(event.start)} → {_fmt_ts(event.end)}\n", "time"))
            chunks.append((f"  {original}\n\n", None))

        self._set_player_cues(player_cues, 0)
        self._set_text(self.preview, "", tagged_chunks=chunks)
        self.notebook.select(0)

    def _preview_merge(self, path_a: Path, path_b: Path):
        if not path_b.exists():
            self.preview_meta.configure(text=f"Not found: {path_b.name}")
            self._set_text(self.preview, "")
            return

        subs_a, note_a = self._load_sub_preview(path_a)
        if note_a:
            self.preview_meta.configure(text=note_a)
            self._set_text(self.preview, note_a)
            return
        subs_b, note_b = self._load_sub_preview(path_b)
        if note_b:
            self.preview_meta.configure(text=note_b)
            self._set_text(self.preview, note_b)
            return

        lang_a, lang_b = detect_language(subs_a), detect_language(subs_b)
        self._update_source_detect_label(lang_a, auto=True)
        self.source_detect_label.configure(
            text=f"Detected: Subtitle 1={lang_name(lang_a)} ({lang_a})  ·  Subtitle 2={lang_name(lang_b)} ({lang_b})"
        )

        # Manual --shift-ms targets Subtitle 2 (subs_b), applied before spine swap.
        try:
            manual_shift = int(self.shift_ms.get())
        except (tk.TclError, ValueError):
            manual_shift = 0
        if manual_shift:
            subs_b = _shift_subs(subs_b, manual_shift)

        script_a, script_b = detect_script(subs_a), detect_script(subs_b)
        if script_a == "cjk" and script_b == "latin":
            primary, secondary = subs_b, subs_a
            spine_name, other_name = path_b.name, path_a.name
        else:
            primary, secondary = subs_a, subs_b
            spine_name, other_name = path_a.name, path_b.name

        if self.auto_shift.get():
            estimated = estimate_time_shift(primary, secondary)
            if estimated:
                secondary = _shift_subs(secondary, estimated)

        dual = merge_subs(
            primary,
            secondary,
            order=self.order.get() or "source-top",
            min_overlap_ms=80,
            include_unmatched=not self.drop_unmatched.get(),
        )
        dual_count = sum(1 for e in dual if "\n" in (e.plaintext or ""))
        single_count = len(dual) - dual_count
        match_pct = (100.0 * dual_count / len(primary)) if primary else 0.0

        self.preview_meta.configure(
            text=(
                f"Combined preview  ·  spine={spine_name} + {other_name}  ·  "
                f"{dual_count}/{len(primary)} paired ({match_pct:.0f}%)"
                + (f"  ·  {single_count} unpaired in spine" if single_count else "")
            )
        )

        chunks = []
        chunks.append((f"COMBINED PREVIEW (first {PREVIEW_CUES} cues)\n", "header"))
        chunks.append(("Player stage above shows how a cue looks. Use Prev/Next to flip samples.\n\n", "time"))

        player_cues = []
        for i, event in enumerate(dual):
            if i >= PREVIEW_CUES:
                chunks.append((f"… ({len(dual) - PREVIEW_CUES} more cues)\n", "time"))
                break
            paired = "\n" in (event.plaintext or "")
            tag = "ok" if paired else "miss"
            time_label = f"{_fmt_ts(event.start)} → {_fmt_ts(event.end)}"
            parts = [p for p in (event.plaintext or "").split("\n") if p]
            if len(parts) >= 2:
                player_cues.append((parts[0], " ".join(parts[1:]), time_label))
            elif parts:
                player_cues.append((parts[0], "", time_label))
            chunks.append((f"{time_label}\n", "time"))
            for line in (event.plaintext or "").splitlines() or [""]:
                chunks.append((f"  {line}\n", tag))
            chunks.append(("\n", None))

        chunks.append(("\n" + "=" * 56 + "\n", "time"))
        chunks.append(("SUBTITLE 1 (sample)\n", "header"))
        chunks.append((_format_subs_preview(subs_a, path_a.name, limit=8) + "\n", None))
        chunks.append(("SUBTITLE 2 (sample)\n", "header"))
        chunks.append((_format_subs_preview(subs_b, path_b.name, limit=8) + "\n", None))

        self._set_player_cues(player_cues, 0)
        self._set_text(self.preview, "", tagged_chunks=chunks)
        self.notebook.select(0)

    def _make_args(self) -> Namespace:
        # Keep target_lang in sync with the picker label.
        label = self.target_lang_box.get()
        if label in self.target_lang_labels:
            self.target_lang.set(self.target_lang_labels[label])
        try:
            shift_ms = int(self.shift_ms.get())
        except (tk.TclError, ValueError):
            shift_ms = 0
        return Namespace(
            source_lang=self.source_lang.get().strip() or "auto",
            target_lang=self.target_lang.get().strip() or "zh-CN",
            order=self.order.get() or "source-top",
            model=self.model.get().strip() or DEFAULT_MODEL,
            batch_size=20,
            workers=6,
            context=self.context.get("1.0", tk.END).strip(),
            merge=None,
            output=None,
            shift_ms=shift_ms,
            auto_shift=bool(self.auto_shift.get()),
            min_overlap_ms=80,
            drop_unmatched=bool(self.drop_unmatched.get()),
            format=self.dual_format.get() or "srt",
            layout=self.dual_layout.get() or "stacked",
        )

    def _run(self):
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("Busy", "Already running.")
            return

        mode = self.mode.get()
        path_a = self.file_a_var.get().strip().strip('"')
        path_b = self.file_b_var.get().strip().strip('"')

        if not path_a:
            messagebox.showerror("Missing file", "Choose an input file.")
            return
        if mode == "merge" and not path_b:
            messagebox.showerror("Missing file", "Choose both subtitle files to merge.")
            return
        if not Path(path_a).exists():
            messagebox.showerror("Not found", f"File not found:\n{path_a}")
            return
        if mode == "merge" and not Path(path_b).exists():
            messagebox.showerror("Not found", f"File not found:\n{path_b}")
            return

        args = self._make_args()
        self.notebook.select(1)
        self._append_log(f"\n--- {mode} ---\n")
        self._cancel_event.clear()
        self._set_busy(True)

        def work():
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = QueueWriter(self.log_q)
            os.environ["PROMPT_ON_EXIT"] = "0"
            try:
                if mode == "merge":
                    process_merge(args, Path(path_a), Path(path_b))
                else:
                    if mode == "translate" and not os.environ.get("NVIDIA_API_KEY"):
                        raise RuntimeError(
                            "NVIDIA_API_KEY is not set. Put it in a .env file next to dual_subs.py."
                        )
                    client = get_client()
                    process_file(
                        client,
                        args,
                        Path(path_a),
                        cancel_event=self._cancel_event,
                        progress_cb=self._progress_cb,
                    )
                print("\nDone.")
            except Exception as e:
                print(f"\n[error] {e}\n")
            finally:
                sys.stdout, sys.stderr = old_out, old_err
                self.after(0, lambda: self._set_busy(False))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()


def run_ui(initial_paths=None):
    if not _HAS_TK:
        raise RuntimeError(
            "The desktop UI requires tkinter, which is not available in this Python install.\n"
            "Use the CLI instead, e.g.: python dual_subs.py movie.srt"
        )
    app = DualSubsApp()

    paths = [Path(p) for p in (initial_paths or []) if p]
    if paths:
        app.file_a_var.set(str(paths[0]))
        # A plausible second subtitle file suggests a merge.
        if len(paths) > 1 and paths[1].suffix.lower() in SUPPORTED_EXTS:
            app.mode.set("merge")
            app._on_mode_change()
            app.file_b_var.set(str(paths[1]))

    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        app._append_log("Tip: create a .env with NVIDIA_API_KEY=... for Translate mode.\n")
    else:
        app._append_log("NVIDIA API key loaded.\n")
    app.mainloop()



def parse_args():
    parser = argparse.ArgumentParser(description="Create dual-language subtitles using an NVIDIA NIM model.")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Subtitle file(s) (.srt/.vtt/.ass/.ssa)",
    )
    parser.add_argument(
        "--merge",
        nargs=2,
        metavar=("FILE_1", "FILE_2"),
        help="Fuse two existing subtitle files into one dual file by time sync",
    )
    parser.add_argument("-o", "--output", default=None, help="Output path for --merge (default: next to first/spine file)")
    parser.add_argument(
        "--shift-ms",
        type=int,
        default=0,
        help="With --merge: shift FILE_2 / Subtitle 2 by this many milliseconds (applied before spine selection; auto-shift is applied separately)",
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
        "--ui",
        action="store_true",
        help="Launch the desktop UI (default when no input paths are given)",
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

    if args.ui or (not args.paths and not args.merge):
        os.environ["PROMPT_ON_EXIT"] = "0"
        run_ui(initial_paths=args.paths)
        return

    if not args.paths:
        print("No files given. Exiting.")
        return

    try:
        client = get_client()
    except RuntimeError as e:
        sys.exit(f"ERROR: {e}")

    failures = 0
    for raw_path in args.paths:
        path = Path(raw_path)
        if not path.exists():
            print(f"\n=== {raw_path} ===\n  [error] file not found")
            failures += 1
            continue
        try:
            if not process_file(client, args, path):
                failures += 1
        except Exception as e:
            print(f"  [error] failed to process {path.name}: {e}", file=sys.stderr)
            failures += 1

    print("\nDone.")
    if failures:
        sys.exit(1)


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
