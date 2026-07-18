"""
Simple desktop UI for dual_subs.

Launch:
    python ui.py
    or double-click "Dual Subs UI.bat"
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import tkinter as tk
from argparse import Namespace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import dual_subs as ds

SUB_TYPES = [
    ("Subtitle / video", "*.srt *.vtt *.ass *.ssa *.mkv *.mp4 *.m4v *.avi *.mov *.webm"),
    ("Subtitles", "*.srt *.vtt *.ass *.ssa"),
    ("Videos", "*.mkv *.mp4 *.m4v *.avi *.mov *.webm"),
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


class DualSubsApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dual Subtitles")
        self.geometry("860x700")
        self.minsize(720, 580)

        self.log_q: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self._preview_job = None
        self._preview_token = 0
        self._translate_preview_worker: threading.Thread | None = None
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
            ("extract", "Extract from video"),
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
        self.file_a_label = ttk.Label(files, text="Input")
        self.file_b_label = ttk.Label(files, text="Second language")

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
            values=["auto"] + [code for code, _ in ds.TARGET_LANG_CHOICES],
            width=14,
        )
        self.source_lang_box.grid(row=0, column=1, sticky=tk.W, padx=8)
        self.source_lang_box.bind("<<ComboboxSelected>>", lambda *_: self._schedule_preview())
        self.source_lang.trace_add("write", lambda *_: self._schedule_preview())

        ttk.Label(opts, text="Target lang").grid(row=0, column=2, sticky=tk.W, padx=(16, 0))
        self.target_lang = tk.StringVar(value="zh-CN")
        self.target_lang_labels = {label: code for code, label in ds.TARGET_LANG_CHOICES}
        self.target_lang_by_code = {code: label for code, label in ds.TARGET_LANG_CHOICES}
        self.target_lang_box = ttk.Combobox(
            opts,
            values=[label for _, label in ds.TARGET_LANG_CHOICES],
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

        ttk.Label(opts, text="Context").grid(row=3, column=0, sticky=tk.NW, pady=4)
        self.context = tk.Text(opts, height=2, width=50, wrap=tk.WORD)
        self.context.grid(row=3, column=1, columnspan=3, sticky=tk.EW, padx=8, pady=4)
        opts.columnconfigure(3, weight=1)

        # Actions
        actions = ttk.Frame(root)
        actions.pack(fill=tk.X, **pad)
        self.run_btn = ttk.Button(actions, text="Run", command=self._run)
        self.run_btn.pack(side=tk.LEFT)
        ttk.Button(actions, text="Refresh preview", command=self._refresh_preview).pack(side=tk.LEFT, padx=8)
        ttk.Button(actions, text="Open output folder", command=self._open_folder).pack(side=tk.LEFT, padx=8)
        self.status = ttk.Label(actions, text="Ready")
        self.status.pack(side=tk.RIGHT)

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
        if self.mode.get() == "merge":
            self.file_a_label.configure(text="Language A")
            self.file_b_label.configure(text="Language B")
        elif self.mode.get() == "extract":
            self.file_a_label.configure(text="Video")
            self.file_b_label.configure(text="Second language")
        else:
            self.file_a_label.configure(text="Subtitle / video")
            self.file_b_label.configure(text="Second language")
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
            return ds.detect_language(subs)
        return selected

    def _update_source_detect_label(self, code: str | None, auto: bool):
        if not code:
            self.source_detect_label.configure(text="Source: auto-detect when a file is loaded")
            return
        name = ds.lang_name(code)
        if auto:
            self.source_detect_label.configure(text=f"Detected source: {name} ({code})")
        else:
            self.source_detect_label.configure(text=f"Source: {name} ({code})")

    def _player_font(self, size: int = 16, bold: bool = False):
        weight = "bold" if bold else "normal"
        # Prefer a CJK-capable UI font on Windows so Chinese shows in the mock player.
        for family in ("Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", "Arial"):
            return (family, size, weight)

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
                self._draw_outlined_text(canvas, cx, h - 40, top, self._player_font(14, bold=True), fill="#ffffff")
                self._draw_outlined_text(canvas, cx, h - 16, bottom, self._player_font(12), fill="#dddddd")
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
        self.status.configure(text="Working…" if busy else "Ready")

    def _open_folder(self):
        path = self.file_a_var.get().strip() or self.file_b_var.get().strip()
        folder = Path(path).parent if path else Path.cwd()
        if folder.exists():
            os.startfile(folder)  # noqa: S606 — Windows explorer open
        else:
            messagebox.showinfo("Open folder", "Pick a file first.")

    def _schedule_preview(self, delay_ms: int = 250):
        if self._preview_job is not None:
            self.after_cancel(self._preview_job)
        self._preview_job = self.after(delay_ms, self._refresh_preview)

    def _load_sub_preview(self, path: Path):
        if path.suffix.lower() in ds.VIDEO_EXTS:
            return None, f"{path.name} is a video — extract/translate to see subtitle text."
        if path.suffix.lower() not in ds.SUPPORTED_EXTS:
            return None, f"Unsupported file type: {path.suffix}"
        return ds.load_subs(path), None

    def _refresh_preview(self):
        self._preview_job = None
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
        script = ds.detect_script(subs)
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

        # Translate mode: show source immediately, then sample-translate into target.
        label = self.target_lang_box.get()
        if label in self.target_lang_labels:
            self.target_lang.set(self.target_lang_labels[label])
        target = self.target_lang.get().strip() or "zh-CN"
        order = self.order.get() or "source-top"

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

        if not os.environ.get("NVIDIA_API_KEY"):
            self.preview_meta.configure(
                text=f"{path.name}  ·  {detected} → {target}  ·  set NVIDIA_API_KEY in .env for live translate preview"
            )
            return

        self._preview_token += 1
        token = self._preview_token
        sample_events = list(subs)[:TRANSLATE_PREVIEW_CUES]
        sample_lines = [e.plaintext for e in sample_events]
        context = self.context.get("1.0", tk.END).strip()

        def work():
            try:
                client = ds.get_client()
                translations = ds.translate_batch(
                    client,
                    ds.DEFAULT_MODEL,
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
        chunks.append((f"TRANSLATE PREVIEW ({len(translations)} cues → {ds.lang_name(tgt)})\n", "header"))
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

        lang_a, lang_b = ds.detect_language(subs_a), ds.detect_language(subs_b)
        self._update_source_detect_label(lang_a, auto=True)
        self.source_detect_label.configure(
            text=f"Detected: A={ds.lang_name(lang_a)} ({lang_a})  ·  B={ds.lang_name(lang_b)} ({lang_b})"
        )

        script_a, script_b = ds.detect_script(subs_a), ds.detect_script(subs_b)
        if script_a == "cjk" and script_b == "latin":
            primary, secondary = subs_b, subs_a
            spine_name, other_name = path_b.name, path_a.name
        else:
            primary, secondary = subs_a, subs_b
            spine_name, other_name = path_a.name, path_b.name

        dual = ds.merge_subs(
            primary,
            secondary,
            order=self.order.get() or "source-top",
            min_overlap_ms=80,
            include_unmatched=False,
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
        chunks.append(("RAW FILE A (sample)\n", "header"))
        chunks.append((_format_subs_preview(subs_a, path_a.name, limit=8) + "\n", None))
        chunks.append(("RAW FILE B (sample)\n", "header"))
        chunks.append((_format_subs_preview(subs_b, path_b.name, limit=8) + "\n", None))

        self._set_player_cues(player_cues, 0)
        self._set_text(self.preview, "", tagged_chunks=chunks)
        self.notebook.select(0)

    def _make_args(self) -> Namespace:
        # Keep target_lang in sync with the picker label.
        label = self.target_lang_box.get()
        if label in self.target_lang_labels:
            self.target_lang.set(self.target_lang_labels[label])
        return Namespace(
            source_lang=self.source_lang.get().strip() or "auto",
            target_lang=self.target_lang.get().strip() or "zh-CN",
            order=self.order.get() or "source-top",
            model=ds.DEFAULT_MODEL,
            batch_size=20,
            workers=6,
            sub_stream=None,
            extract_only=self.mode.get() == "extract",
            context=self.context.get("1.0", tk.END).strip(),
            merge=None,
            output=None,
            shift_ms=0,
            auto_shift=False,
            min_overlap_ms=80,
            drop_unmatched=False,
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
            messagebox.showerror("Missing file", "Choose both language files to merge.")
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
        self._set_busy(True)

        def work():
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = QueueWriter(self.log_q)
            os.environ["PROMPT_ON_EXIT"] = "0"
            try:
                if mode == "merge":
                    ds.process_merge(args, Path(path_a), Path(path_b))
                else:
                    if mode == "translate" and not os.environ.get("NVIDIA_API_KEY"):
                        raise RuntimeError(
                            "NVIDIA_API_KEY is not set. Put it in a .env file next to dual_subs.py."
                        )
                    client = None if args.extract_only else ds.get_client()
                    ds.process_file(client, args, Path(path_a))
                print("\nDone.")
            except Exception as e:
                print(f"\n[error] {e}\n")
            finally:
                sys.stdout, sys.stderr = old_out, old_err
                self.after(0, lambda: self._set_busy(False))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()


def main():
    app = DualSubsApp()
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        app._append_log("Tip: create a .env with NVIDIA_API_KEY=... for Translate mode.\n")
    else:
        app._append_log("NVIDIA API key loaded.\n")
    app.mainloop()


if __name__ == "__main__":
    main()
