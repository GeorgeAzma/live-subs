"""Transparent always-on-top subtitle overlay for Windows.

Usage
-----
    overlay = SubtitleOverlay()
    overlay.on_partial("Hello...")
    overlay.run()  # blocks until window closes
"""

import os
import queue
import tkinter as tk
from ctypes import windll

from main import TextHandler

# ── DPI helpers ──────────────────────────────────────────────────────────


def _enable_dpi_awareness():
    """Mark this process DPI-aware so Tkinter renders at native resolution."""
    for func in (
        ("shcore", "SetProcessDpiAwareness", 2),  # PerMonitorV2  (Win 10+)
        ("user32", "SetProcessDPIAware", None),  # system DPI   (Win 8-)
    ):
        try:
            getattr(windll, func[0]).__getattr__(func[1])(func[2])
            return
        except Exception:
            continue


def _get_dpi_scale() -> float:
    """Return (monitor DPI) / 96."""
    try:
        return windll.user32.GetDpiForSystem() / 96.0
    except Exception:
        return 1.0


# ── Overlay window ───────────────────────────────────────────────────────


class SubtitleOverlay(TextHandler):
    """Always-on-top transparent subtitle window.

    Displays translation text at the bottom of the screen with a
    dark drop shadow for readability.  Drag by left-clicking any
    blank area.  Press Escape to close.
    """

    _TRANSPARENT = "#000000"
    _SHADOW = "#111111"

    def __init__(self, font_size: int = 28):
        _enable_dpi_awareness()
        scale = _get_dpi_scale()

        self.root = tk.Tk()
        self.root.withdraw()
        self.root.title("Transcriber Subtitles")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        w = sw
        h = max(150, int(150 * scale))
        self.root.geometry(f"{w}x{h}+0+{sh - h - int(40 * scale)}")
        self.root.configure(bg=self._TRANSPARENT)

        self._font_size = font_size
        self._scale = scale
        self._canvas_width = w
        self._canvas_height = h
        self._soff = max(3, int(3 * scale))
        self._cx = w // 2
        self._cy = h // 2

        self._canvas = tk.Canvas(self.root, bg=self._TRANSPARENT, highlightthickness=0)
        self._canvas.pack(expand=True, fill="both")

        fs = int(self._font_size * self._scale)
        font = ("Segoe UI", fs, "bold")
        wrap = self._canvas_width - int(100 * self._scale)

        self._shadow_id = self._canvas.create_text(
            self._cx + self._soff, self._cy + self._soff,
            fill=self._SHADOW, font=font, anchor="center", justify="center", width=wrap,
        )
        self._text_id = self._canvas.create_text(
            self._cx, self._cy,
            fill="white", font=font, anchor="center", justify="center", width=wrap,
        )

        self._text = ""
        self._queue: queue.Queue[str] = queue.Queue()

        idle = "Listening..."
        self._canvas.itemconfig(self._shadow_id, text=idle)
        self._canvas.itemconfig(self._text_id, text=idle, fill="#555555")

        try:
            self.root.attributes("-transparentcolor", self._TRANSPARENT)
        except Exception:
            self.root.attributes("-alpha", 0.88)

        self.root.deiconify()
        self.root.update_idletasks()
        self._make_styling()
        self._enable_drag()
        self._enable_close()
        self._enable_scroll()
        self.root.after(50, self._poll)

    def _make_styling(self):
        try:
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x80000
            WS_EX_TOOLWINDOW = 0x80
            hwnd = windll.user32.GetAncestor(self.root.winfo_id(), 2)
            current = windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            windll.user32.SetWindowLongW(
                hwnd,
                GWL_EXSTYLE,
                current | WS_EX_LAYERED | WS_EX_TOOLWINDOW,
            )
        except Exception:
            pass

    # ── mouse drag ─────────────────────────────────────────────────

    def _enable_drag(self):
        self._drag_start = None
        self._canvas.bind("<Button-1>", self._on_drag_start)
        self._canvas.bind("<B1-Motion>", self._on_drag_move)

    def _on_drag_start(self, event):
        self._drag_start = (event.x_root, event.y_root)
        self.root.focus_force()

    def _on_drag_move(self, event):
        if self._drag_start is None:
            return
        dx = event.x_root - self._drag_start[0]
        dy = event.y_root - self._drag_start[1]
        x = self.root.winfo_x() + dx
        y = self.root.winfo_y() + dy
        self.root.geometry(f"+{x}+{y}")
        self._drag_start = (event.x_root, event.y_root)

    # ── font size via scroll ───────────────────────────────────────

    def _enable_scroll(self):
        self._canvas.bind("<MouseWheel>", self._on_scroll)
        self._canvas.bind("<Button-4>", self._on_scroll_up)
        self._canvas.bind("<Button-5>", self._on_scroll_down)

    def _on_scroll(self, event):
        if event.delta > 0:
            self._set_font_size(self._font_size + 2)
        else:
            self._set_font_size(self._font_size - 2)

    def _on_scroll_up(self, event):
        self._set_font_size(self._font_size + 2)

    def _on_scroll_down(self, event):
        self._set_font_size(self._font_size - 2)

    def _set_font_size(self, new_size):
        new_size = max(8, min(120, new_size))
        if new_size == self._font_size:
            return
        self._font_size = new_size
        fs = int(self._font_size * self._scale)
        font = ("Segoe UI", fs, "bold")
        wrap = self._canvas_width - int(100 * self._scale)
        self._canvas.itemconfig(self._shadow_id, font=font, width=wrap)
        self._canvas.itemconfig(self._text_id, font=font, width=wrap)

    # ── close via Escape ───────────────────────────────────────────

    def _enable_close(self):
        self.root.bind("<Escape>", lambda e: os._exit(0))

    # ── TextHandler interface (called from pipeline thread) ────────

    def on_partial(self, text: str):
        self._queue.put(text)

    def on_final(self, text: str):
        self._queue.put(text)

    # ── internal ───────────────────────────────────────────────────

    def _poll(self):
        try:
            while True:
                text = self._queue.get_nowait()
                if text == self._text:
                    continue
                self._text = text
                self._canvas.itemconfig(self._shadow_id, text=text)
                self._canvas.itemconfig(self._text_id, text=text, fill="white")
        except queue.Empty:
            pass
        self.root.after(50, self._poll)

    # ── lifecycle ──────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()

    def stop(self):
        self.root.quit()
