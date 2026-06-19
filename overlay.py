"""Transparent always-on-top subtitle overlay for Windows.

Usage
-----
    overlay = SubtitleOverlay()
    overlay.on_partial("Hello...")
    overlay.run()  # blocks until window closes
"""

import queue
import tkinter as tk
from ctypes import windll

from main import TextHandler


# ── DPI helpers ──────────────────────────────────────────────────────────

def _enable_dpi_awareness():
    """Mark this process DPI-aware so Tkinter renders at native resolution."""
    for func in (
        ("shcore", "SetProcessDpiAwareness", 2),   # PerMonitorV2  (Win 10+)
        ("user32", "SetProcessDPIAware", None),     # system DPI   (Win 8-)
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
    dark drop shadow for readability.  Click-through enabled —
    mouse events pass through to windows beneath.
    """

    _TRANSPARENT = "#000000"   # fully transparent key
    _SHADOW      = "#222222"   # visible shadow (must differ from _TRANSPARENT)

    def __init__(self, font_size: int = 36):
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
        self.root.geometry(f"{w}x{h}+0+{sh - h - int(60 * scale)}")
        self.root.configure(bg=self._TRANSPARENT)

        fs = int(font_size * scale)
        soff = max(4, int(4 * scale))
        font = ("Segoe UI", fs, "bold")

        self._canvas = tk.Canvas(self.root, bg=self._TRANSPARENT, highlightthickness=0)
        self._canvas.pack(expand=True, fill="both")

        cx, cy = w // 2, h // 2
        base = dict(font=font, anchor="center", justify="center", width=w - int(100 * scale))

        self._shadows = []
        for dx, dy in [(-soff, -soff), (-soff, soff), (soff, -soff), (soff, soff)]:
            sid = self._canvas.create_text(cx + dx, cy + dy, fill=self._SHADOW, **base)
            self._shadows.append(sid)

        self._text_id = self._canvas.create_text(cx, cy, fill="white", **base)

        self._text = ""
        self._queue: queue.Queue[str] = queue.Queue()

        idle = "Listening..."
        for sid in self._shadows:
            self._canvas.itemconfig(sid, text=idle)
        self._canvas.itemconfig(self._text_id, text=idle, fill="#555555")

        try:
            self.root.attributes("-transparentcolor", self._TRANSPARENT)
        except Exception:
            self.root.attributes("-alpha", 0.88)

        self.root.deiconify()
        self.root.update_idletasks()
        self._make_click_through()
        self.root.after(50, self._poll)

    def _make_click_through(self):
        try:
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x80000
            WS_EX_TRANSPARENT = 0x20
            WS_EX_TOOLWINDOW = 0x80
            WS_EX_NOACTIVATE = 0x08000000
            hwnd = windll.user32.GetAncestor(self.root.winfo_id(), 2)
            current = windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            windll.user32.SetWindowLongW(
                hwnd,
                GWL_EXSTYLE,
                current | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
            )
        except Exception:
            pass

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
                for sid in self._shadows:
                    self._canvas.itemconfig(sid, text=text)
                self._canvas.itemconfig(self._text_id, text=text, fill="white")
        except queue.Empty:
            pass
        self.root.after(50, self._poll)

    # ── lifecycle ──────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()

    def stop(self):
        self.root.quit()
