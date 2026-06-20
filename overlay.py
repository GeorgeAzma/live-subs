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
            self._cx + self._soff,
            self._cy + self._soff,
            fill=self._SHADOW,
            font=font,
            anchor="center",
            justify="center",
            width=wrap,
        )
        self._text_id = self._canvas.create_text(
            self._cx,
            self._cy,
            fill="white",
            font=font,
            anchor="center",
            justify="center",
            width=wrap,
        )

        # Second nearly-invisible window layered BEHIND the main
        # overlay to capture mouse events across the text's bounding
        # box (including the transparent gaps). The main overlay's
        # transparent pixels are click-through via -transparentcolor,
        # so clicks on gaps fall through to this window. -alpha keeps
        # it click-capturing while making it visually imperceptible.
        # Keeping it behind the text means it never veils the text's
        # antialiased edges.
        self._hit_pad = max(6, int(6 * scale))
        self._input_win = tk.Toplevel(self.root)
        self._input_win.overrideredirect(True)
        self._input_win.attributes("-topmost", True)
        self._input_win.configure(bg=self._TRANSPARENT)
        try:
            self._input_win.attributes("-alpha", 0.01)
        except Exception:
            pass
        # Re-assert root on top so the text renders above the veil.
        self.root.attributes("-topmost", True)

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
        self._update_hit_box()
        self.root.after(50, self._poll)

    def _make_styling(self):
        try:
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x80000
            WS_EX_TOOLWINDOW = 0x80
            for win in (self.root, self._input_win):
                hwnd = windll.user32.GetAncestor(win.winfo_id(), 2)
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
        for win in (self._input_win, self._canvas):
            win.bind("<Button-1>", self._on_drag_start)
            win.bind("<B1-Motion>", self._on_drag_move)

    def _on_drag_start(self, event):
        self._drag_start = (event.x_root, event.y_root)
        self._input_win.focus_force()
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
        self._update_hit_box()

    # ── font size via scroll ───────────────────────────────────────

    def _enable_scroll(self):
        for win in (self._input_win, self._canvas):
            win.bind("<MouseWheel>", self._on_scroll)
            win.bind("<Button-4>", self._on_scroll_up)
            win.bind("<Button-5>", self._on_scroll_down)

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
        self._resize_to_fit_text()
        self._update_hit_box()

    def _resize_to_fit_text(self):
        """Resize the overlay window to fit the rendered text height."""
        self.root.update_idletasks()
        bbox = self._canvas.bbox(self._text_id)
        if not bbox:
            return
        _, y1, _, y2 = bbox
        pad = int(20 * self._scale)
        target_h = (y2 - y1) + 2 * pad
        min_h = max(150, int(150 * self._scale))
        target_h = max(min_h, target_h)

        rx = self.root.winfo_x()
        ry = self.root.winfo_y()
        current_bottom = ry + self._canvas_height
        new_y = current_bottom - target_h

        self.root.geometry(f"{self._canvas_width}x{int(target_h)}+{rx}+{int(new_y)}")
        self._canvas_height = int(target_h)
        self._cy = self._canvas_height // 2

        self._canvas.coords(self._shadow_id, self._cx + self._soff, self._cy + self._soff)
        self._canvas.coords(self._text_id, self._cx, self._cy)

    def _update_hit_box(self):
        """Position the invisible input window over the text bounding box."""
        self.root.update_idletasks()
        bbox = self._canvas.bbox(self._text_id)
        if not bbox:
            return
        x1, y1, x2, y2 = bbox
        pad = self._hit_pad
        rx = self.root.winfo_x()
        ry = self.root.winfo_y()
        ix = rx + x1 - pad
        iy = ry + y1 - pad
        iw = (x2 - x1) + 2 * pad
        ih = (y2 - y1) + 2 * pad
        self._input_win.geometry(f"{iw}x{ih}+{ix}+{iy}")

    # ── close via Escape ───────────────────────────────────────────

    def _enable_close(self):
        self._input_win.bind("<Escape>", lambda e: os._exit(0))
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
                self._resize_to_fit_text()
                self._update_hit_box()
        except queue.Empty:
            pass
        self.root.after(50, self._poll)

    # ── lifecycle ──────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()

    def stop(self):
        self.root.quit()
