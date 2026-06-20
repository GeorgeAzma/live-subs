"""Transparent always-on-top subtitle overlay for Windows.

Usage
-----
    overlay = SubtitleOverlay()
    overlay.on_partial("Hello...")
    overlay.run()  # blocks until window closes
"""

import ctypes
import os
import queue
import tkinter as tk
from ctypes import windll, wintypes
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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


# ── Win32 per-pixel-alpha helpers ───────────────────────────────────────

ULW_ALPHA = 0x02
AC_SRC_OVER = 0x00
AC_SRC_ALPHA = 0x01


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class SIZE(ctypes.Structure):
    _fields_ = [("cx", wintypes.LONG), ("cy", wintypes.LONG)]


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [
        ("BlendOp", wintypes.BYTE),
        ("BlendFlags", wintypes.BYTE),
        ("SourceConstantAlpha", wintypes.BYTE),
        ("AlphaFormat", wintypes.BYTE),
    ]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", BITMAPINFOHEADER),
        ("bmiColors", wintypes.DWORD * 3),
    ]


def _make_bf() -> BLENDFUNCTION:
    bf = BLENDFUNCTION()
    bf.BlendOp = AC_SRC_OVER
    bf.BlendFlags = 0
    bf.SourceConstantAlpha = 255
    bf.AlphaFormat = AC_SRC_ALPHA
    return bf


def _update_layered_window(
    hwnd: int,
    width: int,
    height: int,
    bgra_bytes: bytes,
):
    hdc_screen = windll.user32.GetDC(0)
    hdc_mem = windll.gdi32.CreateCompatibleDC(hdc_screen)

    bmi = BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = width
    bmi.bmiHeader.biHeight = -height
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bmi.bmiHeader.biCompression = 3
    bmi.bmiHeader.biSizeImage = width * height * 4
    bmi.bmiColors[0] = 0x00FF0000
    bmi.bmiColors[1] = 0x0000FF00
    bmi.bmiColors[2] = 0x000000FF

    ppvBits = ctypes.POINTER(ctypes.c_ubyte)()
    hbitmap = windll.gdi32.CreateDIBSection(
        hdc_screen, ctypes.byref(bmi), 0, ctypes.byref(ppvBits), None, 0
    )
    if not hbitmap:
        windll.gdi32.DeleteDC(hdc_mem)
        windll.user32.ReleaseDC(0, hdc_screen)
        return

    ctypes.memmove(ppvBits, bgra_bytes, len(bgra_bytes))

    old = windll.gdi32.SelectObject(hdc_mem, hbitmap)

    rect = wintypes.RECT()
    windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))

    pt_dst = POINT(rect.left, rect.top)
    sz = SIZE(width, height)
    pt_src = POINT(0, 0)
    bf = _make_bf()

    windll.user32.UpdateLayeredWindow(
        hwnd,
        hdc_screen,
        ctypes.byref(pt_dst),
        ctypes.byref(sz),
        hdc_mem,
        ctypes.byref(pt_src),
        0,
        ctypes.byref(bf),
        ULW_ALPHA,
    )

    windll.gdi32.SelectObject(hdc_mem, old)
    windll.gdi32.DeleteObject(hbitmap)
    windll.gdi32.DeleteDC(hdc_mem)
    windll.user32.ReleaseDC(0, hdc_screen)


# ── Overlay window ───────────────────────────────────────────────────────


class SubtitleOverlay(TextHandler):
    """Always-on-top transparent subtitle window.

    Displays translation text at the bottom of the screen with a
    dark drop shadow for readability.  Drag by left-clicking any
    blank area.  Press Escape to close.
    """

    _SHADOW = "#111111"

    def __init__(self, font_size: int = 30):
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

        self._font_size = font_size
        self._scale = scale
        self._canvas_width = w
        self._canvas_height = h
        self._soff = max(3, int(3 * scale))

        self._canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self._canvas.pack(expand=True, fill="both")

        self._hit_pad = max(6, int(6 * scale))
        self._input_win = tk.Toplevel(self.root)
        self._input_win.overrideredirect(True)
        self._input_win.attributes("-topmost", True)
        self._input_win.configure(bg="black")
        try:
            self._input_win.attributes("-alpha", 0.01)
        except Exception:
            pass
        self.root.attributes("-topmost", True)

        self._show_bg = False
        self._text = ""
        self._queue: queue.Queue[str] = queue.Queue()
        self._hwnd: Optional[int] = None

        self.root.deiconify()
        self.root.update_idletasks()
        self._make_styling()
        self._redraw()
        self._enable_drag()
        self._enable_close()
        self._enable_scroll()
        self._enable_toggle()
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

    # ── text rendering (per-pixel alpha via Win32) ───────────────────

    def _render_text_image(self, text: str, is_idle: bool = False):
        """Render text onto an RGBA PIL image with per-pixel alpha."""
        w = self._canvas_width
        h = self._canvas_height
        fs = int(self._font_size * self._scale * 96.0 / 72.0)

        if self._show_bg:
            img = Image.new("RGBA", (w, h), (0, 0, 0, 128))
        else:
            img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("segoeuib.ttf", fs)
        except (IOError, OSError):
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", fs)
            except (IOError, OSError):
                font = ImageFont.load_default()

        wrap = w - int(100 * self._scale)

        lines = self._wrap_text(text, font, wrap)
        if not lines:
            lines = [text]

        ascent, descent = font.getmetrics()
        line_h = ascent + descent
        total_h = len(lines) * line_h
        y_start = max(0, (h - total_h) // 2)

        soff = self._soff

        for i, line in enumerate(lines):
            line_w = font.getlength(line)
            x = int((w - line_w) // 2)
            y = int(y_start + i * line_h)

            if is_idle:
                fg = (180, 180, 180, 255)
            else:
                fg = (255, 255, 255, 255)
            shadow = (0, 0, 0, 255)

            draw.text((x + soff, y + soff), line, font=font, fill=shadow)
            draw.text((x, y), line, font=font, fill=fg)

        return img

    def _wrap_text(self, text: str, font, wrap_width: int):
        lines = []
        for word in text.split():
            if not lines:
                lines.append(word)
                continue
            test = f"{lines[-1]} {word}"
            if font.getlength(test) <= wrap_width:
                lines[-1] = test
            else:
                lines.append(word)
        return lines

    def _redraw(self):
        if self._hwnd is None:
            self._hwnd = windll.user32.GetAncestor(self.root.winfo_id(), 2)

        is_idle = not self._text
        display = self._text if self._text else "Listening..."
        img = self._render_text_image(display, is_idle=is_idle)

        arr = np.array(img, dtype=np.uint8)
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        arr[:, :, :3] = (arr[:, :, :3] * alpha).astype(np.uint8)
        bgra = arr[:, :, [2, 1, 0, 3]]

        _update_layered_window(
            self._hwnd,
            self._canvas_width,
            self._canvas_height,
            bgra.tobytes(),
        )

    # ── background toggle via Space ───────────────────────────────

    def _enable_toggle(self):
        self.root.bind("<Key-space>", self._on_toggle_bg)
        self._input_win.bind("<Key-space>", self._on_toggle_bg)

    def _on_toggle_bg(self, event):
        self._show_bg = not self._show_bg
        self._redraw()

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
        self._resize_to_fit_text()
        self._redraw()
        self._update_hit_box()

    # ── window sizing ──────────────────────────────────────────────

    def _measure_text(self, text: str):
        """Measure rendered text extents without drawing."""
        fs = int(self._font_size * self._scale * 96.0 / 72.0)
        try:
            font = ImageFont.truetype("segoeuib.ttf", fs)
        except (IOError, OSError):
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", fs)
            except (IOError, OSError):
                font = ImageFont.load_default()
        wrap = self._canvas_width - int(100 * self._scale)
        lines = self._wrap_text(text, font, wrap)
        if not lines:
            lines = [text]
        ascent, descent = font.getmetrics()
        line_h = ascent + descent
        total_h = len(lines) * line_h
        max_w = max(font.getlength(l) for l in lines)
        return lines, max_w, total_h, line_h

    def _resize_to_fit_text(self):
        display = self._text if self._text else "Listening..."
        _lines, _max_w, total_h, _line_h = self._measure_text(display)
        pad = int(20 * self._scale)
        target_h = total_h + 2 * pad
        min_h = max(150, int(150 * self._scale))
        target_h = max(min_h, target_h)

        rx = self.root.winfo_x()
        ry = self.root.winfo_y()
        current_bottom = ry + self._canvas_height
        new_y = current_bottom - target_h

        self.root.geometry(f"{self._canvas_width}x{int(target_h)}+{rx}+{int(new_y)}")
        self._canvas_height = int(target_h)

    def _update_hit_box(self):
        display = self._text if self._text else "Listening..."
        lines, max_w, total_h, line_h = self._measure_text(display)
        if not lines:
            lines = [display]

        w = self._canvas_width
        h = self._canvas_height
        # Same positioning logic as _render_text_image
        y_start = max(0, int((h - total_h) // 2))
        x_start = max(0, int((w - max_w) // 2))

        pad = self._hit_pad
        rx = self.root.winfo_x()
        ry = self.root.winfo_y()

        x1 = x_start - pad
        y1 = y_start - pad
        x2 = x_start + int(max_w) + pad
        y2 = y_start + total_h + pad

        self._input_win.geometry(f"{x2 - x1}x{y2 - y1}+{rx + x1}+{ry + y1}")

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
                self._resize_to_fit_text()
                self._redraw()
                self._update_hit_box()
        except queue.Empty:
            pass
        self.root.after(50, self._poll)

    # ── lifecycle ──────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()

    def stop(self):
        self.root.quit()
