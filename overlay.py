import ctypes
import os
import queue
import re
import tkinter as tk
from ctypes import windll, wintypes
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from main import TextHandler


def _enable_dpi_awareness():
    for func in (
        ("shcore", "SetProcessDpiAwareness", 2),
        ("user32", "SetProcessDPIAware", None),
    ):
        try:
            getattr(windll, func[0]).__getattr__(func[1])(func[2])
            return
        except Exception:
            continue


def _get_dpi_scale() -> float:
    try:
        return windll.user32.GetDpiForSystem() / 96.0
    except Exception:
        return 1.0


ULW_ALPHA = 0x02
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
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


def _update_layered_window(hwnd: int, width: int, height: int, bgra_bytes: bytes):
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
    bf = BLENDFUNCTION()
    bf.BlendOp = 0
    bf.BlendFlags = 0
    bf.SourceConstantAlpha = 255
    bf.AlphaFormat = AC_SRC_ALPHA

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


class SubtitleOverlay(TextHandler):
    MAX_SENTENCES = 3
    _SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

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
        # Cap width to prevent overly wide subtitles
        w = min(sw, int(1000 * scale))
        # Fixed height for ~3 lines of text - prevents window jumping
        h = max(180, int(180 * scale))
        self.root.geometry(f"{w}x{h}+{int((sw - w) // 2)}+{sh - h - int(40 * scale)}")

        self._font_size = font_size
        self._scale = scale
        self._canvas_width = w
        self._canvas_height = h

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
        self._soft_shadow = False
        self._translator = None
        self._translating = True
        self._text = ""
        self._partial_suffix = ""
        self._pending_partial = ""
        self._stable_count = 0
        self._STABILIZE_CYCLES = 10
        self._is_partial = False
        self._queue: queue.Queue = queue.Queue()
        self._hwnd: Optional[int] = None

        self.root.deiconify()
        self.root.update_idletasks()
        self._make_styling()
        self._redraw()
        self._enable_keys()
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
                    hwnd, GWL_EXSTYLE, current | WS_EX_LAYERED | WS_EX_TOOLWINDOW
                )
        except Exception:
            pass

    def _render_text_image(self, text: str, is_idle: bool = False, is_partial: bool = False):
        w = self._canvas_width
        h = self._canvas_height
        fs = int(self._font_size * self._scale * 96.0 / 72.0)

        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        font = self._load_font(fs, text)
        wrap = w - int(100 * self._scale)
        lines = self._wrap_text(text, font, wrap)
        if not lines:
            lines = [text]

        ascent, descent = font.getmetrics()
        line_h = ascent + descent
        total_h = len(lines) * line_h
        y_start = max(0, int(fs * 0.15))

        line_widths = [font.getlength(l) for l in lines]
        max_w = max(line_widths)
        left_margin = int(50 * self._scale)

        if self._show_bg:
            hpad = int(fs * 0.5)
            vpad = int(fs * 0.2)
            draw.rounded_rectangle(
                (
                    left_margin - hpad,
                    y_start - vpad,
                    left_margin + int(max_w) + hpad,
                    y_start + total_h + vpad,
                ),
                radius=int(fs * 0.45),
                fill=(0, 0, 0, 128),
            )

        if is_partial and self._text:
            n_stable_words = len(self._text.split())
        else:
            n_stable_words = float('inf')

        if not is_idle:
            if self._soft_shadow:
                soff = max(1, int(fs * 0.04))
                shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
                sdraw = ImageDraw.Draw(shadow)
                for i, line in enumerate(lines):
                    x = left_margin
                    y = int(y_start + i * line_h)
                    sdraw.text((x + soff, y + soff), line, font=font, fill=(0, 0, 0, 255))
                shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(2, int(fs * 0.08))))
                img = Image.alpha_composite(img, shadow)
                draw = ImageDraw.Draw(img)
            else:
                soff = max(3, int(fs * 0.08))
                for i, line in enumerate(lines):
                    x = left_margin
                    y = int(y_start + i * line_h)
                    draw.text((x + soff, y + soff), line, font=font, fill=(0, 0, 0, 255))

        word_pos = 0
        for i, line in enumerate(lines):
            y = int(y_start + i * line_h)
            if is_idle:
                draw.text((left_margin, y), line, font=font, fill=(180, 180, 180, 255))
                continue
            words = line.split()
            x = left_margin
            for word in words:
                is_confirm = word_pos < n_stable_words
                fg = (255, 255, 255, 255) if is_confirm else (120, 120, 120, 255)
                draw.text((x, y), word, font=font, fill=fg)
                x += font.getlength(word + " ")
                word_pos += 1

        return img

    def _load_font(self, fs: int, text: str = ""):
        candidates = ["segoeuib.ttf"]
        if any(0x2E80 <= ord(c) <= 0x9FFF or 0xF900 <= ord(c) <= 0xFAFF for c in text):
            candidates = [
                "C:/Windows/Fonts/msyhbd.ttc",
                "C:/Windows/Fonts/msyh.ttc",
                "segoeuib.ttf",
            ]
        for path in candidates:
            try:
                return ImageFont.truetype(path, fs)
            except (IOError, OSError):
                continue
        return ImageFont.load_default()

    def _wrap_text(self, text: str, font, wrap_width: int):
        lines = []
        for sentence in text.split("\n"):
            words = sentence.split()
            for i, word in enumerate(words):
                if not lines or i == 0:
                    lines.append(word)
                else:
                    test = f"{lines[-1]} {word}"
                    if font.getlength(test) <= wrap_width:
                        lines[-1] = test
                    else:
                        lines.append(word)
        if not lines:
            lines = [text]
        return lines

    def _redraw(self):
        if self._hwnd is None:
            self._hwnd = windll.user32.GetAncestor(self.root.winfo_id(), 2)

        display_text = self._get_display_text()
        is_idle = not display_text
        display = (
            "Transcribing..."
            if (is_idle and not self._translating)
            else (display_text if display_text else "Listening...")
        )
        img = self._render_text_image(display, is_idle=is_idle, is_partial=self._is_partial)

        arr = np.array(img, dtype=np.uint8)
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        arr[:, :, :3] = (arr[:, :, :3] * alpha).astype(np.uint8)
        bgra = arr[:, :, [2, 1, 0, 3]]

        _update_layered_window(
            self._hwnd, self._canvas_width, self._canvas_height, bgra.tobytes()
        )

    def _enable_keys(self):
        for win in (self._input_win, self._canvas):
            win.bind("<Button-1>", self._on_drag_start)
            win.bind("<B1-Motion>", self._on_drag_move)
            win.bind("<MouseWheel>", self._on_scroll)
            win.bind("<Button-4>", self._on_scroll)
            win.bind("<Button-5>", self._on_scroll)
        self.root.bind("<Key-space>", self._on_toggle_bg)
        self._input_win.bind("<Key-space>", self._on_toggle_bg)
        self.root.bind("<Key-t>", self._on_toggle_translate)
        self._input_win.bind("<Key-t>", self._on_toggle_translate)
        self.root.bind("<Key-s>", self._on_toggle_shadow)
        self._input_win.bind("<Key-s>", self._on_toggle_shadow)
        self._input_win.bind("<Escape>", lambda e: os._exit(0))
        self.root.bind("<Escape>", lambda e: os._exit(0))

    def _on_toggle_bg(self, event):
        self._show_bg = not self._show_bg
        self._redraw()

    def _on_toggle_shadow(self, event):
        self._soft_shadow = not self._soft_shadow
        self._redraw()

    def _on_toggle_translate(self, event):
        if self._translator:
            self._translator.toggle_translate()
            self._translating = self._translator._translate_enabled[0]
            self._redraw()

    def set_translator(self, translator):
        self._translator = translator

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

    def _on_scroll(self, event):
        delta = (
            event.delta
            if hasattr(event, "delta") and event.delta
            else (1 if event.num == 4 else -1)
        )
        delta = max(-1, min(1, delta))
        self._set_font_size(self._font_size + max(self._font_size * 0.1, 1) * delta)

    def _set_font_size(self, new_size):
        new_size = max(8, min(120, new_size))
        if new_size == self._font_size:
            return
        self._font_size = new_size
        self._resize_to_fit_text()
        self._redraw()
        self._update_hit_box()

    def _measure_text(self, text: str):
        fs = int(self._font_size * self._scale * 96.0 / 72.0)
        font = self._load_font(fs, text)
        wrap = self._canvas_width - int(100 * self._scale)
        lines = self._wrap_text(text, font, wrap)
        if not lines:
            lines = [text]
        ascent, descent = font.getmetrics()
        line_h = ascent + descent
        return lines, max(font.getlength(l) for l in lines), len(lines) * line_h, line_h

    def _resize_to_fit_text(self):
        display = self._get_display_text() or "Listening..."
        _lines, _max_w, total_h, _line_h = self._measure_text(display)
        fs = int(self._font_size * self._scale * 96.0 / 72.0)
        pad = max(int(20 * self._scale), int(fs * 0.25))
        target_h = max(max(150, int(150 * self._scale)), total_h + 2 * pad)
        rx, ry = self.root.winfo_x(), self.root.winfo_y()
        self.root.geometry(
            f"{self._canvas_width}x{int(target_h)}+{rx}+{int(ry + self._canvas_height - target_h)}"
        )
        self._canvas_height = int(target_h)

    def _update_hit_box(self):
        display = self._get_display_text() or "Listening..."
        lines, max_w, total_h, _ = self._measure_text(display)
        if not lines:
            lines = [display]
        pad = self._hit_pad
        w = int(max_w) + 2 * pad
        h = total_h + 2 * pad
        fs = int(self._font_size * self._scale * 96.0 / 72.0)
        y_start = max(0, int(fs * 0.15))
        x_start = max(0, int(50 * self._scale))
        rx, ry = self.root.winfo_x(), self.root.winfo_y()
        self._input_win.geometry(f"{w}x{h}+{rx + x_start - pad}+{ry + y_start - pad}")

    def on_partial(self, text: str):
        self._queue.put(("partial", text))

    def on_final(self, text: str):
        self._queue.put(("final", text))

    def _get_display_text(self) -> str:
        full = self._text + self._partial_suffix
        if not full:
            return ""
        sentences = [s.strip() for s in self._SENTENCE_SPLIT.split(full) if s.strip()]
        if not sentences:
            return full
        return "\n".join(sentences[-self.MAX_SENTENCES:])

    @staticmethod
    def _common_prefix_length(s1: str, s2: str) -> int:
        i = 0
        while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
            i += 1
        return i

    def _apply_partial(self, text: str):
        lcp = self._common_prefix_length(self._text, text)
        if lcp >= len(self._text):
            self._partial_suffix = text[lcp:]
        elif lcp >= len(self._text) * 0.5:
            self._text = text[:lcp]
            self._partial_suffix = text[lcp:]
        else:
            self._text = text
            self._partial_suffix = ""
        self._is_partial = True
        self._redraw()
        self._update_hit_box()

    def _poll(self):
        try:
            while True:
                kind, text = self._queue.get_nowait()
                raw = self._text + self._partial_suffix
                if text == raw:
                    continue
                if kind == "final":
                    self._text = text
                    self._partial_suffix = ""
                    self._pending_partial = ""
                    self._stable_count = 0
                    self._is_partial = False
                    self._redraw()
                    self._update_hit_box()
                else:
                    if text == self._pending_partial:
                        self._stable_count += 1
                    else:
                        self._pending_partial = text
                        self._stable_count = 0
        except queue.Empty:
            pass

        if self._pending_partial and self._stable_count >= self._STABILIZE_CYCLES:
            raw = self._text + self._partial_suffix
            if self._pending_partial != raw:
                self._apply_partial(self._pending_partial)

        self.root.after(50, self._poll)

    def run(self):
        self.root.mainloop()

    def stop(self):
        self.root.quit()
