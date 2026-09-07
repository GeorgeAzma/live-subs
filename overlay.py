import ctypes
import os
import queue
import re
import time
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


def _is_cjk(ch: str) -> bool:
    return (
        0x2E80 <= ord(ch) <= 0x9FFF
        or 0xF900 <= ord(ch) <= 0xFAFF
        or 0x3000 <= ord(ch) <= 0x303F
        or 0xFF00 <= ord(ch) <= 0xFFEF
    )


def _get_refresh_hz() -> float:
    """Actual refresh rate of the primary monitor (e.g. 144.0), with a safe
    fallback. Per Microsoft docs, a frequency of 0 or 1 means 'unknown'."""
    try:
        dm = wintypes.DEVMODEW()
        dm.dmSize = ctypes.sizeof(dm)
        ENUM_CURRENT_SETTINGS = -1
        if windll.user32.EnumDisplaySettingsW(None, ENUM_CURRENT_SETTINGS, ctypes.byref(dm)):
            hz = float(dm.dmDisplayFrequency)
            if 24.0 <= hz <= 500.0:
                return hz
    except Exception:
        pass
    try:
        dc = windll.user32.GetDC(0)
        hz = float(windll.gdi32.GetDeviceCaps(dc, 4))  # VREFRESH
        windll.user32.ReleaseDC(0, dc)
        if 24.0 <= hz <= 500.0:
            return hz
    except Exception:
        pass
    return 60.0


class SubtitleOverlay(TextHandler):
    """Bottom-anchored, center-aligned live caption surface.

    Readability-first design decisions:
    - *Fixed anchor*: the newest line always sits at the same screen position
      (bottom of the block). The eye parks there; text flows past it.
    - *Centered lines*: every line starts near the horizontal center, so
      saccades to find the line start are short and symmetric.
    - *~42-char measure*: line width is derived from font metrics, never from
      the window width. Short lines eliminate return-sweep regressions.
    - *Immutable history*: a line, once shown, never rewraps or moves except
      for one smooth push-up slide when a new line appears.
    - *High-contrast hierarchy*: stable words are pure white; in-flight words
      and old lines are only gently dimmed so contrast never collapses.
    - *CJK-aware wrapping*: lines can break between CJK characters, not just
      at spaces.
    """

    MAX_LINES = 3                 # history + active, bounded block height
    MEASURE_CHARS = 42            # target characters per line (caption standard)
    _SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+|(?<=[。！？])')

    _C_HIST = 232                 # confirmed previous-utterance text
    _C_IDLE = 180
    _BG_ALPHA = 192               # ~75% black pill: readable over any video
    def __init__(self, font_size: int = 30):
        _enable_dpi_awareness()
        self._scale = _get_dpi_scale()
        self._font_size = font_size
        self._font_cache: dict = {}
        self._mask_cache: dict = {}

        self.root = tk.Tk()
        self.root.withdraw()
        self.root.title("Transcriber Subtitles")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self._sw = self.root.winfo_screenwidth()
        self._sh = self.root.winfo_screenheight()

        self._show_bg = True
        self._soft_shadow = False
        self._translator = None
        self._translating = True

        # Utterance states: hist (old, dim) -> recent (last final, bright)
        # -> active (in-flight partial, conf-dimmed words).
        self._hist_text = ""
        self._recent_text = ""
        self._active_text = ""
        self._status = "Starting..."
        self._queue: queue.Queue = queue.Queue()
        self._hwnd: Optional[int] = None
        self._drag_start = None
        self._anim_y = 0.0          # current render offset (px)
        self._anim_target = 0.0     # where the offset wants to be (0)
        self._anim_last_t = 0.0     # monotonic ts of last anim frame
        self._anim_active = False
        self._anim_after_id = None
        self._clock = time.perf_counter
        self._refresh_hz = _get_refresh_hz()
        # pill edge spring state: current and target [left, top, right, bottom]
        self._pill_cur: Optional[list[float]] = None
        self._last_key = None
        self._last_layout: Optional[dict] = None
        self._canvas_width: int = 0
        self._canvas_height: int = 0

        self._apply_metrics(initial=True)

        self._canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self._canvas.pack(expand=True, fill="both")

        self._hit_pad = max(6, int(6 * self._scale))
        self._input_win = tk.Toplevel(self.root)
        self._input_win.overrideredirect(True)
        self._input_win.attributes("-topmost", True)
        self._input_win.configure(bg="black")
        try:
            self._input_win.attributes("-alpha", 0.01)
        except Exception:
            pass
        self.root.attributes("-topmost", True)

        self.root.deiconify()
        self.root.update_idletasks()
        self._make_styling()
        self._redraw()
        self._enable_keys()
        self._update_hit_box()
        self.root.after(50, self._poll)

    # ------------------------------------------------------------------ metrics

    def _fs(self) -> int:
        return int(self._font_size * self._scale * 96.0 / 72.0)

    def _apply_metrics(self, initial: bool = False):
        """(Re)compute all geometry from font metrics. Window is re-anchored
        keeping its bottom edge and horizontal center fixed."""
        fs = self._fs()
        font = self._get_font(fs)
        avg = font.getlength("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") / 52.0
        asc = getattr(font, "ascent", None) or fs
        desc = getattr(font, "descent", None) or fs // 4
        try:
            fm = font.getmetrics()  # type: ignore[attr-defined]
            asc, desc = int(fm[0]), int(fm[1])
        except (AttributeError, TypeError):
            pass
        self._line_gap = max(2, int(fs * 0.20))
        self._line_h = asc + desc + self._line_gap
        self._wrap_w = min(
            int(avg * self.MEASURE_CHARS),
            max(280, self._sw - int(80 * self._scale)),
        )
        self._hpad = int(fs * 0.65)
        self._vpad = int(fs * 0.32)
        margin = int(20 * self._scale)

        w = min(self._sw, self._wrap_w + 2 * self._hpad + 2 * margin)
        h = self.MAX_LINES * self._line_h + 2 * self._vpad + 2 * margin
        w, h = int(w), int(h)

        prev_w, prev_h = self._canvas_width, self._canvas_height
        self._canvas_width, self._canvas_height = w, h
        if initial or prev_w == 0:
            x = (self._sw - w) // 2
            y = self._sh - h - int(56 * self._scale)
        else:
            x = self.root.winfo_x() + (prev_w - w) // 2
            y = self.root.winfo_y() + (prev_h - h)
        self.root.geometry(f"{w}x{h}+{int(x)}+{int(y)}")
        self._mask_cache.clear()

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

    def _get_font(self, fs: int, text: str = ""):
        candidates = ["segoeuib.ttf"]
        if any(_is_cjk(c) for c in text):
            candidates = ["C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/msyh.ttc", "segoeuib.ttf"]
        path = candidates[0]
        key = (path, fs)
        font = self._font_cache.get(key)
        if font is None:
            if len(self._font_cache) > 64:
                self._font_cache.clear()
            try:
                font = ImageFont.truetype(path, fs)
            except (IOError, OSError):
                font = ImageFont.load_default()
            self._font_cache[key] = font
        return font

    # ------------------------------------------------------------------- layout

    @staticmethod
    def _tokenize(text: str):
        """Split into atomic tokens: space-separated words, with CJK runs
        broken per character (CJK has no spaces but wraps anywhere)."""
        tokens = []  # (text, space_after)
        for word in text.split(" "):
            if not word:
                continue
            chunk = ""
            prev_cjk: Optional[bool] = None
            for ch in word:
                cjk = _is_cjk(ch)
                if prev_cjk is not None and (cjk or prev_cjk):
                    tokens.append((chunk, False))
                    chunk = ""
                chunk += ch
                prev_cjk = cjk
            tokens.append((chunk, True))
        if tokens:
            tokens[-1] = (tokens[-1][0], False)
        return tokens

    def _wrap(self, text: str, font, max_w: int):
        lines = []
        cur = ""
        for token, space_after in self._tokenize(text):
            cand = cur + token
            if cur and font.getlength(cand) > max_w:
                lines.append(cur.rstrip())
                cur = token
            else:
                cur = cand
            if space_after:
                cur += " "
        if cur.strip():
            lines.append(cur.rstrip())
        return lines or [""]

    def _wrap_sentences(self, text: str, font, max_w: int):
        """Wrap per sentence so line breaks land on natural pauses."""
        lines: list[str] = []
        for sentence in self._SENTENCE_SPLIT.split(text):
            if sentence and sentence.strip():
                lines.extend(self._wrap(sentence.strip(), font, max_w))
        return lines or [""]

    def _layout(self, dim_text: str, bright_text: str) -> dict:
        fs = self._fs()
        all_text = (dim_text + " " + bright_text).strip()
        font = self._get_font(fs, all_text)
        dim_lines = self._wrap_sentences(dim_text, font, self._wrap_w) if dim_text.strip() else []
        bright_lines = self._wrap_sentences(bright_text, font, self._wrap_w) if bright_text.strip() else []
        lines = dim_lines + bright_lines
        visible = lines[-self.MAX_LINES:] if lines else [""]
        n_hist_visible = max(0, len(visible) - len(bright_lines))
        hist_flags = [i < n_hist_visible for i in range(len(visible))]
        widths = [font.getlength(l) for l in visible]
        # Text block is bottom-anchored at a fixed position (never moves on
        # its own). The background pill HUGS the text: small when idle, grows
        # smoothly (same exponential spring as the push-up motion) as text
        # appears. Horizontal edges are sprung; vertical edges track the
        # animated text positions exactly, so background and text move as one.
        ys = [float(self._canvas_height - int(20 * self._scale) - self._vpad
                    - (len(visible) - i) * self._line_h + self._line_gap)
              for i in range(len(visible))]
        centers = [(self._canvas_width - wd) / 2.0 for wd in widths]
        # static hugging rect (used for the hit box; the rendered pill is
        # computed in _render with animation applied)
        block_w = (max(widths) if widths else 0) + 2 * self._hpad
        pill = (
            (self._canvas_width - block_w) / 2.0,
            min(ys) - self._vpad if ys else 0.0,
            (self._canvas_width + block_w) / 2.0,
            (max(ys) + self._line_h - self._line_gap + self._vpad) if ys else 0.0,
        )
        return {
            "lines": visible, "widths": widths, "ys": ys, "centers": centers,
            "hist": hist_flags, "font": font, "fs": fs,
            "pill": pill,
        }

    def _layout_single(self, text: str) -> dict:
        return self._layout("", text)

    # ------------------------------------------------------------------- render

    def _render(self, layout: dict, is_idle: bool) -> Image.Image:
        w, h = self._canvas_width, self._canvas_height
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        anim = self._anim_y
        lines = layout["lines"]
        font, fs = layout["font"], layout["fs"]

        # Which lines move during the spring: when a line was ADDED (anim > 0)
        # the new bottom line must NOT move -- it appears directly at the
        # anchor while old lines slide up from one line below their target.
        # When lines were REMOVED (anim < 0) every remaining line eases down.
        def line_offset(i: int) -> float:
            if anim > 0 and i == len(lines) - 1:
                return 0.0
            return anim

        if self._show_bg and self._pill_cur is not None:
            px0, py0, px1, py1 = layout["pill"]
            px0, px1 = self._pill_cur[0], self._pill_cur[2]
            top_off = anim
            bot_off = 0.0 if anim > 0 else anim
            img = self._stamp_bg(img, (px0, py0 + top_off, px1, py1 + bot_off))
        draw = ImageDraw.Draw(img)

        soff = max(1, int(fs * 0.06))
        shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(shadow)
        for i, line in enumerate(lines):
            sdraw.text((layout["centers"][i] + soff, layout["ys"][i] + line_offset(i) + soff),
                       line, font=font, fill=(0, 0, 0, 220))
        if self._soft_shadow:
            shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(1, int(fs * 0.06))))
        img = Image.alpha_composite(img, shadow)
        draw = ImageDraw.Draw(img)

        for i, line in enumerate(lines):
            y = layout["ys"][i] + line_offset(i)
            x = layout["centers"][i]
            if layout["hist"][i]:
                # history lines carry no confidence mapping (conf list belongs
                # to the bright utterance) -> do not advance word_idx
                draw.text((x, y), line, font=font, fill=(self._C_HIST,) * 3 + (255,))
                continue
            if not line.strip():
                continue
            draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
        return img

    def _stamp_bg(self, img: Image.Image, rect) -> Image.Image:
        """Translucent rounded pill with antialiased edges (supersampled mask,
        cached by rect). Clamped inside the canvas so corners never clip."""
        w, h = img.size
        key = tuple(int(v) for v in rect) + (w, h)
        mask = self._mask_cache.get(key)
        if mask is None:
            SS = 4
            x0 = max(1, int(rect[0])); y0 = max(1, int(rect[1]))
            x1 = min(w - 1, int(rect[2])); y1 = min(h - 1, int(rect[3]))
            big = Image.new("L", (w * SS, h * SS), 0)
            d = ImageDraw.Draw(big)
            radius = max(SS, int(min((y1 - y0) * 0.22, (x1 - x0) * 0.5)) * SS)
            d.rounded_rectangle((x0 * SS, y0 * SS, x1 * SS, y1 * SS), radius=radius, fill=255)
            try:
                resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
            except AttributeError:  # Pillow < 9.1
                resample = 1  # Image.LANCZOS == 1
            mask = big.resize((w, h), resample)
            if len(self._mask_cache) > 48:
                self._mask_cache.clear()
            self._mask_cache[key] = mask
        layer = Image.new("RGBA", (w, h), (10, 10, 12, 0))
        layer.putalpha(mask.point(lambda a: a * self._BG_ALPHA // 255))
        return Image.alpha_composite(img, layer)

    # -------------------------------------------------------------------- state

    def set_status(self, text: str):
        """Show a loading/progress message. Safe from any thread."""
        self._queue.put(("status", text or ""))

    def on_partial(self, text: str):
        self._queue.put(("partial", text))

    def on_final(self, text: str):
        self._queue.put(("final", text))

    def _apply_partial(self, text: str):
        if self._recent_text and not self._active_text:
            # a new utterance started: last final becomes history
            self._hist_text = self._recent_text
            self._recent_text = ""
        self._active_text = text
        self._redraw()
        self._update_hit_box()

    def _apply_final(self, text: str):
        if self._active_text:
            # a partial was on screen: it becomes the recent utterance
            self._recent_text = text
            self._active_text = ""
        else:
            # final arrived without any interim (short segment):
            # previous recent becomes history
            self._hist_text = self._recent_text
            self._recent_text = text
        self._redraw()
        self._update_hit_box()

    def _redraw(self):
        if self._hwnd is None:
            self._hwnd = windll.user32.GetAncestor(self.root.winfo_id(), 2)

        is_idle = False
        if self._status:
            layout = self._layout_single(self._status)
            is_idle = True
        elif self._active_text or self._recent_text:
            layout = self._layout(
                self._hist_text,
                self._active_text or self._recent_text,
            )
        else:
            layout = self._layout_single("Listening..." if self._translating else "Transcribing...")
            is_idle = True
        self._last_layout = layout

        key = tuple(layout["lines"])
        if key != self._last_key:
            grew = self._last_key is not None and (
                len(key) > len(self._last_key)
                or (len(key) == self.MAX_LINES and key[0] != self._last_key[-len(key):][0]
                    and self._last_key[-1] in key[:-1])
            )
            shrank = self._last_key is not None and len(key) < len(self._last_key)
            self._last_key = key
            if grew:
                # New line: content sits one line lower than it should; the
                # offset decays smoothly to 0.
                self._anim_y += float(self._line_h)
            elif shrank:
                # History expired: remaining lines would jump down; render
                # them one line higher first and ease into place.
                self._anim_y -= float(self._line_h)

        # Start the spring whenever anything is unsettled: line-count changes
        # (anim_y != 0) OR the pill's target edges moved (text width changed,
        # e.g. "Loading model..." -> "Listening..." on startup). Without this
        # the pill keeps a stale width until the next line-count change.
        pill_unsettled = False
        if self._show_bg:
            tx0, _, tx1, _ = layout["pill"]
            if self._pill_cur is None:
                # first frame after init/toggle: start at the target
                self._pill_cur = [tx0, layout["pill"][1], tx1, layout["pill"][3]]
            else:
                pill_unsettled = (abs(self._pill_cur[0] - tx0) > 0.1
                                  or abs(self._pill_cur[2] - tx1) > 0.1)
        if self._anim_y != 0.0 or pill_unsettled:
            self._start_anim()
        self._draw(layout, is_idle)

    def _draw(self, layout: dict, is_idle: bool):
        img = self._render(layout, is_idle)
        arr = np.array(img, dtype=np.uint8)
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        arr[:, :, :3] = (arr[:, :, :3] * alpha).astype(np.uint8)
        bgra = arr[:, :, [2, 1, 0, 3]]
        hwnd = self._hwnd
        if hwnd is None:
            return
        _update_layered_window(hwnd, self._canvas_width, self._canvas_height, bgra.tobytes())

    # ----------------------------------------------------------------- animation

    _TAU_MOVE = 0.030    # push-up motion time constant
    _TAU_PILL = 0.015    # pill resize: quicker, imperceptible lag behind text

    def _spring_step(self, cur: float, target: float, dt: float, tau: float = _TAU_MOVE) -> float:
        """One critically-damped exponential step (framerate-independent,
        monotonic deceleration, no overshoot, no visible easing pattern)."""
        return cur + (target - cur) * (1.0 - pow(2.718281828, -dt / tau))

    def _start_anim(self):
        """Run the springs at the monitor's refresh rate until settled."""
        if self._anim_active:
            return
        self._anim_active = True
        self._anim_last_t = self._clock()
        self._anim_frame()

    def _anim_frame(self):
        """One spring integration step + repaint, driven at the monitor's
        refresh rate. Animates the push-up offset and the pill edges."""
        now = self._clock()
        dt = min(now - self._anim_last_t, 0.1)  # clamp tab-switch hiccups
        self._anim_last_t = now

        # tau = 30ms: pure exponential decay, settles in ~110ms.
        self._anim_y = self._spring_step(self._anim_y, self._anim_target, dt)

        # pill horizontal edges ease toward the text-hugging target
        if self._pill_cur is not None and self._last_layout is not None:
            tx0, _, tx1, _ = self._last_layout["pill"]
            self._pill_cur[0] = self._spring_step(self._pill_cur[0], tx0, dt, self._TAU_PILL)
            self._pill_cur[2] = self._spring_step(self._pill_cur[2], tx1, dt, self._TAU_PILL)

        if self._last_layout is not None:
            self._draw(self._last_layout, False)

        settled = abs(self._anim_y - self._anim_target) < 0.1
        if self._pill_cur is not None and self._last_layout is not None:
            tx0, _, tx1, _ = self._last_layout["pill"]
            settled = settled and (abs(self._pill_cur[0] - tx0) < 0.1
                                   and abs(self._pill_cur[2] - tx1) < 0.1)
        if settled:
            # snap and stop the loop (zero idle cost); one final paint already
            # happened above with the last sub-threshold step, so no redraw
            self._anim_y = self._anim_target
            if self._pill_cur is not None and self._last_layout is not None:
                self._pill_cur[0] = self._last_layout["pill"][0]
                self._pill_cur[2] = self._last_layout["pill"][2]
            self._anim_active = False
            return

        interval_ms = max(1, round(1000.0 / self._refresh_hz))
        self._anim_after_id = self.root.after(interval_ms, self._anim_frame)

    # -------------------------------------------------------------------- input

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
        self._pill_cur = None
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
        self._anim_y = self._anim_target = 0.0
        self._anim_active = False
        self._pill_cur = None
        if self._anim_after_id is not None:
            try:
                self.root.after_cancel(self._anim_after_id)
            except Exception:
                pass
            self._anim_after_id = None
        self._apply_metrics()
        self._last_key = None
        self._redraw()
        self._update_hit_box()

    def _update_hit_box(self):
        layout = self._last_layout
        if layout is None or self._canvas_width is None:
            return
        rx, ry = self.root.winfo_x(), self.root.winfo_y()
        x0, y0, x1, y1 = layout["pill"]
        pad = self._hit_pad
        w = int(x1 - x0) + 2 * pad
        h = int(y1 - y0) + 2 * pad
        self._input_win.geometry(f"{w}x{h}+{int(rx + x0) - pad}+{int(ry + y0) - pad}")

    # --------------------------------------------------------------------- poll

    _POLL_MS = 10   # delivery latency: results hit the screen within this

    def _poll(self):
        try:
            while True:
                item = self._queue.get_nowait()
                kind = item[0]
                text = item[1] if len(item) > 1 else ""
                if kind == "status":
                    if self._status != text:
                        self._status = text
                        self._redraw()
                        self._update_hit_box()
                elif kind == "final":
                    self._apply_final(text)
                elif text != self._active_text:
                    self._apply_partial(text)
        except queue.Empty:
            pass

        self.root.after(self._POLL_MS, self._poll)

    def run(self):
        self.root.mainloop()

    def stop(self):
        self.root.quit()
