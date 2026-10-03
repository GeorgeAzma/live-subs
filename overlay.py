import ctypes
import json
import math
import os
import queue
import re
import time
import tkinter as tk
import traceback
from ctypes import windll, wintypes
from typing import Callable, Optional

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


_gdi32 = windll.gdi32
_gdi32.CreateCompatibleDC.restype = wintypes.HDC
_gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
_gdi32.CreateDIBSection.restype = wintypes.HBITMAP
_gdi32.CreateDIBSection.argtypes = [
    wintypes.HDC, ctypes.c_void_p, wintypes.UINT,
    ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, wintypes.DWORD,
]
_gdi32.SelectObject.restype = wintypes.HGDIOBJ
_gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
_gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
_gdi32.DeleteDC.argtypes = [wintypes.HDC]

_user32 = windll.user32
_user32.MonitorFromPoint.restype = wintypes.HANDLE
_user32.MonitorFromPoint.argtypes = [POINT, wintypes.DWORD]


def _on_screen(x: int, y: int, w: int, h: int) -> bool:
    """True if the window's center lies on some connected monitor."""
    MONITOR_DEFAULTTONULL = 0
    try:
        return bool(_user32.MonitorFromPoint(POINT(x + w // 2, y + h // 2), MONITOR_DEFAULTTONULL))
    except Exception:
        return True


class _LayeredSurface:
    """Persistent premultiplied BGRA backbuffer for a layered window.

    The DIB section and its memory DC live until the size changes, and
    `pixels` is a numpy view straight onto the DIB memory, so a frame is
    rendered in place and presented with no allocation or copy."""

    def __init__(self):
        self.width = self.height = 0
        self.pixels: Optional[np.ndarray] = None
        self._hdc = None
        self._hbitmap = None
        self._old = None

    def ensure(self, width: int, height: int):
        if (width, height) == (self.width, self.height) and self.pixels is not None:
            return
        self._release()
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
        bits = ctypes.c_void_p()
        hdc = _gdi32.CreateCompatibleDC(None)
        hbitmap = _gdi32.CreateDIBSection(hdc, ctypes.byref(bmi), 0, ctypes.byref(bits), None, 0)
        if not hbitmap:
            _gdi32.DeleteDC(hdc)
            return
        self._hdc, self._hbitmap = hdc, hbitmap
        self._old = _gdi32.SelectObject(hdc, hbitmap)
        buf = (ctypes.c_ubyte * (width * height * 4)).from_address(bits.value)
        self.pixels = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
        self.width, self.height = width, height

    def present(self, hwnd: int):
        if self.pixels is None:
            return
        rect = wintypes.RECT()
        windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
        pt_dst = POINT(rect.left, rect.top)
        sz = SIZE(self.width, self.height)
        pt_src = POINT(0, 0)
        bf = BLENDFUNCTION()
        bf.BlendOp = 0
        bf.BlendFlags = 0
        bf.SourceConstantAlpha = 255
        bf.AlphaFormat = AC_SRC_ALPHA
        windll.user32.UpdateLayeredWindow(
            hwnd,
            None,
            ctypes.byref(pt_dst),
            ctypes.byref(sz),
            wintypes.HDC(self._hdc),
            ctypes.byref(pt_src),
            0,
            ctypes.byref(bf),
            ULW_ALPHA,
        )

    def _release(self):
        self.pixels = None
        if self._hdc:
            _gdi32.SelectObject(self._hdc, self._old)
            _gdi32.DeleteObject(self._hbitmap)
            _gdi32.DeleteDC(self._hdc)
        self._hdc = self._hbitmap = self._old = None
        self.width = self.height = 0


def _blit_over(dst: np.ndarray, src: np.ndarray, inv_alpha: np.ndarray,
               x: int, y: int, opacity: float = 1.0):
    """Premultiplied 'over' of src (with precomputed 1 - alpha) onto dst at
    (x, y), scaled by opacity, clipped to dst."""
    h, w = dst.shape[:2]
    sh, sw = src.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + sw, w), min(y + sh, h)
    if x0 >= x1 or y0 >= y1:
        return
    d = dst[y0:y1, x0:x1]
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    inv = inv_alpha[y0 - y:y1 - y, x0 - x:x1 - x]
    if opacity >= 0.999:
        d *= inv
        d += s
    else:
        d *= inv * opacity + (1.0 - opacity)
        d += s * opacity


def _sprite_arrays(img: Image.Image):
    """PIL RGBA -> (premultiplied float BGRA, 1 - alpha at 4-channel shape).
    The inverse alpha is stored full-shape: a broadcast multiply over the
    length-4 axis is ~8x slower than an elementwise one."""
    rgba = np.asarray(img, dtype=np.float32)
    arr = rgba[:, :, [2, 1, 0, 3]]  # BGRA, the DIB's byte order
    arr[:, :, :3] *= arr[:, :, 3:4] * (1.0 / 255.0)
    inv_alpha = np.repeat(1.0 - arr[:, :, 3:4] * (1.0 / 255.0), 4, axis=2)
    return arr, inv_alpha


def _is_cjk(ch: str) -> bool:
    return (
        0x2E80 <= ord(ch) <= 0x9FFF
        or 0xF900 <= ord(ch) <= 0xFAFF
        or 0x3000 <= ord(ch) <= 0x303F
        or 0xFF00 <= ord(ch) <= 0xFFEF
    )


_FONT_DIR = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts")
_FONTS_LATIN = ("segoeuib.ttf",)
_FONTS_ZH = ("msyhbd.ttc", "msyh.ttc", "segoeuib.ttf")
_FONTS_JA = ("YuGothB.ttc", "msyhbd.ttc", "segoeuib.ttf")
_FONTS_KO = ("malgunbd.ttf", "malgun.ttf", "segoeuib.ttf")


def _font_candidates(text: str) -> tuple:
    """Pick a font family by script. Segoe UI has no CJK glyphs, and
    Japanese/Korean need their own fonts: a Chinese font draws kanji with
    Chinese glyph shapes and has no Hangul at all."""
    han = False
    for ch in text:
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F:
            return _FONTS_KO
        if 0x3040 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF or 0xFF66 <= o <= 0xFF9F:
            return _FONTS_JA  # kana is unambiguous; kanji alone is not
        han = han or _is_cjk(ch)
    return _FONTS_ZH if han else _FONTS_LATIN


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


# ------------------------------------------------------------------ settings

_SETTINGS_PATH = os.path.join(
    os.environ.get("APPDATA") or os.path.expanduser("~"), "Transcriber", "settings.json")
_DEFAULT_SETTINGS = {
    "font_size": 30,
    "x": None,             # window position; None = default bottom-center
    "y": None,
    "show_bg": True,
    "soft_shadow": False,
    "hide_idle": True,
    "translate": True,
}


def _load_settings() -> dict:
    settings = dict(_DEFAULT_SETTINGS)
    try:
        with open(_SETTINGS_PATH, encoding="utf-8") as f:
            saved = json.load(f)
        for key, default in _DEFAULT_SETTINGS.items():
            v = saved.get(key)
            if isinstance(default, bool):
                ok = isinstance(v, bool)
            else:
                ok = isinstance(v, (int, float)) and not isinstance(v, bool)
            if ok:
                settings[key] = v
    except (OSError, ValueError, AttributeError):
        pass
    return settings


def _save_settings(settings: dict):
    try:
        os.makedirs(os.path.dirname(_SETTINGS_PATH), exist_ok=True)
        tmp = _SETTINGS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
        os.replace(tmp, _SETTINGS_PATH)
    except OSError:
        pass


# -------------------------------------------------------------------- overlay

class _Line:
    """One caption line on screen, animated toward its targets."""
    __slots__ = ("text", "color", "split", "font", "x", "y", "a", "tx", "ty", "ta", "tau_a", "exiting")

    def __init__(self, text: str, color: int, split: int, font, x: float, y: float, tau_a: float):
        self.text, self.color, self.split, self.font = text, color, split, font
        self.x = self.tx = x
        self.y = self.ty = y
        self.a, self.ta = 0.0, 1.0
        self.tau_a = tau_a
        self.exiting = False


class SubtitleOverlay(TextHandler):
    """Bottom-anchored, center-aligned live caption surface.

    Readability-first design decisions:
    - *Fixed anchor*: the newest line always sits at the same screen position
      (bottom of the block). The eye parks there; text flows past it.
    - *Centered lines*: every line starts near the horizontal center, so
      saccades to find the line start are short and symmetric.
    - *~42-char measure*: line width is derived from font metrics, never from
      the window width. Short lines eliminate return-sweep regressions.
    - *Stable lines*: a line keeps its identity (utterance, line number) for
      as long as it is shown; it only slides up when a new line arrives and
      fades out when it scrolls off the top.
    - *Quiet when idle*: after a few seconds without speech the block
      shrinks to a small handle; hovering it brings the last lines back.
      The handle is always there to drag, right-click, or hover, and the
      tray icon is always there to exit.
    - *CJK-aware wrapping*: lines can break between CJK characters, not just
      at spaces.

    Rendering is retained-mode: text and state changes only set targets
    (`_retarget`); `_anim_frame` springs every line and the background pill
    toward them and paints, running only while something is unsettled.
    """

    MAX_LINES = 3                 # history + active, bounded block height
    MEASURE_CHARS = 42            # target characters per line (caption standard)
    _SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+|(?<=[。！？])')

    IDLE_SECONDS = 4.0            # no new text this long -> shrink to the handle
    HOVER_GRACE = 0.8             # stay revealed this long after the pointer leaves
    TOAST_SECONDS = 1.5           # how long a setting-change message shows

    _C_TEXT = 255                 # current utterance
    _C_TENTATIVE = 190            # its uncommitted tail: may still change
    _C_HIST = 232                 # previous utterance
    _C_STATUS = 200               # loading / placeholder / toast messages
    _BG_RGB = (10, 10, 12)
    _BG_ALPHA = 192               # ~75% black pill: readable over any video
    _HANDLE_ALPHA = 200

    # spring time constants (s); ~3.5 tau to settle
    _TAU_MOVE = 0.035             # line slides and pill resizing
    _TAU_FADE_IN = 0.04
    _TAU_FADE_OUT = 0.08
    _TAU_REVEAL = 0.05
    _TAU_COLLAPSE = 0.12          # calm shrink to the idle handle

    _POLL_MS = 10                 # delivery latency: results hit the screen within this

    def __init__(self, font_size: Optional[float] = None):
        _enable_dpi_awareness()
        self._scale = _get_dpi_scale()
        self._settings = _load_settings()
        self._font_size = font_size or self._settings["font_size"]
        self._show_bg: bool = self._settings["show_bg"]
        self._soft_shadow: bool = self._settings["soft_shadow"]
        self._hide_idle: bool = self._settings["hide_idle"]
        self._translating: bool = self._settings["translate"]
        self._font_cache: dict = {}
        self._sprite_cache: dict = {}
        self._dots_cache: dict = {}
        self._surface = _LayeredSurface()
        self._fine_timer = False
        self._translator = None
        self._exit_hooks: list[Callable[[], None]] = []
        self._state_hooks: list[Callable[[], None]] = []
        self._save_after_id = None

        self.root = tk.Tk()
        self.root.withdraw()
        self.root.title("Transcriber Subtitles")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self._sw = self.root.winfo_screenwidth()
        self._sh = self.root.winfo_screenheight()

        # Utterances: hist (previous, dim) and current, which is either
        # active (in-flight partial) or recent (its final). Each has a
        # sequence key so its lines keep their identity across updates.
        self._hist_text = ""
        self._recent_text = ""
        self._active_text = ""
        self._active_stable = 0       # active_text[:stable] is committed
        self._hist_key = 0
        self._cur_key = 0
        self._seq = 0
        self._stale_history = False   # set on collapse: next utterance starts fresh
        self._status = "Starting..."
        self._toast = ""
        self._toast_until = 0.0
        self._last_activity = time.monotonic()
        self._hover = False
        self._hover_left = 0.0
        self._collapsed = False

        self._queue: queue.Queue = queue.Queue()
        self._hwnd: Optional[int] = None
        self._drag_start = None
        self._dragged = False
        self._clock = time.perf_counter
        self._refresh_hz = _get_refresh_hz()
        self._anim_active = False
        self._anim_after_id = None
        self._anim_last_t = 0.0

        # animated scene: lines by key, and the pill as
        # [x0, y0, x1, y1, bg alpha, handle-dots opacity] current/target
        self._lines: dict = {}
        self._pill: Optional[list[float]] = None
        self._pill_target: Optional[list[float]] = None
        self._pill_tau = self._TAU_MOVE
        self._font = None

        self._canvas_width: int = 0
        self._canvas_height: int = 0
        self._win_x = self._win_y = 0
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
        self._hwnd = windll.user32.GetAncestor(self.root.winfo_id(), 2)
        self._make_styling()
        self._menu = self._build_menu()
        self._enable_input()
        self._retarget(snap=True)
        self.root.after(50, self._poll)

    # ------------------------------------------------------------------ metrics

    def _fs(self) -> int:
        return int(self._font_size * self._scale * 96.0 / 72.0)

    def _default_position(self, w: int, h: int) -> tuple[int, int]:
        return (self._sw - w) // 2, self._sh - h - int(56 * self._scale)

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
        self._corner_r = self._line_h * 0.32
        self._handle_w = self._line_h * 1.4
        self._handle_h = max(6.0, self._line_h * 0.3)
        margin = int(20 * self._scale)

        w = min(self._sw, self._wrap_w + 2 * self._hpad + 2 * margin)
        h = self.MAX_LINES * self._line_h + 2 * self._vpad + 2 * margin
        w, h = int(w), int(h)
        # draw-y of the bottom line; everything stacks up from here
        self._base_y = h - margin - self._vpad - self._line_h + self._line_gap

        prev_w, prev_h = self._canvas_width, self._canvas_height
        self._canvas_width, self._canvas_height = w, h
        if initial or prev_w == 0:
            x, y = self._settings["x"], self._settings["y"]
            if x is None or y is None or not _on_screen(int(x), int(y), w, h):
                x, y = self._default_position(w, h)
        else:
            x = self._win_x + (prev_w - w) // 2
            y = self._win_y + (prev_h - h)
            if self._settings["x"] is not None:
                self._settings["x"], self._settings["y"] = int(x), int(y)
        self._move_window(int(x), int(y), w, h)
        self._sprite_cache.clear()

    def _move_window(self, x: int, y: int, w: Optional[int] = None, h: Optional[int] = None):
        self._win_x, self._win_y = x, y
        if w is None:
            self.root.geometry(f"+{x}+{y}")
        else:
            self.root.geometry(f"{w}x{h}+{x}+{y}")

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
        candidates = _font_candidates(text)
        key = (candidates, fs)
        font = self._font_cache.get(key)
        if font is None:
            if len(self._font_cache) > 64:
                self._font_cache.clear()
            for name in candidates:
                try:
                    font = ImageFont.truetype(os.path.join(_FONT_DIR, name), fs)
                    break
                except (IOError, OSError):
                    continue
            else:
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
        return lines

    def _wrap_sentences(self, text: str, font, max_w: int):
        """Wrap per sentence so line breaks land on natural pauses."""
        lines: list[str] = []
        for sentence in self._SENTENCE_SPLIT.split(text):
            if sentence and sentence.strip():
                lines.extend(self._wrap(sentence.strip(), font, max_w))
        return lines

    # -------------------------------------------------------------- text state

    def set_status(self, text: str):
        """Show a loading/progress message. Safe from any thread."""
        self._queue.put(("status", text or ""))

    def on_partial(self, text: str, stable: int = 0):
        self._queue.put(("partial", (text, stable)))

    def on_final(self, text: str):
        self._queue.put(("final", text))

    def post_command(self, name: str):
        """Run a UI command (see _command) on the Tk thread. Safe from any thread."""
        self._queue.put(("cmd", name))

    def _touch(self):
        self._last_activity = time.monotonic()

    def _begin_utterance(self):
        if self._recent_text and not self._stale_history:
            self._hist_text, self._hist_key = self._recent_text, self._cur_key
        else:
            self._hist_text = ""
        self._recent_text = ""
        self._stale_history = False
        self._seq += 1
        self._cur_key = self._seq

    def _apply_partial(self, text: str, stable: int) -> bool:
        if not text or (text, stable) == (self._active_text, self._active_stable):
            return False
        if not self._active_text:
            self._begin_utterance()
        self._active_text, self._active_stable = text, stable
        self._touch()
        return True

    def _apply_final(self, text: str) -> bool:
        if self._active_text:
            # the partial becomes the final; an empty final retracts it
            self._active_text = ""
            self._recent_text = text
        elif text:
            self._begin_utterance()
            self._recent_text = text
        else:
            return False
        self._touch()
        return True

    def _show_toast(self, text: str):
        self._toast = text
        self._toast_until = time.monotonic() + self.TOAST_SECONDS
        self._touch()
        self._retarget()

    # ------------------------------------------------------------------ targets

    def _should_collapse(self, now: float) -> bool:
        if not self._hide_idle or self._status or self._toast:
            return False
        if self._hover or self._drag_start is not None or now - self._hover_left < self.HOVER_GRACE:
            return False
        return now - self._last_activity > self.IDLE_SECONDS

    def _content(self) -> list:
        """[(key, text, color, stable_chars)] for what should be on screen
        when expanded; text past stable_chars is drawn as tentative."""
        if self._status:
            return [("status", self._status, self._C_STATUS, len(self._status))]
        if self._toast:
            return [("toast", self._toast, self._C_STATUS, len(self._toast))]
        if self._active_text or self._recent_text:
            items = []
            if self._hist_text:
                items.append((("u", self._hist_key), self._hist_text, self._C_HIST, len(self._hist_text)))
            if self._active_text:
                items.append((("u", self._cur_key), self._active_text, self._C_TEXT, self._active_stable))
            else:
                items.append((("u", self._cur_key), self._recent_text, self._C_TEXT, len(self._recent_text)))
            return items
        placeholder = "Listening..." if self._translating else "Transcribing..."
        return [("placeholder", placeholder, self._C_STATUS, len(placeholder))]

    def _retarget(self, snap: bool = False):
        """Recompute where every line and the pill should be, then animate."""
        was_collapsed = self._collapsed
        self._collapsed = self._should_collapse(time.monotonic())
        if self._collapsed and not was_collapsed:
            self._stale_history = True
        W = self._canvas_width

        targets = {}
        if not self._collapsed:
            items = self._content()
            fs = self._fs()
            font = self._get_font(fs, " ".join(item[1] for item in items))
            self._font = font
            rows = []
            for key, text, color, stable in items:
                pos = 0
                for i, line in enumerate(self._wrap_sentences(text, font, self._wrap_w)):
                    # lines are substrings of text: locate this one to know
                    # how much of it is committed
                    start = text.find(line, pos)
                    start = pos if start < 0 else start
                    pos = start + len(line)
                    rows.append(((key, i), line, color, max(0, min(len(line), stable - start))))
            rows = rows[-self.MAX_LINES:]
            for idx, (key, line, color, split) in enumerate(rows):
                width = font.getlength(line)
                ty = float(self._base_y - (len(rows) - 1 - idx) * self._line_h)
                targets[key] = (line, color, width, (W - width) / 2.0, ty, split)

        # existing lines: retarget, or start exiting
        top_y = min((t[4] for t in targets.values()), default=None)
        moved_up = any(key in targets and targets[key][4] < ln.ty - 0.5
                       for key, ln in self._lines.items())
        for key, ln in self._lines.items():
            t = targets.get(key)
            if t is not None:
                ln.text, ln.color, ln.split, ln.font = t[0], t[1], t[5], self._font
                ln.tx, ln.ty = t[3], t[4]
                if ln.exiting or ln.ta < 1.0:
                    ln.tau_a = self._TAU_FADE_IN
                ln.ta, ln.exiting = 1.0, False
            elif not ln.exiting:
                ln.exiting, ln.ta = True, 0.0
                if self._collapsed:
                    ln.tau_a = self._TAU_COLLAPSE
                elif moved_up and top_y is not None and ln.ty <= top_y + 0.5:
                    # scrolled off the top: keep moving up with the others
                    ln.ty -= self._line_h
                    ln.tau_a = self._TAU_FADE_OUT
                else:
                    # replaced in place: clear out fast so the old and new
                    # text don't visibly cross-fade over each other
                    ln.tau_a = self._TAU_FADE_IN
        for key, t in targets.items():
            if key not in self._lines:
                tau = self._TAU_REVEAL if was_collapsed else self._TAU_FADE_IN
                self._lines[key] = _Line(t[0], t[1], t[5], self._font, t[3], t[4], tau)

        # pill: hug the target text, or shrink to the idle handle
        bottom = self._base_y + self._line_h - self._line_gap + self._vpad
        handle = [W / 2.0 - self._handle_w / 2, bottom - self._handle_h,
                  W / 2.0 + self._handle_w / 2, bottom]
        if self._collapsed:
            self._pill_target = handle + [float(self._HANDLE_ALPHA), 1.0]
            self._pill_tau = self._TAU_COLLAPSE
        elif not targets:  # nothing printable to show
            self._pill_target = handle + [0.0, 0.0]
            self._pill_tau = self._TAU_FADE_OUT
        else:
            half = max(t[2] for t in targets.values()) / 2.0 + self._hpad
            rect = [W / 2.0 - half, top_y - self._vpad, W / 2.0 + half, bottom]
            alpha = float(self._BG_ALPHA) if self._show_bg else 0.0
            self._pill_target = rect + [alpha, 0.0]
            self._pill_tau = self._TAU_REVEAL if was_collapsed else self._TAU_MOVE

        if snap or self._pill is None:
            self._pill = list(self._pill_target)
        if snap:
            for ln in self._lines.values():
                ln.x, ln.y, ln.a = ln.tx, ln.ty, ln.ta
        self._update_hit_box()
        self._ensure_anim()

    # ----------------------------------------------------------------- animation

    def _ensure_anim(self):
        """Run the springs at the monitor's refresh rate until settled."""
        if self._anim_active:
            return
        self._anim_active = True
        self._set_fine_timer(True)
        self._anim_last_t = self._clock()
        self._anim_frame()

    def _set_fine_timer(self, fine: bool):
        """1 ms system timer resolution while animating; otherwise Tk's
        after() rounds frame intervals up to the ~15.6 ms default tick."""
        if fine == self._fine_timer:
            return
        self._fine_timer = fine
        try:
            if fine:
                windll.winmm.timeBeginPeriod(1)
            else:
                windll.winmm.timeEndPeriod(1)
        except Exception:
            pass

    def _anim_frame(self):
        try:
            self._anim_step()
        except Exception:
            # stop cleanly instead of leaving the loop marked active forever
            # (which would freeze the subtitles); the next retarget restarts it
            traceback.print_exc()
            self._anim_active = False
            self._anim_after_id = None
            self._set_fine_timer(False)

    def _anim_step(self):
        """One spring integration step + repaint. Critically-damped
        exponential steps: framerate-independent, no overshoot."""
        now = self._clock()
        dt = min(now - self._anim_last_t, 0.1)  # clamp hiccups
        self._anim_last_t = now

        k_move = 1.0 - math.exp(-dt / self._TAU_MOVE)
        for key in list(self._lines):
            ln = self._lines[key]
            ln.x += (ln.tx - ln.x) * k_move
            ln.y += (ln.ty - ln.y) * k_move
            ln.a += (ln.ta - ln.a) * (1.0 - math.exp(-dt / ln.tau_a))
            if ln.exiting and ln.a < 0.004:
                del self._lines[key]
        p, t = self._pill, self._pill_target
        k_pill = 1.0 - math.exp(-dt / self._pill_tau)
        for i in range(6):
            p[i] += (t[i] - p[i]) * k_pill

        settled = self._settled()
        if settled:
            # snap so the resting frame is exact
            for ln in self._lines.values():
                ln.x, ln.y, ln.a = ln.tx, ln.ty, ln.ta
            self._pill[:] = self._pill_target
        self._draw()

        if settled:
            self._anim_active = False
            self._anim_after_id = None
            self._set_fine_timer(False)
            return
        # frame period minus the time this frame took to render
        interval_ms = max(1, round(1000.0 / self._refresh_hz - (self._clock() - now) * 1000.0))
        self._anim_after_id = self.root.after(interval_ms, self._anim_frame)

    def _settled(self) -> bool:
        for ln in self._lines.values():
            if (ln.exiting or abs(ln.x - ln.tx) > 0.1 or abs(ln.y - ln.ty) > 0.1
                    or abs(ln.a - ln.ta) > 0.004):
                return False
        p, t = self._pill, self._pill_target
        return (all(abs(p[i] - t[i]) < 0.1 for i in range(4))
                and abs(p[4] - t[4]) < 0.5 and abs(p[5] - t[5]) < 0.004)

    # ------------------------------------------------------------------- render

    def _draw(self):
        if self._hwnd is None:
            return
        self._surface.ensure(self._canvas_width, self._canvas_height)
        if self._surface.pixels is None:
            return
        self._render(self._surface.pixels)
        self._surface.present(self._hwnd)

    def _render(self, out: np.ndarray):
        """Compose one frame into `out` (premultiplied BGRA, uint8).

        Lines and handle dots are cached premultiplied sprites, so a frame is
        a pill fill plus a few blits, done in float over the bounding box of
        what is visible."""
        w, h = self._canvas_width, self._canvas_height
        p = self._pill
        fs = self._fs()

        pill = None
        if p[4] > 0.5:
            # clamped inside the canvas so corners never clip
            pill = (max(1.0, p[0]), max(1.0, p[1]), min(w - 1.0, p[2]), min(h - 1.0, p[3]))
            if pill[2] - pill[0] < 1 or pill[3] - pill[1] < 1:
                pill = None

        layers = []  # (premultiplied, inv_alpha, x, y, opacity)
        if p[5] > 0.004:
            arr, inv = self._dots_sprite()
            x = int(round((p[0] + p[2] - arr.shape[1]) / 2))
            y = int(round((p[1] + p[3] - arr.shape[0]) / 2))
            layers.append((arr, inv, x, y, p[5]))
        for ln in self._lines.values():
            if ln.a <= 0.004:
                continue
            (arr, inv), ox, oy = self._line_sprite(ln.text, ln.font, fs, ln.color, ln.split)
            layers.append((arr, inv, int(round(ln.x)) - ox, int(round(ln.y)) - oy, ln.a))

        # bounding box of everything visible, clipped to the canvas
        boxes = [(x, y, x + a.shape[1], y + a.shape[0]) for a, _, x, y, _ in layers]
        if pill is not None:
            boxes.append((int(pill[0]), int(pill[1]), int(pill[2]) + 1, int(pill[3]) + 1))
        out.fill(0)
        if not boxes:
            return
        bx0 = max(0, min(b[0] for b in boxes)); by0 = max(0, min(b[1] for b in boxes))
        bx1 = min(w, max(b[2] for b in boxes)); by1 = min(h, max(b[3] for b in boxes))
        if bx0 >= bx1 or by0 >= by1:
            return

        acc = np.zeros((by1 - by0, bx1 - bx0, 4), dtype=np.float32)
        if pill is not None:
            self._stamp_bg(acc, pill, bx0, by0, p[4])
        for arr, inv, x, y, opacity in layers:
            _blit_over(acc, arr, inv, x - bx0, y - by0, opacity)
        acc += 0.5
        out[by0:by1, bx0:bx1] = acc

    def _line_sprite(self, line: str, font, fs: int, color: int, split: int):
        """Premultiplied image of one line (drop shadow + text), cached;
        line[split:] is drawn in the tentative color.
        Returns ((array, inv_alpha), ox, oy): the text origin in the array."""
        key = (line, getattr(font, "path", None), fs, color, split, self._soft_shadow)
        hit = self._sprite_cache.get(key)
        if hit is not None:
            return hit
        soff = max(1, int(fs * 0.06))
        blur = max(1, int(fs * 0.06)) if self._soft_shadow else 0
        pad = 3 * blur + 1
        l, t, r, b = font.getbbox(line)
        ox, oy = pad - min(0, int(l)), pad - min(0, int(t))
        sw = ox + int(r) + soff + pad + 1
        sh = oy + int(b) + soff + pad + 1
        shadow = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
        ImageDraw.Draw(shadow).text((ox + soff, oy + soff), line, font=font, fill=(0, 0, 0, 220))
        if blur:
            shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur))
        text = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text)
        draw.text((ox, oy), line[:split], font=font, fill=(color,) * 3 + (255,))
        if split < len(line):
            x = ox + font.getlength(line[:split])
            draw.text((x, oy), line[split:], font=font, fill=(self._C_TENTATIVE,) * 3 + (255,))
        if len(self._sprite_cache) > 64:
            self._sprite_cache.clear()
        hit = self._sprite_cache[key] = (_sprite_arrays(Image.alpha_composite(shadow, text)), ox, oy)
        return hit

    def _dots_sprite(self):
        """Three small dots: the idle handle's 'more here' affordance."""
        d = max(3, int(round(self._handle_h * 0.3)))
        hit = self._dots_cache.get(d)
        if hit is None:
            SS = 4
            gap = d * 1.3
            w, h = int(3 * d + 2 * gap) + 2, d + 2
            img = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            for i in range(3):
                x0 = (1 + i * (d + gap)) * SS
                draw.ellipse((x0, SS, x0 + d * SS, SS + d * SS), fill=(235, 235, 235, 230))
            hit = self._dots_cache[d] = _sprite_arrays(img.resize((w, h), Image.LANCZOS))
        return hit

    def _stamp_bg(self, acc: np.ndarray, rect, ox: int, oy: int, alpha: float):
        """Translucent rounded pill as an analytic signed-distance coverage
        mask: exact antialiasing at sub-pixel positions, so the springs move
        it smoothly, and cheap enough to evaluate every frame. Drawn first,
        onto the empty accumulator `acc`, whose (0, 0) is canvas (ox, oy)."""
        x0, y0, x1, y1 = rect
        radius = max(0.5, min(self._corner_r, (y1 - y0) / 2, (x1 - x0) / 2))
        ix0, iy0 = max(int(x0), ox), max(int(y0), oy)
        ix1 = min(int(x1) + 1, ox + acc.shape[1])
        iy1 = min(int(y1) + 1, oy + acc.shape[0])
        if ix0 >= ix1 or iy0 >= iy1:
            return
        # pixel centers, folded into one quadrant of the rect
        qx = np.abs(np.arange(ix0, ix1, dtype=np.float32) + 0.5 - (x0 + x1) / 2) - ((x1 - x0) / 2 - radius)
        qy = np.abs(np.arange(iy0, iy1, dtype=np.float32) + 0.5 - (y0 + y1) / 2) - ((y1 - y0) / 2 - radius)
        # Off the corners the distance is separable, max(qx, qy) - radius,
        # so coverage is the min of two 1-D edge ramps...
        cov = np.minimum(np.clip(0.5 + radius - qx, 0, 1)[None, :],
                         np.clip(0.5 + radius - qy, 0, 1)[:, None])
        # ...and only the corner patches (qx > 0 and qy > 0) need the arc.
        cols = np.flatnonzero(qx > 0)
        rows = np.flatnonzero(qy > 0)
        for cs in np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1):
            for rs in np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1):
                if len(cs) and len(rs):
                    d = np.hypot(qx[cs][None, :], qy[rs][:, None]) - radius
                    cov[rs[0]:rs[-1] + 1, cs[0]:cs[-1] + 1] = np.clip(0.5 - d, 0, 1)
        region = acc[iy0 - oy:iy1 - oy, ix0 - ox:ix1 - ox]
        r, g, b = self._BG_RGB
        # per channel: numpy broadcasting over a length-4 axis is ~6x slower
        for c, v in enumerate((b, g, r, 255)):
            np.multiply(cov, v * alpha / 255.0, out=region[:, :, c])

    # ----------------------------------------------------------------- commands

    @property
    def translating(self) -> bool:
        return self._translating

    @property
    def hide_idle(self) -> bool:
        return self._hide_idle

    @property
    def show_bg(self) -> bool:
        return self._show_bg

    @property
    def soft_shadow(self) -> bool:
        return self._soft_shadow

    def set_translator(self, translator):
        self._translator = translator
        translator.set_translate(self._translating)

    def add_exit_hook(self, fn: Callable[[], None]):
        """Called on the Tk thread just before the process exits."""
        self._exit_hooks.append(fn)

    def add_state_hook(self, fn: Callable[[], None]):
        """Called on the Tk thread after any setting changes."""
        self._state_hooks.append(fn)

    def _command(self, name: str):
        if name == "translate":
            self._translating = not self._translating
            if self._translator:
                self._translator.set_translate(self._translating)
            self._show_toast("Translating to English" if self._translating else "Showing original language")
        elif name == "hide_idle":
            self._hide_idle = not self._hide_idle
            self._show_toast("Subtitles shrink when idle" if self._hide_idle else "Subtitles stay on screen")
        elif name == "bg":
            self._show_bg = not self._show_bg
            self._retarget()
        elif name == "shadow":
            self._soft_shadow = not self._soft_shadow
            self._ensure_anim()  # repaint with the other sprites
        elif name == "show":
            self._reset_position()
            self._touch()
            self._retarget()
        elif name == "exit":
            self.exit()
            return
        self._save_settings_soon()
        for hook in self._state_hooks:
            try:
                hook()
            except Exception:
                pass

    def _save_settings_soon(self):
        if self._save_after_id is not None:
            self.root.after_cancel(self._save_after_id)
        self._save_after_id = self.root.after(400, self._save_settings_now)

    def _save_settings_now(self):
        self._save_after_id = None
        self._settings.update(
            font_size=self._font_size, show_bg=self._show_bg, soft_shadow=self._soft_shadow,
            hide_idle=self._hide_idle, translate=self._translating,
        )
        _save_settings(self._settings)

    def exit(self):
        if self._save_after_id is not None:
            self._save_settings_now()
        for hook in self._exit_hooks:
            try:
                hook()
            except Exception:
                pass
        os._exit(0)

    # -------------------------------------------------------------------- input

    def _build_menu(self) -> tk.Menu:
        self._menu_vars = {k: tk.BooleanVar(master=self.root) for k in ("translate", "hide_idle", "bg", "shadow")}
        menu = tk.Menu(self.root, tearoff=0)
        for key, label in (("translate", "Translate to English"), ("hide_idle", "Shrink when idle"),
                           ("bg", "Background"), ("shadow", "Soft shadow")):
            menu.add_checkbutton(label=label, variable=self._menu_vars[key],
                                 command=lambda k=key: self._command(k))
        menu.add_separator()
        menu.add_command(label="Reset position", command=lambda: self._command("show"))
        menu.add_separator()
        menu.add_command(label="Exit", command=lambda: self._command("exit"))
        return menu

    def _on_right_click(self, event):
        for key, value in (("translate", self._translating), ("hide_idle", self._hide_idle),
                           ("bg", self._show_bg), ("shadow", self._soft_shadow)):
            self._menu_vars[key].set(value)
        try:
            self._menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._menu.grab_release()

    def _enable_input(self):
        for win in (self._input_win, self._canvas):
            win.bind("<Button-1>", self._on_drag_start)
            win.bind("<B1-Motion>", self._on_drag_move)
            win.bind("<ButtonRelease-1>", self._on_drag_end)
            win.bind("<Button-3>", self._on_right_click)
            win.bind("<MouseWheel>", self._on_scroll)
        self._input_win.bind("<Enter>", self._on_enter)
        self._input_win.bind("<Leave>", self._on_leave)
        for win in (self.root, self._input_win):
            win.bind("<Key-space>", lambda e: self._command("bg"))
            win.bind("<Key-t>", lambda e: self._command("translate"))
            win.bind("<Key-s>", lambda e: self._command("shadow"))
            win.bind("<Key-h>", lambda e: self._command("hide_idle"))
            win.bind("<Escape>", lambda e: self.exit())

    def _on_enter(self, event):
        self._hover = True
        if self._collapsed:
            self._retarget()

    def _on_leave(self, event):
        self._hover = False
        self._hover_left = time.monotonic()

    def _on_drag_start(self, event):
        self._drag_start = (event.x_root, event.y_root)
        self._dragged = False
        self._input_win.focus_force()
        self.root.focus_force()

    def _on_drag_move(self, event):
        if self._drag_start is None:
            return
        dx = event.x_root - self._drag_start[0]
        dy = event.y_root - self._drag_start[1]
        self._move_window(self._win_x + dx, self._win_y + dy)
        self._drag_start = (event.x_root, event.y_root)
        self._dragged = True
        self._update_hit_box()

    def _on_drag_end(self, event):
        self._drag_start = None
        if self._dragged:
            self._settings["x"], self._settings["y"] = self._win_x, self._win_y
            self._save_settings_soon()

    def _reset_position(self):
        self._settings["x"] = self._settings["y"] = None
        x, y = self._default_position(self._canvas_width, self._canvas_height)
        self._move_window(x, y)
        for win in (self.root, self._input_win):
            win.attributes("-topmost", True)
            win.lift()

    def _on_scroll(self, event):
        delta = max(-1, min(1, event.delta))
        self._set_font_size(self._font_size + max(self._font_size * 0.1, 1) * delta)

    def _set_font_size(self, new_size):
        new_size = max(8, min(120, new_size))
        if new_size == self._font_size:
            return
        self._font_size = new_size
        self._apply_metrics()
        self._lines.clear()
        self._pill = None
        self._retarget(snap=True)
        self._save_settings_soon()

    def _update_hit_box(self):
        """Size the near-invisible input window to the pill (or the idle
        handle), so clicks elsewhere pass through to what is underneath."""
        if self._pill_target is None:
            return
        x0, y0, x1, y1 = self._pill_target[:4]
        pad = self._hit_pad * (2 if self._collapsed else 1)
        w = int(x1 - x0) + 2 * pad
        h = int(y1 - y0) + 2 * pad
        self._input_win.geometry(f"{w}x{h}+{int(self._win_x + x0) - pad}+{int(self._win_y + y0) - pad}")

    # --------------------------------------------------------------------- poll

    def _poll(self):
        try:
            self._poll_once()
        except Exception:
            traceback.print_exc()
        # always reschedule: one bad update must not stop all future ones
        self.root.after(self._POLL_MS, self._poll)

    def _poll_once(self):
        # Apply every queued update in order, then retarget once: a burst of
        # updates arriving in one tick costs a single layout.
        changed = False
        try:
            while True:
                kind, text = self._queue.get_nowait()
                if kind == "status":
                    if self._status != text:
                        self._status = text
                        self._touch()
                        changed = True
                elif kind == "partial":
                    changed |= self._apply_partial(*text)
                elif kind == "final":
                    changed |= self._apply_final(text)
                elif kind == "cmd":
                    self._command(text)
        except queue.Empty:
            pass
        now = time.monotonic()
        if self._toast and now >= self._toast_until:
            self._toast = ""
            changed = True
        if changed or self._should_collapse(now) != self._collapsed:
            self._retarget()

    def run(self):
        self.root.mainloop()
