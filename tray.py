"""Notification-area icon: an always-reachable way to control or exit the
overlay, even when the subtitles are collapsed, off-screen, or unfocused."""

import threading

import pystray
from PIL import Image, ImageDraw


def _icon_image(size: int = 64) -> Image.Image:
    """A dark caption box with two subtitle lines."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = size / 64
    d.rounded_rectangle((2 * s, 8 * s, 62 * s, 56 * s), radius=10 * s, fill=(20, 20, 24, 255),
                        outline=(235, 235, 235, 255), width=max(1, int(3 * s)))
    d.rounded_rectangle((13 * s, 24 * s, 51 * s, 31 * s), radius=3 * s, fill=(255, 255, 255, 255))
    d.rounded_rectangle((19 * s, 36 * s, 45 * s, 43 * s), radius=3 * s, fill=(190, 190, 190, 255))
    return img


class TrayIcon:
    def __init__(self, overlay):
        o = overlay

        def item(text, command, checked=None, default=False):
            # menu callbacks run on the tray thread: hand off to the Tk thread
            return pystray.MenuItem(text, lambda icon, it: o.post_command(command),
                                    checked=checked, default=default)

        menu = pystray.Menu(
            item("Show subtitles", "show", default=True),  # also: left-click the icon
            pystray.Menu.SEPARATOR,
            item("Translate to English", "translate", checked=lambda it: o.translating),
            item("Shrink when idle", "hide_idle", checked=lambda it: o.hide_idle),
            item("Background", "bg", checked=lambda it: o.show_bg),
            item("Soft shadow", "shadow", checked=lambda it: o.soft_shadow),
            pystray.Menu.SEPARATOR,
            item("Exit", "exit"),
        )
        self._icon = pystray.Icon("transcriber", _icon_image(), "Transcriber", menu)

    def start(self):
        threading.Thread(target=self._icon.run, daemon=True).start()

    def refresh(self):
        """Rebuild the menu so check marks follow changes made elsewhere."""
        try:
            self._icon.update_menu()
        except Exception:
            pass

    def stop(self):
        # remove the icon synchronously: os._exit right after would otherwise
        # leave a dead icon in the tray until the user hovers it
        try:
            self._icon.visible = False
            self._icon.stop()
        except Exception:
            pass
