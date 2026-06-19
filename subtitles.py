#!/usr/bin/env python3
"""Desktop audio translator with always-on-top subtitle overlay.

Run:
    python subtitles.py

Press Escape to close the overlay and stop translation.
"""

import threading

from main import LiveTranslator
from overlay import SubtitleOverlay


def main():
    translator = LiveTranslator()

    print("Loading models (this takes a moment)...")
    translator.start()
    print("Models loaded.\n")

    overlay = SubtitleOverlay()
    translator.set_output(overlay)

    def shutdown():
        threading.Thread(target=translator.stop, daemon=True).start()
        overlay.stop()

    overlay.root.bind("<Escape>", lambda e: shutdown())

    t = threading.Thread(target=translator.run, daemon=True)
    t.start()

    print("Subtitle overlay opened (Escape to exit).")
    try:
        overlay.run()
    except KeyboardInterrupt:
        shutdown()
    finally:
        translator.stop()


if __name__ == "__main__":
    main()
