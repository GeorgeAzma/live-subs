#!/usr/bin/env python3
"""Desktop audio translator with always-on-top subtitle overlay.

Run:
    python subtitles.py

Press Escape to close the overlay and stop translation.
"""

import os
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

    t = threading.Thread(target=translator.run, daemon=True)
    t.start()

    print("Subtitle overlay opened (Escape to exit).")
    try:
        overlay.run()
    except KeyboardInterrupt:
        pass
    os._exit(0)


if __name__ == "__main__":
    main()
