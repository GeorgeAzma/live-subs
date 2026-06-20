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
    overlay.set_translator(translator)
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
