import os
import threading

from main import LiveTranslator
from overlay import SubtitleOverlay


def main():
    # Show the overlay window immediately so the user gets feedback right away,
    # then load the heavy models in a background thread.
    overlay = SubtitleOverlay()
    overlay.set_status("Loading...")

    translator = LiveTranslator()
    overlay.set_translator(translator)
    translator.set_output(overlay)
    translator.set_status_callback(overlay.set_status)

    def _load():
        try:
            translator.start()
            # Clear the status line; the overlay switches to "Listening...".
            overlay.set_status("")
        except Exception as e:
            overlay.set_status(f"Failed to start: {e}")
            raise

    threading.Thread(target=_load, daemon=True).start()

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
