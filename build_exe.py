"""Build a distributable .exe for the Transcriber application.

Usage:
    python build_exe.py

Output is written to dist/Transcriber/.
The Whisper model is NOT bundled -- it downloads from HuggingFace on first run.
"""

import os
import subprocess
import sys


def venv_python():
    venv = os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe")
    return venv if os.path.isfile(venv) else sys.executable


def main():
    python = venv_python()

    try:
        import PyInstaller  # noqa: F811
    except ImportError:
        subprocess.check_call([python, "-m", "pip", "install", "pyinstaller"])

    root = os.path.dirname(__file__)
    dist_dir = os.path.join(root, "dist")

    cmd = [
        python, "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",
        "--name", "Transcriber",
        "--distpath", dist_dir,
        "--add-data", f"{os.path.join(root, 'subtitles.py')}{os.pathsep}.",
        # collect package data
        "--collect-data", "silero_vad",
        "--collect-all", "pyaudiowpatch",
        "--collect-all", "sentencepiece",
        "--hidden-import", "torch",
        "--hidden-import", "transformers",
        "--hidden-import", "accelerate",
        "--hidden-import", "numpy",
        "--hidden-import", "PIL",
        "--hidden-import", "PIL._tkinter_finder",
        "--hidden-import", "silero_vad",
        "--hidden-import", "pyaudiowpatch",
        "--hidden-import", "ctypes",
        "--hidden-import", "ctypes.wintypes",
        "--hidden-import", "queue",
        "--hidden-import", "threading",
        # entry point
        os.path.join(root, "subtitles.py"),
    ]

    print("Building Transcriber executable...")
    subprocess.check_call(cmd, cwd=root)
    print(f"\nDone! Executable is at: {os.path.join(dist_dir, 'Transcriber', 'Transcriber.exe')}")
    print("Run it directly -- the Whisper model will download on first launch.")


if __name__ == "__main__":
    main()
