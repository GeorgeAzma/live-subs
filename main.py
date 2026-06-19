import queue
import sys
import threading
import warnings

import numpy as np
import pyaudiowpatch as pyaudio
import torch
import transformers

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

DEVICE_INDEX = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
MIN_SECS = 2
MAX_SECS = 30
SILENCE_SECS = 0.8


def find_loopback_device(p):
    loopback_devices = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        host = p.get_host_api_info_by_index(info["hostApi"])["name"]
        if "wasapi" in host.lower() and info["maxInputChannels"] > 0 and "loopback" in info["name"].lower():
            loopback_devices.append((i, info))
    if not loopback_devices:
        print("No loopback device found.")
        sys.exit(1)
    preferred = [d for d in loopback_devices if "headphone" in d[1]["name"].lower() or "headset" in d[1]["name"].lower()]
    return preferred[0] if preferred else loopback_devices[0]


def to_mono_16k(data, orig_rate, channels):
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if orig_rate == SAMPLE_RATE:
        return (data / 32768.0).astype(np.float32)
    target = int(len(data) * SAMPLE_RATE / orig_rate)
    return np.interp(np.linspace(0, len(data) - 1, target), np.arange(len(data)), data.astype(np.float64)).astype(np.float32) / 32768.0


def is_silence(audio, threshold=0.01):
    return np.sqrt(np.mean(audio ** 2)) < threshold


def worker(pipe, q, stop_event):
    while not stop_event.is_set():
        try:
            segment = q.get(timeout=0.5)
        except queue.Empty:
            continue
        if segment is None:
            break
        result = pipe(
            {"array": segment, "sampling_rate": SAMPLE_RATE},
            generate_kwargs={"task": "translate", "temperature": 0},
        )
        text = result["text"].strip()
        if text:
            sys.stdout.write(f"\r\033[K{text}\n")
            sys.stdout.flush()


def main():
    global DEVICE_INDEX

    print(f"Loading Whisper large-v3-turbo on {DEVICE.upper()}...")
    pipe = transformers.pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        device=DEVICE,
    )
    print("Model loaded.\n")

    p = pyaudio.PyAudio()
    if DEVICE_INDEX is None:
        dev_idx, dev_info = find_loopback_device(p)
    else:
        dev_idx = DEVICE_INDEX
        dev_info = p.get_device_info_by_index(dev_idx)

    native_rate = int(dev_info["defaultSampleRate"])
    channels = int(dev_info["maxInputChannels"])

    print(f"Device:  {dev_info['name']}")
    print(f"Rate:    {native_rate} Hz | Channels: {channels}")
    print("Listening... Ctrl+C to stop.\n")

    stream = p.open(format=pyaudio.paInt16, channels=channels, rate=native_rate, input=True, input_device_index=dev_idx, frames_per_buffer=1024)

    q = queue.Queue(maxsize=8)
    stop_event = threading.Event()
    t = threading.Thread(target=worker, args=(pipe, q, stop_event), daemon=True)
    t.start()

    buf = np.array([], dtype=np.float32)
    segment = np.array([], dtype=np.float32)
    silent_chunks = 0
    active = False
    frame_samples = 1600  # 100ms at 16kHz
    min_samples = SAMPLE_RATE * MIN_SECS
    max_samples = SAMPLE_RATE * MAX_SECS
    silence_frames = int(SILENCE_SECS / 0.1)  # 8 frames of 100ms

    def flush_segment():
        nonlocal segment, silent_chunks, active
        if len(segment) >= min_samples:
            q.put_nowait(segment.copy())
            sys.stdout.write("\r\033[K\x1b[90m───\x1b[0m\n")
            sys.stdout.flush()
        segment = np.array([], dtype=np.float32)
        silent_chunks = 0
        active = False

    try:
        while True:
            raw = stream.read(1024, exception_on_overflow=False)
            samples = np.frombuffer(raw, dtype=np.int16)
            rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
            sys.stdout.write(f"\r{rms_to_bar(rms)}  ")
            sys.stdout.flush()

            buf = np.concatenate([buf, to_mono_16k(samples, native_rate, channels)])
            n = (len(buf) // frame_samples) * frame_samples
            if n < frame_samples:
                continue

            frame = buf[:n]
            buf = buf[n:]

            for i in range(0, n, frame_samples):
                blk = frame[i:i + frame_samples]
                if is_silence(blk):
                    if active:
                        silent_chunks += 1
                        if silent_chunks >= silence_frames:
                            flush_segment()
                else:
                    silent_chunks = 0
                    segment = np.concatenate([segment, blk])
                    if not active:
                        active = True
                    if len(segment) >= max_samples:
                        flush_segment()

    except KeyboardInterrupt:
        if len(segment) >= min_samples:
            q.put_nowait(segment.copy())
        print("\n\nStopping...")
    finally:
        stop_event.set()
        q.put(None)
        t.join(timeout=5)
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Done.")


def rms_to_bar(rms):
    db = 20 * np.log10(max(rms, 1e-10))
    bar_len = int(min(max((db + 60) / 60, 0), 1) * 40)
    bar = "\u2588" * bar_len + "\u2591" * (40 - bar_len)
    return f"[{bar}] {db:+.1f} dB"


if __name__ == "__main__":
    main()
