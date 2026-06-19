import queue
import sys
import threading
import time
import warnings
from collections import deque

import numpy as np
import pyaudiowpatch as pyaudio
import silero_vad
import torch
import transformers

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

DEVICE_INDEX = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
MIN_SECS = 1
MAX_SECS = 8
INTERIM_INTERVAL = 2.0


def find_loopback_device(p):
    loopback_devices = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        host = p.get_host_api_info_by_index(info["hostApi"])["name"]
        if (
            "wasapi" in host.lower()
            and info["maxInputChannels"] > 0
            and "loopback" in info["name"].lower()
        ):
            loopback_devices.append((i, info))
    if not loopback_devices:
        print("No loopback device found.")
        sys.exit(1)
    preferred = [
        d
        for d in loopback_devices
        if "headphone" in d[1]["name"].lower() or "headset" in d[1]["name"].lower()
    ]
    return preferred[0] if preferred else loopback_devices[0]


def to_mono_16k(data, orig_rate, channels):
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if orig_rate == SAMPLE_RATE:
        return (data / 32768.0).astype(np.float32)
    target = int(len(data) * SAMPLE_RATE / orig_rate)
    return (
        np.interp(
            np.linspace(0, len(data) - 1, target),
            np.arange(len(data)),
            data.astype(np.float64),
        ).astype(np.float32)
        / 32768.0
    )


def worker(model, processor, q, partial_q, stop_event):
    while not stop_event.is_set():
        try:
            item = q.get(timeout=0.5)
        except queue.Empty:
            continue
        if item is None:
            break
        audio, is_final = item
        inputs = processor(audio, return_tensors="pt", sampling_rate=SAMPLE_RATE)
        input_features = inputs.input_features.to(DEVICE, dtype=model.dtype)
        with torch.no_grad():
            generated = model.generate(
                input_features,
                temperature=0,
            )
        text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        if not text:
            if is_final:
                partial_q.put(None)
            continue
        if is_final:
            partial_q.put(None)
            sys.stdout.write(f"\r\033[K{text}\n")
            sys.stdout.flush()
        else:
            partial_q.put(text)


def main():
    global DEVICE_INDEX

    print(f"Loading Whisper large-v3 on {DEVICE.upper()}...")
    model = transformers.WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3",
    ).to(DEVICE)
    model.generation_config.forced_decoder_ids = [[1, None], [2, 50359]]
    processor = transformers.AutoProcessor.from_pretrained("openai/whisper-large-v3")
    print("Model loaded.\n")

    print("Loading Silero VAD...")
    vad_model = silero_vad.load_silero_vad()
    vad_iterator = silero_vad.VADIterator(
        model=vad_model,
        threshold=0.5,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=200,
        speech_pad_ms=50,
    )
    print("VAD loaded.\n")

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

    stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=native_rate,
        input=True,
        input_device_index=dev_idx,
        frames_per_buffer=1024,
    )

    q = queue.Queue(maxsize=4)
    partial_q: queue.Queue[str | None] = queue.Queue()
    stop_event = threading.Event()
    t = threading.Thread(
        target=worker,
        args=(model, processor, q, partial_q, stop_event),
        daemon=True,
    )
    t.start()

    VAD_WINDOW = 512
    min_samples = SAMPLE_RATE * MIN_SECS
    max_samples = SAMPLE_RATE * MAX_SECS

    vad_buf: deque[float] = deque()
    speech_chunks: list[np.ndarray] = []
    speech_samples = 0
    triggered = False
    lookbehind: deque[np.ndarray] = deque(maxlen=2)

    last_interim_time = 0.0
    partial_text = ""

    def push_segment(is_final: bool):
        nonlocal speech_chunks, speech_samples, triggered
        if is_final:
            if speech_samples >= min_samples:
                try:
                    q.put_nowait((np.concatenate(speech_chunks), True))
                except queue.Full:
                    pass
            speech_chunks = []
            speech_samples = 0
            triggered = False
        else:
            if speech_chunks:
                try:
                    q.put_nowait((np.concatenate(speech_chunks), False))
                except queue.Full:
                    pass

    try:
        while True:
            raw = stream.read(1024, exception_on_overflow=False)
            samples = np.frombuffer(raw, dtype=np.int16)
            rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))

            # Drain latest partial text from worker
            while True:
                try:
                    msg = partial_q.get_nowait()
                    if msg is None:
                        partial_text = ""
                    else:
                        partial_text = msg
                except queue.Empty:
                    break

            # Write meter line with partial if active
            sys.stdout.write(f"\r{rms_to_bar(rms)}")
            if partial_text:
                sys.stdout.write(f"  \033[90m▸\033[0m {partial_text}")
            sys.stdout.write("  ")
            sys.stdout.flush()

            mono = to_mono_16k(samples, native_rate, channels)
            vad_buf.extend(mono.tolist())

            while len(vad_buf) >= VAD_WINDOW:
                chunk = np.array(
                    [vad_buf.popleft() for _ in range(VAD_WINDOW)], dtype=np.float32
                )

                result = vad_iterator(torch.from_numpy(chunk), return_seconds=False)

                if result is None:
                    if triggered:
                        speech_chunks.append(chunk)
                        speech_samples += VAD_WINDOW
                        if speech_samples >= max_samples:
                            push_segment(is_final=True)
                elif "start" in result:
                    triggered = True
                    speech_chunks = [*lookbehind, chunk]
                    speech_samples = VAD_WINDOW * len(speech_chunks)
                    last_interim_time = time.monotonic()
                elif "end" in result:
                    if triggered:
                        speech_chunks.append(chunk)
                        speech_samples += VAD_WINDOW
                        push_segment(is_final=True)

                lookbehind.append(chunk)

                # Push interim result periodically during active speech
                if triggered and speech_samples >= min_samples:
                    now = time.monotonic()
                    if now - last_interim_time >= INTERIM_INTERVAL:
                        push_segment(is_final=False)
                        last_interim_time = now

    except KeyboardInterrupt:
        if triggered and speech_samples >= min_samples:
            try:
                q.put_nowait((np.concatenate(speech_chunks), True))
            except queue.Full:
                pass
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
