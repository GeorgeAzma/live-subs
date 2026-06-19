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

SAMPLE_RATE = 16000
VAD_WINDOW = 512
MIN_SEGMENT_SAMPLES = SAMPLE_RATE * 1
MAX_SEGMENT_SAMPLES = SAMPLE_RATE * 8
INTERIM_INTERVAL = 2.0
TRANSLATE_TOKEN = 50359


def find_loopback_device(p):
    devices = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        host = p.get_host_api_info_by_index(info["hostApi"])["name"]
        if "wasapi" in host.lower() and info["maxInputChannels"] > 0 and "loopback" in info["name"].lower():
            devices.append((i, info))
    if not devices:
        print("No loopback device found.")
        sys.exit(1)
    preferred = [d for d in devices if "headphone" in d[1]["name"].lower() or "headset" in d[1]["name"].lower()]
    return preferred[0] if preferred else devices[0]


def to_mono_16k(data, orig_rate, channels):
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if orig_rate == SAMPLE_RATE:
        return (data / 32768.0).astype(np.float32)
    target = int(len(data) * SAMPLE_RATE / orig_rate)
    return (
        np.interp(
            np.linspace(0, len(data) - 1, target), np.arange(len(data)), data.astype(np.float64)
        ).astype(np.float32)
        / 32768.0
    )


def rms_db(samples):
    rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
    return 20 * np.log10(max(rms / 32768.0, 1e-10))


def meter_bar(db):
    n = max(0, min(int((db + 60) / 60 * 20), 20))
    return "\u2588" * n + "\u2591" * (20 - n)


def asr_worker(model, processor, inference_queue, result_queue, stop_event):
    while not stop_event.is_set():
        try:
            audio, is_final = inference_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if audio is None:
            break
        inputs = processor(audio, return_tensors="pt", sampling_rate=SAMPLE_RATE)
        input_features = inputs.input_features.to(model.device, dtype=model.dtype)
        with torch.no_grad():
            generated = model.generate(input_features, temperature=0)
        text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        result_queue.put((text, is_final))


def main():
    device_index = None

    print(f"Loading Whisper large-v3 on {'CUDA' if torch.cuda.is_available() else 'CPU'}...")
    model = transformers.WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.generation_config.forced_decoder_ids = [[1, None], [2, TRANSLATE_TOKEN]]
    processor = transformers.AutoProcessor.from_pretrained("openai/whisper-large-v3")
    print("Model loaded.\n")

    print("Loading Silero VAD...")
    vad_model = silero_vad.load_silero_vad()
    vad = silero_vad.VADIterator(
        model=vad_model,
        threshold=0.5,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=200,
        speech_pad_ms=50,
    )
    print("VAD loaded.\n")

    p = pyaudio.PyAudio()
    if device_index is None:
        dev_idx, dev_info = find_loopback_device(p)
    else:
        dev_idx = device_index
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

    inference_queue = queue.Queue(maxsize=4)
    result_queue = queue.Queue()
    stop_event = threading.Event()
    worker_thread = threading.Thread(
        target=asr_worker, args=(model, processor, inference_queue, result_queue, stop_event), daemon=True
    )
    worker_thread.start()

    # ── State ──────────────────────────────────────────────────────
    raw_buf = deque()                                     # 16 kHz float samples pending VAD
    speech_chunks = []                                     # list of VAD_WINDOW arrays for current utterance
    speech_samples = 0
    speaking = False
    lookbehind = deque(maxlen=2)                          # last 2 VAD windows for speech onset
    last_interim_time = 0.0
    partial_text = ""

    def push_final():
        nonlocal speech_chunks, speech_samples, speaking
        if speech_samples >= MIN_SEGMENT_SAMPLES:
            try:
                inference_queue.put_nowait((np.concatenate(speech_chunks), True))
            except queue.Full:
                pass
        speech_chunks = []
        speech_samples = 0
        speaking = False

    def push_interim():
        nonlocal speech_chunks
        if speech_chunks:
            try:
                inference_queue.put_nowait((np.concatenate(speech_chunks), False))
            except queue.Full:
                pass

    def drain_results():
        nonlocal partial_text
        while True:
            try:
                text, is_final = result_queue.get_nowait()
                if text:
                    if is_final:
                        partial_text = ""
                        sys.stdout.write(f"\r\033[K{text}\n")
                    else:
                        partial_text = text
            except queue.Empty:
                break

    try:
        while True:
            raw = stream.read(1024, exception_on_overflow=False)
            samples = np.frombuffer(raw, dtype=np.int16)
            db = rms_db(samples)

            drain_results()

            if partial_text:
                sys.stdout.write(f"\r\033[90m▸\033[0m {partial_text}  ")
            else:
                bar = meter_bar(db)
                dot = "\033[92m\u25CF\033[0m" if speaking else "\033[90m\u25CB\033[0m"
                sys.stdout.write(f"\r{dot} {bar} {db:+.0f} dB  ")
            sys.stdout.flush()

            mono = to_mono_16k(samples, native_rate, channels)
            raw_buf.extend(mono.tolist())

            while len(raw_buf) >= VAD_WINDOW:
                chunk = np.array([raw_buf.popleft() for _ in range(VAD_WINDOW)], dtype=np.float32)

                event = vad(torch.from_numpy(chunk))

                if event is None:
                    if speaking:
                        speech_chunks.append(chunk)
                        speech_samples += VAD_WINDOW
                        if speech_samples >= MAX_SEGMENT_SAMPLES:
                            push_final()
                elif "start" in event:
                    speaking = True
                    speech_chunks = [*lookbehind, chunk]
                    speech_samples = VAD_WINDOW * len(speech_chunks)
                    last_interim_time = time.monotonic()
                elif "end" in event:
                    if speaking:
                        speech_chunks.append(chunk)
                        speech_samples += VAD_WINDOW
                        push_final()

                lookbehind.append(chunk)

                if speaking and speech_samples >= MIN_SEGMENT_SAMPLES:
                    now = time.monotonic()
                    if now - last_interim_time >= INTERIM_INTERVAL:
                        push_interim()
                        last_interim_time = now

    except KeyboardInterrupt:
        if speaking and speech_samples >= MIN_SEGMENT_SAMPLES:
            try:
                inference_queue.put_nowait((np.concatenate(speech_chunks), True))
            except queue.Full:
                pass
        print("\n\nStopping...")
    finally:
        stop_event.set()
        inference_queue.put((None, False))
        worker_thread.join(timeout=5)
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Done.")


if __name__ == "__main__":
    main()
