import queue
import sys
import threading
import warnings

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
MIN_SECS = 2
MAX_SECS = 10


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


def worker(model, processor, q, stop_event):
    while not stop_event.is_set():
        try:
            segment = q.get(timeout=0.5)
        except queue.Empty:
            continue
        if segment is None:
            break
        inputs = processor(segment, return_tensors="pt", sampling_rate=SAMPLE_RATE)
        input_features = inputs.input_features.to(DEVICE, dtype=model.dtype)
        with torch.no_grad():
            generated = model.generate(
                input_features,
                temperature=0,
            )
        text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        if text:
            sys.stdout.write(f"\r\033[K{text}\n")
            sys.stdout.flush()


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
        min_silence_duration_ms=500,
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
    stop_event = threading.Event()
    t = threading.Thread(
        target=worker, args=(model, processor, q, stop_event), daemon=True
    )
    t.start()

    VAD_WINDOW = 512
    min_samples = SAMPLE_RATE * MIN_SECS
    max_samples = SAMPLE_RATE * MAX_SECS

    vad_buf = np.array([], dtype=np.float32)
    speech_segment = np.array([], dtype=np.float32)
    triggered = False
    segment_samples = 0

    def queue_segment():
        nonlocal speech_segment, triggered, segment_samples
        if len(speech_segment) >= min_samples:
            try:
                q.put_nowait(speech_segment.copy())
                sys.stdout.write("\r\033[K\x1b[90m───\x1b[0m\n")
                sys.stdout.flush()
            except queue.Full:
                pass
        speech_segment = np.array([], dtype=np.float32)
        triggered = False
        segment_samples = 0

    try:
        while True:
            raw = stream.read(1024, exception_on_overflow=False)
            samples = np.frombuffer(raw, dtype=np.int16)
            rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
            sys.stdout.write(f"\r{rms_to_bar(rms)}  ")
            sys.stdout.flush()

            mono = to_mono_16k(samples, native_rate, channels)
            vad_buf = np.concatenate([vad_buf, mono])

            while len(vad_buf) >= VAD_WINDOW:
                chunk = vad_buf[:VAD_WINDOW]
                vad_buf = vad_buf[VAD_WINDOW:]

                chunk_t = torch.from_numpy(chunk)
                result = vad_iterator(chunk_t, return_seconds=False)

                if result is None:
                    if triggered:
                        speech_segment = np.concatenate([speech_segment, chunk])
                        segment_samples += VAD_WINDOW
                        if segment_samples >= max_samples:
                            queue_segment()
                elif "start" in result:
                    triggered = True
                    speech_segment = chunk.copy()
                    segment_samples = VAD_WINDOW
                elif "end" in result:
                    if triggered:
                        speech_segment = np.concatenate([speech_segment, chunk])
                        segment_samples += VAD_WINDOW
                        queue_segment()

    except KeyboardInterrupt:
        if triggered and len(speech_segment) >= min_samples:
            try:
                q.put_nowait(speech_segment.copy())
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
