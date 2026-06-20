import queue
import sys
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pyaudiowpatch as pyaudio
import silero_vad
import torch
import transformers

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()


@dataclass
class Config:
    model_name: str = "openai/whisper-large-v3"
    sample_rate: int = 16000
    vad_threshold: float = 0.5
    min_silence_ms: int = 200
    speech_pad_ms: int = 50
    min_segment_seconds: float = 1.0
    max_segment_seconds: float = 4.0
    interim_interval: float = 1.0
    first_interim_min_seconds: float = 0.8
    inference_queue_size: int = 16
    vad_window: int = 512


class AudioCapture:
    def __init__(self, device_index: Optional[int] = None):
        self.device_index = device_index
        self._p: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self.native_rate: int = 0
        self.channels: int = 0
        self.device_name: str = ""

    @staticmethod
    def _find_loopback(p: pyaudio.PyAudio):
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

    def open(self):
        self._p = pyaudio.PyAudio()
        if self.device_index is None:
            idx, info = self._find_loopback(self._p)
        else:
            idx = self.device_index
            info = self._p.get_device_info_by_index(idx)
        self.native_rate = int(info["defaultSampleRate"])
        self.channels = int(info["maxInputChannels"])
        self.device_name = info["name"]
        self._stream = self._p.open(
            format=pyaudio.paInt16, channels=self.channels, rate=self.native_rate,
            input=True, input_device_index=idx, frames_per_buffer=1024,
        )
        return self

    def read(self) -> np.ndarray:
        raw = self._stream.read(1024, exception_on_overflow=False)
        return np.frombuffer(raw, dtype=np.int16)

    def to_mono_16k(self, data: np.ndarray) -> np.ndarray:
        if self.channels > 1:
            data = data.reshape(-1, self.channels).mean(axis=1)
        sr = Config.sample_rate
        if self.native_rate == sr:
            return (data / 32768.0).astype(np.float32)
        target = int(len(data) * sr / self.native_rate)
        return np.interp(np.linspace(0, len(data) - 1, target), np.arange(len(data)), data.astype(np.float64)).astype(np.float32) / 32768.0

    def close(self):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._p:
            self._p.terminate()


class VoiceDetector:
    def __init__(self, cfg: Config):
        model = silero_vad.load_silero_vad()
        self._vad = silero_vad.VADIterator(
            model=model, threshold=cfg.vad_threshold, sampling_rate=cfg.sample_rate,
            min_silence_duration_ms=cfg.min_silence_ms, speech_pad_ms=cfg.speech_pad_ms,
        )
        self.window = cfg.vad_window
        self.sample_rate = cfg.sample_rate
        self.min_segment = int(cfg.sample_rate * cfg.min_segment_seconds)
        self.max_segment = int(cfg.sample_rate * cfg.max_segment_seconds)
        self.interim_interval = cfg.interim_interval
        self.first_interim_min = int(cfg.sample_rate * cfg.first_interim_min_seconds)
        self.speaking = False
        self.speech_samples = 0
        self.speech_chunks: list[np.ndarray] = []
        self.lookbehind: deque[np.ndarray] = deque(maxlen=2)
        self.last_interim_time = 0.0

    def reset(self):
        self.speaking = False
        self.speech_samples = 0
        self.speech_chunks.clear()
        self.lookbehind.clear()
        self.last_interim_time = 0.0
        self._vad.reset_states()

    def process(self, chunk: np.ndarray) -> dict:
        event = self._vad(torch.from_numpy(chunk))
        self.lookbehind.append(chunk)

        if event is None:
            if self.speaking:
                self.speech_chunks.append(chunk)
                self.speech_samples += self.window
                if self.speech_samples >= self.max_segment:
                    audio = np.concatenate(self.speech_chunks)
                    self.speech_chunks.clear()
                    self.speech_samples = 0
                    return {"type": "final", "audio": audio}
            return {}

        if "start" in event:
            self.speaking = True
            self.speech_chunks = [*self.lookbehind, chunk]
            self.speech_samples = self.window * len(self.speech_chunks)
            self.last_interim_time = 0
            return {}

        if "end" in event:
            if self.speaking:
                self.speech_chunks.append(chunk)
                self.speech_samples += self.window
                result = {"type": "final", "audio": np.concatenate(self.speech_chunks)}
                self._clear_utterance()
                return result
            return {}

        return {}

    def _clear_utterance(self):
        self.speech_chunks.clear()
        self.speech_samples = 0
        self.speaking = False

    def should_push_interim(self) -> Optional[np.ndarray]:
        if not self.speaking or not self.speech_chunks:
            return None
        now = time.monotonic()
        if self.last_interim_time == 0:
            if self.speech_samples < self.first_interim_min:
                return None
            self.last_interim_time = now
            return np.concatenate(self.speech_chunks)
        if now - self.last_interim_time >= self.interim_interval:
            self.last_interim_time = now
            return np.concatenate(self.speech_chunks)
        return None

    def snapshot_segment(self) -> Optional[np.ndarray]:
        if self.speech_samples >= self.min_segment:
            return np.concatenate(self.speech_chunks)
        return None


def _clean_text(text: str) -> str:
    return "".join(c for c in text if c != "\ufffd" and (c.isprintable() or c in "\n\r\t"))


ALLOWED_LANGUAGES = ("en", "zh", "ja", "ko", "ru", "es")
_LANG_NAMES = {
    "en": "english", "zh": "chinese", "ja": "japanese",
    "ko": "korean", "ru": "russian", "es": "spanish",
}
_LANG_IDS: dict[str, int] = {}


def _detect_language(model, input_features, processor) -> str:
    if not _LANG_IDS:
        _LANG_IDS.update(
            (c, processor.tokenizer.convert_tokens_to_ids(f"<|{c}|>"))
            for c in ALLOWED_LANGUAGES
        )
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]], device=model.device)
    with torch.no_grad():
        outputs = model(input_features, decoder_input_ids=decoder_input_ids)
        logits = outputs.logits[:, 0, :]
    best = max(ALLOWED_LANGUAGES, key=lambda c: logits[0, _LANG_IDS[c]].item())
    return _LANG_NAMES[best]


def asr_worker(
    model: transformers.WhisperForConditionalGeneration,
    processor: transformers.WhisperProcessor,
    inference_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    translate_ref: list,
):
    while not stop_event.is_set():
        try:
            audio, is_final = inference_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if audio is None:
            break
        inputs = processor(audio, return_tensors="pt", sampling_rate=Config.sample_rate)
        input_features = inputs.input_features.to(model.device, dtype=model.dtype)
        with torch.no_grad():
            generated = model.generate(
                input_features, temperature=0,
                task="translate" if translate_ref[0] else "transcribe",
                language=_detect_language(model, input_features, processor),
            )
        text = _clean_text(processor.batch_decode(generated, skip_special_tokens=True)[0].strip())
        result_queue.put((text, is_final))


class TextHandler:
    def on_partial(self, text: str): ...
    def on_final(self, text: str): ...


class PrintHandler(TextHandler):
    def on_partial(self, text: str):
        sys.stdout.write(f"\r\033[90m>\033[0m {text}  ")
        sys.stdout.flush()

    def on_final(self, text: str):
        sys.stdout.write(f"\r\033[K{text}\n")
        sys.stdout.flush()

    def on_meter(self, db: float, speaking: bool):
        bar_n = max(0, min(int((db + 60) / 60 * 20), 20))
        bar = "=" * bar_n + "-" * (20 - bar_n)
        dot = "\033[92m*\033[0m" if speaking else "\033[90mo\033[0m"
        sys.stdout.write(f"\r{dot} [{bar}] {db:+.0f} dB  ")
        sys.stdout.flush()


class LiveTranslator:
    def __init__(self, cfg: Optional[Config] = None, output: Optional[TextHandler] = None, device_index: Optional[int] = None):
        self.cfg = cfg or Config()
        self._device_index = device_index
        self._output = output or PrintHandler()
        self._running = False
        self._model: Optional[transformers.WhisperForConditionalGeneration] = None
        self._processor: Optional[transformers.WhisperProcessor] = None
        self._audio: Optional[AudioCapture] = None
        self._vad: Optional[VoiceDetector] = None
        self._inference_queue: Optional[queue.Queue] = None
        self._result_queue: Optional[queue.Queue] = None
        self._stop_event: Optional[threading.Event] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._translate_enabled = [True]

    def set_output(self, handler: TextHandler):
        self._output = handler

    def start(self):
        self._load_models()
        self._audio = AudioCapture(device_index=self._device_index).open()
        self._vad = VoiceDetector(self.cfg)
        self._inference_queue = queue.Queue(maxsize=self.cfg.inference_queue_size)
        self._result_queue = queue.Queue()
        self._stop_event = threading.Event()

        self._worker_thread = threading.Thread(
            target=asr_worker,
            args=(self._model, self._processor, self._inference_queue, self._result_queue, self._stop_event, self._translate_enabled),
            daemon=True,
        )
        self._worker_thread.start()
        self._running = True

    def run(self):
        try:
            self._pipeline_loop()
        finally:
            self.stop()

    def stop(self):
        self._running = False
        if self._stop_event:
            self._stop_event.set()
        if self._inference_queue:
            self._inference_queue.put((None, False))
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        if self._audio:
            self._audio.close()

    def _load_models(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {self.cfg.model_name} on {device.upper()}...")
        self._model = transformers.WhisperForConditionalGeneration.from_pretrained(self.cfg.model_name).to(device)
        self._processor = transformers.AutoProcessor.from_pretrained(self.cfg.model_name)
        print("Model loaded.")
        print("Loading Silero VAD...")
        silero_vad.load_silero_vad()
        print("VAD loaded.\n")

    def toggle_translate(self):
        self._translate_enabled[0] = not self._translate_enabled[0]

    def _push_inference(self, audio: np.ndarray, is_final: bool):
        try:
            self._inference_queue.put_nowait((audio, is_final))
        except queue.Full:
            pass

    def _pipeline_loop(self):
        audio = self._audio
        vad = self._vad
        cfg = self.cfg
        output = self._output
        raw_buf: deque = deque()
        current_partial = ""
        use_meter = hasattr(output, "on_meter")

        print(f"Device:  {audio.device_name}")
        print(f"Rate:    {audio.native_rate} Hz | Channels: {audio.channels}")
        print("Listening... Ctrl+C to stop.\n")

        try:
            while self._running:
                samples = audio.read()
                db = 20 * np.log10(max(np.sqrt(np.mean(samples.astype(np.float64) ** 2)) / 32768.0, 1e-10))

                while True:
                    try:
                        text, is_final = self._result_queue.get_nowait()
                        if is_final:
                            if text:
                                output.on_final(text)
                            current_partial = ""
                        else:
                            if text:
                                output.on_partial(text)
                                current_partial = text
                    except queue.Empty:
                        break

                if current_partial:
                    output.on_partial(current_partial)
                elif use_meter:
                    output.on_meter(db, vad.speaking)

                mono = audio.to_mono_16k(samples)
                raw_buf.extend(mono.tolist())

                while len(raw_buf) >= vad.window:
                    chunk = np.array([raw_buf.popleft() for _ in range(vad.window)], dtype=np.float32)
                    result = vad.process(chunk)

                    if result.get("type") == "final":
                        self._push_inference(result["audio"], True)

                    interim = vad.should_push_interim()
                    if interim is not None:
                        self._push_inference(interim, False)

        except KeyboardInterrupt:
            remaining = vad.snapshot_segment()
            if remaining is not None:
                self._push_inference(remaining, True)
            print("\n\nStopping...")


def main():
    translator = LiveTranslator()
    translator.start()
    translator.run()


if __name__ == "__main__":
    main()
