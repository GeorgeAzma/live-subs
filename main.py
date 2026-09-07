from __future__ import annotations

import queue
import sys
import threading
import time
import warnings
import zlib
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

warnings.filterwarnings("ignore")

# Heavy modules (torch, transformers, silero_vad, pyaudio) are imported lazily
# via _import_heavy() so that importing this module -- and therefore showing
# the overlay UI -- is near-instant.
torch = None
transformers = None
silero_vad = None
pyaudio = None
_heavy_imported = False


def _import_heavy():
    """Import the slow ML/audio modules once, on demand."""
    global torch, transformers, silero_vad, pyaudio, _heavy_imported
    if _heavy_imported:
        return
    import pyaudiowpatch as _pyaudio
    import silero_vad as _silero_vad
    import torch as _torch
    import transformers as _transformers

    pyaudio, silero_vad, torch, transformers = (
        _pyaudio, _silero_vad, _torch, _transformers,
    )
    _transformers.logging.set_verbosity_error()
    # TF32: Ampere+ GPUs run fp32 matmuls ~3x faster with no perceptible
    # accuracy loss for decoding
    _torch.backends.cuda.matmul.allow_tf32 = True
    _torch.backends.cudnn.allow_tf32 = True
    _heavy_imported = True


@dataclass
class Config:
    model_name: str = "openai/whisper-large-v3"
    sample_rate: int = 16000
    vad_threshold: float = 0.5
    min_silence_ms: int = 400
    speech_pad_ms: int = 50
    min_segment_seconds: float = 1.0
    max_segment_seconds: float = 8.0
    inference_queue_size: int = 16
    vad_window: int = 512
    interim_interval: float = 0.8
    first_interim_seconds: float = 0.8
    interim_max_new_tokens: int = 96
    # Whisper generation quality guards.
    fallback_temperatures: tuple = (0.0, 0.4, 0.8)
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    # previous-final text used as decoder prompt (context/language anchor)
    prompt_max_chars: int = 200
    # translation finals: beam search (large accuracy win for zh->en);
    # beam and temperature-fallback don't compose in generate()
    translate_num_beams: int = 5


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
            self.last_interim_time = 0.0
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
        self.last_interim_time = 0.0

    def should_push_interim(self, first_min: int, interval: float) -> Optional[np.ndarray]:
        """Return the current complete utterance when its update is due."""
        if not self.speaking or not self.speech_chunks:
            return None
        now = time.monotonic()
        if self.last_interim_time == 0.0:
            if self.speech_samples < first_min:
                return None
        elif now - self.last_interim_time < interval:
            return None
        self.last_interim_time = now
        return np.concatenate(self.speech_chunks)

    def snapshot_segment(self) -> Optional[np.ndarray]:
        if self.speech_samples >= self.min_segment:
            return np.concatenate(self.speech_chunks)
        return None


def _clean_text(text: str) -> str:
    return "".join(c for c in text if c != "\ufffd" and (c.isprintable() or c in "\n\r\t"))


def _repetitive(text: str) -> bool:
    """Reject only sufficiently long text that compresses like a loop."""
    words = text.split()
    if len(words) < 8:
        return False
    data = text.encode("utf-8")
    return len(data) / max(1, len(zlib.compress(data))) > Config.compression_ratio_threshold


_HALLUCINATION_PHRASES = (
    "subtitles by the amara.org community", "subtitles by amara.org community",
    "subtitles by amara.org", "amara.org community", "amara.org",
    "please subscribe", "thank you for watching", "thanks for watching",
)


def _strip_hallucinations(text: str) -> str:
    lowered = text.lower()
    if any(phrase in lowered for phrase in _HALLUCINATION_PHRASES):
        return ""
    return _clean_text(" ".join(text.split()))




ALLOWED_LANGUAGES = ("en", "zh", "ja", "ko", "es")
_LANG_NAMES = {
    "en": "english", "zh": "chinese", "ja": "japanese",
    "ko": "korean", "es": "spanish",
}
_LANG_IDS: dict[str, int] = {}


def _detect_language(model, input_features, processor) -> str:
    """Return the language code with the highest first-token logit."""
    if not _LANG_IDS:
        _LANG_IDS.update(
            (c, processor.tokenizer.convert_tokens_to_ids(f"<|{c}|>"))
            for c in ALLOWED_LANGUAGES
        )
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]], device=model.device)
    with torch.no_grad():
        logits = model(input_features, decoder_input_ids=decoder_input_ids).logits[:, 0, :]
    return max(ALLOWED_LANGUAGES, key=lambda c: logits[0, _LANG_IDS[c]].item())


def asr_worker(
    model: transformers.WhisperForConditionalGeneration,
    processor: transformers.WhisperProcessor,
    inference_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    translate_ref: list,
    prompt_ref: list,
):
    detected_language: Optional[str] = None
    tok = processor.tokenizer
    if not _LANG_IDS:
        _LANG_IDS.update(
            (c, tok.convert_tokens_to_ids(f"<|{c}|>")) for c in ALLOWED_LANGUAGES
        )
    special_ids = set(tok.all_special_ids)
    with torch.inference_mode():
        while not stop_event.is_set():
            try:
                audio, is_final, prefix_text = inference_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if audio is None:
                break
            try:
                inputs = processor(audio, return_tensors="pt", sampling_rate=Config.sample_rate)
                input_features = inputs.input_features.to(model.device, dtype=model.dtype)
                if detected_language is None:
                    detected_language = _detect_language(model, input_features, processor)
                language = _LANG_NAMES[detected_language]

                prompt_ids = None
                if is_final and prompt_ref[0]:
                    prompt_ids = tok.get_prompt_ids(prompt_ref[0][-Config.prompt_max_chars:], return_tensors="pt").to(model.device)

                gen_kwargs = dict(
                    input_features=input_features,
                    temperature=0.0,
                    task="translate" if translate_ref[0] else "transcribe",
                    language=language,
                )
                if prompt_ids is not None:
                    gen_kwargs["prompt_ids"] = prompt_ids
                if translate_ref[0]:
                    gen_kwargs["num_beams"] = Config.translate_num_beams
                elif is_final:
                    gen_kwargs.update(
                        temperature=Config.fallback_temperatures,
                        compression_ratio_threshold=Config.compression_ratio_threshold,
                        logprob_threshold=Config.logprob_threshold,
                        no_speech_threshold=Config.no_speech_threshold,
                    )

                if not is_final:
                    gen_kwargs.update(
                        do_sample=False,
                        max_new_tokens=Config.interim_max_new_tokens,
                    )
                generated = model.generate(**gen_kwargs)

                token_ids = generated[0].tolist()
                first_text = next((i for i, t in enumerate(token_ids) if t not in special_ids), len(token_ids))
                text_token_ids = [t for t in token_ids[first_text:] if t not in special_ids]
                text = _strip_hallucinations(_clean_text(tok.decode(text_token_ids, skip_special_tokens=True).strip()))

                if not is_final:
                    if _repetitive(text) or not text:
                        result_queue.put(("partial", ""))
                    else:
                        result_queue.put(("partial", text))
                    continue

                result_queue.put(("final", text))
            except Exception as e:
                # never let one bad segment kill the worker: log and continue
                print(f"[asr_worker] segment error: {e}", file=sys.stderr)
                result_queue.put(("final", ""))
            finally:
                result_queue.put(("done", None))


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
        self._status_cb: Optional[Callable[[str], None]] = None
        self._running = False
        self._model: Optional[transformers.WhisperForConditionalGeneration] = None
        self._processor: Optional[transformers.WhisperProcessor] = None
        self._audio: Optional[AudioCapture] = None
        self._vad: Optional[VoiceDetector] = None
        self._inference_queue: Optional[queue.Queue] = None
        self._result_queue: Optional[queue.Queue] = None
        self._stop_event: Optional[threading.Event] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._started = threading.Event()  # set once models/audio are ready
        self._translate_enabled = [True]
        self._prompt_ref = [""]  # last final text, fed back as decoder context
        self._current_partial = ""       # text shown for the active utterance

    def set_output(self, handler: TextHandler):
        self._output = handler

    def set_status_callback(self, cb: Optional[Callable[[str], None]]):
        """Register a callback invoked with progress messages during startup."""
        self._status_cb = cb

    def _status(self, msg: str):
        if self._status_cb:
            try:
                self._status_cb(msg)
            except Exception:
                pass
        print(msg)

    def start(self):
        self._status("Importing libraries (this takes a moment)...")
        _import_heavy()
        self._load_models()
        self._inference_queue = queue.Queue(maxsize=self.cfg.inference_queue_size)
        self._result_queue = queue.Queue()
        self._stop_event = threading.Event()

        self._worker_thread = threading.Thread(
            target=asr_worker,
            args=(self._model, self._processor, self._inference_queue, self._result_queue, self._stop_event, self._translate_enabled, self._prompt_ref),
            daemon=True,
        )
        self._worker_thread.start()
        self._running = True
        self._started.set()
        self._status("Ready.")

    def run(self):
        # Block until start() finishes loading models and audio, so the
        # pipeline loop can be launched in a thread before loading completes.
        self._started.wait()
        try:
            self._pipeline_loop()
        finally:
            self.stop()

    def stop(self):
        self._running = False
        if self._stop_event:
            self._stop_event.set()
        if self._inference_queue:
            self._inference_queue.put((None, False, ""))
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        if self._audio:
            self._audio.close()

    def _load_models(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._status(f"Loading {self.cfg.model_name} on {device.upper()}...")
        kwargs = {
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
        }
        if device == "cuda":
            # Fast check instead of a full `import flash_attn`, which can
            # pull in CUDA extensions and noticeably slow startup.
            from importlib.util import find_spec
            if find_spec("flash_attn") is not None:
                kwargs["attn_implementation"] = "flash_attention_2"
            kwargs["device_map"] = {"": "cuda"}

        def _load_weights():
            self._model = transformers.WhisperForConditionalGeneration.from_pretrained(
                self.cfg.model_name, **kwargs)

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as pool:
            job = pool.submit(_load_weights)
            # tiny tasks, overlapped with the multi-second weight load
            self._processor = transformers.AutoProcessor.from_pretrained(self.cfg.model_name)
            self._status("Loading VAD + opening audio...")
            self._vad = VoiceDetector(self.cfg)
            self._audio = AudioCapture(device_index=self._device_index).open()
            job.result()
        self._status("Model loaded.")

    def toggle_translate(self):
        self._translate_enabled[0] = not self._translate_enabled[0]

    def _push_inference(self, audio: np.ndarray, is_final: bool, prefix_text: str = ""):
        q = self._inference_queue
        if q is None:
            return
        if not is_final and not q.empty():
            return
        try:
            q.put_nowait((audio, is_final, prefix_text))
            return
        except queue.Full:
            pass
        if not is_final:
            return  # never let an interim displace a queued final
        # A final must get through: it supersedes any queued interims, and
        # as a last resort the oldest final.
        kept = []
        evicted = False
        try:
            while True:
                item = q.get_nowait()
                if not evicted and not item[1]:
                    evicted = True
                    continue
                kept.append(item)
        except queue.Empty:
            pass
        if not evicted and kept:
            kept.pop(0)  # queue full of finals: sacrifice the oldest
        for item in kept:
            q.put_nowait(item)
        try:
            q.put_nowait((audio, True, prefix_text))
        except queue.Full:
            pass

    def _pipeline_loop(self):
        audio = self._audio
        vad = self._vad
        cfg = self.cfg
        output = self._output
        raw_buf: deque = deque()
        use_meter = hasattr(output, "on_meter")
        self._current_partial = ""
        interim_first = int(Config.sample_rate * Config.first_interim_seconds)

        print(f"Device:  {audio.device_name}")
        print(f"Rate:    {audio.native_rate} Hz | Channels: {audio.channels}")
        print("Listening... Ctrl+C to stop.\n")

        try:
            while self._running:
                samples = audio.read()
                db = 20 * np.log10(max(np.sqrt(np.mean(samples.astype(np.float64) ** 2)) / 32768.0, 1e-10))

                self._drain_results(output)
                if not self._current_partial and not vad.speaking and use_meter:
                    output.on_meter(db, vad.speaking)

                mono = audio.to_mono_16k(samples)
                raw_buf.extend(mono.tolist())

                while len(raw_buf) >= vad.window:
                    chunk = np.array([raw_buf.popleft() for _ in range(vad.window)], dtype=np.float32)
                    result = vad.process(chunk)

                    if result.get("type") == "final":
                        self._push_inference(result["audio"], True)
                    else:
                        interim = vad.should_push_interim(interim_first, Config.interim_interval)
                        if interim is not None:
                            self._push_inference(interim, False, self._current_partial)

                self._drain_results(output)

        except KeyboardInterrupt:
            remaining = vad.snapshot_segment()
            if remaining is not None:
                self._push_inference(remaining, True)
            print("\n\nStopping...")

    def _drain_results(self, output: TextHandler) -> bool:
        """Deliver every finished result; return True if any was delivered.
        Message protocol: (kind, payload) with kind in
        {final, partial, done}. The worker owns the text: every partial is
        the full authoritative sentence so far, and the overlay's diff makes
        revisions look word-by-word."""
        got_any = False
        while True:
            try:
                kind, payload = self._result_queue.get_nowait()
            except queue.Empty:
                return got_any
            got_any = True
            if kind == "done":
                continue
            elif kind == "final":
                if payload:
                    output.on_final(payload)
                    self._prompt_ref[0] = (self._prompt_ref[0] + " " + payload)[-Config.prompt_max_chars:]
                self._current_partial = ""
            elif kind == "partial":
                if payload != self._current_partial:
                    old_words = self._current_partial.split()
                    new_words = payload.split()
                    is_extension = (
                        len(new_words) >= len(old_words)
                        and new_words[:len(old_words)] == old_words
                    )
                    if is_extension and len(new_words) > len(old_words):
                        # Whisper returns complete hypotheses, not an online
                        # token stream. Publish only the newly completed
                        # words, in order, instead of making the UI jump from
                        # one word straight to a whole sentence.
                        for end in range(len(old_words) + 1, len(new_words) + 1):
                            output.on_partial(" ".join(new_words[:end]))
                    else:
                        # A correction is authoritative and must replace the
                        # old hypothesis once; never append both versions.
                        output.on_partial(payload)
                    self._current_partial = payload


def main():
    translator = LiveTranslator()
    translator.start()
    translator.run()


if __name__ == "__main__":
    main()
