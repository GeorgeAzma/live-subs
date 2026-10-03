from __future__ import annotations

import copy
import os
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

# Heavy modules (torch, transformers, silero_vad, pyaudio, soxr) are imported lazily
# via _import_heavy() so that importing this module -- and therefore showing
# the overlay UI -- is near-instant.
torch = None
transformers = None
silero_vad = None
pyaudio = None
soxr = None
_heavy_imported = False


def _import_heavy():
    """Import the slow ML/audio modules once, on demand."""
    global torch, transformers, silero_vad, pyaudio, soxr, _heavy_imported
    if _heavy_imported:
        return
    import pyaudiowpatch as _pyaudio
    import silero_vad as _silero_vad
    import soxr as _soxr
    import torch as _torch
    import transformers as _transformers

    pyaudio, silero_vad, soxr, torch, transformers = (
        _pyaudio, _silero_vad, _soxr, _torch, _transformers,
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
    # Speech without a pause is force-cut at this length. Committed words
    # keep long utterances stable on screen, so this can be long: forced
    # cuts are the one place where words can still be split or reworded.
    max_segment_seconds: float = 15.0
    # a forced cut lands at the quietest moment in this final stretch
    cut_search_seconds: float = 1.5
    inference_queue_size: int = 16
    vad_window: int = 512
    # interims only decode the uncommitted tail (~250 ms on a modern GPU),
    # so they can run often; the queue drops interims while one is pending
    interim_interval: float = 0.5
    first_interim_seconds: float = 0.8
    interim_max_new_tokens: int = 96
    # caps runaway repetition loops; an 8 s segment needs well under 100
    final_max_new_tokens: int = 160
    # Whisper generation quality guards.
    fallback_temperatures: tuple = (0.0, 0.4, 0.8)
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    # P(<|nospeech|>) at the start of a decode. Measured on large-v3: speech
    # <= 0.02 (even one 0.4 s word, or under noise/music at 0 dB SNR);
    # silence, music, clicks, hum 0.7-0.8; white noise 0.08-0.4 -- all of
    # which otherwise come out as "Thank you." / "Okay."
    no_speech_threshold: float = 0.2
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
        # loopback of the default output device: where the user hears audio
        try:
            info = p.get_default_wasapi_loopback()
            return info["index"], info
        except (LookupError, OSError):
            pass
        devices = []
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            host = p.get_host_api_info_by_index(info["hostApi"])["name"]
            if "wasapi" in host.lower() and info["maxInputChannels"] > 0 and "loopback" in info["name"].lower():
                devices.append((i, info))
        if not devices:
            # raise, not sys.exit: this runs on the loader thread, where
            # SystemExit would kill the thread silently and hang the overlay
            raise RuntimeError("No WASAPI loopback device found")
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
        # Streaming band-limited resampler: keeps filter state across blocks,
        # so there are no seams at block edges and no aliasing of >8 kHz
        # content into the 16 kHz signal Whisper sees.
        self._resampler = None
        if self.native_rate != Config.sample_rate:
            self._resampler = soxr.ResampleStream(
                self.native_rate, Config.sample_rate, 1, dtype="float32", quality="HQ")
        self._blocks: queue.Queue = queue.Queue()

        def _callback(in_data, frame_count, time_info, status):
            self._blocks.put(in_data)
            return (None, pyaudio.paContinue)

        self._stream = self._p.open(
            format=pyaudio.paInt16, channels=self.channels, rate=self.native_rate,
            input=True, input_device_index=idx, frames_per_buffer=1024,
            stream_callback=_callback,
        )
        return self

    def read(self, timeout: float) -> Optional[np.ndarray]:
        """Next captured block, or None if nothing arrived within `timeout`.
        WASAPI loopback delivers no blocks at all while nothing is playing,
        so a blocking read would stall the whole pipeline on silence."""
        try:
            raw = self._blocks.get(timeout=timeout)
        except queue.Empty:
            return None
        return np.frombuffer(raw, dtype=np.int16)

    def to_mono_16k(self, data: np.ndarray) -> np.ndarray:
        mono = data.reshape(-1, self.channels).mean(axis=1, dtype=np.float32) * np.float32(1 / 32768)
        if self._resampler is None:
            return mono
        return self._resampler.resample_chunk(mono)

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
        self.cut_search = max(1, int(cfg.sample_rate * cfg.cut_search_seconds) // cfg.vad_window)
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
                    # Forced cut: split right after the quietest chunk of the
                    # final stretch (most likely a gap between words); the
                    # rest starts the next segment.
                    lo = max(1, len(self.speech_chunks) - self.cut_search)
                    energy = [float(np.dot(c, c)) for c in self.speech_chunks[lo:]]
                    cut = lo + int(np.argmin(energy)) + 1
                    audio = np.concatenate(self.speech_chunks[:cut])
                    self.speech_chunks = self.speech_chunks[cut:]
                    self.speech_samples = self.window * len(self.speech_chunks)
                    self.last_interim_time = 0.0
                    return {"type": "final", "audio": audio, "cut": True}
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


def _normalize(text: str) -> str:
    return _clean_text(" ".join(text.split()))


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


def _is_hallucination(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _HALLUCINATION_PHRASES)


_SENTENCE_END = ".?!。？！…"


class LocalAgreement:
    """LocalAgreement-2 (Machacek et al., 2023) over the decoder tokens of
    the utterance in flight: a token is committed once two consecutive
    hypotheses agree on it. Committed tokens are forced as the decoder
    prefix of every later decode of the utterance, so they never change on
    screen, and each decode only generates the uncommitted tail."""

    def __init__(self, token_str: Callable[[int], str], decode: Callable[[list], str]):
        self._token_str = token_str
        self._decode = decode
        self.committed: list[int] = []
        self._prev: list[int] = []

    def reset(self):
        self.committed = []
        self._prev = []

    def update(self, hyp: list[int], spaced: bool):
        """Feed the next full hypothesis (committed prefix + new tail)."""
        k = 0
        for a, b in zip(self._prev, hyp):
            if a != b:
                break
            k += 1
        self._prev = hyp
        # Never commit the last token: the hypothesis may have stopped
        # mid-word because the audio did.
        k = min(k, len(hyp) - 1)
        while k > len(self.committed) and not self._boundary(hyp, k, spaced):
            k -= 1
        if k > len(self.committed):
            self.committed = hyp[:k]

    def _boundary(self, hyp: list[int], k: int, spaced: bool) -> bool:
        """May the committed prefix end right before hyp[k]?"""
        head = self._decode(hyp[:k]).rstrip()
        if head.endswith("\ufffd"):
            return False  # mid multi-byte character
        # A prefix ending in a full sentence makes Whisper continue past the
        # end of speech and hallucinate ("Thank you for watching.") instead
        # of stopping, so the end punctuation stays in the tail.
        if head[-1:] in _SENTENCE_END:
            return False
        if not spaced:
            return True  # CJK: any character boundary is a word boundary
        nxt = self._token_str(hyp[k])
        return nxt.startswith("\u0120") or not nxt[:1].isalnum()  # "Ġ" = leading space


ALLOWED_LANGUAGES = ("en", "zh", "ja", "ko", "es")
_LANG_NAMES = {
    "en": "english", "zh": "chinese", "ja": "japanese",
    "ko": "korean", "es": "spanish",
}
_UNSPACED_LANGUAGES = ("zh", "ja")
_LANG_IDS: dict[str, int] = {}


def _probe_start(model, encoder_outputs, no_speech_id: int) -> tuple[str, float]:
    """One decoder step from <|startoftranscript|>: the likeliest allowed
    language, and P(<|nospeech|>) (Whisper's own silence/noise detector)."""
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]], device=model.device)
    logits = model(encoder_outputs=encoder_outputs, decoder_input_ids=decoder_input_ids).logits[0, 0].float()
    lang = max(ALLOWED_LANGUAGES, key=lambda c: logits[_LANG_IDS[c]].item())
    return lang, torch.softmax(logits, dim=-1)[no_speech_id].item()


# a language switch must be confirmed by this many consecutive finals, so one
# misdetected segment (music, noise, a loanword) can't flip the language
LANGUAGE_SWITCH_VOTES = 2


def asr_worker(
    model: transformers.WhisperForConditionalGeneration,
    processor: transformers.WhisperProcessor,
    inference_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    translate_ref: list,
    prompt_ref: list,
):
    """Decode queued utterance audio. Emits ("partial", (text, stable_chars))
    for interims, where text[:stable_chars] is committed and will not change,
    and ("final", text) when an utterance ends."""
    detected_language: Optional[str] = None
    candidate_language: Optional[str] = None
    candidate_votes = 0
    tok = processor.tokenizer
    if not _LANG_IDS:
        _LANG_IDS.update(
            (c, tok.convert_tokens_to_ids(f"<|{c}|>")) for c in ALLOWED_LANGUAGES
        )
    special_ids = set(tok.all_special_ids)
    sot, no_timestamps, no_speech_id = tok.convert_tokens_to_ids(
        ["<|startoftranscript|>", "<|notimestamps|>", "<|nospeech|>"])
    task_ids = {t: tok.convert_tokens_to_ids(f"<|{t}|>") for t in ("transcribe", "translate")}

    def decode(ids):
        return tok.decode(ids, skip_special_tokens=True)

    agreement = LocalAgreement(tok.convert_ids_to_tokens, decode)
    agreement_task = None
    # After a forced prefix the first new token may legitimately be the end
    # of text, which the default config suppresses at that position.
    prefixed_config = copy.deepcopy(model.generation_config)
    prefixed_config.begin_suppress_tokens = None
    with torch.inference_mode():
        while not stop_event.is_set():
            try:
                audio, is_final, cut = inference_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if audio is None:
                break
            try:
                inputs = processor(audio, return_tensors="pt", sampling_rate=Config.sample_rate)
                input_features = inputs.input_features.to(model.device, dtype=model.dtype)
                # run the encoder once; shared by the start probe and generate
                encoder_outputs = model.get_encoder()(input_features)

                task = "translate" if translate_ref[0] else "transcribe"
                if task != agreement_task or cut:
                    # other output language / audio split mid-utterance: the
                    # committed tokens no longer describe this audio
                    agreement.reset()
                    agreement_task = task
                prefix = agreement.committed

                if detected_language is None or is_final or not prefix:
                    lang, no_speech = _probe_start(model, encoder_outputs, no_speech_id)
                    if not prefix and no_speech > Config.no_speech_threshold:
                        # music/noise/silence: Whisper would invent text
                        # ("Thank you."); committed words prove speech, so
                        # only uncommitted audio is checked
                        if is_final:
                            agreement.reset()
                            result_queue.put(("final", ""))
                        continue
                    if detected_language is None or lang == detected_language:
                        detected_language = lang
                        candidate_language, candidate_votes = None, 0
                    elif is_final:
                        if lang == candidate_language:
                            candidate_votes += 1
                        else:
                            candidate_language, candidate_votes = lang, 1
                        if candidate_votes >= LANGUAGE_SWITCH_VOTES:
                            detected_language = lang
                            candidate_language, candidate_votes = None, 0
                prompt = []
                if prompt_ref[0]:
                    prompt = tok.get_prompt_ids(prompt_ref[0][-Config.prompt_max_chars:]).tolist()
                decoder_input_ids = torch.tensor(
                    [prompt + [sot, _LANG_IDS[detected_language], task_ids[task], no_timestamps] + prefix],
                    device=model.device)

                gen_kwargs = dict(
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=decoder_input_ids,
                    temperature=0.0,
                    task=task,
                    language=_LANG_NAMES[detected_language],
                )
                if prefix:
                    gen_kwargs["generation_config"] = prefixed_config
                if not is_final:
                    # interims stay greedy: beams would multiply their latency
                    gen_kwargs.update(do_sample=False, max_new_tokens=Config.interim_max_new_tokens)
                else:
                    gen_kwargs["max_new_tokens"] = Config.final_max_new_tokens
                    if task == "translate":
                        # stop once all beams are finished: after a forced
                        # prefix, beams otherwise keep exploring continuations
                        # up to max_new_tokens (seconds per final)
                        gen_kwargs.update(num_beams=Config.translate_num_beams, early_stopping=True)
                    else:
                        gen_kwargs.update(
                            temperature=Config.fallback_temperatures,
                            compression_ratio_threshold=Config.compression_ratio_threshold,
                            logprob_threshold=Config.logprob_threshold,
                        )
                generated = model.generate(**gen_kwargs)[0].tolist()

                # generate returns only the tokens after decoder_input_ids
                tail = [t for t in generated if t not in special_ids]
                if _is_hallucination(_normalize(decode(tail))):
                    tail = []
                hyp = prefix + tail
                text = _normalize(decode(hyp))
                if _repetitive(text):
                    hyp = prefix
                    text = _normalize(decode(hyp))

                if is_final:
                    agreement.reset()
                    result_queue.put(("final", text))
                    continue
                spaced = task == "translate" or detected_language not in _UNSPACED_LANGUAGES
                agreement.update(hyp, spaced)
                stable = _normalize(decode(agreement.committed))
                result_queue.put(("partial", (text, len(os.path.commonprefix([stable, text])))))
            except Exception as e:
                # never let one bad segment kill the worker: log and continue
                print(f"[asr_worker] segment error: {e}", file=sys.stderr)
                if is_final:
                    agreement.reset()
                    result_queue.put(("final", ""))


class TextHandler:
    # text[:stable] is committed: later updates of this utterance keep it
    def on_partial(self, text: str, stable: int = 0): ...
    # an empty final means the utterance turned out to be non-speech:
    # retract its partial
    def on_final(self, text: str): ...


class PrintHandler(TextHandler):
    def on_partial(self, text: str, stable: int = 0):
        sys.stdout.write(f"\r\033[90m>\033[0m {text[:stable]}\033[90m{text[stable:]}\033[0m  ")
        sys.stdout.flush()

    def on_final(self, text: str):
        sys.stdout.write(f"\r\033[K{text}\n" if text else "\r\033[K")
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
        self._current_partial = None     # (text, stable) shown for the active utterance

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
        threading.Thread(target=self._deliver_results, daemon=True).start()
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
            self._inference_queue.put((None, False, False))
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        if self._audio:
            self._audio.close()

    def _load_models(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            from huggingface_hub import try_to_load_from_cache
            cached = isinstance(try_to_load_from_cache(self.cfg.model_name, "model.safetensors"), str)
        except Exception:
            cached = True
        if cached:
            self._status(f"Loading {self.cfg.model_name} on {device.upper()}...")
        else:
            # the first launch silently downloads ~3 GB; say so, or it looks hung
            self._status(f"Downloading {self.cfg.model_name} (~3 GB, first run only)...")
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

    @property
    def translating(self) -> bool:
        return self._translate_enabled[0]

    def set_translate(self, enabled: bool):
        if enabled != self._translate_enabled[0]:
            self._translate_enabled[0] = enabled
            # the decoder prompt is in the old output language; keeping it
            # would pull the next segments toward that language
            self._prompt_ref[0] = ""

    def toggle_translate(self):
        self.set_translate(not self._translate_enabled[0])

    def _push_inference(self, audio: np.ndarray, is_final: bool, cut: bool = False):
        q = self._inference_queue
        if q is None:
            return
        if not is_final and not q.empty():
            return
        try:
            q.put_nowait((audio, is_final, cut))
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
            q.put_nowait((audio, True, cut))
        except queue.Full:
            pass

    # No audio block for this long means the device went quiet (nothing is
    # playing). Synthesized real-time silence then keeps the VAD running, so
    # the utterance in flight still ends and gets its final.
    SILENCE_GAP_SECONDS = 0.25

    def _pipeline_loop(self):
        audio = self._audio
        vad = self._vad
        output = self._output
        pending = np.zeros(0, dtype=np.float32)
        use_meter = hasattr(output, "on_meter")
        interim_first = int(Config.sample_rate * Config.first_interim_seconds)
        last_block = time.monotonic()

        print(f"Device:  {audio.device_name}")
        print(f"Rate:    {audio.native_rate} Hz | Channels: {audio.channels}")
        print("Listening... Ctrl+C to stop.\n")

        try:
            while self._running:
                samples = audio.read(timeout=0.1)
                now = time.monotonic()
                if samples is None:
                    if now - last_block < self.SILENCE_GAP_SECONDS:
                        continue
                    samples = np.zeros(int(audio.native_rate * (now - last_block)) * audio.channels, dtype=np.int16)
                else:
                    if use_meter and not self._current_partial and not vad.speaking:
                        rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2)) / 32768.0
                        output.on_meter(20 * np.log10(max(rms, 1e-10)), vad.speaking)
                mono = audio.to_mono_16k(samples)
                last_block = now

                pending = np.concatenate((pending, mono))
                n_chunks = len(pending) // vad.window
                for i in range(n_chunks):
                    chunk = pending[i * vad.window:(i + 1) * vad.window]
                    result = vad.process(chunk)

                    if result.get("type") == "final":
                        self._push_inference(result["audio"], True, result.get("cut", False))
                    else:
                        interim = vad.should_push_interim(interim_first, Config.interim_interval)
                        if interim is not None:
                            self._push_inference(interim, False)
                pending = pending[n_chunks * vad.window:]

        except KeyboardInterrupt:
            remaining = vad.snapshot_segment()
            if remaining is not None:
                self._push_inference(remaining, True)
            print("\n\nStopping...")

    def _deliver_results(self):
        """Forward worker results to the output as soon as they exist, on
        its own thread so delivery never waits for the next audio block.
        Message protocol: (kind, payload) with kind in {final, partial};
        every partial is the full authoritative hypothesis so far, as
        (text, stable_chars)."""
        while True:
            kind, payload = self._result_queue.get()
            if kind == "final":
                self._output.on_final(payload)
                if payload:
                    self._prompt_ref[0] = (self._prompt_ref[0] + " " + payload)[-Config.prompt_max_chars:]
                self._current_partial = None
            elif kind == "partial" and payload[0] and payload != self._current_partial:
                self._output.on_partial(*payload)
                self._current_partial = payload

def main():
    translator = LiveTranslator()
    translator.start()
    translator.run()


if __name__ == "__main__":
    main()
