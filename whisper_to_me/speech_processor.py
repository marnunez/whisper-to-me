"""
Speech Processing Module

Provides speech-to-text transcription using FasterWhisper locally, or by
posting recorded audio to a remote transcription service.
"""

from __future__ import annotations

import io
import json
import re
import urllib.error
import urllib.request
import uuid
from typing import Any

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

from whisper_to_me.config_constants import (
    DEFAULT_OPENAI_TRANSCRIPTION_MODEL,
    DEFAULT_QWEN_ASR_MODEL,
)
from whisper_to_me.logger import get_logger

_LOCAL_BACKEND = "local"
_QWEN_ASR_BACKEND = "qwen-asr"
_SIMPLE_REMOTE_BACKENDS = {"remote", "whisper-asr"}
_OPENAI_BACKEND = "openai"
_OPENAI_COMPATIBLE_BACKENDS = {_OPENAI_BACKEND, _QWEN_ASR_BACKEND}
_QWEN_ASR_SHORT_AUDIO_CONTEXT_THRESHOLD_SECONDS = 1.0
_QWEN_ASR_MAX_CONTEXT_CHARS = 400
_QWEN_ASR_TEXT_PREFIX = re.compile(
    r"^\s*language\s+([^<\r\n]+)<asr_text>",
    re.IGNORECASE,
)


class SpeechProcessor:
    """
    Speech-to-text processor.

    Features:
    - Local FasterWhisper transcription
    - Optional remote/offloaded transcription over HTTP
    - Support for multiple languages and model sizes
    - Optimized parameters for real-time usage
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        language: str | None = None,
        allowed_languages: list[str] | None = None,
        vad_filter: bool = True,
        initial_prompt: str = "",
        task: str = "transcribe",
        beam_size: int = 5,
        best_of: int = 5,
        temperature: float = 0.0,
        condition_on_previous_text: bool = False,
        no_speech_threshold: float = 0.6,
        log_prob_threshold: float = -1.0,
        compression_ratio_threshold: float = 2.4,
        hallucination_silence_threshold: float | None = None,
        hotwords: str = "",
        min_silence_duration_ms: int = 2000,
        speech_pad_ms: int = 400,
        transcription_backend: str = _LOCAL_BACKEND,
        remote_url: str = "",
        remote_model: str = DEFAULT_OPENAI_TRANSCRIPTION_MODEL,
        remote_api_key: str = "",
        remote_timeout: int = 30,
        remote_fallback_to_local: bool = False,
        context_builder: Any = None,
    ):
        """
        Initialize the speech processor.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3)
            device: Processing device (cpu, cuda) for local transcription
            language: Target language for transcription (None for auto-detection, en, es, fr, etc.)
            allowed_languages: Restrict auto-detection to these languages (e.g. ["en", "es"])
            vad_filter: Enable Voice Activity Detection to filter silence locally
            initial_prompt: Initial prompt to guide transcription (max 224 tokens)
            task: Whisper task, either "transcribe" or "translate"
            beam_size: Beam size for beam-search decoding
            best_of: Number of candidates for non-zero-temperature sampling
            temperature: Decoding temperature
            condition_on_previous_text: Condition each segment on previous output
            no_speech_threshold: Whisper no-speech threshold
            log_prob_threshold: Average log-probability failure threshold
            compression_ratio_threshold: Gzip compression-ratio failure threshold
            hallucination_silence_threshold: Optional silence threshold for hallucination suppression
            hotwords: Optional hotwords/context words for faster-whisper
            min_silence_duration_ms: Minimum duration of silence to split segments (in milliseconds)
            speech_pad_ms: Amount of padding to keep around detected speech (in milliseconds)
            transcription_backend: "local", "whisper-asr"/"remote", "qwen-asr", or "openai"
            remote_url: Remote transcription endpoint URL
            remote_model: Model field for OpenAI-compatible endpoints
            remote_api_key: Optional bearer token for remote endpoints
            remote_timeout: HTTP timeout in seconds
            remote_fallback_to_local: Fall back to local FasterWhisper if remote transcription fails
            context_builder: Optional shared context builder for ASR-capable remote backends
        """
        self.model_size = model_size
        self.device = device
        self.language = language
        self.allowed_languages = allowed_languages
        self.vad_filter = vad_filter
        self.initial_prompt = initial_prompt
        self.task = task
        self.beam_size = beam_size
        self.best_of = best_of
        self.temperature = temperature
        self.condition_on_previous_text = condition_on_previous_text
        self.no_speech_threshold = no_speech_threshold
        self.log_prob_threshold = log_prob_threshold
        self.compression_ratio_threshold = compression_ratio_threshold
        self.hallucination_silence_threshold = hallucination_silence_threshold
        self.hotwords = hotwords
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.transcription_backend = transcription_backend
        self.remote_url = remote_url
        self.remote_model = remote_model
        self.remote_api_key = remote_api_key
        self.remote_timeout = remote_timeout
        self.remote_fallback_to_local = remote_fallback_to_local
        self.context_builder = context_builder
        self.model: WhisperModel | None = None
        self.logger = get_logger()

        if self.transcription_backend == _LOCAL_BACKEND:
            self._load_model()
        else:
            self.logger.info(
                f"Using remote transcription backend: {self.transcription_backend}",
                "speech",
            )

        # Log initial prompt if configured
        if self.initial_prompt:
            self.logger.info(
                f"Initial prompt configured: {self.initial_prompt[:50]}...", "speech"
            )
            if self.transcription_backend == _LOCAL_BACKEND:
                self._validate_initial_prompt()

    def _load_model(self) -> None:
        try:
            self.logger.info(
                f"Loading Whisper model: {self.model_size} on {self.device}", "model"
            )
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type="float32" if self.device == "cpu" else "float16",
            )
            self.logger.success("Model loaded successfully", "model")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}", "model")

    def _ensure_local_model(self) -> None:
        """Load the local FasterWhisper model if it is not already loaded."""
        if self.model is None:
            self._load_model()
        if self.model is None:
            raise RuntimeError("Model not loaded")

    def _detect_among(self, audio_data: np.ndarray, allowed: list[str]) -> str | None:
        """Detect language from a restricted set using probability scores."""
        if self.model is None:
            return None
        try:
            _lang, _prob, all_probs = self.model.detect_language(audio_data)
            best_lang = None
            best_prob = -1.0
            for code, prob in all_probs:
                if code in allowed and prob > best_prob:
                    best_lang = code
                    best_prob = prob
            if best_lang:
                self.logger.debug(
                    f"Detected {best_lang} (p={best_prob:.2f}) from {allowed}",
                    "speech",
                )
            return best_lang
        except Exception:
            return None

    def _validate_initial_prompt(self) -> None:
        """Validate initial prompt length for local FasterWhisper usage."""
        if not self.model or not self.initial_prompt:
            return

        try:
            from faster_whisper.tokenizer import Tokenizer

            tokenizer = Tokenizer(
                self.model.hf_tokenizer,
                self.model.model.is_multilingual,
                task="transcribe",
                language=self.language or "en",
            )
            tokens = tokenizer.encode(" " + self.initial_prompt.strip())
            token_count = len(tokens)
            max_prompt_tokens = 224

            if token_count > max_prompt_tokens:
                self.logger.warning(
                    f"Initial prompt has {token_count} tokens, exceeds limit of {max_prompt_tokens}. "
                    f"Only the last {max_prompt_tokens} tokens will be used.",
                    "prompt",
                )
            else:
                self.logger.debug(
                    f"Initial prompt validated: {token_count} tokens", "prompt"
                )
        except Exception as e:
            self.logger.debug(f"Could not validate initial prompt: {e}", "prompt")

    def _check_initial_prompt_truncation(self, detected_language: str) -> None:
        """Check if initial prompt was truncated and warn user."""
        if not self.model or not self.initial_prompt:
            return

        try:
            # Import tokenizer module
            from faster_whisper.tokenizer import Tokenizer

            # Create tokenizer with detected language
            tokenizer = Tokenizer(
                self.model.hf_tokenizer,
                self.model.model.is_multilingual,
                task="transcribe",
                language=detected_language,
            )

            # Encode with leading space (as faster-whisper does internally)
            tokens = tokenizer.encode(" " + self.initial_prompt.strip())
            token_count = len(tokens)

            # Get max_length from the model and calculate threshold
            max_length = self.model.max_length
            threshold = max_length // 2

            # Check if it would be truncated
            if token_count >= threshold:
                self.logger.warning(
                    f"Initial prompt has {token_count} tokens, exceeds limit of {threshold - 1}. "
                    f"Only the first {threshold - 1} tokens are being used.",
                    "prompt",
                )
        except Exception as e:
            # Don't fail if check fails, just log debug
            self.logger.debug(
                f"Could not check initial prompt truncation: {e}", "prompt"
            )

    def transcribe(self, audio_data: np.ndarray) -> tuple[str, float, str, float]:
        if audio_data is None or len(audio_data) == 0:
            return "", 0.0, "", 0.0

        if self.transcription_backend != _LOCAL_BACKEND:
            try:
                return self._transcribe_remote(audio_data)
            except Exception as e:
                if not self.remote_fallback_to_local:
                    self.logger.error(f"Remote transcription failed: {e}", "speech")
                    return "", 0.0, "", 0.0

                self.logger.warning(
                    f"Remote transcription failed, falling back to local model: {e}",
                    "speech",
                )

        return self._transcribe_local(audio_data)

    def _build_transcribe_params(self) -> dict[str, Any]:
        """Build faster-whisper-compatible transcription parameters."""
        params: dict[str, Any] = {
            "task": self.task,
            "beam_size": self.beam_size,
            "best_of": self.best_of,
            "temperature": self.temperature,
            "condition_on_previous_text": self.condition_on_previous_text,
            "no_speech_threshold": self.no_speech_threshold,
            "log_prob_threshold": self.log_prob_threshold,
            "compression_ratio_threshold": self.compression_ratio_threshold,
        }
        if self.hallucination_silence_threshold is not None:
            params["hallucination_silence_threshold"] = (
                self.hallucination_silence_threshold
            )
        if self.hotwords:
            params["hotwords"] = self.hotwords
        return params

    def _add_language_param(
        self, params: dict[str, Any], audio_data: np.ndarray
    ) -> None:
        """Add explicit or restricted auto-detected language to parameters."""
        if self.language is not None:
            params["language"] = self.language
        elif self.allowed_languages:
            detected = self._detect_among(audio_data, self.allowed_languages)
            if detected:
                params["language"] = detected

    def _add_prompt_param(self, params: dict[str, Any]) -> None:
        """Add the initial prompt when configured."""
        if self.initial_prompt:
            params["initial_prompt"] = self.initial_prompt
            self.logger.debug(
                f"Using initial_prompt: {self.initial_prompt[:50]}...", "speech"
            )

    def _add_vad_params(self, params: dict[str, Any]) -> None:
        """Add VAD configuration to parameters."""
        if self.vad_filter:
            params.update(
                {
                    "vad_filter": True,
                    "vad_parameters": {
                        "min_silence_duration_ms": self.min_silence_duration_ms,
                        "speech_pad_ms": self.speech_pad_ms,
                    },
                }
            )
        else:
            params["vad_filter"] = False

    def _transcribe_local(
        self, audio_data: np.ndarray
    ) -> tuple[str, float, str, float]:
        self._ensure_local_model()
        assert self.model is not None

        try:
            transcribe_params = self._build_transcribe_params()
            self._add_language_param(transcribe_params, audio_data)
            self._add_prompt_param(transcribe_params)
            self._add_vad_params(transcribe_params)

            self.logger.debug(f"Transcribe params: {transcribe_params}", "speech")
            segments, info = self.model.transcribe(audio_data, **transcribe_params)

            text_segments = []
            total_duration = 0.0

            for segment in segments:
                text_segments.append(segment.text.strip())
                total_duration = max(total_duration, segment.end)

            full_text = " ".join(text_segments).strip()

            # Check if initial_prompt was truncated after we have the detected language
            if self.initial_prompt and info.language:
                self._check_initial_prompt_truncation(info.language)

            return full_text, total_duration, info.language, info.language_probability

        except Exception as e:
            self.logger.error(f"Error during transcription: {e}", "speech")
            return "", 0.0, "", 0.0

    def _transcribe_remote(
        self, audio_data: np.ndarray
    ) -> tuple[str, float, str, float]:
        """Transcribe audio by posting it to a remote HTTP service."""
        if not self.remote_url:
            raise RuntimeError(
                "transcription.url is required when transcription.backend is remote"
            )

        wav_bytes = self._encode_wav(audio_data)
        url = self._resolve_remote_url()
        audio_duration = len(audio_data) / 16000

        if self.transcription_backend in _SIMPLE_REMOTE_BACKENDS:
            include_nonstandard = True
        elif self.transcription_backend in _OPENAI_COMPATIBLE_BACKENDS:
            include_nonstandard = False
        else:
            raise RuntimeError(
                f"Unknown transcription backend: {self.transcription_backend}"
            )

        fields = self._remote_transcription_fields(
            include_nonstandard=include_nonstandard,
            audio_duration=audio_duration,
        )
        response, response_body = self._post_multipart(url, wav_bytes, fields=fields)

        text = str(response.get("text") or "").strip()
        detected_language = ""
        if self.transcription_backend == _QWEN_ASR_BACKEND:
            text, detected_language = self._normalise_qwen_asr_text(text)
        if not text:
            self.logger.warning(
                "Remote transcription returned empty text. "
                f"Response body: {self._truncate_for_log(response_body)}",
                "speech",
            )
        duration = self._coerce_float(
            response.get("duration_seconds", response.get("duration")),
            default=len(audio_data) / 16000,
        )
        language = str(
            response.get("language") or detected_language or self.language or ""
        )
        confidence = self._coerce_float(
            response.get("language_probability", response.get("confidence")),
            default=0.0,
        )

        return text, duration, language, confidence

    def _remote_transcription_fields(
        self,
        include_nonstandard: bool,
        audio_duration: float | None = None,
    ) -> dict[str, str]:
        """Build multipart form fields for remote transcription services."""
        fields = {
            "model": self._resolve_remote_model(),
            "response_format": "json",
        }
        if self.language is not None:
            fields["language"] = self.language

        asr_context = ""
        if include_nonstandard or self.transcription_backend == _QWEN_ASR_BACKEND:
            asr_context = self._build_asr_context(audio_duration=audio_duration)
            if asr_context:
                self.logger.debug(
                    f"Using ASR context ({len(asr_context)} chars)", "speech"
                )

        if self.transcription_backend == _QWEN_ASR_BACKEND:
            # llama.cpp exposes Qwen3-ASR through the OpenAI transcription schema,
            # which has one prompt field and no separate context field.
            prompt_parts = [part for part in (asr_context, self.initial_prompt) if part]
            if prompt_parts:
                fields["prompt"] = "\n\n".join(prompt_parts)
        elif self.initial_prompt:
            fields["prompt"] = self.initial_prompt

        fields["temperature"] = str(self.temperature)

        if include_nonstandard:
            if asr_context:
                fields["context"] = asr_context

            fields.update(
                {
                    "task": self.task,
                    "beam_size": str(self.beam_size),
                    "best_of": str(self.best_of),
                    "condition_on_previous_text": self._bool_field(
                        self.condition_on_previous_text
                    ),
                    "vad_filter": self._bool_field(self.vad_filter),
                    "min_silence_duration_ms": str(self.min_silence_duration_ms),
                    "speech_pad_ms": str(self.speech_pad_ms),
                    "no_speech_threshold": str(self.no_speech_threshold),
                    "log_prob_threshold": str(self.log_prob_threshold),
                    "compression_ratio_threshold": str(
                        self.compression_ratio_threshold
                    ),
                }
            )
            if self.initial_prompt and self.transcription_backend != _QWEN_ASR_BACKEND:
                fields["initial_prompt"] = self.initial_prompt
            if self.hallucination_silence_threshold is not None:
                fields["hallucination_silence_threshold"] = str(
                    self.hallucination_silence_threshold
                )
            if self.hotwords:
                fields["hotwords"] = self.hotwords
        return fields

    def _resolve_remote_model(self) -> str:
        """Resolve the model name to send to the selected remote backend."""
        configured = (self.remote_model or "").strip()
        if self.transcription_backend == _QWEN_ASR_BACKEND and (
            not configured or configured == DEFAULT_OPENAI_TRANSCRIPTION_MODEL
        ):
            return DEFAULT_QWEN_ASR_MODEL
        return configured or DEFAULT_OPENAI_TRANSCRIPTION_MODEL

    def _build_asr_context(self, audio_duration: float | None = None) -> str:
        """Build ASR context if a shared context builder is configured."""
        if not self.context_builder:
            return ""
        if (
            self.transcription_backend == _QWEN_ASR_BACKEND
            and audio_duration is not None
            and audio_duration < _QWEN_ASR_SHORT_AUDIO_CONTEXT_THRESHOLD_SECONDS
        ):
            self.logger.debug(
                f"Skipping ASR context for short Qwen-ASR clip ({audio_duration:.2f}s)",
                "speech",
            )
            return ""
        try:
            context = str(self.context_builder.build_asr_context() or "").strip()
        except Exception as e:
            self.logger.debug(f"Could not build ASR context: {e}", "speech")
            return ""

        if (
            self.transcription_backend == _QWEN_ASR_BACKEND
            and len(context) > _QWEN_ASR_MAX_CONTEXT_CHARS
        ):
            self.logger.debug(
                "Truncating Qwen-ASR context "
                f"from {len(context)} to {_QWEN_ASR_MAX_CONTEXT_CHARS} chars",
                "speech",
            )
            return context[:_QWEN_ASR_MAX_CONTEXT_CHARS].rstrip()
        return context

    @staticmethod
    def _normalise_qwen_asr_text(text: str) -> tuple[str, str]:
        """Strip llama.cpp's Qwen3-ASR metadata prefix from transcript text."""
        match = _QWEN_ASR_TEXT_PREFIX.match(text)
        if not match:
            return text.strip(), ""
        language = match.group(1).strip()
        return text[match.end() :].strip(), language

    @staticmethod
    def _bool_field(value: bool) -> str:
        return "true" if value else "false"

    def _resolve_remote_url(self) -> str:
        """Resolve the configured remote URL for the selected backend."""
        url = self.remote_url.rstrip("/")
        if self.transcription_backend not in _OPENAI_COMPATIBLE_BACKENDS:
            return self.remote_url

        endpoint = (
            "audio/translations" if self.task == "translate" else "audio/transcriptions"
        )
        if url.endswith("/audio/transcriptions") or url.endswith("/audio/translations"):
            return url
        if url.endswith("/v1"):
            return f"{url}/{endpoint}"
        return f"{url}/v1/{endpoint}"

    @staticmethod
    def _encode_wav(audio_data: np.ndarray) -> bytes:
        """Encode normalized mono 16kHz audio as an in-memory WAV file."""
        buffer = io.BytesIO()
        sf.write(buffer, audio_data.astype(np.float32), 16000, format="WAV")
        return buffer.getvalue()

    def _post_multipart(
        self, url: str, wav_bytes: bytes, fields: dict[str, str]
    ) -> tuple[dict[str, Any], str]:
        """POST multipart/form-data with a WAV file and parse the JSON response."""
        boundary = f"----whisper-to-me-{uuid.uuid4().hex}"
        body = bytearray()

        for name, value in fields.items():
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
            )
            body.extend(str(value).encode())
            body.extend(b"\r\n")

        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
        )
        body.extend(b"Content-Type: audio/wav\r\n\r\n")
        body.extend(wav_bytes)
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode())

        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        }
        if self.remote_api_key:
            headers["Authorization"] = f"Bearer {self.remote_api_key}"

        request = urllib.request.Request(url, data=bytes(body), headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.remote_timeout) as resp:
                response_body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"HTTP {e.code} from transcription service: {error_body}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Could not reach transcription service: {e}") from e

        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Transcription service returned invalid JSON: {response_body[:200]}"
            ) from e

        if not isinstance(parsed, dict):
            raise RuntimeError("Transcription service returned non-object JSON")
        return parsed, response_body

    @staticmethod
    def _truncate_for_log(value: str, limit: int = 2000) -> str:
        """Make potentially large remote responses safe to print in one log line."""
        single_line = value.replace("\n", "\\n")
        if len(single_line) <= limit:
            return single_line
        return f"{single_line[:limit]}… [truncated {len(single_line) - limit} chars]"

    @staticmethod
    def _coerce_float(value: Any, default: float) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def transcribe_with_timestamps(
        self, audio_data: np.ndarray
    ) -> list[dict[str, Any]]:
        if audio_data is None or len(audio_data) == 0:
            return []

        if self.transcription_backend != _LOCAL_BACKEND:
            text, duration, _language, _confidence = self.transcribe(audio_data)
            if not text:
                return []
            return [{"text": text, "start": 0.0, "end": duration, "words": []}]

        self._ensure_local_model()
        assert self.model is not None

        try:
            transcribe_params = self._build_transcribe_params()
            transcribe_params["word_timestamps"] = True
            self._add_language_param(transcribe_params, audio_data)
            self._add_prompt_param(transcribe_params)
            self._add_vad_params(transcribe_params)

            self.logger.debug(f"Transcribe params: {transcribe_params}", "speech")
            segments, info = self.model.transcribe(audio_data, **transcribe_params)

            # Check if initial_prompt was truncated after we have the detected language
            if self.initial_prompt and info.language:
                self._check_initial_prompt_truncation(info.language)

            result = []
            for segment in segments:
                result.append(
                    {
                        "text": segment.text.strip(),
                        "start": segment.start,
                        "end": segment.end,
                        "words": [
                            {
                                "word": word.word,
                                "start": word.start,
                                "end": word.end,
                                "probability": word.probability,
                            }
                            for word in segment.words
                        ]
                        if hasattr(segment, "words") and segment.words
                        else [],
                    }
                )

            return result

        except Exception as e:
            self.logger.error(
                f"Error during transcription with timestamps: {e}", "speech"
            )
            return []

    def set_language(self, language: str) -> None:
        self.language = language
        self.logger.info(f"Language set to: {language}", "language")

    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_size": self.model_size,
            "device": self.device,
            "language": self.language,
            "loaded": self.model is not None
            or self.transcription_backend != _LOCAL_BACKEND,
            "initial_prompt": self.initial_prompt,
            "task": self.task,
            "beam_size": self.beam_size,
            "best_of": self.best_of,
            "temperature": self.temperature,
            "condition_on_previous_text": self.condition_on_previous_text,
            "no_speech_threshold": self.no_speech_threshold,
            "log_prob_threshold": self.log_prob_threshold,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "hallucination_silence_threshold": self.hallucination_silence_threshold,
            "hotwords": self.hotwords,
            "vad_filter": self.vad_filter,
            "min_silence_duration_ms": self.min_silence_duration_ms,
            "speech_pad_ms": self.speech_pad_ms,
            "transcription_backend": self.transcription_backend,
            "remote_url": self.remote_url,
            "remote_model": self.remote_model,
        }
