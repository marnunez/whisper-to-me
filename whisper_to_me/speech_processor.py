"""
Speech Processing Module

Provides speech-to-text transcription using FasterWhisper with optimized
parameters for accuracy and performance.
"""

from typing import Any

import numpy as np
from faster_whisper import WhisperModel

from whisper_to_me.logger import get_logger


class SpeechProcessor:
    """
    Speech-to-text processor using FasterWhisper.

    Features:
    - High-accuracy transcription with beam search
    - Configurable VAD (Voice Activity Detection)
    - Support for multiple languages and model sizes
    - Optimized parameters for real-time usage
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        language: str | None = None,
        vad_filter: bool = True,
    ):
        """
        Initialize the speech processor.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3)
            device: Processing device (cpu, cuda)
            language: Target language for transcription (None for auto-detection, en, es, fr, etc.)
            vad_filter: Enable Voice Activity Detection to filter silence
        """
        self.model_size = model_size
        self.device = device
        self.language = language
        self.vad_filter = vad_filter
        self.model: WhisperModel | None = None
        self.logger = get_logger()

        self._load_model()

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
            raise

    def transcribe(self, audio_data: np.ndarray) -> tuple[str, float, str, float]:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if audio_data is None or len(audio_data) == 0:
            return "", 0.0, "", 0.0

        try:
            transcribe_params = {
                "beam_size": 5,
                "best_of": 5,
                "temperature": 0.0,
                "condition_on_previous_text": False,
            }

            # Only specify language if set (None enables auto-detection)
            if self.language is not None:
                transcribe_params["language"] = self.language

            if self.vad_filter:
                transcribe_params.update(
                    {
                        "vad_filter": True,
                        "vad_parameters": {
                            "min_silence_duration_ms": 2000,  # Allow longer pauses (2 seconds)
                            "speech_pad_ms": 400,  # Keep more audio around speech
                        },
                    }
                )

            segments, info = self.model.transcribe(audio_data, **transcribe_params)

            text_segments = []
            total_duration = 0.0

            for segment in segments:
                text_segments.append(segment.text.strip())
                total_duration = max(total_duration, segment.end)

            full_text = " ".join(text_segments).strip()

            return full_text, total_duration, info.language, info.language_probability

        except Exception as e:
            self.logger.error(f"Error during transcription: {e}", "speech")
            return "", 0.0, "", 0.0

    def transcribe_with_timestamps(
        self, audio_data: np.ndarray
    ) -> list[dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if audio_data is None or len(audio_data) == 0:
            return []

        try:
            transcribe_params = {
                "beam_size": 5,
                "best_of": 5,
                "temperature": 0.0,
                "word_timestamps": True,
                "condition_on_previous_text": False,
                "vad_filter": True,
            }

            # Only specify language if set (None enables auto-detection)
            if self.language is not None:
                transcribe_params["language"] = self.language

            segments, info = self.model.transcribe(audio_data, **transcribe_params)

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
            "loaded": self.model is not None,
        }
