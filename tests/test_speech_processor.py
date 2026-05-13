"""Test speech processor functionality."""

import json
from unittest.mock import MagicMock, patch

import numpy as np

from whisper_to_me.speech_processor import SpeechProcessor


class TestSpeechProcessor:
    """Test cases for speech processor."""

    @patch("whisper_to_me.speech_processor.WhisperModel")
    def test_initialization_with_initial_prompt(self, mock_whisper_model):
        """Test SpeechProcessor initialization with initial_prompt."""
        # Mock the model
        mock_model = MagicMock()
        mock_whisper_model.return_value = mock_model

        # Initialize with initial_prompt
        processor = SpeechProcessor(
            model_size="base",
            device="cpu",
            language="en",
            vad_filter=True,
            initial_prompt="Test prompt for transcription",
        )

        assert processor.initial_prompt == "Test prompt for transcription"
        assert processor.model_size == "base"
        assert processor.device == "cpu"
        assert processor.language == "en"
        assert processor.vad_filter is True

    @patch("whisper_to_me.speech_processor.WhisperModel")
    @patch("whisper_to_me.speech_processor.get_logger")
    def test_initial_prompt_token_validation_warning(
        self, mock_logger, mock_whisper_model
    ):
        """Test that warning is logged for prompts exceeding 224 tokens."""
        # Mock the model and tokenizer
        mock_model = MagicMock()
        mock_whisper_model.return_value = mock_model

        # Mock tokenizer to return more than 224 tokens
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(250))  # 250 tokens

        # Mock the model attributes
        mock_model.hf_tokenizer = MagicMock()
        mock_model.model.is_multilingual = True

        # Mock logger
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance

        # Patch the Tokenizer import from faster_whisper
        with patch("faster_whisper.tokenizer.Tokenizer", return_value=mock_tokenizer):
            _ = SpeechProcessor(
                model_size="base",
                device="cpu",
                initial_prompt="A very long prompt that would exceed 224 tokens...",
            )

            # Check that warning was logged
            mock_logger_instance.warning.assert_called_with(
                "Initial prompt has 250 tokens, exceeds limit of 224. Only the last 224 tokens will be used.",
                "prompt",
            )

    @patch("whisper_to_me.speech_processor.WhisperModel")
    @patch("whisper_to_me.speech_processor.get_logger")
    def test_initial_prompt_token_validation_valid(
        self, mock_logger, mock_whisper_model
    ):
        """Test that no warning is logged for prompts within 224 tokens."""
        # Mock the model and tokenizer
        mock_model = MagicMock()
        mock_whisper_model.return_value = mock_model

        # Mock tokenizer to return less than 224 tokens
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(100))  # 100 tokens

        # Mock the model attributes
        mock_model.hf_tokenizer = MagicMock()
        mock_model.model.is_multilingual = True

        # Mock logger
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance

        # Patch the Tokenizer import from faster_whisper
        with patch("faster_whisper.tokenizer.Tokenizer", return_value=mock_tokenizer):
            _ = SpeechProcessor(
                model_size="base", device="cpu", initial_prompt="A normal prompt"
            )

            # Check that debug message was logged, not warning
            mock_logger_instance.debug.assert_called_with(
                "Initial prompt validated: 100 tokens", "prompt"
            )

    @patch("whisper_to_me.speech_processor.WhisperModel")
    def test_transcribe_with_initial_prompt(self, mock_whisper_model):
        """Test that initial_prompt is passed to transcribe method."""
        # Mock the model
        mock_model = MagicMock()
        mock_whisper_model.return_value = mock_model

        # Mock transcribe return values
        mock_segment = MagicMock()
        mock_segment.text = "Transcribed text"
        mock_segment.end = 5.0

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.99

        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        # Initialize with initial_prompt
        processor = SpeechProcessor(
            model_size="base",
            device="cpu",
            initial_prompt="Use proper punctuation.",
            task="translate",
            beam_size=3,
            best_of=2,
            temperature=0.2,
            condition_on_previous_text=True,
            no_speech_threshold=0.4,
            log_prob_threshold=-1.5,
            compression_ratio_threshold=2.0,
            hotwords="Whisper, CTranslate2",
        )

        # Create dummy audio data
        audio_data = np.random.rand(16000)  # 1 second of audio

        # Transcribe
        text, duration, lang, prob = processor.transcribe(audio_data)

        # Verify initial_prompt was passed
        call_args = mock_model.transcribe.call_args[1]
        assert "initial_prompt" in call_args
        assert call_args["initial_prompt"] == "Use proper punctuation."
        assert call_args["task"] == "translate"
        assert call_args["beam_size"] == 3
        assert call_args["best_of"] == 2
        assert call_args["temperature"] == 0.2
        assert call_args["condition_on_previous_text"] is True
        assert call_args["no_speech_threshold"] == 0.4
        assert call_args["log_prob_threshold"] == -1.5
        assert call_args["compression_ratio_threshold"] == 2.0
        assert call_args["hotwords"] == "Whisper, CTranslate2"

        # Verify results
        assert text == "Transcribed text"
        assert duration == 5.0
        assert lang == "en"
        assert prob == 0.99

    @patch("whisper_to_me.speech_processor.WhisperModel")
    def test_get_model_info_includes_initial_prompt(self, mock_whisper_model):
        """Test that get_model_info includes initial_prompt."""
        # Mock the model
        mock_model = MagicMock()
        mock_whisper_model.return_value = mock_model

        # Initialize with initial_prompt
        processor = SpeechProcessor(
            model_size="base",
            device="cpu",
            language="en",
            initial_prompt="Medical transcription mode",
        )

        # Get model info
        info = processor.get_model_info()

        assert info["initial_prompt"] == "Medical transcription mode"
        assert info["model_size"] == "base"
        assert info["device"] == "cpu"
        assert info["language"] == "en"
        assert info["loaded"] is True

    @patch("whisper_to_me.speech_processor.WhisperModel")
    def test_empty_initial_prompt(self, mock_whisper_model):
        """Test that empty initial_prompt doesn't break functionality."""
        # Mock the model
        mock_model = MagicMock()
        mock_whisper_model.return_value = mock_model

        # Initialize with empty initial_prompt
        processor = SpeechProcessor(model_size="base", device="cpu", initial_prompt="")

        assert processor.initial_prompt == ""

        # Mock transcribe
        mock_segment = MagicMock()
        mock_segment.text = "Test"
        mock_segment.end = 1.0

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.99

        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        # Create dummy audio data
        audio_data = np.random.rand(16000)

        # Transcribe - should not include initial_prompt in call
        text, _, _, _ = processor.transcribe(audio_data)

        # Verify initial_prompt was NOT passed when empty
        call_args = mock_model.transcribe.call_args[1]
        assert "initial_prompt" not in call_args

        assert text == "Test"

    @patch("whisper_to_me.speech_processor.WhisperModel")
    @patch("whisper_to_me.speech_processor.urllib.request.urlopen")
    def test_remote_whisper_asr_transcription(self, mock_urlopen, mock_whisper_model):
        """Test simple remote whisper-asr transcription backend."""

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "text": "Remote text",
                        "duration_seconds": 2.5,
                        "language": "en",
                    }
                ).encode()

        mock_urlopen.return_value = FakeResponse()

        processor = SpeechProcessor(
            transcription_backend="whisper-asr",
            remote_url="http://asr.example:8080/transcribe",
        )
        audio_data = np.zeros(16000, dtype=np.float32)

        text, duration, language, confidence = processor.transcribe(audio_data)

        assert text == "Remote text"
        assert duration == 2.5
        assert language == "en"
        assert confidence == 0.0
        mock_whisper_model.assert_not_called()

        request = mock_urlopen.call_args.args[0]
        assert request.full_url == "http://asr.example:8080/transcribe"
        assert b'name="file"; filename="audio.wav"' in request.data
        assert b'name="task"' in request.data
        assert b'transcribe' in request.data
        assert b'name="beam_size"' in request.data
        assert b'name="vad_filter"' in request.data

    @patch("whisper_to_me.speech_processor.urllib.request.urlopen")
    def test_openai_remote_url_resolution_and_fields(self, mock_urlopen):
        """Test OpenAI-compatible transcription endpoint shape."""

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"text":"OpenAI text"}'

        mock_urlopen.return_value = FakeResponse()

        processor = SpeechProcessor(
            transcription_backend="openai",
            remote_url="https://example.test",
            remote_model="whisper-1",
            remote_api_key="secret",
            language="en",
        )
        text, _duration, language, _confidence = processor.transcribe(
            np.zeros(16000, dtype=np.float32)
        )

        assert text == "OpenAI text"
        assert language == "en"
        request = mock_urlopen.call_args.args[0]
        assert request.full_url == "https://example.test/v1/audio/transcriptions"
        assert request.get_header("Authorization") == "Bearer secret"
        assert b'name="model"' in request.data
        assert b"whisper-1" in request.data
        assert b'name="language"' in request.data
        assert b'name="temperature"' in request.data
        assert b'name="beam_size"' not in request.data

    def test_openai_remote_translation_url_resolution(self):
        """OpenAI-compatible translate task uses the translations endpoint."""
        processor = SpeechProcessor(
            transcription_backend="openai",
            remote_url="https://api.openai.com/v1",
            task="translate",
        )

        assert (
            processor._resolve_remote_url()
            == "https://api.openai.com/v1/audio/translations"
        )

    def test_openai_remote_url_resolution_accepts_v1_base(self):
        """OpenAI-compatible backend accepts either server root or /v1 base URL."""
        processor = SpeechProcessor(
            transcription_backend="openai",
            remote_url="https://api.openai.com/v1",
        )

        assert (
            processor._resolve_remote_url()
            == "https://api.openai.com/v1/audio/transcriptions"
        )

    def test_openai_remote_url_resolution_accepts_full_endpoint(self):
        """OpenAI-compatible backend leaves full transcription endpoints unchanged."""
        processor = SpeechProcessor(
            transcription_backend="openai",
            remote_url="https://api.openai.com/v1/audio/transcriptions",
        )

        assert (
            processor._resolve_remote_url()
            == "https://api.openai.com/v1/audio/transcriptions"
        )
