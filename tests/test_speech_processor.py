"""Test speech processor functionality."""

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
            model_size="base", device="cpu", initial_prompt="Use proper punctuation."
        )

        # Create dummy audio data
        audio_data = np.random.rand(16000)  # 1 second of audio

        # Transcribe
        text, duration, lang, prob = processor.transcribe(audio_data)

        # Verify initial_prompt was passed
        call_args = mock_model.transcribe.call_args[1]
        assert "initial_prompt" in call_args
        assert call_args["initial_prompt"] == "Use proper punctuation."

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
