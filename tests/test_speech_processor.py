"""Test speech processor functionality."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from whisper_to_me import SpeechProcessor


class TestSpeechProcessor:
    """Test SpeechProcessor functionality."""

    @patch("whisper_to_me.speech_processor.WhisperModel")
    def test_init_default_params(self, mock_whisper_model):
        """Test SpeechProcessor initialization with default parameters."""
        mock_model_instance = Mock()
        mock_whisper_model.return_value = mock_model_instance

        processor = SpeechProcessor()

        assert processor.model_size == "base"
        assert processor.device == "cpu"
        assert processor.language is None
        assert processor.vad_filter is True
        assert processor.model == mock_model_instance

        # Should call WhisperModel with correct parameters
        mock_whisper_model.assert_called_once_with(
            "base", device="cpu", compute_type="float32"
        )

    @patch("whisper_to_me.speech_processor.WhisperModel")
    def test_init_custom_params(self, mock_whisper_model):
        """Test SpeechProcessor initialization with custom parameters."""
        mock_model_instance = Mock()
        mock_whisper_model.return_value = mock_model_instance

        processor = SpeechProcessor(
            model_size="large-v3", device="cuda", language="en", vad_filter=False
        )

        assert processor.model_size == "large-v3"
        assert processor.device == "cuda"
        assert processor.language == "en"
        assert processor.vad_filter is False
        assert processor.model == mock_model_instance

        mock_whisper_model.assert_called_once_with(
            "large-v3",
            device="cuda",
            compute_type="float16",  # Should use float16 for CUDA
        )

    @patch("whisper_to_me.speech_processor.WhisperModel")
    def test_init_model_loading_error(self, mock_whisper_model):
        """Test SpeechProcessor initialization with model loading error."""
        mock_whisper_model.side_effect = Exception("Model loading failed")

        with pytest.raises(Exception) as exc_info:
            SpeechProcessor()

        assert "Model loading failed" in str(exc_info.value)

    @patch("whisper_to_me.speech_processor.WhisperModel")
    def test_load_model_cpu_compute_type(self, mock_whisper_model):
        """Test _load_model with CPU compute type."""
        mock_model_instance = Mock()
        mock_whisper_model.return_value = mock_model_instance

        SpeechProcessor(device="cpu")

        mock_whisper_model.assert_called_once_with(
            "base", device="cpu", compute_type="float32"
        )

    @patch("whisper_to_me.speech_processor.WhisperModel")
    def test_load_model_cuda_compute_type(self, mock_whisper_model):
        """Test _load_model with CUDA compute type."""
        mock_model_instance = Mock()
        mock_whisper_model.return_value = mock_model_instance

        SpeechProcessor(device="cuda")

        mock_whisper_model.assert_called_once_with(
            "base", device="cuda", compute_type="float16"
        )

    @patch("whisper_to_me.speech_processor.WhisperModel")
    def test_transcribe_no_model(self, mock_whisper_model):
        """Test transcribe when model is None."""
        processor = SpeechProcessor.__new__(SpeechProcessor)  # Skip __init__
        processor.model = None

        with pytest.raises(RuntimeError) as exc_info:
            processor.transcribe(np.array([0.1, 0.2, 0.3]))

        assert "Model not loaded" in str(exc_info.value)

    @patch("whisper_to_me.speech_processor.WhisperModel")
    def test_transcribe_empty_audio(self, mock_whisper_model):
        """Test transcribe with empty audio data."""
        mock_model_instance = Mock()
        mock_whisper_model.return_value = mock_model_instance

        processor = SpeechProcessor()

        # Test with None
        result = processor.transcribe(None)
        assert result == ("", 0.0, "", 0.0)

        # Test with empty array
        result = processor.transcribe(np.array([]))
        assert result == ("", 0.0, "", 0.0)

    @patch("whisper_to_me.speech_processor.WhisperModel")
    def test_transcribe_success(self, mock_whisper_model):
        """Test successful transcription."""
        # Mock model and transcription results
        mock_model_instance = Mock()
        mock_whisper_model.return_value = mock_model_instance

        # Mock segments
        mock_segment1 = Mock()
        mock_segment1.text = "  Hello  "
        mock_segment1.end = 2.5

        mock_segment2 = Mock()
        mock_segment2.text = "  world  "
        mock_segment2.end = 4.8

        mock_segments = [mock_segment1, mock_segment2]

        # Mock transcription info
        mock_info = Mock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95

        mock_model_instance.transcribe.return_value = (mock_segments, mock_info)

        processor = SpeechProcessor(language="en")
        audio_data = np.array([0.1, 0.2, 0.3, 0.4])

        result = processor.transcribe(audio_data)

        assert result == ("Hello world", 4.8, "en", 0.95)

        # Verify transcribe was called with correct parameters
        call_args = mock_model_instance.transcribe.call_args
        assert np.array_equal(call_args[0][0], audio_data)
        call_kwargs = call_args[1]
        assert call_kwargs["language"] == "en"
        assert call_kwargs["beam_size"] == 5
        assert call_kwargs["best_of"] == 5
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["condition_on_previous_text"] is False

    @patch("whisper_to_me.speech_processor.WhisperModel")
    def test_transcribe_auto_language(self, mock_whisper_model):
        """Test transcription with auto language detection."""
        mock_model_instance = Mock()
        mock_whisper_model.return_value = mock_model_instance

        mock_segment = Mock()
        mock_segment.text = "Test"
        mock_segment.end = 1.0

        mock_info = Mock()
        mock_info.language = "es"
        mock_info.language_probability = 0.87

        mock_model_instance.transcribe.return_value = ([mock_segment], mock_info)

        processor = SpeechProcessor(language=None)  # Auto detection
        audio_data = np.array([0.1, 0.2])

        result = processor.transcribe(audio_data)

        assert result == ("Test", 1.0, "es", 0.87)

        # Should not pass language parameter for auto detection
        call_kwargs = mock_model_instance.transcribe.call_args[1]
        assert "language" not in call_kwargs

    @patch("whisper_to_me.speech_processor.WhisperModel")
    def test_transcribe_with_vad(self, mock_whisper_model):
        """Test transcription with VAD enabled."""
        mock_model_instance = Mock()
        mock_whisper_model.return_value = mock_model_instance

        mock_segment = Mock()
        mock_segment.text = "Test"
        mock_segment.end = 1.0

        mock_info = Mock()
        mock_info.language = "en"
        mock_info.language_probability = 0.9

        mock_model_instance.transcribe.return_value = ([mock_segment], mock_info)

        processor = SpeechProcessor(vad_filter=True)
        audio_data = np.array([0.1, 0.2])

        processor.transcribe(audio_data)

        # Should include VAD parameters
        call_kwargs = mock_model_instance.transcribe.call_args[1]
        assert call_kwargs["vad_filter"] is True
        assert "vad_parameters" in call_kwargs
        vad_params = call_kwargs["vad_parameters"]
        assert vad_params["min_silence_duration_ms"] == 2000
        assert vad_params["speech_pad_ms"] == 400

    @patch("whisper_to_me.speech_processor.WhisperModel")
    def test_transcribe_without_vad(self, mock_whisper_model):
        """Test transcription with VAD disabled."""
        mock_model_instance = Mock()
        mock_whisper_model.return_value = mock_model_instance

        mock_segment = Mock()
        mock_segment.text = "Test"
        mock_segment.end = 1.0

        mock_info = Mock()
        mock_info.language = "en"
        mock_info.language_probability = 0.9

        mock_model_instance.transcribe.return_value = ([mock_segment], mock_info)

        processor = SpeechProcessor(vad_filter=False)
        audio_data = np.array([0.1, 0.2])

        processor.transcribe(audio_data)

        # Should not include VAD parameters
        call_kwargs = mock_model_instance.transcribe.call_args[1]
        assert "vad_filter" not in call_kwargs
        assert "vad_parameters" not in call_kwargs

    @patch("whisper_to_me.speech_processor.WhisperModel")
    def test_transcribe_error_handling(self, mock_whisper_model):
        """Test transcription error handling."""
        mock_model_instance = Mock()
        mock_model_instance.transcribe.side_effect = Exception("Transcription failed")
        mock_whisper_model.return_value = mock_model_instance

        processor = SpeechProcessor()
        audio_data = np.array([0.1, 0.2])

        result = processor.transcribe(audio_data)

        # Should return empty result on error
        assert result == ("", 0.0, "", 0.0)

    @patch("whisper_to_me.speech_processor.WhisperModel")
    def test_transcribe_with_timestamps_no_model(self, mock_whisper_model):
        """Test transcribe_with_timestamps when model is None."""
        processor = SpeechProcessor.__new__(SpeechProcessor)
        processor.model = None

        with pytest.raises(RuntimeError) as exc_info:
            processor.transcribe_with_timestamps(np.array([0.1, 0.2]))

        assert "Model not loaded" in str(exc_info.value)

    @patch("whisper_to_me.speech_processor.WhisperModel")
    def test_transcribe_with_timestamps_empty_audio(self, mock_whisper_model):
        """Test transcribe_with_timestamps with empty audio."""
        mock_model_instance = Mock()
        mock_whisper_model.return_value = mock_model_instance

        processor = SpeechProcessor()

        # Test with None
        result = processor.transcribe_with_timestamps(None)
        assert result == []

        # Test with empty array
        result = processor.transcribe_with_timestamps(np.array([]))
        assert result == []

    @patch("whisper_to_me.speech_processor.WhisperModel")
    def test_transcribe_with_timestamps_success(self, mock_whisper_model):
        """Test successful transcribe_with_timestamps."""
        mock_model_instance = Mock()
        mock_whisper_model.return_value = mock_model_instance

        # Mock word objects
        mock_word1 = Mock()
        mock_word1.word = "Hello"
        mock_word1.start = 0.0
        mock_word1.end = 0.5
        mock_word1.probability = 0.95

        mock_word2 = Mock()
        mock_word2.word = "world"
        mock_word2.start = 0.6
        mock_word2.end = 1.0
        mock_word2.probability = 0.92

        # Mock segment with words
        mock_segment = Mock()
        mock_segment.text = "Hello world"
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.words = [mock_word1, mock_word2]

        mock_info = Mock()
        mock_model_instance.transcribe.return_value = ([mock_segment], mock_info)

        processor = SpeechProcessor()
        audio_data = np.array([0.1, 0.2])

        result = processor.transcribe_with_timestamps(audio_data)

        expected = [
            {
                "text": "Hello world",
                "start": 0.0,
                "end": 1.0,
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.95},
                    {"word": "world", "start": 0.6, "end": 1.0, "probability": 0.92},
                ],
            }
        ]

        assert result == expected

        # Should include word_timestamps parameter
        call_kwargs = mock_model_instance.transcribe.call_args[1]
        assert call_kwargs["word_timestamps"] is True

    @patch("whisper_to_me.speech_processor.WhisperModel")
    def test_transcribe_with_timestamps_no_words(self, mock_whisper_model):
        """Test transcribe_with_timestamps when segment has no words."""
        mock_model_instance = Mock()
        mock_whisper_model.return_value = mock_model_instance

        mock_segment = Mock()
        mock_segment.text = "Test"
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        # Simulate segment without words attribute or empty words
        del mock_segment.words

        mock_info = Mock()
        mock_model_instance.transcribe.return_value = ([mock_segment], mock_info)

        processor = SpeechProcessor()
        audio_data = np.array([0.1, 0.2])

        result = processor.transcribe_with_timestamps(audio_data)

        expected = [{"text": "Test", "start": 0.0, "end": 1.0, "words": []}]

        assert result == expected

    @patch("whisper_to_me.speech_processor.WhisperModel")
    def test_transcribe_with_timestamps_error(self, mock_whisper_model):
        """Test transcribe_with_timestamps error handling."""
        mock_model_instance = Mock()
        mock_model_instance.transcribe.side_effect = Exception("Timestamp error")
        mock_whisper_model.return_value = mock_model_instance

        processor = SpeechProcessor()
        audio_data = np.array([0.1, 0.2])

        result = processor.transcribe_with_timestamps(audio_data)

        assert result == []

    @patch("whisper_to_me.speech_processor.WhisperModel")
    def test_set_language(self, mock_whisper_model):
        """Test set_language method."""
        mock_model_instance = Mock()
        mock_whisper_model.return_value = mock_model_instance

        processor = SpeechProcessor(language="en")
        assert processor.language == "en"

        processor.set_language("fr")
        assert processor.language == "fr"

    @patch("whisper_to_me.speech_processor.WhisperModel")
    def test_get_model_info(self, mock_whisper_model):
        """Test get_model_info method."""
        mock_model_instance = Mock()
        mock_whisper_model.return_value = mock_model_instance

        processor = SpeechProcessor(model_size="large-v3", device="cuda", language="es")

        info = processor.get_model_info()

        expected = {
            "model_size": "large-v3",
            "device": "cuda",
            "language": "es",
            "loaded": True,
        }

        assert info == expected

    @patch("whisper_to_me.speech_processor.WhisperModel")
    def test_get_model_info_not_loaded(self, mock_whisper_model):
        """Test get_model_info when model is not loaded."""
        processor = SpeechProcessor.__new__(SpeechProcessor)
        processor.model_size = "base"
        processor.device = "cpu"
        processor.language = None
        processor.model = None

        info = processor.get_model_info()

        expected = {
            "model_size": "base",
            "device": "cpu",
            "language": None,
            "loaded": False,
        }

        assert info == expected

    @patch("whisper_to_me.speech_processor.WhisperModel")
    def test_multiple_transcriptions(self, mock_whisper_model):
        """Test multiple transcription calls."""
        mock_model_instance = Mock()
        mock_whisper_model.return_value = mock_model_instance

        # Mock different results for each call
        mock_segment1 = Mock()
        mock_segment1.text = "First"
        mock_segment1.end = 1.0

        mock_segment2 = Mock()
        mock_segment2.text = "Second"
        mock_segment2.end = 1.5

        mock_info = Mock()
        mock_info.language = "en"
        mock_info.language_probability = 0.9

        mock_model_instance.transcribe.side_effect = [
            ([mock_segment1], mock_info),
            ([mock_segment2], mock_info),
        ]

        processor = SpeechProcessor()

        result1 = processor.transcribe(np.array([0.1, 0.2]))
        result2 = processor.transcribe(np.array([0.3, 0.4]))

        assert result1 == ("First", 1.0, "en", 0.9)
        assert result2 == ("Second", 1.5, "en", 0.9)
        assert mock_model_instance.transcribe.call_count == 2
