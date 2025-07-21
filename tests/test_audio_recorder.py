"""Test audio recorder functionality."""

import sys
import os
import numpy as np
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from audio_recorder import AudioRecorder


class TestAudioRecorder:
    """Test AudioRecorder functionality."""

    @patch("sounddevice.InputStream")
    @patch("sounddevice.query_devices")
    @patch("sounddevice.check_input_settings")
    def test_init_default_params(self, mock_check, mock_query, mock_stream):
        """Test AudioRecorder initialization with default parameters."""
        # Mock device query
        mock_query.return_value = {"default_samplerate": 44100}
        mock_check.return_value = None  # No exception

        # Mock stream
        mock_stream_instance = Mock()
        mock_stream.return_value = mock_stream_instance

        recorder = AudioRecorder()

        assert recorder.device_id is None
        assert recorder.device_name is None
        assert recorder.channels == 1
        assert recorder.chunk_size == 512
        assert recorder.sample_rate == 16000  # Should prefer 16kHz
        assert recorder.is_recording is False
        assert recorder.audio_data == []
        assert recorder.stream is not None

        # Should start the stream
        mock_stream_instance.start.assert_called_once()

    @patch("sounddevice.InputStream")
    @patch("sounddevice.query_devices")
    @patch("sounddevice.check_input_settings")
    def test_init_with_custom_params(self, mock_check, mock_query, mock_stream):
        """Test AudioRecorder initialization with custom parameters."""
        mock_query.return_value = {"default_samplerate": 48000}
        mock_check.return_value = None
        mock_stream_instance = Mock()
        mock_stream.return_value = mock_stream_instance

        recorder = AudioRecorder(
            sample_rate=48000,
            channels=2,
            chunk_size=1024,
            device_id=3,
            device_name="Test Device",
        )

        assert recorder.device_id == 3
        assert recorder.device_name == "Test Device"
        assert recorder.channels == 2
        assert recorder.chunk_size == 1024
        assert recorder.sample_rate == 48000

    @patch("sounddevice.InputStream")
    @patch("sounddevice.query_devices")
    @patch("sounddevice.check_input_settings")
    def test_get_best_sample_rate_requested_valid(
        self, mock_check, mock_query, mock_stream
    ):
        """Test _get_best_sample_rate with valid requested rate."""
        mock_check.return_value = None  # No exception for requested rate
        mock_stream.return_value = Mock()

        recorder = AudioRecorder(sample_rate=22050)

        assert recorder.sample_rate == 22050
        mock_check.assert_called_with(device=None, samplerate=22050)

    @patch("sounddevice.InputStream")
    @patch("sounddevice.query_devices")
    @patch("sounddevice.check_input_settings")
    @patch("sounddevice.default")
    def test_get_best_sample_rate_fallback_to_16k(
        self, mock_default, mock_check, mock_query, mock_stream
    ):
        """Test _get_best_sample_rate falling back to 16kHz."""

        # Requested rate fails, but 16kHz works
        def check_side_effect(device=None, samplerate=None):
            if samplerate == 22050:  # Requested rate fails
                raise Exception("Rate not supported")
            # 16kHz works
            return None

        mock_check.side_effect = check_side_effect
        mock_default.device = [None, None]
        mock_query.return_value = {"default_samplerate": 44100}
        mock_stream.return_value = Mock()

        recorder = AudioRecorder(sample_rate=22050)

        assert recorder.sample_rate == 16000

    @patch("sounddevice.InputStream")
    @patch("sounddevice.query_devices")
    @patch("sounddevice.check_input_settings")
    @patch("sounddevice.default")
    def test_get_best_sample_rate_fallback_to_device_default(
        self, mock_default, mock_check, mock_query, mock_stream
    ):
        """Test _get_best_sample_rate falling back to device default."""

        # Both requested rate and 16kHz fail, use device default
        def check_side_effect(device=None, samplerate=None):
            if samplerate in [22050, 16000]:
                raise Exception("Rate not supported")
            # Device default works
            return None

        mock_check.side_effect = check_side_effect
        mock_default.device = [None, None]
        mock_query.return_value = {"default_samplerate": 48000}
        mock_stream.return_value = Mock()

        recorder = AudioRecorder(sample_rate=22050)

        assert recorder.sample_rate == 48000

    @patch("sounddevice.InputStream")
    @patch("sounddevice.query_devices")
    @patch("sounddevice.check_input_settings")
    def test_get_best_sample_rate_query_device_error(
        self, mock_check, mock_query, mock_stream
    ):
        """Test _get_best_sample_rate with device query error."""
        mock_check.side_effect = Exception("All rates fail")
        mock_query.side_effect = Exception("Query failed")
        mock_stream.return_value = Mock()

        recorder = AudioRecorder()

        # Should fallback to 16000 when everything fails
        assert recorder.sample_rate == 16000

    @patch("sounddevice.InputStream")
    @patch("sounddevice.query_devices")
    def test_initialize_stream_with_device_name(self, mock_query, mock_stream):
        """Test _initialize_stream with device name."""
        mock_stream_instance = Mock()
        mock_stream.return_value = mock_stream_instance

        # Don't call the real _get_best_sample_rate
        with patch.object(AudioRecorder, "_get_best_sample_rate", return_value=16000):
            AudioRecorder(device_name="Test Device")

        # Should create stream with correct parameters
        mock_stream.assert_called_once()
        call_kwargs = mock_stream.call_args[1]
        assert call_kwargs["samplerate"] == 16000
        assert call_kwargs["channels"] == 1
        assert call_kwargs["blocksize"] == 512
        assert call_kwargs["dtype"] == np.float32
        assert call_kwargs["latency"] == "low"
        assert call_kwargs["device"] is None

        mock_stream_instance.start.assert_called_once()

    @patch("sounddevice.InputStream")
    @patch("sounddevice.query_devices")
    def test_initialize_stream_query_device_info(self, mock_query, mock_stream):
        """Test _initialize_stream querying device info for display."""
        mock_stream_instance = Mock()
        mock_stream.return_value = mock_stream_instance
        mock_query.return_value = {"name": "Queried Device Name"}

        with patch.object(AudioRecorder, "_get_best_sample_rate", return_value=16000):
            AudioRecorder(device_id=5)

        # Should query device info for display
        mock_query.assert_called_with(5)

    def test_audio_callback_not_recording(self):
        """Test _audio_callback when not recording."""
        with (
            patch("sounddevice.InputStream"),
            patch.object(AudioRecorder, "_get_best_sample_rate", return_value=16000),
            patch.object(AudioRecorder, "_initialize_stream"),
        ):
            recorder = AudioRecorder()
            recorder.is_recording = False

            # Create mock audio data
            indata = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)

            recorder._audio_callback(indata, 3, None, None)

            # Should not store audio data when not recording
            assert len(recorder.audio_data) == 0

    def test_audio_callback_recording(self):
        """Test _audio_callback when recording."""
        with (
            patch("sounddevice.InputStream"),
            patch.object(AudioRecorder, "_get_best_sample_rate", return_value=16000),
            patch.object(AudioRecorder, "_initialize_stream"),
        ):
            recorder = AudioRecorder()
            recorder.is_recording = True

            # Create mock audio data
            indata = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)

            recorder._audio_callback(indata, 3, None, None)

            # Should store copy of audio data
            assert len(recorder.audio_data) == 1
            np.testing.assert_array_equal(recorder.audio_data[0], indata)

    def test_audio_callback_with_status(self):
        """Test _audio_callback with status message."""
        with (
            patch("sounddevice.InputStream"),
            patch.object(AudioRecorder, "_get_best_sample_rate", return_value=16000),
            patch.object(AudioRecorder, "_initialize_stream"),
        ):
            recorder = AudioRecorder()

            indata = np.array([[0.1]], dtype=np.float32)
            status = Mock()
            status.__bool__ = Mock(return_value=True)  # Make status truthy

            # Should not raise exception, just log the status
            recorder._audio_callback(indata, 1, None, status)

    def test_start_recording(self):
        """Test start_recording method."""
        with (
            patch("sounddevice.InputStream"),
            patch.object(AudioRecorder, "_get_best_sample_rate", return_value=16000),
            patch.object(AudioRecorder, "_initialize_stream"),
        ):
            recorder = AudioRecorder()
            recorder.is_recording = False
            recorder.audio_data = [np.array([1, 2, 3])]  # Old data

            recorder.start_recording()

            assert recorder.is_recording is True
            assert len(recorder.audio_data) == 0  # Should clear old data

    def test_start_recording_already_recording(self):
        """Test start_recording when already recording."""
        with (
            patch("sounddevice.InputStream"),
            patch.object(AudioRecorder, "_get_best_sample_rate", return_value=16000),
            patch.object(AudioRecorder, "_initialize_stream"),
        ):
            recorder = AudioRecorder()
            recorder.is_recording = True
            original_data = [np.array([1, 2, 3])]
            recorder.audio_data = original_data.copy()

            recorder.start_recording()

            # Should not change state or clear data
            assert recorder.is_recording is True
            assert recorder.audio_data == original_data

    def test_stop_recording_not_recording(self):
        """Test stop_recording when not recording."""
        with (
            patch("sounddevice.InputStream"),
            patch.object(AudioRecorder, "_get_best_sample_rate", return_value=16000),
            patch.object(AudioRecorder, "_initialize_stream"),
        ):
            recorder = AudioRecorder()
            recorder.is_recording = False

            result = recorder.stop_recording()

            assert result is None

    def test_stop_recording_no_data(self):
        """Test stop_recording with no audio data."""
        with (
            patch("sounddevice.InputStream"),
            patch.object(AudioRecorder, "_get_best_sample_rate", return_value=16000),
            patch.object(AudioRecorder, "_initialize_stream"),
        ):
            recorder = AudioRecorder()
            recorder.is_recording = True
            recorder.audio_data = []
            recorder.sample_rate = 16000

            result = recorder.stop_recording()

            assert result is None
            assert recorder.is_recording is False

    def test_stop_recording_with_data(self):
        """Test stop_recording with audio data."""
        with (
            patch("sounddevice.InputStream"),
            patch.object(AudioRecorder, "_get_best_sample_rate", return_value=16000),
            patch.object(AudioRecorder, "_initialize_stream"),
        ):
            recorder = AudioRecorder()
            recorder.is_recording = True
            recorder.sample_rate = 16000

            # Create mock audio chunks
            chunk1 = np.array([[0.1], [0.2]], dtype=np.float32)
            chunk2 = np.array([[0.3], [0.4]], dtype=np.float32)
            recorder.audio_data = [chunk1, chunk2]

            result = recorder.stop_recording()

            assert recorder.is_recording is False
            assert len(recorder.audio_data) == 0  # Should clear data

            # Should concatenate and flatten chunks
            expected = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
            np.testing.assert_array_equal(result, expected)

    def test_get_audio_data_for_whisper_empty(self):
        """Test get_audio_data_for_whisper with empty/None data."""
        with (
            patch("sounddevice.InputStream"),
            patch.object(AudioRecorder, "_get_best_sample_rate", return_value=16000),
            patch.object(AudioRecorder, "_initialize_stream"),
        ):
            recorder = AudioRecorder()

            # Test with None
            result = recorder.get_audio_data_for_whisper(None)
            assert len(result) == 0

            # Test with empty array
            result = recorder.get_audio_data_for_whisper(np.array([]))
            assert len(result) == 0

    def test_get_audio_data_for_whisper_no_resampling(self):
        """Test get_audio_data_for_whisper with 16kHz data (no resampling needed)."""
        with (
            patch("sounddevice.InputStream"),
            patch.object(AudioRecorder, "_get_best_sample_rate", return_value=16000),
            patch.object(AudioRecorder, "_initialize_stream"),
        ):
            recorder = AudioRecorder()
            recorder.sample_rate = 16000

            # Create test audio data
            audio_data = np.array([0.1, 0.2, 0.8, 0.4], dtype=np.float32)

            result = recorder.get_audio_data_for_whisper(audio_data)

            # Should normalize but not resample
            max_val = np.abs(audio_data).max()
            expected = audio_data / max_val
            np.testing.assert_array_almost_equal(result, expected)

    def test_get_audio_data_for_whisper_with_resampling(self):
        """Test get_audio_data_for_whisper with resampling needed."""
        with (
            patch("sounddevice.InputStream"),
            patch.object(AudioRecorder, "_get_best_sample_rate", return_value=44100),
            patch.object(AudioRecorder, "_initialize_stream"),
        ):
            recorder = AudioRecorder()
            recorder.sample_rate = 44100

            # Create test audio data
            audio_data = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

            # Mock resampling method
            resampled_data = np.array([0.1, 0.3], dtype=np.float32)
            with patch.object(
                recorder, "_resample_audio", return_value=resampled_data
            ) as mock_resample:
                result = recorder.get_audio_data_for_whisper(audio_data)

            # Should call resampling
            mock_resample.assert_called_once()
            call_args = mock_resample.call_args[0]
            np.testing.assert_array_equal(call_args[0], audio_data)
            assert call_args[1] == 44100
            assert call_args[2] == 16000

            # Should normalize resampled data
            expected = resampled_data / np.abs(resampled_data).max()
            np.testing.assert_array_almost_equal(result, expected)

    def test_get_audio_data_for_whisper_multichannel(self):
        """Test get_audio_data_for_whisper with multi-channel data."""
        with (
            patch("sounddevice.InputStream"),
            patch.object(AudioRecorder, "_get_best_sample_rate", return_value=16000),
            patch.object(AudioRecorder, "_initialize_stream"),
        ):
            recorder = AudioRecorder()
            recorder.sample_rate = 16000

            # Create 2-channel test audio data
            audio_data = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

            result = recorder.get_audio_data_for_whisper(audio_data)

            # Should average channels and normalize
            expected_mono = np.array([0.15, 0.35])  # Mean of channels
            expected_normalized = expected_mono / np.abs(expected_mono).max()
            np.testing.assert_array_almost_equal(result, expected_normalized)

    def test_resample_audio_same_rate(self):
        """Test _resample_audio with same source and target rate."""
        with (
            patch("sounddevice.InputStream"),
            patch.object(AudioRecorder, "_get_best_sample_rate", return_value=16000),
            patch.object(AudioRecorder, "_initialize_stream"),
        ):
            recorder = AudioRecorder()

            audio = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
            result = recorder._resample_audio(audio, 16000, 16000)

            # Should return original audio unchanged
            np.testing.assert_array_equal(result, audio)

    def test_resample_audio_downsample(self):
        """Test _resample_audio with downsampling."""
        with (
            patch("sounddevice.InputStream"),
            patch.object(AudioRecorder, "_get_best_sample_rate", return_value=16000),
            patch.object(AudioRecorder, "_initialize_stream"),
        ):
            recorder = AudioRecorder()

            # Create audio at 44.1kHz, downsample to 16kHz
            audio = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
            result = recorder._resample_audio(audio, 44100, 16000)

            # Should have fewer samples (downsampled)
            expected_length = int(len(audio) * 16000 / 44100)
            assert len(result) == expected_length
            assert result.dtype == np.float32

    def test_resample_audio_upsample(self):
        """Test _resample_audio with upsampling."""
        with (
            patch("sounddevice.InputStream"),
            patch.object(AudioRecorder, "_get_best_sample_rate", return_value=16000),
            patch.object(AudioRecorder, "_initialize_stream"),
        ):
            recorder = AudioRecorder()

            # Create audio at 8kHz, upsample to 16kHz
            audio = np.array([0.0, 0.2, 0.4], dtype=np.float32)
            result = recorder._resample_audio(audio, 8000, 16000)

            # Should have more samples (upsampled)
            expected_length = int(len(audio) * 16000 / 8000)
            assert len(result) == expected_length
            assert result.dtype == np.float32

    def test_is_recording_active(self):
        """Test is_recording_active method."""
        with (
            patch("sounddevice.InputStream"),
            patch.object(AudioRecorder, "_get_best_sample_rate", return_value=16000),
            patch.object(AudioRecorder, "_initialize_stream"),
        ):
            recorder = AudioRecorder()

            recorder.is_recording = False
            assert recorder.is_recording_active() is False

            recorder.is_recording = True
            assert recorder.is_recording_active() is True

    def test_get_audio_data_for_whisper_zero_max(self):
        """Test get_audio_data_for_whisper with all-zero audio (no normalization)."""
        with (
            patch("sounddevice.InputStream"),
            patch.object(AudioRecorder, "_get_best_sample_rate", return_value=16000),
            patch.object(AudioRecorder, "_initialize_stream"),
        ):
            recorder = AudioRecorder()
            recorder.sample_rate = 16000

            # All-zero audio data
            audio_data = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

            result = recorder.get_audio_data_for_whisper(audio_data)

            # Should not normalize (divide by zero), return as-is
            np.testing.assert_array_equal(result, audio_data)
