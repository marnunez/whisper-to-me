"""Test component factory functionality."""

from unittest.mock import Mock, patch

import pytest

from tests.conftest import create_test_audio_devices, create_test_hostapis
from whisper_to_me import (
    ApplicationStateManager,
    AudioDeviceManager,
    AudioRecorder,
    ComponentFactory,
    KeystrokeHandler,
    SpeechProcessor,
    TrayIcon,
)


@pytest.mark.usefixtures("isolated_xdg_runtime_dir")
@patch("whisper_to_me.audio_device_manager.sd")
@patch("whisper_to_me.audio_recorder.sd")
class TestComponentFactory:
    """Test ComponentFactory functionality."""

    @pytest.fixture(autouse=True)
    def setup(self, default_test_config, config_manager_with_temp_file):
        """Set up test environment."""
        self.config = default_test_config
        self.config.recording.audio_device = {
            "name": "Test Device",
            "hostapi_name": "ALSA",
        }

        self.config_manager = config_manager_with_temp_file
        self.factory = ComponentFactory(self.config, self.config_manager)

    def test_init(self, mock_sd_recorder, mock_sd_manager):
        """Test ComponentFactory initialization."""
        assert self.factory.config == self.config
        assert self.factory.config_manager == self.config_manager
        assert isinstance(self.factory.state_manager, ApplicationStateManager)
        assert self.factory.logger is not None

    def test_create_device_manager(self, mock_sd_recorder, mock_sd_manager):
        """Test create_device_manager method."""
        device_manager = self.factory.create_device_manager()

        assert isinstance(device_manager, AudioDeviceManager)
        # Should pass the audio device config
        expected_config = {"name": "Test Device", "hostapi_name": "ALSA"}
        assert device_manager._device_config == expected_config

    def test_create_audio_recorder_success(self, mock_sd_recorder, mock_sd_manager):
        """Test successful audio recorder creation with real components."""
        # Set up mock sounddevice responses
        test_devices = create_test_audio_devices()
        test_hostapis = create_test_hostapis()

        def query_devices_side_effect(device=None):
            if device is None:
                return test_devices
            return test_devices[device] if device < len(test_devices) else None

        mock_sd_manager.query_devices.side_effect = query_devices_side_effect
        mock_sd_manager.query_hostapis.return_value = test_hostapis
        
        # Mock successful InputStream creation
        mock_sd_recorder.InputStream.return_value = Mock()
        mock_sd_recorder.query_devices.side_effect = query_devices_side_effect
        mock_sd_recorder.check_input_settings.return_value = None

        # Create real device manager
        device_manager = self.factory.create_device_manager()

        # Create audio recorder with real AudioRecorder class
        recorder = self.factory.create_audio_recorder(device_manager)

        assert isinstance(recorder, AudioRecorder)
        # Device might be None if the test device isn't found
        assert recorder.device_id is None or isinstance(recorder.device_id, int)

    def test_create_audio_recorder_device_failure_with_fallback(self, mock_sd_recorder, mock_sd_manager):
        """Test audio recorder creation with device failure and successful fallback."""
        # Set up mock sounddevice responses
        test_devices = create_test_audio_devices()
        test_hostapis = create_test_hostapis()

        def query_devices_side_effect(device=None):
            if device is None:
                return test_devices
            return test_devices[device] if device < len(test_devices) else None
        
        mock_sd_manager.query_devices.side_effect = query_devices_side_effect
        mock_sd_manager.query_hostapis.return_value = test_hostapis
        
        mock_sd_recorder.query_devices.side_effect = query_devices_side_effect
        mock_sd_recorder.check_input_settings.return_value = None

        # Make the device initialization fail
        def stream_side_effect(*args, **kwargs):
            if kwargs.get("device") == 0:
                raise Exception("Device initialization failed")
            return Mock()  # Fallback succeeds

        mock_sd_recorder.InputStream.side_effect = stream_side_effect

        # Create real device manager
        device_manager = self.factory.create_device_manager()

        # Create recorder - should fall back to default device
        recorder = self.factory.create_audio_recorder(device_manager)

        assert isinstance(recorder, AudioRecorder)
        assert recorder.device_id is None  # Fallback to default
        assert recorder.device_name is None

        # Device manager should be reset
        assert device_manager._device_config is None
        assert device_manager._current_device is None

    def test_create_audio_recorder_default_device_failure(self, mock_sd_recorder, mock_sd_manager):
        """Test audio recorder creation when even default device fails."""
        # Set up device manager with no specific device
        self.config.recording.audio_device = None
        device_manager = self.factory.create_device_manager()

        # Make all device initialization fail
        mock_sd_recorder.InputStream.side_effect = Exception(
            "No audio devices available"
        )

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Audio recorder initialization failed"):
            self.factory.create_audio_recorder(device_manager)

    def test_create_audio_recorder_with_no_device(self, mock_sd_recorder, mock_sd_manager):
        """Test audio recorder creation with no specific device configured."""
        # No audio device configured
        self.config.recording.audio_device = None
        device_manager = self.factory.create_device_manager()

        # Mock successful stream creation
        mock_sd_recorder.InputStream.return_value = Mock()
        
        # Should create recorder with default device
        recorder = self.factory.create_audio_recorder(device_manager)

        assert isinstance(recorder, AudioRecorder)
        assert recorder.device_id is None
        assert recorder.device_name is None

    @patch("whisper_to_me.speech_processor.WhisperModel")
    def test_create_speech_processor(self, mock_whisper_model, mock_sd_recorder, mock_sd_manager):
        """Test create_speech_processor method."""
        processor = self.factory.create_speech_processor()

        assert isinstance(processor, SpeechProcessor)
        # Whisper model is created during initialization
        mock_whisper_model.assert_called_once_with(
            "tiny",  # model from test config
            device="cpu",
            compute_type="float32",  # CPU uses float32
        )

    def test_create_keystroke_handler(self, mock_sd_recorder, mock_sd_manager):
        """Test create_keystroke_handler method."""
        handler = self.factory.create_keystroke_handler()

        assert isinstance(handler, KeystrokeHandler)
        # KeystrokeHandler doesn't have a trailing_space attribute
        # It's passed as a parameter to type_text method

    @patch("whisper_to_me.component_factory.TrayIcon")
    def test_create_tray_icon(self, mock_tray_class, mock_sd_recorder, mock_sd_manager):
        """Test create_tray_icon method with callbacks."""
        # Create mock callbacks
        on_quit = Mock()
        on_profile_change = Mock()
        on_device_change = Mock()
        get_devices = Mock(return_value=[])
        get_current_device = Mock(return_value=None)

        # Mock tray instance
        mock_tray = Mock(spec=TrayIcon)
        mock_tray_class.return_value = mock_tray

        # Create tray icon
        tray = self.factory.create_tray_icon(
            on_quit=on_quit,
            on_profile_change=on_profile_change,
            on_device_change=on_device_change,
            get_devices=get_devices,
            get_current_device=get_current_device,
        )

        assert tray == mock_tray

        # Verify TrayIcon was created with correct arguments (order doesn't matter)
        mock_tray_class.assert_called_once()
        call_kwargs = mock_tray_class.call_args[1]
        assert call_kwargs["on_quit"] == on_quit
        assert call_kwargs["on_profile_change"] == on_profile_change
        assert call_kwargs["get_profiles"] == self.config_manager.get_profile_names
        assert (
            call_kwargs["get_current_profile"]
            == self.config_manager.get_current_profile
        )
        assert call_kwargs["on_device_change"] == on_device_change
        assert call_kwargs["get_devices"] == get_devices
        assert call_kwargs["get_current_device"] == get_current_device

    @patch("whisper_to_me.speech_processor.WhisperModel")
    def test_recreate_speech_processor_no_change(self, mock_whisper_model, mock_sd_recorder, mock_sd_manager):
        """Test recreate_speech_processor when model hasn't changed."""
        old_config = self.config
        new_config = self.config  # Same config

        result = self.factory.recreate_speech_processor(old_config, new_config)

        assert result is None  # No recreation needed
        mock_whisper_model.assert_not_called()

    @patch("whisper_to_me.speech_processor.WhisperModel")
    def test_recreate_speech_processor_model_change(self, mock_whisper_model, mock_sd_recorder, mock_sd_manager):
        """Test recreate_speech_processor when model has changed."""
        old_config = self.config

        # Create new config with different model
        from copy import deepcopy

        new_config = deepcopy(self.config)
        new_config.general.model = "base"

        processor = self.factory.recreate_speech_processor(old_config, new_config)

        assert isinstance(processor, SpeechProcessor)
        mock_whisper_model.assert_called_once_with(
            "base",
            device="cpu",
            compute_type="float32",  # CPU uses float32
        )

    @patch("whisper_to_me.speech_processor.WhisperModel")
    def test_recreate_speech_processor_device_change(self, mock_whisper_model, mock_sd_recorder, mock_sd_manager):
        """Test recreate_speech_processor when device has changed."""
        old_config = self.config

        # Create new config with different device
        from copy import deepcopy

        new_config = deepcopy(self.config)
        new_config.general.device = "cuda"

        processor = self.factory.recreate_speech_processor(old_config, new_config)

        assert isinstance(processor, SpeechProcessor)
        mock_whisper_model.assert_called_once_with(
            "tiny",
            device="cuda",
            compute_type="float16",  # Should use float16 for CUDA
        )

    @patch("whisper_to_me.speech_processor.WhisperModel")
    def test_recreate_speech_processor_language_change(self, mock_whisper_model, mock_sd_recorder, mock_sd_manager):
        """Test recreate_speech_processor when language has changed."""
        old_config = self.config

        # Create new config with different language
        from copy import deepcopy

        new_config = deepcopy(self.config)
        new_config.general.language = "fr"

        processor = self.factory.recreate_speech_processor(old_config, new_config)

        assert isinstance(processor, SpeechProcessor)
        # Language change requires model recreation
        mock_whisper_model.assert_called_once()

    def test_get_state_manager(self, mock_sd_recorder, mock_sd_manager):
        """Test get_state_manager method."""
        state_manager = self.factory.get_state_manager()

        assert state_manager is self.factory.state_manager
        assert isinstance(state_manager, ApplicationStateManager)
