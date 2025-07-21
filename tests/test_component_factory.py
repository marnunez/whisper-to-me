"""Test component factory functionality."""

from unittest.mock import Mock, patch

import pytest

from whisper_to_me import (
    AdvancedConfig,
    AppConfig,
    ApplicationStateManager,
    AudioDevice,
    AudioDeviceManager,
    AudioRecorder,
    ComponentFactory,
    ConfigManager,
    GeneralConfig,
    KeystrokeHandler,
    RecordingConfig,
    SpeechProcessor,
    TrayIcon,
    UIConfig,
)


class TestComponentFactory:
    """Test ComponentFactory functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.config = AppConfig(
            general=GeneralConfig(
                model="base",
                device="cpu",
                language="en",
                debug=False,
                last_profile="default",
                trailing_space=False,
            ),
            recording=RecordingConfig(
                mode="push-to-talk",
                trigger_key="<scroll_lock>",
                discard_key="<esc>",
                audio_device={"name": "Test Device", "hostapi_name": "ALSA"},
            ),
            ui=UIConfig(use_tray=True),
            advanced=AdvancedConfig(sample_rate=16000, chunk_size=512, vad_filter=True),
            profiles={},
        )

        self.config_manager = Mock(spec=ConfigManager)
        self.factory = ComponentFactory(self.config, self.config_manager)

    def test_init(self):
        """Test ComponentFactory initialization."""
        assert self.factory.config == self.config
        assert self.factory.config_manager == self.config_manager
        assert isinstance(self.factory.state_manager, ApplicationStateManager)

    def test_create_device_manager(self):
        """Test create_device_manager method."""
        device_manager = self.factory.create_device_manager()

        assert isinstance(device_manager, AudioDeviceManager)
        # Should pass the audio device config
        expected_config = {"name": "Test Device", "hostapi_name": "ALSA"}
        assert device_manager._device_config == expected_config

    @patch("whisper_to_me.component_factory.AudioRecorder")
    def test_create_audio_recorder_success(self, mock_recorder_class):
        """Test successful audio recorder creation."""
        # Mock device manager
        mock_device = AudioDevice(
            id=3, name="Test Device", hostapi_name="ALSA", channels=2, sample_rate=44100
        )
        mock_device_manager = Mock(spec=AudioDeviceManager)
        mock_device_manager.get_current_device.return_value = mock_device
        mock_device_manager.get_current_device_id.return_value = 3

        # Mock AudioRecorder constructor
        mock_recorder_instance = Mock(spec=AudioRecorder)
        mock_recorder_class.return_value = mock_recorder_instance

        recorder = self.factory.create_audio_recorder(mock_device_manager)

        assert recorder == mock_recorder_instance
        mock_recorder_class.assert_called_once_with(
            device_id=3, device_name="Test Device"
        )

    @patch("whisper_to_me.component_factory.AudioRecorder")
    def test_create_audio_recorder_device_failure_with_fallback(
        self, mock_recorder_class
    ):
        """Test audio recorder creation with device failure and successful fallback."""
        # Mock device manager
        mock_device = AudioDevice(
            id=3, name="Test Device", hostapi_name="ALSA", channels=2, sample_rate=44100
        )
        mock_device_manager = Mock(spec=AudioDeviceManager)
        mock_device_manager.get_current_device.return_value = mock_device
        mock_device_manager.get_current_device_id.return_value = 3

        # Mock available devices for error message
        mock_available_devices = [
            AudioDevice(
                id=1,
                name="Device 1",
                hostapi_name="ALSA",
                channels=1,
                sample_rate=16000,
            ),
            AudioDevice(
                id=2,
                name="Device 2",
                hostapi_name="JACK",
                channels=2,
                sample_rate=48000,
            ),
        ]
        mock_device_manager.list_devices.return_value = mock_available_devices

        # Mock AudioRecorder - first call fails, second (fallback) succeeds
        mock_fallback_recorder = Mock(spec=AudioRecorder)

        def recorder_side_effect(*args, **kwargs):
            if kwargs.get("device_id") == 3:
                raise Exception("Device initialization failed")
            return mock_fallback_recorder

        mock_recorder_class.side_effect = recorder_side_effect

        recorder = self.factory.create_audio_recorder(mock_device_manager)

        assert recorder == mock_fallback_recorder
        # Should have called AudioRecorder twice - once with device, once with None (fallback)
        assert mock_recorder_class.call_count == 2
        mock_recorder_class.assert_any_call(device_id=3, device_name="Test Device")
        mock_recorder_class.assert_any_call(device_id=None)

        # Should have reset device manager config
        assert mock_device_manager._device_config is None
        assert mock_device_manager._current_device is None

    @patch("whisper_to_me.component_factory.AudioRecorder")
    def test_create_audio_recorder_default_device_failure(self, mock_recorder_class):
        """Test audio recorder creation with default device failure."""
        # Mock device manager with no current device
        mock_device_manager = Mock(spec=AudioDeviceManager)
        mock_device_manager.get_current_device.return_value = None
        mock_device_manager.get_current_device_id.return_value = None

        # Mock AudioRecorder to always fail
        mock_recorder_class.side_effect = Exception("All devices failed")

        with pytest.raises(RuntimeError) as exc_info:
            self.factory.create_audio_recorder(mock_device_manager)

        assert "Audio recorder initialization failed" in str(exc_info.value)
        mock_recorder_class.assert_called_once_with(device_id=None, device_name=None)

    @patch("whisper_to_me.component_factory.SpeechProcessor")
    def test_create_speech_processor(self, mock_processor_class):
        """Test create_speech_processor method."""
        mock_processor_instance = Mock(spec=SpeechProcessor)
        mock_processor_class.return_value = mock_processor_instance

        processor = self.factory.create_speech_processor()

        assert processor == mock_processor_instance
        mock_processor_class.assert_called_once_with(
            model_size="base", device="cpu", language="en"
        )

    @patch("whisper_to_me.component_factory.SpeechProcessor")
    def test_create_speech_processor_auto_language(self, mock_processor_class):
        """Test create_speech_processor with auto language detection."""
        # Set language to auto
        self.config.general.language = "auto"

        mock_processor_instance = Mock(spec=SpeechProcessor)
        mock_processor_class.return_value = mock_processor_instance

        processor = self.factory.create_speech_processor()

        assert processor == mock_processor_instance
        mock_processor_class.assert_called_once_with(
            model_size="base",
            device="cpu",
            language=None,  # Should pass None for auto detection
        )

    @patch("whisper_to_me.component_factory.KeystrokeHandler")
    def test_create_keystroke_handler(self, mock_handler_class):
        """Test create_keystroke_handler method."""
        mock_handler_instance = Mock(spec=KeystrokeHandler)
        mock_handler_class.return_value = mock_handler_instance

        handler = self.factory.create_keystroke_handler()

        assert handler == mock_handler_instance
        mock_handler_class.assert_called_once_with()

    @patch("whisper_to_me.component_factory.TrayIcon")
    def test_create_tray_icon_enabled(self, mock_tray_class):
        """Test create_tray_icon when tray is enabled."""
        mock_tray_instance = Mock(spec=TrayIcon)
        mock_tray_class.return_value = mock_tray_instance

        # Mock callback functions
        on_quit = Mock()
        on_profile_change = Mock()
        on_device_change = Mock()
        get_devices = Mock()
        get_current_device = Mock()

        tray = self.factory.create_tray_icon(
            on_quit=on_quit,
            on_profile_change=on_profile_change,
            on_device_change=on_device_change,
            get_devices=get_devices,
            get_current_device=get_current_device,
        )

        assert tray == mock_tray_instance

        # Verify TrayIcon was called with correct parameters
        mock_tray_class.assert_called_once()
        call_kwargs = mock_tray_class.call_args[1]
        assert call_kwargs["on_quit"] == on_quit
        assert call_kwargs["on_profile_change"] == on_profile_change
        assert call_kwargs["on_device_change"] == on_device_change
        assert call_kwargs["get_devices"] == get_devices
        assert call_kwargs["get_current_device"] == get_current_device
        assert call_kwargs["get_profiles"] == self.config_manager.get_profile_names
        assert (
            call_kwargs["get_current_profile"]
            == self.config_manager.get_current_profile
        )

    def test_create_tray_icon_disabled(self):
        """Test create_tray_icon when tray is disabled."""
        # Disable tray
        self.config.ui.use_tray = False

        tray = self.factory.create_tray_icon(
            on_quit=Mock(),
            on_profile_change=Mock(),
            on_device_change=Mock(),
            get_devices=Mock(),
            get_current_device=Mock(),
        )

        assert tray is None

    def test_get_state_manager(self):
        """Test get_state_manager method."""
        state_manager = self.factory.get_state_manager()

        assert isinstance(state_manager, ApplicationStateManager)
        assert state_manager == self.factory.state_manager

    @patch("whisper_to_me.component_factory.SpeechProcessor")
    def test_recreate_speech_processor_language_change(self, mock_processor_class):
        """Test recreate_speech_processor with language change."""
        mock_processor_instance = Mock(spec=SpeechProcessor)
        mock_processor_class.return_value = mock_processor_instance

        # Create old config with different language
        old_config = AppConfig(
            general=GeneralConfig(
                model="base",
                device="cpu",
                language="fr",  # Different language
                debug=False,
                last_profile="default",
                trailing_space=False,
            ),
            recording=RecordingConfig(),
            ui=UIConfig(),
            advanced=AdvancedConfig(),
            profiles={},
        )

        new_processor = self.factory.recreate_speech_processor(old_config, self.config)

        assert new_processor == mock_processor_instance
        assert self.factory.config == self.config  # Should update factory config
        mock_processor_class.assert_called_once_with(
            model_size="base", device="cpu", language="en"
        )

    @patch("whisper_to_me.component_factory.SpeechProcessor")
    def test_recreate_speech_processor_model_change(self, mock_processor_class):
        """Test recreate_speech_processor with model change."""
        mock_processor_instance = Mock(spec=SpeechProcessor)
        mock_processor_class.return_value = mock_processor_instance

        # Create old config with different model
        old_config = AppConfig(
            general=GeneralConfig(
                model="tiny",
                device="cpu",
                language="en",  # Different model
                debug=False,
                last_profile="default",
                trailing_space=False,
            ),
            recording=RecordingConfig(),
            ui=UIConfig(),
            advanced=AdvancedConfig(),
            profiles={},
        )

        new_processor = self.factory.recreate_speech_processor(old_config, self.config)

        assert new_processor == mock_processor_instance

    @patch("whisper_to_me.component_factory.SpeechProcessor")
    def test_recreate_speech_processor_device_change(self, mock_processor_class):
        """Test recreate_speech_processor with device change."""
        mock_processor_instance = Mock(spec=SpeechProcessor)
        mock_processor_class.return_value = mock_processor_instance

        # Create old config with different device
        old_config = AppConfig(
            general=GeneralConfig(
                model="base",
                device="cuda",
                language="en",  # Different device
                debug=False,
                last_profile="default",
                trailing_space=False,
            ),
            recording=RecordingConfig(),
            ui=UIConfig(),
            advanced=AdvancedConfig(),
            profiles={},
        )

        new_processor = self.factory.recreate_speech_processor(old_config, self.config)

        assert new_processor == mock_processor_instance

    def test_recreate_speech_processor_no_change(self):
        """Test recreate_speech_processor with no changes."""
        # Create identical old config
        old_config = AppConfig(
            general=GeneralConfig(
                model="base",
                device="cpu",
                language="en",  # Same as self.config
                debug=False,
                last_profile="default",
                trailing_space=False,
            ),
            recording=RecordingConfig(),
            ui=UIConfig(),
            advanced=AdvancedConfig(),
            profiles={},
        )

        new_processor = self.factory.recreate_speech_processor(old_config, self.config)

        assert new_processor is None  # No recreation needed

    @patch("whisper_to_me.component_factory.SpeechProcessor")
    def test_recreate_speech_processor_multiple_changes(self, mock_processor_class):
        """Test recreate_speech_processor with multiple changes."""
        mock_processor_instance = Mock(spec=SpeechProcessor)
        mock_processor_class.return_value = mock_processor_instance

        # Create old config with multiple differences
        old_config = AppConfig(
            general=GeneralConfig(
                model="tiny",
                device="cuda",
                language="fr",  # All different
                debug=False,
                last_profile="default",
                trailing_space=False,
            ),
            recording=RecordingConfig(),
            ui=UIConfig(),
            advanced=AdvancedConfig(),
            profiles={},
        )

        new_processor = self.factory.recreate_speech_processor(old_config, self.config)

        assert new_processor == mock_processor_instance
        mock_processor_class.assert_called_once_with(
            model_size="base",  # New values
            device="cpu",
            language="en",
        )

    def test_config_update_after_recreation(self):
        """Test that factory config is updated after speech processor recreation."""
        original_config = self.factory.config

        # Create new config
        new_config = AppConfig(
            general=GeneralConfig(
                model="large-v3",
                device="cuda",
                language="es",
                debug=True,
                last_profile="test",
                trailing_space=True,
            ),
            recording=RecordingConfig(mode="tap-mode"),
            ui=UIConfig(use_tray=False),
            advanced=AdvancedConfig(sample_rate=48000),
            profiles={"test": {}},
        )

        with patch("whisper_to_me.component_factory.SpeechProcessor"):
            self.factory.recreate_speech_processor(original_config, new_config)

        # Factory config should be updated
        assert self.factory.config == new_config
        assert self.factory.config != original_config
