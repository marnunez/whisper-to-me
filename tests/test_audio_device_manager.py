"""Test audio device manager functionality."""

from unittest.mock import patch

import pytest

from whisper_to_me import (
    AudioConfigurationError,
    AudioDevice,
    AudioDeviceInitializationError,
    AudioDeviceManager,
    AudioDeviceTestError,
    NoAudioDevicesError,
)


class TestAudioDevice:
    """Test AudioDevice data class."""

    def test_audio_device_creation(self):
        """Test AudioDevice creation with all parameters."""
        device = AudioDevice(
            id=1,
            name="Test Device",
            hostapi_name="ALSA",
            channels=2,
            sample_rate=44100.0,
        )

        assert device.id == 1
        assert device.name == "Test Device"
        assert device.hostapi_name == "ALSA"
        assert device.channels == 2
        assert device.sample_rate == 44100.0

    def test_to_config(self):
        """Test AudioDevice to_config method."""
        device = AudioDevice(
            id=1,
            name="Test Device",
            hostapi_name="ALSA",
            channels=2,
            sample_rate=44100.0,
        )

        config = device.to_config()
        expected = {"name": "Test Device", "hostapi_name": "ALSA"}
        assert config == expected

    def test_from_config(self):
        """Test AudioDevice from_config class method."""
        config = {"name": "Test Device", "hostapi_name": "ALSA"}
        device_info = {
            "id": 1,
            "name": "Test Device",
            "hostapi_name": "ALSA",
            "channels": 2,
            "default_samplerate": 44100.0,
        }

        device = AudioDevice.from_config(config, device_info)

        assert device.id == 1
        assert device.name == "Test Device"
        assert device.hostapi_name == "ALSA"
        assert device.channels == 2
        assert device.sample_rate == 44100.0

    def test_matches_config_exact_match(self):
        """Test AudioDevice matches_config with exact match."""
        device = AudioDevice(
            id=1,
            name="Test Device",
            hostapi_name="ALSA",
            channels=2,
            sample_rate=44100.0,
        )

        config = {"name": "Test Device", "hostapi_name": "ALSA"}
        assert device.matches_config(config) is True

    def test_matches_config_name_only(self):
        """Test AudioDevice matches_config with name-only match."""
        device = AudioDevice(
            id=1,
            name="Test Device",
            hostapi_name="ALSA",
            channels=2,
            sample_rate=44100.0,
        )

        config = {"name": "Test Device"}
        assert device.matches_config(config) is True

    def test_matches_config_no_match(self):
        """Test AudioDevice matches_config with no match."""
        device = AudioDevice(
            id=1,
            name="Test Device",
            hostapi_name="ALSA",
            channels=2,
            sample_rate=44100.0,
        )

        config = {"name": "Different Device", "hostapi_name": "ALSA"}
        assert device.matches_config(config) is False

    def test_matches_config_none(self):
        """Test AudioDevice matches_config with None config."""
        device = AudioDevice(
            id=1,
            name="Test Device",
            hostapi_name="ALSA",
            channels=2,
            sample_rate=44100.0,
        )

        assert device.matches_config(None) is False


class TestAudioDeviceManager:
    """Test AudioDeviceManager functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.mock_devices = [
            {
                "name": "Device 1",
                "max_input_channels": 2,
                "default_samplerate": 44100.0,
                "hostapi": 0,
            },
            {
                "name": "Device 2",
                "max_input_channels": 1,
                "default_samplerate": 48000.0,
                "hostapi": 1,
            },
            {
                "name": "Output Only",
                "max_input_channels": 0,  # Should be filtered out
                "default_samplerate": 44100.0,
                "hostapi": 0,
            },
        ]

        self.mock_hostapis = [{"name": "ALSA"}, {"name": "JACK Audio Connection Kit"}]

    def test_init_no_config(self):
        """Test AudioDeviceManager initialization without config."""
        manager = AudioDeviceManager()

        assert manager._device_config is None
        assert manager._current_device is None
        assert manager._devices_cache is None

    def test_init_with_config(self):
        """Test AudioDeviceManager initialization with config."""
        config = {"name": "Test Device", "hostapi_name": "ALSA"}
        manager = AudioDeviceManager(config)

        assert manager._device_config == config
        assert manager._current_device is None
        assert manager._devices_cache is None

    @patch("sounddevice.query_devices")
    @patch("sounddevice.query_hostapis")
    def test_enumerate_devices(self, mock_query_hostapis, mock_query_devices):
        """Test device enumeration."""
        mock_query_devices.return_value = self.mock_devices
        mock_query_hostapis.return_value = self.mock_hostapis

        manager = AudioDeviceManager()
        devices = manager._enumerate_devices()

        # Should only include devices with input channels > 0
        assert len(devices) == 2
        assert devices[0].name == "Device 1"
        assert devices[0].hostapi_name == "ALSA"
        assert devices[1].name == "Device 2"
        assert devices[1].hostapi_name == "JACK Audio Connection Kit"

    @patch("sounddevice.query_devices")
    @patch("sounddevice.query_hostapis")
    def test_enumerate_devices_error(self, mock_query_hostapis, mock_query_devices):
        """Test device enumeration with sounddevice error."""
        mock_query_devices.side_effect = Exception("Sounddevice error")

        manager = AudioDeviceManager()

        with pytest.raises(AudioDeviceInitializationError):
            manager._enumerate_devices()

    @patch("sounddevice.query_devices")
    @patch("sounddevice.query_hostapis")
    def test_enumerate_devices_no_input_devices(
        self, mock_query_hostapis, mock_query_devices
    ):
        """Test device enumeration when no input devices available."""
        # All devices have 0 input channels
        mock_devices = [
            {
                "name": "Output Only 1",
                "max_input_channels": 0,
                "default_samplerate": 44100.0,
                "hostapi": 0,
            },
            {
                "name": "Output Only 2",
                "max_input_channels": 0,
                "default_samplerate": 44100.0,
                "hostapi": 0,
            },
        ]

        mock_query_devices.return_value = mock_devices
        mock_query_hostapis.return_value = self.mock_hostapis

        manager = AudioDeviceManager()

        with pytest.raises(NoAudioDevicesError):
            manager._enumerate_devices()

    @patch("sounddevice.query_devices")
    @patch("sounddevice.query_hostapis")
    def test_list_devices(self, mock_query_hostapis, mock_query_devices):
        """Test list_devices method."""
        mock_query_devices.return_value = self.mock_devices
        mock_query_hostapis.return_value = self.mock_hostapis

        manager = AudioDeviceManager()

        # First call should enumerate devices
        devices = manager.list_devices()
        assert len(devices) == 2

        # Second call should use cache
        devices2 = manager.list_devices()
        assert devices is devices2  # Same object reference

        # Refresh should re-enumerate
        devices3 = manager.list_devices(refresh=True)
        assert len(devices3) == 2

    @patch("sounddevice.query_devices")
    @patch("sounddevice.query_hostapis")
    def test_get_current_device_no_config(
        self, mock_query_hostapis, mock_query_devices
    ):
        """Test get_current_device with no config (should return None for default)."""
        manager = AudioDeviceManager()

        device = manager.get_current_device()
        assert device is None

    @patch("sounddevice.query_devices")
    @patch("sounddevice.query_hostapis")
    def test_get_current_device_with_valid_config(
        self, mock_query_hostapis, mock_query_devices
    ):
        """Test get_current_device with valid config."""
        mock_query_devices.return_value = self.mock_devices
        mock_query_hostapis.return_value = self.mock_hostapis

        config = {"name": "Device 1", "hostapi_name": "ALSA"}
        manager = AudioDeviceManager(config)

        device = manager.get_current_device()

        assert device is not None
        assert device.name == "Device 1"
        assert device.hostapi_name == "ALSA"

    @patch("sounddevice.query_devices")
    @patch("sounddevice.query_hostapis")
    def test_get_current_device_invalid_config_format(
        self, mock_query_hostapis, mock_query_devices
    ):
        """Test get_current_device with invalid config format."""
        config = "invalid_config"  # Should be dict
        manager = AudioDeviceManager(config)

        with pytest.raises(AudioConfigurationError):
            manager.get_current_device()

    @patch("sounddevice.query_devices")
    @patch("sounddevice.query_hostapis")
    def test_get_current_device_missing_name(
        self, mock_query_hostapis, mock_query_devices
    ):
        """Test get_current_device with config missing name."""
        config = {"hostapi_name": "ALSA"}  # Missing "name" key
        manager = AudioDeviceManager(config)

        with pytest.raises(AudioConfigurationError):
            manager.get_current_device()

    @patch("sounddevice.query_devices")
    @patch("sounddevice.query_hostapis")
    def test_get_current_device_device_not_found(
        self, mock_query_hostapis, mock_query_devices
    ):
        """Test get_current_device when configured device not found."""
        mock_query_devices.return_value = self.mock_devices
        mock_query_hostapis.return_value = self.mock_hostapis

        config = {"name": "Nonexistent Device", "hostapi_name": "ALSA"}
        manager = AudioDeviceManager(config)

        # Should reset config and return None
        device = manager.get_current_device()

        assert device is None
        assert manager._device_config is None

    def test_get_current_device_id(self):
        """Test get_current_device_id method."""
        manager = AudioDeviceManager()

        # Mock current device
        mock_device = AudioDevice(
            id=5, name="Test", hostapi_name="ALSA", channels=2, sample_rate=44100
        )
        manager._current_device = mock_device

        device_id = manager.get_current_device_id()
        assert device_id == 5

        # Test with None device
        manager._current_device = None
        device_id = manager.get_current_device_id()
        assert device_id is None

    @patch("sounddevice.check_input_settings")
    def test_switch_device_success(self, mock_check_input):
        """Test successful device switching."""
        mock_check_input.return_value = None  # No exception means success

        device = AudioDevice(
            id=1, name="Test Device", hostapi_name="ALSA", channels=2, sample_rate=44100
        )

        manager = AudioDeviceManager()
        manager.switch_device(device)

        assert manager._current_device == device
        assert manager._device_config == device.to_config()
        mock_check_input.assert_called_once_with(device=1, samplerate=16000)

    @patch("sounddevice.check_input_settings")
    def test_switch_device_failure(self, mock_check_input):
        """Test device switching failure."""
        mock_check_input.side_effect = Exception("Device test failed")

        device = AudioDevice(
            id=1, name="Test Device", hostapi_name="ALSA", channels=2, sample_rate=44100
        )

        manager = AudioDeviceManager()

        with pytest.raises(AudioDeviceTestError) as exc_info:
            manager.switch_device(device)

        assert "Test Device" in str(exc_info.value)
        assert "input_settings" in str(exc_info.value)

    def test_get_device_config(self):
        """Test get_device_config method."""
        manager = AudioDeviceManager()

        # Test with no current device
        config = manager.get_device_config()
        assert config is None

        # Test with current device
        device = AudioDevice(
            id=1, name="Test Device", hostapi_name="ALSA", channels=2, sample_rate=44100
        )
        manager._current_device = device

        config = manager.get_device_config()
        expected = {"name": "Test Device", "hostapi_name": "ALSA"}
        assert config == expected

    @patch("sounddevice.default")
    @patch("sounddevice.query_devices")
    @patch("sounddevice.query_hostapis")
    def test_get_default_device(
        self, mock_query_hostapis, mock_query_devices, mock_default
    ):
        """Test get_default_device method."""
        mock_default.device = [3, None]  # Input device ID = 3
        mock_query_devices.return_value = {
            "name": "Default Device",
            "hostapi": 0,
            "max_input_channels": 2,
            "default_samplerate": 44100.0,
        }
        mock_query_hostapis.return_value = [{"name": "ALSA"}]

        manager = AudioDeviceManager()
        device = manager.get_default_device()

        assert device is not None
        assert device.id == 3
        assert device.name == "Default Device"
        assert device.hostapi_name == "ALSA"

    @patch("sounddevice.default")
    def test_get_default_device_none(self, mock_default):
        """Test get_default_device when no default device."""
        mock_default.device = [None, None]

        manager = AudioDeviceManager()
        device = manager.get_default_device()

        assert device is None

    @patch("sounddevice.default")
    def test_get_default_device_error(self, mock_default):
        """Test get_default_device with error."""
        mock_default.device = [3, None]

        # Mock query_devices to raise exception
        with patch("sounddevice.query_devices", side_effect=Exception("Error")):
            manager = AudioDeviceManager()
            device = manager.get_default_device()
            assert device is None

    @patch("sounddevice.query_devices")
    @patch("sounddevice.query_hostapis")
    def test_group_devices_by_hostapi(self, mock_query_hostapis, mock_query_devices):
        """Test group_devices_by_hostapi method."""
        mock_query_devices.return_value = self.mock_devices
        mock_query_hostapis.return_value = self.mock_hostapis

        manager = AudioDeviceManager()
        grouped = manager.group_devices_by_hostapi()

        assert "ALSA" in grouped
        assert "JACK Audio Connection Kit" in grouped
        assert len(grouped["ALSA"]) == 1
        assert len(grouped["JACK Audio Connection Kit"]) == 1
        assert grouped["ALSA"][0].name == "Device 1"
        assert grouped["JACK Audio Connection Kit"][0].name == "Device 2"

    @patch("sounddevice.query_devices")
    @patch("sounddevice.query_hostapis")
    def test_enumerate_devices_with_bad_device(
        self, mock_query_hostapis, mock_query_devices
    ):
        """Test device enumeration with one bad device that causes exception."""
        # Create mock devices where one will cause an exception during processing
        mock_devices_with_bad = [
            {
                "name": "Good Device",
                "max_input_channels": 2,
                "default_samplerate": 44100.0,
                "hostapi": 0,
            },
            {
                "name": "Bad Device",
                "max_input_channels": 1,
                "default_samplerate": 48000.0,
                "hostapi": 999,  # Invalid hostapi index
            },
        ]

        mock_query_devices.return_value = mock_devices_with_bad
        mock_query_hostapis.return_value = self.mock_hostapis

        manager = AudioDeviceManager()
        devices = manager._enumerate_devices()

        # Should handle bad device gracefully by using "Unknown" hostapi
        assert len(devices) == 2
        assert devices[0].name == "Good Device"
        assert devices[0].hostapi_name == "ALSA"
        assert devices[1].name == "Bad Device"
        assert devices[1].hostapi_name == "Unknown"
