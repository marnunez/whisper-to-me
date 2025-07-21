"""
Audio Device Management Module

Centralized management of audio input devices, including enumeration,
selection, persistence, and fallback handling.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import sounddevice as sd
from whisper_to_me.audio_exceptions import (
    AudioDeviceTestError,
    AudioDeviceInitializationError,
    NoAudioDevicesError,
    AudioConfigurationError,
)
from whisper_to_me.logger import get_logger


@dataclass
class AudioDevice:
    """Represents an audio input device with all relevant properties."""

    id: int
    name: str
    hostapi_name: str
    channels: int
    sample_rate: float

    def to_config(self) -> Dict[str, str]:
        """Convert to configuration format (minimal data for persistence)."""
        return {"name": self.name, "hostapi_name": self.hostapi_name}

    @classmethod
    def from_config(
        cls, config: Dict[str, str], device_info: Dict[str, Any]
    ) -> "AudioDevice":
        """Create AudioDevice from config dict and full device info."""
        return cls(
            id=device_info["id"],
            name=device_info["name"],
            hostapi_name=device_info["hostapi_name"],
            channels=device_info["channels"],
            sample_rate=device_info["default_samplerate"],
        )

    def matches_config(self, config: Optional[Dict[str, str]]) -> bool:
        """Check if this device matches a configuration dict."""
        if not config:
            return False

        # Exact match: both name and hostapi
        if config.get("hostapi_name"):
            return self.name == config.get("name") and self.hostapi_name == config.get(
                "hostapi_name"
            )

        # Name-only match
        return self.name == config.get("name")


class AudioDeviceManager:
    """Manages audio device enumeration, selection, and persistence."""

    def __init__(self, device_config: Optional[Dict[str, str]] = None):
        """
        Initialize the audio device manager.

        Args:
            device_config: Optional device configuration dict with 'name' and 'hostapi_name'
        """
        self._device_config = device_config
        self._current_device: Optional[AudioDevice] = None
        self._devices_cache: Optional[List[AudioDevice]] = None
        self.logger = get_logger()

    def get_current_device(self) -> Optional[AudioDevice]:
        """
        Get the currently selected audio device.

        Returns:
            AudioDevice object or None for default device
        """
        if self._current_device is None:
            self._resolve_device()
        return self._current_device

    def get_current_device_id(self) -> Optional[int]:
        """
        Get the ID of the current device for AudioRecorder.

        Returns:
            Device ID or None for default device
        """
        device = self.get_current_device()
        return device.id if device else None

    def list_devices(self, refresh: bool = False) -> List[AudioDevice]:
        """
        List all available audio input devices.

        Args:
            refresh: Force refresh of device list

        Returns:
            List of AudioDevice objects
        """
        if self._devices_cache is None or refresh:
            self._devices_cache = self._enumerate_devices()
        return self._devices_cache

    def switch_device(self, device: AudioDevice) -> None:
        """
        Switch to a different audio device.

        Args:
            device: AudioDevice to switch to

        Raises:
            AudioDeviceTestError: If device fails testing
        """
        # Test if device works
        try:
            sd.check_input_settings(device=device.id, samplerate=16000)
            self._current_device = device
            self._device_config = device.to_config()
        except Exception as e:
            raise AudioDeviceTestError(
                device.name, device.id, "input_settings", e
            ) from e

    def get_device_config(self) -> Optional[Dict[str, str]]:
        """Get the current device configuration for persistence."""
        if self._current_device:
            return self._current_device.to_config()
        return self._device_config

    def get_default_device(self) -> Optional[AudioDevice]:
        """Get information about the default input device."""
        try:
            default_id = sd.default.device[0]
            if default_id is None:
                return None

            device_info = sd.query_devices(default_id)
            hostapis = sd.query_hostapis()
            hostapi_index = device_info["hostapi"]
            hostapi_name = (
                hostapis[hostapi_index]["name"]
                if hostapi_index < len(hostapis)
                else "Unknown"
            )

            return AudioDevice(
                id=default_id,
                name=device_info["name"],
                hostapi_name=hostapi_name,
                channels=device_info["max_input_channels"],
                sample_rate=device_info["default_samplerate"],
            )
        except Exception:
            return None

    def _resolve_device(self) -> None:
        """
        Resolve the configured device to an actual AudioDevice.

        Raises:
            AudioDeviceNotFoundError: If configured device is not found
        """
        if not self._device_config:
            self._current_device = None
            return

        # Validate configuration
        if (
            not isinstance(self._device_config, dict)
            or "name" not in self._device_config
        ):
            raise AudioConfigurationError(
                "Invalid device configuration format", "Expected dict with 'name' key"
            )

        # Find device matching config
        devices = self.list_devices()
        if not devices:
            raise NoAudioDevicesError()

        for device in devices:
            if device.matches_config(self._device_config):
                self._current_device = device
                return

        # Device not found - provide helpful information
        device_name = self._device_config.get("name")
        self.logger.warning(
            f"Configured audio device '{device_name}' not found", "device"
        )

        # Show available devices
        self.logger.info("Available audio devices:", "device")
        for device in devices:
            self.logger.info(
                f"   {device.id}: {device.name} ({device.hostapi_name})", "device"
            )

        # Reset to allow fallback to default
        self._device_config = None
        self._current_device = None

    def _enumerate_devices(self) -> List[AudioDevice]:
        """
        Enumerate all audio input devices.

        Returns:
            List of available audio input devices

        Raises:
            AudioDeviceInitializationError: If device enumeration fails
        """
        try:
            devices_info = sd.query_devices()
            hostapis = sd.query_hostapis()
        except Exception as e:
            raise AudioDeviceInitializationError("system", e) from e

        audio_devices = []

        for i, device_info in enumerate(devices_info):
            if device_info["max_input_channels"] > 0:
                try:
                    hostapi_index = device_info["hostapi"]
                    hostapi_name = (
                        hostapis[hostapi_index]["name"]
                        if hostapi_index < len(hostapis)
                        else "Unknown"
                    )

                    audio_devices.append(
                        AudioDevice(
                            id=i,
                            name=device_info["name"],
                            hostapi_name=hostapi_name,
                            channels=device_info["max_input_channels"],
                            sample_rate=device_info["default_samplerate"],
                        )
                    )
                except Exception as e:
                    # Log but don't fail enumeration for one bad device
                    self.logger.warning(
                        f"Skipping device {i} due to error: {e}", "device"
                    )
                    continue

        if not audio_devices:
            raise NoAudioDevicesError()

        return audio_devices

    def group_devices_by_hostapi(self) -> Dict[str, List[AudioDevice]]:
        """
        Group devices by their host API for organized display.

        Returns:
            Dict mapping hostapi_name to list of devices
        """
        devices = self.list_devices()
        grouped = {}

        for device in devices:
            if device.hostapi_name not in grouped:
                grouped[device.hostapi_name] = []
            grouped[device.hostapi_name].append(device)

        return grouped
