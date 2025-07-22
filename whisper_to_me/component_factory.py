"""
Component Factory Module

Handles creation and configuration of application components with dependency injection.
Reduces tight coupling in the main application class.
"""

from collections.abc import Callable

from whisper_to_me.application_state_manager import ApplicationStateManager
from whisper_to_me.audio_device_manager import AudioDeviceManager
from whisper_to_me.audio_recorder import AudioRecorder
from whisper_to_me.config import AppConfig, ConfigManager
from whisper_to_me.keystroke_handler import KeystrokeHandler
from whisper_to_me.logger import get_logger
from whisper_to_me.speech_processor import SpeechProcessor
from whisper_to_me.tray_icon import TrayIcon


class ComponentFactory:
    """
    Factory class for creating and configuring application components.

    Features:
    - Centralized component creation
    - Dependency injection
    - Configuration-driven initialization
    - Error handling for component failures
    """

    def __init__(self, config: AppConfig, config_manager: ConfigManager):
        """
        Initialize the component factory.

        Args:
            config: Application configuration
            config_manager: Configuration manager for profile operations
        """
        self.config = config
        self.config_manager = config_manager
        self.state_manager = ApplicationStateManager()
        self.logger = get_logger()

    def create_device_manager(self) -> AudioDeviceManager:
        """
        Create and configure audio device manager.

        Returns:
            Configured AudioDeviceManager instance
        """
        return AudioDeviceManager(self.config.recording.audio_device)

    def create_audio_recorder(
        self, device_manager: AudioDeviceManager
    ) -> AudioRecorder:
        """
        Create and configure audio recorder.

        Args:
            device_manager: Audio device manager for device configuration

        Returns:
            Configured AudioRecorder instance

        Raises:
            RuntimeError: If audio recorder initialization fails
        """
        device = device_manager.get_current_device()

        try:
            return AudioRecorder(
                device_id=device_manager.get_current_device_id(),
                device_name=device.name if device else None,
            )
        except Exception as e:
            if device is not None:
                self.logger.error(
                    f"Audio device '{device.name}' failed to initialize: {e}", "audio"
                )

                # Show available devices to help user
                available_devices = device_manager.list_devices()
                if available_devices:
                    self.logger.info("Available audio devices:", "device")
                    for dev in available_devices:
                        self.logger.info(
                            f"   {dev.id}: {dev.name} ({dev.hostapi_name})", "device"
                        )

                self.logger.info(
                    "Falling back to default audio device...", "audio", "processing"
                )
                device_manager._device_config = None
                device_manager._current_device = None
                return AudioRecorder(device_id=None)
            else:
                # Even default device failed
                self.logger.critical(
                    f"Failed to initialize default audio device: {e}", "audio"
                )
                raise RuntimeError(f"Audio recorder initialization failed: {e}") from e

    def create_speech_processor(self) -> SpeechProcessor:
        """
        Create and configure speech processor.

        Returns:
            Configured SpeechProcessor instance
        """
        return SpeechProcessor(
            model_size=self.config.general.model,
            device=self.config.general.device,
            language=self.config.general.language
            if self.config.general.language != "auto"
            else None,
            vad_filter=self.config.advanced.vad_filter,
            initial_prompt=self.config.advanced.initial_prompt,
        )

    def create_keystroke_handler(self) -> KeystrokeHandler:
        """
        Create and configure keystroke handler.

        Returns:
            Configured KeystrokeHandler instance
        """
        return KeystrokeHandler()

    def create_tray_icon(
        self,
        on_quit: Callable,
        on_profile_change: Callable[[str], None],
        on_device_change: Callable[[int], None],
        get_devices: Callable[[], list[dict]],
        get_current_device: Callable[[], dict | None],
    ) -> TrayIcon | None:
        """
        Create and configure tray icon if enabled.

        Args:
            on_quit: Callback for quit action
            on_profile_change: Callback for profile changes
            on_device_change: Callback for device changes
            get_devices: Function to get available devices
            get_current_device: Function to get current device

        Returns:
            Configured TrayIcon instance or None if disabled
        """
        if not self.config.ui.use_tray:
            return None

        return TrayIcon(
            on_quit=on_quit,
            on_profile_change=on_profile_change,
            get_profiles=self.config_manager.get_profile_names,
            get_current_profile=self.config_manager.get_current_profile,
            on_device_change=on_device_change,
            get_devices=get_devices,
            get_current_device=get_current_device,
        )

    def get_state_manager(self) -> ApplicationStateManager:
        """
        Get the application state manager.

        Returns:
            ApplicationStateManager instance
        """
        return self.state_manager

    def recreate_speech_processor(
        self, old_config: AppConfig, new_config: AppConfig
    ) -> SpeechProcessor | None:
        """
        Recreate speech processor if configuration changed.

        Args:
            old_config: Previous configuration
            new_config: New configuration

        Returns:
            New SpeechProcessor instance if recreation needed, None otherwise
        """
        if (
            old_config.general.language != new_config.general.language
            or old_config.general.model != new_config.general.model
            or old_config.general.device != new_config.general.device
            or old_config.advanced.initial_prompt != new_config.advanced.initial_prompt
        ):
            if old_config.general.language != new_config.general.language:
                self.logger.info(
                    f"Language changed: {old_config.general.language} → {new_config.general.language}",
                    "config",
                    "language",
                )

            if (
                old_config.general.model != new_config.general.model
                or old_config.general.device != new_config.general.device
            ):
                self.logger.info(
                    f"Model/device changed: {old_config.general.model}@{old_config.general.device} → {new_config.general.model}@{new_config.general.device}",
                    "config",
                    "model",
                )

            # Create new speech processor with updated config
            self.config = new_config
            return self.create_speech_processor()

        return None
