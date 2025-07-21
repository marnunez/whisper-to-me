"""Whisper-to-Me source package."""

# Core configuration and management classes
from .application_state_manager import ApplicationStateManager

# Audio-related classes
from .audio_device_manager import AudioDevice, AudioDeviceManager

# Exceptions
from .audio_exceptions import (
    AudioConfigurationError,
    AudioDeviceInitializationError,
    AudioDeviceNotFoundError,
    AudioDeviceTestError,
    AudioError,
    NoAudioDevicesError,
)
from .audio_recorder import AudioRecorder
from .component_factory import ComponentFactory
from .config import (
    AdvancedConfig,
    AppConfig,
    ConfigManager,
    GeneralConfig,
    RecordingConfig,
    UIConfig,
)
from .hotkey_manager import HotkeyManager

# Input/Output handling
from .keystroke_handler import KeystrokeHandler

# Utilities
from .logger import get_logger
from .menu_builder import (
    DeviceMenuFormatter,
    MenuBuilder,
    ProfileMenuFormatter,
    TrayMenuBuilder,
)
from .profile_manager import ProfileManager
from .single_instance import SingleInstance
from .speech_processor import SpeechProcessor

# UI components
from .tray_icon import TrayIcon

__version__ = "0.3.0"
__all__ = [
    # Configuration
    "AppConfig",
    "ConfigManager",
    "GeneralConfig",
    "RecordingConfig",
    "UIConfig",
    "AdvancedConfig",
    # Management
    "ComponentFactory",
    "ProfileManager",
    # Audio
    "AudioDeviceManager",
    "AudioDevice",
    "AudioRecorder",
    "SpeechProcessor",
    # Input/Output
    "KeystrokeHandler",
    "HotkeyManager",
    # UI
    "TrayIcon",
    "MenuBuilder",
    "TrayMenuBuilder",
    "ProfileMenuFormatter",
    "DeviceMenuFormatter",
    # Utilities
    "get_logger",
    "SingleInstance",
    "ApplicationStateManager",
    # Exceptions
    "AudioError",
    "AudioDeviceNotFoundError",
    "AudioDeviceTestError",
    "AudioDeviceInitializationError",
    "NoAudioDevicesError",
    "AudioConfigurationError",
]
