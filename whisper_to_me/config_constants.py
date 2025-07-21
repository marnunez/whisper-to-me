"""
Configuration Constants Module

Defines constants for configuration section names and keys to eliminate magic strings
throughout the codebase.
"""

from typing import Final

# Configuration section names
GENERAL_SECTION: Final[str] = "general"
RECORDING_SECTION: Final[str] = "recording"
UI_SECTION: Final[str] = "ui"
ADVANCED_SECTION: Final[str] = "advanced"
PROFILES_SECTION: Final[str] = "profiles"

# All configuration sections
ALL_SECTIONS: Final[list[str]] = [
    GENERAL_SECTION,
    RECORDING_SECTION,
    UI_SECTION,
    ADVANCED_SECTION,
    PROFILES_SECTION,
]

# Required sections for validation
REQUIRED_SECTIONS: Final[list[str]] = [
    GENERAL_SECTION,
    RECORDING_SECTION,
    UI_SECTION,
    ADVANCED_SECTION,
]

# Default profile name
DEFAULT_PROFILE: Final[str] = "default"


# Configuration field names
class GeneralFields:
    """Field names for general configuration section."""

    MODEL: Final[str] = "model"
    DEVICE: Final[str] = "device"
    LANGUAGE: Final[str] = "language"
    DEBUG: Final[str] = "debug"
    LAST_PROFILE: Final[str] = "last_profile"
    TRAILING_SPACE: Final[str] = "trailing_space"


class RecordingFields:
    """Field names for recording configuration section."""

    MODE: Final[str] = "mode"
    TRIGGER_KEY: Final[str] = "trigger_key"
    DISCARD_KEY: Final[str] = "discard_key"
    AUDIO_DEVICE: Final[str] = "audio_device"


class UIFields:
    """Field names for UI configuration section."""

    USE_TRAY: Final[str] = "use_tray"


class AdvancedFields:
    """Field names for advanced configuration section."""

    SAMPLE_RATE: Final[str] = "sample_rate"
    CHUNK_SIZE: Final[str] = "chunk_size"
    VAD_FILTER: Final[str] = "vad_filter"


# Recording modes
class RecordingModes:
    """Valid recording mode constants."""

    PUSH_TO_TALK: Final[str] = "push-to-talk"
    TAP_MODE: Final[str] = "tap-mode"


# Device types
class DeviceTypes:
    """Valid device type constants."""

    CPU: Final[str] = "cpu"
    CUDA: Final[str] = "cuda"


# Model sizes
class ModelSizes:
    """Valid Whisper model size constants."""

    TINY: Final[str] = "tiny"
    BASE: Final[str] = "base"
    SMALL: Final[str] = "small"
    MEDIUM: Final[str] = "medium"
    LARGE_V3: Final[str] = "large-v3"


# Language codes
class Languages:
    """Common language code constants."""

    AUTO: Final[str] = "auto"
    ENGLISH: Final[str] = "en"
    SPANISH: Final[str] = "es"
    FRENCH: Final[str] = "fr"
    GERMAN: Final[str] = "de"
    ITALIAN: Final[str] = "it"
    PORTUGUESE: Final[str] = "pt"
    CHINESE: Final[str] = "zh"
    JAPANESE: Final[str] = "ja"
    KOREAN: Final[str] = "ko"
