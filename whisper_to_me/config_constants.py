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
PROCESSING_SECTION: Final[str] = "processing"
TRANSCRIPTION_SECTION: Final[str] = "transcription"
CONTEXT_SECTION: Final[str] = "context"
PROFILES_SECTION: Final[str] = "profiles"

# All configuration sections
ALL_SECTIONS: Final[list[str]] = [
    GENERAL_SECTION,
    RECORDING_SECTION,
    UI_SECTION,
    ADVANCED_SECTION,
    PROCESSING_SECTION,
    TRANSCRIPTION_SECTION,
    CONTEXT_SECTION,
    PROFILES_SECTION,
]

# Required sections for validation
REQUIRED_SECTIONS: Final[list[str]] = [
    GENERAL_SECTION,
    RECORDING_SECTION,
    UI_SECTION,
    ADVANCED_SECTION,
    PROCESSING_SECTION,
    TRANSCRIPTION_SECTION,
    CONTEXT_SECTION,
]

# Default profile name
DEFAULT_PROFILE: Final[str] = "default"

# Remote ASR model defaults
DEFAULT_OPENAI_TRANSCRIPTION_MODEL: Final[str] = "whisper-1"
DEFAULT_QWEN_ASR_MODEL: Final[str] = "Qwen/Qwen3-ASR-1.7B"


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

    CHUNK_SIZE: Final[str] = "chunk_size"
    VAD_FILTER: Final[str] = "vad_filter"
    INITIAL_PROMPT: Final[str] = "initial_prompt"
    FAST_TYPING_DELAY_MS: Final[str] = "fast_typing_delay_ms"


class ProcessingFields:
    """Field names for processing configuration section."""

    ENABLED: Final[str] = "enabled"
    BACKEND: Final[str] = "backend"
    MODEL: Final[str] = "model"
    API_URL: Final[str] = "api_url"
    API_KEY: Final[str] = "api_key"
    TEMPERATURE: Final[str] = "temperature"
    SYSTEM_PROMPT: Final[str] = "system_prompt"
    TIMEOUT: Final[str] = "timeout"


class TranscriptionFields:
    """Field names for transcription backend configuration."""

    BACKEND: Final[str] = "backend"
    URL: Final[str] = "url"
    MODEL: Final[str] = "model"
    API_KEY: Final[str] = "api_key"
    TIMEOUT: Final[str] = "timeout"
    FALLBACK_TO_LOCAL: Final[str] = "fallback_to_local"


class ContextFields:
    """Field names for shared ASR/processing context configuration."""

    ENABLED: Final[str] = "enabled"
    INCLUDE_WINDOW_TITLE: Final[str] = "include_window_title"
    BASE: Final[str] = "base"
    ASR_PROMPT: Final[str] = "asr_prompt"
    PROCESSING_PROMPT: Final[str] = "processing_prompt"
    TERMS: Final[str] = "terms"
    ROLLING_GLOSSARY_ENABLED: Final[str] = "rolling_glossary_enabled"
    ROLLING_GLOSSARY_RESET_ON_CONTEXT_CHANGE: Final[str] = "rolling_glossary_reset_on_context_change"
    ROLLING_GLOSSARY_MAX_TERMS: Final[str] = "rolling_glossary_max_terms"
    ROLLING_GLOSSARY_CONTEXT_TERMS: Final[str] = "rolling_glossary_context_terms"
    MAX_ASR_CHARS: Final[str] = "max_asr_chars"
    MAX_PROCESSING_CHARS: Final[str] = "max_processing_chars"
    RULES: Final[str] = "rules"


# Processing backends
class ProcessingBackends:
    """Valid processing backend constants."""

    OLLAMA: Final[str] = "ollama"
    OPENAI: Final[str] = "openai"
    OPENAI_CODEX: Final[str] = "openai-codex"
    ANTHROPIC: Final[str] = "anthropic"
    PI: Final[str] = "pi"  # alias for anthropic, uses pi's OAuth credentials


class TranscriptionBackends:
    """Valid speech-to-text backend constants."""

    LOCAL: Final[str] = "local"
    WHISPER_ASR: Final[str] = "whisper-asr"
    REMOTE: Final[str] = "remote"  # alias for whisper-asr/simple multipart APIs
    QWEN_ASR: Final[str] = "qwen-asr"
    OPENAI: Final[str] = "openai"


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
