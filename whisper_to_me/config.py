"""
Configuration Management Module

Provides TOML-based configuration management with profile support for
Whisper-to-Me application.
"""

import os
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tomli_w
from pynput import keyboard

from whisper_to_me.config_constants import (
    ADVANCED_SECTION,
    DEFAULT_PROFILE,
    GENERAL_SECTION,
    PROFILES_SECTION,
    RECORDING_SECTION,
    REQUIRED_SECTIONS,
    UI_SECTION,
    DeviceTypes,
    Languages,
    ModelSizes,
    RecordingModes,
)
from whisper_to_me.config_differ import ConfigSectionDiffer
from whisper_to_me.config_validator import ConfigValidator, ValidationError
from whisper_to_me.logger import get_logger


@dataclass
class RecordingConfig:
    """Recording-specific configuration."""

    mode: str = RecordingModes.PUSH_TO_TALK
    trigger_key: str = "<scroll_lock>"
    discard_key: str = "<esc>"
    audio_device: dict[str, str] | None = None  # {"name": str, "hostapi_name": str}


@dataclass
class UIConfig:
    """User interface configuration."""

    use_tray: bool = True


@dataclass
class AdvancedConfig:
    """Advanced configuration options."""

    chunk_size: int = 512
    vad_filter: bool = True
    initial_prompt: str = ""


@dataclass
class GeneralConfig:
    """General application configuration."""

    model: str = ModelSizes.LARGE_V3
    device: str = DeviceTypes.CUDA
    language: str = Languages.AUTO
    debug: bool = False
    last_profile: str = DEFAULT_PROFILE
    trailing_space: bool = False


@dataclass
class AppConfig:
    """Complete application configuration."""

    general: GeneralConfig
    recording: RecordingConfig
    ui: UIConfig
    advanced: AdvancedConfig
    profiles: dict[str, dict[str, Any]]

    def __post_init__(self):
        """Ensure profiles dict exists."""
        if not hasattr(self, "profiles") or self.profiles is None:
            self.profiles = {}


class ConfigManager:
    """
    Manages TOML configuration files and profiles for Whisper-to-Me.

    Features:
    - TOML configuration file support
    - Multiple profiles with inheritance
    - Hot-reloading configuration changes
    - Validation and fallback to defaults
    """

    def __init__(self, config_file: str | None = None):
        """Initialize configuration manager.

        Args:
            config_file: Path to custom configuration file. If None, uses default location.
        """
        if config_file:
            self.config_file = Path(config_file)
            self.config_dir = self.config_file.parent
        else:
            # Use XDG_CONFIG_HOME if set, otherwise use default
            xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
            if xdg_config_home:
                self.config_dir = Path(xdg_config_home) / "whisper-to-me"
            else:
                self.config_dir = Path.home() / ".config" / "whisper-to-me"
            self.config_file = self.config_dir / "config.toml"

        self.current_profile = DEFAULT_PROFILE
        self._config: AppConfig | None = None
        self._config_differ = ConfigSectionDiffer()
        self._validator = ConfigValidator()
        self.logger = get_logger()
        self._ensure_config_dir()

    def _ensure_config_dir(self) -> None:
        """Create config directory if it doesn't exist."""
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def _filter_config_fields(
        self, config_dict: dict[str, Any], config_class: type
    ) -> dict[str, Any]:
        """Filter config dictionary to only include valid fields for the given dataclass.

        Args:
            config_dict: Configuration dictionary from TOML
            config_class: Dataclass type to filter for

        Returns:
            Filtered dictionary containing only valid fields
        """
        import dataclasses

        # Get valid field names from the dataclass
        valid_fields = {field.name for field in dataclasses.fields(config_class)}

        # Filter the config dict
        filtered = {}
        for key, value in config_dict.items():
            if key in valid_fields:
                filtered[key] = value
            else:
                self.logger.warning(
                    f"Ignoring unknown configuration field '{key}' in {config_class.__name__}. "
                    f"This field is no longer supported and will be removed from your config file.",
                    "config",
                )

        return filtered

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration structure."""
        return {
            GENERAL_SECTION: {
                "model": ModelSizes.LARGE_V3,
                "device": DeviceTypes.CUDA,
                "language": Languages.AUTO,
                "debug": False,
                "last_profile": DEFAULT_PROFILE,
                "trailing_space": False,
            },
            RECORDING_SECTION: {
                "mode": RecordingModes.PUSH_TO_TALK,
                "trigger_key": "<scroll_lock>",
                "discard_key": "<esc>",
                "audio_device": None,
            },
            UI_SECTION: {"use_tray": True},
            ADVANCED_SECTION: {
                "chunk_size": 512,
                "vad_filter": True,
                "initial_prompt": "",
            },
            PROFILES_SECTION: {},
        }

    def _create_default_config(self) -> None:
        """Create default configuration file."""
        default_config = self._get_default_config()

        # Add some example profiles
        default_config["profiles"] = {
            "work": {
                "general": {"language": "en", "model": "medium"},
                "recording": {"trigger_key": "<caps_lock>"},
            },
            "spanish": {"general": {"language": "es", "model": "large-v3"}},
            "quick": {
                "general": {"model": "tiny", "device": "cpu"},
                "recording": {"mode": "tap-mode"},
            },
        }

        # Audio device will be None by default

        self._save_config_to_file(default_config)

    def _save_config_to_file(self, config_dict: dict[str, Any]) -> None:
        """Save configuration dictionary to TOML file."""

        # Remove None values for TOML compatibility
        def remove_none_values(obj):
            if isinstance(obj, dict):
                return {
                    k: remove_none_values(v) for k, v in obj.items() if v is not None
                }
            elif isinstance(obj, list):
                return [remove_none_values(item) for item in obj]
            else:
                return obj

        sanitized_config = remove_none_values(config_dict)

        with open(self.config_file, "wb") as f:
            tomli_w.dump(sanitized_config, f)

    def _load_config_from_file(self) -> dict[str, Any]:
        """Load configuration from TOML file."""
        if not self.config_file.exists():
            self._create_default_config()

        try:
            with open(self.config_file, "rb") as f:
                return tomllib.load(f)
        except Exception as e:
            self.logger.error(f"Error loading config file: {e}", "config")
            self.logger.info("Using default configuration", "config")
            return self._get_default_config()

    def _validate_config(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        """Validate and sanitize configuration."""
        default = self._get_default_config()

        # Ensure all required sections exist
        for section in REQUIRED_SECTIONS:
            if section not in config_dict:
                config_dict[section] = default[section]
            else:
                # Fill missing keys with defaults
                for key, value in default[section].items():
                    if key not in config_dict[section]:
                        config_dict[section][key] = value

        # Ensure profiles section exists
        if PROFILES_SECTION not in config_dict:
            config_dict[PROFILES_SECTION] = {}

        return config_dict

    def load_config(self) -> AppConfig:
        """Load configuration from file and return AppConfig object."""
        config_dict = self._load_config_from_file()
        config_dict = self._validate_config(config_dict)

        self._config = AppConfig(
            general=GeneralConfig(
                **self._filter_config_fields(
                    config_dict[GENERAL_SECTION], GeneralConfig
                )
            ),
            recording=RecordingConfig(
                **self._filter_config_fields(
                    config_dict[RECORDING_SECTION], RecordingConfig
                )
            ),
            ui=UIConfig(
                **self._filter_config_fields(config_dict[UI_SECTION], UIConfig)
            ),
            advanced=AdvancedConfig(
                **self._filter_config_fields(
                    config_dict[ADVANCED_SECTION], AdvancedConfig
                )
            ),
            profiles=config_dict[PROFILES_SECTION],
        )

        # Set current profile from config
        self.current_profile = self._config.general.last_profile
        return self._config

    def save_config(self) -> None:
        """Save current configuration to file."""
        if self._config is None:
            return

        config_dict = {
            GENERAL_SECTION: asdict(self._config.general),
            RECORDING_SECTION: asdict(self._config.recording),
            UI_SECTION: asdict(self._config.ui),
            ADVANCED_SECTION: asdict(self._config.advanced),
            PROFILES_SECTION: self._config.profiles,
        }

        self._save_config_to_file(config_dict)

    def get_profile_names(self) -> list[str]:
        """Get list of available profile names."""
        if self._config is None:
            self.load_config()

        profiles = [DEFAULT_PROFILE]  # Always include default
        if self._config and self._config.profiles:
            profiles.extend(self._config.profiles.keys())

        return sorted(profiles)

    def get_current_profile(self) -> str:
        """Get current active profile name."""
        return self.current_profile

    def apply_profile(self, profile_name: str) -> AppConfig:
        """Apply a profile and return the merged configuration."""
        if self._config is None:
            self.load_config()

        if profile_name == DEFAULT_PROFILE:
            # Use base configuration
            self.current_profile = DEFAULT_PROFILE
            return self._config

        if profile_name not in self._config.profiles:
            self.logger.warning(
                f"Profile '{profile_name}' not found, using default", "profile"
            )
            return self._config

        # Create a copy of the base config
        profile_config = AppConfig(
            general=GeneralConfig(**asdict(self._config.general)),
            recording=RecordingConfig(**asdict(self._config.recording)),
            ui=UIConfig(**asdict(self._config.ui)),
            advanced=AdvancedConfig(**asdict(self._config.advanced)),
            profiles=self._config.profiles,
        )

        # Apply profile overrides using the differ
        profile_data = self._config.profiles[profile_name]
        self._config_differ.apply_profile_data(profile_config, profile_data)

        self.current_profile = profile_name
        profile_config.general.last_profile = profile_name

        return profile_config

    def create_profile(self, name: str, config: AppConfig) -> bool:
        """Create a new profile from the given configuration."""
        if self._config is None:
            self.load_config()

        # Convert current config to profile format using the differ
        default = self._get_default_config()
        profile_data = self._config_differ.create_profile_data(config, default)

        # Save profile
        self._config.profiles[name] = profile_data
        self.save_config()
        return True

    def delete_profile(self, name: str) -> bool:
        """Delete a profile."""
        if self._config is None:
            self.load_config()

        if name == DEFAULT_PROFILE:
            return False  # Cannot delete default profile

        if name in self._config.profiles:
            del self._config.profiles[name]

            # If deleting current profile, switch to default
            if self.current_profile == name:
                self.current_profile = DEFAULT_PROFILE
                self._config.general.last_profile = DEFAULT_PROFILE

            self.save_config()
            return True

        return False

    def get_config_file_path(self) -> str:
        """Get the path to the configuration file."""
        return str(self.config_file)

    def parse_key_combination(self, key_str: str) -> set[keyboard.Key]:
        """Parse a key combination string into a set of pynput Key objects.

        Uses pynput's built-in HotKey.parse for robust parsing.
        Format: '<ctrl>+<shift>+r', '<scroll_lock>', 'a', '+'
        """
        try:
            return self._validator.validate_key_combination(key_str)
        except ValidationError as e:
            raise ValueError(str(e)) from e

    def parse_key_string(self, key_str: str) -> keyboard.Key:
        """Parse a single key string into a pynput Key object.

        Uses the same format as parse_key_combination but ensures only single keys.
        """
        try:
            return self._validator.validate_single_key(key_str)
        except ValidationError as e:
            raise ValueError(str(e)) from e
