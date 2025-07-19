"""
Configuration Management Module

Provides TOML-based configuration management with profile support for
Whisper-to-Me application.
"""

import tomllib
import tomli_w
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pynput import keyboard


@dataclass
class RecordingConfig:
    """Recording-specific configuration."""

    mode: str = "push-to-talk"  # "push-to-talk" or "tap-mode"
    trigger_key: str = "scroll_lock"
    discard_key: str = "esc"
    audio_device: Optional[int] = None


@dataclass
class UIConfig:
    """User interface configuration."""

    use_tray: bool = True


@dataclass
class AdvancedConfig:
    """Advanced configuration options."""

    sample_rate: int = 16000
    chunk_size: int = 512
    vad_filter: bool = True


@dataclass
class GeneralConfig:
    """General application configuration."""

    model: str = "large-v3"
    device: str = "cuda"
    language: str = "auto"
    debug: bool = False
    last_profile: str = "default"


@dataclass
class AppConfig:
    """Complete application configuration."""

    general: GeneralConfig
    recording: RecordingConfig
    ui: UIConfig
    advanced: AdvancedConfig
    profiles: Dict[str, Dict[str, Any]]

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

    def __init__(self):
        """Initialize configuration manager."""
        self.config_dir = Path.home() / ".config" / "whisper-to-me"
        self.config_file = self.config_dir / "config.toml"
        self.current_profile = "default"
        self._config: Optional[AppConfig] = None
        self._ensure_config_dir()

    def _ensure_config_dir(self) -> None:
        """Create config directory if it doesn't exist."""
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration structure."""
        return {
            "general": {
                "model": "large-v3",
                "device": "cuda",
                "language": "auto",
                "debug": False,
                "last_profile": "default",
            },
            "recording": {
                "mode": "push-to-talk",
                "trigger_key": "scroll_lock",
                "discard_key": "esc",
                "audio_device": None,
            },
            "ui": {"use_tray": True},
            "advanced": {"sample_rate": 16000, "chunk_size": 512, "vad_filter": True},
            "profiles": {},
        }

    def _create_default_config(self) -> None:
        """Create default configuration file."""
        default_config = self._get_default_config()

        # Add some example profiles
        default_config["profiles"] = {
            "work": {
                "general": {"language": "en", "model": "medium"},
                "recording": {"trigger_key": "caps_lock"},
            },
            "spanish": {"general": {"language": "es", "model": "large-v3"}},
            "quick": {
                "general": {"model": "tiny", "device": "cpu"},
                "recording": {"mode": "tap-mode"},
            },
        }

        # Ensure audio_device is explicitly set
        if default_config["recording"]["audio_device"] is None:
            default_config["recording"]["audio_device"] = ""

        self._save_config_to_file(default_config)

    def _save_config_to_file(self, config_dict: Dict[str, Any]) -> None:
        """Save configuration dictionary to TOML file."""

        # Convert None values to empty strings for TOML compatibility
        def sanitize_for_toml(obj):
            if isinstance(obj, dict):
                return {k: sanitize_for_toml(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_for_toml(item) for item in obj]
            elif obj is None:
                return ""  # Convert None to empty string
            else:
                return obj

        sanitized_config = sanitize_for_toml(config_dict)

        with open(self.config_file, "wb") as f:
            tomli_w.dump(sanitized_config, f)

    def _load_config_from_file(self) -> Dict[str, Any]:
        """Load configuration from TOML file."""
        if not self.config_file.exists():
            self._create_default_config()

        try:
            with open(self.config_file, "rb") as f:
                return tomllib.load(f)
        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Using default configuration")
            return self._get_default_config()

    def _validate_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize configuration."""
        default = self._get_default_config()

        # Convert empty strings back to None for specific fields
        def restore_none_values(obj):
            if isinstance(obj, dict):
                result = {}
                for k, v in obj.items():
                    if k == "audio_device" and v == "":
                        result[k] = None
                    elif isinstance(v, dict):
                        result[k] = restore_none_values(v)
                    else:
                        result[k] = v
                return result
            else:
                return obj

        config_dict = restore_none_values(config_dict)

        # Ensure all required sections exist
        for section in ["general", "recording", "ui", "advanced"]:
            if section not in config_dict:
                config_dict[section] = default[section]
            else:
                # Fill missing keys with defaults
                for key, value in default[section].items():
                    if key not in config_dict[section]:
                        config_dict[section][key] = value

        # Ensure profiles section exists
        if "profiles" not in config_dict:
            config_dict["profiles"] = {}

        return config_dict

    def load_config(self) -> AppConfig:
        """Load configuration from file and return AppConfig object."""
        config_dict = self._load_config_from_file()
        config_dict = self._validate_config(config_dict)

        self._config = AppConfig(
            general=GeneralConfig(**config_dict["general"]),
            recording=RecordingConfig(**config_dict["recording"]),
            ui=UIConfig(**config_dict["ui"]),
            advanced=AdvancedConfig(**config_dict["advanced"]),
            profiles=config_dict["profiles"],
        )

        # Set current profile from config
        self.current_profile = self._config.general.last_profile
        return self._config

    def save_config(self) -> None:
        """Save current configuration to file."""
        if self._config is None:
            return

        config_dict = {
            "general": asdict(self._config.general),
            "recording": asdict(self._config.recording),
            "ui": asdict(self._config.ui),
            "advanced": asdict(self._config.advanced),
            "profiles": self._config.profiles,
        }

        self._save_config_to_file(config_dict)

    def get_profile_names(self) -> List[str]:
        """Get list of available profile names."""
        if self._config is None:
            self.load_config()

        profiles = ["default"]  # Always include default
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

        if profile_name == "default":
            # Use base configuration
            self.current_profile = "default"
            return self._config

        if profile_name not in self._config.profiles:
            print(f"Profile '{profile_name}' not found, using default")
            return self._config

        # Create a copy of the base config
        profile_config = AppConfig(
            general=GeneralConfig(**asdict(self._config.general)),
            recording=RecordingConfig(**asdict(self._config.recording)),
            ui=UIConfig(**asdict(self._config.ui)),
            advanced=AdvancedConfig(**asdict(self._config.advanced)),
            profiles=self._config.profiles,
        )

        # Apply profile overrides
        profile_data = self._config.profiles[profile_name]

        if "general" in profile_data:
            for key, value in profile_data["general"].items():
                if hasattr(profile_config.general, key):
                    setattr(profile_config.general, key, value)

        if "recording" in profile_data:
            for key, value in profile_data["recording"].items():
                if hasattr(profile_config.recording, key):
                    setattr(profile_config.recording, key, value)

        if "ui" in profile_data:
            for key, value in profile_data["ui"].items():
                if hasattr(profile_config.ui, key):
                    setattr(profile_config.ui, key, value)

        if "advanced" in profile_data:
            for key, value in profile_data["advanced"].items():
                if hasattr(profile_config.advanced, key):
                    setattr(profile_config.advanced, key, value)

        self.current_profile = profile_name
        profile_config.general.last_profile = profile_name

        return profile_config

    def create_profile(self, name: str, config: AppConfig) -> bool:
        """Create a new profile from the given configuration."""
        if self._config is None:
            self.load_config()

        # Convert current config to profile format (only non-default values)
        default = self._get_default_config()
        profile_data = {}

        # Compare general settings
        general_diff = {}
        current_general = asdict(config.general)
        for key, value in current_general.items():
            if key != "last_profile" and value != default["general"].get(key):
                general_diff[key] = value
        if general_diff:
            profile_data["general"] = general_diff

        # Compare recording settings
        recording_diff = {}
        current_recording = asdict(config.recording)
        for key, value in current_recording.items():
            if value != default["recording"].get(key):
                recording_diff[key] = value
        if recording_diff:
            profile_data["recording"] = recording_diff

        # Compare UI settings
        ui_diff = {}
        current_ui = asdict(config.ui)
        for key, value in current_ui.items():
            if value != default["ui"].get(key):
                ui_diff[key] = value
        if ui_diff:
            profile_data["ui"] = ui_diff

        # Compare advanced settings
        advanced_diff = {}
        current_advanced = asdict(config.advanced)
        for key, value in current_advanced.items():
            if value != default["advanced"].get(key):
                advanced_diff[key] = value
        if advanced_diff:
            profile_data["advanced"] = advanced_diff

        # Save profile
        self._config.profiles[name] = profile_data
        self.save_config()
        return True

    def delete_profile(self, name: str) -> bool:
        """Delete a profile."""
        if self._config is None:
            self.load_config()

        if name == "default":
            return False  # Cannot delete default profile

        if name in self._config.profiles:
            del self._config.profiles[name]

            # If deleting current profile, switch to default
            if self.current_profile == name:
                self.current_profile = "default"
                self._config.general.last_profile = "default"

            self.save_config()
            return True

        return False

    def get_config_file_path(self) -> str:
        """Get the path to the configuration file."""
        return str(self.config_file)

    def parse_key_string(self, key_str: str) -> keyboard.Key:
        """Parse a key string into a pynput Key object."""
        key_map = {
            "ctrl": keyboard.Key.ctrl_l,
            "alt": keyboard.Key.alt_l,
            "shift": keyboard.Key.shift_l,
            "caps": keyboard.Key.caps_lock,
            "caps_lock": keyboard.Key.caps_lock,
            "tab": keyboard.Key.tab,
            "scroll_lock": keyboard.Key.scroll_lock,
            "pause": keyboard.Key.pause,
            "esc": keyboard.Key.esc,
            "escape": keyboard.Key.esc,
            "del": keyboard.Key.delete,
            "delete": keyboard.Key.delete,
            "backspace": keyboard.Key.backspace,
        }

        key = key_map.get(key_str.lower())
        if key is None and hasattr(keyboard.Key, key_str.lower()):
            key = getattr(keyboard.Key, key_str.lower())

        return key or keyboard.Key.scroll_lock  # fallback
