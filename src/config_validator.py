"""
Configuration Validator Module

Provides centralized validation logic for configuration values and key combinations.
"""

from typing import Dict, Any, Optional
from pynput import keyboard


class ValidationError(Exception):
    """Raised when configuration validation fails."""

    pass


class ConfigValidator:
    """
    Centralized configuration validation with detailed error messages.

    Features:
    - Key combination validation using pynput
    - Configuration section validation
    - Custom validation rules
    - Helpful error messages with suggestions
    """

    # Valid model sizes for Whisper
    VALID_MODELS = {"tiny", "base", "small", "medium", "large-v3"}

    # Valid processing devices
    VALID_DEVICES = {"cpu", "cuda"}

    # Valid recording modes
    VALID_RECORDING_MODES = {"push-to-talk", "tap-mode"}

    def __init__(self):
        """Initialize the configuration validator."""
        pass

    def validate_key_combination(self, key_str: str) -> set[keyboard.Key]:
        """
        Validate and parse a key combination string.

        Args:
            key_str: Key combination string to validate

        Returns:
            Set of pynput Key objects

        Raises:
            ValidationError: If key combination is invalid
        """
        try:
            parsed_keys = keyboard.HotKey.parse(key_str)
            return set(parsed_keys)
        except ValueError as e:
            raise ValidationError(
                f"Invalid key combination: '{key_str}'. "
                f"Use format like '<ctrl>+<shift>+r', '<scroll_lock>', or 'a'. "
                f"Original error: {e}"
            ) from e

    def validate_single_key(self, key_str: str) -> keyboard.Key:
        """
        Validate and parse a single key string.

        Args:
            key_str: Single key string to validate

        Returns:
            pynput Key object

        Raises:
            ValidationError: If key is invalid or is a combination
        """
        parsed_keys = self.validate_key_combination(key_str)
        if len(parsed_keys) != 1:
            raise ValidationError(
                f"Expected single key, got combination: '{key_str}'. "
                f"Use format like '<esc>', '<delete>', or 'a'"
            )
        return list(parsed_keys)[0]

    def validate_model_size(self, model: str) -> str:
        """
        Validate Whisper model size.

        Args:
            model: Model size string

        Returns:
            Validated model string

        Raises:
            ValidationError: If model is invalid
        """
        if model not in self.VALID_MODELS:
            raise ValidationError(
                f"Invalid model '{model}'. Valid options: {', '.join(sorted(self.VALID_MODELS))}"
            )
        return model

    def validate_device(self, device: str) -> str:
        """
        Validate processing device.

        Args:
            device: Device string

        Returns:
            Validated device string

        Raises:
            ValidationError: If device is invalid
        """
        if device not in self.VALID_DEVICES:
            raise ValidationError(
                f"Invalid device '{device}'. Valid options: {', '.join(sorted(self.VALID_DEVICES))}"
            )
        return device

    def validate_recording_mode(self, mode: str) -> str:
        """
        Validate recording mode.

        Args:
            mode: Recording mode string

        Returns:
            Validated mode string

        Raises:
            ValidationError: If mode is invalid
        """
        if mode not in self.VALID_RECORDING_MODES:
            raise ValidationError(
                f"Invalid recording mode '{mode}'. Valid options: {', '.join(sorted(self.VALID_RECORDING_MODES))}"
            )
        return mode

    def validate_language_code(self, language: str) -> str:
        """
        Validate language code.

        Args:
            language: Language code or 'auto'

        Returns:
            Validated language code

        Raises:
            ValidationError: If language code is invalid
        """
        if language == "auto":
            return language

        # Basic validation for language codes (2-3 characters)
        if (
            not isinstance(language, str)
            or not (2 <= len(language) <= 3)
            or not language.isalpha()
        ):
            raise ValidationError(
                f"Invalid language code '{language}'. Use 'auto' for detection or valid codes like 'en', 'es', 'fr'"
            )

        return language.lower()

    def validate_audio_device_config(
        self, device_config: Optional[Dict[str, str]]
    ) -> Optional[Dict[str, str]]:
        """
        Validate audio device configuration.

        Args:
            device_config: Audio device configuration dict

        Returns:
            Validated device config

        Raises:
            ValidationError: If device config is invalid
        """
        if device_config is None:
            return None

        if not isinstance(device_config, dict):
            raise ValidationError("Audio device config must be a dictionary or None")

        required_keys = {"name"}
        optional_keys = {"hostapi_name"}

        if not required_keys.issubset(device_config.keys()):
            missing = required_keys - device_config.keys()
            raise ValidationError(
                f"Audio device config missing required keys: {missing}"
            )

        extra_keys = device_config.keys() - (required_keys | optional_keys)
        if extra_keys:
            raise ValidationError(
                f"Audio device config has unexpected keys: {extra_keys}"
            )

        return device_config

    def validate_config_section(self, section_name: str, section_data: Any) -> Any:
        """
        Validate a complete configuration section.

        Args:
            section_name: Name of the configuration section
            section_data: Configuration section data (dataclass instance)

        Returns:
            Validated section data

        Raises:
            ValidationError: If section is invalid
        """
        if section_name == "general":
            return self._validate_general_config(section_data)
        elif section_name == "recording":
            return self._validate_recording_config(section_data)
        elif section_name == "ui":
            return self._validate_ui_config(section_data)
        elif section_name == "advanced":
            return self._validate_advanced_config(section_data)
        else:
            raise ValidationError(f"Unknown configuration section: {section_name}")

    def _validate_general_config(self, config) -> Any:
        """Validate general configuration section."""
        self.validate_model_size(config.model)
        self.validate_device(config.device)
        self.validate_language_code(config.language)

        if not isinstance(config.debug, bool):
            raise ValidationError("debug must be a boolean")

        if not isinstance(config.trailing_space, bool):
            raise ValidationError("trailing_space must be a boolean")

        return config

    def _validate_recording_config(self, config) -> Any:
        """Validate recording configuration section."""
        self.validate_recording_mode(config.mode)
        self.validate_key_combination(config.trigger_key)
        self.validate_single_key(config.discard_key)
        self.validate_audio_device_config(config.audio_device)

        return config

    def _validate_ui_config(self, config) -> Any:
        """Validate UI configuration section."""
        if not isinstance(config.use_tray, bool):
            raise ValidationError("use_tray must be a boolean")

        return config

    def _validate_advanced_config(self, config) -> Any:
        """Validate advanced configuration section."""
        if not isinstance(config.sample_rate, int) or config.sample_rate <= 0:
            raise ValidationError("sample_rate must be a positive integer")

        if not isinstance(config.chunk_size, int) or config.chunk_size <= 0:
            raise ValidationError("chunk_size must be a positive integer")

        if not isinstance(config.vad_filter, bool):
            raise ValidationError("vad_filter must be a boolean")

        return config

    def get_validation_help(self, section_name: str, field_name: str) -> str:
        """
        Get help text for a specific configuration field.

        Args:
            section_name: Configuration section name
            field_name: Field name within the section

        Returns:
            Help text for the field
        """
        help_text = {
            (
                "general",
                "model",
            ): f"Valid models: {', '.join(sorted(self.VALID_MODELS))}",
            (
                "general",
                "device",
            ): f"Valid devices: {', '.join(sorted(self.VALID_DEVICES))}",
            (
                "general",
                "language",
            ): "Use 'auto' for detection or language codes like 'en', 'es', 'fr'",
            (
                "recording",
                "mode",
            ): f"Valid modes: {', '.join(sorted(self.VALID_RECORDING_MODES))}",
            (
                "recording",
                "trigger_key",
            ): "Examples: '<scroll_lock>', '<ctrl>+<shift>+r', 'a'",
            (
                "recording",
                "discard_key",
            ): "Single key only. Examples: '<esc>', '<delete>', 'x'",
        }

        return help_text.get(
            (section_name, field_name), "No help available for this field"
        )
