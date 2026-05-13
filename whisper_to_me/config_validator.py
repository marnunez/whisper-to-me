"""
Configuration Validator Module

Provides centralized validation logic for configuration values and key combinations.
"""

from typing import Any

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

    # Valid transcription backends
    VALID_TRANSCRIPTION_BACKENDS = {"local", "whisper-asr", "remote", "openai"}

    def __init__(self):
        """Initialize the configuration validator."""
        pass

    def validate_key_combination(self, key_str: str) -> set[Any]:
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

    def validate_single_key(self, key_str: str) -> Any:
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
        self, device_config: dict[str, str] | None
    ) -> dict[str, str] | None:
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
        elif section_name == "transcription":
            return self._validate_transcription_config(section_data)
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
        if not isinstance(config.chunk_size, int) or config.chunk_size <= 0:
            raise ValidationError("chunk_size must be a positive integer")

        if not isinstance(config.vad_filter, bool):
            raise ValidationError("vad_filter must be a boolean")

        if config.task not in {"transcribe", "translate"}:
            raise ValidationError("task must be either 'transcribe' or 'translate'")

        if not isinstance(config.beam_size, int) or config.beam_size <= 0:
            raise ValidationError("beam_size must be a positive integer")

        if not isinstance(config.best_of, int) or config.best_of <= 0:
            raise ValidationError("best_of must be a positive integer")

        if not isinstance(config.temperature, int | float) or config.temperature < 0:
            raise ValidationError("temperature must be a non-negative number")

        if not isinstance(config.condition_on_previous_text, bool):
            raise ValidationError("condition_on_previous_text must be a boolean")

        for field_name in (
            "no_speech_threshold",
            "log_prob_threshold",
            "compression_ratio_threshold",
        ):
            value = getattr(config, field_name)
            if not isinstance(value, int | float):
                raise ValidationError(f"{field_name} must be a number")

        if config.hallucination_silence_threshold is not None and (
            not isinstance(config.hallucination_silence_threshold, int | float)
            or config.hallucination_silence_threshold < 0
        ):
            raise ValidationError(
                "hallucination_silence_threshold must be null or a non-negative number"
            )

        if not isinstance(config.hotwords, str):
            raise ValidationError("hotwords must be a string")

        if (
            not isinstance(config.min_silence_duration_ms, int)
            or config.min_silence_duration_ms <= 0
        ):
            raise ValidationError("min_silence_duration_ms must be a positive integer")

        if not isinstance(config.speech_pad_ms, int) or config.speech_pad_ms < 0:
            raise ValidationError("speech_pad_ms must be a non-negative integer")

        if (
            not isinstance(config.fast_typing_delay_ms, int)
            or config.fast_typing_delay_ms < 0
        ):
            raise ValidationError("fast_typing_delay_ms must be a non-negative integer")

        return config

    def _validate_transcription_config(self, config) -> Any:
        """Validate transcription backend configuration section."""
        if config.backend not in self.VALID_TRANSCRIPTION_BACKENDS:
            raise ValidationError(
                f"Invalid transcription backend '{config.backend}'. Valid options: "
                f"{', '.join(sorted(self.VALID_TRANSCRIPTION_BACKENDS))}"
            )

        if not isinstance(config.url, str):
            raise ValidationError("transcription.url must be a string")

        if not isinstance(config.model, str):
            raise ValidationError("transcription.model must be a string")

        if not isinstance(config.api_key, str):
            raise ValidationError("transcription.api_key must be a string")

        if not isinstance(config.timeout, int | float) or config.timeout <= 0:
            raise ValidationError("transcription.timeout must be a positive number")

        if not isinstance(config.fallback_to_local, bool):
            raise ValidationError("transcription.fallback_to_local must be a boolean")

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
            (
                "transcription",
                "backend",
            ): f"Valid backends: {', '.join(sorted(self.VALID_TRANSCRIPTION_BACKENDS))}",
        }

        return help_text.get(
            (section_name, field_name), "No help available for this field"
        )
