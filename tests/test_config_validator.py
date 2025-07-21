"""Test configuration validator functionality."""

import pytest

from whisper_to_me.config_validator import ConfigValidator, ValidationError


class TestConfigValidator:
    """Test cases for configuration validation."""

    def setup_method(self):
        """Set up test environment."""
        self.validator = ConfigValidator()

    def test_valid_key_combinations(self):
        """Test validation of valid key combinations."""
        valid_keys = [
            "<scroll_lock>",
            "<caps_lock>",
            "<ctrl>+<shift>+r",
            "<alt>+<space>",
            "a",
            "+",
            "<esc>",
        ]

        for key_str in valid_keys:
            # Should not raise exception
            result = self.validator.validate_key_combination(key_str)
            assert len(result) > 0

    def test_invalid_key_combinations(self):
        """Test validation of invalid key combinations."""
        invalid_keys = [
            "",
            "invalid_key",
            "<nonexistent>",
            "<ctrl>+<invalid>",
        ]

        for key_str in invalid_keys:
            with pytest.raises(ValidationError):
                self.validator.validate_key_combination(key_str)

    def test_valid_single_keys(self):
        """Test validation of valid single keys."""
        valid_single_keys = ["<esc>", "<delete>", "<backspace>", "a", "x"]

        for key_str in valid_single_keys:
            # Should not raise exception
            result = self.validator.validate_single_key(key_str)
            assert result is not None

    def test_invalid_single_keys_combinations(self):
        """Test that key combinations are rejected for single keys."""
        key_combinations = ["<ctrl>+<shift>+r", "<alt>+<space>", "<ctrl>+a"]

        for key_str in key_combinations:
            with pytest.raises(ValidationError) as exc_info:
                self.validator.validate_single_key(key_str)
            assert "Expected single key, got combination" in str(exc_info.value)

    def test_valid_model_sizes(self):
        """Test validation of valid model sizes."""
        valid_models = ["tiny", "base", "small", "medium", "large-v3"]

        for model in valid_models:
            result = self.validator.validate_model_size(model)
            assert result == model

    def test_invalid_model_sizes(self):
        """Test validation of invalid model sizes."""
        invalid_models = ["invalid", "large", "huge", ""]

        for model in invalid_models:
            with pytest.raises(ValidationError) as exc_info:
                self.validator.validate_model_size(model)
            assert "Invalid model" in str(exc_info.value)

    def test_valid_devices(self):
        """Test validation of valid devices."""
        valid_devices = ["cpu", "cuda"]

        for device in valid_devices:
            result = self.validator.validate_device(device)
            assert result == device

    def test_invalid_devices(self):
        """Test validation of invalid devices."""
        invalid_devices = ["gpu", "opencl", "", "invalid"]

        for device in invalid_devices:
            with pytest.raises(ValidationError) as exc_info:
                self.validator.validate_device(device)
            assert "Invalid device" in str(exc_info.value)

    def test_valid_recording_modes(self):
        """Test validation of valid recording modes."""
        valid_modes = ["push-to-talk", "tap-mode"]

        for mode in valid_modes:
            result = self.validator.validate_recording_mode(mode)
            assert result == mode

    def test_invalid_recording_modes(self):
        """Test validation of invalid recording modes."""
        invalid_modes = ["voice-activation", "continuous", "", "invalid"]

        for mode in invalid_modes:
            with pytest.raises(ValidationError) as exc_info:
                self.validator.validate_recording_mode(mode)
            assert "Invalid recording mode" in str(exc_info.value)

    def test_valid_language_codes(self):
        """Test validation of valid language codes."""
        valid_languages = ["auto", "en", "es", "fr", "de", "zh", "ja"]

        for language in valid_languages:
            result = self.validator.validate_language_code(language)
            assert result == language.lower()

    def test_invalid_language_codes(self):
        """Test validation of invalid language codes."""
        invalid_languages = ["", "a", "invalid", "toolong", "1", "english"]

        for language in invalid_languages:
            with pytest.raises(ValidationError) as exc_info:
                self.validator.validate_language_code(language)
            assert "Invalid language code" in str(exc_info.value)

    def test_valid_audio_device_config(self):
        """Test validation of valid audio device configurations."""
        valid_configs = [
            None,
            {"name": "Test Device"},
            {"name": "Test Device", "hostapi_name": "ALSA"},
        ]

        for config in valid_configs:
            result = self.validator.validate_audio_device_config(config)
            assert result == config

    def test_invalid_audio_device_config(self):
        """Test validation of invalid audio device configurations."""
        invalid_configs = [
            {},  # Missing required 'name' key
            {"hostapi_name": "ALSA"},  # Missing 'name'
            {"name": "Test", "invalid_key": "value"},  # Extra key
            "not_a_dict",  # Wrong type
        ]

        for config in invalid_configs:
            with pytest.raises(ValidationError):
                self.validator.validate_audio_device_config(config)

    def test_get_validation_help(self):
        """Test validation help text generation."""
        help_text = self.validator.get_validation_help("general", "model")
        assert "Valid models:" in help_text
        assert "tiny" in help_text

        help_text = self.validator.get_validation_help("recording", "trigger_key")
        assert "Examples:" in help_text
        assert "<scroll_lock>" in help_text

        # Test unknown field
        help_text = self.validator.get_validation_help("unknown", "field")
        assert "No help available" in help_text
