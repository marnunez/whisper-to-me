"""Test profile switching functionality."""

import os
import shutil
import tempfile
from pathlib import Path

from whisper_to_me import (
    AdvancedConfig,
    AppConfig,
    ConfigManager,
    GeneralConfig,
    RecordingConfig,
    UIConfig,
)


class TestConfigManager:
    """Test cases for configuration management."""

    def setup_method(self):
        """Set up test environment with temporary config directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_home = os.environ.get("HOME")
        self.original_xdg_config_home = os.environ.get("XDG_CONFIG_HOME")

        # Clear XDG_CONFIG_HOME to ensure we use HOME/.config
        if "XDG_CONFIG_HOME" in os.environ:
            del os.environ["XDG_CONFIG_HOME"]

        os.environ["HOME"] = self.temp_dir
        self.config_manager = ConfigManager()

    def teardown_method(self):
        """Clean up test environment."""
        if self.original_home:
            os.environ["HOME"] = self.original_home

        # Restore XDG_CONFIG_HOME if it was set
        if self.original_xdg_config_home:
            os.environ["XDG_CONFIG_HOME"] = self.original_xdg_config_home
        elif "XDG_CONFIG_HOME" in os.environ:
            del os.environ["XDG_CONFIG_HOME"]

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_default_config(self):
        """Test loading default configuration."""
        config = self.config_manager.load_config()

        assert isinstance(config, AppConfig)
        assert config.general.model == "large-v3"
        assert config.general.device == "cuda"
        assert config.general.language == "auto"
        assert config.recording.mode == "push-to-talk"
        assert config.recording.trigger_key == "<scroll_lock>"
        assert config.ui.use_tray is True
        assert config.advanced.initial_prompt == ""

    def test_profile_creation_and_loading(self):
        """Test creating and loading custom profiles."""
        # Load base config
        config = self.config_manager.load_config()

        # Modify settings
        config.general.model = "medium"
        config.general.language = "en"
        config.recording.trigger_key = "<caps_lock>"
        config.advanced.initial_prompt = "Test prompt"

        # Create profile
        assert self.config_manager.create_profile("test_profile", config) is True

        # Verify profile exists
        profiles = self.config_manager.get_profile_names()
        assert "test_profile" in profiles

        # Apply profile
        profile_config = self.config_manager.apply_profile("test_profile")
        assert profile_config.general.model == "medium"
        assert profile_config.general.language == "en"
        assert profile_config.recording.trigger_key == "<caps_lock>"
        assert profile_config.advanced.initial_prompt == "Test prompt"

    def test_profile_switching(self):
        """Test switching between profiles."""
        # Load and create profiles
        self.config_manager.load_config()

        # Create Spanish profile
        spanish_config = AppConfig(
            general=GeneralConfig(
                model="large-v3",
                device="cuda",
                language="es",
                debug=False,
                last_profile="default",
            ),
            recording=RecordingConfig(
                mode="push-to-talk",
                trigger_key="<scroll_lock>",
                discard_key="esc",
                audio_device=None,
            ),
            ui=UIConfig(use_tray=True),
            advanced=AdvancedConfig(chunk_size=512, vad_filter=True, initial_prompt=""),
            profiles={},
        )
        self.config_manager.create_profile("spanish", spanish_config)

        # Create work profile
        work_config = AppConfig(
            general=GeneralConfig(
                model="medium",
                device="cpu",
                language="en",
                debug=True,
                last_profile="default",
            ),
            recording=RecordingConfig(
                mode="tap-mode",
                trigger_key="<caps_lock>",
                discard_key="esc",
                audio_device=None,
            ),
            ui=UIConfig(use_tray=False),
            advanced=AdvancedConfig(
                chunk_size=1024, vad_filter=False, initial_prompt=""
            ),
            profiles={},
        )
        self.config_manager.create_profile("work", work_config)

        # Test switching to Spanish profile
        spanish_applied = self.config_manager.apply_profile("spanish")
        assert spanish_applied.general.language == "es"
        assert self.config_manager.get_current_profile() == "spanish"

        # Test switching to work profile
        work_applied = self.config_manager.apply_profile("work")
        assert work_applied.general.model == "medium"
        assert work_applied.general.device == "cpu"
        assert work_applied.recording.mode == "tap-mode"
        assert work_applied.recording.trigger_key == "<caps_lock>"
        assert work_applied.ui.use_tray is False
        assert self.config_manager.get_current_profile() == "work"

        # Test switching back to default
        default_applied = self.config_manager.apply_profile("default")
        assert default_applied.general.language == "auto"
        assert default_applied.general.model == "large-v3"
        assert self.config_manager.get_current_profile() == "default"

    def test_profile_deletion(self):
        """Test deleting profiles."""
        # Load config and create test profile
        config = self.config_manager.load_config()
        config.general.model = "tiny"
        self.config_manager.create_profile("temp_profile", config)

        # Verify profile exists
        profiles = self.config_manager.get_profile_names()
        assert "temp_profile" in profiles

        # Delete profile
        assert self.config_manager.delete_profile("temp_profile") is True

        # Verify profile is gone
        profiles = self.config_manager.get_profile_names()
        assert "temp_profile" not in profiles

        # Test deleting non-existent profile
        assert self.config_manager.delete_profile("non_existent") is False

        # Test cannot delete default profile
        assert self.config_manager.delete_profile("default") is False

    def test_profile_inheritance(self):
        """Test that profiles inherit from base config."""
        # Load base config
        config = self.config_manager.load_config()

        # Create profile that only changes model
        minimal_config = AppConfig(
            general=GeneralConfig(
                model="tiny",
                device="cuda",
                language="auto",
                debug=False,
                last_profile="default",
            ),
            recording=config.recording,
            ui=config.ui,
            advanced=config.advanced,
            profiles={},
        )
        self.config_manager.create_profile("minimal", minimal_config)

        # Apply profile
        applied = self.config_manager.apply_profile("minimal")

        # Only model should be different
        assert applied.general.model == "tiny"
        assert applied.general.device == "cuda"  # inherited
        assert applied.general.language == "auto"  # inherited
        assert applied.recording.mode == "push-to-talk"  # inherited
        assert applied.ui.use_tray is True  # inherited

    def test_invalid_profile_fallback(self):
        """Test that invalid profile names fall back to default."""
        self.config_manager.load_config()

        # Try to apply non-existent profile
        result = self.config_manager.apply_profile("non_existent")

        # Should return default config
        assert result.general.model == "large-v3"
        assert self.config_manager.get_current_profile() == "default"

    def test_config_persistence(self):
        """Test that profiles persist across manager instances."""
        # Create profile with first manager
        config = self.config_manager.load_config()
        config.general.model = "medium"
        config.general.language = "fr"
        self.config_manager.create_profile("persistent", config)

        # Create new manager instance
        new_manager = ConfigManager()
        new_manager.load_config()

        # Profile should exist
        profiles = new_manager.get_profile_names()
        assert "persistent" in profiles

        # Apply profile
        applied = new_manager.apply_profile("persistent")
        assert applied.general.model == "medium"
        assert applied.general.language == "fr"

    def test_current_profile_switching_updates_last_profile(self):
        """Test that switching profiles updates the last_profile setting."""
        # Load config and create profile
        config = self.config_manager.load_config()
        config.general.model = "small"
        self.config_manager.create_profile("test_last", config)

        # Switch to profile
        applied = self.config_manager.apply_profile("test_last")

        # last_profile should be updated
        assert applied.general.last_profile == "test_last"

        # Switch back to default
        default_applied = self.config_manager.apply_profile("default")
        assert default_applied.general.last_profile == "default"

    def test_profile_with_all_sections(self):
        """Test profile that modifies all configuration sections."""
        self.config_manager.load_config()

        # Create comprehensive profile
        comprehensive_config = AppConfig(
            general=GeneralConfig(
                model="base",
                device="cpu",
                language="de",
                debug=True,
                last_profile="default",
            ),
            recording=RecordingConfig(
                mode="tap-mode",
                trigger_key="<caps_lock>",
                discard_key="delete",
                audio_device=1,
            ),
            ui=UIConfig(use_tray=False),
            advanced=AdvancedConfig(
                chunk_size=2048, vad_filter=False, initial_prompt=""
            ),
            profiles={},
        )

        self.config_manager.create_profile("comprehensive", comprehensive_config)

        # Apply and verify all sections
        applied = self.config_manager.apply_profile("comprehensive")

        assert applied.general.model == "base"
        assert applied.general.device == "cpu"
        assert applied.general.language == "de"
        assert applied.general.debug is True

        assert applied.recording.mode == "tap-mode"
        assert applied.recording.trigger_key == "<caps_lock>"
        assert applied.recording.discard_key == "delete"
        assert applied.recording.audio_device == 1

        assert applied.ui.use_tray is False

        # sample_rate was removed from config
        assert applied.advanced.chunk_size == 2048
        assert applied.advanced.vad_filter is False

    def test_custom_config_file_location(self):
        """Test using a custom config file location."""
        # Create a custom config file path
        custom_config_path = Path(self.temp_dir) / "custom" / "config.toml"

        # Create ConfigManager with custom path
        custom_manager = ConfigManager(config_file=str(custom_config_path))

        # Verify the config file is created at custom location
        assert custom_manager.config_file == custom_config_path
        assert custom_manager.config_dir == custom_config_path.parent

        # Load config to trigger file creation
        config = custom_manager.load_config()
        assert custom_config_path.exists()

        # Create a profile
        config.general.model = "tiny"
        custom_manager.create_profile("custom_profile", config)

        # Create new manager with same custom path and verify profile exists
        another_manager = ConfigManager(config_file=str(custom_config_path))
        profiles = another_manager.get_profile_names()
        assert "custom_profile" in profiles

    def test_xdg_config_home_support(self):
        """Test that XDG_CONFIG_HOME is respected when no custom config is specified."""
        # Set XDG_CONFIG_HOME
        xdg_dir = Path(self.temp_dir) / "xdg_config"
        os.environ["XDG_CONFIG_HOME"] = str(xdg_dir)

        try:
            # Create new ConfigManager
            xdg_manager = ConfigManager()

            # Verify it uses XDG_CONFIG_HOME
            expected_path = xdg_dir / "whisper-to-me" / "config.toml"
            assert xdg_manager.config_file == expected_path

            # Load config to create the file
            xdg_manager.load_config()
            assert expected_path.exists()
        finally:
            # Clean up XDG_CONFIG_HOME
            if "XDG_CONFIG_HOME" in os.environ:
                del os.environ["XDG_CONFIG_HOME"]

    def test_profile_creation_validation(self):
        """Test profile creation with various configurations."""
        self.config_manager.load_config()

        # Test creating profile with minimal changes
        minimal = AppConfig(
            general=GeneralConfig(
                model="tiny",  # Only change model
                device="cuda",
                language="auto",
                debug=False,
                last_profile="default",
            ),
            recording=RecordingConfig(
                mode="push-to-talk",
                trigger_key="<scroll_lock>",
                discard_key="esc",
                audio_device=None,
            ),
            ui=UIConfig(use_tray=True),
            advanced=AdvancedConfig(chunk_size=512, vad_filter=True, initial_prompt=""),
            profiles={},
        )

        assert self.config_manager.create_profile("minimal_change", minimal) is True

        # Verify the profile only stores the difference
        config_dict = self.config_manager._config.profiles["minimal_change"]
        # Should have the general section with model change
        assert "general" in config_dict
        assert config_dict["general"]["model"] == "tiny"
        # The profile stores minimal differences from the base config

    def test_profile_creation_with_empty_name(self):
        """Test that profile creation handles edge cases."""
        config = self.config_manager.load_config()

        # Test with empty name (should still work, though not recommended)
        assert self.config_manager.create_profile("", config) is True
        assert "" in self.config_manager.get_profile_names()

        # Test with whitespace name
        assert self.config_manager.create_profile("   ", config) is True
        assert "   " in self.config_manager.get_profile_names()

    def test_profile_overwrite(self):
        """Test that creating a profile with existing name overwrites it."""
        config = self.config_manager.load_config()

        # Create initial profile
        config.general.model = "small"
        assert self.config_manager.create_profile("overwrite_test", config) is True

        # Apply and verify
        applied = self.config_manager.apply_profile("overwrite_test")
        assert applied.general.model == "small"

        # Overwrite with different config
        config.general.model = "large-v3"
        config.general.language = "es"
        assert self.config_manager.create_profile("overwrite_test", config) is True

        # Apply and verify it was overwritten
        applied = self.config_manager.apply_profile("overwrite_test")
        assert applied.general.model == "large-v3"
        assert applied.general.language == "es"

    def test_profile_with_special_characters(self):
        """Test profile names with special characters."""
        config = self.config_manager.load_config()

        # Test various special character profile names
        special_names = [
            "profile-with-dashes",
            "profile_with_underscores",
            "profile.with.dots",
            "profile@work",
            "profile#1",
            "profile (home)",
            "profile[test]",
            "über-profile",
            "профиль",  # Cyrillic
            "プロファイル",  # Japanese
        ]

        for name in special_names:
            config.general.model = "tiny"
            assert self.config_manager.create_profile(name, config) is True
            assert name in self.config_manager.get_profile_names()

            # Verify we can apply the profile
            applied = self.config_manager.apply_profile(name)
            assert applied.general.model == "tiny"

    def test_load_config_with_unknown_fields(self):
        """Test loading config with deprecated/unknown fields shows warnings."""
        # Create a config file with unknown fields
        config_dict = {
            "general": {
                "model": "tiny",
                "device": "cpu",
                "language": "en",
                "debug": False,
                "last_profile": "default",
                "trailing_space": False,
                "unknown_field": "should be ignored",
            },
            "recording": {
                "mode": "push-to-talk",
                "trigger_key": "<scroll_lock>",
                "discard_key": "<esc>",
                "audio_device": None,
                "deprecated_option": True,
            },
            "ui": {
                "use_tray": True,
                "future_feature": "not yet implemented",
            },
            "advanced": {
                "chunk_size": 512,
                "vad_filter": True,
                "initial_prompt": "",
                "sample_rate": 16000,  # This was removed
                "old_setting": "deprecated",
            },
            "profiles": {},
        }

        # Write the config with unknown fields
        self.config_manager._save_config_to_file(config_dict)

        # Capture log output
        from unittest.mock import patch

        with patch.object(self.config_manager.logger, "warning") as mock_warning:
            # Load the config - should succeed despite unknown fields
            config = self.config_manager.load_config()

            # Verify the config loaded correctly with known fields
            assert config.general.model == "tiny"
            assert config.general.device == "cpu"
            assert config.recording.mode == "push-to-talk"
            assert config.ui.use_tray is True
            assert config.advanced.chunk_size == 512

            # Verify warnings were logged for unknown fields
            warning_calls = [call[0][0] for call in mock_warning.call_args_list]
            assert any("unknown_field" in msg for msg in warning_calls)
            assert any("deprecated_option" in msg for msg in warning_calls)
            assert any("future_feature" in msg for msg in warning_calls)
            assert any("sample_rate" in msg for msg in warning_calls)
            assert any("old_setting" in msg for msg in warning_calls)

    def test_profile_with_unknown_fields(self):
        """Test applying profile with deprecated/unknown fields shows warnings."""
        # First create a valid config
        self.config_manager.load_config()

        # Create a profile with some unknown fields in the saved data
        profile_data = {
            "general": {
                "model": "large-v3",
                "unknown_general_field": "test",
            },
            "advanced": {
                "vad_filter": False,
                "sample_rate": 44100,  # Deprecated field
            },
        }

        # Manually add the profile with unknown fields
        self.config_manager._config.profiles["test_unknown"] = profile_data
        self.config_manager.save_config()

        # Apply the profile and capture warnings
        from unittest.mock import patch

        with patch.object(self.config_manager.logger, "warning") as mock_warning:
            applied = self.config_manager.apply_profile("test_unknown")

            # Verify known fields were applied
            assert applied.general.model == "large-v3"
            assert applied.advanced.vad_filter is False

            # Verify warnings were logged for unknown fields
            warning_calls = [call[0][0] for call in mock_warning.call_args_list]
            # Note: These warnings come from config_differ.py
            assert any("unknown_general_field" in msg for msg in warning_calls)
            assert any("sample_rate" in msg for msg in warning_calls)
