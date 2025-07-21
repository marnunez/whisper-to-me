"""Test profile manager functionality."""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from whisper_to_me import (
    AdvancedConfig,
    AppConfig,
    ComponentFactory,
    ConfigManager,
    GeneralConfig,
    ProfileManager,
    RecordingConfig,
    SpeechProcessor,
    UIConfig,
)


class TestProfileManager:
    """Test ProfileManager functionality."""

    def setup_method(self):
        """Set up test environment with real config files."""
        # Create temporary directory for config
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_config.toml"

        # Create real ConfigManager with custom config file
        self.config_manager = ConfigManager(config_file=str(self.config_file))

        # Load initial config
        self.default_config = self.config_manager.load_config()

        # Create real ComponentFactory with config
        self.component_factory = ComponentFactory(self.default_config, self.config_manager)

        # Create callback mock
        self.on_config_changed = Mock()

        # Create ProfileManager with real components
        self.manager = ProfileManager(
            self.config_manager, self.component_factory, self.on_config_changed
        )

        # Create test profiles
        self._create_test_profiles()

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_profiles(self):
        """Create test profiles in the config."""
        # Create work profile
        work_config = AppConfig(
            general=GeneralConfig(
                model="large-v3",
                device="cuda",
                language="fr",
                debug=True,
                last_profile="default",
            ),
            recording=RecordingConfig(
                mode="tap-mode",
                trigger_key="<caps_lock>",
                discard_key="delete",
                audio_device=None,
            ),
            ui=UIConfig(use_tray=False),
            advanced=AdvancedConfig(
                sample_rate=22050, chunk_size=1024, vad_filter=False
            ),
            profiles={},
        )
        self.config_manager.create_profile("work", work_config)

        # Create gaming profile
        gaming_config = AppConfig(
            general=GeneralConfig(
                model="tiny",
                device="cpu",
                language="en",
                debug=False,
                last_profile="default",
            ),
            recording=RecordingConfig(
                mode="push-to-talk",
                trigger_key="<f9>",
                discard_key="esc",
                audio_device=None,
            ),
            ui=UIConfig(use_tray=True),
            advanced=AdvancedConfig(
                sample_rate=16000, chunk_size=512, vad_filter=True
            ),
            profiles={},
        )
        self.config_manager.create_profile("gaming", gaming_config)

    def test_init(self):
        """Test ProfileManager initialization."""
        assert self.manager.config_manager == self.config_manager
        assert self.manager.component_factory == self.component_factory
        assert self.manager.on_config_changed == self.on_config_changed
        assert self.manager.current_config is None
        assert self.manager.logger is not None

    def test_switch_profile_success(self):
        """Test successful profile switching."""
        # Mock only the speech processor recreation
        with patch.object(self.component_factory, 'recreate_speech_processor', return_value=None):
            result = self.manager.switch_profile("work")

        # Verify the switch was successful
        assert result.general.model == "large-v3"
        assert result.general.device == "cuda"
        assert result.general.language == "fr"
        assert result.recording.mode == "tap-mode"
        assert result.recording.trigger_key == "<caps_lock>"

        # Verify current profile is updated
        assert self.config_manager.get_current_profile() == "work"
        assert self.manager.current_config == result

        # Verify callback was called
        self.on_config_changed.assert_called_once_with(result)

    def test_switch_profile_invalid_profile(self):
        """Test switching to invalid profile."""
        with pytest.raises(ValueError) as exc_info:
            self.manager.switch_profile("nonexistent")

        assert "Profile 'nonexistent' not found" in str(exc_info.value)
        # Check that available profiles are listed
        assert "Available:" in str(exc_info.value)
        assert "default" in str(exc_info.value)
        assert "gaming" in str(exc_info.value)
        assert "work" in str(exc_info.value)

        # Should not call callback on failure
        self.on_config_changed.assert_not_called()

    def test_switch_profile_with_speech_processor_recreation(self):
        """Test profile switch that requires speech processor recreation."""
        # Create a mock speech processor
        new_speech_processor = Mock(spec=SpeechProcessor)

        # Set speech processor changed callback
        speech_callback = Mock()
        self.manager.set_speech_processor_changed_callback(speech_callback)

        # Mock the recreation to return a new processor
        with patch.object(
            self.component_factory,
            'recreate_speech_processor',
            return_value=new_speech_processor
        ) as mock_recreate:
            self.manager.switch_profile("work")

        # Should recreate speech processor with old and new configs
        mock_recreate.assert_called_once()
        call_args = mock_recreate.call_args[0]
        assert call_args[0].general.model == "large-v3"  # default config
        assert call_args[1].general.model == "large-v3"  # work config

        # Should call speech processor callback
        speech_callback.assert_called_once_with(new_speech_processor)

    def test_switch_profile_no_old_config(self):
        """Test profile switch when no current config exists."""
        self.manager.current_config = None

        with patch.object(self.component_factory, 'recreate_speech_processor', return_value=None):
            self.manager.switch_profile("work")

        # Should load current config for comparison
        assert self.manager.current_config.general.model == "large-v3"

    def test_switch_profile_no_config_changed_callback(self):
        """Test profile switch without config changed callback."""
        # Create manager without callback
        manager = ProfileManager(self.config_manager, self.component_factory, None)

        with patch.object(self.component_factory, 'recreate_speech_processor', return_value=None):
            # Should not raise exception
            result = manager.switch_profile("work")
            assert result.general.model == "large-v3"

    def test_get_current_profile_name(self):
        """Test get_current_profile_name method."""
        # Initially should be default
        assert self.manager.get_current_profile_name() == "default"

        # Switch profile
        with patch.object(self.component_factory, 'recreate_speech_processor', return_value=None):
            self.manager.switch_profile("gaming")

        # Should return new profile
        assert self.manager.get_current_profile_name() == "gaming"

    def test_get_available_profiles(self):
        """Test get_available_profiles method."""
        profiles = self.manager.get_available_profiles()

        # Should include our created profiles plus any defaults
        assert "default" in profiles
        assert "gaming" in profiles
        assert "work" in profiles
        # May also include "quick" and "spanish" from default config

    def test_create_profile_success(self):
        """Test successful profile creation."""
        new_config = AppConfig(
            general=GeneralConfig(
                model="small",
                device="cpu",
                language="de",
                debug=False,
                last_profile="default",
            ),
            recording=RecordingConfig(
                mode="push-to-talk",
                trigger_key="<f12>",
                discard_key="esc",
                audio_device=None,
            ),
            ui=UIConfig(use_tray=True),
            advanced=AdvancedConfig(
                sample_rate=16000, chunk_size=512, vad_filter=True
            ),
            profiles={},
        )

        result = self.manager.create_profile("german", new_config)

        assert result is True
        assert "german" in self.config_manager.get_profile_names()

        # Verify profile was created correctly
        applied = self.config_manager.apply_profile("german")
        assert applied.general.language == "de"
        assert applied.general.model == "small"

    def test_create_profile_exception(self):
        """Test profile creation with exception."""
        # Force an exception by making config file read-only
        os.chmod(self.config_file, 0o444)

        config = self.config_manager.load_config()
        result = self.manager.create_profile("readonly_test", config)

        # Should return False on exception
        assert result is False

        # Restore permissions
        os.chmod(self.config_file, 0o644)

    def test_delete_profile_success(self):
        """Test successful profile deletion."""
        result = self.manager.delete_profile("gaming")

        assert result is True
        assert "gaming" not in self.config_manager.get_profile_names()

    def test_delete_profile_default(self):
        """Test that default profile cannot be deleted."""
        result = self.manager.delete_profile("default")

        assert result is False
        assert "default" in self.config_manager.get_profile_names()

    def test_delete_profile_current_profile(self):
        """Test deleting the current profile."""
        # Switch to gaming profile
        with patch.object(self.component_factory, 'recreate_speech_processor', return_value=None):
            self.manager.switch_profile("gaming")

        # Delete current profile
        result = self.manager.delete_profile("gaming")

        assert result is True
        # Should be back on default
        assert self.manager.get_current_profile_name() == "default"
        # Callback should have been called
        assert self.on_config_changed.call_count == 2  # Once for switch, once for delete

    def test_delete_profile_nonexistent(self):
        """Test deleting non-existent profile."""
        result = self.manager.delete_profile("nonexistent")

        assert result is False

    def test_validate_profile_config_valid(self):
        """Test validating a valid profile configuration."""
        config = self.config_manager.load_config()

        result = self.manager.validate_profile_config(config)

        assert result is True

    def test_validate_profile_config_invalid_key(self):
        """Test validating profile with invalid key combination."""
        config = self.config_manager.load_config()
        config.recording.trigger_key = "invalid_key_format"

        result = self.manager.validate_profile_config(config)

        assert result is False

    def test_get_profile_summary_existing(self):
        """Test getting profile summary for existing profile."""
        summary = self.manager.get_profile_summary("work")

        assert summary is not None
        assert summary["name"] == "work"
        assert summary["model"] == "large-v3"
        assert summary["device"] == "cuda"
        assert summary["language"] == "fr"
        assert summary["mode"] == "tap-mode"
        assert summary["trigger_key"] == "<caps_lock>"

    def test_get_profile_summary_default(self):
        """Test getting profile summary for default profile."""
        summary = self.manager.get_profile_summary("default")

        assert summary is not None
        assert summary["name"] == "default"
        assert summary["model"] == "large-v3"
        assert summary["device"] == "cuda"
        assert summary["language"] == "auto"

    def test_get_profile_summary_nonexistent(self):
        """Test getting profile summary for non-existent profile."""
        # First check what profiles actually exist
        available_profiles = self.manager.get_available_profiles()

        # Use a profile name that definitely doesn't exist
        nonexistent_name = "definitely_nonexistent_profile_12345"
        assert nonexistent_name not in available_profiles

        summary = self.manager.get_profile_summary(nonexistent_name)

        # When profile doesn't exist, apply_profile returns default config
        # So we get a summary with default values
        assert summary is not None
        assert summary["name"] == nonexistent_name
        # Should have default config values
        default_summary = self.manager.get_profile_summary("default")
        assert summary["model"] == default_summary["model"]
        assert summary["device"] == default_summary["device"]
        assert summary["language"] == default_summary["language"]

    def test_profile_lifecycle(self):
        """Test complete profile lifecycle: create, switch, delete."""
        # Create profile
        test_config = AppConfig(
            general=GeneralConfig(
                model="base",
                device="cpu",
                language="ja",
                debug=False,
                last_profile="default",
            ),
            recording=RecordingConfig(
                mode="push-to-talk",
                trigger_key="<f10>",
                discard_key="esc",
                audio_device=None,
            ),
            ui=UIConfig(use_tray=True),
            advanced=AdvancedConfig(
                sample_rate=16000, chunk_size=512, vad_filter=True
            ),
            profiles={},
        )

        create_result = self.manager.create_profile("test_lifecycle", test_config)
        assert create_result is True

        # Switch to profile
        with patch.object(self.component_factory, 'recreate_speech_processor', return_value=None):
            switch_result = self.manager.switch_profile("test_lifecycle")
        assert switch_result.general.language == "ja"

        # Delete profile
        delete_result = self.manager.delete_profile("test_lifecycle")
        assert delete_result is True
        assert "test_lifecycle" not in self.manager.get_available_profiles()

    def test_concurrent_managers(self):
        """Test that multiple ProfileManager instances see the same profiles."""
        # Create another manager with same config file
        another_manager = ProfileManager(
            self.config_manager,
            ComponentFactory(self.default_config, self.config_manager),
            Mock()
        )

        # Both should see same profiles
        manager1_profiles = sorted(self.manager.get_available_profiles())
        manager2_profiles = sorted(another_manager.get_available_profiles())
        assert manager1_profiles == manager2_profiles

        # Create profile with first manager
        config = self.config_manager.load_config()
        config.general.model = "tiny"
        self.manager.create_profile("shared_test", config)

        # Second manager should see it immediately
        assert "shared_test" in another_manager.get_available_profiles()
