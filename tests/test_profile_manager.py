"""Test profile manager functionality."""

from unittest.mock import Mock, patch

import pytest

from whisper_to_me import (
    AdvancedConfig,
    AppConfig,
    GeneralConfig,
    ProfileManager,
    RecordingConfig,
    SpeechProcessor,
    UIConfig,
)


class TestProfileManager:
    """Test ProfileManager functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.config_manager = Mock()
        self.component_factory = Mock()
        self.on_config_changed = Mock()

        # Create test configs
        self.default_config = self._create_test_config("base", "cpu", "en")
        self.work_config = self._create_test_config("large-v3", "cuda", "fr")

        with patch("whisper_to_me.profile_manager.get_logger"):
            self.manager = ProfileManager(
                self.config_manager, self.component_factory, self.on_config_changed
            )

    def _create_test_config(self, model="base", device="cpu", language="en"):
        """Create a test configuration."""
        general = GeneralConfig()
        general.model = model
        general.device = device
        general.language = language

        recording = RecordingConfig()
        recording.trigger_key = "<ctrl>+<shift>+r"
        recording.discard_key = "<esc>"
        recording.mode = "push-to-talk"

        ui = UIConfig()
        advanced = AdvancedConfig()
        profiles = {}

        config = AppConfig(
            general=general,
            recording=recording,
            ui=ui,
            advanced=advanced,
            profiles=profiles,
        )

        return config

    @patch("whisper_to_me.profile_manager.get_logger")
    def test_init(self, mock_get_logger):
        """Test ProfileManager initialization."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        manager = ProfileManager(
            self.config_manager, self.component_factory, self.on_config_changed
        )

        assert manager.config_manager == self.config_manager
        assert manager.component_factory == self.component_factory
        assert manager.on_config_changed == self.on_config_changed
        assert manager.current_config is None
        assert manager.logger == mock_logger

    def test_switch_profile_success(self):
        """Test successful profile switching."""
        # Setup mocks
        self.config_manager.get_profile_names.return_value = [
            "default",
            "work",
            "gaming",
        ]
        self.config_manager.load_config.return_value = self.default_config
        self.config_manager.apply_profile.return_value = self.work_config
        self.component_factory.recreate_speech_processor.return_value = None

        result = self.manager.switch_profile("work")

        # Should validate profile exists
        self.config_manager.get_profile_names.assert_called_once()

        # Should apply profile and update factory
        self.config_manager.apply_profile.assert_called_once_with("work")
        assert self.component_factory.config == self.work_config
        assert self.manager.current_config == self.work_config

        # Should save config and notify callback
        self.config_manager.save_config.assert_called_once()
        self.on_config_changed.assert_called_once_with(self.work_config)

        assert result == self.work_config

    def test_switch_profile_invalid_profile(self):
        """Test switching to invalid profile."""
        self.config_manager.get_profile_names.return_value = ["default", "work"]

        with pytest.raises(ValueError) as exc_info:
            self.manager.switch_profile("nonexistent")

        assert "Profile 'nonexistent' not found" in str(exc_info.value)
        assert "['default', 'work']" in str(exc_info.value)

        # Should not save or notify
        self.config_manager.save_config.assert_not_called()
        self.on_config_changed.assert_not_called()

    def test_switch_profile_with_speech_processor_recreation(self):
        """Test profile switch that requires speech processor recreation."""
        # Setup mocks
        self.config_manager.get_profile_names.return_value = ["default", "work"]
        self.config_manager.load_config.return_value = self.default_config
        self.config_manager.apply_profile.return_value = self.work_config

        new_speech_processor = Mock(spec=SpeechProcessor)
        self.component_factory.recreate_speech_processor.return_value = (
            new_speech_processor
        )

        # Set speech processor changed callback
        speech_callback = Mock()
        self.manager.set_speech_processor_changed_callback(speech_callback)

        self.manager.switch_profile("work")

        # Should recreate speech processor
        self.component_factory.recreate_speech_processor.assert_called_once_with(
            self.default_config, self.work_config
        )

        # Should call speech processor callback
        speech_callback.assert_called_once_with(new_speech_processor)

    def test_switch_profile_no_old_config(self):
        """Test profile switch when no current config exists."""
        self.manager.current_config = None

        self.config_manager.get_profile_names.return_value = ["default", "work"]
        self.config_manager.load_config.return_value = self.default_config
        self.config_manager.apply_profile.return_value = self.work_config
        self.component_factory.recreate_speech_processor.return_value = None

        self.manager.switch_profile("work")

        # Should load current config for comparison
        self.config_manager.load_config.assert_called_once()
        self.component_factory.recreate_speech_processor.assert_called_once_with(
            self.default_config, self.work_config
        )

    def test_switch_profile_no_config_changed_callback(self):
        """Test profile switch without config changed callback."""
        manager = ProfileManager(self.config_manager, self.component_factory, None)

        self.config_manager.get_profile_names.return_value = ["default", "work"]
        self.config_manager.load_config.return_value = self.default_config
        self.config_manager.apply_profile.return_value = self.work_config
        self.component_factory.recreate_speech_processor.return_value = None

        # Should not raise exception
        result = manager.switch_profile("work")
        assert result == self.work_config

    def test_get_current_profile_name(self):
        """Test get_current_profile_name method."""
        self.config_manager.get_current_profile.return_value = "gaming"

        result = self.manager.get_current_profile_name()

        assert result == "gaming"
        self.config_manager.get_current_profile.assert_called_once()

    def test_get_available_profiles(self):
        """Test get_available_profiles method."""
        profiles = ["default", "work", "gaming", "presentation"]
        self.config_manager.get_profile_names.return_value = profiles

        result = self.manager.get_available_profiles()

        assert result == profiles
        self.config_manager.get_profile_names.assert_called_once()

    def test_create_profile_success(self):
        """Test successful profile creation."""
        self.config_manager.create_profile.return_value = True

        result = self.manager.create_profile("new_profile", self.work_config)

        assert result is True
        self.config_manager.create_profile.assert_called_once_with(
            "new_profile", self.work_config
        )

    def test_create_profile_failure(self):
        """Test failed profile creation."""
        self.config_manager.create_profile.return_value = False

        result = self.manager.create_profile("new_profile", self.work_config)

        assert result is False
        self.config_manager.create_profile.assert_called_once_with(
            "new_profile", self.work_config
        )

    def test_create_profile_exception(self):
        """Test profile creation with exception."""
        self.config_manager.create_profile.side_effect = Exception("Creation failed")

        result = self.manager.create_profile("new_profile", self.work_config)

        assert result is False

    def test_delete_profile_success(self):
        """Test successful profile deletion."""
        self.config_manager.delete_profile.return_value = True
        self.config_manager.get_current_profile.return_value = (
            "work"  # Different from deleted
        )

        result = self.manager.delete_profile("gaming")

        assert result is True
        self.config_manager.delete_profile.assert_called_once_with("gaming")

    def test_delete_profile_current_profile_deleted(self):
        """Test deleting the current profile."""
        self.config_manager.delete_profile.return_value = True
        self.config_manager.get_current_profile.return_value = (
            "default"  # Now on default
        )
        self.config_manager.load_config.return_value = self.default_config

        result = self.manager.delete_profile("work")

        assert result is True
        # Should reload config and notify callback
        self.config_manager.load_config.assert_called_once()
        assert self.manager.current_config == self.default_config
        self.on_config_changed.assert_called_once_with(self.default_config)

    def test_delete_profile_default_profile(self):
        """Test trying to delete default profile."""
        result = self.manager.delete_profile("default")

        assert result is False
        self.config_manager.delete_profile.assert_not_called()

    def test_delete_profile_failure(self):
        """Test failed profile deletion."""
        self.config_manager.delete_profile.return_value = False

        result = self.manager.delete_profile("nonexistent")

        assert result is False
        self.config_manager.delete_profile.assert_called_once_with("nonexistent")

    def test_delete_profile_exception(self):
        """Test profile deletion with exception."""
        self.config_manager.delete_profile.side_effect = Exception("Deletion failed")

        result = self.manager.delete_profile("work")

        assert result is False

    def test_set_speech_processor_changed_callback(self):
        """Test setting speech processor changed callback."""
        callback = Mock()
        self.manager.set_speech_processor_changed_callback(callback)

        assert self.manager._speech_processor_changed_callback == callback

    def test_validate_profile_config_success(self):
        """Test successful profile config validation."""
        # Setup config manager to succeed parsing
        self.config_manager.parse_key_combination.return_value = None
        self.config_manager.parse_key_string.return_value = None

        result = self.manager.validate_profile_config(self.work_config)

        assert result is True
        self.config_manager.parse_key_combination.assert_called_once_with(
            "<ctrl>+<shift>+r"
        )
        self.config_manager.parse_key_string.assert_called_once_with("<esc>")

    def test_validate_profile_config_invalid_trigger_key(self):
        """Test profile config validation with invalid trigger key."""
        self.config_manager.parse_key_combination.side_effect = ValueError(
            "Invalid trigger key"
        )

        result = self.manager.validate_profile_config(self.work_config)

        assert result is False

    def test_validate_profile_config_invalid_discard_key(self):
        """Test profile config validation with invalid discard key."""
        self.config_manager.parse_key_combination.return_value = None
        self.config_manager.parse_key_string.side_effect = ValueError(
            "Invalid discard key"
        )

        result = self.manager.validate_profile_config(self.work_config)

        assert result is False

    def test_get_profile_summary_default(self):
        """Test get_profile_summary for default profile."""
        self.config_manager.load_config.return_value = self.default_config

        result = self.manager.get_profile_summary("default")

        expected = {
            "name": "default",
            "model": "base",
            "device": "cpu",
            "language": "en",
            "mode": "push-to-talk",
            "trigger_key": "<ctrl>+<shift>+r",
        }

        assert result == expected
        self.config_manager.load_config.assert_called_once()

    def test_get_profile_summary_named_profile(self):
        """Test get_profile_summary for named profile."""
        self.config_manager.apply_profile.return_value = self.work_config

        result = self.manager.get_profile_summary("work")

        expected = {
            "name": "work",
            "model": "large-v3",
            "device": "cuda",
            "language": "fr",
            "mode": "push-to-talk",
            "trigger_key": "<ctrl>+<shift>+r",
        }

        assert result == expected
        self.config_manager.apply_profile.assert_called_once_with("work")

    def test_get_profile_summary_exception(self):
        """Test get_profile_summary with exception."""
        self.config_manager.apply_profile.side_effect = Exception("Profile not found")

        result = self.manager.get_profile_summary("nonexistent")

        assert result is None

    def test_switch_profile_logging(self):
        """Test that profile switching logs appropriately."""
        mock_logger = Mock()
        self.manager.logger = mock_logger

        # Setup successful switch
        self.config_manager.get_profile_names.return_value = ["default", "work"]
        self.config_manager.load_config.return_value = self.default_config
        self.config_manager.apply_profile.return_value = self.work_config
        self.component_factory.recreate_speech_processor.return_value = None

        self.manager.switch_profile("work")

        # Should log profile switch and success
        mock_logger.profile_switched.assert_called_once_with("work")
        mock_logger.success.assert_called_once_with(
            "Profile switch completed", "profile"
        )

    def test_create_profile_logging_success(self):
        """Test profile creation success logging."""
        mock_logger = Mock()
        self.manager.logger = mock_logger
        self.config_manager.create_profile.return_value = True

        self.manager.create_profile("test", self.work_config)

        mock_logger.success.assert_called_once_with(
            "Profile 'test' created successfully", "profile"
        )

    def test_create_profile_logging_failure(self):
        """Test profile creation failure logging."""
        mock_logger = Mock()
        self.manager.logger = mock_logger
        self.config_manager.create_profile.return_value = False

        self.manager.create_profile("test", self.work_config)

        mock_logger.error.assert_called_once_with(
            "Failed to create profile 'test'", "profile"
        )

    def test_delete_profile_logging_success(self):
        """Test profile deletion success logging."""
        mock_logger = Mock()
        self.manager.logger = mock_logger
        self.config_manager.delete_profile.return_value = True
        self.config_manager.get_current_profile.return_value = "other"

        self.manager.delete_profile("test")

        mock_logger.success.assert_called_once_with(
            "Profile 'test' deleted successfully", "profile"
        )

    def test_delete_profile_logging_not_found(self):
        """Test profile deletion not found logging."""
        mock_logger = Mock()
        self.manager.logger = mock_logger
        self.config_manager.delete_profile.return_value = False

        self.manager.delete_profile("test")

        mock_logger.warning.assert_called_once_with(
            "Profile 'test' not found", "profile"
        )

    def test_delete_profile_logging_default_warning(self):
        """Test trying to delete default profile logging."""
        mock_logger = Mock()
        self.manager.logger = mock_logger

        self.manager.delete_profile("default")

        mock_logger.warning.assert_called_once_with(
            "Cannot delete default profile", "profile"
        )

    def test_validate_profile_config_logging_error(self):
        """Test profile config validation error logging."""
        mock_logger = Mock()
        self.manager.logger = mock_logger
        self.config_manager.parse_key_combination.side_effect = ValueError(
            "Invalid key"
        )

        self.manager.validate_profile_config(self.work_config)

        mock_logger.error.assert_called_once_with(
            "Invalid profile configuration: Invalid key", "profile"
        )

    def test_full_profile_lifecycle(self):
        """Test complete profile management lifecycle."""
        # Create profile
        self.config_manager.create_profile.return_value = True
        create_result = self.manager.create_profile("test_lifecycle", self.work_config)
        assert create_result is True

        # List profiles (should include new one)
        self.config_manager.get_profile_names.return_value = [
            "default",
            "test_lifecycle",
        ]
        profiles = self.manager.get_available_profiles()
        assert "test_lifecycle" in profiles

        # Get profile summary
        self.config_manager.apply_profile.return_value = self.work_config
        summary = self.manager.get_profile_summary("test_lifecycle")
        assert summary["name"] == "test_lifecycle"

        # Switch to profile
        self.config_manager.load_config.return_value = self.default_config
        self.component_factory.recreate_speech_processor.return_value = None
        switch_result = self.manager.switch_profile("test_lifecycle")
        assert switch_result == self.work_config

        # Validate profile config
        self.config_manager.parse_key_combination.return_value = None
        self.config_manager.parse_key_string.return_value = None
        validation = self.manager.validate_profile_config(self.work_config)
        assert validation is True

        # Delete profile
        self.config_manager.delete_profile.return_value = True
        self.config_manager.get_current_profile.return_value = "default"
        delete_result = self.manager.delete_profile("test_lifecycle")
        assert delete_result is True

    def test_no_callback_scenario(self):
        """Test operations when callbacks are not set."""
        manager = ProfileManager(self.config_manager, self.component_factory)

        # Should not raise exceptions when callbacks are None
        self.config_manager.get_profile_names.return_value = ["default", "work"]
        self.config_manager.load_config.return_value = self.default_config
        self.config_manager.apply_profile.return_value = self.work_config
        self.component_factory.recreate_speech_processor.return_value = None

        result = manager.switch_profile("work")
        assert result == self.work_config

        # Test profile deletion that triggers fallback to default
        self.config_manager.delete_profile.return_value = True
        self.config_manager.get_current_profile.return_value = "default"

        delete_result = manager.delete_profile("work")
        assert delete_result is True
