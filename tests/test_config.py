"""Test profile switching functionality."""

import tempfile
import shutil
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import ConfigManager, AppConfig, GeneralConfig, RecordingConfig, UIConfig, AdvancedConfig


class TestConfigManager:
    """Test cases for configuration management."""

    def setup_method(self):
        """Set up test environment with temporary config directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_home = os.environ.get("HOME")
        os.environ["HOME"] = self.temp_dir
        self.config_manager = ConfigManager()

    def teardown_method(self):
        """Clean up test environment."""
        if self.original_home:
            os.environ["HOME"] = self.original_home
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_default_config(self):
        """Test loading default configuration."""
        config = self.config_manager.load_config()
        
        assert isinstance(config, AppConfig)
        assert config.general.model == "large-v3"
        assert config.general.device == "cuda"
        assert config.general.language == "auto"
        assert config.recording.mode == "push-to-talk"
        assert config.recording.trigger_key == "scroll_lock"
        assert config.ui.use_tray is True

    def test_profile_creation_and_loading(self):
        """Test creating and loading custom profiles."""
        # Load base config
        config = self.config_manager.load_config()
        
        # Modify settings
        config.general.model = "medium"
        config.general.language = "en"
        config.recording.trigger_key = "caps_lock"
        
        # Create profile
        assert self.config_manager.create_profile("test_profile", config) is True
        
        # Verify profile exists
        profiles = self.config_manager.get_profile_names()
        assert "test_profile" in profiles
        
        # Apply profile
        profile_config = self.config_manager.apply_profile("test_profile")
        assert profile_config.general.model == "medium"
        assert profile_config.general.language == "en"
        assert profile_config.recording.trigger_key == "caps_lock"

    def test_profile_switching(self):
        """Test switching between profiles."""
        # Load and create profiles
        self.config_manager.load_config()
        
        # Create Spanish profile
        spanish_config = AppConfig(
            general=GeneralConfig(model="large-v3", device="cuda", language="es", debug=False, last_profile="default"),
            recording=RecordingConfig(mode="push-to-talk", trigger_key="scroll_lock", discard_key="esc", audio_device=None),
            ui=UIConfig(use_tray=True),
            advanced=AdvancedConfig(sample_rate=16000, chunk_size=512, vad_filter=True),
            profiles={}
        )
        self.config_manager.create_profile("spanish", spanish_config)
        
        # Create work profile
        work_config = AppConfig(
            general=GeneralConfig(model="medium", device="cpu", language="en", debug=True, last_profile="default"),
            recording=RecordingConfig(mode="tap-mode", trigger_key="caps_lock", discard_key="esc", audio_device=None),
            ui=UIConfig(use_tray=False),
            advanced=AdvancedConfig(sample_rate=22050, chunk_size=1024, vad_filter=False),
            profiles={}
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
        assert work_applied.recording.trigger_key == "caps_lock"
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
            general=GeneralConfig(model="tiny", device="cuda", language="auto", debug=False, last_profile="default"),
            recording=config.recording,
            ui=config.ui,
            advanced=config.advanced,
            profiles={}
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
                last_profile="default"
            ),
            recording=RecordingConfig(
                mode="tap-mode",
                trigger_key="caps_lock",
                discard_key="delete",
                audio_device=1
            ),
            ui=UIConfig(use_tray=False),
            advanced=AdvancedConfig(
                sample_rate=44100,
                chunk_size=2048,
                vad_filter=False
            ),
            profiles={}
        )
        
        self.config_manager.create_profile("comprehensive", comprehensive_config)
        
        # Apply and verify all sections
        applied = self.config_manager.apply_profile("comprehensive")
        
        assert applied.general.model == "base"
        assert applied.general.device == "cpu"
        assert applied.general.language == "de"
        assert applied.general.debug is True
        
        assert applied.recording.mode == "tap-mode"
        assert applied.recording.trigger_key == "caps_lock"
        assert applied.recording.discard_key == "delete"
        assert applied.recording.audio_device == 1
        
        assert applied.ui.use_tray is False
        
        assert applied.advanced.sample_rate == 44100
        assert applied.advanced.chunk_size == 2048
        assert applied.advanced.vad_filter is False