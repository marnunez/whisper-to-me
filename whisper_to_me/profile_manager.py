"""
Profile Manager Module

Handles profile switching logic and configuration updates for different use cases.
Extracted from the main application to improve separation of concerns.
"""

from typing import Callable, Optional
from whisper_to_me.config import ConfigManager, AppConfig
from whisper_to_me.speech_processor import SpeechProcessor
from whisper_to_me.component_factory import ComponentFactory
from whisper_to_me.logger import get_logger


class ProfileManager:
    """
    Manages profile switching and related configuration updates.

    Features:
    - Profile validation and switching
    - Component recreation when needed
    - Configuration persistence
    - Error handling for profile operations
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        component_factory: ComponentFactory,
        on_config_changed: Optional[Callable[[AppConfig], None]] = None,
    ):
        """
        Initialize the profile manager.

        Args:
            config_manager: Configuration manager for profile operations
            component_factory: Factory for recreating components
            on_config_changed: Callback when configuration changes
        """
        self.config_manager = config_manager
        self.component_factory = component_factory
        self.on_config_changed = on_config_changed
        self.current_config: Optional[AppConfig] = None
        self.logger = get_logger()

    def switch_profile(self, profile_name: str) -> AppConfig:
        """
        Switch to a different profile and update configuration.

        Args:
            profile_name: Name of the profile to switch to

        Returns:
            New configuration after profile switch

        Raises:
            ValueError: If profile name is invalid
        """
        self.logger.profile_switched(profile_name)

        # Validate profile exists
        available_profiles = self.config_manager.get_profile_names()
        if profile_name not in available_profiles:
            raise ValueError(
                f"Profile '{profile_name}' not found. Available: {available_profiles}"
            )

        # Store old config for comparison
        old_config = self.current_config or self.config_manager.load_config()

        # Apply the new profile
        new_config = self.config_manager.apply_profile(profile_name)

        # Update component factory with new config
        self.component_factory.config = new_config
        self.current_config = new_config

        # Check if speech processor needs recreation
        new_speech_processor = self.component_factory.recreate_speech_processor(
            old_config, new_config
        )
        if new_speech_processor is not None:
            # Notify caller that speech processor changed
            if hasattr(self, "_speech_processor_changed_callback"):
                self._speech_processor_changed_callback(new_speech_processor)

        # Save the profile switch
        self.config_manager.save_config()

        # Notify about config change
        if self.on_config_changed:
            self.on_config_changed(new_config)

        self.logger.success("Profile switch completed", "profile")
        return new_config

    def get_current_profile_name(self) -> str:
        """
        Get the name of the current active profile.

        Returns:
            Current profile name
        """
        return self.config_manager.get_current_profile()

    def get_available_profiles(self) -> list[str]:
        """
        Get list of available profile names.

        Returns:
            List of profile names
        """
        return self.config_manager.get_profile_names()

    def create_profile(self, name: str, config: AppConfig) -> bool:
        """
        Create a new profile from the given configuration.

        Args:
            name: Name for the new profile
            config: Configuration to save as profile

        Returns:
            True if profile was created successfully
        """
        try:
            success = self.config_manager.create_profile(name, config)
            if success:
                self.logger.success(f"Profile '{name}' created successfully", "profile")
            else:
                self.logger.error(f"Failed to create profile '{name}'", "profile")
            return success
        except Exception as e:
            self.logger.error(f"Error creating profile '{name}': {e}", "profile")
            return False

    def delete_profile(self, name: str) -> bool:
        """
        Delete a profile.

        Args:
            name: Name of the profile to delete

        Returns:
            True if profile was deleted successfully
        """
        if name == "default":
            self.logger.warning("Cannot delete default profile", "profile")
            return False

        try:
            success = self.config_manager.delete_profile(name)
            if success:
                self.logger.success(f"Profile '{name}' deleted successfully", "profile")

                # If we deleted the current profile, we're now on default
                if self.get_current_profile_name() == "default":
                    self.current_config = self.config_manager.load_config()
                    if self.on_config_changed:
                        self.on_config_changed(self.current_config)
            else:
                self.logger.warning(f"Profile '{name}' not found", "profile")
            return success
        except Exception as e:
            self.logger.error(f"Error deleting profile '{name}': {e}", "profile")
            return False

    def set_speech_processor_changed_callback(
        self, callback: Callable[[SpeechProcessor], None]
    ) -> None:
        """
        Set callback for when speech processor needs to be updated.

        Args:
            callback: Function to call with new SpeechProcessor
        """
        self._speech_processor_changed_callback = callback

    def validate_profile_config(self, config: AppConfig) -> bool:
        """
        Validate a configuration for profile creation.

        Args:
            config: Configuration to validate

        Returns:
            True if configuration is valid
        """
        try:
            # Try to parse key combinations
            self.config_manager.parse_key_combination(config.recording.trigger_key)
            self.config_manager.parse_key_string(config.recording.discard_key)
            return True
        except ValueError as e:
            self.logger.error(f"Invalid profile configuration: {e}", "profile")
            return False

    def get_profile_summary(self, profile_name: str) -> Optional[dict]:
        """
        Get a summary of profile settings for display.

        Args:
            profile_name: Name of the profile

        Returns:
            Dictionary with profile information or None if not found
        """
        try:
            if profile_name == "default":
                config = self.config_manager.load_config()
            else:
                config = self.config_manager.apply_profile(profile_name)

            return {
                "name": profile_name,
                "model": config.general.model,
                "device": config.general.device,
                "language": config.general.language,
                "mode": config.recording.mode,
                "trigger_key": config.recording.trigger_key,
            }
        except Exception:
            return None
