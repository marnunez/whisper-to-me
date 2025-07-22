"""
Configuration Differ Module

Generic utility for comparing configuration sections and creating profile diffs.
Eliminates code duplication in configuration management.
"""

from dataclasses import asdict
from typing import Any


class ConfigDiffer[T]:
    """
    Generic configuration differ for comparing and creating section diffs.

    Features:
    - Generic type support for any dataclass configuration section
    - Automatic diff generation between configurations
    - Exclusion of specific fields from comparison
    - Clean API for profile creation workflows
    """

    def __init__(self, exclude_fields: set[str] = None):
        """
        Initialize the configuration differ.

        Args:
            exclude_fields: Set of field names to exclude from diff comparison
        """
        self.exclude_fields = exclude_fields or set()

    def create_diff(
        self, current_config: T, default_config: dict, section_name: str
    ) -> dict[str, Any]:
        """
        Create a diff between current config and defaults for a specific section.

        Args:
            current_config: Current configuration object (dataclass instance)
            default_config: Default configuration dictionary
            section_name: Name of the section in default_config

        Returns:
            Dictionary containing only the changed values
        """
        if section_name not in default_config:
            return {}

        current_dict = asdict(current_config)
        default_section = default_config[section_name]
        diff = {}

        for key, value in current_dict.items():
            # Skip excluded fields
            if key in self.exclude_fields:
                continue

            # Include only values that differ from defaults
            if value != default_section.get(key):
                diff[key] = value

        return diff

    def apply_diff(self, base_config: T, diff: dict[str, Any]) -> None:
        """
        Apply a diff to a configuration object in place.

        Args:
            base_config: Configuration object to modify
            diff: Dictionary of changes to apply
        """
        from whisper_to_me.logger import get_logger

        logger = get_logger()

        for key, value in diff.items():
            if hasattr(base_config, key):
                setattr(base_config, key, value)
            else:
                # Unknown field - log warning but continue
                logger.warning(
                    f"Ignoring unknown profile field '{key}' in {type(base_config).__name__}. "
                    f"This field is no longer supported.",
                    "config",
                )

    def has_changes(
        self, current_config: T, default_config: dict, section_name: str
    ) -> bool:
        """
        Check if current config has any changes from defaults.

        Args:
            current_config: Current configuration object
            default_config: Default configuration dictionary
            section_name: Name of the section in default_config

        Returns:
            True if there are changes from defaults
        """
        diff = self.create_diff(current_config, default_config, section_name)
        return len(diff) > 0

    def merge_configs(self, base_config: T, override_config: dict[str, Any]) -> T:
        """
        Create a new config by merging base with overrides.

        Args:
            base_config: Base configuration object
            override_config: Dictionary of values to override

        Returns:
            New configuration object with overrides applied
        """
        # Create a copy of the base config
        base_dict = asdict(base_config)

        # Apply overrides
        for key, value in override_config.items():
            if key in base_dict:
                base_dict[key] = value

        # Create new instance (assumes dataclass constructor accepts all fields)
        config_type = type(base_config)
        return config_type(**base_dict)


class ConfigSectionDiffer:
    """
    Specialized differ for handling multiple configuration sections.

    Manages different types of configuration sections with appropriate exclusions.
    """

    def __init__(self):
        """Initialize section-specific differs."""
        # General config should exclude last_profile from diffs
        self.general_differ = ConfigDiffer(exclude_fields={"last_profile"})
        self.recording_differ = ConfigDiffer()
        self.ui_differ = ConfigDiffer()
        self.advanced_differ = ConfigDiffer()

    def create_profile_data(
        self, config, default_config: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Create profile data from configuration by diffing against defaults.

        Args:
            config: AppConfig instance with current settings
            default_config: Default configuration dictionary

        Returns:
            Profile data dictionary with only non-default values
        """
        profile_data = {}

        # Compare each section and include only non-default values
        general_diff = self.general_differ.create_diff(
            config.general, default_config, "general"
        )
        if general_diff:
            profile_data["general"] = general_diff

        recording_diff = self.recording_differ.create_diff(
            config.recording, default_config, "recording"
        )
        if recording_diff:
            profile_data["recording"] = recording_diff

        ui_diff = self.ui_differ.create_diff(config.ui, default_config, "ui")
        if ui_diff:
            profile_data["ui"] = ui_diff

        advanced_diff = self.advanced_differ.create_diff(
            config.advanced, default_config, "advanced"
        )
        if advanced_diff:
            profile_data["advanced"] = advanced_diff

        return profile_data

    def apply_profile_data(self, base_config, profile_data: dict[str, Any]) -> None:
        """
        Apply profile data to a configuration object.

        Args:
            base_config: AppConfig instance to modify
            profile_data: Profile data dictionary to apply
        """
        if "general" in profile_data:
            self.general_differ.apply_diff(base_config.general, profile_data["general"])

        if "recording" in profile_data:
            self.recording_differ.apply_diff(
                base_config.recording, profile_data["recording"]
            )

        if "ui" in profile_data:
            self.ui_differ.apply_diff(base_config.ui, profile_data["ui"])

        if "advanced" in profile_data:
            self.advanced_differ.apply_diff(
                base_config.advanced, profile_data["advanced"]
            )
