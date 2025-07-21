"""
Audio Exceptions Module

Custom exceptions for audio device management and recording operations.
Provides specific error types for better error handling and user feedback.
"""

from typing import Optional


class AudioError(Exception):
    """Base exception for all audio-related errors."""

    def __init__(self, message: str, device_info: Optional[dict] = None):
        """
        Initialize audio error.

        Args:
            message: Error message
            device_info: Optional device information for context
        """
        super().__init__(message)
        self.device_info = device_info
        self.message = message


class AudioDeviceNotFoundError(AudioError):
    """Raised when a specified audio device cannot be found."""

    def __init__(self, device_name: str, device_id: Optional[int] = None):
        """
        Initialize device not found error.

        Args:
            device_name: Name of the device that wasn't found
            device_id: Optional device ID
        """
        message = f"Audio device '{device_name}' not found"
        if device_id is not None:
            message += f" (ID: {device_id})"

        super().__init__(message)
        self.device_name = device_name
        self.device_id = device_id


class AudioDeviceInitializationError(AudioError):
    """Raised when audio device initialization fails."""

    def __init__(self, device_name: str, original_error: Exception):
        """
        Initialize device initialization error.

        Args:
            device_name: Name of the device that failed to initialize
            original_error: The original exception that caused the failure
        """
        message = f"Failed to initialize audio device '{device_name}': {original_error}"
        super().__init__(message)
        self.device_name = device_name
        self.original_error = original_error


class AudioDeviceTestError(AudioError):
    """Raised when audio device testing fails."""

    def __init__(
        self,
        device_name: str,
        device_id: int,
        test_type: str,
        original_error: Exception,
    ):
        """
        Initialize device test error.

        Args:
            device_name: Name of the device that failed testing
            device_id: ID of the device
            test_type: Type of test that failed (e.g., "input_settings", "sample_rate")
            original_error: The original exception that caused the failure
        """
        message = f"Device '{device_name}' (ID: {device_id}) failed {test_type} test: {original_error}"
        super().__init__(message)
        self.device_name = device_name
        self.device_id = device_id
        self.test_type = test_type
        self.original_error = original_error


class NoAudioDevicesError(AudioError):
    """Raised when no audio input devices are available."""

    def __init__(self):
        """Initialize no devices error."""
        super().__init__(
            "No audio input devices found. Please check that audio input devices are connected and enabled."
        )


class AudioRecordingError(AudioError):
    """Raised when audio recording operations fail."""

    def __init__(self, operation: str, original_error: Exception):
        """
        Initialize recording error.

        Args:
            operation: The operation that failed (e.g., "start", "stop", "process")
            original_error: The original exception that caused the failure
        """
        message = f"Audio recording {operation} failed: {original_error}"
        super().__init__(message)
        self.operation = operation
        self.original_error = original_error


class AudioConfigurationError(AudioError):
    """Raised when audio device configuration is invalid."""

    def __init__(self, config_issue: str, suggestion: Optional[str] = None):
        """
        Initialize configuration error.

        Args:
            config_issue: Description of the configuration issue
            suggestion: Optional suggestion for fixing the issue
        """
        message = f"Audio configuration error: {config_issue}"
        if suggestion:
            message += f". Suggestion: {suggestion}"

        super().__init__(message)
        self.config_issue = config_issue
        self.suggestion = suggestion


class AudioPermissionError(AudioError):
    """Raised when audio device access is denied due to permissions."""

    def __init__(self, device_name: str):
        """
        Initialize permission error.

        Args:
            device_name: Name of the device with permission issues
        """
        message = (
            f"Permission denied accessing audio device '{device_name}'. "
            f"Check that the application has microphone permissions."
        )
        super().__init__(message)
        self.device_name = device_name
