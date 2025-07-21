"""Test audio exceptions functionality."""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from audio_exceptions import (
    AudioError,
    AudioDeviceNotFoundError,
    AudioDeviceTestError,
    AudioDeviceInitializationError,
    NoAudioDevicesError,
    AudioRecordingError,
    AudioConfigurationError,
    AudioPermissionError,
)


class TestAudioError:
    """Test base AudioError class."""

    def test_audio_error_creation(self):
        """Test basic AudioError creation."""
        error = AudioError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
        assert error.message == "Test error message"

    def test_audio_error_with_device_info(self):
        """Test AudioError with device_info parameter."""
        device_info = {"id": 1, "name": "Test Device"}
        error = AudioError("Main message", device_info)
        assert str(error) == "Main message"
        assert error.device_info == device_info
        assert error.message == "Main message"

    def test_audio_error_without_device_info(self):
        """Test AudioError without device_info parameter."""
        error = AudioError("Main message")
        assert str(error) == "Main message"
        assert error.device_info is None
        assert error.message == "Main message"


class TestAudioDeviceNotFoundError:
    """Test AudioDeviceNotFoundError class."""

    def test_device_not_found_error_with_name_only(self):
        """Test AudioDeviceNotFoundError with device name only."""
        error = AudioDeviceNotFoundError("Test Device")

        expected = "Audio device 'Test Device' not found"
        assert str(error) == expected
        assert error.device_name == "Test Device"
        assert error.device_id is None

    def test_device_not_found_error_with_name_and_id(self):
        """Test AudioDeviceNotFoundError with device name and ID."""
        error = AudioDeviceNotFoundError("Test Device", 5)

        expected = "Audio device 'Test Device' not found (ID: 5)"
        assert str(error) == expected
        assert error.device_name == "Test Device"
        assert error.device_id == 5

    def test_device_not_found_error_inheritance(self):
        """Test AudioDeviceNotFoundError inheritance."""
        error = AudioDeviceNotFoundError("Device")
        assert isinstance(error, AudioError)
        assert isinstance(error, Exception)


class TestAudioDeviceInitializationError:
    """Test AudioDeviceInitializationError class."""

    def test_device_initialization_error_creation(self):
        """Test AudioDeviceInitializationError creation."""
        original_error = Exception("Init failed")
        error = AudioDeviceInitializationError("Test Device", original_error)

        error_str = str(error)
        assert "Test Device" in error_str
        assert "failed to initialize" in error_str.lower()
        assert "Init failed" in error_str

        assert error.device_name == "Test Device"
        assert error.original_error == original_error

    def test_device_initialization_error_inheritance(self):
        """Test AudioDeviceInitializationError inheritance."""
        error = AudioDeviceInitializationError("Device", Exception("error"))
        assert isinstance(error, AudioError)
        assert isinstance(error, Exception)


class TestAudioDeviceTestError:
    """Test AudioDeviceTestError class."""

    def test_device_test_error_creation(self):
        """Test AudioDeviceTestError creation."""
        original_error = Exception("Original error")
        error = AudioDeviceTestError("Test Device", 3, "settings_check", original_error)

        error_str = str(error)
        assert "Test Device" in error_str
        assert "ID: 3" in error_str
        assert "settings_check" in error_str
        assert "Original error" in error_str

        assert error.device_name == "Test Device"
        assert error.device_id == 3
        assert error.test_type == "settings_check"
        assert error.original_error == original_error

    def test_device_test_error_inheritance(self):
        """Test AudioDeviceTestError inheritance."""
        error = AudioDeviceTestError("Device", 1, "test", Exception("error"))
        assert isinstance(error, AudioError)
        assert isinstance(error, Exception)


class TestNoAudioDevicesError:
    """Test NoAudioDevicesError class."""

    def test_no_audio_devices_error_creation(self):
        """Test NoAudioDevicesError creation."""
        error = NoAudioDevicesError()

        error_str = str(error)
        assert "No audio input devices found" in error_str
        assert "check that audio input devices are connected" in error_str.lower()

    def test_no_audio_devices_error_inheritance(self):
        """Test NoAudioDevicesError inheritance."""
        error = NoAudioDevicesError()
        assert isinstance(error, AudioError)
        assert isinstance(error, Exception)


class TestAudioRecordingError:
    """Test AudioRecordingError class."""

    def test_audio_recording_error_creation(self):
        """Test AudioRecordingError creation."""
        original_error = Exception("Recording failed")
        error = AudioRecordingError("start", original_error)

        error_str = str(error)
        assert "Audio recording start failed" in error_str
        assert "Recording failed" in error_str

        assert error.operation == "start"
        assert error.original_error == original_error

    def test_audio_recording_error_inheritance(self):
        """Test AudioRecordingError inheritance."""
        error = AudioRecordingError("stop", Exception("error"))
        assert isinstance(error, AudioError)
        assert isinstance(error, Exception)

    def test_audio_recording_error_different_operations(self):
        """Test AudioRecordingError with different operations."""
        operations = ["start", "stop", "process", "save"]

        for op in operations:
            error = AudioRecordingError(op, Exception("test error"))
            assert f"Audio recording {op} failed" in str(error)
            assert error.operation == op


class TestAudioConfigurationError:
    """Test AudioConfigurationError class."""

    def test_audio_configuration_error_creation(self):
        """Test AudioConfigurationError creation."""
        error = AudioConfigurationError("Invalid config", "Use correct format")

        error_str = str(error)
        assert "Audio configuration error: Invalid config" in error_str
        assert "Suggestion: Use correct format" in error_str

        assert error.config_issue == "Invalid config"
        assert error.suggestion == "Use correct format"

    def test_audio_configuration_error_inheritance(self):
        """Test AudioConfigurationError inheritance."""
        error = AudioConfigurationError("Issue")
        assert isinstance(error, AudioError)
        assert isinstance(error, Exception)

    def test_audio_configuration_error_without_suggestion(self):
        """Test AudioConfigurationError without suggestion."""
        error = AudioConfigurationError("Config problem")

        error_str = str(error)
        assert "Audio configuration error: Config problem" in error_str
        assert "Suggestion:" not in error_str

        assert error.config_issue == "Config problem"
        assert error.suggestion is None


class TestAudioPermissionError:
    """Test AudioPermissionError class."""

    def test_audio_permission_error_creation(self):
        """Test AudioPermissionError creation."""
        error = AudioPermissionError("Microphone Device")

        error_str = str(error)
        assert (
            "Permission denied accessing audio device 'Microphone Device'" in error_str
        )
        assert "microphone permissions" in error_str.lower()

        assert error.device_name == "Microphone Device"

    def test_audio_permission_error_inheritance(self):
        """Test AudioPermissionError inheritance."""
        error = AudioPermissionError("Device")
        assert isinstance(error, AudioError)
        assert isinstance(error, Exception)


class TestExceptionHierarchy:
    """Test exception hierarchy and relationships."""

    def test_all_exceptions_inherit_from_audio_error(self):
        """Test that all audio exceptions inherit from AudioError."""
        exceptions = [
            AudioDeviceNotFoundError("Device"),
            AudioDeviceTestError("Device", 1, "test", Exception()),
            AudioDeviceInitializationError("Device", Exception()),
            NoAudioDevicesError(),
            AudioRecordingError("start", Exception()),
            AudioConfigurationError("Issue"),
            AudioPermissionError("Device"),
        ]

        for exc in exceptions:
            assert isinstance(exc, AudioError)
            assert isinstance(exc, Exception)

    def test_exception_catching(self):
        """Test that specific exceptions can be caught by base AudioError."""
        try:
            raise AudioDeviceNotFoundError("Test Device")
        except AudioError as e:
            assert isinstance(e, AudioDeviceNotFoundError)
            assert "Test Device" in str(e)
        except Exception:
            assert False, "Should have been caught by AudioError"

    def test_exception_chaining(self):
        """Test exception chaining with original errors."""
        original = ValueError("Original problem")

        # Test exceptions that take original errors
        exceptions_with_original = [
            AudioDeviceTestError("Device", 1, "test", original),
            AudioDeviceInitializationError("Device", original),
            AudioRecordingError("start", original),
        ]

        for exc in exceptions_with_original:
            assert exc.original_error == original
            assert str(original) in str(exc)

    def test_error_message_formatting(self):
        """Test that error messages are properly formatted."""
        errors = [
            (AudioDeviceNotFoundError("Test"), "Audio device 'Test' not found"),
            (
                AudioPermissionError("Mic"),
                "Permission denied accessing audio device 'Mic'",
            ),
            (
                AudioConfigurationError("Bad config"),
                "Audio configuration error: Bad config",
            ),
            (
                AudioRecordingError("stop", Exception("Failed")),
                "Audio recording stop failed: Failed",
            ),
        ]

        for error, expected_substring in errors:
            assert expected_substring in str(error)
