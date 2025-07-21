"""
Application State Manager Module

Manages application state including recording status, counters, and state transitions.
Extracted from the main WhisperToMe class to improve separation of concerns.
"""


class ApplicationStateManager:
    """
    Manages application state for recording and transcription workflow.

    Features:
    - Recording state tracking
    - Recording counter management
    - State validation
    - Thread-safe state transitions
    """

    def __init__(self):
        """Initialize the application state manager."""
        self.is_recording = False
        self.recording_counter = 0
        self.trigger_pressed = False
        self._shutting_down = False

    def start_recording(self) -> bool:
        """
        Start recording state.

        Returns:
            True if recording started, False if already recording
        """
        if self.is_recording or self._shutting_down:
            return False

        self.is_recording = True
        return True

    def stop_recording(self) -> bool:
        """
        Stop recording state.

        Returns:
            True if recording stopped, False if not recording
        """
        if not self.is_recording:
            return False

        self.is_recording = False
        self.trigger_pressed = False
        return True

    def set_trigger_pressed(self, pressed: bool) -> None:
        """Set the trigger key pressed state."""
        self.trigger_pressed = pressed

    def is_trigger_pressed(self) -> bool:
        """Check if trigger key is currently pressed."""
        return self.trigger_pressed

    def increment_recording_counter(self) -> int:
        """
        Increment and return the recording counter.

        Returns:
            New counter value
        """
        self.recording_counter += 1
        return self.recording_counter

    def get_recording_counter(self) -> int:
        """Get current recording counter value."""
        return self.recording_counter

    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutting_down

    def request_shutdown(self) -> None:
        """Request application shutdown."""
        self._shutting_down = True
        self.is_recording = False

    def can_start_recording(self) -> bool:
        """
        Check if recording can be started.

        Returns:
            True if recording can start
        """
        return not self.is_recording and not self._shutting_down

    def get_state_summary(self) -> dict:
        """
        Get current state summary for debugging.

        Returns:
            Dictionary with current state information
        """
        return {
            "is_recording": self.is_recording,
            "recording_counter": self.recording_counter,
            "trigger_pressed": self.trigger_pressed,
            "shutting_down": self._shutting_down,
        }
