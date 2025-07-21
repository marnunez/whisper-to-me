"""Test application state manager functionality."""

from whisper_to_me import ApplicationStateManager


class TestApplicationStateManager:
    """Test cases for application state management."""

    def setup_method(self):
        """Set up test environment."""
        self.state_manager = ApplicationStateManager()

    def test_initial_state(self):
        """Test initial state is correct."""
        assert not self.state_manager.is_recording
        assert self.state_manager.recording_counter == 0
        assert not self.state_manager.trigger_pressed
        assert not self.state_manager.is_shutdown_requested()
        assert self.state_manager.can_start_recording()

    def test_start_recording(self):
        """Test starting recording."""
        result = self.state_manager.start_recording()

        assert result is True
        assert self.state_manager.is_recording
        assert not self.state_manager.can_start_recording()

    def test_start_recording_already_recording(self):
        """Test starting recording when already recording."""
        self.state_manager.start_recording()
        result = self.state_manager.start_recording()

        assert result is False
        assert self.state_manager.is_recording

    def test_stop_recording(self):
        """Test stopping recording."""
        self.state_manager.start_recording()
        result = self.state_manager.stop_recording()

        assert result is True
        assert not self.state_manager.is_recording
        assert not self.state_manager.trigger_pressed
        assert self.state_manager.can_start_recording()

    def test_stop_recording_not_recording(self):
        """Test stopping recording when not recording."""
        result = self.state_manager.stop_recording()

        assert result is False
        assert not self.state_manager.is_recording

    def test_trigger_pressed_state(self):
        """Test trigger pressed state management."""
        assert not self.state_manager.is_trigger_pressed()

        self.state_manager.set_trigger_pressed(True)
        assert self.state_manager.is_trigger_pressed()

        self.state_manager.set_trigger_pressed(False)
        assert not self.state_manager.is_trigger_pressed()

    def test_recording_counter(self):
        """Test recording counter functionality."""
        assert self.state_manager.get_recording_counter() == 0

        counter = self.state_manager.increment_recording_counter()
        assert counter == 1
        assert self.state_manager.get_recording_counter() == 1

        counter = self.state_manager.increment_recording_counter()
        assert counter == 2
        assert self.state_manager.get_recording_counter() == 2

    def test_shutdown_workflow(self):
        """Test shutdown workflow."""
        # Start recording first
        self.state_manager.start_recording()
        assert self.state_manager.can_start_recording() is False

        # Request shutdown
        self.state_manager.request_shutdown()

        assert self.state_manager.is_shutdown_requested()
        assert not self.state_manager.is_recording  # Recording should be stopped
        assert not self.state_manager.can_start_recording()

    def test_state_summary(self):
        """Test state summary generation."""
        summary = self.state_manager.get_state_summary()

        expected_keys = {
            "is_recording",
            "recording_counter",
            "trigger_pressed",
            "shutting_down",
        }
        assert set(summary.keys()) == expected_keys

        # Test with different states
        self.state_manager.start_recording()
        self.state_manager.set_trigger_pressed(True)
        self.state_manager.increment_recording_counter()

        summary = self.state_manager.get_state_summary()
        assert summary["is_recording"] is True
        assert summary["recording_counter"] == 1
        assert summary["trigger_pressed"] is True
        assert summary["shutting_down"] is False

    def test_shutdown_prevents_recording_start(self):
        """Test that shutdown prevents new recordings."""
        self.state_manager.request_shutdown()

        result = self.state_manager.start_recording()
        assert result is False
        assert not self.state_manager.is_recording

    def test_stop_recording_clears_trigger_state(self):
        """Test that stopping recording clears trigger pressed state."""
        self.state_manager.start_recording()
        self.state_manager.set_trigger_pressed(True)

        self.state_manager.stop_recording()

        assert not self.state_manager.is_trigger_pressed()
