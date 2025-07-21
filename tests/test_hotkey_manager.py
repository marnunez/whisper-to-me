"""Test hotkey manager functionality."""

from unittest.mock import Mock

import pytest
from pynput import keyboard

from whisper_to_me import AppConfig, HotkeyManager, RecordingConfig


class TestHotkeyManager:
    """Test HotkeyManager functionality."""

    def setup_method(self):
        """Set up test environment."""
        # Create test config
        self.recording_config = RecordingConfig()
        self.recording_config.trigger_key = "<ctrl>+<shift>+r"
        self.recording_config.discard_key = "<esc>"
        self.recording_config.mode = "push-to-talk"

        self.config = Mock(spec=AppConfig)
        self.config.recording = self.recording_config

    def test_init_push_to_talk_mode(self, mock_keyboard_hooks):
        """Test HotkeyManager initialization in push-to-talk mode."""
        # Use real key parsing
        from pynput import keyboard

        mock_keyboard_hooks.HotKey.parse = keyboard.HotKey.parse

        manager = HotkeyManager(self.config)

        assert manager.config == self.config
        assert manager.listener is None
        assert manager.trigger_hotkey is not None
        assert manager.discard_hotkey is None  # Not used in push-to-talk

        # Verify hotkey was created with correct parsed keys
        expected_trigger_keys = keyboard.HotKey.parse("<ctrl>+<shift>+r")
        mock_keyboard_hooks.HotKey.assert_called_once_with(
            expected_trigger_keys, manager._handle_trigger_press
        )

    def test_init_tap_mode(self, mock_keyboard_hooks):
        """Test HotkeyManager initialization in tap mode."""
        self.config.recording.mode = "tap-mode"

        # Use real key parsing
        from pynput import keyboard

        mock_keyboard_hooks.HotKey.parse = keyboard.HotKey.parse

        # Track created hotkeys
        created_hotkeys = []

        def track_hotkey(*args, **kwargs):
            hotkey = Mock()
            created_hotkeys.append((args, kwargs))
            return hotkey

        mock_keyboard_hooks.HotKey.side_effect = track_hotkey

        manager = HotkeyManager(self.config)

        assert manager.trigger_hotkey is not None
        assert manager.discard_hotkey is not None

        # Should create both hotkeys with correct parsed keys
        assert len(created_hotkeys) == 2

        # Check trigger hotkey
        trigger_keys = keyboard.HotKey.parse("<ctrl>+<shift>+r")
        assert created_hotkeys[0][0][0] == trigger_keys

        # Check discard hotkey
        discard_keys = keyboard.HotKey.parse("<esc>")
        assert created_hotkeys[1][0][0] == discard_keys

    def test_real_key_parsing(self):
        """Test that real key parsing works correctly."""
        # Test various key combinations with real parser
        test_cases = [
            ("<ctrl>+a", ["ctrl", "a"]),
            ("<shift>+<f1>", ["shift", "f1"]),
            ("<cmd>+<space>", ["cmd", "space"]),
            ("x", ["x"]),
            ("<esc>", ["esc"]),
        ]

        for key_string, expected_keys in test_cases:
            parsed = keyboard.HotKey.parse(key_string)
            # The parser returns a list of key objects
            assert len(parsed) == len(expected_keys), f"Failed parsing {key_string}"

    def test_set_callbacks(self, mock_keyboard_hooks):
        """Test setting callback functions."""
        manager = HotkeyManager(self.config)

        # Mock callback functions
        on_trigger_press = Mock()
        on_trigger_tap = Mock()
        on_discard_tap = Mock()
        on_trigger_release = Mock()

        manager.set_callbacks(
            on_trigger_press=on_trigger_press,
            on_trigger_tap=on_trigger_tap,
            on_discard_tap=on_discard_tap,
            on_trigger_release=on_trigger_release,
        )

        assert manager.on_trigger_press == on_trigger_press
        assert manager.on_trigger_tap == on_trigger_tap
        assert manager.on_discard_tap == on_discard_tap
        assert manager.on_trigger_release == on_trigger_release

    def test_start_listening(self, mock_keyboard_hooks):
        """Test starting the keyboard listener."""
        manager = HotkeyManager(self.config)
        manager.start_listening()

        # Should create and start listener
        mock_keyboard_hooks.Listener.assert_called_once()
        listener_instance = mock_keyboard_hooks.Listener.return_value
        listener_instance.start.assert_called_once()
        assert manager.listener == listener_instance

    def test_stop_listening(self, mock_keyboard_hooks):
        """Test stopping the keyboard listener."""
        manager = HotkeyManager(self.config)
        manager.start_listening()

        # Now stop it
        manager.stop_listening()

        listener_instance = mock_keyboard_hooks.Listener.return_value
        listener_instance.stop.assert_called_once()
        assert manager.listener is None

    def test_stop_listening_no_active_listener(self, mock_keyboard_hooks):
        """Test stopping when no listener is active."""
        manager = HotkeyManager(self.config)

        # Should not raise exception
        manager.stop_listening()
        assert manager.listener is None

    def test_handle_trigger_press_with_callback(self, mock_keyboard_hooks):
        """Test handling trigger press with callback."""
        manager = HotkeyManager(self.config)

        # Set callback
        on_press = Mock()
        manager.set_callbacks(on_trigger_press=on_press)

        # Simulate trigger press
        manager._handle_trigger_press()

        on_press.assert_called_once()

    def test_handle_trigger_press_no_callback(self, mock_keyboard_hooks):
        """Test handling trigger press without callback."""
        manager = HotkeyManager(self.config)

        # Should not raise exception
        manager._handle_trigger_press()

    def test_handle_trigger_tap_with_callback(self, mock_keyboard_hooks):
        """Test handling trigger tap with callback."""
        self.config.recording.mode = "tap-mode"
        manager = HotkeyManager(self.config)

        # Set callback
        on_tap = Mock()
        manager.set_callbacks(on_trigger_tap=on_tap)

        # Simulate trigger tap
        manager._handle_trigger_tap()

        on_tap.assert_called_once()

    def test_handle_discard_tap_with_callback(self, mock_keyboard_hooks):
        """Test handling discard tap with callback."""
        self.config.recording.mode = "tap-mode"
        manager = HotkeyManager(self.config)

        # Set callback
        on_discard = Mock()
        manager.set_callbacks(on_discard_tap=on_discard)

        # Simulate discard tap
        manager._handle_discard_tap()

        on_discard.assert_called_once()

    def test_on_key_press_event(self, mock_keyboard_hooks):
        """Test on_key_press event handling."""
        manager = HotkeyManager(self.config)
        manager.start_listening()

        # Mock listener canonical method
        mock_listener = mock_keyboard_hooks.Listener.return_value
        mock_listener.canonical.return_value = keyboard.Key.ctrl

        # Create mock hotkey
        mock_trigger_hotkey = Mock()
        manager.trigger_hotkey = mock_trigger_hotkey

        # Simulate key press
        test_key = keyboard.Key.ctrl
        manager.on_key_press(test_key)

        # Should pass to trigger hotkey
        mock_trigger_hotkey.press.assert_called_once_with(keyboard.Key.ctrl)

    def test_on_key_release_event_push_to_talk(self, mock_keyboard_hooks):
        """Test on_key_release event in push-to-talk mode."""
        manager = HotkeyManager(self.config)
        manager.start_listening()

        # Mock listener canonical method
        mock_listener = mock_keyboard_hooks.Listener.return_value
        mock_listener.canonical.return_value = keyboard.Key.ctrl

        # Set up mocks
        mock_trigger_hotkey = Mock()
        manager.trigger_hotkey = mock_trigger_hotkey

        on_release_callback = Mock()
        manager.set_callbacks(on_trigger_release=on_release_callback)

        # Simulate releasing a key
        test_key = keyboard.Key.ctrl
        manager.on_key_release(test_key)

        # Should check trigger hotkey and call release callback
        mock_trigger_hotkey.release.assert_called_once_with(keyboard.Key.ctrl)
        on_release_callback.assert_called_once()

    def test_on_key_release_event_tap_mode(self, mock_keyboard_hooks):
        """Test on_key_release event in tap mode."""
        self.config.recording.mode = "tap-mode"
        manager = HotkeyManager(self.config)
        manager.start_listening()

        # Mock listener canonical method
        mock_listener = mock_keyboard_hooks.Listener.return_value
        mock_listener.canonical.return_value = keyboard.Key.esc

        # Set up mocks
        mock_trigger_hotkey = Mock()
        mock_discard_hotkey = Mock()
        manager.trigger_hotkey = mock_trigger_hotkey
        manager.discard_hotkey = mock_discard_hotkey

        # Simulate key release
        test_key = keyboard.Key.esc
        manager.on_key_release(test_key)

        # Should pass to both hotkeys
        mock_trigger_hotkey.release.assert_called_once_with(keyboard.Key.esc)
        mock_discard_hotkey.release.assert_called_once_with(keyboard.Key.esc)

    def test_config_update(self, mock_keyboard_hooks):
        """Test updating configuration recreates hotkeys."""
        # Use real key parsing
        from pynput import keyboard

        mock_keyboard_hooks.HotKey.parse = keyboard.HotKey.parse

        HotkeyManager(self.config)

        # Change trigger key
        self.config.recording.trigger_key = "<f9>"

        # Recreate manager with new config
        new_manager = HotkeyManager(self.config)

        # Should parse new key combination successfully
        assert new_manager.trigger_hotkey is not None
        assert new_manager.config.recording.trigger_key == "<f9>"

    def test_invalid_key_format(self, mock_keyboard_hooks):
        """Test handling invalid key format."""
        # Use real parsing to test error handling
        from pynput import keyboard

        mock_keyboard_hooks.HotKey.parse = keyboard.HotKey.parse

        self.config.recording.trigger_key = "invalid_key_format"

        # Should raise ValueError from real parser
        with pytest.raises(ValueError):
            HotkeyManager(self.config)

    def test_complex_key_combinations(self, mock_keyboard_hooks):
        """Test complex key combinations with real parsing."""
        from pynput import keyboard

        mock_keyboard_hooks.HotKey.parse = keyboard.HotKey.parse

        # Test various complex combinations
        test_configs = [
            "<ctrl>+<alt>+<shift>+a",
            "<ctrl>+<f12>",
        ]

        for key_combo in test_configs:
            self.config.recording.trigger_key = key_combo
            manager = HotkeyManager(self.config)

            # Should successfully create manager with parsed keys
            assert manager.trigger_hotkey is not None
