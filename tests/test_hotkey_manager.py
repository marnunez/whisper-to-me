"""Test hotkey manager functionality."""

from unittest.mock import Mock, patch, call

from whisper_to_me.hotkey_manager import HotkeyManager
from whisper_to_me.config import AppConfig, RecordingConfig


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

    @patch("whisper_to_me.hotkey_manager.keyboard.HotKey")
    def test_init_push_to_talk_mode(self, mock_hotkey_class):
        """Test HotkeyManager initialization in push-to-talk mode."""
        mock_trigger_hotkey = Mock()
        mock_hotkey_class.return_value = mock_trigger_hotkey

        with patch("whisper_to_me.hotkey_manager.keyboard.HotKey.parse") as mock_parse:
            mock_parse.side_effect = [
                ["ctrl", "shift", "r"],  # trigger key
                ["esc"],  # discard key
            ]

            manager = HotkeyManager(self.config)

            assert manager.config == self.config
            assert manager.listener is None
            assert manager.trigger_hotkey == mock_trigger_hotkey
            assert manager.discard_hotkey is None  # Not used in push-to-talk

            # Should parse both keys
            assert mock_parse.call_count == 2
            mock_parse.assert_any_call("<ctrl>+<shift>+r")
            mock_parse.assert_any_call("<esc>")

            # Should create trigger hotkey only
            mock_hotkey_class.assert_called_once_with(
                ["ctrl", "shift", "r"], manager._handle_trigger_press
            )

    @patch("whisper_to_me.hotkey_manager.keyboard.HotKey")
    def test_init_tap_mode(self, mock_hotkey_class):
        """Test HotkeyManager initialization in tap mode."""
        self.config.recording.mode = "tap-mode"

        mock_trigger_hotkey = Mock()
        mock_discard_hotkey = Mock()
        mock_hotkey_class.side_effect = [mock_trigger_hotkey, mock_discard_hotkey]

        with patch("whisper_to_me.hotkey_manager.keyboard.HotKey.parse") as mock_parse:
            mock_parse.side_effect = [
                ["ctrl", "shift", "r"],  # trigger key
                ["esc"],  # discard key
            ]

            manager = HotkeyManager(self.config)

            assert manager.trigger_hotkey == mock_trigger_hotkey
            assert manager.discard_hotkey == mock_discard_hotkey

            # Should create both hotkeys
            expected_calls = [
                call(["ctrl", "shift", "r"], manager._handle_trigger_tap),
                call(["esc"], manager._handle_discard_tap),
            ]
            mock_hotkey_class.assert_has_calls(expected_calls)

    @patch("whisper_to_me.hotkey_manager.keyboard.HotKey")
    def test_set_callbacks(self, mock_hotkey_class):
        """Test setting callback functions."""
        with patch("whisper_to_me.hotkey_manager.keyboard.HotKey.parse"):
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

    @patch("whisper_to_me.hotkey_manager.keyboard.HotKey")
    def test_set_callbacks_partial(self, mock_hotkey_class):
        """Test setting only some callbacks."""
        with patch("whisper_to_me.hotkey_manager.keyboard.HotKey.parse"):
            manager = HotkeyManager(self.config)

        on_trigger_press = Mock()

        manager.set_callbacks(on_trigger_press=on_trigger_press)

        assert manager.on_trigger_press == on_trigger_press
        assert manager.on_trigger_tap is None
        assert manager.on_discard_tap is None
        assert manager.on_trigger_release is None

    @patch("whisper_to_me.hotkey_manager.keyboard.HotKey")
    def test_handle_trigger_press(self, mock_hotkey_class):
        """Test _handle_trigger_press method."""
        with patch("whisper_to_me.hotkey_manager.keyboard.HotKey.parse"):
            manager = HotkeyManager(self.config)

        # Test with no callback
        manager._handle_trigger_press()  # Should not raise

        # Test with callback
        callback = Mock()
        manager.on_trigger_press = callback

        manager._handle_trigger_press()
        callback.assert_called_once()

    @patch("whisper_to_me.hotkey_manager.keyboard.HotKey")
    def test_handle_trigger_tap(self, mock_hotkey_class):
        """Test _handle_trigger_tap method."""
        with patch("whisper_to_me.hotkey_manager.keyboard.HotKey.parse"):
            manager = HotkeyManager(self.config)

        # Test with no callback
        manager._handle_trigger_tap()  # Should not raise

        # Test with callback
        callback = Mock()
        manager.on_trigger_tap = callback

        manager._handle_trigger_tap()
        callback.assert_called_once()

    @patch("whisper_to_me.hotkey_manager.keyboard.HotKey")
    def test_handle_discard_tap(self, mock_hotkey_class):
        """Test _handle_discard_tap method."""
        with patch("whisper_to_me.hotkey_manager.keyboard.HotKey.parse"):
            manager = HotkeyManager(self.config)

        # Test with no callback
        manager._handle_discard_tap()  # Should not raise

        # Test with callback
        callback = Mock()
        manager.on_discard_tap = callback

        manager._handle_discard_tap()
        callback.assert_called_once()

    @patch("whisper_to_me.hotkey_manager.keyboard.HotKey")
    def test_on_key_press_no_listener(self, mock_hotkey_class):
        """Test on_key_press when no listener is active."""
        with patch("whisper_to_me.hotkey_manager.keyboard.HotKey.parse"):
            manager = HotkeyManager(self.config)

        mock_key = Mock()
        manager.on_key_press(mock_key)  # Should not raise

    @patch("whisper_to_me.hotkey_manager.keyboard.HotKey")
    @patch("whisper_to_me.hotkey_manager.keyboard.Listener")
    def test_on_key_press_with_listener(self, mock_listener_class, mock_hotkey_class):
        """Test on_key_press with active listener."""
        mock_trigger_hotkey = Mock()
        mock_hotkey_class.return_value = mock_trigger_hotkey

        mock_listener = Mock()
        mock_listener.canonical.return_value = "canonical_key"
        mock_listener_class.return_value = mock_listener

        with patch("whisper_to_me.hotkey_manager.keyboard.HotKey.parse"):
            manager = HotkeyManager(self.config)
            manager.listener = mock_listener
            manager.trigger_hotkey = mock_trigger_hotkey
            manager.discard_hotkey = None

        mock_key = Mock()
        manager.on_key_press(mock_key)

        # Should canonicalize key and pass to trigger hotkey
        mock_listener.canonical.assert_called_once_with(mock_key)
        mock_trigger_hotkey.press.assert_called_once_with("canonical_key")

    @patch("whisper_to_me.hotkey_manager.keyboard.HotKey")
    @patch("whisper_to_me.hotkey_manager.keyboard.Listener")
    def test_on_key_press_with_both_hotkeys(
        self, mock_listener_class, mock_hotkey_class
    ):
        """Test on_key_press with both trigger and discard hotkeys."""
        self.config.recording.mode = "tap-mode"

        mock_trigger_hotkey = Mock()
        mock_discard_hotkey = Mock()
        mock_hotkey_class.side_effect = [mock_trigger_hotkey, mock_discard_hotkey]

        mock_listener = Mock()
        mock_listener.canonical.return_value = "canonical_key"
        mock_listener_class.return_value = mock_listener

        with patch("whisper_to_me.hotkey_manager.keyboard.HotKey.parse"):
            manager = HotkeyManager(self.config)
            manager.listener = mock_listener

        mock_key = Mock()
        manager.on_key_press(mock_key)

        # Should pass to both hotkeys
        mock_trigger_hotkey.press.assert_called_once_with("canonical_key")
        mock_discard_hotkey.press.assert_called_once_with("canonical_key")

    @patch("whisper_to_me.hotkey_manager.keyboard.HotKey")
    def test_on_key_release_no_listener(self, mock_hotkey_class):
        """Test on_key_release when no listener is active."""
        with patch("whisper_to_me.hotkey_manager.keyboard.HotKey.parse"):
            manager = HotkeyManager(self.config)

        mock_key = Mock()
        manager.on_key_release(mock_key)  # Should not raise

    @patch("whisper_to_me.hotkey_manager.keyboard.HotKey")
    @patch("whisper_to_me.hotkey_manager.keyboard.Listener")
    def test_on_key_release_push_to_talk_mode(
        self, mock_listener_class, mock_hotkey_class
    ):
        """Test on_key_release in push-to-talk mode."""
        mock_trigger_hotkey = Mock()
        mock_hotkey_class.return_value = mock_trigger_hotkey

        mock_listener = Mock()
        mock_listener.canonical.return_value = "canonical_key"

        with patch("whisper_to_me.hotkey_manager.keyboard.HotKey.parse"):
            manager = HotkeyManager(self.config)
            manager.listener = mock_listener
            manager.trigger_hotkey = mock_trigger_hotkey
            manager.discard_hotkey = None

        # Test with callback
        callback = Mock()
        manager.on_trigger_release = callback

        mock_key = Mock()
        manager.on_key_release(mock_key)

        # Should call release on hotkey and trigger callback
        mock_trigger_hotkey.release.assert_called_once_with("canonical_key")
        callback.assert_called_once()

    @patch("whisper_to_me.hotkey_manager.keyboard.HotKey")
    @patch("whisper_to_me.hotkey_manager.keyboard.Listener")
    def test_on_key_release_tap_mode(self, mock_listener_class, mock_hotkey_class):
        """Test on_key_release in tap mode."""
        self.config.recording.mode = "tap-mode"

        mock_trigger_hotkey = Mock()
        mock_discard_hotkey = Mock()
        mock_hotkey_class.side_effect = [mock_trigger_hotkey, mock_discard_hotkey]

        mock_listener = Mock()
        mock_listener.canonical.return_value = "canonical_key"

        with patch("whisper_to_me.hotkey_manager.keyboard.HotKey.parse"):
            manager = HotkeyManager(self.config)
            manager.listener = mock_listener

        callback = Mock()
        manager.on_trigger_release = callback

        mock_key = Mock()
        manager.on_key_release(mock_key)

        # Should call release on both hotkeys but NOT trigger callback in tap mode
        mock_trigger_hotkey.release.assert_called_once_with("canonical_key")
        mock_discard_hotkey.release.assert_called_once_with("canonical_key")
        callback.assert_not_called()

    @patch("whisper_to_me.hotkey_manager.keyboard.HotKey")
    @patch("whisper_to_me.hotkey_manager.keyboard.Listener")
    def test_start_listening(self, mock_listener_class, mock_hotkey_class):
        """Test start_listening method."""
        mock_listener = Mock()
        mock_listener_class.return_value = mock_listener

        with patch("whisper_to_me.hotkey_manager.keyboard.HotKey.parse"):
            manager = HotkeyManager(self.config)

        manager.start_listening()

        # Should create and start listener
        mock_listener_class.assert_called_once_with(
            on_press=manager.on_key_press, on_release=manager.on_key_release
        )
        mock_listener.start.assert_called_once()
        assert manager.listener == mock_listener

    @patch("whisper_to_me.hotkey_manager.keyboard.HotKey")
    @patch("whisper_to_me.hotkey_manager.keyboard.Listener")
    def test_start_listening_already_active(
        self, mock_listener_class, mock_hotkey_class
    ):
        """Test start_listening when already active."""
        mock_listener = Mock()
        mock_listener_class.return_value = mock_listener

        with patch("whisper_to_me.hotkey_manager.keyboard.HotKey.parse"):
            manager = HotkeyManager(self.config)
            manager.listener = mock_listener  # Already active

        manager.start_listening()

        # Should not create new listener
        mock_listener_class.assert_not_called()
        mock_listener.start.assert_not_called()

    @patch("whisper_to_me.hotkey_manager.keyboard.HotKey")
    def test_stop_listening(self, mock_hotkey_class):
        """Test stop_listening method."""
        mock_listener = Mock()

        with patch("whisper_to_me.hotkey_manager.keyboard.HotKey.parse"):
            manager = HotkeyManager(self.config)
            manager.listener = mock_listener

        manager.stop_listening()

        # Should stop and clear listener
        mock_listener.stop.assert_called_once()
        assert manager.listener is None

    @patch("whisper_to_me.hotkey_manager.keyboard.HotKey")
    def test_stop_listening_no_listener(self, mock_hotkey_class):
        """Test stop_listening when no listener active."""
        with patch("whisper_to_me.hotkey_manager.keyboard.HotKey.parse"):
            manager = HotkeyManager(self.config)

        manager.stop_listening()  # Should not raise

    @patch("whisper_to_me.hotkey_manager.keyboard.HotKey")
    def test_join_listener(self, mock_hotkey_class):
        """Test join_listener method."""
        mock_listener = Mock()

        with patch("whisper_to_me.hotkey_manager.keyboard.HotKey.parse"):
            manager = HotkeyManager(self.config)
            manager.listener = mock_listener

        manager.join_listener()

        mock_listener.join.assert_called_once()

    @patch("whisper_to_me.hotkey_manager.keyboard.HotKey")
    def test_join_listener_no_listener(self, mock_hotkey_class):
        """Test join_listener when no listener active."""
        with patch("whisper_to_me.hotkey_manager.keyboard.HotKey.parse"):
            manager = HotkeyManager(self.config)

        manager.join_listener()  # Should not raise

    @patch("whisper_to_me.hotkey_manager.keyboard.HotKey")
    def test_update_config(self, mock_hotkey_class):
        """Test update_config method."""
        old_trigger_hotkey = Mock()
        mock_hotkey_class.return_value = old_trigger_hotkey

        with patch("whisper_to_me.hotkey_manager.keyboard.HotKey.parse"):
            manager = HotkeyManager(self.config)

        # Create new config
        new_recording_config = RecordingConfig()
        new_recording_config.trigger_key = "<alt>+r"
        new_recording_config.discard_key = "<del>"
        new_recording_config.mode = "tap-mode"

        new_config = Mock(spec=AppConfig)
        new_config.recording = new_recording_config

        # Mock new hotkeys
        new_trigger_hotkey = Mock()
        new_discard_hotkey = Mock()
        mock_hotkey_class.side_effect = [new_trigger_hotkey, new_discard_hotkey]

        with patch("whisper_to_me.hotkey_manager.keyboard.HotKey.parse") as mock_parse:
            mock_parse.side_effect = [
                ["alt", "r"],  # new trigger key
                ["del"],  # new discard key
            ]

            manager.update_config(new_config)

        # Should update config and recreate hotkeys
        assert manager.config == new_config
        assert manager.trigger_hotkey == new_trigger_hotkey
        assert manager.discard_hotkey == new_discard_hotkey

    @patch("whisper_to_me.hotkey_manager.keyboard.HotKey")
    def test_get_trigger_key_display(self, mock_hotkey_class):
        """Test get_trigger_key_display method."""
        with patch("whisper_to_me.hotkey_manager.keyboard.HotKey.parse"):
            manager = HotkeyManager(self.config)

        result = manager.get_trigger_key_display()
        assert result == "<ctrl>+<shift>+r"

    @patch("whisper_to_me.hotkey_manager.keyboard.HotKey")
    def test_get_discard_key_display(self, mock_hotkey_class):
        """Test get_discard_key_display method."""
        with patch("whisper_to_me.hotkey_manager.keyboard.HotKey.parse"):
            manager = HotkeyManager(self.config)

        result = manager.get_discard_key_display()
        assert result == "<esc>"

    @patch("whisper_to_me.hotkey_manager.keyboard.HotKey")
    def test_is_tap_mode_false(self, mock_hotkey_class):
        """Test is_tap_mode returns False for push-to-talk."""
        with patch("whisper_to_me.hotkey_manager.keyboard.HotKey.parse"):
            manager = HotkeyManager(self.config)

        assert manager.is_tap_mode() is False

    @patch("whisper_to_me.hotkey_manager.keyboard.HotKey")
    def test_is_tap_mode_true(self, mock_hotkey_class):
        """Test is_tap_mode returns True for tap mode."""
        self.config.recording.mode = "tap-mode"

        with patch("whisper_to_me.hotkey_manager.keyboard.HotKey.parse"):
            manager = HotkeyManager(self.config)

        assert manager.is_tap_mode() is True

    @patch("whisper_to_me.hotkey_manager.keyboard.HotKey")
    @patch("whisper_to_me.hotkey_manager.keyboard.Listener")
    def test_full_lifecycle_push_to_talk(self, mock_listener_class, mock_hotkey_class):
        """Test complete lifecycle in push-to-talk mode."""
        mock_listener = Mock()
        mock_listener.canonical.return_value = "canonical_key"
        mock_listener_class.return_value = mock_listener

        mock_trigger_hotkey = Mock()
        mock_hotkey_class.return_value = mock_trigger_hotkey

        # Setup callbacks
        on_press = Mock()
        on_release = Mock()

        with patch("whisper_to_me.hotkey_manager.keyboard.HotKey.parse"):
            manager = HotkeyManager(self.config)
            manager.set_callbacks(
                on_trigger_press=on_press, on_trigger_release=on_release
            )

        # Start listening
        manager.start_listening()
        assert manager.listener == mock_listener

        # Simulate key press
        mock_key = Mock()
        manager.on_key_press(mock_key)
        mock_trigger_hotkey.press.assert_called_with("canonical_key")

        # Simulate key release
        manager.on_key_release(mock_key)
        mock_trigger_hotkey.release.assert_called_with("canonical_key")
        on_release.assert_called_once()

        # Stop listening
        manager.stop_listening()
        mock_listener.stop.assert_called_once()
        assert manager.listener is None

    @patch("whisper_to_me.hotkey_manager.keyboard.HotKey")
    @patch("whisper_to_me.hotkey_manager.keyboard.Listener")
    def test_full_lifecycle_tap_mode(self, mock_listener_class, mock_hotkey_class):
        """Test complete lifecycle in tap mode."""
        self.config.recording.mode = "tap-mode"

        mock_listener = Mock()
        mock_listener.canonical.return_value = "canonical_key"
        mock_listener_class.return_value = mock_listener

        mock_trigger_hotkey = Mock()
        mock_discard_hotkey = Mock()
        mock_hotkey_class.side_effect = [mock_trigger_hotkey, mock_discard_hotkey]

        # Setup callbacks
        on_tap = Mock()
        on_discard = Mock()

        with patch("whisper_to_me.hotkey_manager.keyboard.HotKey.parse"):
            manager = HotkeyManager(self.config)
            manager.set_callbacks(on_trigger_tap=on_tap, on_discard_tap=on_discard)

        # Start listening
        manager.start_listening()

        # Simulate key press/release
        mock_key = Mock()
        manager.on_key_press(mock_key)
        manager.on_key_release(mock_key)

        # Should handle both hotkeys
        mock_trigger_hotkey.press.assert_called_with("canonical_key")
        mock_trigger_hotkey.release.assert_called_with("canonical_key")
        mock_discard_hotkey.press.assert_called_with("canonical_key")
        mock_discard_hotkey.release.assert_called_with("canonical_key")

        # Test callbacks
        manager._handle_trigger_tap()
        on_tap.assert_called_once()

        manager._handle_discard_tap()
        on_discard.assert_called_once()

        manager.stop_listening()
