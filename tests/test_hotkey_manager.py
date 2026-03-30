"""Test hotkey manager functionality."""

from unittest.mock import Mock

import pytest
from pynput import keyboard

from whisper_to_me.config import AppConfig, RecordingConfig
from whisper_to_me.display_backend import DisplayBackend
from whisper_to_me.hotkey_manager import HotkeyManager


class TestHotkeyManager:
    """Test HotkeyManager functionality (X11/pynput backend)."""

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
        from pynput import keyboard

        mock_keyboard_hooks.HotKey.parse = keyboard.HotKey.parse

        manager = HotkeyManager(self.config, backend=DisplayBackend.X11)

        assert manager.config == self.config
        assert manager.trigger_hotkey is not None
        assert manager.discard_hotkey is None  # Not used in push-to-talk

        expected_trigger_keys = keyboard.HotKey.parse("<ctrl>+<shift>+r")
        mock_keyboard_hooks.HotKey.assert_called_once_with(
            expected_trigger_keys, manager._backend._handle_trigger_press
        )

    def test_init_tap_mode(self, mock_keyboard_hooks):
        """Test HotkeyManager initialization in tap mode."""
        self.config.recording.mode = "tap-mode"

        from pynput import keyboard

        mock_keyboard_hooks.HotKey.parse = keyboard.HotKey.parse

        created_hotkeys = []

        def track_hotkey(*args, **kwargs):
            hotkey = Mock()
            created_hotkeys.append((args, kwargs))
            return hotkey

        mock_keyboard_hooks.HotKey.side_effect = track_hotkey

        manager = HotkeyManager(self.config, backend=DisplayBackend.X11)

        assert manager.trigger_hotkey is not None
        assert manager.discard_hotkey is not None
        assert len(created_hotkeys) == 2

        trigger_keys = keyboard.HotKey.parse("<ctrl>+<shift>+r")
        assert created_hotkeys[0][0][0] == trigger_keys

        discard_keys = keyboard.HotKey.parse("<esc>")
        assert created_hotkeys[1][0][0] == discard_keys

    def test_real_key_parsing(self):
        """Test that real key parsing works correctly."""
        test_cases = [
            ("<ctrl>+a", ["ctrl", "a"]),
            ("<shift>+<f1>", ["shift", "f1"]),
            ("<cmd>+<space>", ["cmd", "space"]),
            ("x", ["x"]),
            ("<esc>", ["esc"]),
        ]

        for key_string, expected_keys in test_cases:
            parsed = keyboard.HotKey.parse(key_string)
            assert len(parsed) == len(expected_keys), f"Failed parsing {key_string}"

    def test_set_callbacks(self, mock_keyboard_hooks):
        """Test setting callback functions."""
        manager = HotkeyManager(self.config, backend=DisplayBackend.X11)

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
        manager = HotkeyManager(self.config, backend=DisplayBackend.X11)
        manager.start_listening()

        mock_keyboard_hooks.Listener.assert_called_once()
        listener_instance = mock_keyboard_hooks.Listener.return_value
        listener_instance.start.assert_called_once()
        assert manager.listener == listener_instance

    def test_stop_listening(self, mock_keyboard_hooks):
        """Test stopping the keyboard listener."""
        manager = HotkeyManager(self.config, backend=DisplayBackend.X11)
        manager.start_listening()

        manager.stop_listening()

        listener_instance = mock_keyboard_hooks.Listener.return_value
        listener_instance.stop.assert_called_once()
        assert manager.listener is None

    def test_stop_listening_no_active_listener(self, mock_keyboard_hooks):
        """Test stopping when no listener is active."""
        manager = HotkeyManager(self.config, backend=DisplayBackend.X11)
        manager.stop_listening()
        assert manager.listener is None

    def test_handle_trigger_press_with_callback(self, mock_keyboard_hooks):
        """Test handling trigger press with callback."""
        manager = HotkeyManager(self.config, backend=DisplayBackend.X11)

        on_press = Mock()
        manager.set_callbacks(on_trigger_press=on_press)

        manager._handle_trigger_press()
        on_press.assert_called_once()

    def test_handle_trigger_press_no_callback(self, mock_keyboard_hooks):
        """Test handling trigger press without callback."""
        manager = HotkeyManager(self.config, backend=DisplayBackend.X11)
        manager._handle_trigger_press()

    def test_handle_trigger_tap_with_callback(self, mock_keyboard_hooks):
        """Test handling trigger tap with callback."""
        self.config.recording.mode = "tap-mode"
        manager = HotkeyManager(self.config, backend=DisplayBackend.X11)

        on_tap = Mock()
        manager.set_callbacks(on_trigger_tap=on_tap)

        manager._handle_trigger_tap()
        on_tap.assert_called_once()

    def test_handle_discard_tap_with_callback(self, mock_keyboard_hooks):
        """Test handling discard tap with callback."""
        self.config.recording.mode = "tap-mode"
        manager = HotkeyManager(self.config, backend=DisplayBackend.X11)

        on_discard = Mock()
        manager.set_callbacks(on_discard_tap=on_discard)

        manager._handle_discard_tap()
        on_discard.assert_called_once()

    def test_on_key_press_event(self, mock_keyboard_hooks):
        """Test on_key_press event handling."""
        manager = HotkeyManager(self.config, backend=DisplayBackend.X11)
        manager.start_listening()

        mock_listener = mock_keyboard_hooks.Listener.return_value
        mock_listener.canonical.return_value = keyboard.Key.ctrl

        mock_trigger_hotkey = Mock()
        manager._backend.trigger_hotkey = mock_trigger_hotkey

        test_key = keyboard.Key.ctrl
        manager.on_key_press(test_key)

        mock_trigger_hotkey.press.assert_called_once_with(keyboard.Key.ctrl)

    def test_on_key_release_event_push_to_talk(self, mock_keyboard_hooks):
        """Test on_key_release event in push-to-talk mode."""
        manager = HotkeyManager(self.config, backend=DisplayBackend.X11)
        manager.start_listening()

        mock_listener = mock_keyboard_hooks.Listener.return_value
        mock_listener.canonical.return_value = keyboard.Key.ctrl

        mock_trigger_hotkey = Mock()
        manager._backend.trigger_hotkey = mock_trigger_hotkey

        on_release_callback = Mock()
        manager.set_callbacks(on_trigger_release=on_release_callback)

        test_key = keyboard.Key.ctrl
        manager.on_key_release(test_key)

        mock_trigger_hotkey.release.assert_called_once_with(keyboard.Key.ctrl)
        on_release_callback.assert_called_once()

    def test_on_key_release_event_tap_mode(self, mock_keyboard_hooks):
        """Test on_key_release event in tap mode."""
        self.config.recording.mode = "tap-mode"
        manager = HotkeyManager(self.config, backend=DisplayBackend.X11)
        manager.start_listening()

        mock_listener = mock_keyboard_hooks.Listener.return_value
        mock_listener.canonical.return_value = keyboard.Key.esc

        mock_trigger_hotkey = Mock()
        mock_discard_hotkey = Mock()
        manager._backend.trigger_hotkey = mock_trigger_hotkey
        manager._backend.discard_hotkey = mock_discard_hotkey

        test_key = keyboard.Key.esc
        manager.on_key_release(test_key)

        mock_trigger_hotkey.release.assert_called_once_with(keyboard.Key.esc)
        mock_discard_hotkey.release.assert_called_once_with(keyboard.Key.esc)

    def test_config_update(self, mock_keyboard_hooks):
        """Test updating configuration recreates hotkeys."""
        from pynput import keyboard

        mock_keyboard_hooks.HotKey.parse = keyboard.HotKey.parse

        HotkeyManager(self.config, backend=DisplayBackend.X11)

        self.config.recording.trigger_key = "<f9>"

        new_manager = HotkeyManager(self.config, backend=DisplayBackend.X11)

        assert new_manager.trigger_hotkey is not None
        assert new_manager.config.recording.trigger_key == "<f9>"

    def test_invalid_key_format(self, mock_keyboard_hooks):
        """Test handling invalid key format."""
        from pynput import keyboard

        mock_keyboard_hooks.HotKey.parse = keyboard.HotKey.parse

        self.config.recording.trigger_key = "invalid_key_format"

        with pytest.raises(ValueError):
            HotkeyManager(self.config, backend=DisplayBackend.X11)

    def test_complex_key_combinations(self, mock_keyboard_hooks):
        """Test complex key combinations with real parsing."""
        from pynput import keyboard

        mock_keyboard_hooks.HotKey.parse = keyboard.HotKey.parse

        test_configs = [
            "<ctrl>+<alt>+<shift>+a",
            "<ctrl>+<f12>",
        ]

        for key_combo in test_configs:
            self.config.recording.trigger_key = key_combo
            manager = HotkeyManager(self.config, backend=DisplayBackend.X11)

            assert manager.trigger_hotkey is not None


class TestEvdevKeyMapping:
    """Test evdev key parsing utilities."""

    def test_parse_single_key(self):
        """Test parsing a single key."""
        import evdev.ecodes as ec

        from whisper_to_me.hotkey_manager import _parse_pynput_key_string

        codes = _parse_pynput_key_string("<scroll_lock>")
        assert codes == [ec.KEY_SCROLLLOCK]

    def test_parse_modifier_combo(self):
        """Test parsing a modifier + key combo."""
        import evdev.ecodes as ec

        from whisper_to_me.hotkey_manager import _parse_pynput_key_string

        codes = _parse_pynput_key_string("<ctrl>+<shift>+r")
        expected = sorted({ec.KEY_LEFTCTRL, ec.KEY_LEFTSHIFT, ec.KEY_R})
        assert codes == expected

    def test_parse_esc(self):
        """Test parsing escape key."""
        import evdev.ecodes as ec

        from whisper_to_me.hotkey_manager import _parse_pynput_key_string

        codes = _parse_pynput_key_string("<esc>")
        assert codes == [ec.KEY_ESC]

    def test_parse_function_key(self):
        """Test parsing function keys."""
        import evdev.ecodes as ec

        from whisper_to_me.hotkey_manager import _parse_pynput_key_string

        codes = _parse_pynput_key_string("<f9>")
        assert codes == [ec.KEY_F9]

    def test_parse_invalid_key(self):
        """Test that an invalid key raises ValueError."""
        from whisper_to_me.hotkey_manager import _parse_pynput_key_string

        with pytest.raises(ValueError, match="Cannot map pynput key token"):
            _parse_pynput_key_string("<nonexistent_key>")

    def test_modifier_canonicalisation(self):
        """Test that left/right modifiers are canonicalised."""
        from whisper_to_me.hotkey_manager import _parse_pynput_key_string

        left = _parse_pynput_key_string("<ctrl_l>+a")
        right = _parse_pynput_key_string("<ctrl_r>+a")
        generic = _parse_pynput_key_string("<ctrl>+a")

        assert left == generic
        # ctrl_r maps to KEY_RIGHTCTRL but is canonical'd to KEY_LEFTCTRL
        assert right == generic


class TestEvdevHotkeyBackend:
    """Test the evdev hotkey backend."""

    def setup_method(self):
        """Set up test config."""
        self.recording_config = RecordingConfig()
        self.recording_config.trigger_key = "<scroll_lock>"
        self.recording_config.discard_key = "<esc>"
        self.recording_config.mode = "push-to-talk"

        self.config = Mock(spec=AppConfig)
        self.config.recording = self.recording_config

    def test_evdev_backend_creation(self):
        """Test that the evdev backend can be created."""
        from whisper_to_me.hotkey_manager import _EvdevHotkeyBackend

        backend = _EvdevHotkeyBackend(self.config)
        assert backend is not None

    def test_evdev_backend_callbacks(self):
        """Test that callbacks are wired correctly."""
        from whisper_to_me.hotkey_manager import _EvdevHotkeyBackend

        backend = _EvdevHotkeyBackend(self.config)

        on_press = Mock()
        backend.on_trigger_press = on_press

        # Simulate the trigger key press
        import evdev.ecodes as ec

        backend._handle_event(ec.KEY_SCROLLLOCK, 1)  # key down

        on_press.assert_called_once()

    def test_evdev_backend_release_callback(self):
        """Test push-to-talk release callback."""
        from whisper_to_me.hotkey_manager import _EvdevHotkeyBackend

        backend = _EvdevHotkeyBackend(self.config)

        on_press = Mock()
        on_release = Mock()
        backend.on_trigger_press = on_press
        backend.on_trigger_release = on_release

        import evdev.ecodes as ec

        # Press trigger
        backend._handle_event(ec.KEY_SCROLLLOCK, 1)
        on_press.assert_called_once()

        # Release trigger
        backend._handle_event(ec.KEY_SCROLLLOCK, 0)
        on_release.assert_called_once()

    def test_evdev_backend_tap_mode(self):
        """Test tap mode with evdev backend."""
        self.config.recording.mode = "tap-mode"

        from whisper_to_me.hotkey_manager import _EvdevHotkeyBackend

        backend = _EvdevHotkeyBackend(self.config)

        on_tap = Mock()
        on_discard = Mock()
        backend.on_trigger_tap = on_tap
        backend.on_discard_tap = on_discard

        import evdev.ecodes as ec

        # Trigger tap
        backend._handle_event(ec.KEY_SCROLLLOCK, 1)
        on_tap.assert_called_once()

        # Release (shouldn't trigger discard)
        backend._handle_event(ec.KEY_SCROLLLOCK, 0)

        # Discard tap
        backend._handle_event(ec.KEY_ESC, 1)
        on_discard.assert_called_once()

    def test_evdev_backend_combo_key(self):
        """Test key combination matching with evdev backend."""
        self.config.recording.trigger_key = "<ctrl>+<shift>+r"

        from whisper_to_me.hotkey_manager import _EvdevHotkeyBackend

        backend = _EvdevHotkeyBackend(self.config)

        on_press = Mock()
        backend.on_trigger_press = on_press

        import evdev.ecodes as ec

        # Press ctrl, shift, r in sequence
        backend._handle_event(ec.KEY_LEFTCTRL, 1)
        assert on_press.call_count == 0

        backend._handle_event(ec.KEY_LEFTSHIFT, 1)
        assert on_press.call_count == 0

        backend._handle_event(ec.KEY_R, 1)
        on_press.assert_called_once()

    def test_evdev_backend_right_modifier(self):
        """Test that right modifier keys also match."""
        self.config.recording.trigger_key = "<ctrl>+a"

        from whisper_to_me.hotkey_manager import _EvdevHotkeyBackend

        backend = _EvdevHotkeyBackend(self.config)

        on_press = Mock()
        backend.on_trigger_press = on_press

        import evdev.ecodes as ec

        # Use RIGHT ctrl
        backend._handle_event(ec.KEY_RIGHTCTRL, 1)
        backend._handle_event(ec.KEY_A, 1)
        on_press.assert_called_once()
