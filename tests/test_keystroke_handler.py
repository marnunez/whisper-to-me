"""Test keystroke handler functionality."""

from unittest.mock import Mock, call, patch

from whisper_to_me.display_backend import DisplayBackend
from whisper_to_me.keystroke_handler import KeystrokeHandler


class TestKeystrokeHandler:
    """Test KeystrokeHandler functionality (X11/pynput backend)."""

    def setup_method(self):
        """Set up test environment."""
        with patch("pynput.keyboard.Controller"):
            self.handler = KeystrokeHandler(backend=DisplayBackend.X11)

    def test_init_default(self):
        """Test KeystrokeHandler initialization with defaults."""
        with patch("pynput.keyboard.Controller") as mock_controller:
            mock_instance = Mock()
            mock_controller.return_value = mock_instance

            handler = KeystrokeHandler(backend=DisplayBackend.X11)

            assert handler.typing_speed == 0.01
            assert handler.keyboard_controller == mock_instance
            mock_controller.assert_called_once()

    def test_init_custom_speed(self):
        """Test KeystrokeHandler initialization with custom typing speed."""
        with patch("pynput.keyboard.Controller") as mock_controller:
            mock_instance = Mock()
            mock_controller.return_value = mock_instance

            handler = KeystrokeHandler(typing_speed=0.05, backend=DisplayBackend.X11)

            assert handler.typing_speed == 0.05
            assert handler.keyboard_controller == mock_instance

    @patch("time.sleep")
    @patch("pynput.keyboard.Controller")
    def test_type_text_simple(self, mock_controller, mock_sleep):
        """Test type_text with simple text."""
        mock_keyboard = Mock()
        mock_controller.return_value = mock_keyboard

        handler = KeystrokeHandler(typing_speed=0.02, backend=DisplayBackend.X11)
        handler.type_text("Hello")

        # Should type each character with delays
        expected_calls = [call("H"), call("e"), call("l"), call("l"), call("o")]
        mock_keyboard.type.assert_has_calls(expected_calls)

        # Should have called sleep between each character
        assert mock_sleep.call_count == 5
        mock_sleep.assert_has_calls([call(0.02)] * 5)

    @patch("pynput.keyboard.Controller")
    def test_type_text_empty(self, mock_controller):
        """Test type_text with empty text."""
        mock_keyboard = Mock()
        mock_controller.return_value = mock_keyboard

        handler = KeystrokeHandler(backend=DisplayBackend.X11)
        handler.type_text("")

        # Should not type anything
        mock_keyboard.type.assert_not_called()

    @patch("pynput.keyboard.Controller")
    def test_type_text_whitespace_only(self, mock_controller):
        """Test type_text with whitespace-only text."""
        mock_keyboard = Mock()
        mock_controller.return_value = mock_keyboard

        handler = KeystrokeHandler(backend=DisplayBackend.X11)
        handler.type_text("   \t\n  ")

        # Should not type anything for whitespace-only text
        mock_keyboard.type.assert_not_called()

    @patch("time.sleep")
    @patch("pynput.keyboard.Controller")
    def test_type_text_with_trailing_space(self, mock_controller, mock_sleep):
        """Test type_text with trailing space."""
        mock_keyboard = Mock()
        mock_controller.return_value = mock_keyboard

        handler = KeystrokeHandler(backend=DisplayBackend.X11)
        handler.type_text("Hi", trailing_space=True)

        # Should type each character
        expected_calls = [call("H"), call("i")]
        mock_keyboard.type.assert_has_calls(expected_calls)

        # Should press space key
        mock_keyboard.press.assert_called_once()
        mock_keyboard.release.assert_called_once()

    @patch("time.sleep")
    @patch("pynput.keyboard.Controller")
    def test_type_text_strips_input(self, mock_controller, mock_sleep):
        """Test that type_text strips whitespace from input."""
        mock_keyboard = Mock()
        mock_controller.return_value = mock_keyboard

        handler = KeystrokeHandler(backend=DisplayBackend.X11)
        handler.type_text("  Hello  ")

        # Should only type the stripped text
        expected_calls = [call("H"), call("e"), call("l"), call("l"), call("o")]
        mock_keyboard.type.assert_has_calls(expected_calls)

    @patch("pynput.keyboard.Controller")
    def test_type_text_fast_simple(self, mock_controller):
        """Test type_text_fast with simple text."""
        mock_keyboard = Mock()
        mock_controller.return_value = mock_keyboard

        handler = KeystrokeHandler(backend=DisplayBackend.X11)
        handler.type_text_fast("Hello World")

        # Should type the whole text at once (stripped)
        mock_keyboard.type.assert_called_once_with("Hello World")

    @patch("pynput.keyboard.Controller")
    def test_type_text_fast_empty(self, mock_controller):
        """Test type_text_fast with empty text."""
        mock_keyboard = Mock()
        mock_controller.return_value = mock_keyboard

        handler = KeystrokeHandler(backend=DisplayBackend.X11)
        handler.type_text_fast("")

        # Should not type anything
        mock_keyboard.type.assert_not_called()

    @patch("pynput.keyboard.Controller")
    def test_type_text_fast_whitespace_only(self, mock_controller):
        """Test type_text_fast with whitespace-only text."""
        mock_keyboard = Mock()
        mock_controller.return_value = mock_keyboard

        handler = KeystrokeHandler(backend=DisplayBackend.X11)
        handler.type_text_fast("   \n\t   ")

        # Should not type anything for whitespace-only text
        mock_keyboard.type.assert_not_called()

    @patch("pynput.keyboard.Controller")
    def test_type_text_fast_with_trailing_space(self, mock_controller):
        """Test type_text_fast with trailing space."""
        mock_keyboard = Mock()
        mock_controller.return_value = mock_keyboard

        handler = KeystrokeHandler(backend=DisplayBackend.X11)
        handler.type_text_fast("Hello", trailing_space=True)

        # Should type the text
        mock_keyboard.type.assert_called_once_with("Hello")

        # Should press space key
        mock_keyboard.press.assert_called_once()
        mock_keyboard.release.assert_called_once()

    @patch("pynput.keyboard.Controller")
    def test_type_text_fast_strips_input(self, mock_controller):
        """Test that type_text_fast strips whitespace from input."""
        mock_keyboard = Mock()
        mock_controller.return_value = mock_keyboard

        handler = KeystrokeHandler(backend=DisplayBackend.X11)
        handler.type_text_fast("  Hello World  ")

        # Should only type the stripped text
        mock_keyboard.type.assert_called_once_with("Hello World")

    @patch("pynput.keyboard.Controller")
    def test_press_key(self, mock_controller):
        """Test press_key method."""
        mock_keyboard = Mock()
        mock_controller.return_value = mock_keyboard

        from pynput import keyboard

        handler = KeystrokeHandler(backend=DisplayBackend.X11)
        handler.press_key(keyboard.Key.enter)

        # Should press and release the key
        mock_keyboard.press.assert_called_once_with(keyboard.Key.enter)
        mock_keyboard.release.assert_called_once_with(keyboard.Key.enter)

    @patch("pynput.keyboard.Controller")
    def test_add_space(self, mock_controller):
        """Test add_space method."""
        mock_keyboard = Mock()
        mock_controller.return_value = mock_keyboard

        from pynput import keyboard

        handler = KeystrokeHandler(backend=DisplayBackend.X11)
        handler.add_space()

        # Should press and release space key
        mock_keyboard.press.assert_called_once_with(keyboard.Key.space)
        mock_keyboard.release.assert_called_once_with(keyboard.Key.space)

    @patch("pynput.keyboard.Controller")
    def test_add_newline(self, mock_controller):
        """Test add_newline method."""
        mock_keyboard = Mock()
        mock_controller.return_value = mock_keyboard

        from pynput import keyboard

        handler = KeystrokeHandler(backend=DisplayBackend.X11)
        handler.add_newline()

        # Should press and release enter key
        mock_keyboard.press.assert_called_once_with(keyboard.Key.enter)
        mock_keyboard.release.assert_called_once_with(keyboard.Key.enter)

    @patch("time.sleep")
    @patch("pynput.keyboard.Controller")
    def test_typing_speed_respected(self, mock_controller, mock_sleep):
        """Test that typing speed is respected."""
        mock_keyboard = Mock()
        mock_controller.return_value = mock_keyboard

        custom_speed = 0.1
        handler = KeystrokeHandler(
            typing_speed=custom_speed, backend=DisplayBackend.X11
        )
        handler.type_text("Hi")

        # Should sleep with the custom speed
        mock_sleep.assert_has_calls([call(custom_speed), call(custom_speed)])

    @patch("pynput.keyboard.Controller")
    def test_multiple_operations(self, mock_controller):
        """Test multiple operations in sequence."""
        mock_keyboard = Mock()
        mock_controller.return_value = mock_keyboard

        from pynput import keyboard

        handler = KeystrokeHandler(backend=DisplayBackend.X11)

        handler.type_text_fast("Hello")
        handler.add_space()
        handler.type_text_fast("World")
        handler.add_newline()

        # Verify all operations
        assert mock_keyboard.type.call_count == 2
        mock_keyboard.type.assert_any_call("Hello")
        mock_keyboard.type.assert_any_call("World")

        # Should have pressed space and enter
        mock_keyboard.press.assert_any_call(keyboard.Key.space)
        mock_keyboard.press.assert_any_call(keyboard.Key.enter)
        assert mock_keyboard.press.call_count == 2
        assert mock_keyboard.release.call_count == 2

    @patch("pynput.keyboard.Controller")
    def test_text_with_special_characters(self, mock_controller):
        """Test typing text with special characters."""
        mock_keyboard = Mock()
        mock_controller.return_value = mock_keyboard

        handler = KeystrokeHandler(backend=DisplayBackend.X11)
        special_text = "Hello! @#$%^&*() 123"

        handler.type_text_fast(special_text)

        mock_keyboard.type.assert_called_once_with(special_text)

    @patch("pynput.keyboard.Controller")
    def test_text_with_unicode(self, mock_controller):
        """Test typing text with unicode characters."""
        mock_keyboard = Mock()
        mock_controller.return_value = mock_keyboard

        handler = KeystrokeHandler(backend=DisplayBackend.X11)
        unicode_text = "Héllo Wörld 🌍"

        handler.type_text_fast(unicode_text)

        mock_keyboard.type.assert_called_once_with(unicode_text)

    @patch("time.sleep")
    @patch("pynput.keyboard.Controller")
    def test_no_sleep_on_fast_typing(self, mock_controller, mock_sleep):
        """Test that type_text_fast doesn't call sleep."""
        mock_keyboard = Mock()
        mock_controller.return_value = mock_keyboard

        handler = KeystrokeHandler(backend=DisplayBackend.X11)
        handler.type_text_fast("Hello World")

        # Should not call sleep for fast typing
        mock_sleep.assert_not_called()

    @patch("pynput.keyboard.Controller")
    def test_press_key_with_different_keys(self, mock_controller):
        """Test press_key with various key types."""
        mock_keyboard = Mock()
        mock_controller.return_value = mock_keyboard

        from pynput import keyboard

        handler = KeystrokeHandler(backend=DisplayBackend.X11)

        # Test different key types
        keys_to_test = [
            keyboard.Key.enter,
            keyboard.Key.space,
            keyboard.Key.tab,
            keyboard.Key.esc,
            "a",  # Character key
        ]

        for key in keys_to_test:
            mock_keyboard.reset_mock()
            handler.press_key(key)

            mock_keyboard.press.assert_called_once()
            mock_keyboard.release.assert_called_once()


class TestWtypeKeystrokeBackend:
    """Test WtypeKeystrokeBackend (Wayland)."""

    @patch("subprocess.run")
    def test_type_text_fast(self, mock_run):
        """Test fast text typing via wtype."""
        handler = KeystrokeHandler(backend=DisplayBackend.WAYLAND)
        handler.type_text_fast("Hello World")

        mock_run.assert_called_once_with(["wtype", "--", "Hello World"], check=True)

    @patch("subprocess.run")
    def test_type_text_with_delay(self, mock_run):
        """Test text typing with delay via wtype."""
        handler = KeystrokeHandler(typing_speed=0.05, backend=DisplayBackend.WAYLAND)
        handler.type_text("Hello")

        mock_run.assert_called_once_with(
            ["wtype", "-d", "50", "--", "Hello"], check=True
        )

    @patch("subprocess.run")
    def test_add_space(self, mock_run):
        """Test add_space via wtype."""
        handler = KeystrokeHandler(backend=DisplayBackend.WAYLAND)
        handler.add_space()

        mock_run.assert_called_once_with(["wtype", "-k", "space"], check=True)

    @patch("subprocess.run")
    def test_add_newline(self, mock_run):
        """Test add_newline via wtype."""
        handler = KeystrokeHandler(backend=DisplayBackend.WAYLAND)
        handler.add_newline()

        mock_run.assert_called_once_with(["wtype", "-k", "Return"], check=True)

    @patch("subprocess.run")
    def test_type_text_empty(self, mock_run):
        """Test that empty text doesn't call wtype."""
        handler = KeystrokeHandler(backend=DisplayBackend.WAYLAND)
        handler.type_text("")
        mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_type_text_fast_with_trailing_space(self, mock_run):
        """Test fast typing with trailing space via wtype."""
        handler = KeystrokeHandler(backend=DisplayBackend.WAYLAND)
        handler.type_text_fast("Hello", trailing_space=True)

        assert mock_run.call_count == 2
        mock_run.assert_any_call(["wtype", "--", "Hello"], check=True)
        mock_run.assert_any_call(["wtype", "-k", "space"], check=True)

    @patch("subprocess.run")
    def test_press_key_string(self, mock_run):
        """Test pressing a named key via wtype."""
        handler = KeystrokeHandler(backend=DisplayBackend.WAYLAND)
        handler.press_key("enter")

        mock_run.assert_called_once_with(["wtype", "-k", "Return"], check=True)
