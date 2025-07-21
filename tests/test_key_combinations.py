"""Test key combination parsing functionality using pynput HotKey.parse."""

import os
import sys
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import ConfigManager
from pynput import keyboard


class TestKeyCombinations:
    """Test cases for key combination parsing using pynput HotKey.parse."""

    def setup_method(self):
        """Set up test environment."""
        self.config_manager = ConfigManager()

    def test_parse_single_keys(self):
        """Test parsing single keys in pynput format."""
        # Test that parse_key_combination works with pynput format
        result = self.config_manager.parse_key_combination("<scroll_lock>")
        assert isinstance(result, set)
        assert len(result) >= 1  # Should contain at least one key

        result = self.config_manager.parse_key_combination("<esc>")
        assert isinstance(result, set)
        assert len(result) >= 1

        result = self.config_manager.parse_key_combination("a")
        assert isinstance(result, set)
        assert len(result) >= 1

    def test_parse_key_combinations(self):
        """Test parsing key combinations in pynput format."""
        # Test that complex combinations parse successfully
        result = self.config_manager.parse_key_combination("<ctrl>+<shift>+r")
        assert isinstance(result, set)
        assert len(result) == 3  # Should have all three keys

        result = self.config_manager.parse_key_combination("<alt>+<tab>")
        assert isinstance(result, set)
        assert len(result) == 2  # Should have both keys

        result = self.config_manager.parse_key_combination("<ctrl>+-")
        assert isinstance(result, set)
        assert len(result) == 2  # Should have ctrl and minus

    def test_invalid_combinations_raise_errors(self):
        """Test that invalid key combinations raise ValueError."""
        # Invalid key name
        with pytest.raises(ValueError, match="Invalid key combination"):
            self.config_manager.parse_key_combination("<invalid_key>")

        # Old format should fail
        with pytest.raises(ValueError, match="Invalid key combination"):
            self.config_manager.parse_key_combination("ctrl+shift+r")

        # Empty string
        with pytest.raises(ValueError, match="Invalid key combination"):
            self.config_manager.parse_key_combination("")

        # Invalid combination
        with pytest.raises(ValueError, match="Invalid key combination"):
            self.config_manager.parse_key_combination("<ctrl>+asdf")

    def test_parse_key_string_single_only(self):
        """Test that parse_key_string only accepts single keys."""
        # Valid single keys should work
        result = self.config_manager.parse_key_string("<esc>")
        assert result is not None

        result = self.config_manager.parse_key_string("a")
        assert result is not None

        # Combinations should fail
        with pytest.raises(ValueError, match="Expected single key, got combination"):
            self.config_manager.parse_key_string("<ctrl>+<esc>")

        with pytest.raises(ValueError, match="Expected single key, got combination"):
            self.config_manager.parse_key_string("<alt>+f")

    def test_hotkey_integration(self):
        """Test that parsed keys work with pynput HotKey."""
        # Test that our parsing is compatible with HotKey
        trigger_keys = self.config_manager.parse_key_combination("<ctrl>+<shift>+r")

        # Should be able to create a HotKey with parsed keys
        callback_called = False

        def test_callback():
            nonlocal callback_called
            callback_called = True

        # This should not raise an exception
        hotkey = keyboard.HotKey(trigger_keys, test_callback)
        assert hotkey is not None

        # Test single key too
        single_keys = self.config_manager.parse_key_combination("<esc>")
        hotkey2 = keyboard.HotKey(single_keys, test_callback)
        assert hotkey2 is not None
