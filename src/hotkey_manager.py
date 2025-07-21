"""
Hotkey Manager Module

Handles global hotkey detection and management with support for different recording modes.
Extracted from the main application to improve separation of concerns.
"""

from typing import Callable, Optional
from pynput import keyboard
from config import AppConfig


class HotkeyManager:
    """
    Manages global hotkey detection and handling for recording triggers.

    Features:
    - Push-to-talk and tap-mode support
    - HotKey state management using pynput
    - Mode-specific callback handling
    - Clean listener lifecycle management
    """

    def __init__(self, config: AppConfig):
        """
        Initialize the hotkey manager.

        Args:
            config: Application configuration containing key mappings
        """
        self.config = config
        self.listener: Optional[keyboard.Listener] = None
        self.trigger_hotkey: Optional[keyboard.HotKey] = None
        self.discard_hotkey: Optional[keyboard.HotKey] = None

        # Callbacks
        self.on_trigger_press: Optional[Callable] = None
        self.on_trigger_tap: Optional[Callable] = None
        self.on_discard_tap: Optional[Callable] = None
        self.on_trigger_release: Optional[Callable] = None

        self._setup_hotkeys()

    def _setup_hotkeys(self) -> None:
        """Setup HotKey objects based on current configuration."""
        # Parse key combinations
        trigger_keys = keyboard.HotKey.parse(self.config.recording.trigger_key)
        discard_keys = keyboard.HotKey.parse(self.config.recording.discard_key)

        # Create HotKey instances based on mode
        if self.config.recording.mode == "tap-mode":
            self.trigger_hotkey = keyboard.HotKey(
                trigger_keys, self._handle_trigger_tap
            )
            self.discard_hotkey = keyboard.HotKey(
                discard_keys, self._handle_discard_tap
            )
        else:  # push-to-talk mode
            self.trigger_hotkey = keyboard.HotKey(
                trigger_keys, self._handle_trigger_press
            )
            self.discard_hotkey = None  # Not used in push-to-talk mode

    def set_callbacks(
        self,
        on_trigger_press: Optional[Callable] = None,
        on_trigger_tap: Optional[Callable] = None,
        on_discard_tap: Optional[Callable] = None,
        on_trigger_release: Optional[Callable] = None,
    ) -> None:
        """
        Set callback functions for hotkey events.

        Args:
            on_trigger_press: Called when trigger is pressed (push-to-talk mode)
            on_trigger_tap: Called when trigger is tapped (tap mode)
            on_discard_tap: Called when discard key is tapped (tap mode)
            on_trigger_release: Called when trigger is released (push-to-talk mode)
        """
        self.on_trigger_press = on_trigger_press
        self.on_trigger_tap = on_trigger_tap
        self.on_discard_tap = on_discard_tap
        self.on_trigger_release = on_trigger_release

    def _handle_trigger_press(self) -> None:
        """Handle trigger key press in push-to-talk mode."""
        if self.on_trigger_press:
            self.on_trigger_press()

    def _handle_trigger_tap(self) -> None:
        """Handle trigger key tap in tap mode."""
        if self.on_trigger_tap:
            self.on_trigger_tap()

    def _handle_discard_tap(self) -> None:
        """Handle discard key tap in tap mode."""
        if self.on_discard_tap:
            self.on_discard_tap()

    def on_key_press(self, key) -> None:
        """
        Handle key press events.

        Args:
            key: The pressed key
        """
        if not self.listener:
            return

        # Let HotKey objects handle the state tracking
        canonical_key = self.listener.canonical(key)
        if self.trigger_hotkey:
            self.trigger_hotkey.press(canonical_key)
        if self.discard_hotkey:
            self.discard_hotkey.press(canonical_key)

    def on_key_release(self, key) -> None:
        """
        Handle key release events.

        Args:
            key: The released key
        """
        if not self.listener:
            return

        # Let HotKey objects handle the state tracking
        canonical_key = self.listener.canonical(key)
        if self.trigger_hotkey:
            self.trigger_hotkey.release(canonical_key)
        if self.discard_hotkey:
            self.discard_hotkey.release(canonical_key)

        # In push-to-talk mode, check if we should stop recording
        if self.config.recording.mode == "push-to-talk" and self.on_trigger_release:
            self.on_trigger_release()

    def start_listening(self) -> None:
        """Start the keyboard listener."""
        if self.listener is not None:
            return

        self.listener = keyboard.Listener(
            on_press=self.on_key_press, on_release=self.on_key_release
        )
        self.listener.start()

    def stop_listening(self) -> None:
        """Stop the keyboard listener."""
        if self.listener is not None:
            self.listener.stop()
            self.listener = None

    def join_listener(self) -> None:
        """Wait for the listener to finish (blocking call)."""
        if self.listener is not None:
            self.listener.join()

    def update_config(self, new_config: AppConfig) -> None:
        """
        Update hotkey configuration.

        Args:
            new_config: New configuration to apply
        """
        self.config = new_config
        self._setup_hotkeys()

    def get_trigger_key_display(self) -> str:
        """
        Get formatted trigger key for display.

        Returns:
            Formatted trigger key string
        """
        return self.config.recording.trigger_key

    def get_discard_key_display(self) -> str:
        """
        Get formatted discard key for display.

        Returns:
            Formatted discard key string
        """
        return self.config.recording.discard_key

    def is_tap_mode(self) -> bool:
        """
        Check if currently in tap mode.

        Returns:
            True if in tap mode, False for push-to-talk
        """
        return self.config.recording.mode == "tap-mode"
