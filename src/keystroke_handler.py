"""
Keystroke Simulation Module

Provides keyboard simulation functionality for typing transcribed text
into any application.
"""

from pynput import keyboard
import time


class KeystrokeHandler:
    """
    Handles keyboard simulation for typing transcribed text.

    Features:
    - Configurable typing speed
    - Fast and slow typing modes
    - Special key support (space, enter, etc.)
    """

    def __init__(self, typing_speed: float = 0.01):
        """
        Initialize the keystroke handler.

        Args:
            typing_speed: Delay between keystrokes in seconds (0.01 = fast)
        """
        self.typing_speed = typing_speed
        self.keyboard_controller = keyboard.Controller()

    def type_text(self, text: str) -> None:
        """
        Simulate typing the given text
        """
        if not text or not text.strip():
            return

        # Clean up the text
        text = text.strip()

        for char in text:
            self.keyboard_controller.type(char)
            time.sleep(self.typing_speed)

    def type_text_fast(self, text: str) -> None:
        """
        Type text without delays (faster)
        """
        if not text or not text.strip():
            return

        text = text.strip()
        self.keyboard_controller.type(text)

    def press_key(self, key) -> None:
        """
        Press a specific key
        """
        self.keyboard_controller.press(key)
        self.keyboard_controller.release(key)

    def add_space(self) -> None:
        """
        Add a space
        """
        self.press_key(keyboard.Key.space)

    def add_newline(self) -> None:
        """
        Add a newline
        """
        self.press_key(keyboard.Key.enter)
