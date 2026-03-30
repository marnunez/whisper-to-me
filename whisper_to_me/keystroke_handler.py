"""
Keystroke Simulation Module

Provides keyboard simulation functionality for typing transcribed text
into any application.  Supports both X11 (via pynput) and Wayland (via wtype).
"""

from __future__ import annotations

import subprocess
import time
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from whisper_to_me.display_backend import DisplayBackend


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class KeystrokeBackend(Protocol):
    """Interface that every keystroke backend must implement."""

    def type_text(self, text: str) -> None:
        """Type *text* character-by-character (or equivalent)."""
        ...

    def type_text_fast(self, text: str) -> None:
        """Type *text* as fast as possible."""
        ...

    def press_key(self, key: str) -> None:
        """Press and release a named key (e.g. ``"space"``, ``"Return"``)."""
        ...


# ---------------------------------------------------------------------------
# X11 backend — pynput
# ---------------------------------------------------------------------------


class PynputKeystrokeBackend:
    """Keystroke backend using pynput (X11 / XWayland)."""

    def __init__(self, typing_speed: float = 0.01):
        from pynput import keyboard

        self._keyboard = keyboard
        self._controller = keyboard.Controller()
        self.typing_speed = typing_speed

    def type_text(self, text: str) -> None:
        for char in text:
            self._controller.type(char)
            time.sleep(self.typing_speed)

    def type_text_fast(self, text: str) -> None:
        self._controller.type(text)

    def press_key(self, key: str) -> None:
        """Press a named key.

        *key* can be a :class:`pynput.keyboard.Key` member name (``"space"``,
        ``"enter"``, …) or a single character.
        """
        try:
            resolved = getattr(self._keyboard.Key, key)
        except AttributeError:
            resolved = key
        self._controller.press(resolved)
        self._controller.release(resolved)

    # Convenience: allow callers that still hold a pynput Key object
    def press_pynput_key(self, key) -> None:  # type: ignore[no-untyped-def]
        """Press a raw pynput key object (kept for backward compat)."""
        self._controller.press(key)
        self._controller.release(key)


# ---------------------------------------------------------------------------
# Wayland backend — wtype
# ---------------------------------------------------------------------------

# Map common key names to xkb names accepted by wtype
_WTYPE_KEY_MAP: dict[str, str] = {
    "space": "space",
    "enter": "Return",
    "return": "Return",
    "tab": "Tab",
    "backspace": "BackSpace",
    "delete": "Delete",
    "escape": "Escape",
    "esc": "Escape",
    "up": "Up",
    "down": "Down",
    "left": "Left",
    "right": "Right",
    "home": "Home",
    "end": "End",
    "page_up": "Prior",
    "page_down": "Next",
}


class WtypeKeystrokeBackend:
    """Keystroke backend using ``wtype`` (Wayland)."""

    def __init__(self, typing_speed: float = 0.01):
        self.typing_speed = typing_speed

    @staticmethod
    def _run_wtype(*args: str) -> None:
        subprocess.run(["wtype", *args], check=True)  # noqa: S603, S607

    def type_text(self, text: str) -> None:
        delay_ms = max(1, int(self.typing_speed * 1000))
        self._run_wtype("-d", str(delay_ms), "--", text)

    def type_text_fast(self, text: str) -> None:
        self._run_wtype("--", text)

    def press_key(self, key: str) -> None:
        xkb_name = _WTYPE_KEY_MAP.get(key.lower() if isinstance(key, str) else key, key)
        self._run_wtype("-k", xkb_name)


# ---------------------------------------------------------------------------
# Public handler — wraps a backend
# ---------------------------------------------------------------------------


class KeystrokeHandler:
    """
    Handles keyboard simulation for typing transcribed text.

    Features:
    - Configurable typing speed
    - Fast and slow typing modes
    - Special key support (space, enter, etc.)
    - Transparent X11 / Wayland backend selection
    """

    def __init__(
        self,
        typing_speed: float = 0.01,
        backend: DisplayBackend | None = None,
    ):
        """
        Initialise the keystroke handler.

        Args:
            typing_speed: Delay between keystrokes in seconds (0.01 = fast).
            backend: Force a specific display backend.  *None* means
                auto-detect.
        """
        self.typing_speed = typing_speed

        if backend is None:
            from whisper_to_me.display_backend import detect_backend

            backend = detect_backend()

        from whisper_to_me.display_backend import DisplayBackend

        if backend == DisplayBackend.WAYLAND:
            self._backend: KeystrokeBackend = WtypeKeystrokeBackend(typing_speed)
        else:
            self._backend = PynputKeystrokeBackend(typing_speed)

        self._display_backend = backend

        # Backward compat: expose keyboard_controller for X11 callers
        if isinstance(self._backend, PynputKeystrokeBackend):
            self.keyboard_controller = self._backend._controller

    # ------------------------------------------------------------------
    # Public API (unchanged from original)
    # ------------------------------------------------------------------

    def type_text(self, text: str, trailing_space: bool = False) -> None:
        """
        Simulate typing the given text.

        Args:
            text: The text to type.
            trailing_space: Whether to add a space after the text.
        """
        if not text or not text.strip():
            return

        self._backend.type_text(text.strip())

        if trailing_space:
            self.add_space()

    def type_text_fast(self, text: str, trailing_space: bool = False) -> None:
        """
        Type text without delays (faster).

        Args:
            text: The text to type.
            trailing_space: Whether to add a space after the text.
        """
        if not text or not text.strip():
            return

        self._backend.type_text_fast(text.strip())

        if trailing_space:
            self.add_space()

    def press_key(self, key) -> None:  # type: ignore[no-untyped-def]
        """Press a specific key.

        *key* may be a string name (``"space"``) or a ``pynput.keyboard.Key``
        object (X11 backend only).
        """
        if isinstance(key, str):
            self._backend.press_key(key)
        elif isinstance(self._backend, PynputKeystrokeBackend):
            self._backend.press_pynput_key(key)
        else:
            # Attempt to map pynput Key → string name for wtype
            self._backend.press_key(key.name if hasattr(key, "name") else str(key))

    def add_space(self) -> None:
        """Add a space."""
        self._backend.press_key("space")

    def add_newline(self) -> None:
        """Add a newline."""
        self._backend.press_key("enter")
