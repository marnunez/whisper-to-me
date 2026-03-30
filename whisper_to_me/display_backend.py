"""
Display Backend Detection Module

Detects whether the session is running on Wayland or X11 and provides
a centralised point for backend selection.
"""

import os
from enum import Enum

from whisper_to_me.logger import get_logger


class DisplayBackend(Enum):
    """Supported display backends."""

    X11 = "x11"
    WAYLAND = "wayland"


def detect_backend() -> DisplayBackend:
    """
    Auto-detect the display backend from environment variables.

    Checks ``$XDG_SESSION_TYPE`` first, then falls back to the presence of
    ``$WAYLAND_DISPLAY``.  Returns :attr:`DisplayBackend.X11` if neither
    hint points to Wayland.

    Returns:
        The detected :class:`DisplayBackend`.
    """
    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if session_type == "wayland":
        return DisplayBackend.WAYLAND
    if session_type == "x11":
        return DisplayBackend.X11

    # Fallback: check for WAYLAND_DISPLAY
    if os.environ.get("WAYLAND_DISPLAY"):
        return DisplayBackend.WAYLAND

    return DisplayBackend.X11


def resolve_backend(override: str | None = None) -> DisplayBackend:
    """
    Resolve the display backend, with an optional manual override.

    Args:
        override: ``"wayland"``, ``"x11"``, or *None* for auto-detection.

    Returns:
        The resolved :class:`DisplayBackend`.

    Raises:
        ValueError: If *override* is not a recognised value.
    """
    logger = get_logger()

    if override is not None:
        try:
            backend = DisplayBackend(override.lower())
        except ValueError:
            raise ValueError(
                f"Unknown display backend '{override}'. "
                f"Valid options: {', '.join(b.value for b in DisplayBackend)}"
            ) from None
        logger.info(f"Display backend (manual): {backend.value}", "backend")
        return backend

    backend = detect_backend()
    logger.info(f"Display backend (auto-detected): {backend.value}", "backend")
    return backend
