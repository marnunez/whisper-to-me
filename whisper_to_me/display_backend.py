"""
Display Backend Detection Module

Detects whether the session is running on Wayland or X11 and provides
a centralised point for backend selection.
"""

import json
import os
import subprocess
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


def get_focused_app(backend: DisplayBackend | None = None) -> str | None:
    """
    Get the app_id or WM_CLASS of the currently focused window.

    Returns a lowercase string identifying the focused application,
    or None if detection fails.

    Args:
        backend: Display backend to use. Auto-detected if None.
    """
    if backend is None:
        backend = detect_backend()

    try:
        if backend == DisplayBackend.WAYLAND:
            result = subprocess.run(
                ["swaymsg", "-t", "get_tree"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode != 0:
                return None
            tree = json.loads(result.stdout)
            return _find_focused_app(tree)
        else:
            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowclassname"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode != 0:
                return None
            return result.stdout.strip().lower() or None
    except Exception:
        return None


def _find_focused_app(node: dict) -> str | None:
    """Recursively find the focused window's app_id in a sway tree."""
    if node.get("focused") and (node.get("app_id") or node.get("window_properties")):
        # Prefer app_id (native Wayland), fall back to X11 class
        app_id = node.get("app_id")
        if app_id:
            return app_id.lower()
        props = node.get("window_properties", {})
        return (props.get("class") or props.get("instance") or "").lower() or None
    for child in node.get("nodes", []) + node.get("floating_nodes", []):
        result = _find_focused_app(child)
        if result:
            return result
    return None
