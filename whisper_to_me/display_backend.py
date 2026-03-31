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


def get_focused_window(
    backend: DisplayBackend | None = None,
) -> tuple[str | None, str | None]:
    """
    Get the app_id and window title of the currently focused window.

    Returns:
        (app_id, title) — both lowercase. Either or both may be None.

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
                return None, None
            tree = json.loads(result.stdout)
            return _find_focused_window(tree)
        else:
            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowclassname"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode != 0:
                return None, None
            app = result.stdout.strip().lower() or None
            # X11: get title separately
            title = None
            try:
                title_result = subprocess.run(
                    ["xdotool", "getactivewindow", "getwindowname"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if title_result.returncode == 0:
                    title = title_result.stdout.strip() or None
            except Exception:
                pass
            return app, title
    except Exception:
        return None, None


def get_focused_app(backend: DisplayBackend | None = None) -> str | None:
    """Get the app_id of the currently focused window. Convenience wrapper."""
    app, _title = get_focused_window(backend)
    return app


def _find_focused_window(node: dict) -> tuple[str | None, str | None]:
    """Recursively find the focused window's app_id and title in a sway tree."""
    if node.get("focused") and (node.get("app_id") or node.get("window_properties")):
        app_id = node.get("app_id")
        if app_id:
            app = app_id.lower()
        else:
            props = node.get("window_properties", {})
            app = (props.get("class") or props.get("instance") or "").lower() or None
        title = node.get("name")
        return app, title
    for child in node.get("nodes", []) + node.get("floating_nodes", []):
        result = _find_focused_window(child)
        if result[0] is not None or result[1] is not None:
            return result
    return None, None
