"""Test display backend detection."""

import os
from unittest.mock import patch

import pytest

from whisper_to_me.display_backend import (
    DisplayBackend,
    detect_backend,
    resolve_backend,
)


class TestDetectBackend:
    """Test auto-detection logic."""

    def test_detect_wayland_via_session_type(self):
        """Detect Wayland when XDG_SESSION_TYPE is set."""
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}, clear=False):
            assert detect_backend() == DisplayBackend.WAYLAND

    def test_detect_x11_via_session_type(self):
        """Detect X11 when XDG_SESSION_TYPE is set."""
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}, clear=False):
            assert detect_backend() == DisplayBackend.X11

    def test_detect_wayland_via_display_env(self):
        """Fallback to WAYLAND_DISPLAY when session type is missing."""
        env = {"WAYLAND_DISPLAY": "wayland-0"}
        with patch.dict(os.environ, env, clear=False):
            # Remove XDG_SESSION_TYPE if present
            os.environ.pop("XDG_SESSION_TYPE", None)
            assert detect_backend() == DisplayBackend.WAYLAND

    def test_detect_x11_default(self):
        """Default to X11 when no hints present."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure clean env
            os.environ.pop("XDG_SESSION_TYPE", None)
            os.environ.pop("WAYLAND_DISPLAY", None)
            assert detect_backend() == DisplayBackend.X11

    def test_case_insensitive_session_type(self):
        """XDG_SESSION_TYPE should be matched case-insensitively."""
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "Wayland"}, clear=False):
            assert detect_backend() == DisplayBackend.WAYLAND


class TestResolveBackend:
    """Test resolve_backend with overrides."""

    def test_override_wayland(self):
        """Manual override to wayland."""
        assert resolve_backend("wayland") == DisplayBackend.WAYLAND

    def test_override_x11(self):
        """Manual override to x11."""
        assert resolve_backend("x11") == DisplayBackend.X11

    def test_override_case_insensitive(self):
        """Override is case-insensitive."""
        assert resolve_backend("WAYLAND") == DisplayBackend.WAYLAND
        assert resolve_backend("X11") == DisplayBackend.X11

    def test_override_invalid(self):
        """Invalid override raises ValueError."""
        with pytest.raises(ValueError, match="Unknown display backend"):
            resolve_backend("mir")

    def test_none_auto_detects(self):
        """None triggers auto-detection."""
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}, clear=False):
            assert resolve_backend(None) == DisplayBackend.WAYLAND
