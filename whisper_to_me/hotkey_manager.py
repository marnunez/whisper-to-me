"""
Hotkey Manager Module

Handles global hotkey detection and management with support for different
recording modes.  Supports both X11 (pynput) and Wayland (evdev) backends.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

from pynput import keyboard

from whisper_to_me.config import AppConfig
from whisper_to_me.logger import get_logger

if TYPE_CHECKING:
    from whisper_to_me.display_backend import DisplayBackend


# ---------------------------------------------------------------------------
# Key-name ↔ evdev-keycode mapping  (kept outside the class so tests can
# inspect it and the import cost is zero when using pynput)
# ---------------------------------------------------------------------------

# Built lazily the first time the evdev backend is instantiated.
_PYNPUT_TO_EVDEV: dict[str, int] | None = None


def _build_pynput_to_evdev_map() -> dict[str, int]:
    """Build a mapping from pynput key-string tokens to evdev keycodes.

    pynput key strings look like ``<ctrl>``, ``<shift>``, ``<scroll_lock>``,
    ``<f1>`` or single characters such as ``r``.
    """
    import evdev.ecodes as ec

    m: dict[str, int] = {
        # Modifiers — pynput gives *generic* modifier names; map each to both
        # left and right evdev codes so we can match either physical key.
        "ctrl": ec.KEY_LEFTCTRL,
        "ctrl_l": ec.KEY_LEFTCTRL,
        "ctrl_r": ec.KEY_RIGHTCTRL,
        "shift": ec.KEY_LEFTSHIFT,
        "shift_l": ec.KEY_LEFTSHIFT,
        "shift_r": ec.KEY_RIGHTSHIFT,
        "alt": ec.KEY_LEFTALT,
        "alt_l": ec.KEY_LEFTALT,
        "alt_r": ec.KEY_RIGHTALT,
        "alt_gr": ec.KEY_RIGHTALT,
        "cmd": ec.KEY_LEFTMETA,
        "cmd_l": ec.KEY_LEFTMETA,
        "cmd_r": ec.KEY_RIGHTMETA,
        "super": ec.KEY_LEFTMETA,
        # Lock / special keys
        "scroll_lock": ec.KEY_SCROLLLOCK,
        "caps_lock": ec.KEY_CAPSLOCK,
        "num_lock": ec.KEY_NUMLOCK,
        "pause": ec.KEY_PAUSE,
        "print_screen": ec.KEY_SYSRQ,
        # Navigation
        "space": ec.KEY_SPACE,
        "enter": ec.KEY_ENTER,
        "return": ec.KEY_ENTER,
        "tab": ec.KEY_TAB,
        "backspace": ec.KEY_BACKSPACE,
        "delete": ec.KEY_DELETE,
        "insert": ec.KEY_INSERT,
        "home": ec.KEY_HOME,
        "end": ec.KEY_END,
        "page_up": ec.KEY_PAGEUP,
        "page_down": ec.KEY_PAGEDOWN,
        "up": ec.KEY_UP,
        "down": ec.KEY_DOWN,
        "left": ec.KEY_LEFT,
        "right": ec.KEY_RIGHT,
        "esc": ec.KEY_ESC,
        "escape": ec.KEY_ESC,
        # Function keys
        **{f"f{n}": getattr(ec, f"KEY_F{n}") for n in range(1, 25)},
    }

    # Character keys  a-z → KEY_A .. KEY_Z
    for ch in "abcdefghijklmnopqrstuvwxyz":
        m[ch] = getattr(ec, f"KEY_{ch.upper()}")
    # Digits 0-9 → KEY_0 .. KEY_9
    for ch in "0123456789":
        m[ch] = getattr(ec, f"KEY_{ch}")

    # Common symbols
    m["-"] = ec.KEY_MINUS
    m["="] = ec.KEY_EQUAL
    m["["] = ec.KEY_LEFTBRACE
    m["]"] = ec.KEY_RIGHTBRACE
    m["\\"] = ec.KEY_BACKSLASH
    m[";"] = ec.KEY_SEMICOLON
    m["'"] = ec.KEY_APOSTROPHE
    m[","] = ec.KEY_COMMA
    m["."] = ec.KEY_DOT
    m["/"] = ec.KEY_SLASH
    m["`"] = ec.KEY_GRAVE

    return m


def _pynput_to_evdev_map() -> dict[str, int]:
    global _PYNPUT_TO_EVDEV
    if _PYNPUT_TO_EVDEV is None:
        _PYNPUT_TO_EVDEV = _build_pynput_to_evdev_map()
    return _PYNPUT_TO_EVDEV


# Modifier evdev codes — we treat left/right as equivalent.
def _modifier_groups() -> list[set[int]]:
    import evdev.ecodes as ec

    return [
        {ec.KEY_LEFTCTRL, ec.KEY_RIGHTCTRL},
        {ec.KEY_LEFTSHIFT, ec.KEY_RIGHTSHIFT},
        {ec.KEY_LEFTALT, ec.KEY_RIGHTALT},
        {ec.KEY_LEFTMETA, ec.KEY_RIGHTMETA},
    ]


def _canonical_evdev_code(code: int, modifier_groups: list[set[int]]) -> int:
    """Normalise a modifier keycode to its *left* variant."""
    for group in modifier_groups:
        if code in group:
            return min(group)
    return code


# ---------------------------------------------------------------------------
# Parse pynput-style key strings into evdev codes
# ---------------------------------------------------------------------------


def _parse_pynput_key_string(key_str: str) -> list[int]:
    """Parse a pynput key string like ``<ctrl>+<shift>+r`` into evdev codes.

    Returns a sorted list of *canonical* evdev keycodes (left variant for
    modifiers).
    """
    mapping = _pynput_to_evdev_map()
    mod_groups = _modifier_groups()

    tokens = key_str.split("+")
    codes: list[int] = []
    for token in tokens:
        token = token.strip().strip("<>").lower()
        if not token:
            continue
        code = mapping.get(token)
        if code is None:
            raise ValueError(
                f"Cannot map pynput key token '{token}' to an evdev keycode"
            )
        codes.append(_canonical_evdev_code(code, mod_groups))
    return sorted(set(codes))


# ---------------------------------------------------------------------------
# Evdev hotkey backend
# ---------------------------------------------------------------------------


class _EvdevHotkeyBackend:
    """Global hotkey detection using evdev (works under Wayland)."""

    def __init__(self, config: AppConfig):
        import evdev

        self._evdev = evdev
        self._config = config
        self._logger = get_logger()

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Callbacks
        self.on_trigger_press: Callable | None = None
        self.on_trigger_tap: Callable | None = None
        self.on_discard_tap: Callable | None = None
        self.on_trigger_release: Callable | None = None

        # Parse configured keys
        self._trigger_codes: list[int] = []
        self._discard_codes: list[int] = []
        self._mod_groups = _modifier_groups()
        self._setup_hotkeys()

        # Runtime key state
        self._pressed: set[int] = set()

    # ----- setup ----------------------------------------------------------

    def _setup_hotkeys(self) -> None:
        self._trigger_codes = _parse_pynput_key_string(
            self._config.recording.trigger_key
        )
        self._discard_codes = _parse_pynput_key_string(
            self._config.recording.discard_key
        )
        self._tap_mode = self._config.recording.mode == "tap-mode"
        self._trigger_active = False

    def _find_keyboards(self) -> list:  # list[evdev.InputDevice]
        """Find input devices that look like keyboards."""
        evdev = self._evdev
        devices = []
        for path in evdev.list_devices():
            dev = evdev.InputDevice(path)
            caps = dev.capabilities()
            # EV_KEY = 1; require at least KEY_A(30) and KEY_ENTER(28)
            if 1 in caps and 30 in caps[1] and 28 in caps[1]:
                devices.append(dev)
        return devices

    # ----- listener -------------------------------------------------------

    def start_listening(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop_listening(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def join_listener(self) -> None:
        if self._thread is not None:
            self._thread.join()

    def _run(self) -> None:
        import select as _select

        keyboards = self._find_keyboards()
        if not keyboards:
            self._logger.error(
                "No keyboard devices found in /dev/input/.  "
                "Ensure the user is in the 'input' group.",
                "hotkey",
            )
            return

        names = ", ".join(d.name for d in keyboards)
        self._logger.info(f"Listening on evdev devices: {names}", "hotkey")

        fd_map = {dev.fd: dev for dev in keyboards}
        fds = list(fd_map.keys())

        while not self._stop_event.is_set():
            r, _, _ = _select.select(fds, [], [], 0.2)
            for fd in r:
                dev = fd_map[fd]
                try:
                    for event in dev.read():
                        if event.type == 1:  # EV_KEY
                            self._handle_event(event.code, event.value)
                except OSError:
                    # Device disconnected
                    pass

        for dev in keyboards:
            try:
                dev.close()
            except Exception:
                pass

    def _handle_event(self, code: int, value: int) -> None:
        """Process an EV_KEY event.  value: 0=up, 1=down, 2=repeat."""
        canon = _canonical_evdev_code(code, self._mod_groups)

        if value == 1:  # key down
            self._pressed.add(canon)
            self._check_hotkey_press()
        elif value == 0:  # key up
            self._check_hotkey_release(canon)
            self._pressed.discard(canon)

    def _check_hotkey_press(self) -> None:
        if self._tap_mode:
            if self._match(self._trigger_codes):
                if self.on_trigger_tap:
                    self.on_trigger_tap()
            elif self._match(self._discard_codes):
                if self.on_discard_tap:
                    self.on_discard_tap()
        else:
            if self._match(self._trigger_codes):
                self._trigger_active = True
                if self.on_trigger_press:
                    self.on_trigger_press()

    def _check_hotkey_release(self, released_code: int) -> None:
        if not self._tap_mode and self._trigger_active:
            # In push-to-talk: any key release while trigger was active → stop
            if self.on_trigger_release:
                self.on_trigger_release()
            self._trigger_active = False

    def _match(self, target: list[int]) -> bool:
        """Check whether all codes in *target* are currently pressed."""
        return all(c in self._pressed for c in target)

    # ----- config update --------------------------------------------------

    def update_config(self, new_config: AppConfig) -> None:
        self._config = new_config
        self._setup_hotkeys()


# ---------------------------------------------------------------------------
# Pynput hotkey backend (original implementation)
# ---------------------------------------------------------------------------


class _PynputHotkeyBackend:
    """Global hotkey detection using pynput (X11)."""

    def __init__(self, config: AppConfig):
        self._keyboard = keyboard
        self._config = config

        self.listener: keyboard.Listener | None = None
        self.trigger_hotkey: keyboard.HotKey | None = None
        self.discard_hotkey: keyboard.HotKey | None = None

        # Callbacks
        self.on_trigger_press: Callable | None = None
        self.on_trigger_tap: Callable | None = None
        self.on_discard_tap: Callable | None = None
        self.on_trigger_release: Callable | None = None

        self._setup_hotkeys()

    def _setup_hotkeys(self) -> None:
        keyboard = self._keyboard
        trigger_keys = keyboard.HotKey.parse(self._config.recording.trigger_key)
        discard_keys = keyboard.HotKey.parse(self._config.recording.discard_key)

        if self._config.recording.mode == "tap-mode":
            self.trigger_hotkey = keyboard.HotKey(
                trigger_keys, self._handle_trigger_tap
            )
            self.discard_hotkey = keyboard.HotKey(
                discard_keys, self._handle_discard_tap
            )
        else:
            self.trigger_hotkey = keyboard.HotKey(
                trigger_keys, self._handle_trigger_press
            )
            self.discard_hotkey = None

    # -- internal callbacks ------------------------------------------------

    def _handle_trigger_press(self) -> None:
        if self.on_trigger_press:
            self.on_trigger_press()

    def _handle_trigger_tap(self) -> None:
        if self.on_trigger_tap:
            self.on_trigger_tap()

    def _handle_discard_tap(self) -> None:
        if self.on_discard_tap:
            self.on_discard_tap()

    # -- key events --------------------------------------------------------

    def on_key_press(self, key) -> None:  # type: ignore[no-untyped-def]
        if not self.listener:
            return
        canonical_key = self.listener.canonical(key)
        if self.trigger_hotkey:
            self.trigger_hotkey.press(canonical_key)
        if self.discard_hotkey:
            self.discard_hotkey.press(canonical_key)

    def on_key_release(self, key) -> None:  # type: ignore[no-untyped-def]
        if not self.listener:
            return
        canonical_key = self.listener.canonical(key)
        if self.trigger_hotkey:
            self.trigger_hotkey.release(canonical_key)
        if self.discard_hotkey:
            self.discard_hotkey.release(canonical_key)

        if self._config.recording.mode == "push-to-talk" and self.on_trigger_release:
            self.on_trigger_release()

    # -- lifecycle ---------------------------------------------------------

    def start_listening(self) -> None:
        if self.listener is not None:
            return
        self.listener = self._keyboard.Listener(
            on_press=self.on_key_press, on_release=self.on_key_release
        )
        self.listener.start()

    def stop_listening(self) -> None:
        if self.listener is not None:
            self.listener.stop()
            self.listener = None

    def join_listener(self) -> None:
        if self.listener is not None:
            self.listener.join()

    def update_config(self, new_config: AppConfig) -> None:
        self._config = new_config
        self._setup_hotkeys()


# ---------------------------------------------------------------------------
# Public HotkeyManager — thin façade over the selected backend
# ---------------------------------------------------------------------------


class HotkeyManager:
    """
    Manages global hotkey detection and handling for recording triggers.

    Features:
    - Push-to-talk and tap-mode support
    - Transparent X11 / Wayland backend selection
    - Mode-specific callback handling
    - Clean listener lifecycle management
    """

    def __init__(
        self,
        config: AppConfig,
        backend: DisplayBackend | None = None,
    ):
        """
        Initialise the hotkey manager.

        Args:
            config: Application configuration containing key mappings.
            backend: Force a specific display backend.  *None* means
                auto-detect.
        """
        self.config = config

        if backend is None:
            from whisper_to_me.display_backend import detect_backend

            backend = detect_backend()

        from whisper_to_me.display_backend import DisplayBackend

        self._display_backend = backend

        if backend == DisplayBackend.WAYLAND:
            self._backend: _EvdevHotkeyBackend | _PynputHotkeyBackend = (
                _EvdevHotkeyBackend(config)
            )
        else:
            self._backend = _PynputHotkeyBackend(config)

        # Expose internal state for backward-compat with tests that inspect
        # pynput-specific attributes.
        if isinstance(self._backend, _PynputHotkeyBackend):
            self.listener = self._backend.listener
            self.trigger_hotkey = self._backend.trigger_hotkey
            self.discard_hotkey = self._backend.discard_hotkey
        else:
            self.listener = None
            self.trigger_hotkey = None
            self.discard_hotkey = None

        # Callbacks (mirrored to backend)
        self.on_trigger_press: Callable | None = None
        self.on_trigger_tap: Callable | None = None
        self.on_discard_tap: Callable | None = None
        self.on_trigger_release: Callable | None = None

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def set_callbacks(
        self,
        on_trigger_press: Callable | None = None,
        on_trigger_tap: Callable | None = None,
        on_discard_tap: Callable | None = None,
        on_trigger_release: Callable | None = None,
    ) -> None:
        """Set callback functions for hotkey events."""
        self.on_trigger_press = on_trigger_press
        self.on_trigger_tap = on_trigger_tap
        self.on_discard_tap = on_discard_tap
        self.on_trigger_release = on_trigger_release

        self._backend.on_trigger_press = on_trigger_press
        self._backend.on_trigger_tap = on_trigger_tap
        self._backend.on_discard_tap = on_discard_tap
        self._backend.on_trigger_release = on_trigger_release

    # ------------------------------------------------------------------
    # Internal callback handlers (for backward compat)
    # ------------------------------------------------------------------

    def _handle_trigger_press(self) -> None:
        if self.on_trigger_press:
            self.on_trigger_press()

    def _handle_trigger_tap(self) -> None:
        if self.on_trigger_tap:
            self.on_trigger_tap()

    def _handle_discard_tap(self) -> None:
        if self.on_discard_tap:
            self.on_discard_tap()

    # ------------------------------------------------------------------
    # Key events (pynput backend only — evdev handles its own loop)
    # ------------------------------------------------------------------

    def on_key_press(self, key) -> None:  # type: ignore[no-untyped-def]
        if isinstance(self._backend, _PynputHotkeyBackend):
            self._backend.on_key_press(key)

    def on_key_release(self, key) -> None:  # type: ignore[no-untyped-def]
        if isinstance(self._backend, _PynputHotkeyBackend):
            self._backend.on_key_release(key)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_listening(self) -> None:
        """Start the keyboard listener."""
        self._backend.start_listening()
        if isinstance(self._backend, _PynputHotkeyBackend):
            self.listener = self._backend.listener

    def stop_listening(self) -> None:
        """Stop the keyboard listener."""
        self._backend.stop_listening()
        if isinstance(self._backend, _PynputHotkeyBackend):
            self.listener = self._backend.listener  # now None

    def join_listener(self) -> None:
        """Wait for the listener to finish (blocking call)."""
        self._backend.join_listener()

    def update_config(self, new_config: AppConfig) -> None:
        """Update hotkey configuration."""
        self.config = new_config
        self._backend.update_config(new_config)
        if isinstance(self._backend, _PynputHotkeyBackend):
            self.trigger_hotkey = self._backend.trigger_hotkey
            self.discard_hotkey = self._backend.discard_hotkey

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def get_trigger_key_display(self) -> str:
        return self.config.recording.trigger_key

    def get_discard_key_display(self) -> str:
        return self.config.recording.discard_key

    def is_tap_mode(self) -> bool:
        return self.config.recording.mode == "tap-mode"
