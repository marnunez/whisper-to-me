"""Shared test fixtures and utilities."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from whisper_to_me import (
    AdvancedConfig,
    AppConfig,
    ConfigManager,
    GeneralConfig,
    RecordingConfig,
    UIConfig,
)


@pytest.fixture(autouse=True)
def isolated_xdg_runtime_dir():
    """
    Override XDG_RUNTIME_DIR to avoid single instance collisions with production.

    This ensures test runs don't interfere with any actual whisper-to-me instances
    that might be running on the system by using a separate lock file location.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        old_xdg_runtime = os.environ.get("XDG_RUNTIME_DIR")
        os.environ["XDG_RUNTIME_DIR"] = tmpdir
        yield tmpdir
        # Restore original value
        if old_xdg_runtime is None:
            os.environ.pop("XDG_RUNTIME_DIR", None)
        else:
            os.environ["XDG_RUNTIME_DIR"] = old_xdg_runtime


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for test configs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def default_test_config():
    """Create a default test configuration."""
    return AppConfig(
        general=GeneralConfig(
            model="tiny",  # Use tiny model for tests
            device="cpu",
            language="en",
            debug=False,
            last_profile="default",
            trailing_space=False,
        ),
        recording=RecordingConfig(
            mode="push-to-talk",
            trigger_key="<scroll_lock>",
            discard_key="<esc>",
            audio_device=None,
        ),
        ui=UIConfig(use_tray=True),
        advanced=AdvancedConfig(
            chunk_size=512,
            vad_filter=True,
            initial_prompt="",
            min_silence_duration_ms=2000,
            speech_pad_ms=400,
        ),
        profiles={},
    )


@pytest.fixture
def config_manager_with_temp_file(temp_config_dir):
    """Create a ConfigManager with a temporary config file."""
    config_file = temp_config_dir / "test_config.toml"
    return ConfigManager(config_file=str(config_file))


@pytest.fixture
def mock_sounddevice():
    """Mock sounddevice module for audio tests."""
    with patch("whisper_to_me.audio_recorder.sd") as mock_sd:
        # Set up default mock behaviors
        mock_sd.query_devices.return_value = {
            "name": "Default Device",
            "channels": 2,
            "default_samplerate": 44100,
        }
        mock_sd.check_input_settings.return_value = None
        mock_sd.default.device = [0, 0]  # Default input/output devices

        # Mock InputStream
        mock_stream = Mock()
        mock_stream.start = Mock()
        mock_stream.stop = Mock()
        mock_stream.close = Mock()
        mock_sd.InputStream.return_value = mock_stream

        yield mock_sd


@pytest.fixture
def mock_pystray():
    """Mock pystray for GUI tests."""
    with patch("whisper_to_me.menu_builder.pystray") as mock_pystray:
        # Set up Menu.SEPARATOR
        mock_pystray.Menu.SEPARATOR = "SEPARATOR"
        yield mock_pystray


@pytest.fixture
def mock_whisper_model():
    """Mock WhisperModel for speech processor tests."""
    with patch("whisper_to_me.speech_processor.WhisperModel") as mock_model_class:
        mock_model = Mock()
        mock_model.transcribe.return_value = (
            [{"text": "Test transcription"}],
            {"language": "en", "language_probability": 0.99},
        )
        mock_model_class.return_value = mock_model
        yield mock_model_class


@pytest.fixture
def mock_keyboard_hooks():
    """Mock keyboard hooks for hotkey tests."""
    with patch("whisper_to_me.hotkey_manager.keyboard") as mock_kb:
        # Mock HotKey class
        mock_hotkey = Mock()
        mock_kb.HotKey.return_value = mock_hotkey

        # Mock Listener class
        mock_listener = Mock()
        mock_listener.start = Mock()
        mock_listener.stop = Mock()
        mock_kb.Listener.return_value = mock_listener

        # Keep parse function real by default
        from pynput import keyboard

        mock_kb.HotKey.parse = keyboard.HotKey.parse

        yield mock_kb


def create_test_audio_devices():
    """Create test audio device data."""
    return [
        {
            "id": 0,
            "name": "Built-in Microphone",
            "hostapi": 0,
            "max_input_channels": 2,
            "default_samplerate": 44100.0,
        },
        {
            "id": 1,
            "name": "USB Headset",
            "hostapi": 0,
            "max_input_channels": 1,
            "default_samplerate": 48000.0,
        },
        {
            "id": 2,
            "name": "Virtual Cable",
            "hostapi": 1,
            "max_input_channels": 2,
            "default_samplerate": 16000.0,
        },
    ]


def create_test_hostapis():
    """Create test host API data."""
    return [
        {
            "id": 0,
            "name": "ALSA",
            "devices": [0, 1],
            "default_input_device": 0,
        },
        {
            "id": 1,
            "name": "JACK",
            "devices": [2],
            "default_input_device": 2,
        },
    ]
