#!/usr/bin/env python3
"""
Whisper-to-Me: Voice to Keystroke Application

A real-time voice transcription tool that converts speech to text using FasterWhisper
and types the result directly into any application via simulated keystrokes.

Features:
- Push-to-talk recording with configurable hotkeys
- Local speech recognition (no internet required)
- Global hotkey support across all applications
- Multiple audio device support
- Debug mode for troubleshooting

Usage:
    python whisper_to_me.py [options]

Examples:
    python whisper_to_me.py --key caps_lock
    python whisper_to_me.py --model base --device cpu
    python whisper_to_me.py --debug --audio-device 2
"""

import os

from whisper_to_me.audio_recorder import AudioRecorder
from whisper_to_me.audio_device_manager import AudioDeviceManager
from whisper_to_me.speech_processor import SpeechProcessor
from whisper_to_me.keystroke_handler import KeystrokeHandler
from whisper_to_me.tray_icon import TrayIcon
from whisper_to_me.single_instance import SingleInstance
from whisper_to_me.config import ConfigManager, AppConfig
from whisper_to_me.logger import get_logger, setup_logger, LogLevel
from pynput import keyboard


class WhisperToMe:
    """
    Main application class for voice-to-keystroke transcription.

    Handles global hotkey detection, audio recording coordination,
    speech transcription, and keystroke simulation.
    """

    def __init__(
        self,
        config: AppConfig,
        config_manager: ConfigManager,
    ):
        """
        Initialize the Whisper-to-Me application.

        Args:
            config: Application configuration object
            config_manager: Configuration manager for profile switching
        """
        self.config = config
        self.config_manager = config_manager
        self.is_recording = False
        self.recording_counter = 0
        self.tray_icon = None
        self.listener = None
        self.trigger_hotkey = None
        self.discard_hotkey = None
        self.trigger_pressed = (
            False  # Track if trigger combination is currently pressed
        )

        # Extract current settings from config
        self._update_from_config()

        # Initialize components
        self.logger = get_logger()
        self.logger.info("Initializing Whisper-to-Me...")
        self.logger.model_loaded(self.config.general.model, self.config.general.device)
        self.logger.info(
            f"Using profile: {self.config_manager.get_current_profile()}", "config"
        )

        # Initialize device manager
        self.device_manager = AudioDeviceManager(self.config.recording.audio_device)

        # Initialize audio recorder with managed device
        device = self.device_manager.get_current_device()
        try:
            self.audio_recorder = AudioRecorder(
                device_id=self.device_manager.get_current_device_id(),
                device_name=device.name if device else None,
            )
        except Exception as e:
            if device is not None:
                self.logger.error(
                    f"Audio device '{device.name}' failed to initialize: {e}", "audio"
                )

                # Show available devices to help user
                available_devices = self.device_manager.list_devices()
                if available_devices:
                    self.logger.info("Available audio devices:", "device")
                    for dev in available_devices:
                        self.logger.info(
                            f"   {dev.id}: {dev.name} ({dev.hostapi_name})", "device"
                        )

                self.logger.info(
                    "Falling back to default audio device...", "audio", "processing"
                )
                self.device_manager._device_config = None
                self.device_manager._current_device = None
                self.audio_recorder = AudioRecorder(device_id=None)
            else:
                # Even default device failed
                self.logger.critical(
                    f"Failed to initialize default audio device: {e}", "audio"
                )
                raise
        self.speech_processor = SpeechProcessor(
            model_size=self.config.general.model,
            device=self.config.general.device,
            language=self.config.general.language
            if self.config.general.language != "auto"
            else None,
        )
        self.keystroke_handler = KeystrokeHandler()

        # Initialize tray icon if enabled
        if self.config.ui.use_tray:
            self.tray_icon = TrayIcon(
                on_quit=self.shutdown,
                on_profile_change=self.switch_profile,
                get_profiles=self.config_manager.get_profile_names,
                get_current_profile=self.config_manager.get_current_profile,
                on_device_change=self.switch_audio_device,
                get_devices=lambda: self._convert_devices_for_tray(),
                get_current_device=lambda: self._convert_device_for_tray(
                    self.device_manager.get_current_device()
                ),
            )
            self.tray_icon.start()

        # Format key display for user
        trigger_display = self.config.recording.trigger_key

        self.logger.hotkey_info(
            trigger_display,
            self.config.recording.mode,
            self.config.recording.discard_key
            if self.config.recording.mode == "tap-mode"
            else None,
        )
        self.logger.info("Press Ctrl+C in terminal to exit", "app")
        self.logger.info(
            "Note: This captures keys globally across all applications", "app"
        )

    def _update_from_config(self):
        """Update instance variables from current configuration."""
        # Create HotKey objects for trigger and discard keys
        trigger_keys = keyboard.HotKey.parse(self.config.recording.trigger_key)
        discard_keys = keyboard.HotKey.parse(self.config.recording.discard_key)

        # Create HotKey instances with appropriate callbacks
        if self.config.recording.mode == "tap-mode":
            self.trigger_hotkey = keyboard.HotKey(trigger_keys, self._on_trigger_tap)
            self.discard_hotkey = keyboard.HotKey(discard_keys, self._on_discard_tap)
        else:
            self.trigger_hotkey = keyboard.HotKey(trigger_keys, self._on_trigger_press)
            # In push-to-talk mode, we need to handle release separately
            self.discard_hotkey = None  # Not used in push-to-talk mode
        self.debug = self.config.general.debug
        self.tap_mode = self.config.recording.mode == "tap-mode"

    def switch_profile(self, profile_name: str):
        """Switch to a different profile and update configuration."""
        self.logger.profile_switched(profile_name)

        # Apply the new profile
        new_config = self.config_manager.apply_profile(profile_name)

        # Update current config
        old_language = self.config.general.language
        old_model = self.config.general.model
        old_device = self.config.general.device

        self.config = new_config
        self._update_from_config()

        # Update speech processor if language/model/device changed
        if (
            old_language != new_config.general.language
            or old_model != new_config.general.model
            or old_device != new_config.general.device
        ):
            if old_language != new_config.general.language:
                self.logger.info(
                    f"Language changed: {old_language} → {new_config.general.language}",
                    "config",
                    "language",
                )

            if (
                old_model != new_config.general.model
                or old_device != new_config.general.device
            ):
                self.logger.info(
                    f"Model/device changed: {old_model}@{old_device} → {new_config.general.model}@{new_config.general.device}",
                    "config",
                    "model",
                )

            # Reinitialize speech processor with new settings
            self.speech_processor = SpeechProcessor(
                model_size=new_config.general.model,
                device=new_config.general.device,
                language=new_config.general.language
                if new_config.general.language != "auto"
                else None,
            )

        # Save the profile switch
        self.config_manager.save_config()

        self.logger.success("Profile switch completed", "profile")

    def _convert_devices_for_tray(self):
        """Convert AudioDevice objects to dict format expected by tray."""
        devices = self.device_manager.list_devices()
        return [
            {
                "id": dev.id,
                "name": dev.name,
                "hostapi_name": dev.hostapi_name,
                "channels": dev.channels,
                "default_samplerate": dev.sample_rate,
            }
            for dev in devices
        ]

    def _convert_device_for_tray(self, device):
        """Convert AudioDevice object to dict format expected by tray."""
        if device is None:
            # Try to get default device info
            default = self.device_manager.get_default_device()
            if default:
                return {
                    "id": default.id,
                    "name": default.name,
                    "hostapi_name": default.hostapi_name,
                    "channels": default.channels,
                    "default_samplerate": default.sample_rate,
                }
            return None
        return {
            "id": device.id,
            "name": device.name,
            "hostapi_name": device.hostapi_name,
            "channels": device.channels,
            "default_samplerate": device.sample_rate,
        }

    def switch_audio_device(self, device_id: int):
        """Switch to a different audio device."""
        # Find the device in our list
        devices = self.device_manager.list_devices()
        target_device = None

        for device in devices:
            if device.id == device_id:
                target_device = device
                break

        if not target_device:
            self.logger.error(f"Audio device {device_id} not found", "device")
            return

        self.logger.device_switched(target_device.name)

        # Use device manager to switch
        if self.device_manager.switch_device(target_device):
            try:
                # Create new audio recorder with the new device
                self.audio_recorder = AudioRecorder(
                    device_id=target_device.id, device_name=target_device.name
                )

                # Update config with new device
                self.config.recording.audio_device = (
                    self.device_manager.get_device_config()
                )
                self.config_manager.save_config()

                # Refresh tray menu to update device list
                if hasattr(self, "tray_icon") and self.tray_icon:
                    self.tray_icon.refresh_menu()

                self.logger.success("Audio device switch completed", "device")
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize recorder with device '{target_device.name}': {e}",
                    "audio",
                )
                self.logger.info("Keeping current audio device.", "device")
        else:
            self.logger.info("Keeping current audio device.", "device")

    def _on_trigger_tap(self):
        """Handle trigger key combination in tap mode."""
        if not self.is_recording:
            # Start recording
            self.is_recording = True
            self.audio_recorder.start_recording()
            # Update tray icon
            if self.tray_icon:
                self.tray_icon.update_icon(recording=True)
        else:
            # Stop recording and transcribe
            self._stop_and_transcribe()

    def _on_trigger_press(self):
        """Handle trigger key combination press in push-to-talk mode."""
        self.trigger_pressed = True
        if not self.is_recording:
            self.is_recording = True
            self.audio_recorder.start_recording()
            # Update tray icon
            if self.tray_icon:
                self.tray_icon.update_icon(recording=True)

    def _on_discard_tap(self):
        """Handle discard key combination in tap mode."""
        if self.is_recording:
            self._discard_recording()

    def on_key_press(self, key):
        """Handle key press events using HotKey state tracking."""
        # Let HotKey objects handle the state tracking
        canonical_key = self.listener.canonical(key)
        if self.trigger_hotkey:
            self.trigger_hotkey.press(canonical_key)
        if self.discard_hotkey:
            self.discard_hotkey.press(canonical_key)

    def on_key_release(self, key):
        """Handle key release events using HotKey state tracking."""
        # Let HotKey objects handle the state tracking
        canonical_key = self.listener.canonical(key)
        if self.trigger_hotkey:
            self.trigger_hotkey.release(canonical_key)
        if self.discard_hotkey:
            self.discard_hotkey.release(canonical_key)

        # In push-to-talk mode, stop recording when any key is released after trigger was pressed
        if not self.tap_mode and self.trigger_pressed and self.is_recording:
            self.trigger_pressed = False
            self._stop_and_transcribe()

    def _stop_and_transcribe(self):
        """Stop recording and transcribe the audio."""
        self.is_recording = False

        # Update tray icon
        if self.tray_icon:
            self.tray_icon.update_icon(recording=False)

        # Stop recording and get audio
        audio_data = self.audio_recorder.stop_recording()

        if audio_data is not None and len(audio_data) > 0:
            # Save audio for debugging if enabled
            if self.debug:
                import soundfile as sf
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                debug_filename = f"debug_recording_{timestamp}.wav"
                sf.write(debug_filename, audio_data, self.audio_recorder.sample_rate)
                self.logger.debug(f"Saved audio as {debug_filename}", "debug")

            # Prepare audio for Whisper and transcribe
            whisper_audio = self.audio_recorder.get_audio_data_for_whisper(audio_data)
            text, duration, language, confidence = self.speech_processor.transcribe(
                whisper_audio
            )

            if text and text.strip():
                self.logger.transcription_completed(text, language, confidence)
                if self.debug:
                    self.logger.debug(f"Duration: {duration:.2f}s", "speech")
                self.keystroke_handler.type_text_fast(
                    text, self.config.general.trailing_space
                )
            else:
                self.logger.warning("No speech detected", "speech")

            self.recording_counter += 1
        else:
            self.logger.warning("No audio recorded", "audio")

    def _discard_recording(self):
        """Discard the current recording without transcription."""
        self.is_recording = False

        # Update tray icon
        if self.tray_icon:
            self.tray_icon.update_icon(recording=False)

        # Stop recording and discard audio
        _ = self.audio_recorder.stop_recording()

        self.logger.info("Recording discarded", "audio", "🗑️")

    def run(self):
        """
        Start the main application loop.

        Sets up keyboard listeners and runs until interrupted with Ctrl+C.
        """
        try:
            self.listener = keyboard.Listener(
                on_press=self.on_key_press, on_release=self.on_key_release
            )
            self.listener.start()
            self.listener.join()
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        """Clean shutdown of the application."""
        # Prevent multiple shutdown calls
        if hasattr(self, "_shutting_down") and self._shutting_down:
            return
        self._shutting_down = True

        self.logger.application_shutdown()

        # Stop the tray icon
        if self.tray_icon:
            self.tray_icon.stop()

        # Force immediate exit - keyboard listener won't stop cleanly from another thread

        os._exit(0)


def main():
    """
    Main entry point for the application.

    Parses command line arguments and initializes the WhisperToMe application.
    """
    import argparse

    # Initialize config manager first to determine debug level
    config_manager = ConfigManager()

    # Setup logger early (will be reconfigured after arg parsing)
    setup_logger(min_level=LogLevel.INFO)
    logger = get_logger()

    parser = argparse.ArgumentParser(
        description="Whisper-to-Me: Voice to Keystroke Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Use default settings (push-to-talk)
  %(prog)s --key "<caps_lock>"          # Use caps lock as trigger
  %(prog)s --model base --device cpu    # Use smaller model on CPU
  %(prog)s --debug --audio-device 2     # Debug mode with specific device
  %(prog)s --tap-mode                   # Use tap-to-start/tap-to-stop mode
  %(prog)s --push-to-talk               # Use push-to-talk mode
  %(prog)s --tap-mode --discard-key "<delete>" # Tap mode with delete key to discard
  %(prog)s --list-devices               # Show available audio devices
  %(prog)s --profile work               # Use work profile
  %(prog)s --list-profiles              # Show available profiles
  %(prog)s --model tiny --create-profile quick # Create profile from settings
  %(prog)s --config-path                # Show config file location
        """,
    )

    parser.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model size (tiny, base, small, medium, large-v3)",
    )
    parser.add_argument(
        "--device", default="cuda", help="Processing device (cpu, cuda)"
    )
    parser.add_argument(
        "--key",
        default="<scroll_lock>",
        help="Trigger key (single key or combination, e.g., <scroll_lock>, <ctrl>+<shift>+r, <ctrl>+-)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )
    parser.add_argument("--audio-device-name", help="Audio device name to use")
    parser.add_argument(
        "--audio-device-hostapi",
        help="Audio device host API (e.g., 'ALSA', 'PulseAudio')",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Save recorded audio files for debugging"
    )
    parser.add_argument(
        "--language",
        default="auto",
        help="Target language (auto for detection, en, es, fr, etc.)",
    )
    parser.add_argument(
        "--no-tray",
        action="store_true",
        help="Disable system tray icon",
    )
    parser.add_argument(
        "--tap-mode",
        action="store_true",
        help="Use tap-to-start/tap-to-stop instead of push-to-talk",
    )
    parser.add_argument(
        "--push-to-talk",
        action="store_true",
        help="Use push-to-talk mode instead of tap-to-start/tap-to-stop",
    )
    parser.add_argument(
        "--discard-key",
        default="<esc>",
        help="Key to discard recording in tap mode (default: <esc>)",
    )
    parser.add_argument(
        "--profile",
        help="Use specific profile",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available profiles and exit",
    )
    parser.add_argument(
        "--create-profile",
        help="Create new profile from current CLI arguments",
    )
    parser.add_argument(
        "--config-path",
        action="store_true",
        help="Show configuration file path and exit",
    )
    parser.add_argument(
        "--trailing-space",
        action="store_true",
        help="Add a trailing space after transcribed text",
    )

    args = parser.parse_args()

    # Reconfigure logger with debug level if requested
    if args.debug:
        setup_logger(min_level=LogLevel.DEBUG)
        logger = get_logger()

    # Handle configuration-only commands
    if args.config_path:
        logger.info(
            f"Configuration file: {config_manager.get_config_file_path()}", "config"
        )
        return

    if args.list_profiles:
        logger.info("Available profiles:", "config")
        profiles = config_manager.get_profile_names()
        current = config_manager.get_current_profile()

        # Load base config to show profile details
        base_config = config_manager.load_config()

        for profile in profiles:
            marker = "●" if profile == current else "○"
            logger.info(f"  {marker} {profile}", "config")

            if profile == "default":
                # Show base config details
                logger.info(
                    f"      Model: {base_config.general.model}, Device: {base_config.general.device}",
                    "config",
                )
                logger.info(
                    f"      Mode: {base_config.recording.mode}, Language: {base_config.general.language}",
                    "config",
                )
            else:
                # Apply profile and show its specific settings
                profile_config = config_manager.apply_profile(profile)
                logger.info(
                    f"      Model: {profile_config.general.model}, Device: {profile_config.general.device}",
                    "config",
                )
                logger.info(
                    f"      Mode: {profile_config.recording.mode}, Language: {profile_config.general.language}",
                    "config",
                )
        return

    # List audio devices if requested
    if args.list_devices:
        logger.info("Available audio input devices:", "device")
        device_manager = AudioDeviceManager()
        devices_by_hostapi = device_manager.group_devices_by_hostapi()
        default = device_manager.get_default_device()

        # Display devices grouped by host API
        for hostapi_name in sorted(devices_by_hostapi.keys()):
            logger.info(f"\n{hostapi_name}:", "device")
            for device in devices_by_hostapi[hostapi_name]:
                default_mark = (
                    " (default)" if default and device.id == default.id else ""
                )
                logger.info(f"  {device.id}: {device.name}{default_mark}", "device")
                logger.info(
                    f"      Channels: {device.channels}, Sample rate: {device.sample_rate}",
                    "device",
                )
        return

    # Load base configuration
    config = config_manager.load_config()

    # Apply profile if specified
    profile_to_use = args.profile or config.general.last_profile
    if profile_to_use and profile_to_use != "default":
        config = config_manager.apply_profile(profile_to_use)

    # Override config with CLI arguments
    def override_if_provided(config_value, cli_value):
        """Return CLI value if provided, otherwise config value."""
        return cli_value if cli_value is not None else config_value

    # Override general settings
    config.general.model = override_if_provided(config.general.model, args.model)
    config.general.device = override_if_provided(config.general.device, args.device)
    config.general.debug = override_if_provided(config.general.debug, args.debug)

    # Handle language override
    if args.language and args.language != "auto":
        config.general.language = args.language
    elif args.language == "auto":
        config.general.language = "auto"

    # Handle trailing space override
    config.general.trailing_space = override_if_provided(
        config.general.trailing_space, args.trailing_space
    )

    # Override recording settings
    config.recording.trigger_key = override_if_provided(
        config.recording.trigger_key, args.key
    )
    config.recording.discard_key = override_if_provided(
        config.recording.discard_key, args.discard_key
    )
    # Handle audio device override (convert from name/hostapi to config if specified)
    if args.audio_device_name:
        config.recording.audio_device = {
            "name": args.audio_device_name,
            "hostapi_name": args.audio_device_hostapi,
        }

    if args.tap_mode and args.push_to_talk:
        logger.error("Cannot specify both --tap-mode and --push-to-talk", "config")
        return
    elif args.tap_mode:
        config.recording.mode = "tap-mode"
    elif args.push_to_talk:
        config.recording.mode = "push-to-talk"

    # Override UI settings
    if args.no_tray:
        config.ui.use_tray = False

    # Handle profile creation
    if args.create_profile:
        success = config_manager.create_profile(args.create_profile, config)
        if success:
            logger.success(
                f"Profile '{args.create_profile}' created successfully", "profile"
            )
        else:
            logger.error(f"Failed to create profile '{args.create_profile}'", "profile")
        return

    # Validate key configurations before starting
    try:
        # Validate trigger key combination
        config_manager.parse_key_combination(config.recording.trigger_key)
    except ValueError as e:
        logger.error(f"Error in trigger key configuration: {e}", "config")
        logger.error(f"Invalid trigger key: '{config.recording.trigger_key}'", "config")
        logger.info("Valid trigger keys include:", "config")
        logger.info(
            "  - Single keys: scroll_lock, caps_lock, pause, tab, etc.", "config"
        )
        logger.info("  - Single characters: a-z, 0-9, symbols", "config")
        logger.info("  - Combinations: ctrl+shift+r, alt+space, ctrl+-, etc.", "config")
        return

    try:
        # Validate discard key (single key only)
        config_manager.parse_key_string(config.recording.discard_key)
    except ValueError as e:
        logger.error(f"Error in discard key configuration: {e}", "config")
        logger.error(f"Invalid discard key: '{config.recording.discard_key}'", "config")
        logger.info("Valid discard keys include:", "config")
        logger.info("  - Named keys: esc, delete, backspace, tab, etc.", "config")
        logger.info(
            "  - Note: Discard key must be a single key, not a combination", "config"
        )
        return

    # Ensure single instance and run application
    with SingleInstance():
        app = WhisperToMe(config, config_manager)
        app.run()


if __name__ == "__main__":
    main()
