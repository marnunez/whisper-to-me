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

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from audio_recorder import AudioRecorder
from speech_processor import SpeechProcessor
from keystroke_handler import KeystrokeHandler
from tray_icon import TrayIcon
from single_instance import SingleInstance
from config import ConfigManager, AppConfig
from typing import Optional
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
        print("Initializing Whisper-to-Me...")
        print(
            f"Loading {self.config.general.model} model on {self.config.general.device}..."
        )
        print(f"Using profile: {self.config_manager.get_current_profile()}")

        # Initialize audio recorder with fallback for invalid devices
        device_id = self._resolve_audio_device()
        try:
            self.audio_recorder = AudioRecorder(device_id=device_id)
        except Exception as e:
            if device_id is not None:
                device_name = (
                    self.config.recording.audio_device.get("name")
                    if self.config.recording.audio_device
                    else f"ID {device_id}"
                )
                print(f"❌ Audio device '{device_name}' failed to initialize:")
                print(f"   Error: {e}")

                # Show available devices to help user
                available_devices = AudioRecorder.list_input_devices()
                if available_devices:
                    print("📱 Available audio devices:")
                    for device in available_devices:
                        print(
                            f"   {device['id']}: {device['name']} ({device['hostapi_name']})"
                        )

                print("🔄 Falling back to default audio device...")
                self.config.recording.audio_device = None
                self.audio_recorder = AudioRecorder(device_id=None)
            else:
                # Even default device failed
                print(f"❌ Failed to initialize default audio device: {e}")
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
                get_devices=self.get_available_devices,
                get_current_device=self.get_current_device,
            )
            self.tray_icon.start()

        # Format key display for user
        trigger_display = self.config.recording.trigger_key

        if self.config.recording.mode == "tap-mode":
            print(f"✓ Ready! Tap {trigger_display} to start/stop recording")
            print(
                f"  Press {self.config.recording.discard_key} while recording to discard"
            )
        else:
            print(f"✓ Ready! Press and hold {trigger_display} to record")
        print("Press Ctrl+C in terminal to exit")
        print("Note: This captures keys globally across all applications")

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
        print(f"🔄 Switching to profile: {profile_name}")

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
                print(
                    f"🌐 Language changed: {old_language} → {new_config.general.language}"
                )

            if (
                old_model != new_config.general.model
                or old_device != new_config.general.device
            ):
                print(
                    f"🔄 Model/device changed: {old_model}@{old_device} → {new_config.general.model}@{new_config.general.device}"
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

        print("✓ Profile switch completed")

    def get_available_devices(self):
        """Get list of available audio input devices."""
        return AudioRecorder.list_input_devices()

    def _resolve_audio_device(self) -> Optional[int]:
        """Resolve audio device config to actual device ID."""
        if self.config.recording.audio_device is None:
            return None

        device = AudioRecorder.find_device_by_config(self.config.recording.audio_device)
        if device:
            return device["id"]
        else:
            # Device not found, reset config
            print(
                f"⚠️  Configured audio device '{self.config.recording.audio_device.get('name')}' not found"
            )
            self.config.recording.audio_device = None
            return None

    def get_current_device(self):
        """Get current audio device info."""
        device_id = self._resolve_audio_device()
        if device_id is None:
            return AudioRecorder.get_default_input_device()
        else:
            devices = AudioRecorder.list_input_devices()
            for device in devices:
                if device["id"] == device_id:
                    return device
            return None

    def switch_audio_device(self, device_id: int):
        """Switch to a different audio device."""
        devices = AudioRecorder.list_input_devices()
        target_device = None

        for device in devices:
            if device["id"] == device_id:
                target_device = device
                break

        if not target_device:
            print(f"❌ Audio device {device_id} not found")
            return

        print(f"🎤 Switching to audio device: {target_device['name']}")

        try:
            # Test the device first
            test_recorder = AudioRecorder(device_id=device_id)

            # If successful, update config and replace recorder
            self.config.recording.audio_device = AudioRecorder.device_to_config(
                target_device
            )
            self.audio_recorder = test_recorder

            # Save config change
            self.config_manager.save_config()

            # Refresh tray menu to update device list
            if hasattr(self, "tray_icon") and self.tray_icon:
                self.tray_icon.refresh_menu()

            print("✓ Audio device switch completed")

        except Exception as e:
            print(f"❌ Failed to switch to device '{target_device['name']}':")
            print(f"   Error: {e}")

            # Suggest compatible sample rate if it's a sample rate issue
            if "sample rate" in str(e).lower() or "PaErrorCode -9997" in str(e):
                print(
                    f"   Device supports: {target_device.get('default_samplerate', 'unknown')} Hz"
                )
                print("   Try using a device that supports 16000 Hz sample rate")

            print("   Keeping current audio device.")

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
                print(f"🐛 Debug: Saved audio as {debug_filename}")

            # Prepare audio for Whisper and transcribe
            whisper_audio = self.audio_recorder.get_audio_data_for_whisper(audio_data)
            text, duration, language, confidence = self.speech_processor.transcribe(
                whisper_audio
            )

            if text and text.strip():
                print("✓ Transcription completed")
                if language:
                    print(
                        f"🌐 Detected language: {language} (confidence: {confidence:.2f})"
                    )
                if self.debug:
                    print(f"   '{text}' (duration: {duration:.2f}s)")
                self.keystroke_handler.type_text_fast(
                    text, self.config.general.trailing_space
                )
            else:
                print("✗ No speech detected")

            self.recording_counter += 1
        else:
            print("✗ No audio recorded")

    def _discard_recording(self):
        """Discard the current recording without transcription."""
        self.is_recording = False

        # Update tray icon
        if self.tray_icon:
            self.tray_icon.update_icon(recording=False)

        # Stop recording and discard audio
        _ = self.audio_recorder.stop_recording()

        print("🗑️ Recording discarded")

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

        print("Shutting down...")

        # Stop the tray icon
        if self.tray_icon:
            self.tray_icon.stop()

        print("Goodbye!")

        # Force immediate exit - keyboard listener won't stop cleanly from another thread
        import os

        os._exit(0)


def main():
    """
    Main entry point for the application.

    Parses command line arguments and initializes the WhisperToMe application.
    """
    import argparse

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

    # Initialize config manager
    config_manager = ConfigManager()

    # Handle configuration-only commands
    if args.config_path:
        print(f"Configuration file: {config_manager.get_config_file_path()}")
        return

    if args.list_profiles:
        print("Available profiles:")
        profiles = config_manager.get_profile_names()
        current = config_manager.get_current_profile()
        for profile in profiles:
            marker = "●" if profile == current else "○"
            print(f"  {marker} {profile}")
        return

    # List audio devices if requested
    if args.list_devices:
        print("Available audio input devices:")
        devices = AudioRecorder.list_input_devices()
        default = AudioRecorder.get_default_input_device()

        # Group devices by host API
        devices_by_hostapi = {}
        for device in devices:
            hostapi_name = device.get("hostapi_name", "Unknown")
            if hostapi_name not in devices_by_hostapi:
                devices_by_hostapi[hostapi_name] = []
            devices_by_hostapi[hostapi_name].append(device)

        # Display devices grouped by host API
        for hostapi_name in sorted(devices_by_hostapi.keys()):
            print(f"\n{hostapi_name}:")
            for device in devices_by_hostapi[hostapi_name]:
                default_mark = (
                    " (default)" if default and device["id"] == default["id"] else ""
                )
                print(f"  {device['id']}: {device['name']}{default_mark}")
                print(
                    f"      Channels: {device['channels']}, Sample rate: {device['default_samplerate']}"
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
            "hostapi_name": args.audio_device_hostapi
        }

    if args.tap_mode and args.push_to_talk:
        print("Error: Cannot specify both --tap-mode and --push-to-talk")
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
            print(f"✓ Profile '{args.create_profile}' created successfully")
        else:
            print(f"✗ Failed to create profile '{args.create_profile}'")
        return

    # Validate key configurations before starting
    try:
        # Validate trigger key combination
        config_manager.parse_key_combination(config.recording.trigger_key)
    except ValueError as e:
        print(f"Error in trigger key configuration: {e}")
        print(f"Invalid trigger key: '{config.recording.trigger_key}'")
        print("\nValid trigger keys include:")
        print("  - Single keys: scroll_lock, caps_lock, pause, tab, etc.")
        print("  - Single characters: a-z, 0-9, symbols")
        print("  - Combinations: ctrl+shift+r, alt+space, ctrl+-, etc.")
        return

    try:
        # Validate discard key (single key only)
        config_manager.parse_key_string(config.recording.discard_key)
    except ValueError as e:
        print(f"Error in discard key configuration: {e}")
        print(f"Invalid discard key: '{config.recording.discard_key}'")
        print("\nValid discard keys include:")
        print("  - Named keys: esc, delete, backspace, tab, etc.")
        print("  - Note: Discard key must be a single key, not a combination")
        return

    # Ensure single instance and run application
    with SingleInstance():
        app = WhisperToMe(config, config_manager)
        app.run()


if __name__ == "__main__":
    main()
