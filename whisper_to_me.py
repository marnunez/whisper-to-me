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
from pynput import keyboard


class WhisperToMe:
    """
    Main application class for voice-to-keystroke transcription.

    Handles global hotkey detection, audio recording coordination,
    speech transcription, and keystroke simulation.
    """

    def __init__(
        self,
        trigger_key=keyboard.Key.scroll_lock,
        model_size="large-v3",
        device="cuda",
        audio_device=None,
        debug=False,
        language=None,
        use_tray=True,
    ):
        """
        Initialize the Whisper-to-Me application.

        Args:
            trigger_key: Keyboard key to trigger recording (default: scroll_lock)
            model_size: Whisper model size (tiny, base, small, medium, large-v3)
            device: Processing device (cpu, cuda)
            audio_device: Audio input device ID (None for default)
            debug: Enable debug mode to save audio files
            language: Target language (None for auto-detection, 'en', 'es', 'fr', etc.)
            use_tray: Enable system tray icon
        """
        self.trigger_key = trigger_key
        self.is_recording = False
        self.debug = debug
        self.recording_counter = 0
        self.use_tray = use_tray
        self.tray_icon = None

        # Initialize components
        print("Initializing Whisper-to-Me...")
        print(f"Loading {model_size} model on {device}...")

        self.audio_recorder = AudioRecorder(device_id=audio_device)
        self.speech_processor = SpeechProcessor(
            model_size=model_size, device=device, language=language
        )
        self.keystroke_handler = KeystrokeHandler()
        
        # Initialize tray icon if enabled
        if self.use_tray:
            self.tray_icon = TrayIcon(on_quit=self.shutdown)
            self.tray_icon.start()

        print(f"✓ Ready! Press and hold {self.trigger_key} to record")
        print("Press Ctrl+C in terminal to exit")
        print("Note: This captures keys globally across all applications")

    def on_key_press(self, key):
        """
        Handle key press events. Start recording when trigger key is pressed.

        Args:
            key: The key that was pressed
        """
        if key == self.trigger_key and not self.is_recording:
            self.is_recording = True
            self.audio_recorder.start_recording()
            print("🎤 Recording...")
            
            # Update tray icon
            if self.tray_icon:
                self.tray_icon.update_icon(recording=True)

    def on_key_release(self, key):
        """
        Handle key release events. Stop recording and transcribe when trigger key is released.

        Args:
            key: The key that was released
        """
        if key == self.trigger_key and self.is_recording:
            self.is_recording = False
            
            # Update tray icon
            if self.tray_icon:
                self.tray_icon.update_icon(recording=False)

            # Stop recording and get audio
            audio_data = self.audio_recorder.stop_recording()

            if audio_data is not None and len(audio_data) > 0:
                print("🔄 Transcribing...")

                # Save audio for debugging if enabled
                if self.debug:
                    import soundfile as sf
                    from datetime import datetime

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    debug_filename = f"debug_recording_{timestamp}.wav"
                    sf.write(
                        debug_filename, audio_data, self.audio_recorder.sample_rate
                    )
                    print(f"🐛 Debug: Saved audio as {debug_filename}")

                # Prepare audio for Whisper and transcribe
                whisper_audio = self.audio_recorder.get_audio_data_for_whisper(
                    audio_data
                )
                text, duration = self.speech_processor.transcribe(whisper_audio)

                if text and text.strip():
                    print(f"✓ '{text}' (duration: {duration:.2f}s)")
                    self.keystroke_handler.type_text_fast(text)
                else:
                    print("✗ No speech detected")

                self.recording_counter += 1
            else:
                print("✗ No audio recorded")

    def run(self):
        """
        Start the main application loop.

        Sets up keyboard listeners and runs until interrupted with Ctrl+C.
        """
        try:
            with keyboard.Listener(
                on_press=self.on_key_press, on_release=self.on_key_release
            ) as listener:
                listener.join()
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.shutdown()
            
    def shutdown(self):
        """Clean shutdown of the application."""
        if self.tray_icon:
            self.tray_icon.stop()
        print("Goodbye!")


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
  %(prog)s                              # Use default settings
  %(prog)s --key caps_lock              # Use caps lock as trigger
  %(prog)s --model base --device cpu    # Use smaller model on CPU
  %(prog)s --debug --audio-device 2     # Debug mode with specific device
  %(prog)s --list-devices               # Show available audio devices
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
        default="scroll_lock",
        help="Trigger key (scroll_lock, pause, ctrl, alt, caps, etc)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )
    parser.add_argument(
        "--audio-device", type=int, help="Audio device ID to use (see --list-devices)"
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

    args = parser.parse_args()

    # List devices if requested
    if args.list_devices:
        print("Available audio input devices:")
        devices = AudioRecorder.list_input_devices()
        default = AudioRecorder.get_default_input_device()

        for device in devices:
            default_mark = (
                " (default)" if default and device["id"] == default["id"] else ""
            )
            print(f"  {device['id']}: {device['name']}{default_mark}")
            print(
                f"      Channels: {device['channels']}, Sample rate: {device['default_samplerate']}"
            )
        return

    # Parse and validate trigger key
    trigger_key_map = {
        "ctrl": keyboard.Key.ctrl_l,
        "alt": keyboard.Key.alt_l,
        "shift": keyboard.Key.shift_l,
        "caps": keyboard.Key.caps_lock,
        "caps_lock": keyboard.Key.caps_lock,
        "tab": keyboard.Key.tab,
        "scroll_lock": keyboard.Key.scroll_lock,
        "pause": keyboard.Key.pause,
    }

    trigger_key = trigger_key_map.get(args.key.lower())
    if trigger_key is None and hasattr(keyboard.Key, args.key.lower()):
        trigger_key = getattr(keyboard.Key, args.key.lower())
    elif trigger_key is None:
        trigger_key = keyboard.Key.scroll_lock
        print(f"Warning: Unknown key '{args.key}', using scroll_lock instead")

    # Process language parameter
    language = None if args.language.lower() == "auto" else args.language

    # Initialize and run application
    app = WhisperToMe(
        trigger_key=trigger_key,
        model_size=args.model,
        device=args.device,
        audio_device=args.audio_device,
        debug=args.debug,
        language=language,
        use_tray=not args.no_tray,
    )

    app.run()


if __name__ == "__main__":
    main()
