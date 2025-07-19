# Whisper-to-Me

A real-time voice transcription tool that converts speech to text using
FasterWhisper and types the result directly into any application via simulated
keystrokes.

## Features

- **Push-to-talk recording** with configurable hotkeys
- **Local speech recognition** (no internet required)
- **Global hotkey support** across all applications
- **Multiple language support** with auto-detection
- **Multiple audio device support**
- **System tray integration** with visual recording indicator
- **Single instance protection** - prevents multiple instances
- **Debug mode** for troubleshooting
- **High-accuracy transcription** using FasterWhisper
- **Real-time performance** optimized for responsiveness

## Requirements

- Python 3.12+
- CUDA-capable GPU (optional, CPU mode available)
- Audio input device (microphone)
- Linux operating system

## Installation

### From PyPI (Recommended)

```bash
# Install using pip
pip install whisper-to-me

# Or using uv (faster)
uv tool install whisper-to-me
```

### From Source

1. Install system dependencies:

```bash
# Ubuntu/Debian
sudo apt install portaudio19-dev libsndfile1-dev

# Fedora
sudo dnf install portaudio-devel libsndfile-devel

# Arch Linux
sudo pacman -S portaudio libsndfile
```

1. Clone and install:

```bash
git clone https://github.com/marnunez/whisper-to-me.git
cd whisper-to-me
uv tool install .
```

## Usage

### Basic Usage

Simply run the command after installation:

```bash
whisper-to-me
```

The application will:

1. Load the Whisper model (first run may take a moment)
2. Show a system tray icon (microphone)
3. Listen for the trigger key (Scroll Lock by default)
4. Press and hold the trigger key to record
5. Release to transcribe and type the text

### Command Line Options

```bash
whisper-to-me [options]

Options:
  --model MODEL         Whisper model size (tiny, base, small, medium, large-v3)
  --device DEVICE       Processing device (cpu, cuda)
  --key KEY            Trigger key (scroll_lock, pause, ctrl, alt, caps, etc.)
  --language LANG      Target language (auto, en, es, fr, etc.)
  --list-devices       List available audio input devices
  --audio-device ID    Audio device ID to use
  --debug             Save recorded audio files for debugging
  --no-tray           Disable system tray icon
  --help              Show help message
```

### Examples

```bash
# Use default settings (large-v3 model, CUDA, scroll lock key, auto language)
whisper-to-me

# Use smaller model on CPU with caps lock trigger
whisper-to-me --model base --device cpu --key caps_lock

# Spanish transcription with debug mode
whisper-to-me --language es --debug --audio-device 2

# Run without system tray (terminal only)
whisper-to-me --no-tray

# List available audio devices
whisper-to-me --list-devices
```

### System Tray

The system tray icon shows:

- **Gray microphone**: Ready to record
- **Red microphone**: Currently recording
- **Right-click menu**: View status and quit

## How It Works

1. **Single Instance Protection**: Ensures only one instance runs at a time
2. **Global Hotkey Detection**: Monitors for configured trigger key across all applications
3. **Audio Recording**: Captures microphone input while key is held
4. **Speech Processing**: Uses FasterWhisper for local speech-to-text
   conversion
5. **Keystroke Simulation**: Types the transcribed text directly into the
   active application
6. **System Integration**: Shows status in system tray with visual feedback

## Performance Notes

- **First Run**: May take longer as the Whisper model downloads (~1-3GB)
- **GPU Acceleration**: CUDA significantly improves transcription speed
- **Model Sizes**:
  - `tiny`: Fastest, least accurate (~39MB)
  - `base`: Good balance (~74MB)
  - `small`: Better accuracy (~244MB)
  - `medium`: High accuracy (~769MB)
  - `large-v3`: Best accuracy (~1550MB, default)
- **Audio Quality**: Better microphone input improves transcription accuracy

## Troubleshooting

### Common Issues

1. **"Already running" error**: Only one instance allowed - check system
   tray or use `pkill whisper-to-me`
2. **Permission errors**: May need permissions for global key capture and
   microphone access
3. **Audio issues**: Check microphone permissions with `--list-devices`
4. **CUDA errors**: Install CUDA drivers or use `--device cpu`
5. **Trigger key not working**: Try different keys like `--key caps_lock`

### Debug Mode

Use `--debug` to save recorded audio files for troubleshooting:

```bash
whisper-to-me --debug
```

### System Requirements Check

```bash
# Check audio devices
whisper-to-me --list-devices

# Test with smaller model
whisper-to-me --model tiny --device cpu
```

## Uninstallation

```bash
# If installed with pip
pip uninstall whisper-to-me

# If installed with uv tool
uv tool uninstall whisper-to-me
```

## Development

### Setup Development Environment

```bash
git clone https://github.com/marnunez/whisper-to-me.git
cd whisper-to-me
uv sync --all-extras --dev
```

### Run Tests

```bash
uv run pytest
```

### Code Quality

```bash
uv run ruff check
uv run ruff format
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Ensure code quality (`uv run ruff check && uv run pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the
[LICENSE](LICENSE) file for details.

## Acknowledgments

- [FasterWhisper](https://github.com/guillaumekln/faster-whisper) for fast
  speech recognition
- [OpenAI Whisper](https://github.com/openai/whisper) for the underlying model
- [PyNput](https://github.com/moses-palmer/pynput) for cross-platform input
  control
