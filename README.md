# Whisper-to-Me

A real-time voice transcription tool that converts speech to text using
FasterWhisper and types the result directly into any application via simulated
keystrokes.

## Features

- **Push-to-talk and tap-to-start** recording modes with configurable hotkeys
- **Local speech recognition** (no internet required)
- **Global hotkey support** across all applications
- **Multiple language support** with auto-detection
- **Multiple audio device support**
- **System tray integration** with visual recording indicator
- **Single instance protection** - prevents multiple instances
- **Recording discard option** in tap mode (press Esc to cancel)
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

**Push-to-talk mode (default):**
4. Press and hold the trigger key to record
5. Release to transcribe and type the text

**Tap mode (--tap-mode):**
4. Tap the trigger key to start recording
5. Tap again to stop and transcribe, or press Esc to discard

### Command Line Options

```bash
whisper-to-me [options]

Options:
  --model MODEL         Whisper model size (tiny, base, small, medium, large-v3)
  --device DEVICE       Processing device (cpu, cuda)
  --key KEY            Trigger key (single key or combination, e.g.,
                       <scroll_lock>, <ctrl>+<shift>+r)
  --language LANG      Target language (auto, en, es, fr, etc.)
  --list-devices       List available audio input devices
  --audio-device ID    Audio device ID to use
  --debug             Save recorded audio files for debugging
  --no-tray           Disable system tray icon
  --tap-mode          Use tap-to-start/tap-to-stop instead of push-to-talk
  --discard-key KEY   Key to discard recording in tap mode (default: esc)
  --min-silence-duration-ms MS  Min silence duration to split segments
                                (default: 2000)
  --speech-pad-ms MS  Padding around detected speech (default: 400)
  --help              Show help message
```

### Examples

```bash
# Use default settings (large-v3 model, CUDA, scroll lock key, auto language)
whisper-to-me

# Use smaller model on CPU with caps lock trigger
whisper-to-me --model base --device cpu --key "<caps_lock>"

# Use key combination as trigger (Ctrl+Shift+R)
whisper-to-me --key "<ctrl>+<shift>+r"

# Use Ctrl+- (minus) as trigger
whisper-to-me --key "<ctrl>+-"

# Spanish transcription with debug mode
whisper-to-me --language es --debug --audio-device 2

# Run without system tray (terminal only)
whisper-to-me --no-tray

# List available audio devices
whisper-to-me --list-devices

# Use tap-to-start/tap-to-stop mode
whisper-to-me --tap-mode

# Tap mode with delete key to discard recordings
whisper-to-me --tap-mode --discard-key "<delete>"

# Fast VAD for quick commands
whisper-to-me --min-silence-duration-ms 500 --speech-pad-ms 100

# Slow VAD for dictation with pauses
whisper-to-me --min-silence-duration-ms 5000 --speech-pad-ms 800
```

## Configuration

Whisper-to-Me supports persistent configuration through a TOML config file and
multiple profiles for different use cases.

### Configuration File

**Location**: `~/.config/whisper-to-me/config.toml`

View the config file location:

```bash
whisper-to-me --config-path
```

### Configuration Sections

#### General Settings (`[general]`)

- **`model`**: Whisper model size
  - Options: `"tiny"`, `"base"`, `"small"`, `"medium"`, `"large-v3"` (default)
  - Affects: Transcription accuracy vs speed trade-off

- **`device`**: Processing device
  - Options: `"cpu"`, `"cuda"` (default)
  - Affects: Transcription speed (GPU acceleration)

- **`language`**: Target language
  - Options: `"auto"` (default), `"en"`, `"es"`, `"fr"`, etc.
  - Affects: Transcription accuracy for specific languages

- **`debug`**: Debug mode
  - Options: `true`, `false` (default)
  - Affects: Saves audio files for troubleshooting

#### Recording Settings (`[recording]`)

- **`mode`**: Recording mode
  - Options: `"push-to-talk"` (default), `"tap-mode"`
  - Affects: How recording is triggered

- **`trigger_key`**: Key combination to trigger recording
  - Default: `"<scroll_lock>"`
  - Examples: `"<caps_lock>"`, `"<ctrl>+<shift>+r"`, `"<alt>+<space>"`

- **`discard_key`**: Key to discard recording in tap mode
  - Default: `"<esc>"`
  - Options: Single keys like `"<delete>"`, `"<backspace>"`

- **`audio_device`**: Audio input device ID
  - Default: `""` (system default)
  - Use `--list-devices` to see available devices

#### UI Settings (`[ui]`)

- **`use_tray`**: System tray integration
  - Options: `true` (default), `false`
  - Affects: Shows microphone icon in system tray

#### Advanced Settings (`[advanced]`)

- **`chunk_size`**: Audio processing chunk size
  - Default: `512`
  - Affects: Real-time processing performance

- **`vad_filter`**: Voice Activity Detection filter
  - Default: `true`
  - Affects: Noise filtering during recording

- **`min_silence_duration_ms`**: Minimum silence duration to split audio segments
  - Default: `2000` (2 seconds)
  - Affects: How long pauses need to be before splitting speech
  - Lower values = more responsive, higher values = handles longer pauses

- **`speech_pad_ms`**: Padding around detected speech segments
  - Default: `400` (0.4 seconds)
  - Affects: How much audio is kept before and after speech
  - Lower values = tighter cropping, higher values = more context preserved

### Configuration Profiles

Create and manage multiple configuration profiles for different use cases:

#### Profile Management

```bash
# List available profiles
whisper-to-me --list-profiles

# Use specific profile
whisper-to-me --profile work

# Create new profile from current settings
whisper-to-me --model tiny --device cpu --create-profile quick
```

#### Example Profile Configuration

```toml
[general]
model = "large-v3"
device = "cuda"
language = "auto"
debug = false
last_profile = "default"

[recording]
mode = "push-to-talk"
trigger_key = "<scroll_lock>"
discard_key = "<esc>"
audio_device = ""

[ui]
use_tray = true

[advanced]
chunk_size = 512
vad_filter = true
min_silence_duration_ms = 2000
speech_pad_ms = 400

# Work profile - English only, medium model, caps lock trigger
[profiles.work]
[profiles.work.general]
language = "en"
model = "medium"
[profiles.work.recording]
trigger_key = "<caps_lock>"

# Spanish profile - Spanish language, large model
[profiles.spanish]
[profiles.spanish.general]
language = "es"
model = "large-v3"

# Quick profile - Fast transcription, CPU only
[profiles.quick]
[profiles.quick.general]
model = "tiny"
device = "cpu"
[profiles.quick.recording]
mode = "tap-mode"

# Fast VAD profile - Quick response for short commands
[profiles.fast_vad]
[profiles.fast_vad.advanced]
min_silence_duration_ms = 500   # 0.5 second pauses
speech_pad_ms = 100             # Minimal padding

# Dictation profile - Handles longer pauses in speech
[profiles.dictation]
[profiles.dictation.advanced]
min_silence_duration_ms = 5000  # 5 second pauses
speech_pad_ms = 800             # More padding for context
```

### Configuration Priority

Settings are applied in this order (highest to lowest priority):

1. Command line arguments
2. Profile settings
3. Base configuration file
4. Default values

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

### Model Performance

- **First Run**: Model download required (~39MB to 1.5GB depending on size)
- **Model Sizes and Trade-offs**:
  - `tiny`: ~39MB, ~1s latency, lowest accuracy, good for quick notes
  - `base`: ~74MB, ~1.5s latency, decent accuracy, balanced choice
  - `small`: ~244MB, ~2s latency, good accuracy, recommended minimum
  - `medium`: ~769MB, ~3s latency, high accuracy, good for professional use
  - `large-v3`: ~1550MB, ~5s latency, best accuracy, default choice

### Hardware Optimization

- **GPU (CUDA)**: 5-10x faster than CPU, essential for larger models
- **CPU Mode**: Viable for tiny/base models, expect higher latency
- **Memory Requirements**:
  - Tiny/Base: 1-2GB RAM
  - Small/Medium: 2-4GB RAM
  - Large-v3: 4-8GB RAM
  - Add 2-4GB for CUDA operations

### Audio Optimization

- **Microphone Quality**: USB microphones generally provide better results
- **Noise Environment**: VAD filter helps but quiet environment is best
- **Distance**: Keep microphone 6-12 inches from mouth

### Key Combinations

You can use key combinations as trigger keys:

```bash
# Single keys
whisper-to-me --key "<scroll_lock>"
whisper-to-me --key "<caps_lock>"
whisper-to-me --key "a"           # Single character

# Key combinations  
whisper-to-me --key "<ctrl>+<shift>+r"
whisper-to-me --key "<alt>+<space>"
whisper-to-me --key "<ctrl>+-"    # Ctrl + minus
whisper-to-me --key "<shift>+1"   # Shift + 1
```

Uses standard pynput format:

- **Named keys**: Wrap in angle brackets `<ctrl>`, `<alt>`, `<shift>`, `<esc>`,
  `<tab>`, etc.
- **Single characters**: Use directly `a`, `1`, `-`, `+`, etc.
- **Combinations**: Join with `+` symbol

## Troubleshooting

### Common Issues

1. **"Already running" error**: Only one instance allowed - check system
   tray or use `pkill whisper-to-me`
2. **Permission errors**: May need permissions for global key capture and
   microphone access
3. **Audio issues**: Check microphone permissions with `--list-devices`
4. **CUDA errors**: Install CUDA drivers or use `--device cpu`
5. **Trigger key not working**: Try different keys like `--key "<caps_lock>"`

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
