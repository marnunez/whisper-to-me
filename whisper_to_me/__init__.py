"""Whisper-to-Me source package."""

__version__ = "0.6.0"

# Lazy imports — submodules pull in heavy dependencies (sounddevice, pystray,
# faster-whisper, etc.) that may require system libraries not always present.
# Only the main entry point (main.py) imports what it needs at runtime.
