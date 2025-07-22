"""
Audio Recording Module

Provides real-time audio recording functionality with optimized latency
for speech recognition applications.
"""

import numpy as np
import sounddevice as sd

from whisper_to_me.logger import get_logger


class AudioRecorder:
    """
    Real-time audio recorder optimized for speech recognition.

    Features:
    - Low-latency recording with pre-initialized audio stream
    - Configurable sample rate and audio device
    - Optimized for Whisper speech recognition (16kHz default)
    """

    def __init__(
        self,
        channels: int = 1,
        chunk_size: int = 512,
        device_id: int | None = None,
        device_name: str | None = None,
    ):
        """
        Initialize the audio recorder.

        Args:
            channels: Number of audio channels (1 for mono)
            chunk_size: Audio buffer size (smaller = lower latency)
            device_id: Audio input device ID (None for default)
            device_name: Device name for display (optional)
        """
        self.device_id = device_id
        self.device_name = device_name
        self.channels = channels
        self.chunk_size = chunk_size
        self.logger = get_logger()
        self.sample_rate: int = (
            16000  # Will be set to device default in _initialize_stream
        )

        self.is_recording = False
        self.audio_data: list[np.ndarray] = []
        self.stream: sd.InputStream | None = None

        self._initialize_stream()

    def _initialize_stream(self) -> None:
        """Pre-initialize and start the audio stream to eliminate startup latency"""
        # Get device info to find its default sample rate
        try:
            if self.device_id is not None:
                device_info = sd.query_devices(self.device_id)
            else:
                device_info = sd.query_devices(sd.default.device[0])

            self.sample_rate = int(device_info["default_samplerate"])
            self.logger.debug(
                f"Using device default sample rate: {self.sample_rate} Hz", "audio"
            )
        except Exception as e:
            # Fallback if we can't query the device
            self.sample_rate = 44100  # Common default
            self.logger.warning(
                f"Could not determine device sample rate, using {self.sample_rate} Hz: {e}",
                "audio",
            )

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._audio_callback,
            blocksize=self.chunk_size,
            dtype=np.float32,
            latency="low",  # Request low latency
            device=self.device_id,
        )
        # Start the stream immediately but don't record yet
        self.stream.start()

        # Get device name for user feedback
        if self.device_name:
            device_display = self.device_name
        elif self.device_id is None:
            device_display = "default"
        else:
            try:
                device_info = sd.query_devices(self.device_id)
                device_display = device_info["name"]
            except Exception:
                device_display = f"device {self.device_id}"

        self.logger.info(
            f"Audio stream started and ready for instant recording using {device_display}",
            "audio",
        )

    def _audio_callback(self, indata, frames, time, status):
        if status:
            self.logger.debug(f"Audio callback status: {status}", "audio")
        if self.is_recording:
            self.audio_data.append(indata.copy())

    def start_recording(self) -> None:
        if self.is_recording:
            return

        self.is_recording = True
        self.audio_data = []
        self.logger.recording_started()

    def stop_recording(self) -> np.ndarray | None:
        if not self.is_recording:
            return None

        self.is_recording = False

        if not self.audio_data:
            self.logger.warning("No audio data recorded", "audio")
            return None

        audio_array = np.concatenate(self.audio_data, axis=0)
        audio_array = audio_array.flatten()

        # Clear audio data to free memory
        self.audio_data.clear()

        self.logger.recording_stopped(
            len(audio_array) / self.sample_rate, len(audio_array)
        )
        return audio_array

    def get_audio_data_for_whisper(self, audio_array: np.ndarray) -> np.ndarray:
        if audio_array is None or len(audio_array) == 0:
            return np.array([])

        audio_array = audio_array.astype(np.float32)

        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        # Always resample to 16kHz for optimal Whisper performance
        if self.sample_rate != 16000:
            audio_array = self._resample_audio(audio_array, self.sample_rate, 16000)

        max_val = np.abs(audio_array).max()
        if max_val > 0:
            audio_array = audio_array / max_val

        return audio_array

    def _resample_audio(
        self, audio: np.ndarray, original_sr: int, target_sr: int
    ) -> np.ndarray:
        """
        Simple linear resampling for audio data.

        Args:
            audio: Input audio array
            original_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio array
        """
        if original_sr == target_sr:
            return audio

        # Calculate the resampling ratio
        ratio = target_sr / original_sr

        # Simple linear interpolation resampling
        original_length = len(audio)
        new_length = int(original_length * ratio)

        # Create new time indices
        old_indices = np.arange(original_length)
        new_indices = np.linspace(0, original_length - 1, new_length)

        # Interpolate
        resampled = np.interp(new_indices, old_indices, audio)

        return resampled.astype(np.float32)

    def is_recording_active(self) -> bool:
        return self.is_recording
