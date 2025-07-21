"""
Audio Recording Module

Provides real-time audio recording functionality with optimized latency
for speech recognition applications.
"""

import sounddevice as sd
import numpy as np
from typing import Optional, List, Dict, Any


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
        sample_rate: Optional[int] = None,
        channels: int = 1,
        chunk_size: int = 512,
        device_id: Optional[int] = None,
    ):
        """
        Initialize the audio recorder.

        Args:
            sample_rate: Audio sample rate in Hz (None for device-adaptive)
            channels: Number of audio channels (1 for mono)
            chunk_size: Audio buffer size (smaller = lower latency)
            device_id: Audio input device ID (None for default)
        """
        self.device_id = device_id
        self.channels = channels
        self.chunk_size = chunk_size

        # Get the best sample rate for this device
        self.sample_rate = self._get_best_sample_rate(sample_rate)

        self.is_recording = False
        self.audio_data: List[np.ndarray] = []
        self.stream: Optional[sd.InputStream] = None

        self._initialize_stream()

    def _get_best_sample_rate(self, requested_rate: Optional[int]) -> int:
        """
        Get the best sample rate for the device.

        Args:
            requested_rate: Requested sample rate (None for auto-detect)

        Returns:
            Best supported sample rate
        """
        # If user specified a rate, try that first
        if requested_rate is not None:
            try:
                # Test if the requested rate works
                sd.check_input_settings(
                    device=self.device_id, samplerate=requested_rate
                )
                return requested_rate
            except Exception:
                pass

        # Get device info to find its default sample rate
        try:
            if self.device_id is not None:
                device_info = sd.query_devices(self.device_id)
            else:
                device_info = sd.query_devices(sd.default.device[0])

            device_rate = int(device_info["default_samplerate"])

            # Prefer 16kHz (Whisper's native rate), then device default
            preferred_rates = [16000, device_rate]

            for rate in preferred_rates:
                try:
                    sd.check_input_settings(device=self.device_id, samplerate=rate)
                    print(f"Using sample rate: {rate} Hz")
                    return rate
                except Exception:
                    continue

            # Fallback to device default if nothing else works
            print(f"Using device default sample rate: {device_rate} Hz")
            return device_rate

        except Exception as e:
            print(f"Could not determine device sample rate, using 16000 Hz: {e}")
            return 16000

    def _initialize_stream(self) -> None:
        """Pre-initialize and start the audio stream to eliminate startup latency"""
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
        device_name = (
            "default" if self.device_id is None else f"device {self.device_id}"
        )
        print(
            f"Audio stream started and ready for instant recording using {device_name}"
        )

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}")
        if self.is_recording:
            self.audio_data.append(indata.copy())

    def start_recording(self) -> None:
        if self.is_recording:
            return

        self.is_recording = True
        self.audio_data = []
        print("🎤 Recording started...")

    def stop_recording(self) -> Optional[np.ndarray]:
        if not self.is_recording:
            return None

        self.is_recording = False

        if not self.audio_data:
            print("No audio data recorded")
            return None

        audio_array = np.concatenate(self.audio_data, axis=0)
        audio_array = audio_array.flatten()

        # Clear audio data to free memory
        self.audio_data.clear()

        print(
            f"🔄 Recording stopped. Captured {len(audio_array)} samples ({len(audio_array) / self.sample_rate:.2f}s)"
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

    @staticmethod
    def list_input_devices() -> List[Dict[str, Any]]:
        """
        List all available audio input devices.

        Returns:
            List of dictionaries containing device information:
            - id: Device ID number
            - name: Device name
            - channels: Maximum input channels
            - default_samplerate: Default sample rate
            - hostapi: Host API index
            - hostapi_name: Host API name (e.g., 'ALSA', 'PulseAudio')
        """
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        input_devices = []

        for i, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                hostapi_index = device["hostapi"]
                hostapi_name = (
                    hostapis[hostapi_index]["name"]
                    if hostapi_index < len(hostapis)
                    else "Unknown"
                )

                input_devices.append(
                    {
                        "id": i,
                        "name": device["name"],
                        "channels": device["max_input_channels"],
                        "default_samplerate": device["default_samplerate"],
                        "hostapi": hostapi_index,
                        "hostapi_name": hostapi_name,
                    }
                )

        return input_devices

    @staticmethod
    def get_default_input_device() -> Optional[Dict[str, Any]]:
        """
        Get information about the default input device.

        Returns:
            Dictionary with default device info, or None if no default device
        """
        default_device = sd.default.device[0]
        if default_device is None:
            return None

        device_info = sd.query_devices(default_device)
        hostapis = sd.query_hostapis()
        hostapi_index = device_info["hostapi"]
        hostapi_name = (
            hostapis[hostapi_index]["name"]
            if hostapi_index < len(hostapis)
            else "Unknown"
        )

        return {
            "id": default_device,
            "name": device_info["name"],
            "channels": device_info["max_input_channels"],
            "default_samplerate": device_info["default_samplerate"],
            "hostapi": hostapi_index,
            "hostapi_name": hostapi_name,
        }
