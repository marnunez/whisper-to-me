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
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1, 
                 chunk_size: int = 512, device_id: Optional[int] = None):
        """
        Initialize the audio recorder.
        
        Args:
            sample_rate: Audio sample rate in Hz (16000 optimal for Whisper)
            channels: Number of audio channels (1 for mono)
            chunk_size: Audio buffer size (smaller = lower latency)
            device_id: Audio input device ID (None for default)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device_id = device_id
        self.is_recording = False
        self.audio_data: List[np.ndarray] = []
        self.stream: Optional[sd.InputStream] = None
        
        self._initialize_stream()
    
    def _initialize_stream(self) -> None:
        """Pre-initialize and start the audio stream to eliminate startup latency"""
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._audio_callback,
            blocksize=self.chunk_size,
            dtype=np.float32,
            latency='low',  # Request low latency
            device=self.device_id
        )
        # Start the stream immediately but don't record yet
        self.stream.start()
        device_name = "default" if self.device_id is None else f"device {self.device_id}"
        print(f"Audio stream started and ready for instant recording using {device_name}")
        
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
        print("Recording started...")
    
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
        
        print(f"Recording stopped. Captured {len(audio_array)} samples ({len(audio_array)/self.sample_rate:.2f}s)")
        return audio_array
    
    def get_audio_data_for_whisper(self, audio_array: np.ndarray) -> np.ndarray:
        if audio_array is None or len(audio_array) == 0:
            return np.array([])
        
        audio_array = audio_array.astype(np.float32)
        
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        
        max_val = np.abs(audio_array).max()
        if max_val > 0:
            audio_array = audio_array / max_val
        
        return audio_array
    
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
        """
        devices = sd.query_devices()
        input_devices = []
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'default_samplerate': device['default_samplerate']
                })
        
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
        return {
            'id': default_device,
            'name': device_info['name'],
            'channels': device_info['max_input_channels'],
            'default_samplerate': device_info['default_samplerate']
        }