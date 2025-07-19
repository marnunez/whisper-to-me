import socket
import struct
import numpy as np
import io
import soundfile as sf
from typing import Optional


class WhisperClient:
    def __init__(self, socket_path="/tmp/whisper_server.sock"):
        self.socket_path = socket_path
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Optional[str]:
        """
        Send audio data to the Whisper server and get transcription back
        """
        try:
            # Convert audio to bytes (WAV format)
            audio_io = io.BytesIO()
            sf.write(audio_io, audio_data, sample_rate, format='WAV', subtype='PCM_16')
            audio_bytes = audio_io.getvalue()
            
            # Connect to server
            client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client_socket.connect(self.socket_path)
            
            # Send length + audio data
            length = struct.pack('!I', len(audio_bytes))
            client_socket.send(length + audio_bytes)
            
            # Receive response length
            response_length_bytes = client_socket.recv(4)
            if len(response_length_bytes) != 4:
                return None
            
            response_length = struct.unpack('!I', response_length_bytes)[0]
            
            # Receive response text
            response_bytes = b''
            while len(response_bytes) < response_length:
                chunk = client_socket.recv(min(4096, response_length - len(response_bytes)))
                if not chunk:
                    break
                response_bytes += chunk
            
            client_socket.close()
            
            if len(response_bytes) != response_length:
                return None
            
            return response_bytes.decode('utf-8')
            
        except Exception as e:
            print(f"Error communicating with server: {e}")
            return None
    
    def is_server_running(self) -> bool:
        """
        Check if the Whisper server is running
        """
        try:
            client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client_socket.connect(self.socket_path)
            client_socket.close()
            return True
        except:
            return False