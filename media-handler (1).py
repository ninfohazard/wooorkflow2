import os
import hashlib
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import cv2
import pyaudio
import wave
from cryptography.fernet import Fernet
import psutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StorageType(Enum):
    LOCAL = "local"
    EXTERNAL = "external"
    CLOUD = "cloud"

@dataclass
class StorageDevice:
    name: str
    path: str
    type: StorageType
    available_space: int
    is_writable: bool

class MediaFormat(Enum):
    VIDEO = ["mp4", "mov", "avi", "mkv"]
    AUDIO = ["mp3", "wav", "aac", "m4a"]
    
    @classmethod
    def get_all_supported_formats(cls) -> List[str]:
        return [fmt for formats in cls.__members__.values() for fmt in formats.value]

class MediaHandler:
    def __init__(self, max_file_size: int = 1024 * 1024 * 100):  # 100MB default
        self.max_file_size = max_file_size
        self.temp_dir = Path("./temp_encrypted")
        self.encryption_key = Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key)
        self._setup_temp_directory()

    def _setup_temp_directory(self) -> None:
        """Create encrypted temporary directory if it doesn't exist."""
        try:
            self.temp_dir.mkdir(exist_ok=True, parents=True)
            os.chmod(self.temp_dir, 0o700)  # Restrictive permissions
        except Exception as e:
            logger.error(f"Failed to create temporary directory: {e}")
            raise

    def detect_storage_devices(self) -> List[StorageDevice]:
        """Detect and list all available storage devices."""
        storage_devices = []
        
        try:
            # Local storage
            for partition in psutil.disk_partitions():
                if partition.mountpoint:
                    usage = psutil.disk_usage(partition.mountpoint)
                    storage_devices.append(
                        StorageDevice(
                            name=f"Local Disk ({partition.device})",
                            path=partition.mountpoint,
                            type=StorageType.LOCAL if "fixed" in partition.opts else StorageType.EXTERNAL,
                            available_space=usage.free,
                            is_writable=os.access(partition.mountpoint, os.W_OK)
                        )
                    )
        except Exception as e:
            logger.error(f"Error detecting storage devices: {e}")
            
        return storage_devices

    def validate_file(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """Validate file type, size, and integrity."""
        try:
            if not file_path.exists():
                return False, "File does not exist"

            # Check file size
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                return False, f"File exceeds maximum size of {self.max_file_size} bytes"

            # Check file extension
            file_ext = file_path.suffix.lower().lstrip('.')
            if file_ext not in MediaFormat.get_all_supported_formats():
                return False, f"Unsupported file format: {file_ext}"

            # Validate file integrity
            with open(file_path, 'rb') as f:
                # Read first few bytes to verify file signature
                header = f.read(32)
                # Add specific format validation here based on file type

            return True, None
        except Exception as e:
            return False, f"Validation error: {e}"

    def start_video_capture(self, output_path: Path) -> None:
        """Handle live video capture from device camera."""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Failed to open camera")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, 20.0, (640, 480))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                out.write(frame)
                cv2.imshow('Recording', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            # Encrypt the captured video
            self.encrypt_file(output_path)
            
        except Exception as e:
            logger.error(f"Video capture error: {e}")
            raise

    def start_audio_capture(self, output_path: Path, duration: int = 10) -> None:
        """Handle live audio capture from device microphone."""
        try:
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 2
            RATE = 44100

            p = pyaudio.PyAudio()
            stream = p.open(format=FORMAT,
                          channels=CHANNELS,
                          rate=RATE,
                          input=True,
                          frames_per_buffer=CHUNK)

            frames = []
            for _ in range(0, int(RATE / CHUNK * duration)):
                data = stream.read(CHUNK)
                frames.append(data)

            stream.stop_stream()
            stream.close()
            p.terminate()

            wf = wave.open(str(output_path), 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            # Encrypt the captured audio
            self.encrypt_file(output_path)

        except Exception as e:
            logger.error(f"Audio capture error: {e}")
            raise

    def encrypt_file(self, file_path: Path) -> Path:
        """Encrypt file and save to temporary directory."""
        try:
            with open(file_path, 'rb') as file:
                file_data = file.read()
            
            encrypted_data = self.fernet.encrypt(file_data)
            encrypted_path = self.temp_dir / f"{file_path.stem}_encrypted{file_path.suffix}"
            
            with open(encrypted_path, 'wb') as file:
                file.write(encrypted_data)
            
            return encrypted_path
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            raise

    def decrypt_file(self, encrypted_path: Path, output_path: Path) -> None:
        """Decrypt file from temporary directory."""
        try:
            with open(encrypted_path, 'rb') as file:
                encrypted_data = file.read()
            
            decrypted_data = self.fernet.decrypt(encrypted_data)
            
            with open(output_path, 'wb') as file:
                file.write(decrypted_data)
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up temporary files."""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            raise
