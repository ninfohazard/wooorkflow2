import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union, List
import logging
from dataclasses import dataclass
import whisper
import ffmpeg
from pydantic import BaseModel
import wave
import numpy as np
from transformers import pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('caption_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CaptionFormat(BaseModel):
    start_time: float
    end_time: float
    text: str
    confidence: float
    speaker: Optional[str] = None

class CaptionMetadata(BaseModel):
    file_name: str
    duration: float
    language: str
    processed_date: datetime
    word_count: int
    confidence_score: float

class CaptionDocument(BaseModel):
    metadata: CaptionMetadata
    captions: List[CaptionFormat]

class CaptionProcessor:
    def __init__(self, model_type: str = "base", device: str = "cpu"):
        """
        Initialize the caption processor with specified model and device.
        
        Args:
            model_type: Type of Whisper model to use ("tiny", "base", "small", "medium", "large")
            device: Device to run the model on ("cpu" or "cuda")
        """
        self.model = whisper.load_model(model_type)
        self.device = device
        self.supported_formats = {
            'video': ['.mp4', '.mov', '.avi', '.mkv'],
            'audio': ['.mp3', '.wav', '.m4a', '.flac']
        }
        
        # Initialize speaker diarization model
        self.diarization_pipeline = pipeline(
            "audio-classification",
            model="pyannote/speaker-diarization",
            device=self.device
        )

    def validate_input(self, file_path: Path) -> bool:
        """Validate the input file format and accessibility."""
        try:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            file_extension = file_path.suffix.lower()
            valid_extensions = self.supported_formats['video'] + self.supported_formats['audio']
            
            if file_extension not in valid_extensions:
                raise ValueError(f"Unsupported file format: {file_extension}")

            # Try to open the file to verify accessibility
            with open(file_path, 'rb') as f:
                f.read(1024)  # Read first 1KB to verify file is accessible

            return True
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            raise

    def extract_audio(self, file_path: Path) -> Path:
        """Extract audio from video file or process audio file."""
        try:
            output_path = file_path.with_suffix('.wav')
            
            # Use ffmpeg to extract/convert audio
            stream = ffmpeg.input(str(file_path))
            stream = ffmpeg.output(stream, str(output_path),
                                 acodec='pcm_s16le',
                                 ac=1,
                                 ar='16k')
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
            
            return output_path
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            raise
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            raise

    def perform_diarization(self, audio_path: Path) -> Dict[str, List[dict]]:
        """Perform speaker diarization on the audio file."""
        try:
            diarization = self.diarization_pipeline(str(audio_path))
            return self._process_diarization_results(diarization)
        except Exception as e:
            logger.error(f"Speaker diarization failed: {e}")
            return {}

    def _process_diarization_results(self, diarization) -> Dict[str, List[dict]]:
        """Process raw diarization results into a structured format."""
        speaker_segments = {}
        for segment, track in diarization.itertracks(yield_label=True):
            if track not in speaker_segments:
                speaker_segments[track] = []
            speaker_segments[track].append({
                'start': segment.start,
                'end': segment.end
            })
        return speaker_segments

    def generate_captions(self, file_path: Path) -> CaptionDocument:
        """Generate captions from the audio/video file."""
        try:
            # Validate input
            self.validate_input(file_path)
            
            # Extract audio if needed
            audio_path = self.extract_audio(file_path)
            
            # Perform speech recognition
            result = self.model.transcribe(
                str(audio_path),
                language=None,  # Auto-detect language
                task="transcribe"
            )
            
            # Perform speaker diarization
            speaker_segments = self.perform_diarization(audio_path)
            
            # Process results
            captions = []
            for segment in result["segments"]:
                # Find speaker for this segment
                speaker = self._find_speaker(segment["start"], segment["end"], speaker_segments)
                
                caption = CaptionFormat(
                    start_time=segment["start"],
                    end_time=segment["end"],
                    text=segment["text"].strip(),
                    confidence=segment["confidence"],
                    speaker=speaker
                )
                captions.append(caption)
            
            # Create metadata
            metadata = CaptionMetadata(
                file_name=file_path.name,
                duration=result["duration"],
                language=result["language"],
                processed_date=datetime.now(),
                word_count=sum(len(c.text.split()) for c in captions),
                confidence_score=sum(c.confidence for c in captions) / len(captions)
            )
            
            return CaptionDocument(metadata=metadata, captions=captions)
            
        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            raise
        finally:
            # Cleanup temporary audio file if it was created
            if 'audio_path' in locals():
                audio_path.unlink(missing_ok=True)

    def _find_speaker(self, start: float, end: float, speaker_segments: Dict) -> Optional[str]:
        """Find the speaker for a given time segment."""
        for speaker, segments in speaker_segments.items():
            for segment in segments:
                if (start >= segment['start'] and start < segment['end']) or \
                   (end > segment['start'] and end <= segment['end']):
                    return speaker
        return None

    def save_captions(self, captions: CaptionDocument, output_path: Path, format: str = "json") -> Path:
        """Save captions to file in specified format."""
        try:
            output_path = output_path.with_suffix(f'.{format}')
            
            if format == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(captions.dict(), f, indent=2, default=str)
            
            elif format == "xml":
                root = ET.Element("caption_document")
                
                # Add metadata
                metadata = ET.SubElement(root, "metadata")
                for key, value in captions.metadata.dict().items():
                    ET.SubElement(metadata, key).text = str(value)
                
                # Add captions
                captions_elem = ET.SubElement(root, "captions")
                for caption in captions.captions:
                    caption_elem = ET.SubElement(captions_elem, "caption")
                    for key, value in caption.dict().items():
                        ET.SubElement(caption_elem, key).text = str(value)
                
                tree = ET.ElementTree(root)
                tree.write(output_path, encoding='utf-8', xml_declaration=True)
            
            else:
                raise ValueError(f"Unsupported output format: {format}")
            
            logger.info(f"Captions saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save captions: {e}")
            raise

    def load_captions(self, file_path: Path) -> CaptionDocument:
        """Load captions from a file."""
        try:
            format = file_path.suffix.lower()[1:]
            
            if format == "json":
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return CaptionDocument(**data)
                    
            elif format == "xml":
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                # Parse metadata
                metadata_elem = root.find("metadata")
                metadata = {elem.tag: elem.text for elem in metadata_elem}
                
                # Parse captions
                captions = []
                for caption_elem in root.find("captions"):
                    caption = {elem.tag: elem.text for elem in caption_elem}
                    captions.append(caption)
                
                return CaptionDocument(
                    metadata=CaptionMetadata(**metadata),
                    captions=[CaptionFormat(**caption) for caption in captions]
                )
            
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Failed to load captions: {e}")
            raise
