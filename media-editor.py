import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import json
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
import wave
import logging
from pydantic import BaseModel
from datetime import timedelta
import ffmpeg
import tempfile
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('media_editor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EditOperation(BaseModel):
    """Model for tracking edit operations"""
    operation_type: str
    start_time: float
    end_time: float
    parameters: Dict[str, Any]
    timestamp: float

class Caption(BaseModel):
    """Model for caption data"""
    start_time: float
    end_time: float
    text: str
    speaker: Optional[str]

class EditSession:
    """Tracks editing session state and history"""
    def __init__(self):
        self.operations: List[EditOperation] = []
        self.can_undo: bool = False
        self.can_redo: bool = False
        self.saved_state: bool = True
        
    def add_operation(self, operation: EditOperation):
        self.operations.append(operation)
        self.can_undo = True
        self.saved_state = False
        
    def undo(self) -> Optional[EditOperation]:
        if self.operations:
            op = self.operations.pop()
            self.can_redo = True
            return op
        return None

class MediaEditor:
    def __init__(self, temp_dir: Optional[Path] = None):
        self.temp_dir = temp_dir or Path(tempfile.mkdtemp())
        self.session = EditSession()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.supported_video_formats = {'.mp4', '.mov', '.avi', '.mkv'}
        self.supported_audio_formats = {'.mp3', '.wav', '.aac', '.m4a'}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        self.executor.shutdown(wait=True)

    def load_media(self, media_path: Path, captions_path: Optional[Path] = None) -> Tuple[Any, List[Caption]]:
        """Load media file and associated captions."""
        try:
            # Validate media file
            if not media_path.exists():
                raise FileNotFoundError(f"Media file not found: {media_path}")

            extension = media_path.suffix.lower()
            if extension in self.supported_video_formats:
                clip = VideoFileClip(str(media_path))
            elif extension in self.supported_audio_formats:
                clip = AudioFileClip(str(media_path))
            else:
                raise ValueError(f"Unsupported media format: {extension}")

            # Load captions if provided
            captions = []
            if captions_path and captions_path.exists():
                captions = self.load_captions(captions_path)

            return clip, captions

        except Exception as e:
            logger.error(f"Failed to load media: {e}")
            raise

    def load_captions(self, captions_path: Path) -> List[Caption]:
        """Load captions from file."""
        try:
            with open(captions_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [Caption(**caption) for caption in data['captions']]
        except Exception as e:
            logger.error(f"Failed to load captions: {e}")
            raise

    def trim_clip(self, clip: Any, start_time: float, end_time: float) -> Any:
        """Trim media clip to specified time range."""
        try:
            if end_time > clip.duration or start_time < 0:
                raise ValueError("Invalid trim range")

            trimmed_clip = clip.subclip(start_time, end_time)
            
            # Track operation
            self.session.add_operation(EditOperation(
                operation_type="trim",
                start_time=start_time,
                end_time=end_time,
                parameters={},
                timestamp=time.time()
            ))
            
            return trimmed_clip

        except Exception as e:
            logger.error(f"Failed to trim clip: {e}")
            raise

    def split_clip(self, clip: Any, split_points: List[float]) -> List[Any]:
        """Split media clip at specified points."""
        try:
            if not all(0 <= point <= clip.duration for point in split_points):
                raise ValueError("Invalid split points")

            split_points = sorted(split_points)
            segments = []
            
            start_time = 0
            for point in split_points:
                segments.append(clip.subclip(start_time, point))
                start_time = point
            segments.append(clip.subclip(start_time, clip.duration))
            
            # Track operation
            self.session.add_operation(EditOperation(
                operation_type="split",
                start_time=0,
                end_time=clip.duration,
                parameters={"split_points": split_points},
                timestamp=time.time()
            ))
            
            return segments

        except Exception as e:
            logger.error(f"Failed to split clip: {e}")
            raise

    def merge_clips(self, clips: List[Any]) -> Any:
        """Merge multiple clips into one."""
        try:
            if not clips:
                raise ValueError("No clips provided for merging")

            merged_clip = concatenate_videoclips(clips) if isinstance(clips[0], VideoFileClip) else concatenate_audioclips(clips)
            
            # Track operation
            self.session.add_operation(EditOperation(
                operation_type="merge",
                start_time=0,
                end_time=sum(clip.duration for clip in clips),
                parameters={"clip_count": len(clips)},
                timestamp=time.time()
            ))
            
            return merged_clip

        except Exception as e:
            logger.error(f"Failed to merge clips: {e}")
            raise

    def edit_caption(self, caption: Caption, new_text: str = None, 
                    new_start: float = None, new_end: float = None) -> Caption:
        """Edit caption text or timing."""
        try:
            # Create new caption with updates
            updated_caption = Caption(
                start_time=new_start if new_start is not None else caption.start_time,
                end_time=new_end if new_end is not None else caption.end_time,
                text=new_text if new_text is not None else caption.text,
                speaker=caption.speaker
            )
            
            # Validate timing
            if updated_caption.start_time >= updated_caption.end_time:
                raise ValueError("Invalid caption timing")
            
            # Track operation
            self.session.add_operation(EditOperation(
                operation_type="edit_caption",
                start_time=caption.start_time,
                end_time=caption.end_time,
                parameters={
                    "original_text": caption.text,
                    "new_text": updated_caption.text
                },
                timestamp=time.time()
            ))
            
            return updated_caption

        except Exception as e:
            logger.error(f"Failed to edit caption: {e}")
            raise

    def adjust_captions_timing(self, captions: List[Caption], offset: float) -> List[Caption]:
        """Adjust timing of all captions by specified offset."""
        try:
            adjusted_captions = []
            for caption in captions:
                adjusted_caption = Caption(
                    start_time=max(0, caption.start_time + offset),
                    end_time=caption.end_time + offset,
                    text=caption.text,
                    speaker=caption.speaker
                )
                adjusted_captions.append(adjusted_caption)
            
            # Track operation
            self.session.add_operation(EditOperation(
                operation_type="adjust_timing",
                start_time=0,
                end_time=0,
                parameters={"offset": offset},
                timestamp=time.time()
            ))
            
            return adjusted_captions

        except Exception as e:
            logger.error(f"Failed to adjust captions timing: {e}")
            raise

    def save_media(self, clip: Any, output_path: Path) -> None:
        """Save edited media file."""
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save based on media type
            if isinstance(clip, VideoFileClip):
                clip.write_videofile(str(output_path), codec='libx264', audio_codec='aac')
            else:
                clip.write_audiofile(str(output_path))
            
            self.session.saved_state = True
            
        except Exception as e:
            logger.error(f"Failed to save media: {e}")
            raise

    def save_captions(self, captions: List[Caption], output_path: Path) -> None:
        """Save edited captions."""
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            caption_data = {
                "version": "1.0",
                "captions": [caption.dict() for caption in captions]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(caption_data, f, indent=2)
            
            self.session.saved_state = True
            
        except Exception as e:
            logger.error(f"Failed to save captions: {e}")
            raise

    def undo_last_operation(self) -> bool:
        """Undo the last edit operation."""
        try:
            operation = self.session.undo()
            if operation:
                # Implement reverse operation based on operation type
                if operation.operation_type == "trim":
                    # Restore original clip
                    pass
                elif operation.operation_type == "split":
                    # Merge split segments
                    pass
                # ... implement other operation reversals
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to undo operation: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up temporary files."""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
