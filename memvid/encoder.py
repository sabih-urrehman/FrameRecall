"""
FrameRecallEncoder - Converts text segments into searchable QR-encoded video
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from tqdm import tqdm
import cv2
import numpy as np

from .utils import encode_to_qr, qr_to_frame, create_video_writer, chunk_text
from .index import IndexManager
from .config import get_default_config

logger = logging.getLogger(__name__)


class FrameRecallEncoder:
    """Encodes textual input into a video archive with QR encoding and index search"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_default_config()
        self.chunks = []
        self.index_manager = IndexManager(self.config)

    def add_chunks(self, chunks: List[str]):
        self.chunks.extend(chunks)
        logger.info(f"Inserted {len(chunks)} segments. Total: {len(self.chunks)}")

    def add_text(self, text: str, chunk_size: int = 500, overlap: int = 50):
        segments = chunk_text(text, chunk_size, overlap)
        self.add_chunks(segments)

    def build_video(self, output_file: str, index_file: str,
                    show_progress: bool = True) -> Dict[str, Any]:
        if not self.chunks:
            raise ValueError("No content to encode. Use add_chunks() beforehand.")

        output_path = Path(output_file)
        index_path = Path(index_file)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Encoding video with {len(self.chunks)} segments")

        video_cfg = self.config["video"]
        writer = create_video_writer(str(output_path), video_cfg)

        frame_ids = []
        iterator = enumerate(self.chunks)
        if show_progress:
            iterator = tqdm(iterator, total=len(self.chunks), desc="Encoding segments")

        try:
            for frame_id, content in iterator:
                payload = {
                    "id": frame_id,
                    "text": content,
                    "frame": frame_id
                }
                qr_img = encode_to_qr(json.dumps(payload), self.config)
                frame = qr_to_frame(qr_img, (video_cfg["frame_width"], video_cfg["frame_height"]))
                writer.write(frame)
                frame_ids.append(frame_id)

            logger.info("Generating retrieval index...")
            self.index_manager.add_chunks(self.chunks, frame_ids, show_progress)
            self.index_manager.save(str(index_path.with_suffix('')))

            stats = {
                "total_chunks": len(self.chunks),
                "total_frames": len(frame_ids),
                "video_file": str(output_path),
                "index_file": str(index_path),
                "video_size_mb": output_path.stat().st_size / (1024 * 1024)
                    if output_path.exists() else 0,
                "fps": video_cfg["fps"],
                "duration_seconds": len(frame_ids) / video_cfg["fps"],
                "index_stats": self.index_manager.get_stats()
            }

            logger.info(f"Video complete: {output_path}")
            logger.info(f"Length: {stats['duration_seconds']:.1f}s | Size: {stats['video_size_mb']:.1f}MB")

            return stats

        finally:
            writer.release()

    def clear(self):
        self.chunks = []
        self.index_manager = IndexManager(self.config)
        logger.info("Encoder state reset")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_chunks": len(self.chunks),
            "total_characters": sum(len(c) for c in self.chunks),
            "avg_chunk_size": np.mean([len(c) for c in self.chunks]) if self.chunks else 0,
            "config": self.config
        }

    @classmethod
    def from_file(cls, file_path: str, chunk_size: int = 500,
                  overlap: int = 50, config: Optional[Dict[str, Any]] = None) -> 'FrameRecallEncoder':
        encoder = cls(config)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        encoder.add_text(text, chunk_size, overlap)
        return encoder

    @classmethod
    def from_documents(cls, documents: List[str], chunk_size: int = 500,
                       overlap: int = 50, config: Optional[Dict[str, Any]] = None) -> 'FrameRecallEncoder':
        encoder = cls(config)
        for doc in documents:
            encoder.add_text(doc, chunk_size, overlap)
        return encoder
