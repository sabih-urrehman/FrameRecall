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
from .config import get_default_config, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP

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

    def add_text(self, text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
        segments = chunk_text(text, chunk_size, overlap)
        self.add_chunks(segments)

    def add_pdf(self, pdf_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
        try:
            import PyPDF2
        except ImportError:
            raise ImportError("PyPDF2 is required for PDF support. Install with: pip install PyPDF2")
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            logger.info(f"Extracting text from {num_pages} pages of {Path(pdf_path).name}")
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                text += page_text + "\n\n"
        if text.strip():
            self.add_text(text, chunk_size, overlap)
            logger.info(f"Added PDF content: {len(text)} characters from {Path(pdf_path).name}")
        else:
            logger.warning(f"No text extracted from PDF: {pdf_path}")

    def add_epub(self, epub_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("ebooklib and beautifulsoup4 are required for EPUB support. Install with: pip install ebooklib beautifulsoup4")
        if not Path(epub_path).exists():
            raise FileNotFoundError(f"EPUB file not found: {epub_path}")
        try:
            book = epub.read_epub(epub_path)
            text_content = []
            logger.info(f"Extracting text from EPUB: {Path(epub_path).name}")
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    for script in soup(["script", "style"]):
                        script.decompose()
                    text = soup.get_text()
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)
                    if text.strip():
                        text_content.append(text)
            full_text = "\n\n".join(text_content)
            if full_text.strip():
                self.add_text(full_text, chunk_size, overlap)
                logger.info(f"Added EPUB content: {len(full_text)} characters from {Path(epub_path).name}")
            else:
                logger.warning(f"No text extracted from EPUB: {epub_path}")
        except Exception as e:
            logger.error(f"Error processing EPUB {epub_path}: {e}")
            raise

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
    def from_file(cls, file_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
                  overlap: int = DEFAULT_OVERLAP, config: Optional[Dict[str, Any]] = None) -> 'FrameRecallEncoder':
        encoder = cls(config)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        encoder.add_text(text, chunk_size, overlap)
        return encoder

    @classmethod
    def from_documents(cls, documents: List[str], chunk_size: int = DEFAULT_CHUNK_SIZE,
                       overlap: int = DEFAULT_OVERLAP, config: Optional[Dict[str, Any]] = None) -> 'FrameRecallEncoder':
        encoder = cls(config)
        for doc in documents:
            encoder.add_text(doc, chunk_size, overlap)
        return encoder