

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


class FramerecallEncoder:
    
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        
        self.config = config or get_default_config()
        self.chunks = []
        self.index_manager = IndexManager(self.config)
        
    def add_chunks(self, chunks: List[str]):
        
        self.chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} chunks. Total: {len(self.chunks)}")
    
    def add_text(self, text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
        
        chunks = chunk_text(text, chunk_size, overlap)
        self.add_chunks(chunks)
    
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
            raise ValueError("No chunks to encode. Use add_chunks() first.")
        
        output_path = Path(output_file)
        index_path = Path(index_file)
        

        output_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Building video with {len(self.chunks)} chunks")
        

        video_config = self.config["video"]
        writer = create_video_writer(str(output_path), video_config)
        
        frame_numbers = []
        
        try:

            chunks_iter = enumerate(self.chunks)
            if show_progress:
                chunks_iter = tqdm(chunks_iter, total=len(self.chunks), desc="Encoding chunks to video")
            
            for frame_num, chunk in chunks_iter:

                chunk_data = {
                    "id": frame_num,
                    "text": chunk,
                    "frame": frame_num
                }
                

                qr_image = encode_to_qr(json.dumps(chunk_data), self.config)
                

                frame = qr_to_frame(qr_image, (video_config["frame_width"], video_config["frame_height"]))
                

                writer.write(frame)
                frame_numbers.append(frame_num)
            

            logger.info("Building search index...")
            self.index_manager.add_chunks(self.chunks, frame_numbers, show_progress)
            

            self.index_manager.save(str(index_path.with_suffix('')))
            

            stats = {
                "total_chunks": len(self.chunks),
                "total_frames": len(frame_numbers),
                "video_file": str(output_path),
                "index_file": str(index_path),
                "video_size_mb": output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0,
                "fps": video_config["fps"],
                "duration_seconds": len(frame_numbers) / video_config["fps"],
                "index_stats": self.index_manager.get_stats()
            }
            
            logger.info(f"Successfully built video: {output_path}")
            logger.info(f"Video duration: {stats['duration_seconds']:.1f} seconds")
            logger.info(f"Video size: {stats['video_size_mb']:.1f} MB")
            
            return stats
            
        finally:
            writer.release()
    
    def clear(self):
        
        self.chunks = []
        self.index_manager = IndexManager(self.config)
        logger.info("Cleared all chunks")
    
    def get_stats(self) -> Dict[str, Any]:
        
        return {
            "total_chunks": len(self.chunks),
            "total_characters": sum(len(chunk) for chunk in self.chunks),
            "avg_chunk_size": np.mean([len(chunk) for chunk in self.chunks]) if self.chunks else 0,
            "config": self.config
        }
    
    @classmethod
    def from_file(cls, file_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP,
                  config: Optional[Dict[str, Any]] = None) -> 'FramerecallEncoder':
        
        encoder = cls(config)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        encoder.add_text(text, chunk_size, overlap)
        return encoder
    
    @classmethod
    def from_documents(cls, documents: List[str], chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP,
                       config: Optional[Dict[str, Any]] = None) -> 'FramerecallEncoder':
        
        encoder = cls(config)
        
        for doc in documents:
            encoder.add_text(doc, chunk_size, overlap)
        
        return encoder