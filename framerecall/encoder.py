

import json
import logging
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any
from tqdm import tqdm
import cv2
import numpy as np

from .utils import encode_to_qr, qr_to_frame, chunk_text
from .index import IndexManager
from .config import get_default_config, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from .docker_manager import DockerManager

logger = logging.getLogger(__name__)

class FramerecallEncoder:
    

    def __init__(self, config: Optional[Dict[str, Any]] = None, enable_docker=True):
        self.config = config or get_default_config()
        self.chunks = []
        self.index_manager = IndexManager()


        self.dcker_mngr = DockerManager() if enable_docker else None

        if self.dcker_mngr and not self.dcker_mngr.is_available():
            logger.info("Docker backend not available - using native encoding only")

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

    def create_video_writer(self, output_path: str, codec: str = "mp4v") -> cv2.VideoWriter:
        
        from .config import codec_parameters

        if codec not in codec_parameters:
            raise ValueError(f"Unsupported codec: {codec}")

        codec_config = codec_parameters[codec]


        opencv_codec_map = {
            "mp4v": "mp4v",
            "xvid": "XVID",
            "mjpg": "MJPG"
        }

        opencv_codec = opencv_codec_map.get(codec, codec)
        fourcc = cv2.VideoWriter_fourcc(*opencv_codec)

        return cv2.VideoWriter(
            output_path,
            fourcc,
            codec_config["video_fps"],
            (codec_config["frame_width"], codec_config["frame_height"])
        )

    def _generate_qr_frames(self, temp_dir: Path, show_progress: bool = True) -> Path:
        
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir()

        chunks_iter = enumerate(self.chunks)
        if show_progress:
            chunks_iter = tqdm(chunks_iter, total=len(self.chunks), desc="Generating QR frames")

        for frame_num, chunk in chunks_iter:
            chunk_data = {"id": frame_num, "text": chunk, "frame": frame_num}
            qr_image = encode_to_qr(json.dumps(chunk_data))
            frame_path = frames_dir / f"frame_{frame_num:06d}.png"
            qr_image.save(frame_path)

        created_frames = list(frames_dir.glob("frame_*.png"))
        print(f"🐛 FRAMES: {len(created_frames)} files in {frames_dir}")

        logger.info(f"Generated {len(self.chunks)} QR frames in {frames_dir}")
        return frames_dir

    def _build_ffmpeg_command(self, frames_dir: Path, output_file: Path, codec: str) -> List[str]:
        

        from .config import get_codec_parameters


        codec_config = get_codec_parameters(codec.lower())


        ffmpeg_codec_map = {
            "h265": "libx265", "hevc": "libx265",
            "h264": "libx264", "avc": "libx264",
            "av1": "libaom-av1", "vp9": "libvpx-vp9"
        }

        ffmpeg_codec = ffmpeg_codec_map.get(codec, codec)


        expected_ext = codec_config["video_file_type"]
        if not str(output_file).endswith(expected_ext):
            output_file = output_file.with_suffix(expected_ext)


        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(codec_config["video_fps"]),
            '-i', str(frames_dir / 'frame_%06d.png'),
            '-c:v', ffmpeg_codec,
            '-preset', codec_config["video_preset"],
            '-crf', str(codec_config["video_crf"]),
        ]


        if ffmpeg_codec in ['libx265', 'libx264']:

            target_width = codec_config["frame_width"]
            target_height = codec_config["frame_height"]
            cmd.extend(['-vf', f'scale={target_width}:{target_height}'])
            cmd.extend(['-pix_fmt', codec_config["pix_fmt"]])


            if codec_config.get("video_profile"):
                cmd.extend(['-profile:v', codec_config["video_profile"]])
        else:

            cmd.extend(['-pix_fmt', codec_config["pix_fmt"]])


        import os
        thread_count = min(os.cpu_count() or 4, 16)
        cmd.extend(['-threads', str(thread_count)])


        if codec_config.get("extra_ffmpeg_args"):
            extra_args = codec_config["extra_ffmpeg_args"]
            if isinstance(extra_args, str):

                if ffmpeg_codec == 'libx265':
                    extra_args = f"{extra_args}:threads={thread_count}"
                    cmd.extend(['-x265-params', extra_args])
                elif ffmpeg_codec == 'libx264':
                    extra_args = f"{extra_args}:threads={thread_count}"
                    cmd.extend(['-x264-params', extra_args])
            else:

                cmd.extend(extra_args)


        cmd.extend(['-movflags', '+faststart', '-avoid_negative_ts', 'make_zero'])

        cmd.append(str(output_file))
        return cmd

    def _encode_with_opencv(self, frames_dir: Path, output_file: Path, codec: str,
                            show_progress: bool = True) -> Dict[str, Any]:
        
        from .config import codec_parameters

        if codec not in codec_parameters:
            raise ValueError(f"Unsupported codec: {codec}")

        codec_config = codec_parameters[codec]

        if show_progress:
            logger.info(f"Encoding with OpenCV using {codec} codec...")


        writer = self.create_video_writer(str(output_file), codec)
        frame_numbers = []

        try:

            frame_files = sorted(frames_dir.glob("frame_*.png"))
            frame_iter = enumerate(frame_files)

            if show_progress:
                frame_iter = tqdm(frame_iter, total=len(frame_files), desc="Writing video frames")

            for frame_num, frame_file in frame_iter:

                frame = cv2.imread(str(frame_file))
                if frame is None:
                    logger.warning(f"Could not load frame: {frame_file}")
                    continue


                target_size = (codec_config["frame_width"], codec_config["frame_height"])
                if frame.shape[:2][::-1] != target_size:
                    frame = cv2.resize(frame, target_size)


                writer.write(frame)
                frame_numbers.append(frame_num)

            return {
                "backend": "opencv",
                "codec": codec,
                "total_frames": len(frame_numbers),
                "video_size_mb": output_file.stat().st_size / (1024 * 1024) if output_file.exists() else 0,
                "fps": codec_config["video_fps"],
                "duration_seconds": len(frame_numbers) / codec_config["video_fps"]
            }

        finally:
            writer.release()

    def _encode_with_ffmpeg(self, frames_dir: Path, output_file: Path, codec: str,
                            show_progress: bool = True, auto_build_docker: bool = True) -> Dict[str, Any]:
        

        from .config import codec_parameters

        print(f"🐛 FFMPEG: frames={frames_dir} → docker_mount={frames_dir.parent}")

        cmd = self._build_ffmpeg_command(frames_dir, output_file, codec)

        if self.dcker_mngr and self.dcker_mngr.should_use_docker(codec):
            if show_progress:
                logger.info(f"Encoding with Docker FFmpeg using {codec} codec...")

            result = self.dcker_mngr.execute_ffmpeg(
                cmd, frames_dir.parent, output_file, auto_build=auto_build_docker
            )

            frame_count = len(list(frames_dir.glob("frame_*.png")))
            result.update({
                "codec": codec,
                "total_frames": frame_count,
                "fps": codec_parameters[codec]["video_fps"],
                "duration_seconds": frame_count / codec_parameters[codec]["video_fps"]
            })

            return result

        else:
            if show_progress:
                logger.info(f"Encoding with native FFmpeg using {codec} codec...")

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"Native FFmpeg failed: {result.stderr}")

            frame_count = len(list(frames_dir.glob("frame_*.png")))
            return {
                "backend": "native_ffmpeg",
                "codec": codec,
                "total_frames": frame_count,
                "video_size_mb": output_file.stat().st_size / (1024 * 1024) if output_file.exists() else 0,
                "fps": codec_parameters[codec]["video_fps"],
                "duration_seconds": frame_count / codec_parameters[codec]["video_fps"]
            }

    def build_video(self, output_file: str, index_file: str,
                    codec: str = "mp4v", show_progress: bool = True,
                    auto_build_docker: bool = True, allow_fallback: bool = True) -> Dict[str, Any]:
        
        if not self.chunks:
            raise ValueError("No chunks to encode. Use add_chunks() first.")

        output_path = Path(output_file)
        index_path = Path(index_file)


        output_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Building video with {len(self.chunks)} chunks using {codec} codec")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)


            frames_dir = self._generate_qr_frames(temp_path, show_progress)

            try:
                from .config import codec_parameters
                print(f"🐛 DEBUG: Requested codec: '{codec}'")
                print(f"🐛 DEBUG: Available in full mapping: {list(codec_parameters.keys())}")

                if codec == "mp4v":

                    stats = self._encode_with_opencv(frames_dir, output_path, codec, show_progress)
                else:

                    stats = self._encode_with_ffmpeg(frames_dir, output_path, codec,
                                                     show_progress, auto_build_docker)

            except Exception as e:
                if allow_fallback and codec != "mp4v":
                    warnings.warn(f"{codec} encoding failed: {e}. Falling back to MP4V.", UserWarning)
                    stats = self._encode_with_opencv(frames_dir, output_path, "mp4v", show_progress)
                else:
                    raise


            if show_progress:
                logger.info("Building search index...")

            frame_numbers = list(range(len(self.chunks)))
            self.index_manager.add_chunks(self.chunks, frame_numbers, show_progress)


            self.index_manager.save(str(index_path.with_suffix('')))


            stats.update({
                "total_chunks": len(self.chunks),
                "video_file": str(output_path),
                "index_file": str(index_path),
                "index_stats": self.index_manager.get_stats()
            })

            if show_progress:
                logger.info(f"Successfully built video: {output_path}")
                logger.info(f"Video duration: {stats.get('duration_seconds', 0):.1f} seconds")
                logger.info(f"Video size: {stats.get('video_size_mb', 0):.1f} MB")

            return stats

    def clear(self):
        
        self.chunks = []
        self.index_manager = IndexManager()
        logger.info("Cleared all chunks")

    def get_stats(self) -> Dict[str, Any]:
        
        docker_status = "disabled"
        if self.dcker_mngr:
            docker_status = "available" if self.dcker_mngr.is_available() else "unavailable"

        return {
            "total_chunks": len(self.chunks),
            "total_characters": sum(len(chunk) for chunk in self.chunks),
            "avg_chunk_size": np.mean([len(chunk) for chunk in self.chunks]) if self.chunks else 0,
            "docker_status": docker_status,
            "supported_codecs": list(self.config["codec_parameters"].keys()),
            "config": self.config
        }

    def get_docker_status(self) -> str:
        
        if not self.dcker_mngr:
            return "Docker backend disabled"
        return self.dcker_mngr.get_status_message()

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