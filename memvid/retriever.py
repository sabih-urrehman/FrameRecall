"""
FrameRecallRetriever - Rapid semantic lookup and frame decoding from matrix-encoded video
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from functools import lru_cache
import cv2
from .utils import (
    extract_frame, decode_qr, batch_extract_and_decode,
    extract_and_decode_cached
)
from .index import IndexManager
from .config import get_default_config

logger = logging.getLogger(__name__)

class FrameRecallRetriever:
    """Enables high-speed semantic search and content recovery from encoded video archive"""
    def __init__(self, video_file: str, index_file: str,
                 config: Optional[Dict[str, Any]] = None):
        self.video_file = str(Path(video_file).absolute())
        self.index_file = str(Path(index_file).absolute())
        self.config = config or get_default_config()
        self.index_manager = IndexManager(self.config)
        self.index_manager.load(str(Path(index_file).with_suffix('')))
        self._frame_cache = {}
        self._cache_size = self.config["retrieval"]["cache_size"]
        self._verify_video()
        logger.info(f"FrameRecallRetriever ready with {self.index_manager.get_stats()['total_chunks']} segments")

    def _verify_video(self):
        cap = cv2.VideoCapture(self.video_file)
        if not cap.isOpened():
            raise ValueError(f"Cannot access video: {self.video_file}")
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        logger.info(f"Video check: {self.total_frames} frames @ {self.fps:.2f} fps")

    def search(self, query: str, top_k: int = 5) -> List[str]:
        start = time.time()
        results = self.index_manager.search(query, top_k)
        frame_nums = list({res[2]["frame"] for res in results})
        decoded = self._decode_frames_parallel(frame_nums)
        chunks = []
        for _, _, meta in results:
            f = meta["frame"]
            try:
                data = json.loads(decoded[f]) if f in decoded else meta
                chunks.append(data["text"] if "text" in data else meta["text"])
            except Exception:
                chunks.append(meta["text"])
        logger.info(f"Query complete in {time.time() - start:.3f}s: '{query[:50]}...'")
        return chunks

    def get_chunk_by_id(self, chunk_id: int) -> Optional[str]:
        meta = self.index_manager.get_chunk_by_id(chunk_id)
        if not meta:
            return None
        frame = meta["frame"]
        data = self._decode_single_frame(frame)
        try:
            return json.loads(data)["text"] if data else meta["text"]
        except Exception:
            return meta["text"]

    def _decode_single_frame(self, frame_number: int) -> Optional[str]:
        if frame_number in self._frame_cache:
            return self._frame_cache[frame_number]
        result = extract_and_decode_cached(self.video_file, frame_number)
        if result and len(self._frame_cache) < self._cache_size:
            self._frame_cache[frame_number] = result
        return result

    def _decode_frames_parallel(self, frame_numbers: List[int]) -> Dict[int, str]:
        results = {}
        to_decode = [f for f in frame_numbers if f not in self._frame_cache]
        for f in frame_numbers:
            if f in self._frame_cache:
                results[f] = self._frame_cache[f]
        if not to_decode:
            return results
        decoded = batch_extract_and_decode(
            self.video_file,
            to_decode,
            max_workers=self.config["retrieval"]["max_workers"]
        )
        for f, d in decoded.items():
            results[f] = d
            if len(self._frame_cache) < self._cache_size:
                self._frame_cache[f] = d
        return results

    def search_with_metadata(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        start = time.time()
        results = self.index_manager.search(query, top_k)
        frame_nums = list({res[2]["frame"] for res in results})
        decoded = self._decode_frames_parallel(frame_nums)
        out = []
        for chunk_id, dist, meta in results:
            f = meta["frame"]
            try:
                data = json.loads(decoded[f]) if f in decoded else meta
                text = data["text"] if "text" in data else meta["text"]
            except Exception:
                text = meta["text"]
            out.append({
                "text": text,
                "score": 1.0 / (1.0 + dist),
                "chunk_id": chunk_id,
                "frame": f,
                "metadata": meta
            })
        logger.info(f"Metadata lookup done in {time.time() - start:.3f}s")
        return out

    def get_context_window(self, chunk_id: int, window_size: int = 2) -> List[str]:
        context = []
        for offset in range(-window_size, window_size + 1):
            cid = chunk_id + offset
            chunk = self.get_chunk_by_id(cid)
            if chunk:
                context.append(chunk)
        return context

    def prefetch_frames(self, frame_numbers: List[int]):
        targets = [f for f in frame_numbers if f not in self._frame_cache]
        if targets:
            logger.info(f"Prefetching {len(targets)} frames...")
            decoded = self._decode_frames_parallel(targets)
            logger.info(f"Prefetched {len(decoded)}")

    def clear_cache(self):
        self._frame_cache.clear()
        extract_and_decode_cached.cache_clear()
        logger.info("Frame cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "video_file": self.video_file,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "cache_size": len(self._frame_cache),
            "max_cache_size": self._cache_size,
            "index_stats": self.index_manager.get_stats()
        }