"""
FrameRecall Index Manager - Embedding index and metadata structure for fast lookup
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path
from tqdm import tqdm

from .config import get_default_config

logger = logging.getLogger(__name__)


class IndexManager:
    """Controls embedding operations, FAISS index usage, and context metadata"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_default_config()
        self.embedding_model = SentenceTransformer(self.config["embedding"]["model"])
        self.dimension = self.config["embedding"]["dimension"]
        self.index = self._create_index()
        self.metadata = []
        self.chunk_to_frame = {}
        self.frame_to_chunks = {}

    def _create_index(self) -> faiss.Index:
        index_type = self.config["index"]["type"]
        if index_type == "Flat":
            idx = faiss.IndexFlatL2(self.dimension)
        elif index_type == "IVF":
            quantizer = faiss.IndexFlatL2(self.dimension)
            idx = faiss.IndexIVFFlat(quantizer, self.dimension, self.config["index"]["nlist"])
        else:
            raise ValueError(f"Unsupported index: {index_type}")
        return faiss.IndexIDMap(idx)

    def add_chunks(self, chunks: List[str], frame_numbers: List[int], show_progress: bool = True) -> List[int]:
        if len(chunks) != len(frame_numbers):
            raise ValueError("Mismatch between chunks and frame numbers")

        logger.info(f"Encoding {len(chunks)} segments into vectors")
        embeddings = self.embedding_model.encode(
            chunks,
            show_progress_bar=show_progress,
            batch_size=32
        ).astype('float32')

        start = len(self.metadata)
        ids = list(range(start, start + len(chunks)))

        if isinstance(self.index.index, faiss.IndexIVFFlat) and not self.index.index.is_trained:
            logger.info("Training FAISS IVF index...")
            self.index.index.train(embeddings)

        self.index.add_with_ids(embeddings, np.array(ids, dtype=np.int64))

        for i, (text, frame_id, chunk_id) in enumerate(zip(chunks, frame_numbers, ids)):
            entry = {
                "id": chunk_id,
                "text": text,
                "frame": frame_id,
                "length": len(text)
            }
            self.metadata.append(entry)
            self.chunk_to_frame[chunk_id] = frame_id
            self.frame_to_chunks.setdefault(frame_id, []).append(chunk_id)

        logger.info(f"Indexed {len(ids)} chunks successfully")
        return ids

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float, Dict[str, Any]]]:
        query_vec = self.embedding_model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                meta = self.metadata[idx]
                results.append((idx, float(dist), meta))
        return results

    def get_chunks_by_frame(self, frame_number: int) -> List[Dict[str, Any]]:
        ids = self.frame_to_chunks.get(frame_number, [])
        return [self.metadata[i] for i in ids]

    def get_chunk_by_id(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        if 0 <= chunk_id < len(self.metadata):
            return self.metadata[chunk_id]
        return None

    def save(self, path: str):
        path = Path(path)
        faiss.write_index(self.index, str(path.with_suffix('.faiss')))
        data = {
            "metadata": self.metadata,
            "chunk_to_frame": self.chunk_to_frame,
            "frame_to_chunks": self.frame_to_chunks,
            "config": self.config
        }
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Index saved at: {path}")

    def load(self, path: str):
        path = Path(path)
        self.index = faiss.read_index(str(path.with_suffix('.faiss')))
        with open(path.with_suffix('.json'), 'r') as f:
            data = json.load(f)
        self.metadata = data["metadata"]
        self.chunk_to_frame = {int(k): v for k, v in data["chunk_to_frame"].items()}
        self.frame_to_chunks = {int(k): v for k, v in data["frame_to_chunks"].items()}
        if "config" in data:
            self.config.update(data["config"])
        logger.info(f"Index loaded from: {path}")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_chunks": len(self.metadata),
            "total_frames": len(self.frame_to_chunks),
            "index_type": self.config["index"]["type"],
            "embedding_model": self.config["embedding"]["model"],
            "dimension": self.dimension,
            "avg_chunks_per_frame": np.mean([len(c) for c in self.frame_to_chunks.values()]) if self.frame_to_chunks else 0
        }
