"""
Default parameters and constants for FrameRecall
"""

from typing import Dict, Any

# QR pattern configuration
QR_VERSION = 1
QR_ERROR_CORRECTION = 'M'
QR_BOX_SIZE = 10
QR_BORDER = 4
QR_FILL_COLOR = "black"
QR_BACK_COLOR = "white"

# Video generation parameters
VIDEO_FPS = 30
VIDEO_CODEC = 'mp4v'
FRAME_WIDTH = 512
FRAME_HEIGHT = 512

# Retrieval system defaults
DEFAULT_TOP_K = 5
BATCH_SIZE = 100
MAX_WORKERS = 4
CACHE_SIZE = 1000

# Embedding model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Vector index tuning
INDEX_TYPE = "Flat"
NLIST = 100

# LLM configuration
DEFAULT_LLM_MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 1000
TEMPERATURE = 0.7
CONTEXT_WINDOW = 4096

# Chat system behavior
MAX_HISTORY_LENGTH = 10
CONTEXT_CHUNKS_PER_QUERY = 5

# Runtime performance
PREFETCH_FRAMES = 50
DECODE_TIMEOUT = 10

def get_default_config() -> Dict[str, Any]:
    """Return base configuration dictionary for FrameRecall"""
    return {
        "qr": {
            "version": QR_VERSION,
            "error_correction": QR_ERROR_CORRECTION,
            "box_size": QR_BOX_SIZE,
            "border": QR_BORDER,
            "fill_color": QR_FILL_COLOR,
            "back_color": QR_BACK_COLOR,
        },
        "video": {
            "fps": VIDEO_FPS,
            "codec": VIDEO_CODEC,
            "frame_width": FRAME_WIDTH,
            "frame_height": FRAME_HEIGHT,
        },
        "retrieval": {
            "top_k": DEFAULT_TOP_K,
            "batch_size": BATCH_SIZE,
            "max_workers": MAX_WORKERS,
            "cache_size": CACHE_SIZE,
        },
        "embedding": {
            "model": EMBEDDING_MODEL,
            "dimension": EMBEDDING_DIMENSION,
        },
        "index": {
            "type": INDEX_TYPE,
            "nlist": NLIST,
        },
        "llm": {
            "model": DEFAULT_LLM_MODEL,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "context_window": CONTEXT_WINDOW,
        },
        "chat": {
            "max_history": MAX_HISTORY_LENGTH,
            "context_chunks": CONTEXT_CHUNKS_PER_QUERY,
        },
        "performance": {
            "prefetch_frames": PREFETCH_FRAMES,
            "decode_timeout": DECODE_TIMEOUT,
        }
    }