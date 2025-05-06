"""
Shared utility methods for FrameRecall
"""

import io
import json
import qrcode
import cv2
import numpy as np
from PIL import Image
from pyzbar import pyzbar
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import logging
from tqdm import tqdm
import base64
import gzip

from .config import get_default_config

logger = logging.getLogger(__name__)

def encode_to_qr(data: str, config: Optional[Dict[str, Any]] = None) -> Image.Image:
    if config is None:
        config = get_default_config()["qr"]
    else:
        default_cfg = get_default_config()["qr"]
        config = {**default_cfg, **config.get("qr", config)}

    if len(data) > 100:
        compressed = gzip.compress(data.encode())
        data = "GZ:" + base64.b64encode(compressed).decode()

    qr = qrcode.QRCode(
        version=config["version"],
        error_correction=getattr(qrcode.constants, f"ERROR_CORRECT_{config['error_correction']}"),
        box_size=config["box_size"],
        border=config["border"]
    )
    qr.add_data(data)
    qr.make(fit=True)
    return qr.make_image(fill_color=config["fill_color"], back_color=config["back_color"])

def decode_qr(image: np.ndarray) -> Optional[str]:
    try:
        decoded = pyzbar.decode(image)
        if decoded:
            result = decoded[0].data.decode('utf-8')
            if result.startswith("GZ:"):
                result = gzip.decompress(base64.b64decode(result[3:])).decode()
            return result
    except Exception as e:
        logger.warning(f"QR decode error: {e}")
    return None

def create_video_writer(output_path: str, config: Optional[Dict[str, Any]] = None) -> cv2.VideoWriter:
    cfg = config or get_default_config()["video"]
    fourcc = cv2.VideoWriter_fourcc(*cfg["codec"])
    return cv2.VideoWriter(
        output_path,
        fourcc,
        cfg["fps"],
        (cfg["frame_width"], cfg["frame_height"])
    )

def qr_to_frame(qr_image: Image.Image, frame_size: Tuple[int, int]) -> np.ndarray:
    qr_image = qr_image.resize(frame_size, Image.Resampling.LANCZOS)
    if qr_image.mode != 'RGB':
        qr_image = qr_image.convert('RGB')
    return cv2.cvtColor(np.array(qr_image, dtype=np.uint8), cv2.COLOR_RGB2BGR)

def extract_frame(video_path: str, frame_number: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        return frame if ret else None
    finally:
        cap.release()

@lru_cache(maxsize=1000)
def extract_and_decode_cached(video_path: str, frame_number: int) -> Optional[str]:
    frame = extract_frame(video_path, frame_number)
    return decode_qr(frame) if frame is not None else None

def batch_extract_frames(video_path: str, frame_numbers: List[int], max_workers: int = 4) -> List[Tuple[int, Optional[np.ndarray]]]:
    results = []
    sorted_indices = sorted(frame_numbers)
    cap = cv2.VideoCapture(video_path)
    try:
        for idx in sorted_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            results.append((idx, frame if ret else None))
    finally:
        cap.release()
    return results

def parallel_decode_qr(frames: List[Tuple[int, np.ndarray]], max_workers: int = 4) -> List[Tuple[int, Optional[str]]]:
    def task(pair):
        idx, frame = pair
        return (idx, decode_qr(frame)) if frame is not None else (idx, None)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(task, frames))

def batch_extract_and_decode(video_path: str, frame_numbers: List[int], max_workers: int = 4, show_progress: bool = False) -> Dict[int, str]:
    frames = batch_extract_frames(video_path, frame_numbers)
    if show_progress:
        frames = tqdm(frames, desc="Decoding frames")
    decoded = parallel_decode_qr(frames, max_workers)
    return {fid: data for fid, data in decoded if data is not None}

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    output = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        snippet = text[start:end]
        if end < len(text):
            cutoff = snippet.rfind('.')
            if cutoff > chunk_size * 0.8:
                end = start + cutoff + 1
                snippet = text[start:end]
        output.append(snippet.strip())
        start = end - overlap
    return output

def save_index(index_data: Dict[str, Any], output_path: str):
    with open(output_path, 'w') as f:
        json.dump(index_data, f, indent=2)

def load_index(index_path: str) -> Dict[str, Any]:
    with open(index_path, 'r') as f:
        return json.load(f)
