

import io
import json
import qrcode
import cv2
import numpy as np
from PIL import Image
from pyzbar import pyzbar
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
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

        default_qr_config = get_default_config()["qr"]
        config = {**default_qr_config, **config.get("qr", config)}
    

    if len(data) > 100:
        compressed = gzip.compress(data.encode())
        data = base64.b64encode(compressed).decode()
        data = "GZ:" + data
    
    qr = qrcode.QRCode(
        version=config["version"],
        error_correction=getattr(qrcode.constants, f"ERROR_CORRECT_{config['error_correction']}"),
        box_size=config["box_size"],
        border=config["border"],
    )
    
    qr.add_data(data)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color=config["fill_color"], back_color=config["back_color"])
    return img


def decode_qr(image: np.ndarray) -> Optional[str]:
    
    try:
        decoded_objects = pyzbar.decode(image)
        if decoded_objects:
            data = decoded_objects[0].data.decode('utf-8')
            

            if data.startswith("GZ:"):
                compressed_data = base64.b64decode(data[3:])
                data = gzip.decompress(compressed_data).decode()
            
            return data
    except Exception as e:
        logger.warning(f"QR decode failed: {e}")
    return None


def create_video_writer(output_path: str, config: Optional[Dict[str, Any]] = None) -> cv2.VideoWriter:
    
    if config is None:
        config = get_default_config()["video"]
    
    fourcc = cv2.VideoWriter_fourcc(*config["codec"])
    return cv2.VideoWriter(
        output_path,
        fourcc,
        config["fps"],
        (config["frame_width"], config["frame_height"])
    )


def qr_to_frame(qr_image: Image.Image, frame_size: Tuple[int, int]) -> np.ndarray:
    

    qr_image = qr_image.resize(frame_size, Image.Resampling.LANCZOS)
    

    if qr_image.mode != 'RGB':
        qr_image = qr_image.convert('RGB')
    

    img_array = np.array(qr_image, dtype=np.uint8)
    

    frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return frame


def extract_frame(video_path: str, frame_number: int) -> Optional[np.ndarray]:
    
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            return frame
    finally:
        cap.release()
    return None


@lru_cache(maxsize=1000)
def extract_and_decode_cached(video_path: str, frame_number: int) -> Optional[str]:
    
    frame = extract_frame(video_path, frame_number)
    if frame is not None:
        return decode_qr(frame)
    return None


def batch_extract_frames(video_path: str, frame_numbers: List[int], 
                        max_workers: int = 4) -> List[Tuple[int, Optional[np.ndarray]]]:
    
    results = []
    

    sorted_frames = sorted(frame_numbers)
    
    cap = cv2.VideoCapture(video_path)
    try:
        for frame_num in sorted_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            results.append((frame_num, frame if ret else None))
    finally:
        cap.release()
    
    return results


def parallel_decode_qr(frames: List[Tuple[int, np.ndarray]], 
                      max_workers: int = 4) -> List[Tuple[int, Optional[str]]]:
    
    def decode_frame(item):
        frame_num, frame = item
        if frame is not None:
            data = decode_qr(frame)
            return (frame_num, data)
        return (frame_num, None)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(decode_frame, frames))
    
    return results


def batch_extract_and_decode(video_path: str, frame_numbers: List[int], 
                            max_workers: int = 4, show_progress: bool = False) -> Dict[int, str]:
    

    frames = batch_extract_frames(video_path, frame_numbers)
    

    if show_progress:
        frames = tqdm(frames, desc="Decoding QR frames")
    
    decoded = parallel_decode_qr(frames, max_workers)
    

    result = {}
    for frame_num, data in decoded:
        if data is not None:
            result[frame_num] = data
    
    return result


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        

        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.8:
                end = start + last_period + 1
                chunk = text[start:end]
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks


def save_index(index_data: Dict[str, Any], output_path: str):
    
    with open(output_path, 'w') as f:
        json.dump(index_data, f, indent=2)


def load_index(index_path: str) -> Dict[str, Any]:
    
    with open(index_path, 'r') as f:
        return json.load(f)