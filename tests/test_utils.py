"""Tests for utility functions"""

import pytest
import numpy as np
from PIL import Image
import tempfile
import os

from memvid.utils import (
    encode_to_qr, decode_qr, chunk_text, qr_to_frame,
    save_index, load_index
)


def test_qr_basic_encode_decode():
    """Verify QR encoding and decoding with basic data"""
    sample_text = "Hello, Memvid!"
    
    qr_img = encode_to_qr(sample_text)
    assert isinstance(qr_img, Image.Image)
    
    frame = qr_to_frame(qr_img, (512, 512))
    assert frame.shape == (512, 512, 3)
    
    decoded = decode_qr(frame)
    assert decoded == sample_text


def test_qr_large_data_with_compression():
    """Ensure compression works for larger input strings"""
    long_text = "x" * 1000
    
    qr_img = encode_to_qr(long_text)
    frame = qr_to_frame(qr_img, (512, 512))
    
    decoded = decode_qr(frame)
    assert decoded == long_text


def test_chunking_text_with_overlap():
    """Check chunking logic and overlap behavior"""
    text = "This is a test. " * 50
    
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)
    
    for i in range(len(chunks) - 1):
        # Confirm overlap between chunks
        overlap_words = chunks[i].split()[-5:]
        assert any(word in chunks[i + 1] for word in overlap_words)


def test_index_serialization():
    """Test saving and reloading index data"""
    mock_data = {
        "metadata": [{"id": 1, "text": "mock entry"}],
        "config": {"debug_mode": True}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        path = tmp.name
    
    try:
        save_index(mock_data, path)
        assert os.path.exists(path)
        
        loaded = load_index(path)
        assert loaded == mock_data
    finally:
        os.unlink(path)
