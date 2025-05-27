

import pytest
import numpy as np
from PIL import Image
import tempfile
import os

from framerecall.utils import (
    encode_to_qr, decode_qr, chunk_text, qr_to_frame,
    save_index, load_index
)


def test_qr_encode_decode():
    
    test_data = "Hello, Framerecall!"
    

    qr_image = encode_to_qr(test_data)
    assert isinstance(qr_image, Image.Image)
    

    frame = qr_to_frame(qr_image, (512, 512))
    assert frame.shape == (512, 512, 3)
    

    decoded = decode_qr(frame)
    assert decoded == test_data


def test_qr_encode_decode_large_data():
    
    test_data = "x" * 1000
    

    qr_image = encode_to_qr(test_data)
    frame = qr_to_frame(qr_image, (512, 512))
    

    decoded = decode_qr(frame)
    assert decoded == test_data


def test_chunk_text():
    
    text = "This is a test. " * 50
    

    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)
    

    for i in range(len(chunks) - 1):

        assert any(word in chunks[i+1] for word in chunks[i].split()[-5:])


def test_save_load_index():
    
    test_data = {
        "metadata": [{"id": 1, "text": "test"}],
        "config": {"test": True}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    
    try:

        save_index(test_data, temp_file)
        assert os.path.exists(temp_file)
        

        loaded_data = load_index(temp_file)
        assert loaded_data == test_data
    finally:
        os.unlink(temp_file)