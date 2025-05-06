"""
Tests for FrameRecallEncoder functionality
"""

import pytest
import tempfile
import os
from pathlib import Path

from framerecall import FrameRecallEncoder

def test_encoder_initialization():
    encoder = FrameRecallEncoder()
    assert encoder.chunks == []
    assert encoder.index_manager is not None

def test_add_chunks():
    encoder = FrameRecallEncoder()
    data = ["alpha", "beta", "gamma"]
    encoder.add_chunks(data)
    assert len(encoder.chunks) == 3
    assert encoder.chunks == data

def test_add_text():
    encoder = FrameRecallEncoder()
    text = "Sample sentence. " * 40  # ~700 chars
    encoder.add_text(text, chunk_size=120, overlap=30)
    assert len(encoder.chunks) > 1
    assert all(c.strip() for c in encoder.chunks)

def test_build_video():
    encoder = FrameRecallEncoder()
    segments = [
        "Alpha memory block",
        "Beta memory block",
        "Gamma memory block"
    ]
    encoder.add_chunks(segments)

    with tempfile.TemporaryDirectory() as tmp:
        video_out = os.path.join(tmp, "clip.mp4")
        index_out = os.path.join(tmp, "clip_index.json")
        stats = encoder.build_video(video_out, index_out, show_progress=False)

        assert os.path.exists(video_out)
        assert os.path.exists(index_out)
        assert os.path.exists(index_out.replace('.json', '.faiss'))
        assert stats["total_chunks"] == 3
        assert stats["video_size_mb"] > 0
        assert stats["duration_seconds"] > 0

def test_encoder_stats():
    encoder = FrameRecallEncoder()
    segments = ["tiny", "something medium sized", "a rather lengthy segment of text"]
    encoder.add_chunks(segments)
    stats = encoder.get_stats()
    assert stats["total_chunks"] == 3
    assert stats["total_characters"] == sum(len(s) for s in segments)
    assert stats["avg_chunk_size"] > 0

def test_clear():
    encoder = FrameRecallEncoder()
    encoder.add_chunks(["erase this", "and that"])
    encoder.clear()
    assert encoder.chunks == []
    assert encoder.get_stats()["total_chunks"] == 0
