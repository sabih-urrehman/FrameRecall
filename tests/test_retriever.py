"""
Tests for FrameRecallRetriever
"""

import pytest
import tempfile
import os

from framerecall import FrameRecallEncoder, FrameRecallRetriever


@pytest.fixture
def test_memory_fixture():
    """Prepara ficheiros de teste (vídeo + índice)"""
    encoder = FrameRecallEncoder()
    exemplos = [
        "Large language models process text sequences using transformers",
        "Reinforcement learning optimises decisions through rewards",
        "Graph neural networks work with relational data",
        "Federated learning allows decentralised training",
        "Computer vision enables image understanding"
    ]
    encoder.add_chunks(exemplos)

    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = os.path.join(temp_dir, "memory.mp4")
        index_path = os.path.join(temp_dir, "memory_index.json")

        encoder.build_video(video_path, index_path, show_progress=False)
        yield video_path, index_path, exemplos


def test_initialisation(test_memory_fixture):
    """Verifica inicialização do retriever"""
    video, index, dados = test_memory_fixture
    retriever = FrameRecallRetriever(video, index)
    assert retriever.video_file == video
    assert retriever.total_frames == len(dados)


def test_semantic_query(test_memory_fixture):
    """Verifica pesquisa semântica"""
    video, index, _ = test_memory_fixture
    retriever = FrameRecallRetriever(video, index)

    results = retriever.search("transformers", top_k=3)
    assert any("transformer" in r.lower() for r in results)

    results = retriever.search("image classification", top_k=3)
    assert any("vision" in r.lower() for r in results)


def test_query_with_metadata(test_memory_fixture):
    """Verifica resultados com metadados"""
    video, index, _ = test_memory_fixture
    retriever = FrameRecallRetriever(video, index)

    results = retriever.search_with_metadata("federated", top_k=2)
    assert len(results) > 0
    for r in results:
        assert "text" in r and "score" in r and "chunk_id" in r


def test_chunk_lookup(test_memory_fixture):
    """Verifica acesso a chunk específico por ID"""
    video, index, _ = test_memory_fixture
    retriever = FrameRecallRetriever(video, index)

    chunk = retriever.get_chunk_by_id(0)
    assert chunk and "transformer" in chunk.lower()

    assert retriever.get_chunk_by_id(999) is None


def test_cache_control(test_memory_fixture):
    """Verifica controlo e limpeza de cache"""
    video, index, _ = test_memory_fixture
    retriever = FrameRecallRetriever(video, index)

    assert len(retriever._frame_cache) == 0
    retriever.search("some topic", top_k=2)
    assert len(retriever._frame_cache) >= 0
    retriever.clear_cache()
    assert len(retriever._frame_cache) == 0


def test_retriever_metrics(test_memory_fixture):
    """Verifica estatísticas do retriever"""
    video, index, dados = test_memory_fixture
    retriever = FrameRecallRetriever(video, index)

    stats = retriever.get_stats()
    assert stats["total_frames"] == len(dados)
    assert stats["fps"] > 0
    assert "index_stats" in stats
