#!/usr/bin/env python3


import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_qr_encoding():
    
    print("Testing QR encoding/decoding...")
    
    from framerecall.utils import encode_to_qr, decode_qr, qr_to_frame
    

    test_text = "Hello, Framerecall! This is a test."
    

    qr_image = encode_to_qr(test_text)
    print(f"✓ Created QR code image: {qr_image.size}")
    

    frame = qr_to_frame(qr_image, (512, 512))
    print(f"✓ Converted to video frame: {frame.shape}")
    

    decoded = decode_qr(frame)
    print(f"✓ Decoded text: {decoded}")
    
    assert decoded == test_text, f"Decode failed: expected '{test_text}', got '{decoded}'"
    print("✅ QR encoding/decoding test passed!\n")


def test_text_chunking():
    
    print("Testing text chunking...")
    
    from framerecall.utils import chunk_text
    

    text = "This is a test sentence. " * 20
    

    chunks = chunk_text(text, chunk_size=100, overlap=20)
    print(f"✓ Created {len(chunks)} chunks from {len(text)} characters")
    

    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {len(chunk)} chars - '{chunk[:30]}...'")
    
    assert len(chunks) > 1, "Should create multiple chunks"
    assert all(chunk for chunk in chunks), "No empty chunks"
    print("✅ Text chunking test passed!\n")


def test_encoder_basic():
    
    print("Testing FramerecallEncoder...")
    
    try:
        from framerecall import FramerecallEncoder
        

        encoder = FramerecallEncoder()
        print("✓ Created encoder instance")
        

        test_chunks = [
            "First chunk of test data",
            "Second chunk with different content",
            "Third chunk for testing"
        ]
        encoder.add_chunks(test_chunks)
        print(f"✓ Added {len(test_chunks)} chunks")
        

        stats = encoder.get_stats()
        print(f"✓ Encoder stats: {stats['total_chunks']} chunks, {stats['total_characters']} chars")
        
        print("✅ FramerecallEncoder basic test passed!\n")
        
    except Exception as e:
        print(f"❌ FramerecallEncoder test failed: {e}\n")
        import traceback
        traceback.print_exc()


def test_full_pipeline():
    
    print("Testing full pipeline...")
    
    try:
        from framerecall import FramerecallEncoder, FramerecallRetriever
        

        chunks = [
            "Python is a high-level programming language",
            "Machine learning uses algorithms to learn from data",
            "Neural networks are inspired by the human brain",
            "Cloud computing provides on-demand resources",
            "Data science combines statistics and programming"
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            video_file = os.path.join(temp_dir, "test.mp4")
            index_file = os.path.join(temp_dir, "test_index.json")
            

            print("  Encoding chunks to video...")
            encoder = FramerecallEncoder()
            encoder.add_chunks(chunks)
            stats = encoder.build_video(video_file, index_file, show_progress=False)
            print(f"  ✓ Created video: {stats['video_size_mb']:.2f} MB, {stats['duration_seconds']:.1f}s")
            

            print("  Testing retrieval...")
            retriever = FramerecallRetriever(video_file, index_file)
            

            results = retriever.search("programming", top_k=3)
            print(f"  ✓ Search for 'programming' returned {len(results)} results")
            for i, result in enumerate(results):
                print(f"    {i+1}. {result[:50]}...")
            
            print("✅ Full pipeline test passed!\n")
            
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 60)
    print("Framerecall Library Test Suite")
    print("=" * 60)
    print()
    

    test_qr_encoding()
    test_text_chunking()
    test_encoder_basic()
    test_full_pipeline()
    
    print("All basic tests completed!")