#!/usr/bin/env python3


import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from framerecall import FramerecallRetriever
import logging

logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

def test_retrieval():
    print("Testing Framerecall retrieval flow...")
    print("=" * 50)
    

    video_file = "output/memory.mp4"
    index_file = "output/memory_index.json"
    
    if not os.path.exists(video_file) or not os.path.exists(index_file):
        print("ERROR: Video or index file not found. Run build_memory.py first.")
        return
    
    print(f"\nInitializing retriever with:")
    print(f"  Video: {video_file}")
    print(f"  Index: {index_file}")
    
    retriever = FramerecallRetriever(video_file, index_file)
    

    query = "quantum computing"
    print(f"\nSearching for: '{query}'")
    results = retriever.search(query, top_k=3)
    
    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result[:100]}...")
    

    print("\n\nTesting search_with_metadata to see full flow...")
    metadata_results = retriever.search_with_metadata(query, top_k=3)
    
    for i, result in enumerate(metadata_results):
        print(f"\n{i+1}. Text: {result['text'][:100]}...")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Frame: {result['frame']}")
        print(f"   From metadata: {result['metadata']['text'][:100]}...")

if __name__ == "__main__":
    test_retrieval()