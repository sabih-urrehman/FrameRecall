#!/usr/bin/env python3


import sys
import os
import shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from framerecall import FramerecallRetriever

def test_without_video():
    print("Testing retrieval without video file...")
    print("=" * 50)
    

    video_file = "output/memory.mp4"
    backup_file = "output/memory_backup.mp4"
    
    if os.path.exists(video_file):
        shutil.copy(video_file, backup_file)
        print(f"Backed up video to {backup_file}")
    
    try:

        if os.path.exists(video_file):
            os.remove(video_file)
            print(f"Removed {video_file}")
        

        print("\nTrying to create retriever without video file...")
        try:
            retriever = FramerecallRetriever(video_file, "output/memory_index.json")
            print("Retriever created (unexpected!)")
            

            print("\nTrying to search...")
            results = retriever.search("quantum computing", top_k=3)
            print(f"Search returned {len(results)} results")
            for i, text in enumerate(results):
                print(f"{i+1}. {text[:50]}...")
                
        except Exception as e:
            print(f"Error (expected): {type(e).__name__}: {e}")
            
    finally:

        if os.path.exists(backup_file):
            shutil.copy(backup_file, video_file)
            os.remove(backup_file)
            print(f"\nRestored {video_file}")

if __name__ == "__main__":
    test_without_video()