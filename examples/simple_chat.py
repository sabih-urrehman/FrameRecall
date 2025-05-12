#!/usr/bin/env python3
"""
Minimalistic chat interface with FrameRecall memory
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framerecall import chat_with_memory, quick_chat

def main():
    print("FrameRecall Quick Interaction Examples")
    print("=" * 50)

    memory_video = "output/memory.mp4"
    memory_index = "output/memory_index.json"

    # Verify memory data presence
    if not os.path.exists(memory_video):
        print("\nMissing files: Please execute 'python examples/build_memory.py' beforehand.")
        return

    # Retrieve or define API token
    api_key = "your-api-key-here"
    if not api_key:
        print("\nℹ️  Set your OPENAI_API_KEY environment variable to enable full responses.")
        print("Fallback mode will return plain memory chunks only.\n")

    print("\n1. Single-shot prompt response:")
    print("-" * 30)
    answer = quick_chat(
        memory_video,
        memory_index,
        "How many qubits did the quantum computer achieve?",
        api_key=api_key
    )
    print(f"Response: {answer}")

    print("\n\n2. Open-ended conversation mode:")
    print("-" * 30)
    print("Launching session...\n")

    chat_with_memory(memory_video, memory_index, api_key=api_key)

if __name__ == "__main__":
    main()
