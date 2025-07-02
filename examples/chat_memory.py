#!/usr/bin/env python3
"""
Demo: Updated interactive session using FrameRecallChat
"""

import sys
import os

# Disable tokenizer threading warning for better CLI experience
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framerecall import FrameRecallChat
import time

def display_context(matches):
    """Format matching results for display"""
    print("\nTop semantic matches:")
    print("-" * 50)
    for i, match in enumerate(matches[:3]):
        print(f"\n[{i+1}] Match Score: {match['score']:.3f}")
        print(f"Excerpt: {match['text'][:150]}...")
        print(f"Frame Ref: {match['frame']}")

def main():
    print("FrameRecall Demo: Contextual Chat Interface")
    print("=" * 50)

    # Verify presence of archive files
    archive_path = "output/archive.mp4"
    index_path = "output/search_index.json"

    if not os.path.exists(archive_path) or not os.path.exists(index_path):
        print("\nError: Required memory files are missing!")
        print("Use 'python examples/build_memory.py' to generate them.")
        return

    # Announce memory load
    print(f"\nOpening archive: {archive_path}")

    # Load API key from environment or fallback default (for demo)
    api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    if not api_key:
        print("\nNotice: No API key provided. Operating in retrieval-only mode.")
        print("To enable full conversation, set OPENAI_API_KEY in your environment.")

    chat = FrameRecallChat(archive_path, index_path, llm_api_key=api_key)
    chat.start_session()

    # Retrieve session summary
    stats = chat.get_stats()
    print("\nMemory access successful!")
    print(f"  Entries indexed: {stats['retriever_stats']['index_stats']['total_chunks']}")
    print(f"  LLM integrated: {stats['llm_available']}")
    if stats['llm_available']:
        print(f"  Engine in use: {stats['llm_model']}")

    print("\nQuick Commands:")
    print("- Ask anything to query the memory")
    print("- Use 'search <keywords>' for direct lookups")
    print("- Use 'stats' to view performance data")
    print("- Use 'export' to save dialogue history")
    print("- Type 'exit' or 'quit' to leave")
    print("-" * 50)

    # Run conversation loop
    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['exit', 'quit']:
                print("\nSession closed. See you next time!")
                break

            elif user_input.lower() == 'stats':
                stats = chat.get_stats()
                print("\nSystem Overview:")
                print(f"  Messages exchanged: {stats['message_count']}")
                print(f"  Cache size: {stats['retriever_stats']['cache_size']}")
                print(f"  Total frames: {stats['retriever_stats']['total_frames']}")
                continue

            elif user_input.lower() == 'export':
                export_path = f"output/session_{chat.session_id}.json"
                chat.export_session(export_path)
                print(f"Transcript saved to: {export_path}")
                continue

            elif user_input.lower().startswith('search '):
                query = user_input[7:]
                print(f"\nLooking up: '{query}'")
                start = time.time()
                matches = chat.search_context(query, top_k=5)
                elapsed = time.time() - start
                print(f"Search done in {elapsed:.3f} seconds")
                display_context(matches)
                continue

            # Get LLM response
            print("\nAssistant: ", end="", flush=True)
            start = time.time()
            answer = chat.chat(user_input)
            elapsed = time.time() - start

            print(answer)
            print(f"\n[Latency: {elapsed:.2f}s]")

        except KeyboardInterrupt:
            print("\n\nSession interrupted manually. Exiting.")
            break
        except Exception as e:
            print(f"\nException encountered: {e}")
            continue

    # Save conversation if applicable
    if chat.get_history():
        export_path = f"output/session_{chat.session_id}.json"
        chat.export_session(export_path)
        print(f"\nSession archive stored at: {export_path}")

if __name__ == "__main__":
    main()