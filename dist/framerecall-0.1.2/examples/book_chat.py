#!/usr/bin/env python3


import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()


from framerecall import FramerecallEncoder, chat_with_memory

book_pdf = "data/bitcoin.pdf"

video_path = "output/book_memory.mp4"
index_path = "output/book_memory_index.json"

os.makedirs("output/book_chat", exist_ok=True)

encoder = FramerecallEncoder()
encoder.add_pdf(book_pdf)
encoder.build_video(video_path, index_path)
print(f"Created book memory video: {video_path}")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("\nNote: Set OPENAI_API_KEY environment variable for full LLM responses.")
    print("Without it, you'll only see raw context chunks.\n")

print("\n📚 Chat with your book! Ask questions about the content.")
print("Example questions:")
print("- 'What is this document about?'")
print("- 'What are the key concepts explained?'\n")

chat_with_memory(video_path, index_path, api_key=api_key, session_dir="output/book_chat")