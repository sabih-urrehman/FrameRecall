#!/usr/bin/env python3
"""
Example usage of FrameRecall to interact with book content
"""

import sys
import os
from memvid.config import VIDEO_FILE_TYPE
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from framerecall import FrameRecallEncoder, chat_with_memory

# Path to source PDF — FrameRecall manages parsing automatically
book_source = "data/bitcoin.pdf"  # Adjust to point to your file

# Output paths for video archive and search index
video_output = "output/book_memory.{VIDEO_FILE_TYPE}"
index_output = "output/book_memory_index.json"

# Prepare output directory for saving interaction history
os.makedirs("output/book_chat", exist_ok=True)

# Convert document into video format — handled internally
encoder = FrameRecallEncoder()
encoder.add_pdf(book_source)  # Single-line ingestion
encoder.build_video(video_output, index_output)
print(f"📼 Memory file created: {video_output}")

# Retrieve API token from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("\n⚠️  To enable LLM-based responses, define the OPENAI_API_KEY variable.")
    print("You'll otherwise receive extracted content chunks only.\n")

# Begin dialogue session with document memory
print("\n📖 Chat with your book! Pose queries about its topics.")
print("Try asking:")
print("- 'Give a summary of this file.'")
print("- 'What are the fundamental principles discussed?'\n")

chat_with_memory(video_output, index_output, api_key=api_key, session_dir="output/book_chat")
