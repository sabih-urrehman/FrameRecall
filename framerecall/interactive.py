#!/usr/bin/env python3
"""
Conversational session tool for FrameRecall
"""

import os
import time
from typing import Optional, Dict, Any
from .chat import FrameRecallChat
from .config import VIDEO_FILE_TYPE

def chat_with_memory(
    video_file: str,
    index_file: str,
    api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
    show_stats: bool = True,
    export_on_exit: bool = True,
    session_dir: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
):
    """
    Launch a dialogue interface with a QR code-based archive.
    
    Arguments:
        video_file: Path to encoded video file
        index_file: Metadata and retrieval structure path
        api_key: Token for LLM access (optional)
        llm_model: Language model identifier
        show_stats: Show session details on startup
        export_on_exit: Save transcript upon termination
        session_dir: Output location for chat logs
        config: Additional runtime options

    Available commands:
        - 'search <term>': Return matched results
        - 'stats': Display performance metrics
        - 'export': Write chat log to disk
        - 'clear': Wipe current conversation
        - 'help': List commands
        - 'exit' or 'quit': Terminate session

    Example:
        >>> from framerecall import chat_with_memory
        >>> chat_with_memory(f'knowledge.{VIDEO_FILE_TYPE}', 'knowledge_index.json')
    """
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    if session_dir is None:
        session_dir = "output"
    os.makedirs(session_dir, exist_ok=True)
    if not os.path.exists(video_file):
        print(f"Missing video: {video_file}")
        return
    if not os.path.exists(index_file):
        print(f"Missing index: {index_file}")
        return
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    print("Setting up FrameRecall dialogue...")
    chat = FrameRecallChat(video_file, index_file, llm_api_key=api_key, llm_model=llm_model, config=config)
    chat.start_session()
    if show_stats:
        stats = chat.get_stats()
        print(f"\nLoaded entries: {stats['retriever_stats']['index_stats']['total_chunks']}")
        if stats['llm_available']:
            print(f"LLM connected: {stats['llm_model']}")
        else:
            print("LLM not detected, using offline context only.")
    print("\nEnter 'help' for instructions or 'exit' to close")
    print("-" * 50)
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            command = user_input.lower()
            if command in ['exit', 'quit', 'q']:
                break
            elif command == 'help':
                print("\nAvailable:")
                print("  search <query> - Raw result display")
                print("  stats          - System usage data")
                print("  export         - Save discussion")
                print("  clear          - Reset dialogue")
                print("  help           - This menu")
                print("  exit/quit      - Close interface")
                continue
            elif command == 'stats':
                stats = chat.get_stats()
                print(f"\nExchanges: {stats['message_count']}")
                print(f"Cache entries: {stats['retriever_stats']['cache_size']}")
                print(f"Frame count: {stats['retriever_stats']['total_frames']}")
                continue
            elif command == 'export':
                path = os.path.join(session_dir, f"framerecall_session_{chat.session_id}.json")
                chat.export_session(path)
                print(f"Transcript saved: {path}")
                continue
            elif command == 'clear':
                chat.reset_session()
                chat.start_session()
                print("Dialogue reset.")
                continue
            elif command.startswith('search '):
                query = user_input[7:]
                print(f"\nLooking up: '{query}'")
                start = time.time()
                hits = chat.search_context(query, top_k=5)
                duration = time.time() - start
                print(f"{len(hits)} found in {duration:.3f}s:\n")
                for i, r in enumerate(hits[:3]):
                    print(f"{i+1}. [Score: {r['score']:.3f}] {r['text'][:100]}...")
                continue
            print("\nAssistant: ", end="", flush=True)
            begin = time.time()
            answer = chat.chat(user_input)
            runtime = time.time() - begin
            print(answer)
            print(f"\n[{runtime:.1f}s]", end="")
        except KeyboardInterrupt:
            print("\n\nStopped by user.")
            break
        except Exception as err:
            print(f"\nIssue: {err}")
            continue
    if export_on_exit and chat.get_history():
        path = os.path.join(session_dir, f"framerecall_session_{chat.session_id}.json")
        chat.export_session(path)
        print(f"\nSaved log to: {path}")
    print("Session closed.")

def quick_chat(video_file: str, index_file: str, query: str, api_key: Optional[str] = None) -> str:
    """
    One-shot Q&A execution outside of session mode.
    
    Parameters:
        video_file: FrameRecall video location
        index_file: Lookup structure path
        query: User question
        api_key: LLM token
    
    Returns:
        Generated reply as text
    
    Usage:
        >>> from framerecall import quick_chat
        >>> output = quick_chat(f"knowledge.{VIDEO_FILE_TYPE}", "knowledge_index.json", "What is quantum computing?")
        >>> print(output)
    """
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    chat = FrameRecallChat(video_file, index_file, llm_api_key=api_key)
    return chat.chat(query)