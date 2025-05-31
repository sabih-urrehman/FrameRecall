

import os
import time
from typing import Optional, Dict, Any
from .chat import FramerecallChat
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
    

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    

    if session_dir is None:
        session_dir = "output"
    os.makedirs(session_dir, exist_ok=True)
    

    if not os.path.exists(video_file):
        print(f"Error: Video file not found: {video_file}")
        return
    if not os.path.exists(index_file):
        print(f"Error: Index file not found: {index_file}")
        return
    

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    
    print("Initializing Framerecall Chat...")
    chat = FramerecallChat(video_file, index_file, llm_api_key=api_key, llm_model=llm_model, config=config)
    chat.start_session()
    

    if show_stats:
        stats = chat.get_stats()
        print(f"\nMemory loaded: {stats['retriever_stats']['index_stats']['total_chunks']} chunks")
        if stats['llm_available']:
            print(f"LLM: {stats['llm_model']}")
        else:
            print("LLM: Not available (context-only mode)")
    
    print("\nType 'help' for commands, 'exit' to quit")
    print("-" * 50)
    

    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
                

            lower_input = user_input.lower()
            
            if lower_input in ['exit', 'quit', 'q']:
                break
                
            elif lower_input == 'help':
                print("\nCommands:")
                print("  search <query> - Show raw search results")
                print("  stats         - Show system statistics")
                print("  export        - Save conversation")
                print("  clear         - Clear conversation history")
                print("  help          - Show this help")
                print("  exit/quit     - End session")
                continue
                
            elif lower_input == 'stats':
                stats = chat.get_stats()
                print(f"\nMessages: {stats['message_count']}")
                print(f"Cache size: {stats['retriever_stats']['cache_size']}")
                print(f"Video frames: {stats['retriever_stats']['total_frames']}")
                continue
                
            elif lower_input == 'export':
                export_file = os.path.join(session_dir, f"framerecall_session_{chat.session_id}.json")
                chat.export_session(export_file)
                print(f"Exported to: {export_file}")
                continue
                
            elif lower_input == 'clear':
                chat.reset_session()
                chat.start_session()
                print("Conversation cleared.")
                continue
                
            elif lower_input.startswith('search '):
                query = user_input[7:]
                print(f"\nSearching: '{query}'")
                start_time = time.time()
                results = chat.search_context(query, top_k=5)
                elapsed = time.time() - start_time
                print(f"Found {len(results)} results in {elapsed:.3f}s:\n")
                for i, result in enumerate(results[:3]):
                    print(f"{i+1}. [Score: {result['score']:.3f}] {result['text'][:100]}...")
                continue
            

            print("\nAssistant: ", end="", flush=True)
            start_time = time.time()
            response = chat.chat(user_input)
            elapsed = time.time() - start_time
            
            print(response)
            print(f"\n[{elapsed:.1f}s]", end="")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue
    

    if export_on_exit and chat.get_history():
        export_file = os.path.join(session_dir, f"framerecall_session_{chat.session_id}.json")
        chat.export_session(export_file)
        print(f"\nSession saved to: {export_file}")
    
    print("Goodbye!")


def quick_chat(video_file: str, index_file: str, query: str, api_key: Optional[str] = None) -> str:
    
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    chat = FramerecallChat(video_file, index_file, llm_api_key=api_key)
    return chat.chat(query)