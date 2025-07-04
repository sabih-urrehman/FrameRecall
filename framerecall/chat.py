"""
FrameRecallChat - Conversational layer with semantic video memory access
"""

import json
import os
import logging
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
from .llm_client import LLMClient
from .retriever import FrameRecallRetriever
from .config import get_default_config

logger = logging.getLogger(__name__)

class FrameRecallChat:
    """Controls contextual dialogue via retriever and optional language model"""
    def __init__(
            self,
            video_file: str,
            index_file: str,
            llm_provider: str = 'google',
            llm_model: str = None,
            llm_api_key: str = None,
            config: Optional[Dict] = None,
            retriever_kwargs: Dict = None
    ):
        """
        Initialise FrameRecallChat

        Args:
            video_file: Path to video archive
            index_file: Path to associated index
            llm_api_key: Optional LLM key
            llm_model: Language model to apply
            config: Optional override configuration
        """
        self.video_file = video_file
        self.index_file = index_file
        self.config = config or get_default_config()
        retriever_kwargs = retriever_kwargs or {}
        self.retriever = FrameRecallRetriever(video_file, index_file, self.config)
        try:
            self.llm_client = LLMClient(
                provider=llm_provider,
                model=llm_model,
                api_key=llm_api_key
            )
            self.llm_provider = llm_provider
            logger.info(f"✓ Initialized {llm_provider} LLM client")
        except Exception as e:
            logger.error(f"✗ Failed to initialize LLM client: {e}")
            self.llm_client = None
            self.llm_provider = None
        self.context_chunks = self.config.get("chat", {}).get("context_chunks", 5)
        self.max_history = self.config.get("chat", {}).get("max_history", 10)
        self.session_id = None
        self.system_prompt = None

    def start_session(self, system_prompt: str = None, session_id: str = None):
        self.conversation_history = []
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = self._get_default_system_prompt()
        logger.info(f"Chat session started: {self.session_id}")
        if self.llm_provider:
            print(f"Using {self.llm_provider} for responses.")
        else:
            print("LLM not available - will return context only.")
        print("-" * 50)

    def _get_default_system_prompt(self) -> str:
        return """You are a helpful AI assistant with access to a knowledge base stored in video format. 
When answering questions:
1. Use the provided context from the knowledge base when relevant
2. Be clear about what information comes from the knowledge base vs. your general knowledge
3. If the context doesn't contain enough information, say so clearly
4. Provide helpful, accurate, and concise responses
The context will be provided with each query based on semantic similarity to the user's question."""

    def chat(self, message: str, stream: bool = False, max_context_tokens: int = 2000) -> str:
        if not self.session_id:
            self.start_session()
        if not self.llm_client:
            return self._generate_context_only_response(message)
        context = self._get_context(message, max_context_tokens)
        messages = self._build_messages(message, context)
        self.conversation_history.append({"role": "user", "content": message})
        if stream:
            return self._handle_streaming_response(messages)
        else:
            response = self.llm_client.chat(messages)
            if response:
                self.conversation_history.append({"role": "assistant", "content": response})
                return response
            else:
                return "Sorry, I encountered an error generating a response."

    def _get_context(self, query: str, max_tokens: int = 2000) -> str:
        try:
            context_chunks = self.retriever.search(query, top_k=self.context_chunks)
            context = "\n\n".join([f"[Context {i+1}]: {chunk}"
                                   for i, chunk in enumerate(context_chunks)])
            if len(context) > max_tokens * 4:
                context = context[:max_tokens * 4] + "..."
            return context
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return ""

    def _build_messages(self, message: str, context: str) -> List[Dict[str, str]]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        history_to_include = self.conversation_history[-6:] 
        messages.extend(history_to_include)
        if context.strip():
            enhanced_message = f"""Context from knowledge base:
{context}
User question: {message}"""
        else:
            enhanced_message = message
        messages.append({"role": "user", "content": enhanced_message})
        return messages

    def _handle_streaming_response(self, messages: List[Dict[str, str]]) -> str:
        print("Assistant: ", end="", flush=True)
        full_response = ""
        try:
            for chunk in self.llm_client.chat_stream(messages):
                print(chunk, end="", flush=True)
                full_response += chunk
            if full_response:
                self.conversation_history.append({"role": "assistant", "content": full_response})
            return full_response
        except Exception as e:
            error_msg = f"\nError during streaming: {e}"
            print(error_msg)
            return error_msg

    def _generate_context_only_response(self, query: str) -> str:
        try:
            context_chunks = self.retriever.search(query, top_k=self.context_chunks)
            if not context_chunks:
                return "I couldn't find any relevant information in the knowledge base."
            avg_chunk_length = sum(len(chunk) for chunk in context_chunks) / len(context_chunks)
            if avg_chunk_length < 50:  
                return "I couldn't find any relevant information about that topic in the knowledge base."
            response = "Based on the knowledge base, here's what I found:\n\n"
            for i, chunk in enumerate(context_chunks[:3]):  
                response += f"{i+1}. {chunk[:200]}...\n\n" if len(chunk) > 200 else f"{i+1}. {chunk}\n\n"
            return response.strip()
        except Exception as e:
            return f"Error searching knowledge base: {e}"

    def interactive_chat(self):
        if not self.llm_client:
            print("Warning: LLM client not initialized. Will return context-only responses.")
        self.start_session()
        print("Commands:")
        print("  - Type your questions normally")
        print("  - Type 'quit' or 'exit' to end")
        print("  - Type 'clear' to clear conversation history")
        print("  - Type 'stats' to see session statistics")
        print("=" * 50)
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    if self.conversation_history:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        export_path = f"output/conversation_{timestamp}.json"
                        self.export_conversation(export_path)
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    self.clear_history()
                    continue
                elif user_input.lower() == 'stats':
                    stats = self.get_stats()
                    print(f"Session stats: {stats}")
                    continue
                if not user_input:
                    continue
                if self.llm_client:
                    self.chat(user_input, stream=True)
                else:
                    response = self.chat(user_input, stream=False)
                    print(f"Assistant: {response}")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

    def search_context(self, query: str, top_k: int = 5) -> List[str]:
        try:
            return self.retriever.search(query, top_k)
        except Exception as e:
            logger.error(f"Error in search_context: {e}")
            return []

    def clear_history(self):
        self.conversation_history = []
        print("Conversation history cleared.")

    def export_conversation(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        conversation_data = {
            'session_id': self.session_id,
            'system_prompt': self.system_prompt,
            'llm_provider': self.llm_provider,
            'conversation': self.conversation_history,
            'video_file': self.video_file,
            'index_file': self.index_file,
            'timestamp': datetime.now().isoformat(),
            'stats': self.get_stats()
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        print(f"Conversation exported to {path}")

    def load_session(self, session_file: str):
        with open(session_file, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        self.session_id = session_data.get("session_id")
        self.conversation_history = session_data.get("conversation", [])
        self.system_prompt = session_data.get("system_prompt", self._get_default_system_prompt())

    def reset_session(self):
        self.conversation_history = []
        self.session_id = None
        logger.info("Reset conversation session")

    def get_stats(self) -> Dict:
        return {
            'session_id': self.session_id,
            'messages_exchanged': len(self.conversation_history),
            'llm_provider': self.llm_provider,
            'llm_available': self.llm_client is not None,
            'video_file': self.video_file,
            'index_file': self.index_file,
            'context_chunks_per_query': self.context_chunks,
            'max_history': self.max_history
        }

    def chat_with_memory(video_file: str, index_file: str, api_key: str = None,
                        provider: str = 'google', model: str = None):
        chat = FrameRecallChat(
            video_file=video_file,
            index_file=index_file,
            llm_provider=provider,
            llm_model=model,
            llm_api_key=api_key
        )
        chat.interactive_chat()

    def quick_chat(video_file: str, index_file: str, message: str,
                provider: str = 'google', api_key: str = None) -> str:
        chat = FrameRecallChat(
            video_file=video_file,
            index_file=index_file,
            llm_provider=provider,
            llm_api_key=api_key
        )
        return chat.chat(message)