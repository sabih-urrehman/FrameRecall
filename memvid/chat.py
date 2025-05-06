"""
FrameRecallChat - Conversational layer with semantic video memory access
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

from .retriever import FrameRecallRetriever
from .config import get_default_config

logger = logging.getLogger(__name__)

# Attempt OpenAI import, fallback gracefully
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not found. LLM-based replies will be unavailable.")


class FrameRecallChat:
    """Controls contextual dialogue via retriever and optional language model"""

    def __init__(self, video_file: str, index_file: str,
                 llm_api_key: Optional[str] = None,
                 llm_model: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialise FrameRecallChat

        Args:
            video_file: Path to video archive
            index_file: Path to associated index
            llm_api_key: Optional LLM key
            llm_model: Language model to apply
            config: Optional override configuration
        """
        self.config = config or get_default_config()
        self.retriever = FrameRecallRetriever(video_file, index_file, self.config)

        self.llm_model = llm_model or self.config["llm"]["model"]
        self._init_llm(llm_api_key)

        self.conversation_history = []
        self.session_id = None
        self.context_chunks = self.config["chat"]["context_chunks"]
        self.max_history = self.config["chat"]["max_history"]

    def _init_llm(self, api_key: Optional[str] = None):
        if not OPENAI_AVAILABLE:
            self.llm_client = None
            logger.warning("No OpenAI module. LLM features will be skipped.")
            return

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.llm_client = None
            logger.warning("OPENAI_API_KEY missing. Operating without LLM support.")
            return

        try:
            self.llm_client = OpenAI(api_key=api_key)
            logger.info(f"LLM connected: {self.llm_model}")
        except Exception as e:
            self.llm_client = None
            logger.error(f"LLM init failed: {e}")

    def start_session(self, session_id: Optional[str] = None):
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.conversation_history = []
        logger.info(f"Chat session started: {self.session_id}")

    def chat(self, user_input: str) -> str:
        if not self.session_id:
            self.start_session()

        context_chunks = self.retriever.search(user_input, top_k=self.context_chunks)
        context = "\n\n".join([f"[Context {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)])

        self.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })

        if self.llm_client:
            reply = self._generate_llm_response(user_input, context)
        else:
            reply = self._generate_context_response(context_chunks)

        self.conversation_history.append({
            "role": "assistant",
            "content": reply,
            "timestamp": datetime.now().isoformat(),
            "context_used": len(context_chunks)
        })

        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]

        return reply

    def _generate_llm_response(self, user_input: str, context: str) -> str:
        try:
            messages = [{
                "role": "system",
                "content": (
                    "You are a reasoning assistant backed by encoded video knowledge.\n"
                    "Use supplied context segments to answer clearly.\n"
                    "Reflect on topics covered when summarising answers."
                )
            }]

            history_start = max(0, len(self.conversation_history) - self.max_history)
            for msg in self.conversation_history[history_start:-1]:
                messages.append({"role": msg["role"], "content": msg["content"]})

            messages.append({
                "role": "user",
                "content": f"Video context:\n{context}\n\nQuery: {user_input}"
            })

            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                max_tokens=self.config["llm"]["max_tokens"],
                temperature=self.config["llm"]["temperature"]
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM error: {e}")
            return self._generate_context_response(context.split("\n\n"))

    def _generate_context_response(self, context_chunks: List[str]) -> str:
        if not context_chunks:
            return "No relevant information found in the archive."

        reply = "From memory archive, the following points emerged:\n\n"
        for i, chunk in enumerate(context_chunks[:3]):
            reply += f"{i+1}. {chunk[:200]}...\n\n" if len(chunk) > 200 else f"{i+1}. {chunk}\n\n"

        return reply.strip()

    def search_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.retriever.search_with_metadata(query, top_k)

    def get_history(self) -> List[Dict[str, Any]]:
        return self.conversation_history.copy()

    def export_session(self, output_file: str):
        session_data = {
            "session_id": self.session_id,
            "start_time": self.conversation_history[0]["timestamp"] if self.conversation_history else None,
            "end_time": self.conversation_history[-1]["timestamp"] if self.conversation_history else None,
            "message_count": len(self.conversation_history),
            "history": self.conversation_history,
            "config": {
                "llm_model": self.llm_model,
                "context_chunks": self.context_chunks,
                "max_history": self.max_history
            }
        }

        with open(output_file, 'w') as f:
            json.dump(session_data, f, indent=2)

        logger.info(f"Session exported to {output_file}")

    def load_session(self, session_file: str):
        with open(session_file, 'r') as f:
            session_data = json.load(f)

        self.session_id = session_data["session_id"]
        self.conversation_history = session_data["history"]
        logger.info(f"Session loaded: {self.session_id}")

    def reset_session(self):
        self.conversation_history = []
        logger.info("Chat history cleared")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "message_count": len(self.conversation_history),
            "llm_available": self.llm_client is not None,
            "llm_model": self.llm_model,
            "context_chunks_per_query": self.context_chunks,
            "retriever_stats": self.retriever.get_stats()
        }
