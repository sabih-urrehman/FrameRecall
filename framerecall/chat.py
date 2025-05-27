

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

from .retriever import FramerecallRetriever
from .config import get_default_config

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available. LLM features will be limited.")


class FramerecallChat:
    
    
    def __init__(self, video_file: str, index_file: str,
                 llm_api_key: Optional[str] = None,
                 llm_model: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        
        self.config = config or get_default_config()
        self.retriever = FramerecallRetriever(video_file, index_file, self.config)
        

        self.llm_model = llm_model or self.config["llm"]["model"]
        self._init_llm(llm_api_key)
        

        self.conversation_history = []
        self.session_id = None
        self.context_chunks = self.config["chat"]["context_chunks"]
        self.max_history = self.config["chat"]["max_history"]
        
    def _init_llm(self, api_key: Optional[str] = None):
        
        if not OPENAI_AVAILABLE:
            self.llm_client = None
            logger.warning("OpenAI not available. Chat will return context only.")
            return
        

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.llm_client = None
            logger.warning("No OpenAI API key provided. Chat will return context only.")
            return
        
        try:
            self.llm_client = OpenAI(api_key=api_key)
            logger.info(f"Initialized OpenAI client with model: {self.llm_model}")
        except Exception as e:
            self.llm_client = None
            logger.error(f"Failed to initialize OpenAI client: {e}")
    
    def start_session(self, session_id: Optional[str] = None):
        
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.conversation_history = []
        logger.info(f"Started chat session: {self.session_id}")
    
    def chat(self, user_input: str) -> str:
        
        if not self.session_id:
            self.start_session()
        

        context_chunks = self.retriever.search(user_input, top_k=self.context_chunks)
        

        context = "\n\n".join([f"[Context {i+1}]: {chunk}" 
                               for i, chunk in enumerate(context_chunks)])
        

        self.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        

        if self.llm_client:
            response = self._generate_llm_response(user_input, context)
        else:

            response = self._generate_context_response(context_chunks)
        

        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat(),
            "context_used": len(context_chunks)
        })
        

        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
        
        return response
    
    def _generate_llm_response(self, user_input: str, context: str) -> str:
        
        try:

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant with access to a knowledge base stored in video memory. "
                        "Use the provided context chunks to answer questions accurately. "
                        "If asked about what the context or knowledge base contains, analyze and summarize "
                        "the topics covered based on the context chunks provided. "
                        "Always base your answers on the given context."
                    )
                }
            ]
            

            history_start = max(0, len(self.conversation_history) - self.max_history)
            for msg in self.conversation_history[history_start:-1]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            

            messages.append({
                "role": "user",
                "content": f"Context from knowledge base:\n{context}\n\nUser question: {user_input}"
            })
            

            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                max_tokens=self.config["llm"]["max_tokens"],
                temperature=self.config["llm"]["temperature"]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_context_response(context.split("\n\n"))
    
    def _generate_context_response(self, context_chunks: List[str]) -> str:
        
        if not context_chunks:
            return "I couldn't find any relevant information in the knowledge base."
        
        response = "Based on the knowledge base, here's what I found:\n\n"
        for i, chunk in enumerate(context_chunks[:3]):
            response += f"{i+1}. {chunk[:200]}...\n\n" if len(chunk) > 200 else f"{i+1}. {chunk}\n\n"
        
        return response.strip()
    
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
        
        logger.info(f"Exported session to: {output_file}")
    
    def load_session(self, session_file: str):
        
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        self.session_id = session_data["session_id"]
        self.conversation_history = session_data["history"]
        
        logger.info(f"Loaded session: {self.session_id}")
    
    def reset_session(self):
        
        self.conversation_history = []
        logger.info("Reset conversation history")
    
    def get_stats(self) -> Dict[str, Any]:
        
        return {
            "session_id": self.session_id,
            "message_count": len(self.conversation_history),
            "llm_available": self.llm_client is not None,
            "llm_model": self.llm_model,
            "context_chunks_per_query": self.context_chunks,
            "retriever_stats": self.retriever.get_stats()
        }