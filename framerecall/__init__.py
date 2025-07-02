"""
FrameRecall - Matrix Clip AI Memory Framework
"""

__version__ = "0.1.0"

from .encoder import FrameRecallEncoder
from .retriever import FrameRecallRetriever
from .chat import FrameRecallChat
from .interactive import chat_with_memory, quick_chat
from .llm_client import LLMClient, create_llm_client

__all__ = ["FrameRecallEncoder", "FrameRecallRetriever", "FrameRecallChat", "chat_with_memory", "quick_chat", "LLMClient", "create_llm_client"]
