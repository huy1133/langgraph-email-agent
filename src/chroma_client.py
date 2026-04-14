"""
ChromaDB Singleton Client.

Cung cấp một PersistentClient duy nhất cho toàn bộ process,
tránh việc tạo mới client (mở file lock) mỗi lần request.
Thread-safe thông qua double-checked locking.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from .llm_env import LLMEnv

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_lock = threading.Lock()

_client: chromadb.PersistentClient | None = None
# Cache theo key = "{collection_name}::{embedding_model}"
_collections: dict[str, chromadb.Collection] = {}


def _get_client() -> chromadb.PersistentClient:
    """Khởi tạo PersistentClient một lần duy nhất (lazy singleton)."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                chroma_path = _ROOT / "chroma_data"
                chroma_path.mkdir(parents=True, exist_ok=True)
                _client = chromadb.PersistentClient(path=str(chroma_path))
                logger.info("ChromaDB PersistentClient khởi tạo tại: %s", chroma_path)
    return _client


def get_chroma_collection(
    env: LLMEnv,
    collection_name: str = "email_chunks",
) -> chromadb.Collection:
    """
    Trả về ChromaDB collection đã được cache cho process.

    Args:
        env: Cấu hình LLM (api_key, base_url, embedding_model).
        collection_name: Tên collection trong ChromaDB.

    Returns:
        chromadb.Collection đã được cache hoặc mới tạo.
    """
    cache_key = f"{collection_name}::{env.embedding_model}"

    # Fast path: không cần lock nếu đã có trong cache
    if cache_key in _collections:
        return _collections[cache_key]

    with _lock:
        # Double-check sau khi acquire lock
        if cache_key in _collections:
            return _collections[cache_key]

        emb_kwargs: dict = {
            "api_key": env.api_key,
            "model_name": env.embedding_model,
        }
        if env.base_url:
            emb_kwargs["api_base"] = env.base_url

        ef = embedding_functions.OpenAIEmbeddingFunction(**emb_kwargs)
        client = _get_client()
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef,
        )
        _collections[cache_key] = collection
        logger.info("ChromaDB collection '%s' đã được cache.", collection_name)
        return collection


def invalidate_cache() -> None:
    """Xóa cache collection (dùng cho testing hoặc reload)."""
    global _client, _collections
    with _lock:
        _collections.clear()
        _client = None
    logger.info("ChromaDB singleton cache đã bị xóa.")
