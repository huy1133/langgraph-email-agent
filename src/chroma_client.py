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
# Sửa Lock thành RLock để chống deadlock khi gọi lồng nhau
_lock = threading.RLock()

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


class SafeGeminiEmbeddingFunction:
    def __init__(self, api_key: str, model_name: str, api_base: str | None = None):
        import openai
        # Tắt TỰ ĐỘNG RETRY (max_retries=0) để tránh treo vô hạn nếu dính Rate Limit
        self.client = openai.OpenAI(api_key=api_key, base_url=api_base, max_retries=0, timeout=45.0)
        self.model_name = model_name

    def name(self) -> str:
        return "SafeGeminiEmbeddingFunction"

    def __call__(self, input: list[str]) -> list[list[float]]:
        # Gửi toàn bộ dữ liệu qua API chỉ bằng MỘT VÀI kết nối HTTP (chia chunk 50)
        # Google Gemini Free giới hạn 15 RPM. Gộp mảng (batch array) chỉ tính là 1 request!
        import time
        logger.info(f"==> [SafeGeminiEmbeddingFunction] Nhận yêu cầu tạo vector cho {len(input)} chunks từ ChromaDB...")
        embeddings = []
        batch_size = 50
        for i in range(0, len(input), batch_size):
            batch = input[i:i + batch_size]
            try:
                resp = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                embeddings.extend([d.embedding for d in resp.data])
            except Exception as e:
                logger.error("Embedding API error for batch: %s", e)
                # Fallback: Trả về zero-vector cho các chunk bị lỗi (để giữ nguyên cấu trúc index)
                for _ in range(len(batch)):
                    embeddings.append([0.0] * 3072)
        logger.info(f"==> [SafeGeminiEmbeddingFunction] Trả về {len(embeddings)} vectors cho ChromaDB.")
        return embeddings

    # --- Thêm alias hỗ trợ Duck-typing cho LangChain framework nếu ChromaDB kiểm tra ---
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.__call__(texts)

    def embed_query(self, text: str) -> list[float]:
        # Khác với call là nó chỉ trả 1 dòng list float chứ không phải list của list
        res = self.__call__([text])
        return res[0] if res else [0.0] * 3072


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

        logger.info("TRACING: Instantiating Embedding Function...")
        ef = SafeGeminiEmbeddingFunction(
            api_key=env.api_key,
            model_name=env.embedding_model,
            api_base=env.base_url
        )
        
        logger.info("TRACING: Calling _get_client()...")
        client = _get_client()
        logger.info("TRACING: Calling client.get_or_create_collection()...")
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
