"""Đọc cấu hình LLM từ biến môi trường (.env) — dùng chung cho ingest và (sau này) agent."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _strip_or_none(s: str | None) -> str | None:
    if s is None:
        return None
    t = s.strip()
    return t or None


@dataclass(frozen=True)
class LLMEnv:
    api_key: str
    base_url: str | None
    embedding_model: str
    chat_model: str


def load_llm_env() -> LLMEnv:
    """
    Biến môi trường:
      - LLM_API_KEY: bắt buộc cho API tương thích OpenAI
      - BASE_URL: tuỳ chọn (proxy / Azure-style / local); để trống = mặc định của provider
      - LLM_EMBEDDING_MODEL: model embedding (ingest / RAG)
      - LLM_CHAT_MODEL: model hội thoại (draft/eval sau này; ingest không dùng)
    """
    key = os.getenv("LLM_API_KEY", "").strip()
    raw_url = _strip_or_none(os.getenv("LLM_BASE_URL")) or _strip_or_none(os.getenv("BASE_URL"))
    if raw_url and raw_url.endswith("/"):
        raw_url = raw_url.rstrip("/")
    emb = os.getenv("LLM_EMBEDDING_MODEL", "text-embedding-3-small").strip()
    chat = os.getenv("LLM_CHAT_MODEL", "gpt-4o-mini").strip()
    return LLMEnv(api_key=key, base_url=raw_url, embedding_model=emb, chat_model=chat)
