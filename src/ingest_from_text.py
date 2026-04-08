"""
Đọc một file .txt chứa mail nhiều khách (phân tách bằng <<<CLIENT_ID:...>>>),
chunk → embedding (LLM_EMBEDDING_MODEL, BASE_URL tùy chọn) → lưu ChromaDB cục bộ (./chroma_data).

Biến môi trường: LLM_API_KEY (bắt buộc), LLM_EMBEDDING_MODEL, BASE_URL. LLM_CHAT_MODEL không dùng ở bước này.

Chạy từ thư mục gốc dự án:
  python -m src.ingest_from_text data/emails_mixed.txt
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import tiktoken
from dotenv import load_dotenv

from .llm_env import load_llm_env

# Load .env từ thư mục gốc dự án (cha của thư mục src)
_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError as e:
    print("Thiếu thư viện. Chạy: pip install -r requirements.txt", file=sys.stderr)
    raise e

CLIENT_MARKER = re.compile(r"^<<<CLIENT_ID:([^>\s]+)>>>\s*$", re.MULTILINE)


def strip_comment_lines(raw: str) -> str:
    """Bỏ các dòng chỉ là comment (# ...), giữ nội dung mail."""
    lines = []
    for line in raw.splitlines():
        s = line.strip()
        if s.startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def parse_by_client(text: str) -> list[tuple[str, str]]:
    """
    Tách file thành các cặp (client_id, nội_dung_block).
    Nếu không có marker nào: toàn bộ file gán client_id = 'unknown'.
    """
    text = strip_comment_lines(text)
    if not text:
        return []

    matches = list(CLIENT_MARKER.finditer(text))
    if not matches:
        return [("unknown", text)]

    segments: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        client_id = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            segments.append((client_id, body))

    if not segments and text:
        return [("unknown", text)]
    return segments


def chunk_by_tokens(
    text: str,
    max_tokens: int = 500,
    overlap: int = 50,
    encoding_name: str = "cl100k_base",
) -> list[str]:
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)
    if not tokens:
        return []
    chunks: list[str] = []
    step = max(1, max_tokens - overlap)
    i = 0
    while i < len(tokens):
        piece = tokens[i : i + max_tokens]
        chunks.append(enc.decode(piece))
        i += step
    return chunks


def stable_chunk_id(client_id: str, chunk_index: int, text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{client_id}_{chunk_index}_{h}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest file text hỗn hợp vào ChromaDB.")
    parser.add_argument(
        "input_file",
        type=Path,
        help="Đường dẫn file .txt (UTF-8), dùng marker <<<CLIENT_ID:ten_khach>>>",
    )
    parser.add_argument(
        "--collection",
        default="email_chunks",
        help="Tên collection Chroma (mặc định: email_chunks)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Số token tối đa mỗi chunk (mặc định 500)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Token overlap giữa các chunk (mặc định 50)",
    )
    args = parser.parse_args()

    path = args.input_file.resolve()
    if not path.is_file():
        print(f"Không tìm thấy file: {path}", file=sys.stderr)
        sys.exit(1)

    env = load_llm_env()
    if not env.api_key:
        print(
            "Thiếu LLM_API_KEY. Tạo file .env từ .env.example và điền key.",
            file=sys.stderr,
        )
        sys.exit(1)
    raw = path.read_text(encoding="utf-8", errors="replace")
    pairs = parse_by_client(raw)
    if not pairs:
        print("File rỗng hoặc không còn nội dung sau khi bỏ comment.", file=sys.stderr)
        sys.exit(1)

    all_docs: list[str] = []
    all_ids: list[str] = []
    all_meta: list[dict] = []
    chunk_counter = 0

    for client_id, body in pairs:
        for local_i, chunk in enumerate(
            chunk_by_tokens(body, max_tokens=args.max_tokens, overlap=args.overlap)
        ):
            cid = stable_chunk_id(client_id, chunk_counter, chunk)
            all_docs.append(chunk)
            all_ids.append(cid)
            all_meta.append(
                {
                    "client_id": client_id,
                    "chunk_index": chunk_counter,
                    "source_file": str(path.name),
                }
            )
            chunk_counter += 1

    if not all_docs:
        print("Không tạo được chunk nào.", file=sys.stderr)
        sys.exit(1)

    chroma_path = _ROOT / "chroma_data"
    chroma_path.mkdir(parents=True, exist_ok=True)

    emb_kwargs: dict = {
        "api_key": env.api_key,
        "model_name": env.embedding_model,
    }
    if env.base_url:
        emb_kwargs["api_base"] = env.base_url
    ef = embedding_functions.OpenAIEmbeddingFunction(**emb_kwargs)
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(
        name=args.collection,
        embedding_function=ef,
    )

    # Tránh trùng id khi chạy lại: xóa id cũ trùng (tuỳ chọn đơn giản: add mới — Chroma báo lỗi nếu trùng)
    # Ở đây dùng upsert nếu có; chromadb Collection có upsert
    collection.upsert(documents=all_docs, ids=all_ids, metadatas=all_meta)

    print(f"Đã upsert {len(all_docs)} chunk vào Chroma.")
    print(f"  Thư mục DB: {chroma_path}")
    print(f"  Collection: {args.collection}")
    print("  Metadata mỗi chunk: client_id, chunk_index, source_file")


if __name__ == "__main__":
    main()
