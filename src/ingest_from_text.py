"""
Đọc file text chứa dữ liệu email lộn xộn (ngăn cách bằng ==========).
Sử dụng LLM để trích xuất metadata: name, company, industry. 
Lưu 3 Entities riêng rẽ vào 3 bảng (customers, companies, industries) ở data/customers.db (SQLite).
Lưu Vector Emails vào ./chroma_data (ChromaDB), gắn siêu dữ liệu chứa 3 Khóa Ngoại từ DB SQL.

Biến môi trường: LLM_API_KEY (bắt buộc), LLM_EMBEDDING_MODEL, LLM_CHAT_MODEL, BASE_URL.

Chạy từ thư mục gốc dự án:
  python -m src.ingest_from_text data/emails_10_companies_10_mails.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import io
import sys
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding='utf-8')
if isinstance(sys.stderr, io.TextIOWrapper):
    sys.stderr.reconfigure(encoding='utf-8')

import tiktoken
from dotenv import load_dotenv

from .llm_env import load_llm_env

# Load .env từ thư mục gốc dự án
_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")

try:
    import chromadb
    from chromadb.utils import embedding_functions
    import openai
except ImportError as e:
    print("Thiếu thư viện. Chạy: pip install -r requirements.txt", file=sys.stderr)
    raise e

def parse_raw_emails_by_separator(text: str, separator: str = "==========") -> list[str]:
    """Cắt file thành các chuỗi email thô (raw) dựa vào dấu gạch ngang."""
    raw_blocks = text.split(separator)
    valid_blocks = [b.strip() for b in raw_blocks if b.strip()]
    return valid_blocks

def init_sqlite_db(db_path: Path) -> sqlite3.Connection:
    """Tạo 3 bảng thực thể Độc lập vào DB nếu chưa có"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Bảng Khách hàng
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        )
    """)
    # 2. Bảng Công ty
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        )
    """)
    # 3. Bảng Ngành nghề
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS industries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        )
    """)
    conn.commit()
    return conn

def extract_metadata_via_llm(raw_text: str, env) -> Optional[Dict[str, Any]]:
    """Dùng LLM để rút trích metadata: name, company, industry, goal, tone."""
    client = openai.Client(api_key=env.api_key, base_url=env.base_url)
    
    system_prompt = '''Bạn là một trợ lý phân tích dữ liệu email.
    Trích xuất thông tin dưới dạng JSON từ đoạn nội dung, mail, hoặc ghi chú của người dùng.
    Luôn trả về cấu trúc JSON sau (lưu ý không để thẻ markdown định dạng JSON bên ngoài):
    {
    "name": "Tên khách hàng hoặc tên người (nếu có, không thì 'Unknown')",
    "company": "Tên công ty khách hàng / đối tác (nếu có, không thì 'Unknown')",
    "industry": "Ngành nghề (nếu có, không thì 'Unknown')",
    "body": "Nội dung nguyên bản không thay đổi"        
    }'''

    try:
        response = client.chat.completions.create(
            model=env.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Văn bản raw như sau:\n\n{raw_text}"}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        # Fallback 
        for key in ["name", "company", "industry", "body"]: 
            if key not in parsed:
                parsed[key] = "Unknown"
        return parsed
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return None

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

def stable_chunk_id(chunk_index: int, text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"chk_{chunk_index}_{h}"

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest text hỗn hợp, lưu vào 3 bảng Thực Thể độc lập SQL và Vector Chroma.")
    parser.add_argument("input_file", type=Path, help="File txt có ngăn cách ==========")
    parser.add_argument("--collection", default="email_chunks", help="Tên collection Chroma")
    parser.add_argument("--max-tokens", type=int, default=500)
    args = parser.parse_args()

    path = args.input_file.resolve()
    if not path.is_file():
        print(f"Không tìm thấy file: {path}", file=sys.stderr)
        sys.exit(1)

    env = load_llm_env()
    if not env.api_key:
        print("Thiếu LLM_API_KEY.", file=sys.stderr)
        sys.exit(1)

    # Đọc raw template
    raw = path.read_text(encoding="utf-8", errors="replace")
    raw_blocks = parse_raw_emails_by_separator(raw)
    
    if not raw_blocks:
        print("File rỗng.", file=sys.stderr)
        sys.exit(1)

    # Cài đặt SQL Connection
    sql_db_path = _ROOT / "data" / "customers.db"
    sql_conn = init_sqlite_db(sql_db_path)
    cursor = sql_conn.cursor()

    all_docs: list[str] = []
    all_ids: list[str] = []
    all_meta: list[dict] = []
    chunk_counter = 0

    print(f"Bắt đầu xử lý {len(raw_blocks)} blocks qua LLM ({env.chat_model})...")

    for idx, block in enumerate(raw_blocks, 1):
        print(f" - Xử lý block {idx}/{len(raw_blocks)}...")
        extracted = extract_metadata_via_llm(block, env)
        if not extracted:
            print("   -> Bỏ qua block do LLM lỗi/chặn proxy.")
            continue
            
        c_name = str(extracted.get("name", "Unknown")).strip()
        c_comp = str(extracted.get("company", "Unknown")).strip()
        c_indus = str(extracted.get("industry", "Unknown")).strip()
        
        # INSERT OR IGNORE cho Khách hàng
        cursor.execute("INSERT OR IGNORE INTO customers (name) VALUES (?)", (c_name,))
        cursor.execute("SELECT id FROM customers WHERE name=?", (c_name,))
        customer_id = (cursor.fetchone() or [-1])[0]

        # INSERT OR IGNORE cho Công ty
        cursor.execute("INSERT OR IGNORE INTO companies (name) VALUES (?)", (c_comp,))
        cursor.execute("SELECT id FROM companies WHERE name=?", (c_comp,))
        company_id = (cursor.fetchone() or [-1])[0]

        # INSERT OR IGNORE cho Ngành nghề
        cursor.execute("INSERT OR IGNORE INTO industries (name) VALUES (?)", (c_indus,))
        cursor.execute("SELECT id FROM industries WHERE name=?", (c_indus,))
        industry_id = (cursor.fetchone() or [-1])[0]
        
        # Bọc Tiền Tố vào nội dung Chunk Vector 
        semantic_prefix = (
            f"Tên khách hàng: {c_name}\nCông ty: {c_comp}\nNgành: {c_indus}\n"
            f"\n\nNội dung: "
        )
        
        # Thêm Vector vào Vector DB
        body = extracted.get("body", "")
        for local_i, chunk_body in enumerate(chunk_by_tokens(body, max_tokens=args.max_tokens, overlap=50)):
            full_chunk_text = semantic_prefix + chunk_body
            cid = stable_chunk_id(chunk_counter, full_chunk_text)
            all_docs.append(full_chunk_text)
            all_ids.append(cid)
            # Link độc lập tới 3 bảng SQL bên ngoài
            all_meta.append({
                "customer_id": customer_id, 
                "company_id": company_id,
                "industry_id": industry_id,
                "source_file": str(path.name)
            })
            chunk_counter += 1

    sql_conn.commit()
    sql_conn.close()

    if not all_docs:
        print("Không tạo được vector chunk nào.", file=sys.stderr)
        sys.exit(1)

    chroma_path = _ROOT / "chroma_data"
    chroma_path.mkdir(parents=True, exist_ok=True)

    emb_kwargs: dict = {"api_key": env.api_key, "model_name": env.embedding_model}
    if env.base_url:
        emb_kwargs["api_base"] = env.base_url
        
    ef = embedding_functions.OpenAIEmbeddingFunction(**emb_kwargs)
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(name=args.collection, embedding_function=ef)
    collection.upsert(documents=all_docs, ids=all_ids, metadatas=all_meta)

    print(f"\nThành công! Đã tách dữ liệu 3 bảng 3 ngã:")
    print(f" -> SQL Thực thể Master: {sql_db_path}")
    print(f" -> Vector Chroma: {len(all_docs)} đoạn chứa 3 Khóa Ngoại ForeignKey")

if __name__ == "__main__":
    main()
