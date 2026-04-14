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

from .chroma_client import get_chroma_collection

def init_sqlite_db(db_path: Path) -> sqlite3.Connection:
    """Tạo 1 bảng duy nhất lưu trữ Cả 3 thông tin"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS contacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            company TEXT,
            industry TEXT,
            UNIQUE(name, company, industry)
        )
    """)
    conn.commit()
    return conn

def extract_emails_via_llm(raw_text: str, env) -> list[Dict[str, Any]]:
    """Dùng LLM để rút trích danh sách email/metadata từ văn bản thô."""
    client = openai.Client(api_key=env.api_key, base_url=env.base_url)
    
    system_prompt = '''Bạn là một trợ lý phân tích dữ liệu email.
    Nhiệm vụ của bạn là đọc một đoạn văn bản thô, nhận diện và tách từng email/tin nhắn/ghi chú riêng biệt có trong đó.
    Trả về một MẢNG JSON (lưu ý không để thẻ markdown định dạng JSON bên ngoài). Mỗi phần tử trong mảng là một dict biểu diễn một email:
    [
        {
            "name": "Tên khách hàng hoặc tên người (nếu có, không thì 'Unknown')",
            "company": "Tên công ty khách hàng / đối tác (nếu có, không thì 'Unknown')",
            "industry": "Ngành nghề (nếu có, không thì 'Unknown')",
            "body": "Nội dung nguyên bản của email đó không thay đổi"
        }
    ]'''

    try:
        response = client.chat.completions.create(
            model=env.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Văn bản raw như sau:\n\n{raw_text}"}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content or "[]"
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
            
        parsed = json.loads(content.strip())
        if not isinstance(parsed, list):
            if isinstance(parsed, dict):
                parsed = [parsed]
            else:
                parsed = []
                
        # Fallback 
        for item in parsed:
            for key in ["name", "company", "industry", "body"]: 
                if key not in item:
                    item[key] = "Unknown"
        return parsed
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return []

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

def run_ingest_process(raw_text: str, source_name: str, max_tokens: int = 500, collection_name: str = "email_chunks") -> dict:
    """Xử lý text, trích xuất và lưu vào DB (dùng chung cho Web API và các nguồn khác)."""
    env = load_llm_env()
    if not env.api_key:
        return {"error": "Thiếu LLM_API_KEY ở .env"}

    extracted_emails = extract_emails_via_llm(raw_text, env)
    if not extracted_emails:
        return {"error": "Nội dung rỗng hoặc LLM không tìm thấy email nào.", "logs": []}

    sql_db_path = _ROOT / "data" / "customers.db"
    if not sql_db_path.parent.exists():
        sql_db_path.parent.mkdir(parents=True)
    
    sql_conn = init_sqlite_db(sql_db_path)
    cursor = sql_conn.cursor()

    all_docs = []
    all_ids = []
    all_meta = []
    chunk_counter = 0
    logs = []
    
    for idx, extracted in enumerate(extracted_emails, 1):
        c_name = str(extracted.get("name", "Unknown")).strip()
        c_comp = str(extracted.get("company", "Unknown")).strip()
        c_indus = str(extracted.get("industry", "Unknown")).strip()
        
        cursor.execute("INSERT OR IGNORE INTO contacts (name, company, industry) VALUES (?, ?, ?)", (c_name, c_comp, c_indus))
        cursor.execute("SELECT id FROM contacts WHERE name=? AND company=? AND industry=?", (c_name, c_comp, c_indus))
        contact_id = (cursor.fetchone() or [-1])[0]
        
        semantic_prefix = (
            f"Tên khách hàng: {c_name}\nCông ty: {c_comp}\nNgành: {c_indus}\n\nNội dung: "
        )
        
        body = extracted.get("body", "")
        for chunk_body in chunk_by_tokens(body, max_tokens=max_tokens, overlap=50):
            full_chunk_text = semantic_prefix + chunk_body
            cid = stable_chunk_id(chunk_counter, full_chunk_text)
            all_docs.append(full_chunk_text)
            all_ids.append(cid)
            all_meta.append({
                "contact_id": contact_id,
                "source_file": source_name
            })
            chunk_counter += 1
            
        logs.append(f"Email {idx}: Bóc tách thành công [ {c_name} | {c_comp} | {c_indus} ].")

    sql_conn.commit()
    sql_conn.close()

    if not all_docs:
        return {"error": "Hoàn tất nhưng không có mảnh vector nào được tạo.", "logs": logs}

    collection = get_chroma_collection(env, collection_name=collection_name)
    collection.upsert(documents=all_docs, ids=all_ids, metadatas=all_meta)

    logs.append(f"✅ Hoàn tất lưu Master SQL Data.")
    logs.append(f"✅ Hoàn tất nạp {len(all_docs)} Vector Chunk vào ChromaDB ({collection_name}).")

    return {"status": "success", "logs": logs, "chunks": len(all_docs)}

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest text hỗn hợp, lưu vào 3 bảng Thực Thể độc lập SQL và Vector Chroma.")
    parser.add_argument("input_path", type=Path, help="File txt hoặc thư mục chứa các file txt (không cần ngăn cách ==========)")
    parser.add_argument("--collection", default="email_chunks", help="Tên collection Chroma")
    parser.add_argument("--max-tokens", type=int, default=500)
    args = parser.parse_args()

    path = args.input_path.resolve()
    if not path.exists():
        print(f"Không tìm thấy: {path}", file=sys.stderr)
        sys.exit(1)

    env = load_llm_env()
    if not env.api_key:
        print("Thiếu LLM_API_KEY.", file=sys.stderr)
        sys.exit(1)

    # Lấy danh sách file cần xử lý
    files_to_process = []
    if path.is_file():
        files_to_process.append(path)
    elif path.is_dir():
        files_to_process.extend(path.glob('*.txt'))

    if not files_to_process:
        print("Không có file txt nào để xử lý.", file=sys.stderr)
        sys.exit(1)

    # Cài đặt SQL Connection
    sql_db_path = _ROOT / "data" / "customers.db"
    sql_conn = init_sqlite_db(sql_db_path)
    cursor = sql_conn.cursor()

    all_docs: list[str] = []
    all_ids: list[str] = []
    all_meta: list[dict] = []
    chunk_counter = 0

    print(f"Bắt đầu xử lý {len(files_to_process)} file qua LLM ({env.chat_model})...")

    for fpath in files_to_process:
        print(f"\n- Đang dùng LLM để trích xuất email từ file: {fpath.name}...")
        raw = fpath.read_text(encoding="utf-8", errors="replace")
        if not raw.strip():
            continue
            
        extracted_emails = extract_emails_via_llm(raw, env)
        if not extracted_emails:
            print("   -> Bỏ qua file do LLM không tìm thấy email hoặc có lỗi.")
            continue
            
        print(f"   -> LLM tìm thấy {len(extracted_emails)} email trong file này.")
        
        # Ghi log kết quả JSON từ LLM ra file trước khi insert DB
        log_file = fpath.parent / f"llm_log_{fpath.name}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(extracted_emails, f, ensure_ascii=False, indent=4)
        print(f"   -> Đã ghi log kết quả chuẩn JSON ra file: {log_file}")
        
        for idx, extracted in enumerate(extracted_emails, 1):
            c_name = str(extracted.get("name", "Unknown")).strip()
            c_comp = str(extracted.get("company", "Unknown")).strip()
            c_indus = str(extracted.get("industry", "Unknown")).strip()
            
            cursor.execute("INSERT OR IGNORE INTO contacts (name, company, industry) VALUES (?, ?, ?)", (c_name, c_comp, c_indus))
            cursor.execute("SELECT id FROM contacts WHERE name=? AND company=? AND industry=?", (c_name, c_comp, c_indus))
            contact_id = (cursor.fetchone() or [-1])[0]
            
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
                # Link độc lập tới bảng SQL bên ngoài
                all_meta.append({
                    "contact_id": contact_id,
                    "source_file": str(fpath.name)
                })
                chunk_counter += 1

    sql_conn.commit()
    sql_conn.close()

    if not all_docs:
        print("Không tạo được vector chunk nào.", file=sys.stderr)
        sys.exit(1)

    chroma_path = _ROOT / "chroma_data"
    chroma_path.mkdir(parents=True, exist_ok=True)

    collection = get_chroma_collection(env, collection_name=args.collection)
    collection.upsert(documents=all_docs, ids=all_ids, metadatas=all_meta)

    print(f"\nThành công! Đã tách dữ liệu vào 1 bảng:")
    print(f" -> SQL Thực thể Master: {sql_db_path}")
    print(f" -> Vector Chroma: {len(all_docs)} đoạn chứa Khóa Ngoại contact_id")

if __name__ == "__main__":
    main()
