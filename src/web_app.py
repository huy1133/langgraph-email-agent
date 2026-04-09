from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import io
import os
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

# Import script modules
from src.ingest_from_text import (
    parse_raw_emails_by_separator,
    init_sqlite_db,
    extract_metadata_via_llm,
    chunk_by_tokens,
    stable_chunk_id,
    _ROOT,
    load_llm_env
)

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding='utf-8')

app = FastAPI(title="Email RAG Ingestion Portal")

# Setup Static Files for HTML/CSS
static_dir = _ROOT / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

class IngestRequest(BaseModel):
    raw_text: str

@app.get("/")
async def root():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Không tìm thấy static/index.html</h1>")

@app.post("/api/ingest")
async def api_ingest(payload: IngestRequest):
    env = load_llm_env()
    if not env.api_key:
        return JSONResponse(status_code=500, content={"error": "Thiếu LLM_API_KEY ở .env"})

    raw_blocks = parse_raw_emails_by_separator(payload.raw_text)
    if not raw_blocks:
        return JSONResponse(status_code=400, content={"error": "Nội dung rỗng hoặc không có format hợp lệ."})

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
    
    for idx, block in enumerate(raw_blocks, 1):
        extracted = extract_metadata_via_llm(block, env)
        if not extracted:
            logs.append(f"Block {idx}: Lỗi gọi LLM (bị chặn Proxy hoặc đứt mạng). Ghi nhận THẤT BẠI.")
            continue
            
        c_name = str(extracted.get("name", "Unknown")).strip()
        c_comp = str(extracted.get("company", "Unknown")).strip()
        c_indus = str(extracted.get("industry", "Unknown")).strip()
        
        # INSERT OR IGNORE
        cursor.execute("INSERT OR IGNORE INTO customers (name) VALUES (?)", (c_name,))
        cursor.execute("SELECT id FROM customers WHERE name=?", (c_name,))
        customer_id = (cursor.fetchone() or [-1])[0]

        cursor.execute("INSERT OR IGNORE INTO companies (name) VALUES (?)", (c_comp,))
        cursor.execute("SELECT id FROM companies WHERE name=?", (c_comp,))
        company_id = (cursor.fetchone() or [-1])[0]

        cursor.execute("INSERT OR IGNORE INTO industries (name) VALUES (?)", (c_indus,))
        cursor.execute("SELECT id FROM industries WHERE name=?", (c_indus,))
        industry_id = (cursor.fetchone() or [-1])[0]
        
        semantic_prefix = (
            f"Tên khách hàng: {c_name}\nCông ty: {c_comp}\nNgành: {c_indus}\n\nNội dung: "
        )
        
        body = extracted.get("body", "")
        for chunk_body in chunk_by_tokens(body, max_tokens=500, overlap=50):
            full_chunk_text = semantic_prefix + chunk_body
            cid = stable_chunk_id(chunk_counter, full_chunk_text)
            all_docs.append(full_chunk_text)
            all_ids.append(cid)
            all_meta.append({
                "customer_id": customer_id, 
                "company_id": company_id,
                "industry_id": industry_id,
                "source_file": "Web UI Paste"
            })
            chunk_counter += 1
            
        logs.append(f"Block {idx}: Bóc tách thành công [ {c_name} | {c_comp} | {c_indus} ].")

    sql_conn.commit()
    sql_conn.close()

    if not all_docs:
        return JSONResponse(status_code=400, content={"error": "Hoàn tất nhưng không có mảnh vector nào được tạo.", "logs": logs})

    chroma_path = _ROOT / "chroma_data"
    emb_kwargs = {"api_key": env.api_key, "model_name": env.embedding_model}
    if env.base_url:
        emb_kwargs["api_base"] = env.base_url
        
    ef = embedding_functions.OpenAIEmbeddingFunction(**emb_kwargs)
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(name="email_chunks", embedding_function=ef)
    collection.upsert(documents=all_docs, ids=all_ids, metadatas=all_meta)

    logs.append(f"✅ Hoàn tất lưu Master SQL Data.")
    logs.append(f"✅ Hoàn tất nạp {len(all_docs)} Vector Chunk vào ChromaDB.")

    return {"status": "success", "logs": logs, "chunks": len(all_docs)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.web_app:app", host="127.0.0.1", port=8000, reload=True)
