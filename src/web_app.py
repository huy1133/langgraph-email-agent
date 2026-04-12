from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import io
import os
import sqlite3
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

# Import script modules
from src.ingest_from_text import (
    init_sqlite_db,
    extract_emails_via_llm,
    chunk_by_tokens,
    stable_chunk_id,
    _ROOT,
    load_llm_env,
    run_ingest_process
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

class DraftRequest(BaseModel):
    contact_id: int
    name: str
    company: str
    industry: str
    goal: str
    objective: str = ""
    tone: str = ""
    chat_history: list = []

@app.get("/")
async def root():
    f = static_dir / "chat.html"
    if f.exists():
        return HTMLResponse(content=f.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Không tìm thấy static/chat.html</h1>")

@app.get("/ingest")
async def ingest_page():
    f = static_dir / "ingest.html"
    if f.exists():
        return HTMLResponse(content=f.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Không tìm thấy static/ingest.html</h1>")

@app.get("/chat")
async def chat_page():
    f = static_dir / "chat.html"
    if f.exists():
        return HTMLResponse(content=f.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Không tìm thấy static/chat.html</h1>")

@app.get("/api/contacts")
async def api_contacts():
    sql_db_path = _ROOT / "data" / "customers.db"
    if not sql_db_path.exists():
        return []
    try:
        conn = sqlite3.connect(sql_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, company, industry FROM contacts ORDER BY name ASC")
        rows = cursor.fetchall()
        conn.close()
        
        contacts = []
        for r in rows:
            contacts.append({
                "id": r[0],
                "name": r[1],
                "company": r[2],
                "industry": r[3]
            })
        return contacts
    except Exception as e:
        return []

@app.post("/api/ingest")
async def api_ingest(payload: IngestRequest):
    result = run_ingest_process(payload.raw_text, source_name="Web UI Paste")
    if "error" in result:
        return JSONResponse(status_code=400, content=result)
    return result

@app.post("/api/ingest_file")
async def api_ingest_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        raw_text = content.decode("utf-8")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": "Không thể đọc file text.", "logs": [str(e)]})
        
    result = run_ingest_process(raw_text, source_name=file.filename)
    if "error" in result:
        return JSONResponse(status_code=400, content=result)
    return result

@app.post("/api/generate_draft")
async def api_generate_draft(payload: DraftRequest):
    from src.agent import email_agent_app
    try:
        initial_state = {
            "user_inputs": {
                "contact_id": payload.contact_id,
                "name": payload.name,
                "company": payload.company,
                "industry": payload.industry,
                "goal": payload.goal,
                "objective": payload.objective,
                "tone": payload.tone
            },
            "email_history": "",
            "current_draft": "",
            "feedback": "",
            "evaluate_status": "",
            "iterations": 0,
            "chat_history": payload.chat_history
        }
        
        final_state = email_agent_app.invoke(initial_state)
        return {
            "draft": final_state.get("current_draft", ""),
            "feedback": final_state.get("feedback", ""),
            "iterations": final_state.get("iterations", 0)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

class SaveRequest(BaseModel):
    draft: str
    contact_id: int
    name: str
    company: str
    industry: str

@app.post("/api/save_to_knowledge")
async def api_save_to_knowledge(payload: SaveRequest):
    from src.ingest_from_text import chunk_by_tokens, stable_chunk_id, _ROOT
    from src.llm_env import load_llm_env
    import chromadb
    from chromadb.utils import embedding_functions

    env = load_llm_env()
    if not env.api_key:
        return JSONResponse(status_code=400, content={"error": "Thiếu LLM_API_KEY"})

    try:
        # Bổ sung semantic prefix để vector search chuẩn
        semantic_prefix = (
            f"Tên khách hàng: {payload.name}\n"
            f"Công ty: {payload.company}\n"
            f"Ngành: {payload.industry}\n\nNội dung: "
        )

        all_docs, all_ids, all_meta = [], [], []
        for i, chunk in enumerate(chunk_by_tokens(payload.draft, max_tokens=500, overlap=50)):
            full_text = semantic_prefix + chunk
            cid = stable_chunk_id(i, full_text + "_saved")
            all_docs.append(full_text)
            all_ids.append(cid)
            all_meta.append({"contact_id": payload.contact_id, "source_file": "saved_from_chat"})

        chroma_path = _ROOT / "chroma_data"
        emb_kwargs = {"api_key": env.api_key, "model_name": env.embedding_model}
        if env.base_url:
            emb_kwargs["api_base"] = env.base_url

        ef = embedding_functions.OpenAIEmbeddingFunction(**emb_kwargs)
        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_or_create_collection(name="email_chunks", embedding_function=ef)
        collection.upsert(documents=all_docs, ids=all_ids, metadatas=all_meta)

        return {"status": "success", "chunks": len(all_docs)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.web_app:app", host="127.0.0.1", port=8000, reload=True)
