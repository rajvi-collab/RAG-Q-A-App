import os
import shutil
import numpy as np
import fitz  # PyMuPDF
import faiss
import anthropic

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="RAG Q&A API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
async def serve_ui():
    return FileResponse(os.path.join(frontend_path, "index.html"))

# ── Global state (in-memory; swap for Redis/DB in production) ────────────────

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
claude_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

vector_index: faiss.IndexFlatL2 | None = None
stored_chunks: list[str] = []
stored_metadata: list[dict] = []   # {filename, chunk_index}
uploaded_files: list[str] = []

# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_text(text)


def build_or_update_index(new_chunks: list[str], metadata: list[dict]):
    global vector_index, stored_chunks, stored_metadata

    embeddings = embed_model.encode(new_chunks).astype("float32")

    if vector_index is None:
        dim = embeddings.shape[1]
        vector_index = faiss.IndexFlatL2(dim)

    vector_index.add(embeddings)
    stored_chunks.extend(new_chunks)
    stored_metadata.extend(metadata)


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    if vector_index is None or len(stored_chunks) == 0:
        return []

    query_vec = embed_model.encode([query]).astype("float32")
    distances, indices = vector_index.search(query_vec, min(top_k, len(stored_chunks)))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(stored_chunks):
            results.append({
                "text": stored_chunks[idx],
                "metadata": stored_metadata[idx],
                "score": float(dist),
            })
    return results


def ask_claude(query: str, context_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source {i+1} — {c['metadata']['filename']}]:\n{c['text']}"
        for i, c in enumerate(context_chunks)
    )

    prompt = f"""You are a precise document assistant. Answer the user's question 
using ONLY the provided context below. Cite sources inline as [Source N].
If the context does not contain enough information, say exactly:
"I don't have enough information in the uploaded documents to answer that."

Context:
{context}

Question: {query}

Answer:"""

    response = claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text

# ── Routes ───────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    allowed = {".pdf", ".txt", ".md"}
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Use PDF, TXT, or MD.")

    tmp_path = f"/tmp/{file.filename}"
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    if ext == ".pdf":
        text = extract_text_from_pdf(tmp_path)
    else:
        text = extract_text_from_txt(tmp_path)

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(400, "Could not extract any text from this file.")

    metadata = [{"filename": file.filename, "chunk_index": i} for i in range(len(chunks))]
    build_or_update_index(chunks, metadata)
    uploaded_files.append(file.filename)

    return {
        "message": f"Successfully indexed '{file.filename}'",
        "chunks_indexed": len(chunks),
        "total_chunks": len(stored_chunks),
    }


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


@app.post("/ask")
async def ask(req: QueryRequest):
    if vector_index is None:
        raise HTTPException(400, "No documents uploaded yet. Please upload a file first.")

    relevant = retrieve(req.question, top_k=req.top_k)
    if not relevant:
        raise HTTPException(500, "Retrieval failed.")

    answer = ask_claude(req.question, relevant)

    return {
        "answer": answer,
        "sources": [
            {"filename": r["metadata"]["filename"], "excerpt": r["text"][:200] + "..."}
            for r in relevant
        ],
    }


@app.get("/status")
async def status():
    return {
        "documents": uploaded_files,
        "total_chunks": len(stored_chunks),
        "ready": vector_index is not None,
    }


@app.delete("/reset")
async def reset():
    global vector_index, stored_chunks, stored_metadata, uploaded_files
    vector_index = None
    stored_chunks = []
    stored_metadata = []
    uploaded_files = []
    return {"message": "Index cleared."}
