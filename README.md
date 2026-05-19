# DocMind — RAG Q&A App

An end-to-end Retrieval-Augmented Generation (RAG) pipeline using **Claude API**, **FAISS**, and **sentence-transformers**, served via **FastAPI** with a polished browser UI.

---

## Project Structure

```
rag-qa-app/
├── backend/
│   ├── main.py              # FastAPI app — upload, index, query, reset
│   └── requirements.txt
├── frontend/
│   └── index.html           # Single-file UI (served by FastAPI)
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run

```bash
uvicorn main:app --reload --port 8000
```

### 4. Open the UI

Visit **http://localhost:8000** in your browser.

---

## How It Works

| Step | What happens |
|------|-------------|
| **Upload** | PDF/TXT/MD is parsed and split into 500-token chunks with 50-token overlap |
| **Embed** | Each chunk is embedded using `all-MiniLM-L6-v2` (runs locally, free) |
| **Store** | Embeddings are added to a FAISS `IndexFlatL2` vector store in memory |
| **Query** | User question is embedded → top-5 nearest chunks retrieved |
| **Generate** | Chunks + question are sent to Claude with a strict grounding prompt |
| **Respond** | Claude returns an answer with inline `[Source N]` citations |

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `POST` | `/upload` | Upload and index a document |
| `POST` | `/ask` | Ask a question, get answer + sources |
| `GET`  | `/status` | List indexed docs and chunk count |
| `DELETE` | `/reset` | Clear all documents from memory |

---

## Customization

**Swap vector store** — Replace FAISS with Pinecone or ChromaDB for persistence:
```python
# ChromaDB example
import chromadb
client = chromadb.Client()
collection = client.create_collection("docs")
```

**Adjust chunking** — In `chunk_text()`, change `chunk_size` and `chunk_overlap`:
```python
# Larger chunks = more context per retrieval
# Smaller chunks = more precise retrieval
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
```

**Multi-doc filtering** — Add a `filename` filter to retrieve only from a specific doc:
```python
# Filter stored_metadata by filename before building response
```

---

## Resume Bullet

> Built an end-to-end RAG pipeline using Claude API, FAISS, and sentence-transformers; parses PDFs via FastAPI, retrieves semantically similar chunks, and prompts Claude to return grounded answers with inline source citations — deployed as a single-origin web app with a document management UI.
