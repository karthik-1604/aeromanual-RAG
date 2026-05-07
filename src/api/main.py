from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List
import uvicorn

from src.embeddings.vector_store import load_index, index_exists
from src.pipeline.rag_chain import build_rag_chain, format_response

# Global chain variable
rag_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain
    if not index_exists():
        raise RuntimeError(
            "FAISS index not found. Run: python src/embeddings/vector_store.py first."
        )
    print("Loading FAISS index...")
    vs = load_index()
    retriever = vs.as_retriever(search_kwargs={"k": 5})
    rag_chain = build_rag_chain(retriever)
    print("RAG pipeline ready.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="AeroManual-RAG API",
    description="Natural language Q&A over FAA aerospace maintenance manuals",
    version="1.0.0",
    lifespan=lifespan
)


# --- Request / Response Schemas ---

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the torque specs for cylinder head bolts?",
                "top_k": 5
            }
        }


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    num_sources: int


# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok", "pipeline_loaded": rag_chain is not None}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    try:
        result = rag_chain.invoke({"query": req.question})
        return format_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {
        "message": "AeroManual RAG API",
        "docs": "/docs",
        "health": "/health",
        "query": "POST /query"
    }


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
