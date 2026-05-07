import sys, os

# MUST be set before any HuggingFace imports
# Forces model to download inside project folder, not global cache
os.environ["HF_HOME"] = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "models"
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from pathlib import Path
from typing import List
from tqdm.auto import tqdm
import logging

logger = logging.getLogger(__name__)

# Absolute paths — works from anywhere (notebooks, terminal, tests)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
INDEX_PATH = os.path.join(ROOT_DIR, "data", "faiss_index")
MODEL_CACHE_PATH = os.path.join(ROOT_DIR, "data", "models")


def get_embedder() -> HuggingFaceEmbeddings:
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    Path(MODEL_CACHE_PATH).mkdir(parents=True, exist_ok=True)
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        cache_folder=MODEL_CACHE_PATH,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def build_index(chunks: List[Document]) -> FAISS:
    logger.info(f"Building FAISS index for {len(chunks)} chunks...")
    embedder = get_embedder()

    # Embed in batches with tqdm progress bar
    batch_size = 32
    all_texts = [c.page_content for c in chunks]
    all_metadatas = [c.metadata for c in chunks]
    all_embeddings = []

    print(f"Embedding {len(chunks)} chunks in batches of {batch_size}...")
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding"):
        batch_texts = all_texts[i:i + batch_size]
        batch_embeddings = embedder.embed_documents(batch_texts)
        all_embeddings.extend(batch_embeddings)

    print("Building FAISS index...")
    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip(all_texts, all_embeddings)),
        embedding=embedder,
        metadatas=all_metadatas
    )

    Path(INDEX_PATH).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(INDEX_PATH)
    logger.info(f"Index saved to {INDEX_PATH}")
    return vectorstore


def load_index() -> FAISS:
    embedder = get_embedder()
    return FAISS.load_local(
        INDEX_PATH,
        embedder,
        allow_dangerous_deserialization=True
    )


def index_exists() -> bool:
    return Path(f"{INDEX_PATH}/index.faiss").exists()


if __name__ == "__main__":
    from src.ingestion.pdf_loader import load_all_documents
    from src.ingestion.chunker import pages_to_langchain_docs, chunk_documents

    print("Step 1: Loading PDFs...")
    pages = load_all_documents()

    print("Step 2: Converting to documents...")
    docs = pages_to_langchain_docs(pages)

    print("Step 3: Chunking...")
    chunks = chunk_documents(docs)

    print(f"Step 4: Building FAISS index ({len(chunks)} chunks)...")
    build_index(chunks)
    print("\nDone! Index saved to data/faiss_index/")