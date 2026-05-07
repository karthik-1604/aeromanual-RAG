from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import logging

logger = logging.getLogger(__name__)


def pages_to_langchain_docs(pages) -> List[Document]:
    """Convert DocumentPage objects to LangChain Document objects."""
    return [
        Document(
            page_content=p.content,
            metadata={
                "source": p.source,
                "page": p.page_num,
                "chapter": p.chapter
            }
        )
        for p in pages
    ]


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64
) -> List[Document]:
    """
    Split documents into chunks.
    512 tokens with 64 overlap is a good balance for technical manuals.
    Overlap ensures procedures split across chunks stay coherent.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
        length_function=len
    )
    chunks = splitter.split_documents(docs)

    # Preserve metadata and add chunk_id
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    logger.info(f"Created {len(chunks)} chunks from {len(docs)} pages")
    return chunks


if __name__ == "__main__":
    from src.ingestion.pdf_loader import load_all_documents
    pages = load_all_documents()
    docs = pages_to_langchain_docs(pages)
    chunks = chunk_documents(docs)
    print(f"Total chunks: {len(chunks)}")
    print(f"Sample chunk:\n{chunks[0].page_content[:200]}")
    print(f"Metadata: {chunks[0].metadata}")