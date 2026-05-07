import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentPage:
    content: str
    page_num: int
    source: str
    chapter: str = ""


def load_pdf_pymupdf(pdf_path: str) -> List[DocumentPage]:
    """Primary loader — handles text-based PDFs."""
    pages = []
    doc = fitz.open(pdf_path)
    src = Path(pdf_path).stem
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if len(text) > 50:
            pages.append(DocumentPage(content=text, page_num=i + 1, source=src))
    logger.info(f"Loaded {len(pages)} pages from {src}")
    return pages


def load_pdf_pdfplumber(pdf_path: str) -> List[DocumentPage]:
    """Fallback — better for tables and complex layouts."""
    pages = []
    src = Path(pdf_path).stem
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if len(text.strip()) > 50:
                pages.append(DocumentPage(content=text.strip(), page_num=i + 1, source=src))
    logger.info(f"Loaded {len(pages)} pages from {src} (pdfplumber)")
    return pages


def load_all_documents(data_dir: str = "data/raw") -> List[DocumentPage]:
    all_pages = []
    pdf_files = list(Path(data_dir).glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {data_dir}")
        return []
    for pdf in pdf_files:
        try:
            pages = load_pdf_pymupdf(str(pdf))
            all_pages.extend(pages)
        except Exception as e:
            logger.warning(f"PyMuPDF failed for {pdf}, trying pdfplumber: {e}")
            all_pages.extend(load_pdf_pdfplumber(str(pdf)))
    logger.info(f"Total pages loaded: {len(all_pages)}")
    return all_pages


if __name__ == "__main__":
    pages = load_all_documents()
    print(f"Sample page:\n{pages[0].content[:300]}")