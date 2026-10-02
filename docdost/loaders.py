"""Load documents (Markdown, text, PDF) and split them into chunks for retrieval."""

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from docdost.config import CHUNK_OVERLAP, CHUNK_SIZE
from docdost.embeddings import count_tokens

SUPPORTED = {".md", ".txt", ".pdf"}

_IMAGE_MD = re.compile(r"!\[[^\]]*\]\([^)]*\)")  # ![alt](path)
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")     # <img ...>, <p>, <br/>
_BLANK_LINES = re.compile(r"\n{3,}")
_SOFT_BREAK = re.compile(r"(?<!\n)\n(?!\n)")      # a single line break inside a paragraph


def clean_text(text: str) -> str:
    """Remove images and HTML tags: they carry no meaning for an embedding model."""
    text = _IMAGE_MD.sub("", text)
    text = _HTML_TAG.sub("", text)
    return _BLANK_LINES.sub("\n\n", text).strip()


def load_file(path: Path) -> list[Document]:
    """Read one file into Documents. PDFs give one Document per page."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [
            # PDF text breaks lines wherever the page did; join them back into paragraphs
            Document(page_content=clean_text(_SOFT_BREAK.sub(" ", page.extract_text() or "")),
                     metadata={"source": path.name, "page": i + 1})
            for i, page in enumerate(reader.pages)
            if (page.extract_text() or "").strip()
        ]
    if suffix in {".md", ".txt"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [Document(page_content=text, metadata={"source": path.name})]
    raise ValueError(f"Unsupported file type: {path.name} (use {', '.join(sorted(SUPPORTED))})")


def load_folder(folder: Path) -> list[Document]:
    """Load every supported file in a folder (recursively), sorted for reproducibility."""
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in SUPPORTED)
    docs = []
    for f in files:
        docs.extend(load_file(f))
    return docs


def split_documents(docs: list[Document],
                    chunk_size: int = CHUNK_SIZE,
                    chunk_overlap: int = CHUNK_OVERLAP,
                    add_context: bool = True) -> list[Document]:
    """Split documents into overlapping chunks of at most `chunk_size` tokens.

    Markdown is first split on its headings, so a chunk never mixes two sections and
    each chunk remembers which section it came from (shown as the citation). Long
    sections, text files and PDF pages are then split by token count, using the
    embedding model's own tokenizer.
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in docs:
        if doc.metadata["source"].endswith(".md"):
            sections = header_splitter.split_text(doc.page_content)
            for s in sections:
                s.metadata = {**doc.metadata, **s.metadata}
        else:
            sections = [doc]
        for s in sections:
            s.page_content = clean_text(s.page_content)
        sections = [s for s in sections if len(s.page_content) > 40]
        chunks.extend(size_splitter.split_documents(sections))

    for i, c in enumerate(chunks):
        c.metadata["section"] = c.metadata.get("h3") or c.metadata.get("h2") or c.metadata.get("h1", "")
        c.metadata["chunk_id"] = i
        if add_context:
            c.page_content = f"{context_header(c.metadata)}\n{c.page_content}"
    return chunks


def context_header(metadata: dict) -> str:
    """'kharcha_ai.md > Kharcha AI > Results at a glance': where a chunk comes from.

    A chunk like "| ML: note only | 82.8% |" means little on its own. Adding the file
    name and heading path lets both the embedding and BM25 match questions that name
    the project or section. (Often called "contextual chunk headers".)
    """
    parts = [metadata["source"]] + [metadata[h] for h in ("h1", "h2", "h3") if metadata.get(h)]
    return " > ".join(dict.fromkeys(parts))
