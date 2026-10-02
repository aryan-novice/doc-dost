"""Build the vector index and retrieve chunks with hybrid search.

Three retrievers, so they can be compared in eval/evaluate.py:
  - "bm25":   keyword search (the classic baseline)
  - "dense":  embedding similarity search in ChromaDB
  - "hybrid": both, merged with Reciprocal Rank Fusion (RRF)
"""

import re
import shutil
import uuid
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from docdost.config import COLLECTION, FETCH_K, RRF_K, TOP_K
from docdost.embeddings import MiniLMEmbeddings

_TOKEN = re.compile(r"[a-z0-9₹]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens for BM25 (keeps numbers like 23.5 -> '23', '5')."""
    return _TOKEN.findall(text.lower())


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """Merge several ranked lists of ids. Each id scores sum(1 / (k + rank)).

    RRF only uses ranks, not raw scores, so BM25 scores and cosine similarities
    don't need to be on the same scale.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


class HybridRetriever:
    def __init__(self, chunks: list[Document], vectorstore: Chroma):
        self.chunks = chunks
        self.vectorstore = vectorstore
        self.bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks])

    # ---------- building / loading ----------

    @classmethod
    def build(cls, chunks: list[Document], persist_dir: Path | None = None,
              embeddings=None) -> "HybridRetriever":
        """Embed every chunk and store it in ChromaDB (on disk if persist_dir is given)."""
        if persist_dir is not None and Path(persist_dir).exists():
            shutil.rmtree(persist_dir)  # rebuild from scratch, never mix old and new chunks
        # In-memory Chroma collections are shared inside one process, so each in-memory
        # index gets its own name (otherwise uploaded files would mix with the samples).
        name = COLLECTION if persist_dir else f"{COLLECTION}-{uuid.uuid4().hex[:8]}"
        vs = Chroma.from_documents(
            chunks,
            embedding=embeddings or MiniLMEmbeddings(),
            collection_name=name,
            persist_directory=str(persist_dir) if persist_dir else None,
            ids=[str(c.metadata["chunk_id"]) for c in chunks],
            collection_metadata={"hnsw:space": "cosine"},
        )
        return cls(chunks, vs)

    @classmethod
    def load(cls, persist_dir: Path, embeddings=None) -> "HybridRetriever":
        """Re-open a saved index; BM25 is rebuilt from the chunks stored in Chroma."""
        vs = Chroma(collection_name=COLLECTION, embedding_function=embeddings or MiniLMEmbeddings(),
                    persist_directory=str(persist_dir))
        data = vs.get(include=["documents", "metadatas"])
        chunks = [Document(page_content=t, metadata=m) for t, m in zip(data["documents"], data["metadatas"])]
        chunks.sort(key=lambda c: c.metadata["chunk_id"])
        if not chunks:
            raise FileNotFoundError(f"No index found in {persist_dir}. Run: python -m docdost.cli ingest")
        return cls(chunks, vs)

    # ---------- search ----------

    def _bm25_ids(self, query: str, n: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in order[:n] if scores[i] > 0]

    def _dense_ids(self, query: str, n: int) -> list[int]:
        hits = self.vectorstore.similarity_search(query, k=min(n, len(self.chunks)))
        return [h.metadata["chunk_id"] for h in hits]

    def retrieve(self, query: str, k: int = TOP_K, mode: str = "hybrid") -> list[Document]:
        if mode == "bm25":
            ids = self._bm25_ids(query, k)
        elif mode == "dense":
            ids = self._dense_ids(query, k)
        elif mode == "hybrid":
            ids = reciprocal_rank_fusion([self._dense_ids(query, FETCH_K), self._bm25_ids(query, FETCH_K)])
        else:
            raise ValueError(f"mode must be bm25, dense or hybrid, not {mode!r}")
        by_id = {c.metadata["chunk_id"]: c for c in self.chunks}
        return [by_id[i] for i in ids[:k]]
