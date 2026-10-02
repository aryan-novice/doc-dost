"""Sentence embeddings with all-MiniLM-L6-v2, run through ONNX Runtime.

The same model as sentence-transformers' all-MiniLM-L6-v2 (384 dimensions), but it
runs on CPU without PyTorch, so the app installs fast and fits on a free host.
The weights (~80 MB) download once on first use.
"""

from functools import lru_cache

import numpy as np
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from langchain_core.embeddings import Embeddings


class MiniLMEmbeddings(Embeddings):
    """LangChain wrapper around Chroma's ONNX MiniLM model."""

    def __init__(self):
        self._model = ONNXMiniLM_L6_V2()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [np.asarray(v, dtype=float).tolist() for v in self._model(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def _tokenizer():
    tok = ONNXMiniLM_L6_V2().tokenizer
    tok.no_padding()
    tok.no_truncation()
    return tok


def count_tokens(text: str) -> int:
    """Number of MiniLM WordPiece tokens, including [CLS] and [SEP]."""
    return len(_tokenizer().encode(text).ids)


def cosine_similarity(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
