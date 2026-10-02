"""Settings in one place, so every experiment is easy to repeat."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"
INDEX_DIR = ROOT / "data" / "chroma_index"
COLLECTION = "docdost"

# Chunk size is measured in MiniLM tokens, not characters. The embedding model cuts
# input off at 256 tokens, so a chunk (plus its ~20-token context header) must stay
# under that or its end is never embedded. See notebooks/01_tokens_embeddings_attention.ipynb.
MAX_MODEL_TOKENS = 256
CHUNK_SIZE = 200      # tokens
CHUNK_OVERLAP = 30    # tokens

TOP_K = 4          # chunks handed to the LLM
FETCH_K = 20       # candidates each retriever returns before fusion
RRF_K = 60         # reciprocal rank fusion constant (from the original RRF paper)

# LLM providers that work on a free tier. The first one with an API key set is used.
LLM_PROVIDERS = {
    "gemini": {"env": "GOOGLE_API_KEY", "model": "gemini-2.5-flash"},
    "groq": {"env": "GROQ_API_KEY", "model": "llama-3.1-8b-instant"},
    "ollama": {"env": None, "model": "llama3.2"},  # local, no key; needs `ollama serve`
}
