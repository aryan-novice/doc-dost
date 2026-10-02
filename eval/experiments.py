"""Grid search: chunk size x contextual headers x retriever, on the same test set.

    python eval/experiments.py      # writes reports/chunking_experiments.csv
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docdost.config import DOCS_DIR, ROOT  # noqa: E402
from docdost.embeddings import MiniLMEmbeddings  # noqa: E402
from docdost.loaders import load_folder, split_documents  # noqa: E402
from docdost.retriever import HybridRetriever  # noqa: E402
from eval.evaluate import first_hit_rank  # noqa: E402


def main() -> None:
    questions = pd.read_csv(ROOT / "eval" / "questions.csv")
    docs = load_folder(DOCS_DIR)
    embeddings = MiniLMEmbeddings()
    rows = []
    for size in (64, 128, 200):  # tokens
        for headers in (False, True):
            chunks = split_documents(docs, chunk_size=size, chunk_overlap=size // 7, add_context=headers)
            retriever = HybridRetriever.build(chunks, embeddings=embeddings)
            for mode in ("bm25", "dense", "hybrid"):
                ranks = [first_hit_rank(retriever.retrieve(q.question, k=4, mode=mode), q.evidence, q.source)
                         for q in questions.itertuples()]
                rows.append({
                    "chunk_size": size, "context_headers": headers, "retriever": mode, "n_chunks": len(chunks),
                    "hit@1": sum(r == 1 for r in ranks) / len(ranks),
                    "hit@4": sum(r is not None for r in ranks) / len(ranks),
                })
            retriever.vectorstore.delete_collection()

    res = pd.DataFrame(rows)
    out = ROOT / "reports" / "chunking_experiments.csv"
    res.round(3).to_csv(out, index=False)
    print(res.assign(**{c: (res[c] * 100).round(1) for c in ("hit@1", "hit@4")}).to_string(index=False))
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
