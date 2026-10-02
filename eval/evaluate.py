"""Measure retrieval quality on a hand-written test set.

For each question, eval/questions.csv records the file and a short "evidence" phrase
that a correct answer needs. A retrieval is a hit if one of the top-k chunks contains
that phrase. Without the right chunk, even a perfect LLM can't answer, so retrieval is
the part of RAG worth measuring first.

    python eval/evaluate.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docdost.config import DOCS_DIR, ROOT  # noqa: E402
from docdost.loaders import load_folder, split_documents  # noqa: E402
from docdost.retriever import HybridRetriever  # noqa: E402

MODES = ["bm25", "dense", "hybrid"]
KS = [1, 4]


def norm(text: str) -> str:
    return " ".join(text.replace("*", "").lower().split())


def first_hit_rank(chunks, evidence: str, source: str) -> int | None:
    """1-based rank of the first retrieved chunk holding the evidence, else None."""
    for rank, c in enumerate(chunks, start=1):
        if c.metadata["source"] == source and norm(evidence) in norm(c.page_content):
            return rank
    return None


def main() -> None:
    questions = pd.read_csv(ROOT / "eval" / "questions.csv")
    chunks = split_documents(load_folder(DOCS_DIR))

    # Guard against a broken test set: every evidence phrase must exist in some chunk.
    missing = [e for e in questions.evidence
               if not any(norm(e) in norm(c.page_content) for c in chunks)]
    if missing:
        sys.exit(f"Evidence not found in any chunk: {missing}")

    retriever = HybridRetriever.build(chunks)  # in-memory index
    max_k = max(KS)
    rows = []
    for q in questions.itertuples():
        for mode in MODES:
            rank = first_hit_rank(retriever.retrieve(q.question, k=max_k, mode=mode), q.evidence, q.source)
            rows.append({"question": q.question, "style": q.style, "mode": mode, "rank": rank})
    res = pd.DataFrame(rows)
    for k in KS:
        res[f"hit@{k}"] = res["rank"].apply(lambda r: pd.notna(r) and r <= k)
    res["rr"] = res["rank"].apply(lambda r: 1 / r if pd.notna(r) else 0.0)  # reciprocal rank

    summary = res.groupby("mode")[[f"hit@{k}" for k in KS] + ["rr"]].mean().loc[MODES]
    by_style = res.pivot_table(index="mode", columns="style", values=f"hit@{max_k}").loc[MODES]

    print(f"{len(questions)} questions, {len(chunks)} chunks\n")
    print((summary * 100).round(1).rename(columns={"rr": "MRR"}).to_string())
    print(f"\nhit@{max_k} by question style (%):")
    print((by_style * 100).round(1).to_string())

    misses = res[(res["mode"] == "hybrid") & (~res[f"hit@{max_k}"])]
    if len(misses):
        print("\nHybrid misses:")
        for m in misses.question:
            print(" -", m)

    out = ROOT / "reports" / "retrieval_metrics.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "n_questions": len(questions),
        "n_chunks": len(chunks),
        "overall": (summary.round(3)).to_dict(orient="index"),
        f"hit@{max_k}_by_style": (by_style.round(3)).to_dict(orient="index"),
    }, indent=2))
    print(f"\nSaved {out.relative_to(ROOT)}")
    plot(summary, ROOT / "reports" / "figures" / "retrieval_comparison.png")


def plot(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"bm25": "BM25 (keywords)", "dense": "Dense (MiniLM + ChromaDB)", "hybrid": "Hybrid (RRF)"}
    colors = {"bm25": "#B0B7C3", "dense": "#7EA6E0", "hybrid": "#2F5DA8"}
    metrics = [c for c in summary.columns if c.startswith("hit@")]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    width = 0.26
    for i, mode in enumerate(summary.index):
        vals = summary.loc[mode, metrics] * 100
        bars = ax.bar([x + i * width for x in range(len(metrics))], vals, width,
                      label=labels[mode], color=colors[mode])
        ax.bar_label(bars, fmt="%.0f%%", fontsize=8, padding=2)
    ax.set_xticks([x + width for x in range(len(metrics))], [f"{m} (right chunk in top {m[4:]})" for m in metrics])
    ax.set_ylim(0, 110)
    ax.set_ylabel("% of 40 questions")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Hybrid search gets the right chunk into the top 4 most often", fontsize=11, loc="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
