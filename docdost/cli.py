"""Command line interface.

    python -m docdost.cli ingest [folder]      # build the index (default: data/docs)
    python -m docdost.cli ask "your question"  # one question
    python -m docdost.cli chat                 # interactive chat with memory
"""

import argparse
import sys
from pathlib import Path

from docdost.config import DOCS_DIR, INDEX_DIR
from docdost.loaders import load_folder, split_documents
from docdost.rag import answer_question, get_llm
from docdost.retriever import HybridRetriever


def ingest(folder: Path) -> None:
    docs = load_folder(folder)
    if not docs:
        sys.exit(f"No .md, .txt or .pdf files found in {folder}")
    chunks = split_documents(docs)
    HybridRetriever.build(chunks, persist_dir=INDEX_DIR)
    print(f"Indexed {len(docs)} documents -> {len(chunks)} chunks in {INDEX_DIR}")


def print_answer(ans) -> None:
    print(f"\n{ans.text}\n")
    if ans.mode == "extractive":
        print("(extractive mode: set GOOGLE_API_KEY or GROQ_API_KEY for written answers)")
    print("Sources:")
    for i, c in enumerate(ans.sources, start=1):
        print(f"  [{i}] {c.metadata['source']} > {c.metadata.get('section', '')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DocDost: chat with your documents")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("folder", nargs="?", default=str(DOCS_DIR))
    p_ask = sub.add_parser("ask")
    p_ask.add_argument("question")
    sub.add_parser("chat")
    for p in (p_ask,):
        p.add_argument("--provider", choices=["gemini", "groq", "ollama"])
    args = parser.parse_args()

    if args.cmd == "ingest":
        ingest(Path(args.folder))
        return

    retriever = HybridRetriever.load(INDEX_DIR)
    llm = get_llm(getattr(args, "provider", None))
    if args.cmd == "ask":
        print_answer(answer_question(args.question, retriever, llm))
        return

    history = []
    print("Ask about your documents (empty line to quit).")
    while (q := input("\nyou> ").strip()):
        ans = answer_question(q, retriever, llm, history)
        print_answer(ans)
        history.append((q, ans.text))


if __name__ == "__main__":
    main()
