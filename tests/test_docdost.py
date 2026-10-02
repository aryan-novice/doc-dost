"""Fast tests for chunking, retrieval and the RAG chain (no API key needed)."""

import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docdost.config import DOCS_DIR, MAX_MODEL_TOKENS  # noqa: E402
from docdost.embeddings import count_tokens  # noqa: E402
from docdost.loaders import clean_text, load_folder, split_documents  # noqa: E402
from docdost.rag import NOT_FOUND, answer_question, format_context  # noqa: E402
from docdost.retriever import HybridRetriever, reciprocal_rank_fusion  # noqa: E402


@pytest.fixture(scope="module")
def retriever():
    return HybridRetriever.build(split_documents(load_folder(DOCS_DIR)))


def test_clean_text_removes_images_and_html_but_keeps_comparisons():
    text = "Intro ![chart](a.png)\n<img src='b.png'>\nconfidence < 60% goes to review"
    out = clean_text(text)
    assert "png" not in out and "<img" not in out
    assert "confidence < 60% goes to review" in out


def test_chunks_fit_in_the_embedding_model():
    chunks = split_documents(load_folder(DOCS_DIR))
    assert len(chunks) > 10
    for c in chunks:
        assert c.metadata["source"].endswith(".md")
        assert "chunk_id" in c.metadata
        # header + body must fit, or the end of the chunk would be silently cut off
        assert count_tokens(c.page_content) <= MAX_MODEL_TOKENS


def test_reciprocal_rank_fusion_rewards_agreement():
    # id 2 is 2nd in both lists, so it beats ids that are 1st in only one list
    assert reciprocal_rank_fusion([[1, 2, 3], [4, 2, 5]])[0] == 2


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_retrieval_finds_the_right_document(retriever, mode):
    hits = retriever.retrieve("difference-in-differences weekend breakfast pilot", k=3, mode=mode)
    assert hits[0].metadata["source"] == "plate_to_bin.md"


def test_dense_retrieval_understands_paraphrase(retriever):
    # no shared keywords with "never seen modern Indian scams like KYC updates"
    hits = retriever.retrieve("Can the filter catch fake bank account update messages?", k=4, mode="dense")
    assert any(h.metadata["source"] == "spam_sms_classifier.md" for h in hits)


def test_in_memory_indexes_do_not_mix(retriever):
    other = HybridRetriever.build([Document(page_content="Mangoes are yellow fruits from India.",
                                            metadata={"source": "fruit.txt", "chunk_id": 0})])
    assert other.vectorstore._collection.count() == 1
    assert retriever.vectorstore._collection.count() == len(retriever.chunks)


def test_format_context_numbers_and_labels_chunks():
    docs = [Document(page_content="alpha", metadata={"source": "a.pdf", "page": 2, "section": "Intro"})]
    assert format_context(docs) == "[1] (a.pdf, page 2 > Intro)\nalpha"


def test_rag_chain_passes_context_to_llm(retriever):
    llm = FakeListChatModel(responses=["About 23.5% of food was wasted [1]."])
    ans = answer_question("How much food was wasted?", retriever, llm)
    assert ans.mode == "llm"
    assert ans.text.endswith("[1].")
    assert len(ans.sources) == 4


def test_follow_up_question_is_condensed(retriever):
    llm = FakeListChatModel(responses=["How much did the Plate to Bin pilot save?", "~140 kg [1]"])
    ans = answer_question("and how much did it save?", retriever, llm,
                          history=[("What was the pilot?", "A weekend opt-in [1].")])
    assert ans.question_used == "How much did the Plate to Bin pilot save?"


def test_extractive_mode_refuses_off_topic_questions(retriever):
    ans = answer_question("What is the capital of France?", retriever, llm=None)
    assert ans.mode == "extractive"
    assert ans.text == NOT_FOUND
