"""Retrieval-Augmented Generation: retrieve chunks, then let an LLM answer from them only.

If no LLM is configured, DocDost still works in "extractive" mode: it returns the
sentences from the retrieved chunks that are most similar to the question.
"""

import os
import re
from dataclasses import dataclass, field

import numpy as np
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from docdost.config import LLM_PROVIDERS, TOP_K

SYSTEM_PROMPT = """You are DocDost, a helpful assistant that answers questions about the user's documents.

Rules:
- Answer ONLY from the numbered context below. Do not use outside knowledge.
- Cite the context you used with its number in square brackets, like [1] or [2][3].
- If the answer is not in the context, say "I couldn't find that in the documents." and stop.
- Be concise: 1-4 sentences, or a short list. Keep numbers exactly as written.

Context:
{context}"""

CONDENSE_PROMPT = """Given the chat so far and a follow-up question, rewrite the follow-up as a \
standalone question that makes sense without the chat. Return only the question.

Chat so far:
{history}

Follow-up question: {question}
Standalone question:"""

NOT_FOUND = "I couldn't find that in the documents."


@dataclass
class Answer:
    text: str
    sources: list[Document] = field(default_factory=list)
    mode: str = "llm"            # "llm" or "extractive"
    question_used: str = ""      # the standalone question that was searched


def get_llm(provider: str | None = None, temperature: float = 0.0):
    """Return a LangChain chat model, or None if nothing is configured.

    Picks `provider` if given, otherwise the first provider whose API key is set.
    """
    candidates = [provider] if provider else [p for p, c in LLM_PROVIDERS.items()
                                              if c["env"] and os.getenv(c["env"])]
    for name in candidates:
        cfg = LLM_PROVIDERS[name]
        model = os.getenv("DOCDOST_MODEL", cfg["model"])
        if name == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model, temperature=temperature)
        if name == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(model=model, temperature=temperature)
        if name == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(model=model, temperature=temperature)
    return None


def format_context(chunks: list[Document]) -> str:
    """Number each chunk and label where it came from, so the LLM can cite it."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        where = c.metadata["source"]
        if c.metadata.get("page"):
            where += f", page {c.metadata['page']}"
        if c.metadata.get("section"):
            where += f" > {c.metadata['section']}"
        parts.append(f"[{i}] ({where})\n{c.page_content}")
    return "\n\n".join(parts)


def condense_question(question: str, history: list[tuple[str, str]], llm) -> str:
    """Turn 'and how much did it save?' into a question the retriever can search for."""
    if not history or llm is None:
        return question
    chat = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history[-3:])
    chain = ChatPromptTemplate.from_template(CONDENSE_PROMPT) | llm | StrOutputParser()
    return chain.invoke({"history": chat, "question": question}).strip() or question


_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")


def _plain(text: str) -> str:
    """Strip Markdown so a sentence reads cleanly: links -> their text, no ** or `."""
    return _MD_LINK.sub(r"\1", text).replace("**", "").replace("`", "")


def extractive_answer(question: str, chunks: list[Document], embeddings,
                      n_sentences: int = 3, min_similarity: float = 0.35) -> str:
    """No-LLM fallback: pick the sentences closest in meaning to the question.

    If even the best sentence is not similar enough, say so instead of guessing.
    """
    sentences, seen = [], set()
    for i, c in enumerate(chunks, start=1):
        for s in _SENTENCE.split(c.page_content):
            if s.lstrip().startswith(("#", "```")) or any(ch in s for ch in "├└│"):
                continue  # headings repeat the question's words but don't answer it; trees are code
            s = _plain(s).strip(" *|-")
            if len(s.split()) >= 5 and not s.endswith("?") and s not in seen:
                seen.add(s)
                sentences.append((s, i))
    if not sentences:
        return NOT_FOUND
    q = np.asarray(embeddings.embed_query(question))
    S = np.asarray(embeddings.embed_documents([s for s, _ in sentences]))
    sims = S @ q / (np.linalg.norm(S, axis=1) * np.linalg.norm(q))
    if sims.max() < min_similarity:
        return NOT_FOUND
    best = sorted(np.argsort(-sims)[:n_sentences])  # keep document order so it reads naturally
    return " ".join(f"{sentences[j][0]} [{sentences[j][1]}]" for j in best)


def answer_question(question: str, retriever, llm=None, history=None, k: int = TOP_K) -> Answer:
    """The full RAG pipeline: (condense) -> retrieve -> generate with citations."""
    standalone = condense_question(question, history or [], llm)
    chunks = retriever.retrieve(standalone, k=k)
    if not chunks:
        return Answer(NOT_FOUND, [], "llm" if llm else "extractive", standalone)

    if llm is None:
        text = extractive_answer(standalone, chunks, retriever.vectorstore.embeddings)
        return Answer(text, chunks, "extractive", standalone)

    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])
    chain = prompt | llm | StrOutputParser()
    text = chain.invoke({"context": format_context(chunks), "question": standalone})
    return Answer(text.strip(), chunks, "llm", standalone)
