"""DocDost Streamlit app: upload documents and chat with them.

    streamlit run app/app.py
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docdost.config import DOCS_DIR  # noqa: E402
from docdost.embeddings import MiniLMEmbeddings  # noqa: E402
from docdost.loaders import load_file, load_folder, split_documents  # noqa: E402
from docdost.rag import answer_question, get_llm  # noqa: E402
from docdost.retriever import HybridRetriever  # noqa: E402

st.set_page_config(page_title="DocDost", page_icon="📚", layout="wide")


@st.cache_resource
def embeddings():
    return MiniLMEmbeddings()


@st.cache_resource
def sample_retriever():
    """Index of the bundled sample documents (Aryan's project write-ups)."""
    return HybridRetriever.build(split_documents(load_folder(DOCS_DIR)), embeddings=embeddings())


def build_from_uploads(files) -> tuple[HybridRetriever, int]:
    docs = []
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            path = Path(tmp) / f.name
            path.write_bytes(f.getvalue())
            docs.extend(load_file(path))
    chunks = split_documents(docs)
    if not chunks:
        raise ValueError("No readable text found (scanned PDFs need OCR first).")
    return HybridRetriever.build(chunks, embeddings=embeddings()), len(chunks)


# ---------------- sidebar ----------------
with st.sidebar:
    st.title("📚 DocDost")
    st.caption("Chat with your documents. Answers come only from your files, with citations.")

    st.subheader("1. Documents")
    uploads = st.file_uploader("Upload PDF, Markdown or text files", type=["pdf", "md", "txt"],
                               accept_multiple_files=True)
    if uploads:
        key = tuple((f.name, f.size) for f in uploads)
        if st.session_state.get("upload_key") != key:
            with st.spinner("Reading and indexing..."):
                try:
                    st.session_state.retriever, n = build_from_uploads(uploads)
                    st.session_state.upload_key = key
                    st.session_state.messages = []
                    st.success(f"Indexed {len(uploads)} file(s) into {n} chunks.")
                except ValueError as e:
                    st.error(str(e))
    else:
        st.info("No files uploaded, so you're chatting with the sample documents: "
                "write-ups of my projects Plate to Bin, Kharcha AI and Spam SMS Classifier.")
        if st.session_state.get("upload_key"):
            st.session_state.messages = []
        st.session_state.upload_key = None
        st.session_state.retriever = sample_retriever()

    st.subheader("2. Language model")
    provider = st.selectbox("Provider", ["gemini", "groq", "none (extractive)"],
                            help="Gemini and Groq both have free API keys.")
    if provider != "none (extractive)":
        env = {"gemini": "GOOGLE_API_KEY", "groq": "GROQ_API_KEY"}[provider]
        try:  # a key saved in .streamlit/secrets.toml (or Streamlit Cloud secrets)
            secret = st.secrets.get(env)
        except Exception:  # no secrets file at all
            secret = None
        key_in = st.text_input(f"{env}", type="password", value="",
                               help="Used only in this session, never stored.") or secret
        if key_in:
            os.environ[env] = key_in
    show_chunks = st.toggle("Show retrieved chunks", value=True)
    if st.button("Clear chat"):
        st.session_state.messages = []


def current_llm():
    if provider == "none (extractive)":
        return None
    env = {"gemini": "GOOGLE_API_KEY", "groq": "GROQ_API_KEY"}[provider]
    return get_llm(provider) if os.getenv(env) else None


# ---------------- chat ----------------
st.header("Ask your documents anything")
st.session_state.setdefault("messages", [])

if not st.session_state.messages:
    st.markdown("**Try:** *How much food was wasted in the hostel mess?* · "
                "*Why wasn't 96% accuracy good enough?* · *How does Kharcha AI handle empty notes?*")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    llm = current_llm()
    history = [(a["content"], b["content"]) for a, b in
               zip(st.session_state.messages[:-1:2], st.session_state.messages[1:-1:2])]
    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            try:
                ans = answer_question(question, st.session_state.retriever, llm, history)
            except Exception as e:  # bad API key, rate limit, network
                st.error(f"The language model call failed ({type(e).__name__}). "
                         "Check your API key, or switch to extractive mode.")
                st.stop()
        st.markdown(ans.text)
        if ans.mode == "extractive":
            st.caption("Extractive mode: these are the closest sentences from your documents. "
                       "Add a free Gemini or Groq key in the sidebar for written answers.")
        if ans.question_used != question:
            st.caption(f"Searched for: *{ans.question_used}*")
        if show_chunks and ans.sources:
            with st.expander(f"Sources ({len(ans.sources)} chunks)"):
                for i, c in enumerate(ans.sources, start=1):
                    st.markdown(f"**[{i}] {c.metadata['source']}**"
                                + (f", page {c.metadata['page']}" if c.metadata.get("page") else ""))
                    st.text(c.page_content[:600])
    st.session_state.messages.append({"role": "assistant", "content": ans.text})
