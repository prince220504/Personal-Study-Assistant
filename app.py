import streamlit as st
from src.config import NOTES_DIR, CHROMA_DIR
from src.ingest import build_vectorstore

st.set_page_config(page_title="Study Assistant", page_icon="📚", layout="wide")

st.title("📚 Personal Study Assistant")
st.write(
    "RAG-powered Q&A over your own notes, plus a quiz mode that tests "
    "what you've learned."
)

with st.sidebar:
    st.header("Index")
    st.write(f"**Notes folder:** `{NOTES_DIR.name}/`")
    st.write(f"**Index exists:** {'✅' if CHROMA_DIR.exists() else '❌'}")

    st.divider()
    st.subheader("Rebuild index")
    st.caption("Run this only after adding/changing files in data/notes/")
    if st.button("🔄 Rebuild now", type="secondary"):
        with st.spinner("Re-embedding all documents..."):
            vs = build_vectorstore(force_rebuild=True)
        st.success(f"Indexed {vs._collection.count()} chunks.")

st.divider()

st.markdown("""
### How to use

Use the pages in the left sidebar:

1. **Ask** — type a question about your notes, get an answer with source citations.
2. **Quiz Me** — pick a topic, the assistant generates questions and grades your answers.

---

**Tech under the hood:**
- 🔍 **RAG (Retrieval-Augmented Generation)** for the Ask page — LangChain pipeline
- 🧠 **LangGraph** stateful agent for Quiz mode — handles questions, hints, retries, weak-topic tracking
- ⚡ **Groq** LLM (llama-3.1-8b-instant)
- 📐 **HuggingFace** local embeddings (BAAI/bge-small-en-v1.5)
- 🗂️ **Chroma** vector store (local, file-backed)
""")
