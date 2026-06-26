import streamlit as st

from src.config import ACTIVE_NOTES_DIR
from src.upload_ui import render_pdf_upload_panel


st.set_page_config(page_title="Study Assistant", page_icon="📚", layout="wide")

st.title("📚 Personal Study Assistant")
st.write("Upload your own PDF notes, then ask questions or take a LangGraph-powered quiz.")

ready = render_pdf_upload_panel()

st.divider()

if ready:
    st.success("Your PDF index is ready. Use Ask or Quiz Me from the sidebar.")
else:
    st.info(
        f"Upload PDFs into `{ACTIVE_NOTES_DIR}` with the sidebar, then build the index. "
        "Ask and Quiz Me will use only that uploaded PDF index."
    )

st.markdown("""
### How to use

Use the pages in the left sidebar:

1. **Ask** - type a question about your uploaded PDFs and get an answer with source citations.
2. **Quiz Me** - pick a topic, the LangGraph quiz agent generates questions and grades your answers.

---

**Tech under the hood:**
- **RAG (Retrieval-Augmented Generation)** for the Ask page - LangChain pipeline
- **LangGraph** stateful agent for Quiz mode - handles questions, hints, retries, weak-topic tracking
- **Groq** LLM
- **HuggingFace** local embeddings (BAAI/bge-small-en-v1.5)
- **Chroma** vector store (local, file-backed)
""")
