from pathlib import Path

import streamlit as st

from src.rag_chain import build_rag_chain
from src.upload_ui import render_pdf_upload_panel


@st.cache_resource
def get_chain():
    """Cache chain + retriever across Streamlit reruns."""
    return build_rag_chain()


st.title("📖 Ask")
st.caption("Ask questions about the PDFs you uploaded for this session.")

ready = render_pdf_upload_panel()

if not ready:
    st.warning("Upload one or more PDFs and build the index before asking questions.")
    st.stop()

question = st.text_input(
    "Your question:",
    placeholder="e.g. What is the main idea in this PDF?",
)

if question:
    with st.spinner("Searching your uploaded PDFs and thinking..."):
        chain, retriever = get_chain()
        answer = chain.invoke(question)
        sources = retriever.invoke(question)

    st.markdown("### Answer")
    st.write(answer)

    st.divider()
    st.markdown(f"### Sources ({len(sources)} chunks)")
    for i, doc in enumerate(sources, 1):
        raw_source = doc.metadata.get("source", "?")
        filename = Path(raw_source).name if raw_source != "?" else "?"
        page = doc.metadata.get("page", "?")
        with st.expander(f"Source {i}: {filename} (page {page})"):
            st.text(doc.page_content[:500])
