import streamlit as st
from src.rag_chain import build_rag_chain


@st.cache_resource
def get_chain():
    """Cache chain + retriever across Streamlit reruns."""
    return build_rag_chain()


st.title("📖 Ask")

st.caption("Ask any question about your study notes. The assistant answers from your documents.")

chain, retriever = get_chain()

question = st.text_input(
    "Your question:",
    placeholder="e.g. What is a transformer attention mechanism?",
)

if question:
    with st.spinner("Searching your notes + thinking..."):
        answer = chain.invoke(question)
        sources = retriever.invoke(question)

    st.markdown("### Answer")
    st.write(answer)

    st.divider()
    st.markdown(f"### Sources ({len(sources)} chunks)")
    for i, doc in enumerate(sources, 1):
        filename = doc.metadata.get("source", "?")
        page = doc.metadata.get("page", "?")
        with st.expander(f"Source {i}: {filename} (page {page})"):
            st.text(doc.page_content[:500])
