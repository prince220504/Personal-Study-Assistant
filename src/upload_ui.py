from pathlib import Path

import streamlit as st

from .config import ACTIVE_NOTES_DIR, ACTIVE_CHROMA_DIR
from .ingest import build_vectorstore, vectorstore_has_documents


def list_uploaded_pdfs():
    """Return uploaded PDF paths sorted by filename."""
    if not ACTIVE_NOTES_DIR.exists():
        return []
    return sorted(ACTIVE_NOTES_DIR.glob("*.pdf"))


def save_uploaded_pdfs(uploaded_files):
    """Save Streamlit UploadedFile objects into the active user docs folder."""
    ACTIVE_NOTES_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for uploaded_file in uploaded_files:
        filename = Path(uploaded_file.name).name
        if not filename.lower().endswith(".pdf"):
            continue
        destination = ACTIVE_NOTES_DIR / filename
        destination.write_bytes(uploaded_file.getbuffer())
        saved.append(destination)
    return saved


def clear_runtime_caches():
    """
    Clear cached RAG/quiz resources after rebuilding the index.

    Streamlit keeps cached chains across reruns. Clearing here prevents Ask
    or Quiz from using a retriever pointed at the previous upload index.
    """
    st.cache_resource.clear()
    try:
        from .quiz_graph import reset_quiz_resources

        reset_quiz_resources()
    except Exception:
        pass


def uploaded_index_ready():
    """Return True if uploaded PDFs exist and the user Chroma index has chunks."""
    pdfs = list_uploaded_pdfs()
    if not pdfs or index_needs_rebuild(pdfs):
        return False
    return vectorstore_has_documents()


def index_needs_rebuild(pdfs=None):
    """Return True when uploaded PDFs are newer than the persisted index."""
    pdfs = list_uploaded_pdfs() if pdfs is None else pdfs
    db_file = ACTIVE_CHROMA_DIR / "chroma.sqlite3"
    if not pdfs or not db_file.exists():
        return bool(pdfs)
    newest_pdf = max(pdf.stat().st_mtime for pdf in pdfs)
    return newest_pdf > db_file.stat().st_mtime


def render_pdf_upload_panel(location="sidebar"):
    """
    Render PDF upload + index controls.

    Uploading PDFs only saves files. Rebuilding creates embeddings and stores
    them in the active user Chroma DB, which Ask and Quiz both read from.
    """
    container = st.sidebar if location == "sidebar" else st
    pdfs = list_uploaded_pdfs()

    with container:
        st.header("Your PDFs")
        if pdfs:
            st.caption(f"{len(pdfs)} uploaded file(s)")
            for pdf in pdfs:
                st.write(f"- {pdf.name}")
        else:
            st.caption("Upload at least one PDF to ask questions or take quizzes.")

        uploaded_files = st.file_uploader(
            "Upload PDF notes",
            type=["pdf"],
            accept_multiple_files=True,
            help="These PDFs become the active knowledge source for Ask and Quiz.",
        )

        if uploaded_files:
            if st.button("Save uploaded PDFs", type="secondary"):
                saved = save_uploaded_pdfs(uploaded_files)
                if saved:
                    st.success(f"Saved {len(saved)} PDF(s). Rebuild the index next.")
                else:
                    st.warning("No PDF files were saved.")
                st.rerun()

        rebuild_disabled = not pdfs
        if st.button("Build index from my PDFs", type="primary", disabled=rebuild_disabled):
            clear_runtime_caches()
            with st.spinner("Reading PDFs, chunking text, and building embeddings..."):
                vs = build_vectorstore(force_rebuild=True)
            clear_runtime_caches()
            st.success(f"Indexed {vs._collection.count()} chunks from your PDFs.")
            st.rerun()

        stale = index_needs_rebuild(pdfs)
        ready = uploaded_index_ready()
        st.write(f"**Index ready:** {'yes' if ready else 'no'}")
        if stale:
            st.caption("Index stale: rebuild after latest upload.")
        if ACTIVE_CHROMA_DIR.exists():
            st.caption(f"Index folder: `{ACTIVE_CHROMA_DIR.name}/`")

    return ready
