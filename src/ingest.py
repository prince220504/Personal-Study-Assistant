import shutil
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from .config import (
    NOTES_DIR,
    CHROMA_DIR,
    ACTIVE_NOTES_DIR,
    ACTIVE_CHROMA_DIR,
    EMBED_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def load_documents(notes_dir=ACTIVE_NOTES_DIR):
    """Load PDFs and TXT files from the notes directory."""
    notes_dir = Path(notes_dir)

    # Loader classes mapped to glob patterns
    loaders = [
        (PyPDFLoader, "**/*.pdf"),
        (TextLoader, "**/*.txt"),
    ]

    all_docs = []
    for loader_cls, pattern in loaders:
        files = list(notes_dir.glob(pattern))
        if not files:
            print(f"No files matched: {pattern}")
            continue
        loader = DirectoryLoader(
            str(notes_dir),
            glob=pattern,
            loader_cls=loader_cls,
            show_progress=True,
        )
        docs = loader.load()
        print(f"Loaded {len(docs)} doc(s) from {pattern}")
        all_docs.extend(docs)

    if not all_docs:
        raise FileNotFoundError(
            f"No PDF or TXT files found in {notes_dir}. "
            "Add some files first."
        )
    return all_docs


def vectorstore_has_documents(chroma_dir=ACTIVE_CHROMA_DIR):
    """Return True when a persisted Chroma index exists and has chunks."""
    chroma_dir = Path(chroma_dir)
    if not chroma_dir.exists():
        return False

    vs = Chroma(
        persist_directory=str(chroma_dir),
    )
    return vs._collection.count() > 0


def build_vectorstore(
    force_rebuild=False,
    notes_dir=ACTIVE_NOTES_DIR,
    chroma_dir=ACTIVE_CHROMA_DIR,
):
    """
    Load docs, split, embed, persist to Chroma.

    Set force_rebuild=True to wipe and re-index everything.
    Otherwise -- if chroma_db/ already exists -- just load from disk.
    """
    notes_dir = Path(notes_dir)
    chroma_dir = Path(chroma_dir)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if chroma_dir.exists() and not force_rebuild:
        print(f"Loading existing vector store from {chroma_dir}...")
        vs = Chroma(
            persist_directory=str(chroma_dir),
            embedding_function=embeddings,
        )
        print(f"Loaded {vs._collection.count()} chunks.")
        return vs

    if force_rebuild and chroma_dir.exists():
        shutil.rmtree(chroma_dir)

    print("Wiping existing index..." if force_rebuild else "No index found. Building...")

    docs = load_documents(notes_dir=notes_dir)
    print(f"Total documents loaded: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,        # max chars per chunk
        chunk_overlap=CHUNK_OVERLAP,  # overlap between adjacent chunks
        separators=["\n\n", "\n", ".", " ", ""],  # try to split at paragraph/line/sentence boundaries
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(chroma_dir),
    )
    print(f"Embedded and stored {len(chunks)} chunks at {chroma_dir}.")
    return vs


def build_sample_vectorstore(force_rebuild=False):
    """Build/load the original sample-notes index."""
    return build_vectorstore(
        force_rebuild=force_rebuild,
        notes_dir=NOTES_DIR,
        chroma_dir=CHROMA_DIR,
    )


if __name__ == "__main__":
    vs = build_vectorstore(force_rebuild=True)
    print("Done.")
