from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from .config import NOTES_DIR, CHROMA_DIR, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


def load_documents():
    """Load PDFs and TXT files from the notes directory."""
    # Loader classes mapped to glob patterns
    loaders = [
        (PyPDFLoader, "**/*.pdf"),
        (TextLoader, "**/*.txt"),
    ]

    all_docs = []
    for loader_cls, pattern in loaders:
        files = list(Path(NOTES_DIR).glob(pattern))
        if not files:
            print(f"No files matched: {pattern}")
            continue
        loader = DirectoryLoader(
            str(NOTES_DIR),
            glob=pattern,
            loader_cls=loader_cls,
            show_progress=True,
        )
        docs = loader.load()
        print(f"Loaded {len(docs)} doc(s) from {pattern}")
        all_docs.extend(docs)

    if not all_docs:
        raise FileNotFoundError(
            f"No PDF or TXT files found in {NOTES_DIR}. "
            "Add some files first."
        )
    return all_docs


def build_vectorstore(force_rebuild=False):
    """
    Load docs, split, embed, persist to Chroma.

    Set force_rebuild=True to wipe and re-index everything.
    Otherwise -- if chroma_db/ already exists -- just load from disk.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if CHROMA_DIR.exists() and not force_rebuild:
        print(f"Loading existing vector store from {CHROMA_DIR}...")
        vs = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        )
        print(f"Loaded {vs._collection.count()} chunks.")
        return vs

    print("Wiping existing index..." if force_rebuild else "No index found. Building...")

    docs = load_documents()
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
        persist_directory=str(CHROMA_DIR),
    )
    print(f"Embedded and stored {len(chunks)} chunks at {CHROMA_DIR}.")
    return vs


if __name__ == "__main__":
    vs = build_vectorstore(force_rebuild=True)
    print("Done.")
