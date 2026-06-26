import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
ROOT = Path(__file__).resolve().parent.parent
NOTES_DIR = ROOT / "data" / "notes"
UPLOADS_DIR = ROOT / "data" / "uploads"
CHROMA_DIR = ROOT / "chroma_db"
USER_CHROMA_DIR = ROOT / "user_chroma_db"

# Runtime document source.
# The shipped notes/index can stay on disk as examples, but the app should
# answer and quiz from PDFs uploaded by the current user.
ACTIVE_NOTES_DIR = UPLOADS_DIR
ACTIVE_CHROMA_DIR = USER_CHROMA_DIR

# API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in .env file")

# LLM
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.3

# Embeddings
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Ingestion
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Retrieval
TOP_K = 4
