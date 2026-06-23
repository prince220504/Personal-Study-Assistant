import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
ROOT = Path(__file__).resolve().parent.parent
NOTES_DIR = ROOT / "data" / "notes"
CHROMA_DIR = ROOT / "chroma_db"

# API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in .env file")

# LLM
LLM_MODEL = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0

# Embeddings
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Ingestion
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Retrieval
TOP_K = 4
