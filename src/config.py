import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "documents")
DATA_DIR = os.environ.get("DATA_DIR", "./data")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4
