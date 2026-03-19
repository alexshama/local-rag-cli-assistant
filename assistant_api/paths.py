from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
DATA_DIR = MODULE_DIR / "data"

# Stable storage paths independent of current working directory.
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
DEFAULT_CACHE_DB_PATH = PROJECT_ROOT / "semantic_rag_cache_v2.db"
EVALUATION_CACHE_DB_PATH = PROJECT_ROOT / "api_rag_cache.db"
LOG_FILE_PATH = PROJECT_ROOT / "app.log"

ENV_FILE_PATH = PROJECT_ROOT / ".env"

DEFAULT_DATA_SOURCES = {
    "docs": DATA_DIR / "docs.txt",
    "python": DATA_DIR / "python.txt",
}
