from pathlib import Path

# ----------------------------
# BASE PATHS
# ----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# ----------------------------
# SQLITE
# ----------------------------

SQLITE_DB_PATH = DATA_DIR / "thoughts.db"

# ----------------------------
# EMBEDDINGS
# ----------------------------

# lightweight & fast for laptops
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ----------------------------
# CHROMA (later phase)
# ----------------------------

CHROMA_PERSIST_DIR = DATA_DIR / "chroma"
CHROMA_COLLECTION = "thought_embeddings"
