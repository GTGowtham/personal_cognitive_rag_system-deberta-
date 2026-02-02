# -------------------------------------------------
# PATH FIX — makes project root importable
# -------------------------------------------------

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # Go to 1_THOUGHTS_CLASSIFIER
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -------------------------------------------------
# STANDARD IMPORTS
# -------------------------------------------------

import uuid
import sqlite3
from datetime import datetime

from sentence_transformers import SentenceTransformer

# -------------------------------------------------
# PROJECT IMPORTS
# -------------------------------------------------

from rag_system.settings import SQLITE_DB_PATH, EMBEDDING_MODEL_NAME

# classifier inference function
from classifier.src.inference.predict import classify_thought


# -------------------------------------------------
# EMBEDDING MODEL (load once)
# -------------------------------------------------

_embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)


# -------------------------------------------------
# SQLITE SETUP
# -------------------------------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS thoughts (
    thought_id TEXT PRIMARY KEY,
    created_at TEXT,
    text TEXT,
    category TEXT,
    category_confidence REAL,
    suggestion TEXT,
    model_version TEXT
);
"""


def _get_connection():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.execute(CREATE_TABLE_SQL)
    return conn


# -------------------------------------------------
# INGESTION PIPELINE
# -------------------------------------------------

def ingest_thought(text: str) -> str:

    thought_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()

    # -----------------------------
    # 1) classify
    # -----------------------------

    result = classify_thought(text)

    category = result["category"]
    category_conf = result.get("category_confidence")
    suggestion = result.get("suggestion")
    model_version = result.get("model_version", "v1")

    # -----------------------------
    # 2) embedding
    # -----------------------------

    embedding = _embedder.encode(text).tolist()

    # -----------------------------
    # 3) sqlite insert
    # -----------------------------

    conn = _get_connection()

    conn.execute(
        """
        INSERT INTO thoughts (
            thought_id,
            created_at,
            text,
            category,
            category_confidence,
            suggestion,
            model_version
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            thought_id,
            created_at,
            text,
            category,
            category_conf,
            suggestion,
            model_version,
        ),
    )

    conn.commit()
    conn.close()

    print("\n--- INGESTED THOUGHT ---")
    print("ID:", thought_id)
    print("Category:", category)
    print("Confidence:", category_conf)
    print("Suggestion:", suggestion)
    print("-----------------------\n")

    return thought_id


# -------------------------------------------------
# CLI ENTRY
# -------------------------------------------------

if __name__ == "__main__":

    print("\nEnter a thought (type 'quit' or 'exit' to stop, or leave empty):\n")

    while True:

        user_text = input("> ").strip()

        # Exit conditions
        if not user_text or user_text.lower() in {'quit', 'exit', 'q'}:
            print("\n👋 Goodbye!\n")
            break

        ingest_thought(user_text)