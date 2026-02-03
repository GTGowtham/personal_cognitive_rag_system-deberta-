# -------------------------------------------------
# PATH FIX — makes project root importable
# -------------------------------------------------

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # Go to 1_THOUGHTS_CLASSIFIER
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

import sqlite3
import json
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

from rag_system.settings import SQLITE_DB_PATH, EMBEDDING_MODEL_NAME


# -------------------------------------------------
# EMBEDDING MODEL (load once)
# -------------------------------------------------

_embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)


# -------------------------------------------------
# UTILS
# -------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _embedding_from_json(embedding_str: str) -> np.ndarray:
    """
    Convert stored embedding JSON string back to numpy array.
    """
    return np.array(json.loads(embedding_str), dtype=np.float32)


# -------------------------------------------------
# LOAD THOUGHTS
# -------------------------------------------------

def _load_all_thoughts():
    """Load all thoughts with embeddings from SQLite database."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT
            thought_id,
            created_at,
            text,
            category,
            category_confidence,
            suggestion,
            model_version,
            embedding
        FROM thoughts
        """
    )
    
    rows = cursor.fetchall()
    conn.close()
    
    return rows


# -------------------------------------------------
# SEARCH ENGINE
# -------------------------------------------------

def search_similar(
    query: str,
    top_k: int = 5,
    category: Optional[str] = None,
) -> List[Dict]:
    """
    Search for thoughts similar to the query using semantic similarity.
    Uses PRE-COMPUTED embeddings stored in the database.
    
    Args:
        query: Text to search for
        top_k: Number of results to return
        category: Optional filter by category (Noise/Emotion/Curiosity/Problem/Idea)
    
    Returns:
        List of matching thoughts sorted by similarity score
    """
    
    # Encode query ONCE
    query_vec = _embedder.encode(query)
    
    # Load all thoughts with their PRE-COMPUTED embeddings
    rows = _load_all_thoughts()
    
    if not rows:
        print("⚠️  No thoughts found in database.")
        return []
    
    scored = []
    
    for row in rows:
        (
            thought_id,
            created_at,
            text,
            cat,
            cat_conf,
            suggestion,
            model_version,
            embedding_json,
        ) = row
        
        # Filter by category if specified
        if category and cat != category:
            continue
        
        # Load PRE-COMPUTED embedding from database (not re-encoding!)
        thought_vec = _embedding_from_json(embedding_json)
        
        # Calculate similarity
        sim = _cosine_similarity(query_vec, thought_vec)
        
        scored.append(
            {
                "thought_id": thought_id,
                "created_at": created_at,
                "text": text,
                "category": cat,
                "category_confidence": cat_conf,
                "suggestion": suggestion,
                "model_version": model_version,
                "similarity": sim,
            }
        )
    
    # Sort by similarity (highest first)
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    
    return scored[:top_k]


# -------------------------------------------------
# SEARCH BY CATEGORY
# -------------------------------------------------

def search_by_category(category: str, limit: int = 10) -> List[Dict]:
    """
    Get all thoughts of a specific category.
    
    Args:
        category: Category to filter (Noise/Emotion/Curiosity/Problem/Idea)
        limit: Maximum number of results
    
    Returns:
        List of thoughts in the category
    """
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT
            thought_id,
            created_at,
            text,
            category,
            category_confidence,
            suggestion,
            model_version
        FROM thoughts
        WHERE category = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (category, limit),
    )
    
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        results.append(
            {
                "thought_id": row[0],
                "created_at": row[1],
                "text": row[2],
                "category": row[3],
                "category_confidence": row[4],
                "suggestion": row[5],
                "model_version": row[6],
            }
        )
    
    return results


# -------------------------------------------------
# RECENT THOUGHTS
# -------------------------------------------------

def get_recent_thoughts(limit: int = 10) -> List[Dict]:
    """
    Get most recent thoughts.
    
    Args:
        limit: Number of thoughts to return
    
    Returns:
        List of recent thoughts
    """
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT
            thought_id,
            created_at,
            text,
            category,
            category_confidence,
            suggestion,
            model_version
        FROM thoughts
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        results.append(
            {
                "thought_id": row[0],
                "created_at": row[1],
                "text": row[2],
                "category": row[3],
                "category_confidence": row[4],
                "suggestion": row[5],
                "model_version": row[6],
            }
        )
    
    return results


# -------------------------------------------------
# STATISTICS
# -------------------------------------------------

def get_stats() -> Dict:
    """
    Get database statistics.
    
    Returns:
        Dictionary with counts by category
    """
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT category, COUNT(*) as count
        FROM thoughts
        GROUP BY category
        ORDER BY count DESC
        """
    )
    
    rows = cursor.fetchall()
    
    cursor.execute("SELECT COUNT(*) FROM thoughts")
    total = cursor.fetchone()[0]
    
    conn.close()
    
    stats = {
        "total": total,
        "by_category": {row[0]: row[1] for row in rows},
    }
    
    return stats


# -------------------------------------------------
# CLI DEMO
# -------------------------------------------------

if __name__ == "__main__":
    
    print("\n" + "=" * 60)
    print("🔍 THOUGHT RETRIEVAL SYSTEM")
    print("=" * 60)
    
    # Show stats
    print("\n📊 Database Statistics:")
    stats = get_stats()
    print(f"Total thoughts: {stats['total']}")
    print("\nBy category:")
    for cat, count in stats["by_category"].items():
        print(f"  {cat}: {count}")
    
    # Interactive search
    print("\n" + "=" * 60)
    print("Enter a search query (or 'quit' to exit):")
    print("=" * 60)
    
    while True:
        query = input("\n🔎 Search: ").strip()
        
        if not query or query.lower() in {'quit', 'exit', 'q'}:
            print("\n👋 Goodbye!\n")
            break
        
        print("\nSearching...")
        results = search_similar(query, top_k=5)
        
        if not results:
            print("❌ No results found.")
            continue
        
        print(f"\n✅ Found {len(results)} similar thoughts:\n")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. [{result['category']}] (similarity: {result['similarity']:.3f})")
            print(f"   💭 {result['text']}")
            print(f"   📝 {result['suggestion']}")
            print(f"   🕒 {result['created_at']}")
            print()