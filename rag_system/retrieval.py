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


def _embedding_from_json(embedding_str: str) -> Optional[np.ndarray]:
    """Convert stored embedding JSON string back to numpy array."""
    if not embedding_str:
        return None
    
    try:
        return np.array(json.loads(embedding_str), dtype=np.float32)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


# -------------------------------------------------
# LOAD THOUGHTS
# -------------------------------------------------

def _load_all_thoughts() -> List:
    """Load all thoughts with embeddings from SQLite database. Always returns a list."""
    try:
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
        
        return rows if rows else []
        
    except sqlite3.OperationalError as e:
        print(f"⚠️  Database error: {e}")
        return []
    except Exception as e:
        print(f"⚠️  Error loading thoughts: {e}")
        return []


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
    
    CRITICAL: Always returns a list (never None), even if empty.
    
    Args:
        query: Text to search for
        top_k: Number of results to return
        category: Optional filter by category
    
    Returns:
        List of matching thoughts sorted by similarity (guaranteed list, never None)
    """
    
    # Encode query
    try:
        query_vec = _embedder.encode(query)
    except Exception as e:
        print(f"⚠️  Error encoding query: {e}")
        return []  # Return empty list, not None
    
    # Load all thoughts
    rows = _load_all_thoughts()
    
    if not rows:
        return []  # Return empty list, not None
    
    scored = []
    
    for row in rows:
        try:
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
            
            # Skip if no embedding
            if not embedding_json:
                continue
            
            # Load embedding
            thought_vec = _embedding_from_json(embedding_json)
            
            if thought_vec is None:
                continue
            
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
        except Exception as e:
            # Skip problematic rows
            continue
    
    # Sort by similarity (highest first)
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Return top_k results (always a list)
    return scored[:top_k]


# -------------------------------------------------
# SEARCH BY CATEGORY
# -------------------------------------------------

def search_by_category(category: str, limit: int = 10) -> List[Dict]:
    """Get all thoughts of a specific category."""
    try:
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
        
    except Exception as e:
        print(f"⚠️  Error searching by category: {e}")
        return []


# -------------------------------------------------
# RECENT THOUGHTS
# -------------------------------------------------

def get_recent_thoughts(limit: int = 10) -> List[Dict]:
    """Get most recent thoughts."""
    try:
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
        
    except Exception as e:
        print(f"⚠️  Error getting recent thoughts: {e}")
        return []


# -------------------------------------------------
# STATISTICS
# -------------------------------------------------

def get_stats() -> Dict:
    """Get database statistics."""
    try:
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
        total_result = cursor.fetchone()
        total = total_result[0] if total_result else 0
        
        conn.close()
        
        stats = {
            "total": total,
            "by_category": {row[0]: row[1] for row in rows} if rows else {},
        }
        
        return stats
        
    except Exception as e:
        print(f"⚠️  Error getting stats: {e}")
        return {
            "total": 0,
            "by_category": {},
        }