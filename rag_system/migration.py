#!/usr/bin/env python3
"""
Database Migration Script
Adds the 'embedding' column to existing thoughts table.

Usage:
    python rag_system/migrate_db.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sqlite3
import json
from sentence_transformers import SentenceTransformer
from rag_system.settings import SQLITE_DB_PATH, EMBEDDING_MODEL_NAME


def check_embedding_column_exists():
    """Check if embedding column already exists."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA table_info(thoughts)")
    columns = [row[1] for row in cursor.fetchall()]
    
    conn.close()
    
    return 'embedding' in columns


def add_embedding_column():
    """Add embedding column to thoughts table."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    
    print("Adding 'embedding' column to thoughts table...")
    
    try:
        cursor.execute("ALTER TABLE thoughts ADD COLUMN embedding TEXT")
        conn.commit()
        print("✅ Column added successfully!")
    except sqlite3.OperationalError as e:
        if "duplicate column" in str(e).lower():
            print("⚠️  Column already exists, skipping...")
        else:
            raise
    finally:
        conn.close()


def backfill_embeddings():
    """Generate embeddings for existing thoughts that don't have them."""
    
    print("\nLoading embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("✅ Model loaded!")
    
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    
    # Find thoughts without embeddings
    cursor.execute(
        """
        SELECT thought_id, text 
        FROM thoughts 
        WHERE embedding IS NULL OR embedding = ''
        """
    )
    
    rows = cursor.fetchall()
    
    if not rows:
        print("\n✅ All thoughts already have embeddings!")
        conn.close()
        return
    
    print(f"\n📝 Found {len(rows)} thoughts without embeddings")
    print("Generating embeddings...")
    
    for i, (thought_id, text) in enumerate(rows, 1):
        # Generate embedding
        embedding = embedder.encode(text).tolist()
        embedding_json = json.dumps(embedding)
        
        # Update database
        cursor.execute(
            "UPDATE thoughts SET embedding = ? WHERE thought_id = ?",
            (embedding_json, thought_id)
        )
        
        if i % 10 == 0:
            print(f"  Processed {i}/{len(rows)} thoughts...")
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ Successfully backfilled {len(rows)} embeddings!")


def verify_migration():
    """Verify that all thoughts have embeddings."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM thoughts")
    total = cursor.fetchone()[0]
    
    cursor.execute(
        """
        SELECT COUNT(*) FROM thoughts 
        WHERE embedding IS NOT NULL AND embedding != ''
        """
    )
    with_embeddings = cursor.fetchone()[0]
    
    conn.close()
    
    print("\n" + "=" * 60)
    print("📊 MIGRATION VERIFICATION")
    print("=" * 60)
    print(f"Total thoughts: {total}")
    print(f"With embeddings: {with_embeddings}")
    
    if total == with_embeddings:
        print("\n✅ Migration complete! All thoughts have embeddings.")
    else:
        print(f"\n⚠️  Warning: {total - with_embeddings} thoughts missing embeddings")


def main():
    """Run the migration."""
    
    print("\n" + "=" * 60)
    print("🔧 DATABASE MIGRATION - Add Embedding Support")
    print("=" * 60)
    
    if not SQLITE_DB_PATH.exists():
        print(f"\n❌ Database not found: {SQLITE_DB_PATH}")
        print("Please run ingestion.py first to create the database.")
        return
    
    print(f"\n📁 Database: {SQLITE_DB_PATH}")
    
    # Step 1: Check if column exists
    if check_embedding_column_exists():
        print("\n✅ Embedding column already exists!")
    else:
        # Step 2: Add column
        add_embedding_column()
    
    # Step 3: Backfill embeddings
    backfill_embeddings()
    
    # Step 4: Verify
    verify_migration()
    
    print("\n" + "=" * 60)
    print("🎉 Migration complete!")
    print("=" * 60)
    print("\nYou can now use the retrieval system with stored embeddings.")
    print("Run: python rag_system/query_cli.py")
    print()


if __name__ == "__main__":
    main()