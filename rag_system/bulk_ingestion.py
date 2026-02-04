#!/usr/bin/env python3
"""
Bulk Ingestion Script - Feed Excel Data into RAG System

This script loads thoughts from an Excel file and ingests them
into the RAG system with classification, embeddings, and storage.

Usage:
    python rag_system/bulk_ingest.py path/to/thoughts.xlsx
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import argparse
from tqdm import tqdm
from datetime import datetime

from rag_system.ingestion import ingest_thought
from rag_system.settings import SQLITE_DB_PATH


def load_excel(file_path: str) -> pd.DataFrame:
    """Load thoughts from Excel file."""
    print(f"\n📂 Loading data from: {file_path}")
    
    df = pd.read_excel(file_path)
    
    print(f"✅ Loaded {len(df)} rows")
    print(f"📋 Columns: {list(df.columns)}")
    
    return df


def identify_text_column(df: pd.DataFrame) -> str:
    """Identify which column contains the thought text."""
    possible_names = ['thought_text', 'text', 'thought', 'content', 'message']
    
    for col in possible_names:
        if col in df.columns:
            return col
    
    # If not found, just use first text column
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"⚠️  Using column '{col}' as text column")
            return col
    
    raise ValueError("Could not identify text column in dataset")


def bulk_ingest(
    file_path: str,
    batch_size: int = 100,
    skip_existing: bool = True,
    limit: int = None
):
    """
    Bulk ingest thoughts from Excel file.
    
    Args:
        file_path: Path to Excel file
        batch_size: Number of thoughts to process before showing progress
        skip_existing: Whether to skip if database already has data
        limit: Optional limit on number of thoughts to ingest
    """
    
    print("\n" + "=" * 70)
    print("🚀 BULK INGESTION STARTING")
    print("=" * 70)
    
    # Load data
    df = load_excel(file_path)
    text_col = identify_text_column(df)
    
    print(f"\n📝 Text column: '{text_col}'")
    
    if limit:
        df = df.head(limit)
        print(f"⚠️  Limited to first {limit} thoughts")
    
    # Check existing data
    if SQLITE_DB_PATH.exists() and skip_existing:
        import sqlite3
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM thoughts")
        existing_count = cursor.fetchone()[0]
        conn.close()
        
        if existing_count > 0:
            print(f"\n⚠️  Database already contains {existing_count} thoughts")
            response = input("Continue and add more? (y/n): ").strip().lower()
            if response != 'y':
                print("\n❌ Ingestion cancelled")
                return
    
    # Show preview
    print("\n" + "=" * 70)
    print("📋 DATA PREVIEW")
    print("=" * 70)
    print(f"\nFirst 3 thoughts:")
    for i, text in enumerate(df[text_col].head(3), 1):
        print(f"  {i}. {text}")
    
    # Confirm
    print("\n" + "=" * 70)
    response = input(f"\n🤔 Ingest {len(df)} thoughts? This may take a while. (y/n): ").strip().lower()
    
    if response != 'y':
        print("\n❌ Ingestion cancelled")
        return
    
    # Ingest with progress bar
    print("\n" + "=" * 70)
    print("⚡ INGESTING THOUGHTS...")
    print("=" * 70)
    
    success_count = 0
    error_count = 0
    errors = []
    
    start_time = datetime.now()
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        text = str(row[text_col]).strip()
        
        if not text or len(text) < 3:
            error_count += 1
            continue
        
        try:
            # Suppress print output from ingest_thought
            import io
            import contextlib
            
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                ingest_thought(text)
            
            success_count += 1
            
        except Exception as e:
            error_count += 1
            errors.append((idx, text[:50], str(e)))
            
            if len(errors) <= 5:  # Only store first 5 errors
                pass
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ INGESTION COMPLETE")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"  ✅ Successfully ingested: {success_count}")
    print(f"  ❌ Errors: {error_count}")
    print(f"  ⏱️  Duration: {duration:.2f} seconds")
    print(f"  ⚡ Rate: {success_count/duration:.2f} thoughts/second")
    
    if errors:
        print(f"\n⚠️  First few errors:")
        for idx, text, error in errors[:5]:
            print(f"  Row {idx}: {text}... - {error}")
    
    # Verify database
    import sqlite3
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM thoughts")
    total = cursor.fetchone()[0]
    
    cursor.execute("SELECT category, COUNT(*) FROM thoughts GROUP BY category")
    by_category = dict(cursor.fetchall())
    
    conn.close()
    
    print(f"\n📊 Database now contains {total} thoughts:")
    for cat, count in sorted(by_category.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")
    
    print("\n" + "=" * 70)
    print("🎉 Ready to search! Run: python rag_system/query_cli.py")
    print("=" * 70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Bulk ingest thoughts from Excel file into RAG system"
    )
    parser.add_argument(
        "file",
        type=str,
        help="Path to Excel file containing thoughts"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for progress updates (default: 100)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of thoughts to ingest (for testing)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompts"
    )
    
    args = parser.parse_args()
    
    if not Path(args.file).exists():
        print(f"❌ Error: File not found: {args.file}")
        sys.exit(1)
    
    bulk_ingest(
        file_path=args.file,
        batch_size=args.batch_size,
        limit=args.limit,
        skip_existing=not args.force
    )


if __name__ == "__main__":
    main()