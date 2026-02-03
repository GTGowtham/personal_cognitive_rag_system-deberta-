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

from rag_system.retrieval import search_similar, get_stats, search_by_category, get_recent_thoughts


# -------------------------------------------------
# DISPLAY HELPERS
# -------------------------------------------------

def _print_results(results):
    """Pretty print search results."""
    if not results:
        print("\n❌ No matching thoughts found.\n")
        return
    
    print("\n" + "=" * 70)
    print(f"✅ Found {len(results)} matching thoughts:")
    print("=" * 70 + "\n")
    
    for i, r in enumerate(results, start=1):
        print(f"{i}. 🕒 {r['created_at']}")
        print(f"   💭 Text: {r['text']}")
        print(f"   🏷️  Category: {r['category']} (confidence: {r.get('category_confidence', 0):.2f})")
        
        if 'similarity' in r:
            print(f"   📊 Similarity: {r['similarity']:.4f}")
        
        print(f"   💡 Suggestion: {r['suggestion']}")
        print("-" * 70)
    
    print()


def _print_stats():
    """Display database statistics."""
    stats = get_stats()
    
    print("\n" + "=" * 70)
    print("📊 DATABASE STATISTICS")
    print("=" * 70)
    print(f"\n📝 Total thoughts: {stats['total']}")
    
    if stats['by_category']:
        print("\n📂 By Category:")
        for cat, count in sorted(stats['by_category'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats['total'] * 100) if stats['total'] > 0 else 0
            bar_length = int(percentage / 5)  # Scale to 20 chars max
            bar = "█" * bar_length
            print(f"   {cat:12} {count:4} ({percentage:5.1f}%) {bar}")
    
    print()


def _print_menu():
    """Display main menu."""
    print("\n" + "=" * 70)
    print("🧠 THOUGHT RETRIEVAL SYSTEM")
    print("=" * 70)
    print("\nOptions:")
    print("  1. 🔍 Search similar thoughts")
    print("  2. 📂 Browse by category")
    print("  3. 🕒 View recent thoughts")
    print("  4. 📊 Show statistics")
    print("  5. ❌ Exit")
    print()


def _handle_search():
    """Handle semantic search."""
    query = input("\n🔎 Enter search query: ").strip()
    
    if not query:
        print("⚠️  Query cannot be empty.")
        return
    
    print("\nFilter by category? (leave empty for all)")
    print("Options: Noise, Emotion, Curiosity, Problem, Idea")
    category = input("Category: ").strip()
    
    if category and category not in ['Noise', 'Emotion', 'Curiosity', 'Problem', 'Idea']:
        print(f"⚠️  Invalid category: {category}")
        category = None
    
    if not category:
        category = None
    
    top_k = input("\nHow many results? (default: 5): ").strip()
    try:
        top_k = int(top_k) if top_k else 5
    except ValueError:
        top_k = 5
    
    print(f"\n🔍 Searching for: '{query}'")
    if category:
        print(f"📂 Category filter: {category}")
    print(f"📊 Showing top {top_k} results...\n")
    
    results = search_similar(query, top_k=top_k, category=category)
    _print_results(results)


def _handle_category_browse():
    """Handle category browsing."""
    print("\n📂 Select a category:")
    print("  1. Noise")
    print("  2. Emotion")
    print("  3. Curiosity")
    print("  4. Problem")
    print("  5. Idea")
    
    choice = input("\nEnter number or category name: ").strip()
    
    category_map = {
        '1': 'Noise',
        '2': 'Emotion',
        '3': 'Curiosity',
        '4': 'Problem',
        '5': 'Idea',
    }
    
    if choice in category_map:
        category = category_map[choice]
    elif choice in ['Noise', 'Emotion', 'Curiosity', 'Problem', 'Idea']:
        category = choice
    else:
        print("⚠️  Invalid selection.")
        return
    
    limit = input(f"\nHow many {category} thoughts to show? (default: 10): ").strip()
    try:
        limit = int(limit) if limit else 10
    except ValueError:
        limit = 10
    
    print(f"\n📂 Loading {category} thoughts...\n")
    
    results = search_by_category(category, limit=limit)
    _print_results(results)


def _handle_recent():
    """Handle recent thoughts display."""
    limit = input("\nHow many recent thoughts? (default: 10): ").strip()
    try:
        limit = int(limit) if limit else 10
    except ValueError:
        limit = 10
    
    print(f"\n🕒 Loading {limit} most recent thoughts...\n")
    
    results = get_recent_thoughts(limit=limit)
    _print_results(results)


# -------------------------------------------------
# MAIN CLI
# -------------------------------------------------

def main():
    """Main interactive CLI loop."""
    
    print("\n" + "=" * 70)
    print("Welcome to the Thought Retrieval System! 🧠")
    print("=" * 70)
    
    # Show initial stats
    _print_stats()
    
    while True:
        _print_menu()
        
        choice = input("Select an option (1-5): ").strip()
        
        if choice == '1':
            _handle_search()
        
        elif choice == '2':
            _handle_category_browse()
        
        elif choice == '3':
            _handle_recent()
        
        elif choice == '4':
            _print_stats()
        
        elif choice == '5' or choice.lower() in {'quit', 'exit', 'q'}:
            print("\n👋 Goodbye!\n")
            break
        
        else:
            print("\n⚠️  Invalid option. Please choose 1-5.\n")
        
        # Pause before showing menu again
        input("\nPress Enter to continue...")


# -------------------------------------------------
# SIMPLE MODE (original behavior)
# -------------------------------------------------

def simple_mode():
    """Simple query mode - just search, no menu."""
    print("\n" + "=" * 70)
    print("🔍 Thought Retrieval - Simple Mode")
    print("=" * 70)
    print("\nType a query to search your past thoughts.")
    print("Type 'quit', 'exit', or press ENTER to exit.\n")
    
    while True:
        query = input("> ").strip()
        
        if not query or query.lower() in {'quit', 'exit', 'q'}:
            print("\n👋 Goodbye!\n")
            break
        
        category = input("Filter by category (or press Enter): ").strip()
        if not category:
            category = None
        elif category not in ['Noise', 'Emotion', 'Curiosity', 'Problem', 'Idea']:
            print(f"⚠️  Invalid category: {category}. Searching all categories.")
            category = None
        
        results = search_similar(query, top_k=5, category=category)
        _print_results(results)


# -------------------------------------------------
# ENTRY POINT
# -------------------------------------------------

if __name__ == "__main__":
    import sys
    
    # Check if user wants simple mode
    if len(sys.argv) > 1 and sys.argv[1] in ['--simple', '-s']:
        simple_mode()
    else:
        main()