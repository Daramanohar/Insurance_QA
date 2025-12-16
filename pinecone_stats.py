"""
Show detailed Pinecone index statistics
"""

from pinecone_setup import PineconeManager
import config

print("=" * 70)
print(" Pinecone Index Statistics")
print("=" * 70)

# Initialize
print("\nConnecting to Pinecone...")
manager = PineconeManager()
manager.create_index()
print("[OK] Connected!")

# Get detailed stats
print("\n" + "-" * 70)
print(" INDEX INFORMATION")
print("-" * 70)

print(f"\nIndex Name:      {config.PINECONE_INDEX_NAME}")
print(f"Environment:     {config.PINECONE_ENVIRONMENT}")
print(f"Current Namespace: {config.PINECONE_NAMESPACE or '(default)'}")

# Get stats
stats = manager.index.describe_index_stats()

print("\n" + "-" * 70)
print(" VECTOR STATISTICS")
print("-" * 70)

print(f"\nTotal Vectors:   {stats['total_vector_count']:,}")
print(f"Dimension:       {stats['dimension']}")
print(f"Metric:          {stats['metric']}")
print(f"Index Fullness:  {stats.get('index_fullness', 0) * 100:.2f}%")

# Namespaces
if 'namespaces' in stats and stats['namespaces']:
    print("\n" + "-" * 70)
    print(" NAMESPACES")
    print("-" * 70)
    
    for ns_name, ns_data in stats['namespaces'].items():
        display_name = ns_name if ns_name else '__default__'
        print(f"\n{display_name}:")
        print(f"  Vectors: {ns_data.get('vector_count', 0):,}")

# Configuration
print("\n" + "-" * 70)
print(" CURRENT CONFIGURATION")
print("-" * 70)

print(f"\nEmbedding Model:     {config.EMBEDDING_MODEL}")
print(f"Embedding Dimension: {config.EMBEDDING_DIMENSION}")
print(f"Top K Results:       {config.TOP_K_RESULTS}")
print(f"Similarity Threshold: {config.SIMILARITY_THRESHOLD}")
print(f"Dataset:             {config.DATASET_NAME}")

# Model info
print("\n" + "-" * 70)
print(" EMBEDDING MODEL")
print("-" * 70)

try:
    print(f"\nModel loaded: {config.EMBEDDING_MODEL}")
    print(f"Output dimension: {config.EMBEDDING_DIMENSION}")
    print("Status: Ready")
except Exception as e:
    print(f"Status: Error - {e}")

# Quick search test
print("\n" + "-" * 70)
print(" QUICK SEARCH TEST")
print("-" * 70)

test_query = "insurance"
print(f"\nTesting search with query: '{test_query}'")

try:
    results = manager.search_similar(test_query, top_k=3)
    print(f"[OK] Found {len(results)} results")
    
    if results:
        print("\nTop result:")
        print(f"  ID: {results[0]['id']}")
        print(f"  Score: {results[0]['score']:.4f}")
        print(f"  Question: {results[0]['question'][:60]}...")
except Exception as e:
    print(f"[ERROR] Search failed: {e}")

print("\n" + "=" * 70)
print(" Summary")
print("=" * 70)

print(f"\nYour Pinecone index has {stats['total_vector_count']:,} vectors")
print(f"ready for similarity search!")

if stats['total_vector_count'] == 0:
    print("\nIndex is empty. Run 'python pinecone_setup.py' to populate.")
elif stats['total_vector_count'] < 1000:
    print(f"\nYou have a small dataset. Consider adding more data.")
else:
    print(f"\nYour index is ready for production use!")

print("\n" + "=" * 70)

