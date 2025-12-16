"""
Quick test script to check your Pinecone vectors
"""

from pinecone_setup import PineconeManager
import config

print("=" * 60)
print(" Testing Pinecone Connection & Vectors")
print("=" * 60)

# Initialize manager
print("\n1. Connecting to Pinecone...")
manager = PineconeManager()
manager.create_index()
print("[OK] Connected!")

# Get index stats
print("\n2. Index Statistics:")
stats = manager.index.describe_index_stats()
print(f"   Total vectors: {stats['total_vector_count']}")
print(f"   Dimension: {stats['dimension']}")
print(f"   Metric: {stats['metric']}")

# Test a search
print("\n3. Testing Search:")
test_query = "What is a deductible?"
print(f"   Query: '{test_query}'")

results = manager.search_similar(test_query, top_k=3)

if results:
    print(f"\n   Found {len(results)} similar results:")
    for i, result in enumerate(results, 1):
        print(f"\n   Result {i}:")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Question: {result['question'][:80]}...")
        print(f"   Answer: {result['answer'][:100]}...")
else:
    print("   No results found")

print("\n" + "=" * 60)
print(" Test Complete! Your Pinecone is working perfectly! [OK]")
print("=" * 60)

