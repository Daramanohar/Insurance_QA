"""
Script to fetch and display actual vector IDs from Pinecone
"""

from pinecone_setup import PineconeManager
import random

print("=" * 70)
print(" Fetching Your Pinecone Vector IDs")
print("=" * 70)

# Initialize manager
print("\n1. Connecting to Pinecone...")
manager = PineconeManager()
manager.create_index()
print("[OK] Connected!")

# Get index stats
stats = manager.index.describe_index_stats()
print(f"\nTotal vectors in index: {stats['total_vector_count']}")

# Fetch some sample vectors
print("\n2. Fetching sample vectors to see their IDs...")
print("-" * 70)

# Method 1: Search with a dummy query to get some results
sample_query = "insurance"
results = manager.search_similar(sample_query, top_k=10)

if results:
    print(f"\nFound {len(results)} sample vectors:")
    print("\nSample Vector IDs and their content:")
    print("-" * 70)
    
    for i, result in enumerate(results[:10], 1):
        print(f"\n{i}. ID: {result['id']}")
        print(f"   Similarity: {result['score']:.4f}")
        print(f"   Question: {result['question'][:70]}...")
        print(f"   Answer: {result['answer'][:100]}...")
        
    # Show a copyable list of IDs
    print("\n" + "=" * 70)
    print(" IDs you can try in Pinecone Dashboard:")
    print("=" * 70)
    for result in results[:5]:
        print(f"  {result['id']}")
    
    # Demonstrate how to fetch by ID
    print("\n" + "=" * 70)
    print(" How to Fetch a Vector by ID:")
    print("=" * 70)
    
    test_id = results[0]['id']
    print(f"\nFetching vector with ID: {test_id}")
    
    try:
        # Fetch by ID
        fetch_result = manager.index.fetch(ids=[test_id])
        
        if fetch_result and 'vectors' in fetch_result:
            vector_data = fetch_result['vectors'].get(test_id)
            if vector_data:
                print("\n[OK] Successfully fetched vector!")
                print(f"ID: {test_id}")
                print(f"Dimension: {len(vector_data['values'])}")
                if 'metadata' in vector_data:
                    print(f"Question: {vector_data['metadata'].get('question', 'N/A')[:80]}...")
                    print(f"Answer: {vector_data['metadata'].get('answer', 'N/A')[:100]}...")
    except Exception as e:
        print(f"Error fetching by ID: {e}")

else:
    print("\nNo vectors found. This shouldn't happen!")

print("\n" + "=" * 70)
print(" Tip: Copy one of the IDs above and paste it in Pinecone Dashboard")
print("=" * 70)

