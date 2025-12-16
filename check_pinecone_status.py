"""
Check Pinecone index status and vector count
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone
import config

# Load environment
load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index(config.PINECONE_INDEX_NAME)

print("="*60)
print("PINECONE INDEX STATUS CHECK")
print("="*60)

# Get index stats
stats = index.describe_index_stats()

print(f"\nIndex Name: {config.PINECONE_INDEX_NAME}")
print(f"Configured Namespace: {config.PINECONE_NAMESPACE}")
print("\nIndex Statistics:")
print(f"Total Vectors: {stats.get('total_vector_count', 0)}")
print(f"Dimension: {stats.get('dimension', 'N/A')}")

print("\nNamespaces:")
namespaces = stats.get('namespaces', {})
if namespaces:
    for ns_name, ns_stats in namespaces.items():
        print(f"  - {ns_name}: {ns_stats.get('vector_count', 0)} vectors")
else:
    print("  No namespaces found (vectors might be in default namespace)")

print("\n" + "="*60)

# Try to query with and without namespace
print("\nTesting queries with different namespace settings:")
print("-"*40)

from pinecone_setup import PineconeManager

# Test with current config
print(f"\n1. Using configured namespace '{config.PINECONE_NAMESPACE}':")
pm = PineconeManager()
pm.create_index()

test_query = "What is insurance?"
try:
    # Generate embedding
    query_embedding = pm.model.encode([test_query])[0].tolist()
    
    # Query with namespace
    results = index.query(
        vector=query_embedding,
        top_k=3,
        namespace=config.PINECONE_NAMESPACE,
        include_metadata=True
    )
    
    if results['matches']:
        print(f"   Found {len(results['matches'])} results")
    else:
        print("   No results found")
except Exception as e:
    print(f"   Error: {e}")

# Test with empty namespace
print(f"\n2. Using default namespace (empty string):")
try:
    results = index.query(
        vector=query_embedding,
        top_k=3,
        namespace="",
        include_metadata=True
    )
    
    if results['matches']:
        print(f"   Found {len(results['matches'])} results")
        print("\n   Sample result:")
        match = results['matches'][0]
        metadata = match.get('metadata', {})
        print(f"   Q: {metadata.get('question', 'N/A')[:100]}")
        print(f"   Score: {match.get('score', 'N/A')}")
    else:
        print("   No results found")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "="*60)
print("DIAGNOSIS:")
if stats.get('total_vector_count', 0) == 0:
    print("[!] NO VECTORS IN INDEX - Need to run pinecone_setup.py to upload data")
else:
    print(f"[OK] Index has {stats.get('total_vector_count', 0)} vectors")
    if not namespaces or config.PINECONE_NAMESPACE not in namespaces:
        print(f"[!] Namespace '{config.PINECONE_NAMESPACE}' not found")
        print("    Vectors might be in a different namespace")
print("="*60)
