"""
Test Pinecone retrieval to debug why answers aren't being found
"""

import os
from dotenv import load_dotenv
from pinecone_setup import PineconeManager

# Load environment
load_dotenv()

# Initialize Pinecone
print("Initializing Pinecone...")
pm = PineconeManager()

# Initialize the index connection
print("Connecting to Pinecone index...")
try:
    pm.create_index()  # This connects to existing index
    print("Connected successfully!")
except Exception as e:
    print(f"Connection error: {e}")

# Test queries that should have answers
test_queries = [
    "What is insurance and why is it important?",
    "What is a premium in an insurance policy?",
    "Does Dave Ramsey Recommend Long Term Disability Insurance?",
    "What is a deductible?",
    "How does life insurance work?"
]

print("\n" + "="*60)
print("TESTING PINECONE RETRIEVAL")
print("="*60)

for query in test_queries:
    print(f"\nQuery: {query}")
    print("-" * 40)
    
    try:
        results = pm.search_similar(query, top_k=3)
        
        if results:
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n  Result {i}:")
                print(f"  Q: {result.get('question', 'N/A')[:100]}...")
                print(f"  A: {result.get('answer', 'N/A')[:150]}...")
                print(f"  Score: {result.get('score', 'N/A')}")
        else:
            print("  [X] NO RESULTS FOUND")
            
    except Exception as e:
        print(f"  [X] ERROR: {e}")

print("\n" + "="*60)
print("If you see 'NO RESULTS' above, the vectors may not be properly indexed.")
print("If you see results but the app returns generic answers, the issue is in generation.")
print("="*60)