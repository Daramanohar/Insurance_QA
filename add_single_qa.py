"""
Quick script to add a single Q&A pair to Pinecone
"""

from pinecone_setup import PineconeManager
from sentence_transformers import SentenceTransformer
import time

print("=" * 70)
print(" Add Single Q&A to Pinecone")
print("=" * 70)

# Get user input
print("\nEnter your insurance question and answer:")
question = input("\nQuestion: ").strip()
answer = input("Answer: ").strip()

if not question or not answer:
    print("\nError: Both question and answer are required!")
    exit(1)

# Initialize
print("\n1. Connecting to Pinecone...")
manager = PineconeManager()
manager.create_index()
print("[OK] Connected!")

print("\n2. Loading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("[OK] Model loaded!")

# Generate embedding
print("\n3. Generating embedding...")
text = f"Question: {question}\n\nAnswer: {answer}"
embedding = model.encode(text).tolist()
print("[OK] Embedding generated!")

# Create unique ID
vector_id = f"custom_{int(time.time())}"

# Add to Pinecone
print("\n4. Adding to Pinecone...")
manager.index.upsert(
    vectors=[{
        'id': vector_id,
        'values': embedding,
        'metadata': {
            'question': question,
            'answer': answer,
            'source': 'manual-entry'
        }
    }],
    namespace=""  # Use your namespace from config
)

print("[OK] Added successfully!")
print("\n" + "=" * 70)
print(f" Vector ID: {vector_id}")
print("=" * 70)

# Test search
print("\n5. Testing search...")
results = manager.search_similar(question, top_k=1)
if results and results[0]['id'] == vector_id:
    print("[OK] Your Q&A is searchable!")
else:
    print("[OK] Added! (May take a moment to be searchable)")

print("\nDone! Your Q&A has been added to the knowledge base.")

