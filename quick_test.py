"""Quick working test"""
from pinecone_setup_simple import SimplePineconeManager
from ollama_client import OllamaClient

print("\n" + "="*60)
print(" TESTING YOUR CHATBOT")
print("="*60)

# Initialize
print("\n1. Initializing Pinecone...")
manager = SimplePineconeManager()
manager.create_index()
print("   [OK]")

print("\n2. Initializing Ollama...")
client = OllamaClient()
print("   [OK]")

# Test query
question = "What is a deductible?"
print(f"\n3. Testing query: '{question}'")

# Search
print("   Searching Pinecone...")
results = manager.search_similar(question, top_k=3)
print(f"   [OK] Found {len(results)} results")

if results:
    print(f"\n   Top result:")
    print(f"   - Question: {results[0]['question'][:70]}...")
    print(f"   - Score: {results[0]['score']:.3f}")

# Generate answer
print("\n4. Generating answer with Mistral...")
answer = client.generate_answer(question, results)
print(f"   [OK] Generated! ({len(answer)} chars)\n")
print("="*60)
print(" ANSWER:")
print("="*60)
print(answer)
print("="*60)
print("\n[SUCCESS] Your chatbot is FULLY WORKING!")
print("\nGo to: http://localhost:8501")
print("Click 'Initialize Chatbot' and start chatting!")
print("="*60)

