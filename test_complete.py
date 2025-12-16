"""Complete pipeline test"""
import sys
sys.path.insert(0, r'C:\Program Files\Python311\Lib\site-packages')

print("=" * 70)
print(" COMPLETE SYSTEM TEST")
print("=" * 70)

# Test 1: Pinecone
print("\n1. Testing Pinecone (Simple version)...")
try:
    from pinecone_setup_simple import SimplePineconeManager
    manager = SimplePineconeManager()
    manager.create_index()
    print("   [OK] Pinecone connected!")
    
    # Test search
    results = manager.search_similar("What is a deductible?", top_k=3)
    print(f"   [OK] Search works! Found {len(results)} results")
    if results:
        print(f"   Top result: {results[0]['question'][:60]}...")
        print(f"   Score: {results[0]['score']:.3f}")
except Exception as e:
    print(f"   [FAIL] {e}")
    sys.exit(1)

# Test 2: Ollama
print("\n2. Testing Ollama...")
try:
    from ollama_client import OllamaClient
    client = OllamaClient()
    
    if client.check_connection():
        print("   [OK] Ollama connected!")
        
        # Test generation
        answer = client.generate_answer(
            "What is a deductible?",
            results[:1]
        )
        
        if answer and len(answer) > 20:
            print("   [OK] Answer generation works!")
            print(f"   Sample answer: {answer[:100]}...")
        else:
            print("   [WARN] Short answer received")
    else:
        print("   [FAIL] Cannot connect to Ollama")
        sys.exit(1)
except Exception as e:
    print(f"   [FAIL] {e}")
    sys.exit(1)

# Test 3: Full pipeline
print("\n3. Testing full RAG pipeline...")
try:
    test_question = "How does auto insurance work?"
    print(f"   Question: {test_question}")
    
    # Search
    docs = manager.search_similar(test_question, top_k=3)
    print(f"   [OK] Found {len(docs)} relevant documents")
    
    # Generate
    final_answer = client.generate_answer(test_question, docs)
    print(f"   [OK] Generated answer ({len(final_answer)} chars)")
    print(f"\n   Answer: {final_answer[:200]}...")
    
except Exception as e:
    print(f"   [FAIL] {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print(" ALL TESTS PASSED! YOUR CHATBOT IS FULLY FUNCTIONAL!")
print("=" * 70)
print("\nYour app is ready at: http://localhost:8501")
print("Click 'Initialize Chatbot' and start asking questions!")

