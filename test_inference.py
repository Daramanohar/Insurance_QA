"""
Quick inference test - Run a query through the RAG engine
"""
import sys
import logging
from optimized_rag_engine import OptimizedRAGEngine
from pinecone_setup import PineconeManager
from ollama_client import OllamaClient
from cache_manager import QueryCache
import config

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_inference():
    """Test the RAG engine with a sample question"""
    
    print("=" * 60)
    print("TESTING RAG INFERENCE")
    print("=" * 60)
    
    # Initialize components
    print("\n[1] Initializing components...")
    try:
        pinecone_manager = PineconeManager()
        pinecone_manager.create_index()
        print("   [OK] Pinecone initialized")
        
        ollama_client = OllamaClient()
        if not ollama_client.check_connection():
            print("   [ERROR] Ollama not running!")
            print("   [TIP] Start Ollama: ollama serve")
            return False
        print("   [OK] Ollama connected")
        
        cache_manager = QueryCache(similarity_threshold=0.90)
        print("   [OK] Cache initialized")
        
        rag_engine = OptimizedRAGEngine(
            pinecone_manager,
            ollama_client,
            cache_manager
        )
        print("   [OK] RAG Engine ready")
        
    except Exception as e:
        print(f"   [ERROR] Initialization failed: {e}")
        return False
    
    # Test query
    test_question = "What is a deductible in insurance, and how does choosing a higher deductible affect premiums?"
    
    print(f"\n[2] Testing query:")
    print(f"   Q: {test_question}")
    print(f"\n[3] Processing...")
    
    try:
        result = rag_engine.process_query(
            test_question,
            conversation_history=None,
            use_cache=False
        )
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        
        # Answer
        answer = result.get('answer', 'No answer')
        print(f"\n[ANSWER] ({len(answer)} chars):")
        print("-" * 60)
        print(answer[:500] + ("..." if len(answer) > 500 else ""))
        print("-" * 60)
        
        # Mode and metrics
        mode = result.get('mode', 'unknown')
        print(f"\n[MODE] {mode.upper()}")
        
        if result.get('retrieval_score'):
            print(f"   Top-1 Similarity: {result['retrieval_score']:.1%}")
        if result.get('retrieval_avg_score'):
            print(f"   Avg Top-K: {result['retrieval_avg_score']:.1%}")
        if result.get('faithfulness_score') is not None:
            print(f"   Faithfulness: {result['faithfulness_score']:.1%}")
        
        # Timings
        timings = result.get('timings', {})
        print(f"\n[PERFORMANCE]")
        print(f"   Total: {timings.get('total', 0):.2f}s")
        print(f"   Retrieval: {timings.get('retrieval', 0):.2f}s")
        print(f"   Generation: {timings.get('generation', 0):.2f}s")
        
        # Context docs
        context_docs = result.get('context_docs', [])
        if context_docs:
            print(f"\n[RETRIEVED CONTEXT] ({len(context_docs)} docs):")
            for i, doc in enumerate(context_docs[:2], 1):
                print(f"   {i}. ID: {doc.get('id', 'N/A')}")
                print(f"      Score: {doc.get('score', 0):.3f}")
                print(f"      Q: {doc.get('question', '')[:80]}...")
        
        # Success check
        if answer and len(answer) > 50:
            print("\n" + "=" * 60)
            print("[SUCCESS] TEST PASSED - System is working!")
            print("=" * 60)
            return True
        else:
            print("\n" + "=" * 60)
            print("[WARNING] TEST WARNING - Answer too short")
            print("=" * 60)
            return False
            
    except Exception as e:
        print(f"\n[ERROR] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_inference()
    sys.exit(0 if success else 1)

