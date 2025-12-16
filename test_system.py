"""
System Test Script
Tests all components of the Insurance Q&A Chatbot
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70 + "\n")


def test_imports():
    """Test if all required packages can be imported"""
    print_section("Testing Package Imports")
    
    packages = {
        "streamlit": "Streamlit",
        "datasets": "HuggingFace Datasets",
        "sentence_transformers": "Sentence Transformers",
        "pinecone": "Pinecone",
        "pandas": "Pandas",
        "numpy": "NumPy",
        "dotenv": "Python Dotenv",
        "requests": "Requests"
    }
    
    results = {}
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✅ {name:.<50} OK")
            results[package] = True
        except ImportError as e:
            print(f"❌ {name:.<50} FAILED")
            print(f"   Error: {e}")
            results[package] = False
    
    return all(results.values())


def test_config():
    """Test configuration file"""
    print_section("Testing Configuration")
    
    try:
        import config
        
        print(f"Pinecone Index Name: {config.PINECONE_INDEX_NAME}")
        print(f"Embedding Model: {config.EMBEDDING_MODEL}")
        print(f"Embedding Dimension: {config.EMBEDDING_DIMENSION}")
        print(f"Top K Results: {config.TOP_K_RESULTS}")
        print(f"Dataset Name: {config.DATASET_NAME}")
        print(f"Ollama Model: {config.OLLAMA_MODEL}")
        
        # Check if API key is set
        if config.PINECONE_API_KEY and "your_" not in config.PINECONE_API_KEY.lower():
            print("\n✅ Configuration loaded successfully")
            return True
        else:
            print("\n⚠️  Pinecone API key not set in .env file")
            return False
            
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False


def test_data_loader():
    """Test data loader"""
    print_section("Testing Data Loader")
    
    try:
        from data_loader import InsuranceDataLoader
        
        print("Creating data loader...")
        loader = InsuranceDataLoader("deccan-ai/insuranceQA-v2")
        print("✅ Data loader initialized")
        
        # Note: Not loading full dataset in test to save time
        print("⚠️  Skipping full dataset load (run manually if needed)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing data loader: {e}")
        return False


def test_ollama_connection():
    """Test Ollama connection"""
    print_section("Testing Ollama Connection")
    
    try:
        from ollama_client import OllamaClient
        
        print("Creating Ollama client...")
        client = OllamaClient()
        
        print("Checking connection...")
        if client.check_connection():
            print("✅ Ollama is running and accessible")
            
            # Try a simple generation
            print("\nTesting answer generation...")
            test_docs = [{
                'question': 'What is insurance?',
                'answer': 'Insurance is a contract that provides financial protection against loss.',
                'score': 0.95
            }]
            
            answer = client.generate_answer(
                "What is insurance?",
                test_docs
            )
            
            if answer and len(answer) > 10:
                print(f"✅ Answer generated successfully")
                print(f"   Sample: {answer[:100]}...")
                return True
            else:
                print("⚠️  Answer generation returned short response")
                return False
        else:
            print("❌ Cannot connect to Ollama")
            print("\nMake sure:")
            print("  1. Ollama is installed (https://ollama.ai)")
            print("  2. Run: ollama pull mistral")
            print("  3. Ollama service is running")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Ollama: {e}")
        return False


def test_pinecone_connection():
    """Test Pinecone connection"""
    print_section("Testing Pinecone Connection")
    
    try:
        from pinecone_setup import PineconeManager
        
        print("Creating Pinecone manager...")
        manager = PineconeManager()
        print("✅ Pinecone client initialized")
        
        print("\nChecking index...")
        manager.create_index()
        print("✅ Connected to Pinecone index")
        
        print("\nTesting search...")
        results = manager.search_similar("What is a deductible?", top_k=3)
        
        if results:
            print(f"✅ Search returned {len(results)} results")
            print(f"\nSample result:")
            print(f"  Question: {results[0]['question'][:60]}...")
            print(f"  Score: {results[0]['score']:.3f}")
            return True
        else:
            print("⚠️  No results found")
            print("   Have you run 'python pinecone_setup.py' to populate the database?")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Pinecone: {e}")
        print("\nMake sure:")
        print("  1. PINECONE_API_KEY is set in .env")
        print("  2. Run 'python pinecone_setup.py' to create and populate index")
        return False


def test_full_pipeline():
    """Test the full RAG pipeline"""
    print_section("Testing Full RAG Pipeline")
    
    try:
        from pinecone_setup import PineconeManager
        from ollama_client import OllamaClient
        
        print("Initializing components...")
        pinecone_manager = PineconeManager()
        pinecone_manager.create_index()
        ollama_client = OllamaClient()
        
        print("✅ Components initialized")
        
        # Test query
        test_query = "What does auto insurance cover?"
        print(f"\nTest query: '{test_query}'")
        
        print("\n1. Retrieving similar documents...")
        similar_docs = pinecone_manager.search_similar(test_query, top_k=3)
        
        if not similar_docs:
            print("⚠️  No similar documents found")
            return False
        
        print(f"✅ Found {len(similar_docs)} similar documents")
        for i, doc in enumerate(similar_docs, 1):
            print(f"   {i}. Score: {doc['score']:.3f} - {doc['question'][:50]}...")
        
        print("\n2. Generating answer...")
        answer = ollama_client.generate_answer(
            test_query,
            similar_docs
        )
        
        if answer and len(answer) > 20:
            print("✅ Answer generated successfully")
            print("\nGenerated Answer:")
            print("-" * 70)
            print(answer)
            print("-" * 70)
            return True
        else:
            print("❌ Answer generation failed or returned empty response")
            return False
            
    except Exception as e:
        print(f"❌ Error in full pipeline test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print(" Insurance Q&A Chatbot - System Test")
    print("=" * 70)
    
    results = {}
    
    # Run tests
    results['imports'] = test_imports()
    results['config'] = test_config()
    results['data_loader'] = test_data_loader()
    results['ollama'] = test_ollama_connection()
    results['pinecone'] = test_pinecone_connection()
    
    # Only run full pipeline if basic tests pass
    if all([results['ollama'], results['pinecone']]):
        results['pipeline'] = test_full_pipeline()
    else:
        print_section("Skipping Full Pipeline Test")
        print("Fix Ollama and Pinecone issues first")
        results['pipeline'] = False
    
    # Summary
    print_section("Test Summary")
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.capitalize():.<50} {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if all(results.values()):
        print("\n🎉 All tests passed! Your system is ready.")
        print("\nTo start the chatbot:")
        print("   streamlit run app.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Set up .env file with Pinecone credentials")
        print("  - Install Ollama: https://ollama.ai")
        print("  - Pull Mistral model: ollama pull mistral")
        print("  - Run Pinecone setup: python pinecone_setup.py")
    
    return all(results.values())


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests cancelled by user.")
        sys.exit(1)


