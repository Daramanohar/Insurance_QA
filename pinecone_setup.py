"""
Pinecone Setup and Embedding Storage
This module handles creating embeddings and storing them in Pinecone vector database
"""

import time
from typing import List, Dict
import logging
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import config
from data_loader import InsuranceDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PineconeManager:
    """
    Manages Pinecone vector database operations including:
    - Index creation
    - Embedding generation
    - Vector storage and retrieval
    """
    
    def __init__(self):
        """
        Initialize Pinecone manager with configuration from config.py
        """
        if not config.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index_name = config.PINECONE_INDEX_NAME
        self.index = None
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully")
        
    def create_index(self) -> None:
        """
        Create a Pinecone index if it doesn't exist.
        Uses serverless architecture for cost-effectiveness.
        """
        try:
            # Check if index already exists
            existing_indexes = self.pc.list_indexes()
            index_names = [index.name for index in existing_indexes]
            
            if self.index_name in index_names:
                logger.info(f"Index '{self.index_name}' already exists")
                self.index = self.pc.Index(self.index_name)
                return
            
            # Create new index
            logger.info(f"Creating new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=config.EMBEDDING_DIMENSION,
                metric='cosine',  # Cosine similarity for semantic search
                spec=ServerlessSpec(
                    cloud='aws',
                    region=config.PINECONE_ENVIRONMENT
                )
            )
            
            # Wait for index to be ready
            logger.info("Waiting for index to be ready...")
            time.sleep(5)
            
            self.index = self.pc.Index(self.index_name)
            logger.info("Index created and ready")
            
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using SentenceTransformers.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once
            
        Returns:
            List of embedding vectors
        """
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            embeddings.extend(batch_embeddings.tolist())
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch)}/{len(texts)} texts")
        
        logger.info("Embedding generation complete")
        return embeddings
    
    def upsert_vectors(self, data: List[Dict], batch_size: int = 100) -> None:
        """
        Generate embeddings and upsert vectors to Pinecone.
        
        Args:
            data: List of dictionaries containing Q&A pairs
            batch_size: Number of vectors to upsert at once
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call create_index() first.")
        
        logger.info(f"Preparing to upsert {len(data)} vectors...")
        
        # Generate embeddings for all combined texts
        texts = [item['combined_text'] for item in data]
        embeddings = self.generate_embeddings(texts)
        
        # Prepare vectors for upsert
        vectors = []
        for i, (item, embedding) in enumerate(zip(data, embeddings)):
            vector = {
                'id': item['id'],
                'values': embedding,
                'metadata': {
                    'question': item['question'][:1000],  # Pinecone metadata limits
                    'answer': item['answer'][:1000],
                    'source': item['metadata']['source']
                }
            }
            vectors.append(vector)
        
        # Upsert in batches
        logger.info(f"Upserting vectors to Pinecone (namespace: '{config.PINECONE_NAMESPACE or 'default'}')...")
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=config.PINECONE_NAMESPACE)
            
            if (i // batch_size + 1) % 5 == 0:
                logger.info(f"Upserted {i + len(batch)}/{len(vectors)} vectors")
        
        logger.info("All vectors upserted successfully")
        
        # Wait for index to update
        time.sleep(2)
        
        # Get index stats
        stats = self.index.describe_index_stats()
        logger.info(f"Index stats: {stats}")
    
    def search_similar(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Search for similar vectors in Pinecone given a query.
        
        Args:
            query: User question/query string
            top_k: Number of similar results to return
            
        Returns:
            List of similar Q&A pairs with scores
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call create_index() first.")
        
        if top_k is None:
            top_k = config.TOP_K_RESULTS
        
        # Generate embedding for query
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Search in Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=config.PINECONE_NAMESPACE
        )
        
        # Format results
        similar_docs = []
        for match in results['matches']:
            # Do NOT over-filter by similarity; return top_k with scores and ids
            similar_docs.append({
                'id': match['id'],
                'score': match['score'],
                'question': match['metadata'].get('question', ''),
                'answer': match['metadata'].get('answer', ''),
                'source': match['metadata'].get('source', '')
            })
        
        return similar_docs
    
    def delete_index(self) -> None:
        """
        Delete the Pinecone index (use with caution!)
        """
        try:
            self.pc.delete_index(self.index_name)
            logger.info(f"Index '{self.index_name}' deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting index: {e}")
            raise


def setup_pinecone_database():
    """
    Main function to set up Pinecone database with InsuranceQA data.
    This should be run once to initialize the database.
    """
    print("=" * 60)
    print("Insurance Q&A Chatbot - Pinecone Setup")
    print("=" * 60)
    
    # Step 1: Load and process dataset
    print("\n[1/3] Loading and processing dataset...")
    loader = InsuranceDataLoader(config.DATASET_NAME)
    loader.load_dataset(split=config.DATASET_SPLIT)
    data = loader.process_dataset()
    
    if not data:
        raise ValueError("No data processed. Check dataset loading.")
    
    print(f"[OK] Processed {len(data)} Q&A pairs")
    
    # Step 2: Initialize Pinecone and create index
    print("\n[2/3] Setting up Pinecone...")
    manager = PineconeManager()
    manager.create_index()
    print("[OK] Pinecone index ready")
    
    # Step 3: Generate embeddings and upsert to Pinecone
    print("\n[3/3] Generating embeddings and storing in Pinecone...")
    manager.upsert_vectors(data)
    print("[OK] All vectors stored successfully")
    
    print("\n" + "=" * 60)
    print("Setup Complete! Your chatbot is ready to use.")
    print("=" * 60)
    print("\nRun 'streamlit run app.py' to start the chatbot.")


if __name__ == "__main__":
    setup_pinecone_database()


