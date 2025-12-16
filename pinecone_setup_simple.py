"""
Alternative Pinecone Setup - No PyTorch Required!
Uses TF-IDF embeddings instead of sentence transformers
"""

import time
import numpy as np
from typing import List, Dict
import logging
from pinecone import Pinecone, ServerlessSpec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import config
from data_loader import InsuranceDataLoader
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimplePineconeManager:
    """
    Pinecone manager using sklearn instead of PyTorch
    """
    
    def __init__(self):
        if not config.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found")
        
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index_name = config.PINECONE_INDEX_NAME
        self.index = None
        
        # Use TF-IDF + SVD for embeddings (no PyTorch!)
        logger.info("Initializing TF-IDF vectorizer (No PyTorch needed!)")
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.svd = TruncatedSVD(n_components=384)  # Reduce to 384 dimensions
        self.is_fitted = False
        
    def create_index(self):
        try:
            existing_indexes = self.pc.list_indexes()
            index_names = [index.name for index in existing_indexes]
            
            if self.index_name in index_names:
                logger.info(f"Index '{self.index_name}' already exists")
                self.index = self.pc.Index(self.index_name)
                return
            
            logger.info(f"Creating new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region=config.PINECONE_ENVIRONMENT)
            )
            
            time.sleep(5)
            self.index = self.pc.Index(self.index_name)
            logger.info("Index created and ready")
            
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using TF-IDF + SVD (no PyTorch!)"""
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        if not self.is_fitted:
            # Fit vectorizer and SVD on all texts
            logger.info("Fitting TF-IDF vectorizer...")
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            logger.info("Reducing dimensions with SVD...")
            self.svd.fit(tfidf_matrix)
            self.is_fitted = True
            
            # Save the fitted models
            with open('tfidf_model.pkl', 'wb') as f:
                pickle.dump((self.vectorizer, self.svd), f)
            logger.info("Models saved to tfidf_model.pkl")
        
        # Transform texts to embeddings
        tfidf_matrix = self.vectorizer.transform(texts)
        embeddings = self.svd.transform(tfidf_matrix)
        
        return embeddings.tolist()
    
    def upsert_vectors(self, data: List[Dict], batch_size: int = 100):
        if self.index is None:
            raise ValueError("Index not initialized")
        
        logger.info(f"Preparing to upsert {len(data)} vectors...")
        
        texts = [item['combined_text'] for item in data]
        embeddings = self.generate_embeddings(texts)
        
        vectors = []
        for item, embedding in zip(data, embeddings):
            vector = {
                'id': item['id'],
                'values': embedding,
                'metadata': {
                    'question': item['question'][:1000],
                    'answer': item['answer'][:1000],
                    'source': item['metadata']['source']
                }
            }
            vectors.append(vector)
        
        logger.info("Upserting vectors to Pinecone...")
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=config.PINECONE_NAMESPACE)
            
            if (i // batch_size + 1) % 5 == 0:
                logger.info(f"Upserted {i + len(batch)}/{len(vectors)} vectors")
        
        logger.info("All vectors upserted successfully")
        time.sleep(2)
        
        stats = self.index.describe_index_stats()
        logger.info(f"Index stats: {stats}")
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.index is None:
            raise ValueError("Index not initialized")
        
        # Load fitted models if needed
        if not self.is_fitted:
            try:
                with open('tfidf_model.pkl', 'rb') as f:
                    self.vectorizer, self.svd = pickle.load(f)
                self.is_fitted = True
                logger.info("Loaded saved TF-IDF models")
            except:
                logger.error("Models not found. Run setup first.")
                return []
        
        # Generate query embedding
        query_tfidf = self.vectorizer.transform([query])
        query_embedding = self.svd.transform(query_tfidf)[0].tolist()
        
        # Search Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=config.PINECONE_NAMESPACE
        )
        
        similar_docs = []
        for match in results['matches']:
            if match['score'] >= config.SIMILARITY_THRESHOLD:
                similar_docs.append({
                    'id': match['id'],
                    'score': match['score'],
                    'question': match['metadata'].get('question', ''),
                    'answer': match['metadata'].get('answer', ''),
                    'source': match['metadata'].get('source', '')
                })
        
        return similar_docs


def setup_simple():
    print("=" * 60)
    print("Simple Setup (No PyTorch!)")
    print("=" * 60)
    
    print("\n[1/3] Loading dataset...")
    loader = InsuranceDataLoader(config.DATASET_NAME)
    loader.load_dataset(split=config.DATASET_SPLIT)
    data = loader.process_dataset()
    print(f"[OK] Processed {len(data)} Q&A pairs")
    
    print("\n[2/3] Setting up Pinecone...")
    manager = SimplePineconeManager()
    manager.create_index()
    print("[OK] Pinecone ready")
    
    print("\n[3/3] Generating embeddings (this may take a few minutes)...")
    manager.upsert_vectors(data)
    print("[OK] Complete!")
    
    print("\n" + "=" * 60)
    print("Setup Complete! Run: python -m streamlit run app_simple.py")
    print("=" * 60)


if __name__ == "__main__":
    setup_simple()

