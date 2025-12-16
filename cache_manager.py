"""
Query Caching and Deduplication System
Reduces latency by caching previous answers and detecting similar queries
"""

import hashlib
import json
import pickle
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class QueryCache:
    """
    Intelligent query caching system with semantic similarity detection
    """
    
    def __init__(self, cache_file: str = "query_cache.pkl", similarity_threshold: float = 0.95):
        """
        Initialize query cache
        
        Args:
            cache_file: Path to cache file
            similarity_threshold: Minimum similarity to reuse cached answer (0-1)
        """
        self.cache_file = Path(cache_file)
        self.similarity_threshold = similarity_threshold
        self.cache = self._load_cache()
        
    def _load_cache(self) -> Dict:
        """Load cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                logger.info(f"Loaded cache with {len(cache)} entries")
                return cache
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Could not save cache: {e}")
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalize query for better matching
        
        Args:
            query: User query string
            
        Returns:
            Normalized query
        """
        # Convert to lowercase
        normalized = query.lower().strip()
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        # Remove common filler words for better matching
        fillers = ['please', 'could you', 'can you', 'tell me']
        for filler in fillers:
            normalized = normalized.replace(filler, '')
        
        return normalized.strip()
    
    def _compute_query_hash(self, query: str) -> str:
        """
        Compute semantic hash of query
        
        Args:
            query: Normalized query
            
        Returns:
            Hash string
        """
        return hashlib.md5(query.encode()).hexdigest()
    
    def _compute_similarity(self, query1: str, query2: str) -> float:
        """
        Compute simple similarity between two queries
        Uses Jaccard similarity on word sets
        
        Args:
            query1, query2: Queries to compare
            
        Returns:
            Similarity score (0-1)
        """
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def get_cached_answer(self, query: str) -> Optional[Tuple[str, List[Dict], bool]]:
        """
        Get cached answer if available
        
        Args:
            query: User query
            
        Returns:
            Tuple of (answer, context_docs, from_cache) or None
        """
        normalized = self._normalize_query(query)
        query_hash = self._compute_query_hash(normalized)
        
        # Check exact match
        if query_hash in self.cache:
            entry = self.cache[query_hash]
            logger.info(f"Cache HIT (exact): {query[:50]}...")
            entry['hits'] = entry.get('hits', 0) + 1
            entry['last_accessed'] = datetime.now().isoformat()
            self._save_cache()
            return entry['answer'], entry['context_docs'], True
        
        # Check semantic similarity
        for cached_hash, entry in self.cache.items():
            cached_query = entry['normalized_query']
            similarity = self._compute_similarity(normalized, cached_query)
            
            if similarity >= self.similarity_threshold:
                logger.info(f"Cache HIT (similar {similarity:.2f}): {query[:50]}...")
                entry['hits'] = entry.get('hits', 0) + 1
                entry['last_accessed'] = datetime.now().isoformat()
                self._save_cache()
                return entry['answer'], entry['context_docs'], True
        
        logger.info(f"Cache MISS: {query[:50]}...")
        return None
    
    def cache_answer(self, query: str, answer: str, context_docs: List[Dict]):
        """
        Cache an answer for future reuse
        
        Args:
            query: User query
            answer: Generated answer
            context_docs: Retrieved context documents
        """
        normalized = self._normalize_query(query)
        query_hash = self._compute_query_hash(normalized)
        
        self.cache[query_hash] = {
            'original_query': query,
            'normalized_query': normalized,
            'answer': answer,
            'context_docs': context_docs,
            'timestamp': datetime.now().isoformat(),
            'hits': 0,
            'last_accessed': datetime.now().isoformat()
        }
        
        self._save_cache()
        logger.info(f"Cached answer for: {query[:50]}...")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        if not self.cache:
            return {'total_entries': 0, 'total_hits': 0}
        
        total_hits = sum(entry.get('hits', 0) for entry in self.cache.values())
        
        return {
            'total_entries': len(self.cache),
            'total_hits': total_hits,
            'avg_hits_per_entry': total_hits / len(self.cache) if self.cache else 0
        }
    
    def clear_cache(self):
        """Clear all cached entries"""
        self.cache = {}
        self._save_cache()
        logger.info("Cache cleared")

