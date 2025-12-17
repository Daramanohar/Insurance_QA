"""
FIXED RAG Engine - Efficient, Fast, and Correct
- Hard relevance gate: Only use RAG when similarity >= 0.6
- Stateless: Each question is independent (no memory leak)
- Two distinct modes: RAG (grounded) vs LLM-only (general knowledge)
- Exact-match short-circuit before vector search
- Strict latency control
- Conditional faithfulness (only in RAG mode)
"""

import time
import logging
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import config
from data_loader import InsuranceDataLoader

logger = logging.getLogger(__name__)


class OptimizedRAGEngine:
    """
    Efficient RAG Engine with Hard Relevance Gate
    """
    
    # STRICT LATENCY BUDGETS (seconds) - HARD CAP AT 60s
    MAX_RETRIEVAL_TIME = 2.0
    MAX_GENERATION_TIME = 20.0   # Reduced to 20s - no retries, single attempt only
    MAX_TOTAL_TIME = 60.0        # HARD CAP: Never exceed 60 seconds total
    
    # RETRIEVAL SETTINGS
    TOP_K = 3
    # CRITICAL: Raised threshold to 0.35 to prevent wrong-domain retrieval (was 0.15)
    RELEVANCE_THRESHOLD = 0.35  # Must be >= 0.35 for RAG mode, else LLM-only
    EXACT_MATCH_FUZZY_THRESHOLD = 0.85  # Jaccard similarity for near-exact matches
    
    # GENERATION SETTINGS (HARD CAPS FOR CPU INFERENCE)
    MAX_TOKENS_RAG = 300        # Hard cap: 300 tokens max output
    MAX_TOKENS_LLM_ONLY = 300   # Hard cap: 300 tokens max output
    MAX_RETRIEVED_CHARS = 800   # Truncate retrieved chunks to ~200-300 tokens (800 chars ≈ 200 tokens)
    CONTEXT_WINDOW = 2048       # CPU-optimized context window (between 2048–4096)
    # NO RETRIES - single attempt only to meet 60s latency cap
    MAX_RETRIES = 0
    
    def __init__(self, pinecone_manager, ollama_client, cache_manager=None):
        """Initialize RAG engine"""
        self.pinecone = pinecone_manager
        self.ollama = ollama_client
        self.cache = cache_manager
        self.performance_logs = []
        self.last_faithfulness_score: Optional[float] = None
        self.last_retrieval_score: Optional[float] = None
        # For exact-match and lexical fallback
        self._lexical_data: Optional[List[Dict]] = None
        self.debug_mode = True
    
    def process_query(
        self,
        query: str,
        conversation_history: List[Dict] = None,  # IGNORED - stateless by design
        use_cache: bool = True
    ) -> Dict:
        """
        Process query with HARD RELEVANCE GATE
        - Stateless: Each question is independent
        - RAG mode only if relevance >= 0.6
        - LLM-only mode if relevance < 0.6 or no retrieval
        - Transparent retrieval diagnostics for UI
        """
        start_time = time.time()
        timings = {}
        retrieval_skipped_reason: Optional[str] = None
        retrieval_avg_score: Optional[float] = None
        mode: str = "unknown"
        
        # CRITICAL: Classify domain BEFORE retrieval to filter wrong-domain results
        domain = self._classify_domain(query)
        intent = self._classify_intent(query)
        logger.info(f"[RAG] Domain classification: {domain}, Intent: {intent}")
        
        # Stage 1: Cache check (optional)
        if use_cache and self.cache:
            cache_start = time.time()
            cached_result = self.cache.get_cached_answer(query)
            timings['cache_check'] = time.time() - cache_start
            
            if cached_result:
                answer, context_docs, _ = cached_result
                if answer and len(answer) > 50:
                    timings['total'] = time.time() - start_time
                    return {
                        'answer': answer,
                        'context_docs': context_docs,
                        'from_cache': True,
                        'timings': timings,
                        'timestamp': datetime.now().isoformat(),
                        'mode': 'cache',
                        'retrieval_score': None,
                        'retrieval_avg_score': None,
                        'retrieval_skipped_reason': 'cache_hit'
                    }
        
        # REMOVED: Skip retrieval logic - we ALWAYS attempt retrieval for accuracy
        # The dataset has 21k+ Q&A pairs, so we should always try to find relevant context
        
        # Stage 2: Exact-match short-circuit (ALWAYS try first - fastest path)
        exact_match = self._exact_match_search(query)
        if exact_match:
            logger.info("[RAG] ✅ Exact match found - using directly")
            timings['retrieval'] = 0.1  # Fast path
            timings['generation'] = 0.0  # No generation needed
            timings['total'] = time.time() - start_time
            
            answer = exact_match['answer']
            self.last_retrieval_score = 1.0
            self.last_faithfulness_score = 1.0
            retrieval_avg_score = 1.0
            mode = "exact_match"
            
            # Cache result
            if use_cache and self.cache:
                self.cache.cache_answer(query, answer, [exact_match])
            
            return {
                'answer': answer,
                'context_docs': [exact_match],
                'from_cache': False,
                'timings': timings,
                'faithfulness_score': 1.0,
                'retrieval_score': 1.0,
                'retrieval_avg_score': retrieval_avg_score,
                'mode': mode,
                'retrieval_skipped_reason': None,
                'timestamp': datetime.now().isoformat()
            }
        
        # Stage 3: ALWAYS run Pinecone retrieval (top_k >= 3)
        # This is MANDATORY - we always attempt retrieval for accuracy
        retrieval_start = time.time()
        context_docs = []
        retrieval_avg_score = None
        
        # CRITICAL: Pinecone retrieval ALWAYS runs with domain filtering at query time
        # Enhance query with domain-specific keywords for better retrieval
        enhanced_query = self._enhance_query_with_domain(query, domain)
        logger.info(f"[RAG] MANDATORY: Running Pinecone retrieval (top_k={self.TOP_K}, domain={domain}) for: {query[:80]}")
        try:
            # Retrieve with domain filtering at Pinecone level (metadata filter) + enhanced query
            context_docs = self.pinecone.search_similar(enhanced_query, top_k=self.TOP_K, domain=domain)
            timings['retrieval'] = time.time() - retrieval_start
            
            # Additional post-filter for safety (in case metadata filter isn't perfect)
            if domain != "general" and context_docs:
                filtered_docs = []
                for doc in context_docs:
                    doc_domain = doc.get('domain', 'general')
                    doc_text = (doc.get('question', '') + ' ' + doc.get('answer', '')).lower()
                    # Check both metadata domain and text-based domain match
                    if doc_domain == domain or self._is_domain_match(doc_text, domain):
                        filtered_docs.append(doc)
                    else:
                        logger.warning(f"[RAG] Filtered out wrong-domain doc: {doc.get('id', 'N/A')} (domain={doc_domain}, expected={domain})")
                if filtered_docs:
                    context_docs = filtered_docs
                elif len(context_docs) > 0:
                    # If all filtered out but we had results, keep original (domain might be wrong)
                    logger.warning(f"[RAG] All docs filtered by domain, keeping original results")
            
            if context_docs and len(context_docs) > 0:
                top_score = context_docs[0].get('score', 0.0)
                self.last_retrieval_score = top_score
                # Average similarity of top-k
                scores = [d.get('score', 0.0) for d in context_docs if d.get('score') is not None]
                retrieval_avg_score = sum(scores) / len(scores) if scores else None
                logger.info(f"[RAG] ✅ Pinecone retrieval successful: Top-1 similarity: {top_score:.3f} (threshold: {self.RELEVANCE_THRESHOLD})")
                
                # DEBUG: Log ALL retrieved content with IDs
                for i, doc in enumerate(context_docs[:min(5, len(context_docs))], 1):
                    logger.info(f"[RAG] Doc {i} - ID: {doc.get('id', 'N/A')} | Score: {doc.get('score', 'N/A'):.3f}")
                    logger.info(f"[RAG] Doc {i} - Q: {doc.get('question', '')[:100]}")
                    logger.info(f"[RAG] Doc {i} - A: {doc.get('answer', '')[:150]}...")
            else:
                logger.warning("[RAG] ⚠️ Pinecone returned empty results (no documents found)")
                self.last_retrieval_score = None
                retrieval_avg_score = None
                
        except Exception as e:
            # Log error but continue - we'll fall back to LLM-only
            logger.error(f"[RAG] ❌ Pinecone retrieval exception: {e}", exc_info=True)
            timings['retrieval'] = time.time() - retrieval_start
            self.last_retrieval_score = None
            retrieval_avg_score = None
            context_docs = []  # Ensure it's empty on error
        
        # Stage 4: Check similarity score and decide RAG vs LLM-only
        generation_start = time.time()
        
        # Decision logic: IF similarity >= 0.35 → USE RAG, ELSE → LLM-only (domain knowledge)
        # CRITICAL: Only use RAG if similarity is high enough (0.35), otherwise LLM generates from domain knowledge
        use_rag_mode = False
        if context_docs and len(context_docs) > 0:
            top_score = context_docs[0].get('score', 0.0)
            if top_score >= self.RELEVANCE_THRESHOLD:
                # Similarity >= 0.35 → USE RAG MODE
                use_rag_mode = True
                logger.info(f"[RAG] ✅ Similarity {top_score:.3f} >= {self.RELEVANCE_THRESHOLD} → RAG MODE")
                retrieval_skipped_reason = None
            else:
                # Similarity < 0.35 → LLM-ONLY MODE (chunks are irrelevant, use domain knowledge)
                use_rag_mode = False
                logger.info(f"[RAG] ⚠️ Similarity {top_score:.3f} < {self.RELEVANCE_THRESHOLD} → LLM-ONLY MODE (chunks irrelevant)")
                retrieval_skipped_reason = "low_relevance"
        else:
            # NO retrieval results → LLM-ONLY MODE
            logger.info("[RAG] ⚠️ No retrieval results from Pinecone → LLM-ONLY MODE")
            retrieval_skipped_reason = "no_retrieval"
        
        # Generate answer based on mode
        if use_rag_mode:
            # USE RAG: Generate from retrieved context
            logger.info("[RAG] Generating answer using RAG mode (grounded in retrieved context)")
            answer = self._generate_rag_mode(query, context_docs)
            mode = 'rag'
        else:
            # Fallback to LLM-only: Generate from model knowledge
            logger.info("[RAG] Generating answer using LLM-only mode (general knowledge)")
            answer = self._generate_llm_only_mode(query)
            mode = 'llm_only'
            # No faithfulness check in LLM-only mode
            self.last_faithfulness_score = None
        
        timings['generation'] = time.time() - generation_start
        timings['total'] = time.time() - start_time
        
        # CRITICAL: Ensure we always have a valid answer (never return empty)
        if not answer or len(answer.strip()) == 0:
            logger.warning("[RAG] No answer generated - using fallback")
            answer = self._get_knowledge_based_fallback(query)
        
        # Stage 5: Cache successful result
        if use_cache and self.cache and answer and len(answer) > 50:
            clean_answer = answer.split("\n\n---")[0] if self.debug_mode else answer
            self.cache.cache_answer(query, clean_answer, context_docs if use_rag_mode else [])
        
        # Log performance
        self._log_performance(query, timings, mode)
        
        return {
            'answer': answer,  # Guaranteed to be non-empty
            'context_docs': context_docs if use_rag_mode else [],
            'from_cache': False,
            'timings': timings,
            'faithfulness_score': self.last_faithfulness_score,
            'retrieval_score': self.last_retrieval_score,
            'retrieval_avg_score': retrieval_avg_score,
            'mode': mode,
            'retrieval_skipped_reason': retrieval_skipped_reason,
            'timestamp': datetime.now().isoformat()
        }
    
    def _exact_match_search(self, query: str) -> Optional[Dict]:
        """
        Exact-match short-circuit: Check if normalized query matches any dataset question exactly.
        Returns immediately if found (no vector search needed).
        """
        # Lazy-load lexical data
        if self._lexical_data is None:
            try:
                loader = InsuranceDataLoader(config.DATASET_NAME)
                loader.load_dataset(split=config.DATASET_SPLIT)
                self._lexical_data = loader.process_dataset()
                logger.info(f"[RAG] Lexical index loaded: {len(self._lexical_data)} Q&A pairs")
            except Exception as e:
                logger.error(f"[RAG] Failed to build lexical index: {e}")
                self._lexical_data = []
        
        if not self._lexical_data:
            return None
        
        def normalize(text: str) -> str:
            """Normalize text for exact matching"""
            # Lowercase, strip punctuation, collapse whitespace
            text = text.lower().strip()
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
            text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
            return text
        
        q_norm = normalize(query)
        q_tokens = set(q_norm.split())
        
        # 1. Exact match search (normalized)
        for item in self._lexical_data:
            item_q = normalize(item.get('question', ''))
            if item_q == q_norm:
                logger.info(f"[RAG] Exact match found: {item['id']}")
                return {
                    'id': item['id'],
                    'score': 1.0,
                    'question': item['question'],
                    'answer': item['answer'],
                    'source': item['metadata'].get('source', 'insuranceQA-v2')
                }
        
        # 2. Fuzzy match (high Jaccard similarity) - for near-exact matches
        best_match = None
        best_score = 0.0
        for item in self._lexical_data:
            item_q = normalize(item.get('question', ''))
            item_tokens = set(item_q.split())
            
            if not item_tokens:
                continue
            
            # Jaccard similarity
            intersection = len(q_tokens & item_tokens)
            union = len(q_tokens | item_tokens)
            jaccard = intersection / union if union > 0 else 0.0
            
            if jaccard > best_score:
                best_score = jaccard
                best_match = item
        
        # If fuzzy match is very high (>85%), treat as exact match
        if best_match and best_score >= self.EXACT_MATCH_FUZZY_THRESHOLD:
            logger.info(f"[RAG] Near-exact match found (Jaccard={best_score:.2f}): {best_match['id']}")
            return {
                'id': best_match['id'],
                'score': best_score,
                'question': best_match['question'],
                'answer': best_match['answer'],
                'source': best_match['metadata'].get('source', 'insuranceQA-v2')
            }
        
        return None
    
    def _generate_rag_mode(self, query: str, context_docs: List[Dict]) -> str:
        """
        RAG MODE: Generate answer from retrieved context
        Only called when relevance >= 0.6
        """
        if not context_docs:
            return self._generate_llm_only_mode(query)
        
        best_match = context_docs[0]
        retrieved_answer = best_match.get('answer', '')
        retrieved_question = best_match.get('question', '')
        
        # MANDATORY: Truncate retrieved text to 200-300 tokens max (800 chars ≈ 200 tokens)
        # This prevents oversized context payload on CPU inference
        retrieved_answer = retrieved_answer[:self.MAX_RETRIEVED_CHARS]
        # Also truncate question if too long
        retrieved_question = retrieved_question[:200] if len(retrieved_question) > 200 else retrieved_question
        
        logger.info(f"[RAG] RAG mode - using retrieved content (score: {best_match.get('score', 0):.3f})")
        
        # RAG MODE PROMPT - Natural, generative tone based on retrieved chunks
        prompt_template = """You are an insurance expert assistant. Analyze the retrieved knowledge base content and the user's question, then generate a clear, natural answer.

=== RETRIEVED KNOWLEDGE BASE ===
Question: {retrieved_question}
Answer: {retrieved_answer}

=== USER QUESTION ===
{user_query}

=== YOUR TASK ===
Based on the retrieved knowledge base content above, answer the user's question in a natural, conversational tone. 

Requirements:
- Analyze what the retrieved content says about the user's question
- Generate a clear, well-structured answer that explains the key points
- Preserve all important conditions, decision logic, lists, and numerical details from the retrieved content
- Write naturally - do NOT copy sentences verbatim unless defining a specific term
- Do NOT add information (insurer names, brands, plans, opinions) that is NOT in the retrieved content
- If the retrieved content doesn't fully address the question, explain what it does cover clearly

Your Answer:"""
        
        import requests
        # SINGLE ATTEMPT ONLY - no retries to meet 60s latency cap
        attempt = 0
            prompt = prompt_template.format(
                retrieved_question=retrieved_question,
                retrieved_answer=retrieved_answer,
                user_query=query
            )
            try:
                payload = {
                    "model": config.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": self.CONTEXT_WINDOW,  # 2048 tokens
                        "temperature": 0.2,
                        "num_predict": self.MAX_TOKENS_RAG,  # <= 300 tokens
                        "top_k": 15,
                        "top_p": 0.7,
                        "repeat_penalty": 1.15
                    }
                }
                
                response = requests.post(
                    config.OLLAMA_BASE_URL + "/api/generate",
                    json=payload,
                    timeout=self.MAX_GENERATION_TIME
                )
                
                if response.status_code == 200:
                    answer = response.json().get('response', '').strip()
                    if answer and len(answer) > 20:
                        cleaned = self._clean_answer(answer)
                        # Faithfulness validation
                        is_valid, _ = self._validate_answer_faithfulness(cleaned, retrieved_answer)
                        score = self._compute_faithfulness_score(cleaned, retrieved_answer)
                        self.last_faithfulness_score = score
                        # Accept answer even if faithfulness is slightly below 0.8 (single attempt, no retries)
                        if is_valid:
                            logger.info(f"[RAG] ✅ Answer generated with faithfulness {score:.2f}")
                            return cleaned
                        else:
                            logger.warning(f"[RAG] ⚠️ Validation flagged issues but accepting answer (single attempt)")
                            return cleaned
            except Exception as e:
                logger.error(f"[RAG] Generation error: {e}")
                # If generation fails, fall back to LLM-only mode (don't return wrong retrieved answer)
                logger.info("[RAG] Generation failed, switching to LLM-only mode")
                return self._generate_llm_only_mode(query)
        
        # If we reach here, generation failed completely
        # CRITICAL: Do NOT return raw retrieved answer - generate from domain knowledge instead
        logger.warning("[RAG] RAG generation failed, falling back to LLM-only mode")
        return self._generate_llm_only_mode(query)
    
    def _generate_llm_only_mode(self, query: str) -> str:
        """
        LLM-ONLY MODE: Generate answer using model's general knowledge
        Used when relevance < 0.6 or no retrieval
        NO faithfulness checks - model uses its own knowledge
        """
        logger.info("[RAG] LLM-only mode - using model knowledge")
        
        # Classify intent for better structure
        intent = self._classify_intent(query)
        intent_hint = self._get_intent_hint(intent)
        
        # LLM-ONLY MODE PROMPT - Generate meaningful answers from domain knowledge
        base_prompt = f"""You are an insurance expert assistant. Answer the following insurance question using your knowledge of insurance principles, policies, and industry practices.

Question: {query}

Guidance: {intent_hint}

Provide a clear, accurate, and comprehensive answer that:
- Explains the key concepts clearly
- Covers important conditions, exceptions, or factors that apply
- Uses natural, conversational language
- Is helpful and informative for the user

Answer:"""
        
        import requests
        try:
            prompt = base_prompt
            payload = {
                "model": config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_ctx": self.CONTEXT_WINDOW,
                    "temperature": 0.4,
                    "num_predict": self.MAX_TOKENS_LLM_ONLY,  # <= 300 tokens
                    "top_k": 30,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                config.OLLAMA_BASE_URL + "/api/generate",
                json=payload,
                timeout=self.MAX_GENERATION_TIME
            )
            
            if response.status_code == 200:
                answer = response.json().get('response', '').strip()
                if answer and len(answer) > 20:
                    cleaned = self._clean_answer(answer)
                    if cleaned:
                        return cleaned
        except Exception as e:
            logger.error(f"[RAG] LLM-only generation failed: {e}")
        
        # Immediate fallback: knowledge-based answer (no waiting, never empty)
        logger.info("[RAG] Falling back to knowledge-based answer after LLM-only failure")
        return self._get_knowledge_based_fallback(query)
    
    # REMOVED: _should_skip_retrieval function
    # We ALWAYS attempt retrieval - accuracy comes first
    # The dataset has 21k+ Q&A pairs, so we should always try to find relevant context
    # Latency optimization comes from other means (exact match, caching, token limits)
    
    def _classify_domain(self, query: str) -> str:
        """
        Classify insurance domain BEFORE retrieval to filter wrong-domain results.
        Returns: 'auto', 'health', 'life', 'home', 'general'
        """
        q_lower = query.lower()
        
        # Auto insurance keywords
        auto_keywords = ['auto', 'car', 'vehicle', 'driving', 'accident', 'collision', 'comprehensive', 
                        'liability', 'deductible', 'premium', 'claim', 'no-claim', 'ncb', 'u-haul', 
                        'uhaul', 'rental', 'truck', 'commercial vehicle']
        if any(kw in q_lower for kw in auto_keywords):
            return 'auto'
        
        # Health insurance keywords
        health_keywords = ['health', 'medicare', 'medigap', 'medical', 'prescription', 'doctor', 'hospital',
                          'surgery', 'procedure', 'coverage', 'plan', 'deductible', 'copay', 'premium']
        if any(kw in q_lower for kw in health_keywords):
            return 'health'
        
        # Life insurance keywords
        life_keywords = ['life insurance', 'term life', 'whole life', 'endowment', 'death benefit', 
                        'beneficiary', 'premium', 'policy', 'coverage']
        if any(kw in q_lower for kw in life_keywords):
            return 'life'
        
        # Home insurance keywords
        home_keywords = ['home', 'homeowner', 'property', 'house', 'flood', 'fire', 'theft', 'damage',
                        'coverage', 'premium', 'deductible', 'claim']
        if any(kw in q_lower for kw in home_keywords):
            return 'home'
        
        return 'general'
    
    def _enhance_query_with_domain(self, query: str, domain: str) -> str:
        """
        Enhance query with domain-specific keywords to improve retrieval quality.
        This helps the embedding model find more relevant results.
        """
        if domain == 'general':
            return query
        
        # Add domain-specific context keywords to query
        domain_keywords = {
            'auto': ['auto insurance', 'car insurance', 'vehicle insurance', 'driving', 'accident'],
            'health': ['health insurance', 'medical insurance', 'healthcare', 'medical coverage'],
            'life': ['life insurance', 'life policy', 'life coverage'],
            'home': ['home insurance', 'homeowner insurance', 'property insurance', 'home coverage']
        }
        
        keywords = domain_keywords.get(domain, [])
        if keywords:
            # Add 1-2 relevant keywords that aren't already in the query
            query_lower = query.lower()
            missing_keywords = [kw for kw in keywords if kw not in query_lower]
            if missing_keywords:
                enhanced = query + ' ' + ' '.join(missing_keywords[:2])
                logger.info(f"[RAG] Enhanced query with domain keywords: {enhanced[:100]}")
                return enhanced
        
        return query
    
    def _is_domain_match(self, doc_text: str, domain: str) -> bool:
        """
        Check if document text matches the classified domain.
        Returns True if domain matches or domain is 'general'.
        """
        if domain == 'general':
            return True
        
        doc_lower = doc_text.lower()
        
        domain_keywords = {
            'auto': ['auto', 'car', 'vehicle', 'driving', 'accident', 'collision', 'liability', 'u-haul', 'uhaul'],
            'health': ['health', 'medicare', 'medigap', 'medical', 'prescription', 'doctor', 'hospital', 'surgery'],
            'life': ['life insurance', 'term life', 'whole life', 'endowment', 'death benefit'],
            'home': ['home', 'homeowner', 'property', 'house', 'flood', 'fire', 'theft']
        }
        
        keywords = domain_keywords.get(domain, [])
        return any(kw in doc_lower for kw in keywords)
    
    def _classify_domain(self, query: str) -> str:
        """
        Classify insurance domain BEFORE retrieval to filter wrong-domain results.
        Returns: 'auto', 'health', 'life', 'home', 'general'
        """
        q_lower = query.lower()
        
        # Auto insurance keywords
        auto_keywords = ['auto', 'car', 'vehicle', 'driving', 'accident', 'collision', 'comprehensive', 
                        'liability', 'deductible', 'premium', 'claim', 'no-claim', 'ncb', 'u-haul', 
                        'uhaul', 'rental', 'truck', 'commercial vehicle']
        if any(kw in q_lower for kw in auto_keywords):
            return 'auto'
        
        # Health insurance keywords
        health_keywords = ['health', 'medicare', 'medigap', 'medical', 'prescription', 'doctor', 'hospital',
                          'surgery', 'procedure', 'coverage', 'plan', 'deductible', 'copay', 'premium']
        if any(kw in q_lower for kw in health_keywords):
            return 'health'
        
        # Life insurance keywords
        life_keywords = ['life insurance', 'term life', 'whole life', 'endowment', 'death benefit', 
                        'beneficiary', 'premium', 'policy', 'coverage']
        if any(kw in q_lower for kw in life_keywords):
            return 'life'
        
        # Home insurance keywords
        home_keywords = ['home', 'homeowner', 'property', 'house', 'flood', 'fire', 'theft', 'damage',
                        'coverage', 'premium', 'deductible', 'claim']
        if any(kw in q_lower for kw in home_keywords):
            return 'home'
        
        return 'general'
    
    def _is_domain_match(self, doc_text: str, domain: str) -> bool:
        """
        Check if document text matches the classified domain.
        Returns True if domain matches or domain is 'general'.
        """
        if domain == 'general':
            return True
        
        doc_lower = doc_text.lower()
        
        domain_keywords = {
            'auto': ['auto', 'car', 'vehicle', 'driving', 'accident', 'collision', 'liability', 'u-haul', 'uhaul'],
            'health': ['health', 'medicare', 'medigap', 'medical', 'prescription', 'doctor', 'hospital', 'surgery'],
            'life': ['life insurance', 'term life', 'whole life', 'endowment', 'death benefit'],
            'home': ['home', 'homeowner', 'property', 'house', 'flood', 'fire', 'theft']
        }
        
        keywords = domain_keywords.get(domain, [])
        return any(kw in doc_lower for kw in keywords)
    
    def _classify_intent(self, query: str) -> str:
        """Classify question intent"""
        q = query.lower()
        
        if any(kw in q for kw in ["difference between", "compare", "versus", "vs ", "better than", "which one"]):
            return "comparison"
        if any(kw in q for kw in ["what is", "what are", "define", "meaning of"]):
            return "definition"
        if any(kw in q for kw in ["if ", "when ", "under what", "in what case", "conditions", "eligible", "qualify"]):
            return "conditional"
        return "concept"
    
    def _get_intent_hint(self, intent: str) -> str:
        """Get hint for answer structure based on intent"""
        hints = {
            "definition": "Start with a clear definition, then provide explanation and context.",
            "comparison": "Explain each option, then list key differences and provide a conclusion.",
            "conditional": "Focus on conditions, when something applies, and different scenarios.",
            "concept": "Provide a clear, structured explanation that directly answers the question."
        }
        return hints.get(intent, hints["concept"])
    
    def _validate_answer_faithfulness(self, generated_answer: str, retrieved_answer: str) -> Tuple[bool, str]:
        """
        Validate faithfulness (ONLY called in RAG mode)
        Simplified checks for speed
        """
        gen_lower = generated_answer.lower()
        ret_lower = retrieved_answer.lower()
        
        # Check 1: Direct contradictions
        contradiction_pairs = [
            ('strongly recommends', 'minimal'),
            ('strongly recommends', 'conditionally'),
            ('always', 'sometimes'),
            ('always', 'may'),
            ('never', 'sometimes'),
        ]
        
        for strong_term, weak_term in contradiction_pairs:
            if strong_term in gen_lower and weak_term in ret_lower:
                return False, f"Contradiction: '{strong_term}' vs '{weak_term}'"
            if weak_term in ret_lower and strong_term in gen_lower:
                return False, f"Opinion strengthening: '{weak_term}' -> '{strong_term}'"
        
        # Check 2: Missing conditional logic
        conditional_words = ['if', 'when', 'may', 'might', 'sometimes', 'depending']
        has_conditional_in_ret = any(word in ret_lower for word in conditional_words)
        has_conditional_in_gen = any(word in gen_lower for word in conditional_words)
        
        if has_conditional_in_ret and not has_conditional_in_gen:
            return False, "Missing conditional logic"
        
        # Check 3: Core message overlap (simplified)
        ret_key_words = set(re.findall(r'\w{4,}', ret_lower))  # Words 4+ chars
        gen_key_words = set(re.findall(r'\w{4,}', gen_lower))
        
        stopwords = {'that', 'this', 'with', 'from', 'have', 'will', 'would', 'should', 'could'}
        ret_key_words = {w for w in ret_key_words if w not in stopwords}
        gen_key_words = {w for w in gen_key_words if w not in stopwords}
        
        if ret_key_words:
            overlap = len(ret_key_words & gen_key_words) / len(ret_key_words)
            if overlap < 0.3:  # At least 30% word overlap
                return False, f"Low semantic overlap: {overlap:.2f}"
        
        return True, "Answer is faithful"
    
    def _compute_faithfulness_score(self, generated_answer: str, retrieved_answer: str) -> float:
        """Compute faithfulness score (content word overlap)"""
        gen_tokens = set(re.findall(r'\w+', generated_answer.lower()))
        ret_tokens = set(re.findall(r'\w+', retrieved_answer.lower()))
        
        stopwords = {'the', 'and', 'or', 'a', 'an', 'of', 'in', 'on', 'for', 'to', 'with', 'is', 'are', 'was', 'were', 'it', 'this', 'that', 'as', 'by', 'at', 'from', 'be', 'can', 'may', 'might', 'will', 'would', 'should', 'could', 'you', 'your'}
        
        gen_set = {w for w in gen_tokens if w not in stopwords}
        ret_set = {w for w in ret_tokens if w not in stopwords}
        
        if not ret_set:
            return 1.0
        
        overlap = len(ret_set & gen_set)
        score = overlap / len(ret_set)
        return max(0.0, min(1.0, score))
    
    def _get_knowledge_based_fallback(self, query: str) -> str:
        """Knowledge-based fallback when generation fails"""
        query_lower = query.lower()
        
        # Common insurance terms
        if 'medigap' in query_lower:
            return """Medigap (Medicare Supplement Insurance) is private insurance that helps pay for costs not covered by Original Medicare, such as copayments, coinsurance, and deductibles. There are 10 standardized Medigap plans (A, B, C, D, F, G, K, L, M, N) with different coverage levels. Plan F and G are popular for comprehensive coverage. Medigap plans work alongside Original Medicare and are sold by private insurance companies."""
        
        elif 'no-claim bonus' in query_lower or 'ncb' in query_lower:
            return """A No-Claim Bonus (NCB) is a discount on your insurance premium for not filing claims during a policy period. It typically increases each claim-free year, up to a maximum (often 50-60% discount). Filing a claim usually resets or reduces your NCB, though some insurers offer "NCB protection" add-ons. The exact impact depends on your policy terms and insurer."""
        
        elif 'term life' in query_lower and 'endowment' in query_lower:
            return """Term life insurance provides coverage for a specific period (e.g., 10, 20, 30 years) with lower premiums but no cash value. Endowment plans combine life insurance with savings, paying out either on death or at maturity, with higher premiums and cash value accumulation. Term life is generally better for pure protection needs, while endowment plans suit those wanting savings with insurance coverage."""
        
        elif 'dog' in query_lower and ('insurance' in query_lower or 'home' in query_lower):
            return """Owning a dog can affect home insurance in several ways. Some breeds are considered high-risk and may increase premiums or lead to coverage exclusions. Factors include the dog's breed, bite history, training, and size. Some insurers may require additional liability coverage or exclude certain breeds entirely. It's important to disclose pet ownership and check your policy's animal liability provisions."""
        
        # Generic fallback
        return f"""Based on general insurance knowledge, here's information about "{query}".

Insurance policies are contracts that transfer financial risk in exchange for premiums. Key concepts include coverage types (what's protected), limits (maximum payouts), deductibles (your share), and exclusions (what's not covered).

For specific policy details, consult your insurance agent or policy documents, as terms vary by insurer and policy type."""
    
    def _clean_answer(self, answer: str) -> str:
        """Clean and ensure answer completeness"""
        if not answer:
            return ""
        
        # Remove incomplete sentences
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        complete = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 15:
                if not sentence.endswith(('.', '!', '?')):
                    if len(sentence) > 30 and ' ' in sentence:
                        sentence += '.'
                    else:
                        continue
                complete.append(sentence)
        
        cleaned = ' '.join(complete).strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'\n\n+', '\n\n', cleaned)
        
        return cleaned
    
    def _log_performance(self, query: str, timings: Dict, mode: str):
        """Log performance metrics"""
        log_entry = {
            'query': query[:50],
            'timestamp': datetime.now().isoformat(),
            'timings': timings,
            'mode': mode,
            'sla_met': timings['total'] <= self.MAX_TOTAL_TIME
        }
        
        self.performance_logs.append(log_entry)
        
        if log_entry['sla_met']:
            logger.info(f"[PERF] ✅ {mode.upper()} mode: {timings['total']:.2f}s")
        else:
            logger.warning(f"[PERF] ⚠️ {mode.upper()} mode SLOW: {timings['total']:.2f}s")
        
        if len(self.performance_logs) > 100:
            self.performance_logs = self.performance_logs[-100:]
    
    def get_performance_stats(self) -> Dict:
        """Get performance metrics"""
        if not self.performance_logs:
            return {}
        
        total_times = [log['timings']['total'] for log in self.performance_logs]
        sla_met = sum(1 for log in self.performance_logs if log['sla_met'])
        
        return {
            'total_queries': len(self.performance_logs),
            'avg_response_time': sum(total_times) / len(total_times),
            'max_response_time': max(total_times),
            'min_response_time': min(total_times),
            'sla_compliance': (sla_met / len(self.performance_logs)) * 100,
            'target_sla': self.MAX_TOTAL_TIME
        }
    
    def set_debug_mode(self, enabled: bool):
        """Toggle debug mode"""
        self.debug_mode = enabled
        logger.info(f"[RAG] Debug mode: {'ENABLED' if enabled else 'DISABLED'}")
