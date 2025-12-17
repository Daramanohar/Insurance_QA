# 📊 Insurance Q&A RAG Chatbot - Presentation Slides
## Complete Slide-by-Slide Guide for Jury Presentation

---

## **Slide 1: Title Slide**

**Title:** Insurance Knowledge-Base Q&A Chatbot  
**Subtitle:** Retrieval-Augmented Generation (RAG) System  
**Your Name**  
**Date**  
**Institution/Organization**

---

## **Slide 2: Project Overview**

**Title:** What is This Project?

**Content:**
- **AI-powered chatbot** that answers insurance-related questions
- Uses **Retrieval-Augmented Generation (RAG)** architecture
- Combines **curated knowledge base** (21,325 Q&A pairs) with **local LLM** (Mistral-7B)
- **No OpenAI API** - completely open-source and local
- **Domain-specific**: Health, Auto, Life, and Home insurance

**Key Point:** "A production-ready system that delivers accurate, faithful answers by grounding responses in a verified insurance knowledge base."

---

## **Slide 3: Objectives**

**Title:** Project Objectives

**Content:**
1. **Accuracy First**: Answer insurance questions correctly using verified knowledge base
2. **Transparency**: Show users where answers come from (source references, similarity scores)
3. **Local & Open-Source**: Run entirely on local hardware using open-source models
4. **Production-Ready**: Handle edge cases, errors, and provide consistent performance
5. **Domain Expertise**: Specialized for insurance domain with quality controls

**Visual:** 5 bullet points with icons

---

## **Slide 4: High-Level Architecture**

**Title:** System Architecture Overview

**Content:**
```
┌─────────────┐
│   User      │
│  Question   │
└──────┬──────┘
       │
       ▼
┌─────────────────┐     ┌──────────────┐
│  Streamlit UI   │────▶│  RAG Engine  │
└─────────────────┘     └──────┬───────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
        │  Pinecone   │  │  Sentence   │  │   Ollama   │
        │ Vector DB   │  │Transformers │  │ Mistral-7B  │
        └─────────────┘  └─────────────┘  └─────────────┘
```

**Key Components:**
- **UI Layer**: Streamlit chat interface
- **Retrieval Layer**: Pinecone vector database
- **Embedding Layer**: Sentence Transformers
- **Generation Layer**: Mistral-7B via Ollama

---

## **Slide 5: Dataset - HuggingFace InsuranceQA-v2**

**Title:** Knowledge Base: InsuranceQA-v2 Dataset

**Content:**
- **Source**: HuggingFace dataset `deccan-ai/insuranceQA-v2`
- **Size**: 21,325 Q&A pairs (train split)
- **Format**: 
  - Question (user query)
  - Multiple candidate answers
  - Ground truth (correct answer index)
- **Domain Coverage**: Health, Auto, Life, Home insurance
- **Quality**: Expert-verified answers from insurance domain

**Why This Dataset?**
- Large, curated, domain-specific
- Real-world insurance questions
- Expert-validated answers

**Visual:** Screenshot of dataset on HuggingFace or sample Q&A pairs

---

## **Slide 6: Technology Stack**

**Title:** Technology Stack

**Content:**

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Vector DB** | Pinecone | Fast semantic search over 21K+ vectors |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) | Convert text to 384-dim vectors |
| **LLM** | Mistral-7B-Instruct | Answer generation |
| **LLM Server** | Ollama | Local model serving |
| **UI Framework** | Streamlit | Interactive web interface |
| **Language** | Python 3.11 | Backend implementation |

**Key Point:** All open-source, no proprietary APIs

---

## **Slide 7: Why Mistral-7B?**

**Title:** Model Selection: Mistral-7B-Instruct

**Content:**
- **Open-Source**: No API costs, full control
- **7B Parameters**: Good balance of quality and speed
- **Instruct-Tuned**: Better at following prompts and instructions
- **Local Deployment**: Runs on CPU (no GPU required)
- **RAG-Optimized**: Works well with retrieved context
- **Community Support**: Active development and optimization

**Comparison:**
- vs GPT-3.5: Free, local, no API dependency
- vs Larger models (13B+): Faster inference, lower memory
- vs Smaller models (3B): Better quality and instruction following

---

## **Slide 8: RAG Architecture Explained**

**Title:** How RAG Works

**Content:**

**Traditional LLM:**
- User Question → LLM → Answer (may hallucinate)

**RAG (Our Approach):**
1. User Question → **Retrieve** relevant chunks from knowledge base
2. Retrieved Context + Question → **LLM** → Grounded Answer
3. Answer is **faithful** to knowledge base

**Benefits:**
- ✅ Reduces hallucinations
- ✅ Uses verified knowledge base
- ✅ Transparent (shows sources)
- ✅ Updatable (add new Q&A pairs)

**Visual:** Side-by-side comparison diagram

---

## **Slide 9: Data Pipeline - Chunking & Embedding**

**Title:** Data Ingestion Pipeline

**Content:**

**Step 1: Load Dataset**
- Download from HuggingFace
- Process 21,325 Q&A pairs

**Step 2: Format Creation**
- Combine: `"Question: {question}\n\nAnswer: {answer}"`
- Classify domain (auto/health/life/home)
- Attach metadata (source, domain, original index)

**Step 3: Embedding Generation**
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: 384
- Process: Text → 384-dim dense vector

**Step 4: Storage**
- Store in Pinecone with metadata
- Index: `insurance-qa-index`
- Metric: Cosine similarity

**Visual:** Flow diagram showing: Dataset → Processing → Embedding → Storage

---

## **Slide 10: Query Processing Pipeline**

**Title:** Query-to-Answer Pipeline

**Content:**

**Stage 1: Query Input**
- User enters question in Streamlit UI

**Stage 2: Domain Classification**
- Classify: auto/health/life/home/general
- Enhances query with domain keywords

**Stage 3: Embedding**
- Convert query to 384-dim vector
- Same embedding model as indexing

**Stage 4: Vector Search**
- Pinecone cosine similarity search
- Retrieve top-k=3 most similar Q&A pairs
- Domain filtering at database level

**Stage 5: Mode Decision**
- If similarity ≥ 0.35 → RAG mode (use retrieved context)
- If similarity < 0.35 → LLM-only mode (domain knowledge)

**Visual:** Step-by-step flow diagram

---

## **Slide 11: Retrieval - Pinecone Vector Search**

**Title:** Retrieval: Finding Relevant Context

**Content:**

**How It Works:**
1. Query embedding (384-dim vector)
2. Cosine similarity search in Pinecone
3. Returns top-k=3 most similar Q&A pairs
4. Each result includes:
   - Similarity score (0-1)
   - Original question
   - Original answer
   - Pinecone ID

**Domain Filtering:**
- Metadata filter: `domain = 'auto'` (for auto questions)
- Prevents wrong-domain results (e.g., Medicare answers for auto questions)
- Post-filtering for safety

**Example:**
- Query: "Does filing a claim affect No-Claim Bonus?"
- Domain: `auto`
- Retrieves: Auto insurance Q&A about NCB and claims
- Filters out: Health/life/home insurance results

**Visual:** Pinecone dashboard screenshot or retrieval example

---

## **Slide 12: Generation - LLM Analysis & Context**

**Title:** Answer Generation: RAG Mode

**Content:**

**RAG Mode (when similarity ≥ 0.35):**
1. **Context Building**: Retrieved Q&A pairs + user question
2. **Prompt Engineering**: 
   - "Analyze retrieved knowledge base content"
   - "Generate natural answer preserving all conditions and facts"
   - "Do NOT add new information not in retrieved content"
3. **LLM Generation**: Mistral-7B generates explanation
4. **Validation**: Faithfulness check (≥80% content overlap)
5. **Output**: Natural, faithful answer

**LLM-Only Mode (when similarity < 0.35):**
- Uses model's general insurance knowledge
- No retrieved context (chunks are irrelevant)
- Still domain-aware and accurate

**Visual:** Prompt structure diagram or example prompt

---

## **Slide 13: User Interface - Streamlit**

**Title:** User Interface: Streamlit Chat Application

**Content:**

**Features:**
- **Chat Interface**: Natural conversation flow
- **Source References**: Shows retrieved Q&A pairs
- **Transparency Metrics**:
  - Mode (RAG / LLM-only)
  - Similarity scores (Top-1, Avg top-k)
  - Faithfulness score
  - Pinecone IDs
- **Performance Indicators**: Response time, cache hits
- **System Info**: Model name, knowledge base size

**User Experience:**
- Clean, professional design
- Real-time response generation
- Collapsible source references
- Example questions for quick start

**Visual:** Screenshot of the Streamlit UI

---

## **Slide 14: CPU Challenges & Latency**

**Title:** CPU-Only Inference: Challenges & Solutions

**Content:**

**Challenge 1: Large Context Window**
- Context: 2048-4096 tokens
- RAG injects retrieved Q&A into prompt
- Large KV cache builds for each request
- **Impact**: Slower generation

**Challenge 2: Attention Computation**
- Mistral-7B has 7 billion parameters
- Attention matrices recomputed on CPU
- No GPU acceleration
- **Impact**: 25-40 seconds per RAG response

**Challenge 3: Memory Constraints**
- CPU-only inference uses RAM
- Limited parallelization
- **Impact**: Sequential processing

**Our Optimizations:**
- ✅ Context window: 2048 tokens (not 4096)
- ✅ Max tokens: 300 (prevents rambling)
- ✅ Single attempt (no retries)
- ✅ Query caching (repeated questions: <0.1s)
- ✅ Domain filtering (reduces irrelevant context)

**Expected Performance:**
- LLM-only: 10-15 seconds
- RAG mode: 25-40 seconds
- With GPU: 5-15 seconds (typical)

**Visual:** Performance comparison chart (CPU vs GPU)

---

## **Slide 15: Latency Breakdown**

**Title:** Response Time Breakdown

**Content:**

**Typical RAG Query (CPU):**
- Cache check: 0.01s
- Retrieval (Pinecone): 0.5-1.0s
- Query embedding: 0.1s
- LLM generation: 20-30s ⚠️ (bottleneck)
- Validation & formatting: 0.5s
- **Total: 25-40 seconds**

**Cached Query:**
- Cache hit: 0.01s
- **Total: <0.1 seconds** ✅

**Why Generation is Slow:**
1. Large model (7B parameters)
2. CPU-only (no GPU acceleration)
3. Large context (retrieved Q&A + prompt)
4. Attention computation (quadratic complexity)

**Visual:** Timeline diagram showing each stage's time

---

## **Slide 16: How RAG Works - Detailed Flow**

**Title:** RAG Pipeline: Step-by-Step

**Content:**

**Example Query:** "Does filing a claim affect No-Claim Bonus?"

**Step 1: Domain Classification**
- Detected: `auto` insurance
- Enhanced query: "Does filing a claim affect No-Claim Bonus? auto insurance car insurance"

**Step 2: Vector Search**
- Query → 384-dim embedding
- Pinecone search with `domain='auto'` filter
- Top-3 results retrieved (similarity: 0.85, 0.78, 0.72)

**Step 3: Mode Selection**
- Top similarity: 0.85 ≥ 0.35 threshold
- **RAG Mode** selected

**Step 4: Context Building**
- Retrieved Q&A: "Filing a claim typically resets NCB, but some policies offer NCB protection..."
- User question included

**Step 5: LLM Generation**
- Mistral-7B analyzes retrieved content
- Generates natural explanation
- Preserves all conditions and facts

**Step 6: Validation**
- Faithfulness check: 92% (passes)
- Answer displayed with sources

**Visual:** Complete flow diagram with example

---

## **Slide 17: Quality Controls**

**Title:** Ensuring Answer Quality

**Content:**

**Multi-Layer Validation:**
1. **Language Check**: English-only enforcement
2. **Hallucination Detection**: Blocks "based on documents" phrases
3. **Factual Validation**: Prevents known incorrect claims
4. **Faithfulness Score**: Measures content overlap with retrieved answer
5. **Professional Tone**: Manager-ready quality

**Domain Filtering:**
- Prevents wrong-domain answers (Medicare for auto questions)
- Metadata filtering at Pinecone level
- Post-filtering for safety

**Fallback Strategy:**
- If RAG generation fails → LLM-only mode
- If LLM-only fails → Knowledge-based fallback
- **Never returns empty or wrong answers**

**Visual:** Quality control pipeline diagram

---

## **Slide 18: Results & Outcomes**

**Title:** Project Outcomes

**Content:**

**Functional Achievements:**
- ✅ Answers insurance questions accurately
- ✅ Uses verified knowledge base (21K+ Q&A pairs)
- ✅ Transparent (shows sources and scores)
- ✅ Handles edge cases gracefully
- ✅ Domain-aware (filters wrong-domain results)

**Technical Achievements:**
- ✅ Complete RAG pipeline implementation
- ✅ Domain classification and filtering
- ✅ Query caching (60-80% hit rate)
- ✅ CPU-optimized (meets 60s latency cap)
- ✅ Production-ready error handling

**Performance Metrics:**
- Dataset: 21,325 Q&A pairs indexed
- Embedding dimension: 384
- Retrieval: Top-3 with domain filtering
- Generation: 25-40s (CPU), 5-15s (GPU)
- Cache hit rate: 60-80% after warmup

**Visual:** Metrics dashboard or summary table

---

## **Slide 19: Challenges Overcome**

**Title:** Key Challenges & Solutions

**Content:**

**Challenge 1: Wrong-Domain Retrieval**
- **Problem**: Medicare answers for auto questions
- **Solution**: Domain classification + Pinecone metadata filtering

**Challenge 2: Low Similarity Threshold**
- **Problem**: Irrelevant chunks (similarity 0.15-0.20)
- **Solution**: Raised threshold to 0.35, LLM-only fallback

**Challenge 3: High Latency**
- **Problem**: 70-100 seconds per response
- **Solution**: Removed retries, reduced context, query caching

**Challenge 4: Generation Failures**
- **Problem**: Timeouts, wrong fallbacks
- **Solution**: Single attempt, LLM-only fallback (not raw dataset paste)

**Challenge 5: CPU Limitations**
- **Problem**: Slow inference on CPU
- **Solution**: Optimized context window, token limits, caching

**Visual:** Before/After comparison for each challenge

---

## **Slide 20: Technology Deep Dive**

**Title:** Why These Technologies?

**Content:**

**Pinecone:**
- Managed vector database (no infrastructure setup)
- Fast cosine similarity search
- Metadata filtering support
- Scalable (handles millions of vectors)

**Sentence Transformers:**
- State-of-the-art semantic embeddings
- Lightweight (384-dim vs 768-dim)
- Fast inference
- Good for insurance domain

**Ollama:**
- Easy local LLM deployment
- No API keys needed
- Supports multiple models
- CPU and GPU support

**Streamlit:**
- Rapid UI development
- Built-in chat components
- Easy deployment
- Professional appearance

**Visual:** Technology logos or architecture diagram

---

## **Slide 21: Future Enhancements**

**Title:** Future Work & Improvements

**Content:**

**Short-Term:**
- GPU acceleration (reduce latency to 5-15s)
- Hybrid search (keyword + vector)
- Multi-turn conversation support
- Advanced caching strategies

**Long-Term:**
- Fine-tune Mistral-7B on insurance domain
- Add more insurance datasets
- Real-time knowledge base updates
- Multi-language support
- Voice interface

**Scalability:**
- Deploy on cloud with GPU
- Horizontal scaling
- Load balancing
- Monitoring and analytics

**Visual:** Roadmap timeline

---

## **Slide 22: Conclusion**

**Title:** Summary & Key Takeaways

**Content:**

**What We Built:**
- Production-ready RAG chatbot for insurance Q&A
- 21K+ knowledge base with domain filtering
- Transparent, accurate, and faithful answers
- CPU-optimized for local deployment

**Key Innovations:**
- Domain-aware retrieval (prevents wrong-domain answers)
- Multi-layer quality controls
- Query enhancement for better retrieval
- Intelligent caching for performance

**Impact:**
- Demonstrates RAG architecture for domain-specific applications
- Shows CPU-optimization techniques
- Provides transparent, trustworthy AI system

**Thank You!**

**Visual:** Final architecture diagram or project logo

---

## **Slide 23: Q&A / Demo**

**Title:** Questions & Live Demo

**Content:**
- **Live Demo**: Show the chatbot in action
- **Questions**: Open for jury questions
- **Contact**: Your email/contact info

**Demo Script:**
1. Show Streamlit UI
2. Ask: "What is covered by Medigap?"
3. Show retrieval results and scores
4. Show generated answer
5. Show source references
6. Explain latency (CPU inference)

---

## **Presentation Tips:**

1. **Slide 1-3**: Set context (2-3 minutes)
2. **Slide 4-8**: Architecture & tech stack (5-7 minutes)
3. **Slide 9-12**: Pipeline details (7-10 minutes)
4. **Slide 13-15**: UI & CPU challenges (5-7 minutes)
5. **Slide 16-18**: How it works & results (5-7 minutes)
6. **Slide 19-22**: Challenges, future work, conclusion (3-5 minutes)
7. **Slide 23**: Demo & Q&A (5-10 minutes)

**Total Time: 30-40 minutes**

**Key Points to Emphasize:**
- RAG architecture and why it's better than pure LLM
- Domain filtering to prevent wrong answers
- CPU challenges and how you optimized
- Transparency (showing sources and scores)
- Production-ready quality controls

