# 📘 Insurance Q&A RAG Chatbot – Full Project Report

This document provides a **single, end‑to‑end report** of the Insurance_QA project so that any stakeholder (e.g., a manager, engineer, or reviewer) can understand the entire system: architecture, pipeline, quality controls, performance characteristics, and how to run and extend it.

---

## 1. Project Overview

The **Insurance_QA** project is an AI‑powered **Insurance Question‑Answering Chatbot** built with a **Retrieval‑Augmented Generation (RAG)** architecture. It is designed to answer health, auto, life, and general insurance questions using a curated knowledge base and a local, open‑source large language model (LLM).

**Core goals:**

- Deliver **accurate, faithful, and natural‑sounding** answers to insurance questions.
- Use a **RAG pipeline** where **retrieval comes first** and the knowledge base is treated as the **source of truth**.
- Fall back to the model’s domain knowledge **only when retrieval truly fails**.
- Ensure **high quality**, **no hallucinations**, and **English‑only**, manager‑ready responses.
- Run on **local hardware** (CPU) using **Mistral‑7B via Ollama**, with clear performance expectations.

The complete source code and documentation are available in the GitHub repository:  
`https://github.com/Daramanohar/Insurance_QA`  

---

## 2. Architecture & Technology Stack

The system is composed of five main layers:

1. **User Interface (UI)** – Built with **Streamlit** (`app_final.py` / `app_pro.py`), providing a professional chat interface with metrics, status indicators, and transparency features.
2. **Query Processing & Embeddings** – Uses **Sentence Transformers** (`all‑MiniLM‑L6‑v2`, 384‑dim) to convert user questions and Q&A pairs into dense vector representations.
3. **Retrieval Layer** – Uses **Pinecone** as a vector database for fast **cosine similarity** search over ~21K InsuranceQA Q&A vectors.
4. **RAG Engine** – Implemented in `optimized_rag_engine.py`, orchestrating retrieval, mode selection (RAG vs LLM‑only), answer synthesis, validation, caching, and performance logging.
5. **Language Model** – **Mistral‑7B‑Instruct** served via **Ollama** (`ollama_client.py`) for natural‑language answer generation.

### High‑Level Flow

```text
User Question
    ↓
[cache_manager] → Check for exact/near‑duplicate answers
    ↓
[SentenceTransformer] → Encode question (384‑dim embedding)
    ↓
[Pinecone] → top_k=3 semantic search (+ lexical fallback)
    ↓
[RAG Engine] → Decide RAG vs LLM‑only mode
    ↓
[Answer Generation] → RAG‑grounded or LLM‑only
    ↓
[Validation] → English‑only, non‑hallucination, factual & professional
    ↓
[Cache Store] → Save answer for future reuse
    ↓
UI → Answer + Similarity + Faithfulness + Sources + Metrics
```

---

## 3. Data Ingestion & Knowledge Base

**Module:** `data_loader.py` + `pinecone_setup.py` / `pinecone_setup_simple.py`

### Dataset: InsuranceQA‑v2

- Source: [Hugging Face – insuranceqa](https://huggingface.co/datasets/insuranceqa)  
- Size: ~12K–20K Q&A pairs  
- Each record includes:
  - `question` – the user’s question text  
  - `answers` – candidate answers  
  - `ground_truth` – index of the correct answer(s)  

### Processing Steps

1. Load the dataset using `datasets.load_dataset`.
2. For each item:
   - Extract the primary question.
   - Select the correct answer using `ground_truth` index.
   - Build a `combined_text`:
     ```text
     Question: {question}
     
     Answer: {answer}
     ```
   - Attach metadata: `{"source": "insuranceQA-v2", "original_idx": idx}`.
3. Return a list of processed Q&A entries:
   ```python
   {
     "id": f"qa_{idx}_{ans_idx}",
     "question": question,
     "answer": answer,
     "combined_text": combined_text,
     "metadata": {...}
   }
   ```

### Vector Indexing in Pinecone

**Module:** `pinecone_setup.py`

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384‑dim)
- Creates a Pinecone index (`insurance-qa-index`) with:
  - Metric: **cosine**
  - Dimension: **384**
  - On‑demand capacity mode
- Upserts each `combined_text` as a vector with:
  - `id = "qa_{idx}_{ans_idx}"`
  - `values = embedding`
  - `metadata = {question, answer, source, original_idx}`

`pinecone_setup_simple.py` provides a streamlined version for testing without full PyTorch overhead.

---

## 4. Retrieval & RAG Engine

**Module:** `optimized_rag_engine.py`

The RAG engine is responsible for **retrieval‑first** logic, **mode selection**, **answer synthesis**, **quality validation**, and **performance tracking**.

### 4.1 Retrieval‑First Policy (Mandatory)

For **every** user question:

1. Run **vector search** against Pinecone with `top_k = 3` and `include_metadata=True`.
2. If **any** result has `score ≥ 0.15`, retrieval is considered **meaningful** and **RAG mode must be used**.
3. Retrieval is **never skipped** based on heuristics (e.g., “conceptual” detection). This enforces the rule: **Retrieval comes first**.
4. If no match meets the threshold, a **lexical fallback** is attempted using Jaccard similarity on question text.
5. Only if both vector and lexical retrieval fail does the engine switch to **LLM‑only mode**, with `retrieval_skipped_reason = "no_retrieval"`.

### 4.2 Query Deduplication & Caching

**Module:** `cache_manager.py`

- **Normalization** – lowercasing, trimming whitespace, and removing filler words so that variants like:  
  - `"What is a deductible?"`  
  - `"what is a deductible"`  
  - `"Explain deductible"`  
  map to comparable forms.
- **Exact match caching** – uses MD5 hashing of the normalized query string as a key; subsequent identical questions return in ~0.01s.
- **Near‑duplicate detection** – uses **Jaccard similarity** between token sets to reuse answers when similarity ≥ a configurable threshold (for example, 0.9).
- **Metadata tracking** – stores cached answers along with retrieved context and timing, enabling simple analytics (e.g., cache hit rate, most frequent questions).

This dramatically reduces latency and compute for frequent or similar questions, which is especially important when running Mistral‑7B on CPU.

### 4.3 RAG Answer Generation (When Context Exists)

**Function:** `_generate_from_retrieved_content()` in `optimized_rag_engine.py`

When `context_docs` is non‑empty and similarity ≥ threshold:

1. **Select best match** from `context_docs[0]` (highest similarity).
2. Build a structured **RAG prompt**:
   - Includes the **original question**.
   - Includes the **retrieved answer** labeled as the **AUTHORITATIVE source of truth**.
   - Adds **critical rules**, for example:
     - Do **not** contradict the retrieved answer.
     - Do **not** add new insurer/brand/plan names or personal opinions.
     - Preserve all **conditions, decision logic, lists, and numeric details**.
     - Do **not** collapse structured lists into a single vague sentence.
   - Instructs the model to **rewrite/explain** the retrieved answer in clear, user‑friendly English, without copy‑pasting long sentences.
3. Calls **Ollama** with `mistral:7b-instruct` using conservative generation parameters:
   - `num_ctx = 2048`
   - `num_predict = 300`
   - `temperature = 0.2`
   - `top_k = 15`, `top_p = 0.7`, `repeat_penalty = 1.15`
   - `MAX_GENERATION_TIME = 30s` and at most **one retry** (`MAX_RETRIES = 1`).
4. After generation:
   - Cleans and normalizes the answer (`_clean_answer()`).
   - Runs **faithfulness validation** (`_validate_answer_faithfulness()`).
   - Computes a **faithfulness score** via `_compute_faithfulness_score()` (content‑word overlap between retrieved answer and generated explanation).
   - If `score < 0.8` or validation flags contradictions/new entities/missing conditions:
     - The engine retries once with stricter instructions.
5. **Final fallback when validation keeps failing**:
   - Makes one more “simplified paraphrase” call:  
     > “Explain this answer in clear, simple English. Keep all conditions and numbers. Do not add new companies/plans or opinions.”
   - If that also fails (e.g., due to timeout), returns the **truncated retrieved answer** itself as a last resort (never an empty response).

The RAG pipeline therefore ensures every answer is either:

- A **validated, natural explanation** of the KB answer, or  
- The **KB answer itself** (when model generation is unavailable), never a hallucination.

### 4.4 LLM‑Only Mode (When Retrieval Fails)

If both vector and lexical retrieval return no meaningful results:

- The engine sets `mode = "llm_only"` and `retrieval_skipped_reason = "no_retrieval"`.
- `_generate_from_model_knowledge()` is invoked with a **domain‑focused prompt**:
  - “You are an insurance expert. Answer this question using general insurance knowledge…”
  - `intent_hint` (definition / comparison / conditional / concept) shapes the structure.
- Uses similar conservative generation parameters:
  - `num_ctx = 2048`, `num_predict = 300`, `temperature = 0.4`, `top_k = 30`, `top_p = 0.9`.
- A single retry is allowed; if both attempts fail validation, the engine returns a **safe, generic but correct** insurance explanation from `_get_knowledge_based_fallback()`.

The mode (RAG vs LLM‑only vs exact‑match) is surfaced in the UI so users always know how the answer was produced.

### 4.5 Validation & Quality Controls

**Module:** `answer_validator.py` + logic in `optimized_rag_engine.py`

The project implements a robust, multi‑stage validation pipeline to guarantee **English‑only, factually aligned, professional** answers:

1. **Language Check** – `_check_english_only()`:
   - Blocks non‑English characters, Tagalog/Spanish keywords, and mixed‑language output.
2. **Hallucination & Uncertainty Check**:
   - Flags phrases like “based on the documents” or speculative language.
3. **Factual Rule Check** – `_check_factual_violations()`:
   - Pattern‑based rules to prevent known bad claims (e.g., “mileage always cancels NCB”).
4. **Professionalism & Length**:
   - Enforces 50–2000 character range.
   - Disallows all‑caps shouting, slang, or informal tone.
   - Requires a clear, manager‑ready structure (direct answer → explanation → details → conclusion).

If any check fails:

- The engine retries once with tighter parameters and stricter prompts.
- If all attempts fail, a **safe fallback answer** is returned that is correct but conservative, rather than hallucinated.

The **faithfulness score** is also exposed in the UI so stakeholders can see how closely the answer matches the retrieved ground truth.

---

## 5. User Interface & Transparency

**Module:** `app_final.py` (and enhanced `app_pro.py`)

The UI is built with **Streamlit** and is designed to be both **user‑friendly** and **transparent**:

- **Chat Interface** – Uses `st.chat_message` for user and assistant turns.
- **Initialization Flow** – “Initialize Assistant” button sets up `PineconeManager`, `OllamaClient`, and `OptimizedRAGEngine`.
- **System Info Panel** – Shows:
  - LLM Model: `mistral:7b-instruct`
  - Knowledge Base: InsuranceQA‑v2 (21,325 Q&A)
  - Search Engine: Pinecone (cosine similarity)
  - Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
  - Mode: “Stateless Q&A (no cross-question memory)”
- **Performance Notice** – Clearly explains CPU inference behavior:
  - Running Mistral‑7B on CPU via Ollama
  - RAG responses may take **25–40s** on CPU due to large KV cache & attention computation
  - With GPU + quantization, responses typically drop to **5–15s**
- **Per‑Response Metrics & Transparency**:
  - Mode badge: `RAG`, `LLM-only`, or `Exact match`
  - Response time (total & retrieval time)
  - **Top‑1 similarity score** (e.g., `🔍 Top‑1 Sim: 82%`)
  - **Faithfulness score** (0–1 or %)
  - **Pinecone vector IDs** and **retrieved Q&A snippets** in a collapsible “Retrieved Context” panel

These elements make it clear **how** each answer was produced and how confident the system is, which is critical for enterprise and insurance use cases.

---

## 6. Performance & CPU Inference Characteristics

Because the system runs **Mistral‑7B‑Instruct** on CPU via **Ollama**, performance characteristics are carefully managed and documented (see also `README.md` and `OPTIMIZATION_SUMMARY.md`):

- **Context Window**: up to 4096 tokens (configured to 2048 for CPU efficiency).
- **Generation Limits**: `MAX_TOKENS_RAG = 300`, `MAX_TOKENS_LLM_ONLY = 300`.
- **Retrieval Timings**:
  - Pinecone search: typically **0.5–1s** for `top_k=3`.
  - Cache hits: ~**0.01s**.
- **Latency Budgets**:
  - LLM‑only / simple queries: **10–15s** on CPU.
  - RAG / complex queries: **25–40s** on CPU (due to large prompts and attention computation).
  - Hard caps:
    - `MAX_RETRIEVAL_TIME = 2s`
    - `MAX_GENERATION_TIME = 30s`
    - `MAX_TOTAL_TIME = 40s` (never exceed 60s)
- **Caching Impact**:
  - Repeated or similar queries are served from cache in **<0.1s**, dramatically improving UX and reducing compute.

The system explicitly prioritizes **accuracy and faithfulness** over minimal latency, which is appropriate for insurance Q&A where correctness matters more than raw speed.

---

## 7. Setup, Testing & Tooling

### 7.1 Environment & Setup

Key steps (see `README.md` and `QUICKSTART.md` for full details):

1. Install **Python 3.8+** and create a virtual environment (`.venv`).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment variables via `env_template.txt` → `.env`:
   ```env
   PINECONE_API_KEY=your_pinecone_key
   PINECONE_ENVIRONMENT=us-east-1
   PINECONE_INDEX_NAME=insurance-qa-index
   OLLAMA_MODEL=mistral:7b-instruct
   OLLAMA_BASE_URL=http://localhost:11434
   ```
4. Install and start **Ollama**, then pull the model:
   ```bash
   ollama pull mistral
   ```
5. Run `python pinecone_setup.py` once to populate the index.
6. Launch the UI:
   ```bash
   streamlit run app_final.py
   ```

### 7.2 Testing & Verification

The repo includes multiple scripts to validate correct behavior:

- `test_system.py` – end‑to‑end smoke tests (Ollama + Pinecone + RAG).
- `test_pinecone.py`, `check_pinecone_status.py` – verify index connectivity and vector counts.
- `test_retrieval.py` – exercises retrieval correctness for known insurance questions.
- `test_inference.py` – tests LLM + RAG generation on exemplar queries.
- `check_system.py` – validates local environment (Python version, PyTorch DLLs, Ollama availability).

These tools make it easy to confirm everything is configured correctly before a demo or deployment.

---

## 8. Extensibility & Customization

The project is designed for easy extension:

- **Swap or add datasets**:
  - Modify `data_loader.py` to point to a new CSV/JSON or Hugging Face dataset.
  - Regenerate embeddings and re‑run `pinecone_setup.py`.
- **Change embedding model**:
  - Update `EMBEDDING_MODEL` and `EMBEDDING_DIMENSION` in `config.py`.
- **Change LLM**:
  - Edit `OLLAMA_MODEL` in `.env` to any other Ollama‑available model (e.g., `llama2:7b`) while preserving the same RAG pipeline.
- **Tune RAG behavior**:
  - Adjust `TOP_K`, `RELEVANCE_THRESHOLD`, `MAX_TOKENS_*`, and validation thresholds in `optimized_rag_engine.py`.
- **Extend UI**:
  - Add new tabs, filters, or admin views in `app_final.py` / `app_pro.py`.
  - Integrate additional dashboards (e.g., usage analytics, error logs).

Because each concern is encapsulated in its own module (data loading, retrieval, LLM, validation, UI), the codebase is straightforward to adapt to other domains beyond insurance.

---

## 9. Example End‑to‑End Walkthrough

Consider the question:  
> “Does filing a claim always affect my No‑Claim Bonus?”

1. **User asks the question** in the Streamlit UI.
2. `process_query()` normalizes the text and checks the cache. No hit on first run.
3. The query is **embedded** via `SentenceTransformer` and sent to **Pinecone** with `top_k=3`.
4. Pinecone returns a high‑similarity Q&A about No‑Claim Bonus and claims impact → **RAG mode is selected**.
5. The retrieved answer (from InsuranceQA) is passed into `_generate_from_retrieved_content()` along with the user’s question.
6. The LLM is instructed to **explain exactly what the retrieved answer says**, preserving conditions (e.g., when NCB is lost, when it is retained) and avoiding new insurer names or speculative advice.
7. The answer is generated, validated for English‑only language, lack of hallucinations, factual alignment, and professional tone.
8. A **faithfulness score** (e.g., 0.92) is computed and displayed alongside the answer.
9. The UI shows:
   - **Mode**: RAG
   - **Top‑1 similarity** (e.g., 88%)
   - **Faithfulness** (e.g., 92%)
   - **Pinecone ID** and **retrieved Q&A snippet** in the “Retrieved Context” panel
   - Total response time and whether the SLA (<40s) was met
10. The answer is cached, so similar follow‑up questions can be answered in milliseconds without re‑querying the LLM.

---

## 10. Conclusion

The **Insurance_QA RAG Chatbot** is a robust, production‑grade system that demonstrates how to build a **trustworthy, transparent, and performant** domain‑specific assistant:

- Retrieval‑first design that always prioritizes the **knowledge base**  
- Strong **quality controls** to eliminate non‑English output and minimize hallucinations  
- A carefully tuned **CPU‑friendly** RAG pipeline with caching and latency safeguards  
- A professional, transparent **Streamlit UI** with performance and provenance metrics  
- A modular, well‑documented codebase that is easy to extend to new insurance products or entirely different domains  

By following the included documentation (`README.md`, `FULL_PROJECT_REPORT.md`, `PINECONE_GUIDE.md`, `QUALITY_CONTROLS.md`, `OPTIMIZATION_SUMMARY.md`, etc.), a reviewer or manager can fully understand how the system works, how it meets the assignment requirements, and how it can be deployed or extended in a real‑world insurance environment.
