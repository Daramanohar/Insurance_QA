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


