# 📍 Complete Pinecone Setup Guide

This guide explains everything about using Pinecone for this Insurance Q&A Chatbot project, from scratch.

## What is Pinecone?

Pinecone is a **vector database** designed for similarity search. It stores high-dimensional vectors (embeddings) and allows you to quickly find the most similar vectors to a query vector.

### Why Use Pinecone?

1. **Fast Similarity Search**: Sub-second query times even with millions of vectors
2. **Fully Managed**: No infrastructure to maintain
3. **Scalable**: Handles large datasets efficiently
4. **Free Tier**: Generous free tier for learning/development
5. **Easy Integration**: Simple Python API

## How Pinecone Works in Our Project

```
Text → Embedding Model → Vector (384 dimensions) → Pinecone → Similar Vectors
```

### Our Use Case:

1. **Storage**: Store 12,000+ insurance Q&A pairs as vectors
2. **Retrieval**: When user asks a question, find similar Q&A pairs
3. **Context**: Use retrieved Q&A pairs to generate accurate answers

## Step-by-Step Pinecone Setup

### 1. Create Pinecone Account

1. Go to [https://app.pinecone.io/](https://app.pinecone.io/)
2. Click "Sign Up"
3. Choose "Free" plan (no credit card required)
4. Verify your email

**Free Tier Includes:**
- 1 project
- 1 index
- 100K vectors (we need ~12K)
- 5GB storage
- Perfect for this project!

### 2. Get API Key

1. Log in to Pinecone dashboard
2. Click "API Keys" in the left sidebar
3. Copy your API key (looks like: `abc123-def456-ghi789`)
4. Keep it secure (don't commit to Git!)

### 3. Choose Environment

Pinecone offers several cloud regions:

- **us-east-1-aws** (recommended for this project)
- us-west-2-aws
- eu-west-1-aws
- And more...

**For Free Tier**: Use `us-east-1-aws`

### 4. Configure in Your Project

Create a `.env` file:

```env
PINECONE_API_KEY=your_actual_api_key_here
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=insurance-qa-index
```

## Understanding Pinecone Concepts

### Index

An **index** is like a database table for vectors.

**Our Index:**
- Name: `insurance-qa-index`
- Dimension: 384 (matches our embedding model)
- Metric: Cosine similarity
- Type: Serverless

### Vectors

Each vector has:
- **ID**: Unique identifier (e.g., `qa_0_0`)
- **Values**: 384-dimensional array of floats
- **Metadata**: Additional information (question, answer, source)

**Example:**
```python
{
    'id': 'qa_123_0',
    'values': [0.123, -0.456, 0.789, ...],  # 384 numbers
    'metadata': {
        'question': 'What is a deductible?',
        'answer': 'A deductible is the amount...',
        'source': 'insuranceQA-v2'
    }
}
```

### Similarity Metrics

- **Cosine**: Measures angle between vectors (we use this)
- **Euclidean**: Measures distance
- **Dot Product**: Measures magnitude and direction

**Cosine Similarity:**
- Range: -1 to 1
- 1 = identical
- 0 = orthogonal (unrelated)
- -1 = opposite

## Our Pinecone Workflow

### Phase 1: Setup (One Time)

```python
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone
pc = Pinecone(api_key="your_api_key")

# Create index
pc.create_index(
    name="insurance-qa-index",
    dimension=384,
    metric='cosine',
    spec=ServerlessSpec(
        cloud='aws',
        region='us-east-1-aws'
    )
)

# Connect to index
index = pc.Index("insurance-qa-index")
```

### Phase 2: Data Ingestion

```python
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Process each Q&A pair
for item in data:
    # Combine question and answer
    text = f"Question: {item['question']}\nAnswer: {item['answer']}"
    
    # Generate embedding
    embedding = model.encode(text)  # Returns 384-dim vector
    
    # Prepare vector with metadata
    vector = {
        'id': item['id'],
        'values': embedding.tolist(),
        'metadata': {
            'question': item['question'],
            'answer': item['answer'],
            'source': 'insuranceQA-v2'
        }
    }
    
    # Upsert to Pinecone
    index.upsert(vectors=[vector])
```

**Batch Processing** (more efficient):
```python
vectors = []
for item in data[:100]:  # Process 100 at a time
    # ... generate embedding ...
    vectors.append(vector)

# Upsert batch
index.upsert(vectors=vectors)
```

### Phase 3: Query Time

```python
# User asks a question
user_question = "What is a deductible?"

# Generate embedding for question
query_embedding = model.encode(user_question)

# Search Pinecone
results = index.query(
    vector=query_embedding.tolist(),
    top_k=5,  # Get top 5 similar results
    include_metadata=True
)

# Process results
for match in results['matches']:
    print(f"Score: {match['score']}")
    print(f"Question: {match['metadata']['question']}")
    print(f"Answer: {match['metadata']['answer']}")
    print("---")
```

## Code Walkthrough

### 1. PineconeManager Class

```python
class PineconeManager:
    def __init__(self):
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index_name = config.PINECONE_INDEX_NAME
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
```

### 2. Creating Index

```python
def create_index(self):
    # Check if index exists
    existing_indexes = self.pc.list_indexes()
    index_names = [index.name for index in existing_indexes]
    
    if self.index_name not in index_names:
        # Create new index
        self.pc.create_index(
            name=self.index_name,
            dimension=config.EMBEDDING_DIMENSION,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region=config.PINECONE_ENVIRONMENT
            )
        )
    
    # Connect to index
    self.index = self.pc.Index(self.index_name)
```

### 3. Generating Embeddings

```python
def generate_embeddings(self, texts, batch_size=32):
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Encode batch
        batch_embeddings = self.embedding_model.encode(
            batch,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        embeddings.extend(batch_embeddings.tolist())
    
    return embeddings
```

### 4. Upserting Vectors

```python
def upsert_vectors(self, data, batch_size=100):
    # Get all texts
    texts = [item['combined_text'] for item in data]
    
    # Generate embeddings
    embeddings = self.generate_embeddings(texts)
    
    # Prepare vectors
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
    
    # Upsert in batches
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        self.index.upsert(vectors=batch)
```

### 5. Searching

```python
def search_similar(self, query, top_k=5):
    # Generate query embedding
    query_embedding = self.embedding_model.encode([query])[0].tolist()
    
    # Query Pinecone
    results = self.index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # Format results
    similar_docs = []
    for match in results['matches']:
        if match['score'] >= config.SIMILARITY_THRESHOLD:
            similar_docs.append({
                'id': match['id'],
                'score': match['score'],
                'question': match['metadata']['question'],
                'answer': match['metadata']['answer']
            })
    
    return similar_docs
```

## Configuration Options

### config.py Settings

```python
# Embedding Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Retrieval Configuration
TOP_K_RESULTS = 5  # Number of similar docs to retrieve
SIMILARITY_THRESHOLD = 0.5  # Minimum similarity score
```

**Tuning Tips:**

1. **TOP_K_RESULTS**:
   - Lower (3): Faster, more focused
   - Higher (10): More context, slower

2. **SIMILARITY_THRESHOLD**:
   - Lower (0.3): More results, less relevant
   - Higher (0.7): Fewer results, more relevant

## Performance Optimization

### 1. Batch Processing

```python
# Bad: One at a time
for item in data:
    embedding = model.encode(item['text'])
    index.upsert(vectors=[create_vector(item, embedding)])

# Good: Batches
batch_size = 100
for i in range(0, len(data), batch_size):
    batch = data[i:i + batch_size]
    embeddings = model.encode([item['text'] for item in batch])
    vectors = [create_vector(item, emb) for item, emb in zip(batch, embeddings)]
    index.upsert(vectors=vectors)
```

### 2. Metadata Limits

Pinecone has metadata size limits:

```python
# Truncate long text
'question': item['question'][:1000],  # Max 1000 chars
'answer': item['answer'][:1000]
```

### 3. Caching

The embedding model caches results:

```python
# First query: slow (model loading)
results1 = search("What is insurance?")

# Subsequent queries: fast (model cached)
results2 = search("What is a deductible?")
```

## Monitoring & Debugging

### Check Index Stats

```python
stats = index.describe_index_stats()
print(stats)

# Output:
# {
#     'dimension': 384,
#     'index_fullness': 0.12,
#     'namespaces': {'': {'vector_count': 12000}},
#     'total_vector_count': 12000
# }
```

### List All Indexes

```python
indexes = pc.list_indexes()
for idx in indexes:
    print(f"Name: {idx.name}")
    print(f"Dimension: {idx.dimension}")
    print(f"Metric: {idx.metric}")
```

### Test Query

```python
# Simple test
results = index.query(
    vector=[0.1] * 384,  # Dummy vector
    top_k=1,
    include_metadata=True
)
print(f"Found {len(results['matches'])} results")
```

## Common Issues & Solutions

### Issue 1: "API key not found"

**Solution:**
```bash
# Check .env file exists
ls -la .env

# Check content
cat .env

# Should see:
# PINECONE_API_KEY=your_key_here
```

### Issue 2: "Index not found"

**Solution:**
```python
# List all indexes
pc.list_indexes()

# Create if missing
python pinecone_setup.py
```

### Issue 3: "Dimension mismatch"

**Solution:**
- Embedding model outputs 384 dimensions
- Index must be created with dimension=384
- Delete and recreate index if wrong dimension

### Issue 4: "No results found"

**Causes:**
1. Index is empty (run `pinecone_setup.py`)
2. Threshold too high (lower `SIMILARITY_THRESHOLD`)
3. Query is too different from stored data

**Debug:**
```python
# Check vector count
stats = index.describe_index_stats()
print(f"Vectors: {stats['total_vector_count']}")

# Try lower threshold
results = search(query, threshold=0.3)
```

## Cost Considerations

### Free Tier:
- ✅ Sufficient for this project
- ✅ 100K vectors (we use ~12K)
- ✅ Serverless (pay only for usage)

### If Scaling:
- Queries: $0.001 per 10K queries
- Storage: Included in free tier
- No surprise charges

## Alternative to Pinecone (Optional)

If you can't use Pinecone, consider:

1. **FAISS** (Facebook AI Similarity Search)
   - Free, local
   - Requires more setup
   - No cloud infrastructure

2. **Weaviate**
   - Open-source vector database
   - Self-hosted or cloud

3. **ChromaDB**
   - Simple, lightweight
   - Good for small projects

## Testing Your Pinecone Setup

```bash
# Test the complete setup
python -c "
from pinecone_setup import PineconeManager
manager = PineconeManager()
manager.create_index()
results = manager.search_similar('What is insurance?', top_k=3)
print(f'Found {len(results)} results')
for r in results:
    print(f'  Score: {r[\"score\"]:.3f}')
"
```

## Summary

**Pinecone Workflow:**
1. **Setup**: Create account, get API key
2. **Initialize**: Create index with correct dimensions
3. **Ingest**: Generate embeddings and upsert vectors
4. **Query**: Search for similar vectors at runtime
5. **Retrieve**: Get metadata of similar items
6. **Generate**: Use retrieved context with LLM

**Key Advantages:**
- No infrastructure management
- Fast similarity search
- Easy Python integration
- Free for development
- Scales automatically

**In Our Project:**
- Stores 12,000 insurance Q&A pairs
- Returns top 5 similar pairs per query
- Takes 5-10 minutes to set up initially
- Query time: < 1 second

---

## Quick Reference Commands

```bash
# Setup Pinecone database (one time)
python pinecone_setup.py

# Test Pinecone connection
python -c "from pinecone_setup import PineconeManager; m = PineconeManager(); m.create_index(); print('OK')"

# Check index stats
python -c "from pinecone_setup import PineconeManager; m = PineconeManager(); m.create_index(); print(m.index.describe_index_stats())"

# Test search
python -c "from pinecone_setup import PineconeManager; m = PineconeManager(); m.create_index(); print(m.search_similar('insurance', 3))"
```

---

**You're now a Pinecone expert! 🎉**

For more details, visit [Pinecone Documentation](https://docs.pinecone.io/)


