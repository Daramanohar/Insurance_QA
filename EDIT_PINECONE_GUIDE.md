# How to Edit Your Pinecone Setup

## Common Editing Operations

### 1. Change Which Data Gets Stored

**Edit `data_loader.py`** to filter or modify data before storing:

```python
# In data_loader.py, modify process_dataset() method

# Example: Only store health insurance questions
if 'health' in question.lower():
    self.processed_data.append({...})

# Example: Only store first 1000 items
if len(self.processed_data) >= 1000:
    break
```

---

### 2. Update Existing Vectors

**Option A: Update specific vectors**

```python
from pinecone_setup import PineconeManager

manager = PineconeManager()
manager.create_index()

# Update a specific vector
manager.index.upsert(
    vectors=[{
        'id': 'qa_2042_0',
        'values': [0.1, 0.2, ...],  # New embedding
        'metadata': {
            'question': 'Updated question',
            'answer': 'Updated answer',
            'source': 'updated'
        }
    }],
    namespace=""  # or your namespace
)
```

**Option B: Re-run setup to update all**

```bash
# This will re-upload all vectors (overwrites existing with same IDs)
python pinecone_setup.py
```

---

### 3. Delete Vectors

**Delete specific vectors:**

```python
from pinecone_setup import PineconeManager

manager = PineconeManager()
manager.create_index()

# Delete by ID
manager.index.delete(ids=['qa_2042_0', 'qa_4705_0'])

# Delete by filter (if you have indexed metadata)
manager.index.delete(filter={'source': 'old_data'})

# Delete ALL vectors in namespace (careful!)
manager.index.delete(delete_all=True, namespace="")
```

**Delete all and start fresh:**

```bash
# Option 1: Delete the entire index
python -c "from pinecone_setup import PineconeManager; m = PineconeManager(); m.delete_index()"

# Then re-run setup
python pinecone_setup.py
```

---

### 4. Add New Data

**Option A: Add to existing dataset**

1. Edit `data_loader.py` to include new data source
2. Run setup again:
   ```bash
   python pinecone_setup.py
   ```

**Option B: Manually add vectors**

```python
from pinecone_setup import PineconeManager
from sentence_transformers import SentenceTransformer

manager = PineconeManager()
manager.create_index()

# Load embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Create new Q&A pair
new_qa = {
    'question': 'What is term life insurance?',
    'answer': 'Term life insurance provides coverage for a specific period...'
}

# Generate embedding
text = f"Question: {new_qa['question']}\nAnswer: {new_qa['answer']}"
embedding = model.encode(text).tolist()

# Add to Pinecone
manager.index.upsert(
    vectors=[{
        'id': 'custom_001',
        'values': embedding,
        'metadata': new_qa
    }]
)
```

---

### 5. Change Embedding Model

**Edit `config.py`:**

```python
# Current
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Change to different model
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIMENSION = 768  # Update dimension!

# Note: You'll need to delete index and recreate with new dimension
```

**Then:**
1. Delete old index
2. Re-run `python pinecone_setup.py`

---

### 6. Change Dataset Source

**Edit `config.py`:**

```python
# Change dataset
DATASET_NAME = "your-new-dataset"
DATASET_SPLIT = "train"
```

**Or edit `data_loader.py` to use custom data:**

```python
# Add method to load from CSV
def load_from_csv(self, filepath: str):
    df = pd.read_csv(filepath)
    for _, row in df.iterrows():
        self.processed_data.append({
            'id': f"custom_{row['id']}",
            'question': row['question'],
            'answer': row['answer'],
            'combined_text': f"Question: {row['question']}\nAnswer: {row['answer']}",
            'metadata': {'source': 'custom'}
        })
```

---

### 7. Modify Vector Metadata

**Update what gets stored with vectors:**

Edit `pinecone_setup.py`, find the `upsert_vectors` method:

```python
# Current metadata
'metadata': {
    'question': item['question'][:1000],
    'answer': item['answer'][:1000],
    'source': item['metadata']['source']
}

# Add more fields
'metadata': {
    'question': item['question'][:1000],
    'answer': item['answer'][:1000],
    'source': item['metadata']['source'],
    'category': 'health',  # Add category
    'confidence': 0.95,    # Add confidence score
    'date_added': '2024-01-01'  # Add timestamp
}
```

---

### 8. Change Batch Size

**Edit `pinecone_setup.py`:**

```python
# In upsert_vectors method
def upsert_vectors(self, data: List[Dict], batch_size: int = 100):
    # Change default batch_size
    # Larger = faster but more memory
    # Smaller = slower but more stable
```

Or when calling:
```python
manager.upsert_vectors(data, batch_size=50)  # Smaller batches
```

---

### 9. Filter Data Before Upload

**Edit `pinecone_setup.py` main function:**

```python
def setup_pinecone_database():
    # ... existing code ...
    
    # Filter data before uploading
    data = loader.process_dataset()
    
    # Only upload health insurance questions
    filtered_data = [
        item for item in data 
        if 'health' in item['question'].lower()
    ]
    
    manager.upsert_vectors(filtered_data)
```

---

### 10. Change Similarity Threshold

**Edit `config.py`:**

```python
# How similar results need to be
SIMILARITY_THRESHOLD = 0.5  # Current

# More strict (only very similar results)
SIMILARITY_THRESHOLD = 0.7

# Less strict (more results)
SIMILARITY_THRESHOLD = 0.3
```

---

## Quick Edit Scripts

### Script: Delete All Vectors

```python
# delete_all_vectors.py
from pinecone_setup import PineconeManager

manager = PineconeManager()
manager.create_index()

print("Deleting all vectors...")
manager.index.delete(delete_all=True, namespace="")
print("Done!")
```

### Script: Update Single Vector

```python
# update_vector.py
from pinecone_setup import PineconeManager
from sentence_transformers import SentenceTransformer

manager = PineconeManager()
manager.create_index()
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# New data
new_data = {
    'question': 'What is a deductible?',
    'answer': 'UPDATED: A deductible is the amount you pay...'
}

# Generate embedding
text = f"Question: {new_data['question']}\nAnswer: {new_data['answer']}"
embedding = model.encode(text).tolist()

# Update
manager.index.upsert(
    vectors=[{
        'id': 'qa_2042_0',  # Existing ID to update
        'values': embedding,
        'metadata': new_data
    }]
)
print("Vector updated!")
```

### Script: Add Single Q&A

```python
# add_single_qa.py
from pinecone_setup import PineconeManager
from sentence_transformers import SentenceTransformer

manager = PineconeManager()
manager.create_index()
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Your new Q&A
question = input("Enter question: ")
answer = input("Enter answer: ")

# Generate embedding
text = f"Question: {question}\nAnswer: {answer}"
embedding = model.encode(text).tolist()

# Add to Pinecone
import time
vector_id = f"custom_{int(time.time())}"

manager.index.upsert(
    vectors=[{
        'id': vector_id,
        'values': embedding,
        'metadata': {
            'question': question,
            'answer': answer,
            'source': 'manual'
        }
    }]
)
print(f"Added! ID: {vector_id}")
```

---

## Common Workflows

### Workflow 1: Update All Data

```bash
# 1. Backup if needed (optional)
python -c "from pinecone_setup import PineconeManager; m = PineconeManager(); m.create_index(); print(m.index.describe_index_stats())"

# 2. Modify data_loader.py or config.py as needed

# 3. Re-run setup (overwrites existing)
python pinecone_setup.py

# 4. Test
python test_pinecone.py
```

### Workflow 2: Add New Data

```bash
# 1. Create new data file (CSV, JSON, etc.)

# 2. Modify data_loader.py to load new file

# 3. Run setup (adds to existing)
python pinecone_setup.py

# 4. Verify
python list_vector_ids.py
```

### Workflow 3: Clean Start

```bash
# 1. Delete everything
python -c "from pinecone_setup import PineconeManager; m = PineconeManager(); m.delete_index()"

# 2. Re-run setup
python pinecone_setup.py
```

---

## Safety Tips

1. **Always test first**: Use a test namespace
2. **Backup important data**: Export vectors before major changes
3. **Check vector counts**: Before and after changes
4. **Use version control**: Git commit before major edits
5. **Document changes**: Note what you changed and why

---

## Need Help?

Run these to check your current setup:

```bash
# Check configuration
python check_namespace.py

# List some vectors
python list_vector_ids.py

# Test search
python test_pinecone.py

# Check index stats
python -c "from pinecone_setup import PineconeManager; m = PineconeManager(); m.create_index(); print(m.index.describe_index_stats())"
```

