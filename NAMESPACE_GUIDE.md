# Pinecone Namespace Guide

## What Are Namespaces?

Namespaces in Pinecone are like **folders** or **partitions** within your index. They allow you to organize vectors into separate logical groups within the same index.

### Benefits:
- **Organize data** by category (health insurance, auto insurance, etc.)
- **Separate environments** (production, testing, staging)
- **Isolate queries** (search only within specific namespace)
- **Easy management** (delete/update specific groups)

---

## Current Setup

✅ **Your code now supports namespaces!**

By default, your vectors are stored in the **default namespace** (empty string "").

---

## How to Change Namespace

### Option 1: Use Default Namespace (Current)

**In your `.env` file:**
```env
PINECONE_NAMESPACE=
```

This stores/searches vectors in the default namespace (shown as `__default__` in dashboard).

---

### Option 2: Use a Custom Namespace

**In your `.env` file:**
```env
PINECONE_NAMESPACE=production
```

Or any name you want:
- `production`
- `testing`
- `health-insurance`
- `auto-insurance`
- `v1`, `v2`, etc.

---

## Step-by-Step: Create New Namespace

### 1. Edit `.env` File

Open `.env` and change the namespace:

```env
PINECONE_NAMESPACE=my-custom-namespace
```

### 2. Re-run Setup (Optional - Only if you want NEW data)

If you want to store the data in a NEW namespace:

```bash
python pinecone_setup.py
```

This will:
- Read the new namespace from `.env`
- Store all vectors in the new namespace
- Keep your old data in the default namespace

### 3. Update Your App

The app automatically uses the namespace from `.env`. Just restart:

```bash
streamlit run app.py
```

---

## Working with Multiple Namespaces

### Example: Separate by Insurance Type

**1. Create Health Insurance Namespace:**

Edit `.env`:
```env
PINECONE_NAMESPACE=health-insurance
```

Run setup (or manually filter your data for health insurance only):
```bash
python pinecone_setup.py
```

**2. Create Auto Insurance Namespace:**

Edit `.env`:
```env
PINECONE_NAMESPACE=auto-insurance
```

Run setup again with filtered data.

**3. Switch Between Them:**

Just change the namespace in `.env` and restart your app!

---

## Namespace Operations

### List All Namespaces

```python
from pinecone_setup import PineconeManager

manager = PineconeManager()
manager.create_index()

# Get index stats (shows all namespaces)
stats = manager.index.describe_index_stats()
print(stats['namespaces'])

# Output example:
# {
#   '': {'vector_count': 21325},           # default namespace
#   'production': {'vector_count': 21325},
#   'testing': {'vector_count': 1000}
# }
```

### Search Specific Namespace

The code automatically uses the namespace from `config.PINECONE_NAMESPACE`.

To search a different namespace programmatically:

```python
# Search in specific namespace
results = manager.index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True,
    namespace="production"  # Specify namespace
)
```

### Delete a Namespace

**Warning: This deletes all vectors in the namespace!**

```python
# Delete all vectors in a namespace
manager.index.delete(delete_all=True, namespace="testing")
```

---

## Current State

Your current setup:
- **Namespace**: `""` (default)
- **Vectors**: 21,325 in default namespace
- **Location**: Visible as `__default__` in Pinecone dashboard

---

## Practical Use Cases

### Use Case 1: Development vs Production

```env
# Development
PINECONE_NAMESPACE=dev

# Production  
PINECONE_NAMESPACE=prod
```

### Use Case 2: Versioning

```env
# Version 1 of your data
PINECONE_NAMESPACE=v1

# Version 2 with updated data
PINECONE_NAMESPACE=v2
```

### Use Case 3: Multi-tenant Application

```env
# Customer A's data
PINECONE_NAMESPACE=customer-a

# Customer B's data
PINECONE_NAMESPACE=customer-b
```

### Use Case 4: A/B Testing

```env
# Old model
PINECONE_NAMESPACE=model-old

# New model
PINECONE_NAMESPACE=model-new
```

---

## Best Practices

1. **Use descriptive names**: `production` not `prod1`
2. **Document your namespaces**: Keep track of what each contains
3. **Don't overuse**: Too many namespaces can be confusing
4. **Consistent naming**: Use lowercase with hyphens
5. **Backup important namespaces**: Export before deleting

---

## FAQ

**Q: Can I search across multiple namespaces?**
A: No, each query searches only one namespace. You'd need to query each separately and combine results.

**Q: Do namespaces cost extra?**
A: No, namespaces are free. You only pay for total vectors stored.

**Q: Can I rename a namespace?**
A: No, you need to create a new one and migrate data.

**Q: What's the limit?**
A: Pinecone supports 100 namespaces per index (free tier).

**Q: Do I need to change namespace?**
A: No! The default namespace works perfectly. Only change if you need organization.

---

## Quick Reference

```bash
# View current namespace
cat .env | grep NAMESPACE

# Change namespace
# Edit .env and change PINECONE_NAMESPACE=your-namespace

# Re-populate with new namespace
python pinecone_setup.py

# Test search in new namespace
python test_pinecone.py

# Run app with new namespace
streamlit run app.py
```

---

## Your Current Setup

Check your current namespace:

```bash
python -c "import config; print(f'Current namespace: {config.PINECONE_NAMESPACE or \"default\"}')"
```

---

**Need help?** The default namespace works great! Only change if you have a specific organizational need.

