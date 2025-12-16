# 🏥 Insurance Knowledge-Base Q&A Chatbot

An AI-driven chatbot that answers insurance-related questions using RAG (Retrieval Augmented Generation) architecture with Pinecone, Sentence Transformers, and Mistral-7B.

## 🎯 Features

- **Intelligent Q&A**: Ask natural language questions about health, auto, life, and other insurance topics
- **Context-Aware**: Maintains conversation history for follow-up questions
- **Source References**: Shows the knowledge base sources used to generate answers
- **Real-Time Search**: Fast similarity search using Pinecone vector database
- **Local LLM**: Uses Mistral-7B via Ollama (no OpenAI API required)
- **User-Friendly UI**: Clean Streamlit interface with confidence scores

## 🏗️ Architecture

```
User Question
     ↓
[Sentence Transformer] → Generate Query Embedding
     ↓
[Pinecone Vector DB] → Similarity Search (Top K Results)
     ↓
[Retrieved Context] → Relevant Q&A Pairs
     ↓
[Mistral-7B via Ollama] → Generate Answer with Context
     ↓
Display Answer + Sources
```

### Key Components:

1. **Data Loading** (`data_loader.py`): Loads and processes InsuranceQA-v2 dataset from Hugging Face
2. **Embedding & Storage** (`pinecone_setup.py`): Generates embeddings and stores in Pinecone
3. **Vector Search**: Retrieves relevant documents using cosine similarity
4. **Answer Generation** (`ollama_client.py`): Uses Mistral-7B to generate contextual answers
5. **UI** (`app.py`): Streamlit-based chat interface

## 📋 Prerequisites

- Python 3.8 or higher
- Pinecone account (free tier available)
- Ollama installed locally
- 8GB+ RAM recommended

## 🚀 Installation & Setup

### Step 1: Clone and Install Dependencies

```bash
# Navigate to project directory
cd linqura

# Install required packages
pip install -r requirements.txt
```

### Step 2: Set Up Pinecone

1. Create a free account at [Pinecone](https://app.pinecone.io/)
2. Get your API key from the Pinecone dashboard
3. Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_api_key_here
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=insurance-qa-index

OLLAMA_MODEL=mistral:7b
OLLAMA_BASE_URL=http://localhost:11434
```

**Note**: Copy from `env_template.txt` and fill in your actual values.

### Step 3: Install and Set Up Ollama

1. **Install Ollama**:
   - Download from [https://ollama.ai](https://ollama.ai)
   - Follow installation instructions for your OS

2. **Pull Mistral Model**:
   ```bash
   ollama pull mistral
   ```

3. **Verify Ollama is Running**:
   ```bash
   # Ollama should start automatically
   # Test with:
   ollama list
   ```

### Step 4: Initialize Pinecone Database

**This is a one-time setup that loads the dataset and creates embeddings:**

```bash
python pinecone_setup.py
```

This script will:
- Download InsuranceQA-v2 dataset from Hugging Face
- Process ~20,000+ Q&A pairs
- Generate embeddings using Sentence Transformers
- Store vectors in Pinecone (takes 5-10 minutes)

**Expected Output:**
```
[1/3] Loading and processing dataset...
✓ Processed 12000 Q&A pairs

[2/3] Setting up Pinecone...
✓ Pinecone index ready

[3/3] Generating embeddings and storing in Pinecone...
✓ All vectors stored successfully

Setup Complete! Your chatbot is ready to use.
```

### Step 5: Run the Chatbot

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🎮 Usage

### First Time Setup in the App:

1. Click **"Initialize Chatbot"** button in the sidebar
2. Wait for components to load (takes a few seconds)
3. Start asking questions!

### Example Questions:

- "What is a deductible in health insurance?"
- "How does collision coverage work in auto insurance?"
- "What's the difference between term and whole life insurance?"
- "What does comprehensive car insurance cover?"
- "How do I file a claim after an accident?"

### Features to Try:

- **Follow-up Questions**: The chatbot maintains context
- **Source References**: Toggle "Show Source References" to see knowledge base sources
- **Answer Modes**: Choose between Concise and Detailed answers
- **Example Questions**: Quick-start buttons in the sidebar

## 📁 Project Structure

```
linqura/
├── app.py                 # Main Streamlit application
├── config.py              # Configuration settings
├── data_loader.py         # Dataset loading and processing
├── pinecone_setup.py      # Pinecone initialization and embedding storage
├── ollama_client.py       # Ollama API client for LLM
├── requirements.txt       # Python dependencies
├── env_template.txt       # Environment variable template
└── README.md             # This file
```

## 🔧 Configuration

Edit `config.py` to customize:

- **Embedding Model**: Change `EMBEDDING_MODEL` (default: `all-MiniLM-L6-v2`)
- **Top K Results**: Adjust `TOP_K_RESULTS` (default: 5)
- **Similarity Threshold**: Set `SIMILARITY_THRESHOLD` (default: 0.5)
- **Chunk Size**: Modify `CHUNK_SIZE` for different chunking strategies

## 🧪 Testing Components

### Test Data Loader:
```bash
python data_loader.py
```

### Test Ollama Connection:
```bash
python ollama_client.py
```

### Test Pinecone Search:
```python
from pinecone_setup import PineconeManager

manager = PineconeManager()
manager.create_index()
results = manager.search_similar("What is a deductible?", top_k=3)
print(results)
```

## 🐛 Troubleshooting

### "Cannot connect to Pinecone"
- Check your API key in `.env` file
- Verify internet connection
- Ensure you've created an index by running `pinecone_setup.py`

### "Cannot connect to Ollama"
- Make sure Ollama is running (`ollama list`)
- Check if Mistral is installed (`ollama pull mistral`)
- Verify Ollama is on port 11434

### "No vectors found"
- Run `python pinecone_setup.py` to populate the database
- Check Pinecone dashboard to verify index exists

### Slow Response Times
- First query is slower due to model loading
- Subsequent queries should be faster
- Consider reducing `TOP_K_RESULTS` in config.py

### Out of Memory
- Reduce batch size in `pinecone_setup.py`
- Use a smaller embedding model
- Close other applications

## 📊 Performance

- **Setup Time**: ~5-10 minutes (one-time)
- **Query Response**: 2-5 seconds
- **Dataset Size**: ~12,000 Q&A pairs
- **Embedding Dimension**: 384
- **Memory Usage**: ~2-3 GB

## 🎨 Customization

### Add Your Own Data:

1. Modify `data_loader.py` to load custom data
2. Ensure data format matches:
   ```python
   {
       'id': 'unique_id',
       'question': 'question text',
       'answer': 'answer text',
       'combined_text': 'question + answer'
   }
   ```
3. Re-run `python pinecone_setup.py`

### Change LLM Model:

Edit `.env`:
```env
OLLAMA_MODEL=llama2:7b  # or any other Ollama model
```

### Adjust UI:

Modify CSS in `app.py` under `st.markdown("""<style>...</style>""")`

## 📦 Dependencies

Key libraries:
- `streamlit`: Web interface
- `sentence-transformers`: Embeddings
- `pinecone-client`: Vector database
- `datasets`: Hugging Face datasets
- `requests`: Ollama API calls

## 🤝 Contributing

Feel free to:
- Add new features
- Improve prompts
- Optimize performance
- Add more insurance datasets

## 📝 Assignment Submission Checklist

- ✅ Uses open-source LLM (Mistral-7B via Ollama)
- ✅ No OpenAI API
- ✅ InsuranceQA dataset from Hugging Face
- ✅ Streamlit chat interface
- ✅ Retrieval-augmented generation (RAG)
- ✅ Source references and context
- ✅ Multi-turn conversation support
- ✅ Error handling for edge cases
- ✅ Well-commented code
- ✅ Setup instructions
- ✅ Efficient vector search with Pinecone
- ✅ Clean, user-friendly UI

## 🎓 How It Works

### 1. Data Processing:
- Loads InsuranceQA-v2 from Hugging Face
- Processes questions and expert answers
- Creates combined text for better context

### 2. Embedding Generation:
- Uses Sentence Transformers (all-MiniLM-L6-v2)
- Converts text to 384-dimensional vectors
- Captures semantic meaning

### 3. Vector Storage:
- Stores embeddings in Pinecone
- Enables fast cosine similarity search
- Metadata includes questions and answers

### 4. Query Processing:
- User question → embedding
- Similarity search in Pinecone
- Retrieve top K relevant Q&A pairs

### 5. Answer Generation:
- Build context from retrieved documents
- Create prompt with context + conversation history
- Generate answer using Mistral-7B
- Display with source references

## 📞 Support

For issues or questions:
1. Check troubleshooting section
2. Verify all setup steps completed
3. Check logs for error messages

## 📄 License

This project is created for educational purposes.

---

**Built with ❤️ using open-source technologies**


