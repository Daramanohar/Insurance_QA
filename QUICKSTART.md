# 🚀 Quick Start Guide - Insurance Q&A Chatbot

Get up and running in 10 minutes!

## Prerequisites Check

Before starting, ensure you have:
- [ ] Python 3.8 or higher installed
- [ ] pip package manager
- [ ] Internet connection (for downloading models and data)
- [ ] 8GB+ RAM recommended

## Step-by-Step Setup

### 1️⃣ Install Dependencies (2 minutes)

```bash
# Navigate to project directory
cd linqura

# Install required packages
pip install -r requirements.txt
```

Expected output: All packages installed successfully.

---

### 2️⃣ Get Pinecone API Key (3 minutes)

1. Go to [https://app.pinecone.io/](https://app.pinecone.io/)
2. Sign up for a free account (no credit card required)
3. Click "API Keys" in the left sidebar
4. Copy your API key

---

### 3️⃣ Configure Environment (1 minute)

1. Open `env_template.txt`
2. Copy the contents
3. Create a new file named `.env` in the project root
4. Paste the contents and replace `your_pinecone_api_key_here` with your actual API key

**Your .env file should look like:**
```env
PINECONE_API_KEY=abc123-your-actual-key-xyz789
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=insurance-qa-index

OLLAMA_MODEL=mistral:7b
OLLAMA_BASE_URL=http://localhost:11434
```

---

### 4️⃣ Install Ollama & Mistral (3 minutes)

**Windows:**
1. Download from [https://ollama.ai/download](https://ollama.ai/download)
2. Run the installer
3. Open Command Prompt or PowerShell
4. Run: `ollama pull mistral`

**Mac/Linux:**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Mistral model
ollama pull mistral
```

Verify installation:
```bash
ollama list
```

You should see `mistral:7b` in the list.

---

### 5️⃣ Set Up Pinecone Database (5-10 minutes, ONE TIME ONLY)

```bash
python pinecone_setup.py
```

This will:
- Download InsuranceQA-v2 dataset (~10,000 Q&A pairs)
- Generate embeddings
- Store in Pinecone

**Expected output:**
```
[1/3] Loading and processing dataset...
✓ Processed 12000 Q&A pairs

[2/3] Setting up Pinecone...
✓ Pinecone index ready

[3/3] Generating embeddings and storing in Pinecone...
✓ All vectors stored successfully

Setup Complete!
```

⏰ **This takes 5-10 minutes. Be patient!**

---

### 6️⃣ Launch the Chatbot! 🎉

```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

---

## First Use

1. Click **"🚀 Initialize Chatbot"** in the sidebar
2. Wait a few seconds for components to load
3. You'll see "✅ Chatbot Ready"
4. Start asking questions!

### Try These Questions:

- "What is a deductible?"
- "How does collision coverage work in auto insurance?"
- "What's the difference between term and whole life insurance?"
- "What does comprehensive car insurance cover?"

---

## Troubleshooting

### Problem: "Cannot connect to Pinecone"
**Solution:** 
- Check your `.env` file has the correct API key
- Verify internet connection
- Make sure you ran `python pinecone_setup.py`

### Problem: "Cannot connect to Ollama"
**Solution:**
- Make sure Ollama is running (it should start automatically)
- Verify Mistral is installed: `ollama list`
- Try restarting Ollama service

### Problem: "No vectors found"
**Solution:**
- You need to run `python pinecone_setup.py` first
- This only needs to be done once

### Problem: Slow responses
**Solution:**
- First query is slower (model loading)
- Subsequent queries should be 2-5 seconds
- Check your internet connection

### Problem: Module not found
**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

---

## Automated Setup (Alternative)

We provide an interactive setup script:

```bash
python setup_guide.py
```

This will:
- Check all prerequisites
- Install dependencies
- Verify configuration
- Guide you through setup

---

## Testing Your Installation

Run the test suite to verify everything works:

```bash
python test_system.py
```

This will test:
- Package imports
- Configuration
- Ollama connection
- Pinecone connection
- Full RAG pipeline

All tests should pass ✅

---

## Directory Structure

```
linqura/
├── app.py                    # Main app - RUN THIS
├── config.py                 # Configuration
├── data_loader.py            # Dataset loader
├── pinecone_setup.py         # Setup script - RUN ONCE
├── ollama_client.py          # LLM client
├── requirements.txt          # Dependencies
├── .env                      # Your API keys (create this)
├── env_template.txt          # Template for .env
├── README.md                 # Full documentation
├── QUICKSTART.md            # This file
├── setup_guide.py            # Interactive setup
└── test_system.py            # Testing suite
```

---

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Set up database (one time)
python pinecone_setup.py

# Run chatbot
streamlit run app.py

# Test system
python test_system.py

# Interactive setup
python setup_guide.py
```

---

## Usage Tips

1. **Ask Clear Questions**: The more specific, the better
   - Good: "What does collision coverage include in auto insurance?"
   - Okay: "What is collision coverage?"

2. **Use Follow-ups**: The bot remembers context
   - First: "What is a deductible?"
   - Follow-up: "How does it work for health insurance?"

3. **View Sources**: Click "View Source References" to see where answers come from

4. **Adjust Settings**: Use sidebar to toggle context display and answer detail level

---

## Next Steps

- Read the full [README.md](README.md) for detailed information
- Check [PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md) for presentation tips
- Customize `config.py` for your needs
- Add your own insurance data

---

## Getting Help

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section above
2. Run `python test_system.py` to diagnose
3. Read error messages carefully
4. Check logs in the terminal

---

## Success Checklist

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Pinecone account created
- [ ] API key in `.env` file
- [ ] Ollama installed
- [ ] Mistral model pulled (`ollama pull mistral`)
- [ ] Database set up (`python pinecone_setup.py`)
- [ ] App launches (`streamlit run app.py`)
- [ ] Chatbot initializes successfully
- [ ] Questions get answered!

---

## 🎉 You're All Set!

Enjoy your Insurance Q&A Chatbot!

For detailed documentation, see [README.md](README.md)


