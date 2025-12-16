"""
Simple Insurance Q&A Chatbot - NO PYTORCH REQUIRED!
Uses TF-IDF embeddings instead of Sentence Transformers
"""

import streamlit as st
import time
from typing import List, Dict
import logging
from datetime import datetime
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(**config.PAGE_CONFIG)

# PROFESSIONAL CSS - READABLE & ATTRACTIVE
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { background: #f8f9fa; }
    
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1a202c;
        text-align: center;
        padding: 2rem 0 0.5rem 0;
    }
    .header-accent { color: #3b82f6; }
    
    .sub-header {
        font-size: 1.15rem;
        color: #64748b;
        text-align: center;
        padding-bottom: 2rem;
    }
    
    /* SIDEBAR - READABLE WHITE TEXT */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] span {
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: #60a5fa !important;
        font-weight: 700 !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #e2e8f0 !important;
    }
    
    /* PROFESSIONAL BUTTONS */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.25);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.4);
    }
    
    /* CHAT MESSAGES */
    .stChatMessage {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
    }
    
    /* SOURCE CARDS */
    .source-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-left: 4px solid #3b82f6;
        padding: 1.25rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'pinecone_manager' not in st.session_state:
        st.session_state.pinecone_manager = None
    if 'ollama_client' not in st.session_state:
        st.session_state.ollama_client = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False


@st.cache_resource
def load_simple_manager():
    """Load simple Pinecone manager (no PyTorch!)"""
    try:
        from pinecone_setup_simple import SimplePineconeManager
        manager = SimplePineconeManager()
        manager.create_index()
        logger.info("Simple Pinecone manager initialized")
        return manager, None
    except Exception as e:
        return None, f"Error: {str(e)}"


@st.cache_resource
def load_ollama_client():
    """Load Ollama client with phi3 for faster responses"""
    try:
        from ollama_client import OllamaClient
        client = OllamaClient(model="phi3:latest")  # Faster model!
        if not client.check_connection():
            return None, "Cannot connect to Ollama"
        return client, None
    except Exception as e:
        return None, f"Error: {str(e)}"


def process_query(question: str, manager, client):
    try:
        with st.spinner("🔍 Searching..."):
            similar_docs = manager.search_similar(question, top_k=5)
        
        with st.spinner("💭 Generating answer..."):
            if similar_docs:
                answer = client.generate_answer(question, similar_docs)
            else:
                answer = "I couldn't find specific information. Let me try a general answer:\n\n"
                answer += client.generate_simple_answer(question)
        
        return {'answer': answer, 'context_docs': similar_docs}
    except Exception as e:
        logger.error(f"Error: {e}")
        return {
            'answer': "Sorry, I encountered an error. Please try again.",
            'context_docs': []
        }


def main():
    initialize_session_state()
    
    st.markdown('<h1 class="main-header">🏛️ Insurance <span class="header-accent">Advisory AI</span></h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enterprise-Grade Insurance Intelligence • Powered by AI & Phi3</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        if not st.session_state.initialized:
            if st.button("🚀 Initialize Chatbot", type="primary", use_container_width=True):
                with st.spinner("Initializing..."):
                    manager, error = load_simple_manager()
                    if error:
                        st.error(f"❌ Pinecone: {error}")
                        return
                    
                    client, error = load_ollama_client()
                    if error:
                        st.error(f"❌ Ollama: {error}")
                        return
                    
                    st.session_state.pinecone_manager = manager
                    st.session_state.ollama_client = client
                    st.session_state.initialized = True
                    
                    st.success("✅ Ready!")
                    st.rerun()
        else:
            st.success("✅ Chatbot Ready")
            
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_history = []
                st.rerun()
        
        st.divider()
        st.info("""
        **Simple Version**
        - No PyTorch dependency
        - Works on any system
        - Uses TF-IDF embeddings
        - Powered by Mistral-7B
        """)
        
        # Example questions
        st.subheader("💡 Examples")
        examples = [
            "What is a deductible?",
            "How does auto insurance work?",
            "What is term life insurance?"
        ]
        
        for q in examples:
            if st.button(q, key=f"ex_{q}", use_container_width=True):
                if st.session_state.initialized:
                    st.session_state.messages.append({"role": "user", "content": q})
                    st.rerun()
    
    # Main area
    if not st.session_state.initialized:
        st.warning("👈 Click 'Initialize Chatbot' in the sidebar")
        
        st.info("""
        ### ℹ️ This is the Simple Version
        
        **Why this version?**
        - Avoids PyTorch DLL issues
        - Works on any Windows system
        - Lighter and faster startup
        - Still provides good answers!
        
        **What's different?**
        - Uses TF-IDF instead of deep learning for embeddings
        - Slightly less accurate similarity matching
        - But still very functional!
        """)
        return
    
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "context_docs" in message:
                if message["context_docs"]:
                    with st.expander("📚 Sources"):
                        for i, doc in enumerate(message["context_docs"], 1):
                            st.markdown(f"""
                            **Source {i}** (Score: {doc['score']:.2f})
                            
                            Q: {doc['question']}
                            
                            A: {doc['answer'][:200]}...
                            """)
    
    # Chat input
    if prompt := st.chat_input("Ask about insurance..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            result = process_query(
                prompt,
                st.session_state.pinecone_manager,
                st.session_state.ollama_client
            )
            
            st.markdown(result['answer'])
            
            if result['context_docs']:
                with st.expander("📚 Sources"):
                    for i, doc in enumerate(result['context_docs'], 1):
                        st.markdown(f"""
                        **Source {i}** (Score: {doc['score']:.2f})
                        
                        Q: {doc['question']}
                        
                        A: {doc['answer'][:200]}...
                        """)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": result['answer'],
            "context_docs": result['context_docs']
        })
        
        st.session_state.conversation_history.append({
            "user": prompt,
            "assistant": result['answer']
        })
        
        st.rerun()


if __name__ == "__main__":
    main()

