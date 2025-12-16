"""
Insurance Q&A Chatbot - Streamlit Application
Main application file for the interactive chatbot interface
"""

import streamlit as st
import time
from typing import List, Dict
import logging
from datetime import datetime

# Import config only (no PyTorch dependencies yet)
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy import function to avoid loading PyTorch at startup
def lazy_import_modules():
    """Import heavy modules only when needed"""
    try:
        from pinecone_setup import PineconeManager
        from ollama_client import OllamaClient
        return PineconeManager, OllamaClient, None
    except Exception as e:
        return None, None, str(e)


# Page configuration
st.set_page_config(**config.PAGE_CONFIG)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        padding-bottom: 2rem;
    }
    .context-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .source-reference {
        font-size: 0.9rem;
        color: #555;
        font-style: italic;
        padding: 0.5rem;
        background-color: #e8f4f8;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
    }
    .confidence-score {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    .score-high { background-color: #d4edda; color: #155724; }
    .score-medium { background-color: #fff3cd; color: #856404; }
    .score-low { background-color: #f8d7da; color: #721c24; }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 2rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """
    Initialize Streamlit session state variables
    """
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
    
    if 'show_context' not in st.session_state:
        st.session_state.show_context = True
    
    if 'answer_mode' not in st.session_state:
        st.session_state.answer_mode = "Detailed"


@st.cache_resource
def load_pinecone_manager():
    """
    Load and cache Pinecone manager (runs once per session)
    """
    try:
        # Lazy import to avoid loading PyTorch at startup
        PineconeManager, _, error = lazy_import_modules()
        if error:
            return None, f"Import error: {error}"
        
        manager = PineconeManager()
        manager.create_index()  # Connect to existing index
        logger.info("Pinecone manager initialized")
        return manager, None
    except Exception as e:
        error_msg = f"Error initializing Pinecone: {str(e)}"
        logger.error(error_msg)
        return None, error_msg


@st.cache_resource
def load_ollama_client():
    """
    Load and cache Ollama client (runs once per session)
    """
    try:
        # Lazy import to avoid loading dependencies at startup
        _, OllamaClient, error = lazy_import_modules()
        if error:
            return None, f"Import error: {error}"
            
        client = OllamaClient()
        if not client.check_connection():
            return None, "Cannot connect to Ollama. Please ensure Ollama is running."
        logger.info("Ollama client initialized")
        return client, None
    except Exception as e:
        error_msg = f"Error initializing Ollama: {str(e)}"
        logger.error(error_msg)
        return None, error_msg


def get_confidence_badge(score: float) -> str:
    """
    Generate HTML badge for confidence score
    """
    if score >= 0.8:
        css_class = "score-high"
        label = "High Confidence"
    elif score >= 0.6:
        css_class = "score-medium"
        label = "Medium Confidence"
    else:
        css_class = "score-low"
        label = "Low Confidence"
    
    return f'<span class="confidence-score {css_class}">{label} ({score:.2f})</span>'


def display_context_sources(context_docs: List[Dict]):
    """
    Display the source documents used for generating the answer
    """
    if not context_docs:
        return
    
    with st.expander("📚 View Source References", expanded=False):
        for i, doc in enumerate(context_docs, 1):
            score_badge = get_confidence_badge(doc['score'])
            
            st.markdown(f"""
            <div class="source-reference">
                <strong>Reference {i}</strong> {score_badge}
                <br><br>
                <strong>Q:</strong> {doc['question']}
                <br><br>
                <strong>A:</strong> {doc['answer']}
            </div>
            """, unsafe_allow_html=True)


def process_user_query(user_question: str, pinecone_manager, ollama_client) -> Dict:
    """
    Process user query through RAG pipeline
    
    Args:
        user_question: User's question
        pinecone_manager: Pinecone manager instance
        ollama_client: Ollama client instance
    
    Returns:
        Dictionary containing answer and context information
    """
    try:
        # Step 1: Retrieve similar documents from Pinecone
        with st.spinner("🔍 Searching knowledge base..."):
            similar_docs = pinecone_manager.search_similar(
                user_question,
                top_k=config.TOP_K_RESULTS
            )
        
        # Step 2: Generate answer using Ollama with retrieved context
        with st.spinner("💭 Generating answer..."):
            if similar_docs:
                answer = ollama_client.generate_answer(
                    question=user_question,
                    context_docs=similar_docs,
                    conversation_history=st.session_state.conversation_history[-3:]
                )
            else:
                # Fallback if no similar documents found
                answer = ("I couldn't find specific information about this in the insurance knowledge base. "
                         "However, let me try to provide a general answer:\n\n")
                answer += ollama_client.generate_simple_answer(user_question)
        
        return {
            'answer': answer,
            'context_docs': similar_docs,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {
            'answer': "I apologize, but I encountered an error processing your question. Please try again.",
            'context_docs': [],
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }


def main():
    """
    Main application function
    """
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<p class="main-header">🏥 Insurance Q&A Chatbot</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Ask me anything about insurance - health, auto, life, and more!</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Initialize button
        if not st.session_state.initialized:
            if st.button("🚀 Initialize Chatbot", type="primary", use_container_width=True):
                with st.spinner("Initializing components..."):
                    # Load Pinecone
                    pinecone_manager, error = load_pinecone_manager()
                    if error:
                        st.error(f"❌ {error}")
                        st.info("Make sure you've set up Pinecone and run 'python pinecone_setup.py' first.")
                        return
                    
                    # Load Ollama
                    ollama_client, error = load_ollama_client()
                    if error:
                        st.error(f"❌ {error}")
                        st.info("Make sure Ollama is running with Mistral model:\n```\nollama pull mistral\n```")
                        return
                    
                    # Store in session state
                    st.session_state.pinecone_manager = pinecone_manager
                    st.session_state.ollama_client = ollama_client
                    st.session_state.initialized = True
                    
                    st.success("✅ Chatbot initialized successfully!")
                    st.rerun()
        else:
            st.success("✅ Chatbot Ready")
            
            # Answer mode selector
            st.subheader("Answer Settings")
            st.session_state.answer_mode = st.radio(
                "Answer Detail Level:",
                ["Concise", "Detailed"],
                index=1
            )
            
            # Show context toggle
            st.session_state.show_context = st.checkbox(
                "Show Source References",
                value=True
            )
            
            st.divider()
            
            # Clear conversation button
            if st.button("🗑️ Clear Conversation", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_history = []
                st.success("Conversation cleared!")
                st.rerun()
            
            # Statistics
            st.divider()
            st.subheader("📊 Session Stats")
            st.metric("Messages", len(st.session_state.messages))
            st.metric("Conversations", len(st.session_state.conversation_history))
        
        # Info section
        st.divider()
        st.subheader("ℹ️ About")
        st.info("""
        This chatbot uses:
        - **Pinecone** for vector storage
        - **Sentence Transformers** for embeddings
        - **Mistral-7B** via Ollama for generation
        - **InsuranceQA-v2** dataset
        
        Ask questions about insurance policies, coverage, deductibles, and more!
        """)
        
        # Example questions
        st.subheader("💡 Example Questions")
        example_questions = [
            "What is a deductible?",
            "How does collision coverage work?",
            "What's the difference between term and whole life insurance?",
            "What does health insurance cover?",
            "How do I file an insurance claim?"
        ]
        
        for q in example_questions:
            if st.button(q, key=f"example_{q}", use_container_width=True):
                if st.session_state.initialized:
                    st.session_state.messages.append({"role": "user", "content": q})
                    st.rerun()
    
    # Main chat interface
    if not st.session_state.initialized:
        st.warning("👈 Please initialize the chatbot using the sidebar button to get started.")
        
        # Show quick start guide
        st.markdown("### 🚀 Quick Start Guide")
        st.markdown("""
        1. **Set up environment variables** (create a `.env` file):
           - `PINECONE_API_KEY` - Get from [Pinecone](https://app.pinecone.io/)
           - `PINECONE_ENVIRONMENT` - Your Pinecone environment (e.g., us-east-1-aws)
        
        2. **Install Ollama and Mistral**:
           ```bash
           # Install Ollama from https://ollama.ai
           ollama pull mistral
           ```
        
        3. **Set up Pinecone database** (first time only):
           ```bash
           python pinecone_setup.py
           ```
        
        4. **Click "Initialize Chatbot"** in the sidebar
        """)
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display context sources if available
            if message["role"] == "assistant" and "context_docs" in message:
                if st.session_state.show_context and message["context_docs"]:
                    display_context_sources(message["context_docs"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about insurance..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            response_data = process_user_query(
                prompt,
                st.session_state.pinecone_manager,
                st.session_state.ollama_client
            )
            
            answer = response_data['answer']
            context_docs = response_data['context_docs']
            
            st.markdown(answer)
            
            # Display context sources
            if st.session_state.show_context and context_docs:
                display_context_sources(context_docs)
        
        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "context_docs": context_docs
        })
        
        # Update conversation history for context
        st.session_state.conversation_history.append({
            "user": prompt,
            "assistant": answer
        })
        
        st.rerun()


if __name__ == "__main__":
    main()


