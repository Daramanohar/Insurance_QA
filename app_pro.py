"""
Professional Insurance Q&A Chatbot - Production Grade
Optimized RAG pipeline with intelligent caching and premium UI
"""

import streamlit as st
import time
from typing import List, Dict
import logging
from datetime import datetime
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Premium page configuration
st.set_page_config(
    page_title="Insurance Q&A Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Professional Insurance Q&A Assistant powered by AI"
    }
)

# Professional CSS styling
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header styling */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 2rem 0 0.5rem 0;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #64748b;
        text-align: center;
        padding-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        border-radius: 8px;
    }
    
    .stError {
        background-color: #fee2e2;
        border-left: 4px solid #ef4444;
        border-radius: 8px;
    }
    
    /* Context sources */
    .source-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .source-header {
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .confidence-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(255,255,255,0.2);
    }
    
    /* Performance metrics */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3c72;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 0.25rem;
    }
    
    /* Loading animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .thinking {
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    /* Input box styling */
    .stChatInputContainer {
        border-top: 2px solid #e2e8f0;
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'messages': [],
        'conversation_history': [],
        'pinecone_manager': None,
        'ollama_client': None,
        'rag_engine': None,
        'cache_manager': None,
        'initialized': False,
        'total_queries': 0,
        'cache_hits': 0
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def load_components():
    """Load and cache all components"""
    try:
        from pinecone_setup_simple import SimplePineconeManager
        from cache_manager import QueryCache
        from optimized_rag_engine import OptimizedRAGEngine
        
        # Load Pinecone
        pinecone_manager = SimplePineconeManager()
        pinecone_manager.create_index()
        
        # Load Ollama (lightweight, no caching needed)
        from ollama_client import OllamaClient
        ollama_client = OllamaClient()
        
        if not ollama_client.check_connection():
            return None, None, None, "Ollama not running"
        
        # Initialize cache
        cache_manager = QueryCache(similarity_threshold=0.90)
        
        # Create RAG engine
        rag_engine = OptimizedRAGEngine(
            pinecone_manager,
            ollama_client,
            cache_manager
        )
        
        return rag_engine, cache_manager, {
            'pinecone': pinecone_manager,
            'ollama': ollama_client
        }, None
        
    except Exception as e:
        return None, None, None, str(e)


def display_source_card(doc: Dict, index: int):
    """Display a professional source reference card"""
    score = doc['score']
    
    # Color based on confidence
    if score >= 0.8:
        badge_color = "#10b981"
        confidence = "High"
    elif score >= 0.6:
        badge_color = "#f59e0b"
        confidence = "Medium"
    else:
        badge_color = "#6b7280"
        confidence = "Low"
    
    st.markdown(f"""
    <div class="source-card">
        <div class="source-header">
            <span>📄 Reference {index}</span>
            <span class="confidence-badge" style="background-color: {badge_color}">
                {confidence} • {score:.0%}
            </span>
        </div>
        <div style="font-size: 0.9rem; margin-top: 0.75rem;">
            <strong>Q:</strong> {doc['question']}
        </div>
        <div style="font-size: 0.85rem; margin-top: 0.5rem; opacity: 0.9;">
            <strong>A:</strong> {doc['answer'][:150]}...
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">💼 Insurance Advisory Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Answers to Your Insurance Questions</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Control Panel")
        st.markdown("---")
        
        # Initialize button
        if not st.session_state.initialized:
            if st.button("🚀 Initialize Assistant", type="primary", use_container_width=True):
                with st.spinner("🔄 Loading AI components..."):
                    rag_engine, cache_mgr, components, error = load_components()
                    
                    if error:
                        st.error(f"❌ {error}")
                        st.info("**Troubleshooting:**\n- Ensure Ollama is running\n- Check Pinecone connection")
                        return
                    
                    st.session_state.rag_engine = rag_engine
                    st.session_state.cache_manager = cache_mgr
                    st.session_state.pinecone_manager = components['pinecone']
                    st.session_state.ollama_client = components['ollama']
                    st.session_state.initialized = True
                    
                    st.success("✅ Assistant Ready!")
                    time.sleep(0.5)
                    st.rerun()
        
        else:
            st.success("✅ **Assistant Active**")
            
            # Performance metrics
            st.markdown("---")
            st.markdown("### 📊 Performance")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Queries", st.session_state.total_queries)
            with col2:
                cache_rate = (st.session_state.cache_hits / st.session_state.total_queries * 100) if st.session_state.total_queries > 0 else 0
                st.metric("Cache Hit Rate", f"{cache_rate:.0f}%")
            
            # Cache stats
            if st.session_state.cache_manager:
                stats = st.session_state.cache_manager.get_cache_stats()
                st.caption(f"📦 Cached: {stats.get('total_entries', 0)} answers")
            
            # Actions
            st.markdown("---")
            st.markdown("### 🎛️ Actions")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.conversation_history = []
                    st.rerun()
            
            with col2:
                if st.button("🔄 Clear Cache", use_container_width=True):
                    if st.session_state.cache_manager:
                        st.session_state.cache_manager.clear_cache()
                        st.session_state.cache_hits = 0
                    st.success("Cache cleared!")
            
            # Performance graph (if enough data)
            if st.session_state.rag_engine and len(st.session_state.rag_engine.performance_logs) >= 3:
                st.markdown("---")
                st.markdown("### ⚡ Response Times")
                
                perf_stats = st.session_state.rag_engine.get_performance_stats()
                st.caption(f"Avg: {perf_stats['avg_total_time']:.2f}s")
                st.progress(min(perf_stats['avg_total_time'] / 10, 1.0))
        
        # Info section
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.caption("""
        **Technology Stack:**
        - RAG Architecture
        - Pinecone Vector DB
        - Mistral-7B LLM
        - Smart Caching
        - TF-IDF Embeddings
        
        **Coverage:**
        21,325+ Insurance Q&A pairs
        """)
        
        # Example questions
        st.markdown("---")
        st.markdown("### 💡 Try These")
        
        examples = [
            "What is a deductible?",
            "How does collision coverage work?",
            "Explain term vs whole life insurance",
            "What does health insurance cover?",
            "How do I file a claim?"
        ]
        
        for example in examples:
            if st.button(f"💬 {example[:30]}...", key=f"ex_{example}", use_container_width=True):
                if st.session_state.initialized:
                    st.session_state.messages.append({"role": "user", "content": example})
                    st.rerun()
    
    # Main chat area
    if not st.session_state.initialized:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 3rem 0;'>
                <div style='font-size: 4rem; margin-bottom: 1rem;'>💼</div>
                <h2 style='color: #1e3c72; margin-bottom: 1rem;'>Welcome to Insurance Advisory Assistant</h2>
                <p style='font-size: 1.1rem; color: #64748b; line-height: 1.6;'>
                    Get instant, accurate answers to your insurance questions.<br>
                    Powered by advanced AI and 21,000+ expert Q&A pairs.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("""
            ### 🎯 What This Assistant Does
            
            ✅ **Answers Insurance Questions** - Health, Auto, Life, Home  
            ✅ **Provides Source References** - Transparent and trustworthy  
            ✅ **Remembers Context** - Natural follow-up conversations  
            ✅ **Lightning Fast** - Intelligent caching for instant answers  
            ✅ **Always Learning** - Improves with every interaction  
            """)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 👈 Click **'Initialize Assistant'** in the sidebar to begin")
        
        return
    
    # Display conversation
    for msg_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and "context_docs" in message:
                if message.get("context_docs"):
                    with st.expander("📚 Source References", expanded=False):
                        for i, doc in enumerate(message["context_docs"][:3], 1):
                            display_source_card(doc, i)
                
                # Show performance metrics
                if "timings" in message:
                    timings = message["timings"]
                    from_cache = message.get("from_cache", False)
                    
                    cols = st.columns(4)
                    with cols[0]:
                        st.caption(f"⏱️ {timings['total']:.2f}s")
                    with cols[1]:
                        st.caption(f"{'💾 Cached' if from_cache else '🔍 Fresh'}")
                    if not from_cache:
                        with cols[2]:
                            st.caption(f"📡 {timings.get('retrieval', 0):.2f}s")
                        with cols[3]:
                            st.caption(f"🤖 {timings.get('generation', 0):.2f}s")
    
    # Chat input
    if prompt := st.chat_input("💬 Ask me anything about insurance...", key="chat_input"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.total_queries += 1
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            with st.status("🔍 Processing your question...", expanded=False) as status:
                # Process through RAG engine
                result = st.session_state.rag_engine.process_query(
                    prompt,
                    st.session_state.conversation_history[-3:] if st.session_state.conversation_history else None,
                    use_cache=True
                )
                
                # Update cache hit counter
                if result.get('from_cache'):
                    st.session_state.cache_hits += 1
                    status.update(label="✅ Retrieved from cache (instant!)", state="complete")
                else:
                    status.update(label="✅ Answer generated successfully", state="complete")
            
            # Display answer
            st.markdown(result['answer'])
            
            # Show sources
            if result.get('context_docs'):
                with st.expander("📚 Source References", expanded=False):
                    for i, doc in enumerate(result['context_docs'][:3], 1):
                        display_source_card(doc, i)
            
            # Show performance
            if result.get('timings'):
                timings = result['timings']
                from_cache = result.get('from_cache', False)
                
                cols = st.columns(4)
                with cols[0]:
                    st.caption(f"⏱️ Response time: {timings['total']:.2f}s")
                with cols[1]:
                    st.caption(f"{'💾 From cache' if from_cache else '🔍 Fresh answer'}")
                if not from_cache:
                    with cols[2]:
                        st.caption(f"📡 Search: {timings.get('retrieval', 0):.2f}s")
                    with cols[3]:
                        st.caption(f"🤖 Generate: {timings.get('generation', 0):.2f}s")
        
        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result['answer'],
            "context_docs": result.get('context_docs', []),
            "timings": result.get('timings', {}),
            "from_cache": result.get('from_cache', False)
        })
        
        st.session_state.conversation_history.append({
            "user": prompt,
            "assistant": result['answer']
        })
        
        st.rerun()


if __name__ == "__main__":
    main()

