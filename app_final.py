"""
Professional Insurance Q&A Chatbot - FINAL VERSION
- Sentence Transformers for quality embeddings
- Mistral 7B Instruct for high-quality answers via Ollama
- Premium professional UI
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
    page_title="Insurance Advisory AI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# PROFESSIONAL CSS - FIXED COLORS & VISIBILITY
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background - soft professional */
    .main {
        background: #f8f9fa;
    }
    
    /* Premium header */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1a202c;
        text-align: center;
        padding: 2rem 0 0.5rem 0;
        letter-spacing: -0.5px;
    }
    
    .header-accent {
        color: #3b82f6;
    }
    
    .sub-header {
        font-size: 1.15rem;
        color: #64748b;
        text-align: center;
        padding-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Sidebar - PROFESSIONAL DARK WITH READABLE TEXT */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    
    /* Sidebar text - WHITE AND READABLE */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] div {
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] .stCaption {
        color: #cbd5e1 !important;
    }
    
    /* Metrics in sidebar - WHITE TEXT */
    section[data-testid="stSidebar"] .css-1wivap2,
    section[data-testid="stSidebar"] [data-testid="stMetricValue"],
    section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #ffffff !important;
    }
    
    /* Buttons - PROFESSIONAL STYLE */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.25);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.4);
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    }
    
    section[data-testid="stSidebar"] .stButton>button {
        width: 100%;
    }
    
    /* Chat messages - ELEGANT CARDS */
    .stChatMessage {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
    }
    
    [data-testid="stChatMessageContent"] {
        color: #1a202c;
    }
    
    /* Source cards - PREMIUM DESIGN */
    .source-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-left: 4px solid #3b82f6;
        padding: 1.25rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    
    .source-header {
        font-weight: 700;
        font-size: 0.95rem;
        color: #1e293b;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .confidence-high {
        background: #10b981;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .confidence-medium {
        background: #f59e0b;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .confidence-low {
        background: #6b7280;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .source-text {
        color: #475569;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    /* Success/Info messages - VISIBLE */
    .stSuccess {
        background-color: #d1fae5 !important;
        color: #065f46 !important;
        border-left: 4px solid #10b981;
    }
    
    .stInfo {
        background-color: #dbeafe !important;
        color: #1e40af !important;
        border-left: 4px solid #3b82f6;
    }
    
    /* Performance badge */
    .perf-badge {
        display: inline-block;
        background: #f1f5f9;
        color: #475569;
        padding: 0.35rem 0.75rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.25rem;
    }
    
    .cache-badge {
        background: #dcfce7;
        color: #166534;
        font-weight: 600;
    }
    
    /* Input styling */
    .stChatInputContainer {
        border-top: 2px solid #e2e8f0;
        padding-top: 1.5rem;
        background: white;
    }
    
    /* Metric cards - VISIBLE IN SIDEBAR */
    section[data-testid="stSidebar"] .stMetric {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: #60a5fa !important;
        font-weight: 700 !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #e2e8f0 !important;
        font-size: 0.9rem !important;
    }
    
    /* Expander - better styling */
    .streamlit-expanderHeader {
        background-color: #f8fafc;
        border-radius: 8px;
        font-weight: 600;
        color: #1e293b;
    }
    
    /* Welcome card */
    .welcome-card {
        background: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        text-align: center;
        margin: 2rem auto;
        max-width: 800px;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        border-color: #3b82f6;
        transform: translateY(-4px);
        box-shadow: 0 8px 16px rgba(59, 130, 246, 0.15);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .feature-title {
        font-weight: 700;
        color: #1e293b;
        margin: 0.5rem 0;
    }
    
    .feature-desc {
        color: #64748b;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state"""
    defaults = {
        'messages': [],
        'conversation_history': [],
        'pinecone_manager': None,
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
def load_optimized_components():
    """Load all components with sentence transformers"""
    try:
        # Use ORIGINAL pinecone_setup with sentence transformers
        from pinecone_setup import PineconeManager
        from ollama_client import OllamaClient
        from cache_manager import QueryCache
        from optimized_rag_engine import OptimizedRAGEngine
        
        # Load Pinecone with sentence transformers
        pinecone_manager = PineconeManager()
        pinecone_manager.create_index()
        
        # Load Ollama with Mistral (model comes from config)
        ollama_client = OllamaClient()  # Uses config.OLLAMA_MODEL by default
        if not ollama_client.check_connection():
            return None, None, "Ollama not running. Please start Ollama."
        
        # Initialize cache
        cache_manager = QueryCache(similarity_threshold=0.90)
        
        # Create optimized RAG engine
        rag_engine = OptimizedRAGEngine(
            pinecone_manager,
            ollama_client,
            cache_manager
        )
        
        return rag_engine, cache_manager, None
        
    except Exception as e:
        return None, None, f"Initialization error: {str(e)}"


def display_professional_source(doc: Dict, index: int):
    """Display source with professional styling"""
    score = doc.get('score', 0.0)
    vector_id = doc.get('id', 'N/A')
    
    if score >= 0.8:
        badge_class = "confidence-high"
        confidence = "High Confidence"
    elif score >= 0.6:
        badge_class = "confidence-medium"
        confidence = "Medium Confidence"
    else:
        badge_class = "confidence-low"
        confidence = "Low Relevance"
    
    st.markdown(f"""
    <div class="source-card">
        <div class="source-header">
            📄 Reference {index} &nbsp; <code>{vector_id}</code>
            <span class="{badge_class}">{confidence} • {score:.0%}</span>
        </div>
        <div class="source-text">
            <strong>Q:</strong> {doc['question']}<br><br>
            <strong>A:</strong> {doc['answer'][:200]}...
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application"""
    initialize_session_state()
    
    # Professional Header
    st.markdown('''
    <h1 class="main-header">
        🏛️ Insurance <span class="header-accent">Advisory AI</span>
    </h1>
    <p class="sub-header">Enterprise-Grade Insurance Intelligence • Powered by Advanced AI</p>
    ''', unsafe_allow_html=True)
    
    # REDESIGNED SIDEBAR - PROFESSIONAL & READABLE
    with st.sidebar:
        st.markdown("## ⚙️ Control Center")
        st.markdown("---")
        
        # Initialize button
        if not st.session_state.initialized:
            st.markdown("### 🚀 Get Started")
            if st.button("⚡ Initialize AI Assistant", type="primary", use_container_width=True):
                with st.spinner("🔄 Loading AI components..."):
                    progress = st.progress(0)
                    
                    progress.progress(33)
                    time.sleep(0.3)
                    
                    rag_engine, cache_mgr, error = load_optimized_components()
                    progress.progress(66)
                    
                    if error:
                        st.error(f"❌ {error}")
                        progress.progress(100)
                        return
                    
                    st.session_state.rag_engine = rag_engine
                    st.session_state.cache_manager = cache_mgr
                    st.session_state.initialized = True
                    
                    progress.progress(100)
                    st.success("✅ AI Assistant Ready!")
                    time.sleep(0.5)
                    st.rerun()
            
            st.markdown("---")
            st.markdown("### 📋 System Info")
            st.info(f"""
            **LLM Model**: `{config.OLLAMA_MODEL}`  
            **Knowledge Base**: 21,325 Q&A Pairs  
            **Search Engine**: Pinecone Vector DB  
            **Embeddings**: Sentence Transformers  
            **Mode**: Stateless Q&A (no cross-question memory)
            """)

            st.markdown("---")
            st.markdown("### ⚡ Performance Note")
            st.markdown("""
**System Performance Notice**  
You are running **Mistral-7B** on **CPU** via Ollama.  
- On CPU-only inference, **RAG responses** can take **25–40 seconds** for complex questions.  
- This is expected for a 7B parameter model without GPU acceleration.  
- With **GPU acceleration** and **quantization**, responses typically complete in **5–15 seconds**, which is standard for interactive Q&A.

Technical details:
- Context window: up to 4096 tokens (configured at 2048 for CPU efficiency)  
- Retrieval-Augmented Generation (RAG) injects retrieved documents into the prompt  
- Each request builds a KV cache and recomputes attention on CPU, which impacts latency  

**Short version:** Running Mistral-7B on CPU → RAG answers may take **25–40s**. With GPU, **5–15s** is typical.
            """)
        
        else:
            # Active status
            st.markdown("### ✅ System Status")
            st.success("**AI Assistant Active**")
            
            st.markdown("---")
            
            # Performance Dashboard - WHITE TEXT ON DARK
            st.markdown("### 📊 Performance")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="Total Queries",
                    value=str(st.session_state.total_queries)
                )
            with col2:
                cache_rate = (st.session_state.cache_hits / st.session_state.total_queries * 100) if st.session_state.total_queries > 0 else 0
                st.metric(
                    label="Cache Hits",
                    value=f"{cache_rate:.0f}%"
                )
            
            # Cache info
            if st.session_state.cache_manager:
                stats = st.session_state.cache_manager.get_cache_stats()
                st.markdown(f"**💾 Cached Answers**: {stats.get('total_entries', 0)}")
            
            st.markdown("---")
            
            # Actions
            st.markdown("### 🎛️ Quick Actions")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.conversation_history = []
                    st.rerun()
            
            with col2:
                if st.button("💾 Clear Cache", use_container_width=True):
                    if st.session_state.cache_manager:
                        st.session_state.cache_manager.clear_cache()
                        st.session_state.cache_hits = 0
                    st.success("✅ Cache cleared!")
            
            st.markdown("---")
            
            # Example questions - VISIBLE BUTTONS
            st.markdown("### 💡 Quick Questions")
            
            examples = [
                ("💰 What is a deductible?", "What is a deductible in insurance?"),
                ("🚗 Collision coverage?", "How does collision coverage work?"),
                ("❤️ Term vs Whole life?", "What's the difference between term and whole life insurance?"),
                ("🏥 Health insurance?", "What does health insurance cover?"),
                ("📋 File a claim?", "How do I file an insurance claim?")
            ]
            
            for label, question in examples:
                if st.button(label, key=f"ex_{question}", use_container_width=True):
                    if st.session_state.initialized:
                        st.session_state.messages.append({"role": "user", "content": question})
                        st.rerun()
            
            st.markdown("---")
            st.markdown("### 🔧 Debug Settings")
            debug_mode = st.checkbox("Show Retrieval Debug Info", value=False)
            if st.session_state.initialized and debug_mode != st.session_state.get('debug_mode', False):
                st.session_state.rag_engine.set_debug_mode(debug_mode)
                st.session_state.debug_mode = debug_mode
            
            st.markdown("---")
            st.markdown("### ℹ️ About")
            st.markdown("""
            **Version**: Professional 2.0  
            **Status**: Production Ready  
            **Updated**: December 2025
            """)
    
    # Main Content Area
    if not st.session_state.initialized:
        # PROFESSIONAL WELCOME SCREEN
        st.markdown("""
        <div class="welcome-card">
            <div style="font-size: 4.5rem; margin-bottom: 1.5rem;">🏛️</div>
            <h2 style="color: #1e293b; margin-bottom: 1rem; font-size: 2.2rem;">
                Welcome to Insurance Advisory AI
            </h2>
            <p style="font-size: 1.15rem; color: #64748b; line-height: 1.7; margin-bottom: 2rem;">
                Get instant, expert-level answers to all your insurance questions.<br>
                Powered by advanced AI and over 21,000 professional insurance Q&A pairs.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature cards
        st.markdown('<div class="feature-grid">', unsafe_allow_html=True)
        
        features = [
            ("🎯", "Expert Answers", "Professional insurance advice powered by AI"),
            ("⚡", "Lightning Fast", "Smart caching for instant responses"),
            ("📚", "Verified Sources", "Every answer backed by references"),
            ("🧠", "Context Aware", "Remembers your conversation"),
            ("📊", "Performance Tracking", "Monitor response times and efficiency"),
            ("🔒", "100% Private", "No data sent to third parties")
        ]
        
        cols = st.columns(3)
        for idx, (icon, title, desc) in enumerate(features):
            with cols[idx % 3]:
                st.markdown(f"""
                <div class="feature-card">
                    <div class="feature-icon">{icon}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("### 👈 Click **'Initialize AI Assistant'** in the sidebar to begin")
        
        return
    
    # Chat Display
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🤖"
        
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            
            # Show sources
            if message["role"] == "assistant" and message.get("context_docs"):
                with st.expander("📚 View Source References", expanded=False):
                    for i, doc in enumerate(message["context_docs"][:3], 1):
                        display_professional_source(doc, i)
            
            # Show performance
            if message["role"] == "assistant" and message.get("timings"):
                timings = message["timings"]
                from_cache = message.get("from_cache", False)
                
                badges = []
                if from_cache:
                    badges.append(f'<span class="perf-badge cache-badge">💾 Cached • {timings["total"]:.2f}s</span>')
                else:
                    badges.append(f'<span class="perf-badge">⏱️ {timings["total"]:.2f}s total</span>')
                    badges.append(f'<span class="perf-badge">🔍 {timings.get("retrieval", 0):.2f}s search</span>')
                    badges.append(f'<span class="perf-badge">🤖 {timings.get("generation", 0):.2f}s AI</span>')
                
                st.markdown(" ".join(badges), unsafe_allow_html=True)

            # Show RAG / model metrics and mode
            if message["role"] == "assistant":
                model_name = config.OLLAMA_MODEL
                mode = message.get("mode")
                retr = message.get("retrieval_score")
                retr_avg = message.get("retrieval_avg_score")
                faith = message.get("faithfulness_score")
                skipped_reason = message.get("retrieval_skipped_reason")

                if mode == "rag":
                    rag_line = "🧠 **RAG Mode** • Grounded answer from retrieved context"
                    if faith is not None:
                        rag_line += f" • Faithfulness: {faith:.0%}"
                    if retr is not None:
                        rag_line += f" • Top-1 similarity: {retr:.0%}"
                    if retr_avg is not None:
                        rag_line += f" • Avg top-k: {retr_avg:.0%}"
                    rag_line += f" • Model: `{model_name}`"
                    st.markdown(rag_line)

                elif mode == "llm_only":
                    if skipped_reason == "no_retrieval":
                        line = "🧠 **LLM-only Mode** • No relevant knowledge-base entry found — answered using general domain knowledge"
                    elif skipped_reason == "low_relevance" and retr is not None:
                        line = f"🧠 **LLM-only Mode** • RAG disabled due to low relevance (Top-1: {retr:.0%}) — answered using general domain knowledge"
                    else:
                        line = "🧠 **LLM-only Mode** • No relevant knowledge-base entry found — answered using general domain knowledge"
                    line += f" • Model: `{model_name}`"
                    st.markdown(line)

                elif mode == "exact_match":
                    line = "🧠 **RAG Exact Match** • Answer directly from matching dataset entry"
                    if retr is not None:
                        line += f" • Similarity: {retr:.0%}"
                    line += f" • Model: `{model_name}`"
                    st.markdown(line)
    
    # Chat Input
    if prompt := st.chat_input("💬 Ask me about insurance (health, auto, life, home)...", key="chat_input"):
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.total_queries += 1
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            try:
                with st.status("🔍 Processing your question...", expanded=False) as status:
                    # Stateless: Each question is independent (no conversation history)
                    result = st.session_state.rag_engine.process_query(
                        prompt,
                        conversation_history=None  # Stateless by design
                    )
                    
                    if not result:
                        raise ValueError("RAG engine returned None")
                    
                    if not result.get('answer'):
                        raise ValueError(f"RAG engine returned no answer. Result keys: {list(result.keys())}")
                    
                    if result.get('from_cache'):
                        st.session_state.cache_hits += 1
                        status.update(label="✅ Instant answer (from cache)", state="complete")
                    else:
                        status.update(label="✅ Answer generated", state="complete")
                
                # Display answer
                answer = result.get('answer', 'No answer generated.')
                if not answer or len(answer.strip()) == 0:
                    answer = "I apologize, but I couldn't generate an answer. Please try rephrasing your question."
                st.markdown(answer)
                
            except Exception as e:
                error_msg = f"Error generating answer: {str(e)}"
                logger.error(error_msg, exc_info=True)  # Log full traceback
                st.error(f"❌ {error_msg}")
                st.info("💡 **Troubleshooting:**\n- Ensure Ollama is running: `ollama serve`\n- Check that the model is installed: `ollama pull mistral:7b-instruct`\n- Try refreshing the page")
                # Don't stop - show error but continue to save message
                answer = f"Error: {error_msg}. Please try again or check Ollama connection."
                st.markdown(answer)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "error": True
                })
                return  # Exit early but don't stop the app
            
            # Sources (RAG context) - only when RAG was actually used
            if result.get('context_docs'):
                with st.expander("📚 Retrieved Context (RAG Source)", expanded=False):
                    for i, doc in enumerate(result['context_docs'][:3], 1):
                        display_professional_source(doc, i)
            
            # Performance badges
            if result.get('timings'):
                timings = result['timings']
                from_cache = result.get('from_cache', False)
                
                badges = []
                if from_cache:
                    badges.append(f'<span class="perf-badge cache-badge">💾 Cached • {timings["total"]:.2f}s</span>')
                else:
                    badges.append(f'<span class="perf-badge">⏱️ {timings["total"]:.2f}s total</span>')
                    badges.append(f'<span class="perf-badge">🔍 {timings.get("retrieval", 0):.2f}s search</span>')
                    badges.append(f'<span class="perf-badge">🤖 {timings.get("generation", 0):.2f}s generation</span>')
                
                st.markdown(" ".join(badges), unsafe_allow_html=True)

            # Mode / RAG info for the freshly generated answer
            mode = result.get("mode")
            retr = result.get("retrieval_score")
            retr_avg = result.get("retrieval_avg_score")
            faith = result.get("faithfulness_score")
            skipped_reason = result.get("retrieval_skipped_reason")
            model_name = config.OLLAMA_MODEL

            if mode == "rag":
                rag_line = "🧠 **RAG Mode** • Grounded answer from retrieved context"
                if faith is not None:
                    rag_line += f" • Faithfulness: {faith:.0%}"
                if retr is not None:
                    rag_line += f" • Top-1 similarity: {retr:.0%}"
                if retr_avg is not None:
                    rag_line += f" • Avg top-k: {retr_avg:.0%}"
                rag_line += f" • Model: `{model_name}`"
                st.markdown(rag_line)

            elif mode == "llm_only":
                if skipped_reason == "no_retrieval":
                    line = "🧠 **LLM-only Mode** • No relevant knowledge-base entry found — answered using general domain knowledge"
                elif skipped_reason == "low_relevance" and retr is not None:
                    line = f"🧠 **LLM-only Mode** • RAG disabled due to low relevance (Top-1: {retr:.0%}) — answered using general domain knowledge"
                else:
                    line = "🧠 **LLM-only Mode** • No relevant knowledge-base entry found — answered using general domain knowledge"
                line += f" • Model: `{model_name}`"
                st.markdown(line)

            elif mode == "exact_match":
                line = "🧠 **RAG Exact Match** • Answer directly from matching dataset entry"
                if retr is not None:
                    line += f" • Similarity: {retr:.0%}"
                line += f" • Model: `{model_name}`"
                st.markdown(line)
        
        # Save to history (only if we have a valid result)
        if 'result' in locals() and result:
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.get('answer', 'No answer generated.'),
                "context_docs": result.get('context_docs', []),
                "timings": result.get('timings', {}),
                "from_cache": result.get('from_cache', False),
                "faithfulness_score": result.get('faithfulness_score'),
                "retrieval_score": result.get('retrieval_score'),
                "retrieval_avg_score": result.get('retrieval_avg_score'),
                "mode": result.get('mode'),
                "retrieval_skipped_reason": result.get('retrieval_skipped_reason')
            })
            
            st.session_state.conversation_history.append({
                "user": prompt,
                "assistant": result.get('answer', 'No answer generated.')
            })
        
        st.rerun()


if __name__ == "__main__":
    main()

