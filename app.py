import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Add src folder to path so we can import chain.py
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from chain import build_rag_chain, ask_question

load_dotenv()

# ── PAGE CONFIG ──────────────────────────────────────────────────
st.set_page_config(
    page_title="NovaTech HR Assistant",
    page_icon="🤖",
    layout="wide"
)

# ── HEADER ───────────────────────────────────────────────────────
st.title("🤖 NovaTech HR Knowledge Base")
st.markdown("Ask any question about NovaTech's HR policies and get instant answers with source citations.")
st.divider()

# ── LOAD RAG CHAIN (once, cached) ────────────────────────────────
# ELI5: @st.cache_resource means "build the chain ONCE
# and reuse it for every question — don't rebuild every time!"
@st.cache_resource
def load_chain():
    with st.spinner("🔧 Loading AI knowledge base... (first load takes ~30 seconds)"):
        chain, retriever = build_rag_chain()
    return chain, retriever

chain, retriever = load_chain()

# ── CHAT HISTORY ─────────────────────────────────────────────────
# ELI5: st.session_state is like a notepad that remembers
# everything typed in this session
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.sources = []

# ── DISPLAY CHAT HISTORY ─────────────────────────────────────────
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        # Show sources under assistant messages
        if message["role"] == "assistant" and i // 2 < len(st.session_state.sources):
            sources = st.session_state.sources[i // 2]
            if sources:
                with st.expander("📚 View Sources"):
                    for source in sources:
                        st.caption(source)

# ── CHAT INPUT ───────────────────────────────────────────────────
if question := st.chat_input("Ask an HR question... e.g. 'How many sick days do I get?'"):

    # Show user message
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base..."):
            answer = chain.invoke(question)
            source_docs = retriever.invoke(question)

        st.write(answer)

        # Build source citations
        seen = []
        sources = []
        for doc in source_docs:
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            citation = f"📄 {source} — Page {page}"
            if citation not in seen:
                seen.append(citation)
                sources.append(citation)

        # Show sources in expander
        if sources:
            with st.expander("📚 View Sources"):
                for source in sources:
                    st.caption(source)

    # Save to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.sources.append(sources)

# ── SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **RAG-Powered HR Assistant**
    
    This tool uses Retrieval-Augmented Generation (RAG) to answer questions from NovaTech's HR policy documents.
    
    **How it works:**
    1. 📄 Your question is converted to numbers
    2. 🔍 ChromaDB finds relevant policy sections  
    3. 🤖 Groq AI generates a precise answer
    4. 📚 Sources are cited automatically
    
    **Powered by:**
    - LangChain
    - ChromaDB  
    - Groq (Llama 3.3)
    - HuggingFace Embeddings
    """)
    
    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.sources = []
        st.rerun()