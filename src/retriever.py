import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ── CONFIGURATION ────────────────────────────────────────────────
CHROMA_FOLDER = "data/chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# How many chunks to retrieve per question
TOP_K = 3

def load_vectorstore():
    """
    ELI5: Open the filing cabinet (ChromaDB)
    so we can search through it
    """
    print("📂 Loading ChromaDB...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )
    
    vectorstore = Chroma(
        persist_directory=CHROMA_FOLDER,
        embedding_function=embeddings
    )
    
    print("✅ ChromaDB loaded!")
    return vectorstore

def retrieve_chunks(question, vectorstore):
    """
    ELI5: Search the filing cabinet for the
    most relevant flashcards for this question
    """
    print(f"\n🔍 Searching for: '{question}'")
    
    results = vectorstore.similarity_search(
        query=question,
        k=TOP_K
    )
    
    print(f"✅ Found {len(results)} relevant chunks:")
    for i, chunk in enumerate(results):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source: {chunk.metadata.get('source', 'Unknown')}")
        print(f"Page: {chunk.metadata.get('page', 'Unknown')}")
        print(f"Content: {chunk.page_content[:200]}...")
    
    return results

if __name__ == "__main__":
    # Test the retriever
    vectorstore = load_vectorstore()
    
    # Test question
    question = "How many sick leave days do employees get?"
    chunks = retrieve_chunks(question, vectorstore)