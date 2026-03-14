import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ── CONFIGURATION ────────────────────────────────────────────────
# Where your PDFs live
DOCS_FOLDER = "data/sample_docs"

# Where ChromaDB will save the memory
CHROMA_FOLDER = "data/chroma_db"

# The embedding model (converts text to numbers)
# This runs 100% locally on your laptop — no API needed!
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_documents(folder_path):
    """
    ELI5: Walk into the library room and pick up every PDF book
    """
    documents = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"📄 Loading: {filename}")
            
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
            print(f"   ✅ Loaded {len(docs)} pages")
    
    print(f"\n📚 Total pages loaded: {len(documents)}")
    return documents

def chunk_documents(documents):
    """
    ELI5: Cut each book into small flashcards
    Each chunk is ~500 characters with 50 character overlap
    (overlap means chunks share a little text so context isn't lost)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # Each flashcard = 500 characters
        chunk_overlap=50,      # Cards share 50 characters with neighbours
        separators=["\n\n", "\n", ".", " "]  # Where to cut
    )
    
    chunks = splitter.split_documents(documents)
    print(f"✂️  Total chunks created: {len(chunks)}")
    return chunks

def store_in_chromadb(chunks):
    """
    ELI5: Convert each flashcard into numbers and file them
    in the smart filing cabinet (ChromaDB)
    """
    print(f"\n🧠 Loading embedding model: {EMBEDDING_MODEL}")
    print("   (This downloads ~90MB the first time — please wait...)")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}  # Uses CPU (safe for all laptops)
    )
    
    print("✅ Embedding model loaded!")
    print(f"\n📦 Storing {len(chunks)} chunks in ChromaDB...")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_FOLDER
    )
    
    print(f"✅ All chunks stored in ChromaDB at: {CHROMA_FOLDER}")
    return vectorstore

def run_ingestion():
    """
    ELI5: The main function that runs all 3 jobs in order
    """
    print("=" * 50)
    print("🚀 RAG INGESTION PIPELINE STARTING")
    print("=" * 50)
    
    # Job 1 — Read
    print("\n📖 JOB 1: Loading documents...")
    documents = load_documents(DOCS_FOLDER)
    
    if not documents:
        print("❌ No PDFs found! Make sure files are in data/sample_docs/")
        return
    
    # Job 2 — Chop
    print("\n✂️  JOB 2: Chunking documents...")
    chunks = chunk_documents(documents)
    
    # Job 3 — Memorize
    print("\n🧠 JOB 3: Storing in ChromaDB...")
    store_in_chromadb(chunks)
    
    print("\n" + "=" * 50)
    print("✅ INGESTION COMPLETE!")
    print(f"   Your knowledge base is ready.")
    print("=" * 50)

if __name__ == "__main__":
    run_ingestion()