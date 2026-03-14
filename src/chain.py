import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── CONFIGURATION ────────────────────────────────────────────────
CHROMA_FOLDER = "data/chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── PROMPT TEMPLATE ──────────────────────────────────────────────
PROMPT_TEMPLATE = """
You are a helpful HR assistant for NovaTech Solutions.
Answer the employee's question using ONLY the context provided below.
If the answer is not in the context, say exactly:
"I don't have enough information in the knowledge base to answer this."
Never make up information. Always be precise and professional.

Context from HR documents:
{context}

Employee Question: {question}

Your Answer:"""

def format_docs(docs):
    """
    ELI5: Take all the retrieved chunks and
    join them into one big block of text
    """
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain():
    """
    ELI5: Assemble all the workers into one pipeline
    """
    print("🔧 Building RAG chain...")

    # Load embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )

    # Load ChromaDB
    vectorstore = Chroma(
        persist_directory=CHROMA_FOLDER,
        embedding_function=embeddings
    )

    # Set up retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # Set up Groq LLM
    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0  # 0 = factual, prevents hallucinations!
    )

    # Build prompt
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    # ── THE MODERN LCEL CHAIN ────────────────────────────────────
    # ELI5: This is an assembly line
    # Question goes in → retriever finds chunks →
    # chunks + question go into prompt →
    # Groq reads prompt → answer comes out
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✅ RAG chain ready!")
    return chain, retriever

def ask_question(chain, retriever, question):
    """
    ELI5: Ask a question and get an answer WITH sources
    """
    print(f"\n❓ Question: {question}")
    print("⏳ Thinking...")

    # Get answer
    answer = chain.invoke(question)
    print(f"\n💬 Answer:\n{answer}")

    # Get source citations separately
    source_docs = retriever.invoke(question)
    print(f"\n📚 Sources:")
    seen = []
    for doc in source_docs:
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'Unknown')
        citation = f"  • {source} — Page {page}"
        if citation not in seen:
            seen.append(citation)
            print(citation)

    return answer

if __name__ == "__main__":
    chain, retriever = build_rag_chain()

    questions = [
        "How many sick leave days do employees get?",
        "What is the probation period for new employees?",
        "What happens if I get a performance rating of 1?",
        "How do I apply for leave?"
    ]

    for question in questions:
        ask_question(chain, retriever, question)
        print("\n" + "="*50)