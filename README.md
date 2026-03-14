# 🤖 RAG-Powered Enterprise Knowledge Base

> An AI chatbot that lets employees instantly query company 
> documents and get accurate answers with source citations — 
> built to eliminate the "I know it's somewhere in our docs" problem.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![LangChain](https://img.shields.io/badge/LangChain-Latest-green)
![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.3-orange)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

---

## 🎯 The Problem I Was Solving

Large enterprises store thousands of documents — HR policies, 
compliance manuals, product wikis, onboarding guides. 
Employees waste an average of **2.5 hours per day** searching 
for information across these documents.

Traditional keyword search fails because:
- It matches words, not **meaning**
- It returns documents, not **answers**
- It provides no **source verification**

---

## 💡 The Solution

A RAG (Retrieval-Augmented Generation) system that:
1. Ingests company documents (PDFs, etc.)
2. Stores them in a vector database (ChromaDB)
3. Retrieves the most relevant sections per question
4. Generates precise answers using Llama 3.3 via Groq
5. **Always cites the source page** — zero blind trust in AI

---

## 🏗️ Architecture
