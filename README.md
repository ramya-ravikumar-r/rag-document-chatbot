# RAG Document Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot that allows users to upload documents and ask questions based on document content.

## Features
- Document chunking and embedding generation
- Semantic search using FAISS vector database
- Context-aware response generation using Google Gemini LLM
- Interactive Streamlit UI for real-time Q&A
- Reduced hallucinations by grounding responses in retrieved content

## Tech Stack
Python, LangChain, FAISS, Sentence Transformers, Google Gemini LLM, Streamlit

## Workflow
1. Upload document
2. Document chunking and embedding generation
3. Semantic similarity search
4. LLM response generation
5. Interactive chatbot interface
