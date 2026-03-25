# RAG-Based Document Chatbot

An end-to-end Retrieval Augmented Generation (RAG) chatbot that answers questions from uploaded PDF documents — using offline semantic embeddings, FAISS vector search, and Google Gemini 2.5 Flash as the generative LLM. Deployed live via Streamlit.


# Problem Statement
Large language models hallucinate when asked questions outside their training data. RAG solves this by grounding the model's responses in a specific document — making it accurate, verifiable, and domain-specific without expensive fine-tuning.
Use Case: Upload any PDF (research paper, policy document, product manual) and ask questions in plain English. The system retrieves the most relevant chunks and generates a precise, document-faithful answer.

# System Architecture
PDF Upload
    │
    ▼
PyPDFLoader → Text Extraction
    │
    ▼
RecursiveCharacterTextSplitter
(chunk_size=500, chunk_overlap=50)
    │
    ▼
Sentence Transformers → Local Embeddings
(offline — no external API cost)
    │
    ▼
FAISS Vector Index → Semantic Search
    │
    ▼
Top-K Relevant Chunks → Structured Prompt
    │
    ▼
Gemini 2.5 Flash → Grounded Answer
    │
    ▼
Streamlit UI → User Response

# Key Features
FeatureDetailOffline EmbeddingsSentence Transformers — no external embedding API neededVector StoreFAISS — sub-second semantic retrievalLLMGoogle Gemini 2.5 Flash — fast, cost-efficient, document-groundedChunking StrategyRecursiveCharacterTextSplitter (size 500, overlap 50)DeploymentStreamlit — interactive web UIHallucination ControlContext injection via structured prompt template

# Tech Stack
Python LangChain FAISS Sentence Transformers Google Gemini 2.5 Flash PyPDFLoader Streamlit HuggingFace

# How to Run
bashgit clone https://github.com/ramya-ravikumar-r/rag-document-chatbot
cd rag-document-chatbot


# Add your Gemini API key
export GOOGLE_API_KEY="your_api_key_here"

streamlit run app.py

# Why RAG Over Fine-tuning?
ApproachCostAccuracy on Private DocsUpdate SpeedFine-tuningHigh (GPU, time)GoodSlow (retrain)Prompt stuffingMediumLimited by context windowFastRAG ✅LowHigh (grounded)Instant (re-index)

# Possible Extensions

Multi-document support with source attribution
Conversational memory (multi-turn Q&A)
Swap Gemini for a fully local LLM (Ollama + Llama 3)
