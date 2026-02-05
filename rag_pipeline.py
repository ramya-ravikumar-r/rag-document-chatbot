from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

# -------- Configure Gemini --------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("models/gemini-2.5-flash")


# -------- Local Embedding --------
class LocalEmbedding(Embeddings):

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


# -------- PDF Processing --------
def process_pdf(pdf_path):

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    embeddings = LocalEmbedding()
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store


# -------- RAG Query Function --------
def ask_question(vector_store, question):

    retriever = vector_store.as_retriever()

    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer the question using ONLY the context below.

    Context:
    {context}

    Question:
    {question}
    """

    response = model.generate_content(prompt)

    return response.text
