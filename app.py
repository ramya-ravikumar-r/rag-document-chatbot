import streamlit as st
from rag_pipeline import process_pdf, ask_question

st.title("RAG Document Chatbot")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    # Save PDF temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("Document uploaded successfully!")

    # Process document
    vector_store = process_pdf("temp.pdf")

    # Ask Question
    question = st.text_input("Ask a question from document")

    if question:
        answer = ask_question(vector_store, question)
        st.write(answer)
