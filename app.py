import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import tempfile

from dotenv import load_dotenv
load_dotenv()

print("Google API Key:", os.getenv("GOOGLE_API_KEY"))

def process_pdf(pdf_file):
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_file_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Extract text content from chunks for FAISS
        texts = [chunk.page_content for chunk in chunks]
        vector_store = FAISS.from_texts(texts, embeddings)
        print("Vector store created with", len(chunks), "chunks.")
        return vector_store
    finally:
        # Clean up the temporary file
        os.unlink(tmp_file_path)


def qa(vector_store):
    #llm bnaya
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="You are a helpful assistant. Given the context: {context}, answer the question: {question}"
    )

    question = st.text_input("Enter your question:")

    if question:
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        docs = retriever.invoke(question)
        context = "\n".join([doc.page_content for doc in docs])

        final_prompt = prompt.format(context=context, question=question)
        response = llm.invoke(final_prompt)

        st.write("Answer:", response)


def main():
    st.title("📚 PDF Research Q&A Bot")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        st.info("Processing your PDF...")
        vector_store = process_pdf(uploaded_file)
        st.success("✅ PDF processed successfully!")

        qa(vector_store)

if __name__ == "__main__":
    main()