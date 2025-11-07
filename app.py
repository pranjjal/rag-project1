import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, YoutubeLoader
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import tempfile
import time
import re

from dotenv import load_dotenv
load_dotenv()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "youtube_processed" not in st.session_state:
    st.session_state.youtube_processed = False
if "document_type" not in st.session_state:
    st.session_state.document_type = None
if "document_name" not in st.session_state:
    st.session_state.document_name = ""

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


def is_valid_youtube_url(url):
    """Check if the URL is a valid YouTube URL"""
    youtube_regex = re.compile(
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\n]{11})'
    )
    return youtube_regex.match(url) is not None


def process_youtube(youtube_url):
    """Process YouTube video and create vector store"""
    try:
        # Load YouTube video transcript
        loader = YoutubeLoader.from_youtube_url(youtube_url)
        documents = loader.load()
        
        if not documents:
            raise ValueError("Could not load transcript from this YouTube video. The video might not have captions available.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Extract text content from chunks for FAISS
        texts = [chunk.page_content for chunk in chunks]
        vector_store = FAISS.from_texts(texts, embeddings)
        print("Vector store created with", len(chunks), "chunks from YouTube video.")
        return vector_store, documents[0].metadata.get('title', 'YouTube Video')
    except Exception as e:
        raise Exception(f"Error processing YouTube video: {str(e)}")


def get_answer(question, vector_store):
    """Get answer from the PDF using RAG"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant analyzing a PDF document. 
        Based on the following context from the document, please provide a clear and concise answer to the question.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
    )

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(final_prompt)
    
    return response.content if hasattr(response, 'content') else str(response)


def display_chat_message(message, is_user=True):
    """Display a chat message with appropriate styling"""
    if is_user:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end; margin: 10px 0;">
                <div style="background-color: #007bff; color: white; padding: 10px 15px; 
                           border-radius: 20px; max-width: 70%; word-wrap: break-word;">
                    <strong>You:</strong> {message}
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
                <div style="background-color: #f1f3f4; color: #333; padding: 10px 15px; 
                           border-radius: 20px; max-width: 70%; word-wrap: break-word;">
                    <strong>🤖 Assistant:</strong> {message}
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )


def main():
    st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📚", layout="wide")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: flex-start;
        color: #1f77b4;
        margin-bottom: 30px;
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">📚 PDF & YouTube Q&A Chatbot</h1>', unsafe_allow_html=True)
    
    # Sidebar for document upload/input
    with st.sidebar:
        st.header("📄 Document Input")
        
        # Option selector
        input_type = st.radio(
            "Choose input type:",
            ["📄 PDF Upload", "🎥 YouTube URL"],
            key="input_type"
        )
        
        if input_type == "📄 PDF Upload":
            uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
            
            if uploaded_file and not st.session_state.pdf_processed:
                with st.spinner("🔄 Processing your PDF..."):
                    st.session_state.vector_store = process_pdf(uploaded_file)
                    st.session_state.pdf_processed = True
                    st.session_state.youtube_processed = False
                    st.session_state.document_type = "PDF"
                    st.session_state.document_name = uploaded_file.name
                    st.success("✅ PDF processed successfully!")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"📄 I've successfully processed your PDF: **{uploaded_file.name}**. You can now ask me questions about its content!"
                    })
            
            if st.session_state.pdf_processed and st.session_state.document_type == "PDF":
                st.success("📄 PDF Ready!")
                st.info(f"**Document:** {st.session_state.document_name}")
                
        elif input_type == "🎥 YouTube URL":
            youtube_url = st.text_input(
                "Enter YouTube URL:",
                placeholder="https://www.youtube.com/watch?v=..."
            )
            
            if youtube_url and not st.session_state.youtube_processed:
                if is_valid_youtube_url(youtube_url):
                    with st.spinner("� Processing YouTube video..."):
                        try:
                            st.session_state.vector_store, video_title = process_youtube(youtube_url)
                            st.session_state.youtube_processed = True
                            st.session_state.pdf_processed = False
                            st.session_state.document_type = "YouTube"
                            st.session_state.document_name = video_title
                            st.success("✅ YouTube video processed successfully!")
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": f"🎥 I've successfully processed the YouTube video: **{video_title}**. You can now ask me questions about its content!"
                            })
                        except Exception as e:
                            st.error(f"❌ Error processing YouTube video: {str(e)}")
                else:
                    st.error("❌ Please enter a valid YouTube URL")
            
            if st.session_state.youtube_processed and st.session_state.document_type == "YouTube":
                st.success("🎥 YouTube Video Ready!")
                st.info(f"**Video:** {st.session_state.document_name}")
        
        st.divider()
        
        # Document status
        if st.session_state.pdf_processed or st.session_state.youtube_processed:
            st.success(f"📄 {st.session_state.document_type} Ready!")
            st.info("💡 **Tip:** Ask specific questions about the content for better answers.")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
            
        # Reset document button
        if st.session_state.pdf_processed or st.session_state.youtube_processed:
            if st.button("🔄 Load New Document"):
                st.session_state.pdf_processed = False
                st.session_state.youtube_processed = False
                st.session_state.vector_store = None
                st.session_state.document_type = None
                st.session_state.document_name = ""
                st.session_state.messages = []
                st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat history container
        chat_container = st.container()
        
        with chat_container:
            # Display chat messages
            if st.session_state.messages:
                for message in st.session_state.messages:
                    display_chat_message(
                        message["content"], 
                        is_user=(message["role"] == "user")
                    )
            else:
                st.markdown(
                    """
                    <div style="text-align: center; color: #666; padding: 50px; 
                               background-color: #fafafa; border-radius: 10px; margin: 20px 0;">
                        <h3>👋 Welcome to PDF & YouTube Q&A Chatbot!</h3>
                        <p>Choose your input type in the sidebar:</p>
                        <p>📄 Upload a PDF document or 🎥 Enter a YouTube URL</p>
                        <p>Once processed, you can ask questions about the content!</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        # Chat input
        if st.session_state.pdf_processed or st.session_state.youtube_processed:
            doc_type = "PDF" if st.session_state.pdf_processed else "YouTube video"
            placeholder_text = f"What is this {doc_type.lower()} about?" if doc_type == "PDF" else "What topics are discussed in this video?"
            
            question = st.text_input(
                f"💬 Ask a question about your {doc_type}:", 
                placeholder=placeholder_text,
                key="user_input"
            )
            
            col_send, col_clear = st.columns([6, 1])
            with col_send:
                send_button = st.button("📤 Send", use_container_width=True)
            
            if (send_button and question) or (question and question != st.session_state.get("last_question", "")):
                # Add user message to chat
                st.session_state.messages.append({"role": "user", "content": question})
                
                # Get AI response
                with st.spinner("🤔 Thinking..."):
                    try:
                        answer = get_answer(question, st.session_state.vector_store)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
                st.session_state.last_question = question
                st.rerun()
        else:
            st.info("📤 Please upload a PDF document or enter a YouTube URL first to start chatting!")
    
    with col2:
        if st.session_state.pdf_processed or st.session_state.youtube_processed:
            st.markdown("### 🎯 Quick Actions")
            
            # Dynamic content based on document type
            doc_type = "document" if st.session_state.pdf_processed else "video"
            content_type = "document" if st.session_state.pdf_processed else "video content"
            
            if st.button(f"📋 Summarize {doc_type.title()}", use_container_width=True):
                summary_question = f"Please provide a comprehensive summary of this {content_type}."
                st.session_state.messages.append({"role": "user", "content": summary_question})
                with st.spinner("📝 Creating summary..."):
                    try:
                        answer = get_answer(summary_question, st.session_state.vector_store)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.rerun()
            
            if st.button("🔑 Key Points", use_container_width=True):
                key_points_question = f"What are the main key points and important information in this {content_type}?"
                st.session_state.messages.append({"role": "user", "content": key_points_question})
                with st.spinner("🎯 Finding key points..."):
                    try:
                        answer = get_answer(key_points_question, st.session_state.vector_store)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.rerun()
            
            # YouTube-specific quick action
            if st.session_state.youtube_processed:
                if st.button("🎬 Video Topics", use_container_width=True):
                    topics_question = "What are the main topics and themes discussed in this video? Please organize them clearly."
                    st.session_state.messages.append({"role": "user", "content": topics_question})
                    with st.spinner("🎬 Analyzing video topics..."):
                        try:
                            answer = get_answer(topics_question, st.session_state.vector_store)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                        except Exception as e:
                            error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.rerun()

if __name__ == "__main__":
    main()