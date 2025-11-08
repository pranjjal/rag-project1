import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, YoutubeLoader
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
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
if "search_mode" not in st.session_state:
    st.session_state.search_mode = "📄 Document Only"  # Initialize with actual radio option

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


def search_web(query):
    """Perform web search using DuckDuckGo"""
    try:
        search = DuckDuckGoSearchAPIWrapper()
        results = search.run(query)
        return results
    except Exception as e:
        return f"Error performing web search: {str(e)}"


def get_web_answer(question):
    """Get answer from web search"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2
    )

    # First, search the web
    search_results = search_web(question)
    
    prompt = PromptTemplate(
        input_variables=["question", "search_results"],
        template="""You are a helpful assistant that provides accurate information based on web search results.
        
        Question: {question}
        
        Web Search Results: {search_results}
        
        Based on the search results above, provide a comprehensive and accurate answer to the question. 
        If the search results don't contain enough information, mention that clearly.
        
        Answer:"""
    )

    final_prompt = prompt.format(question=question, search_results=search_results)
    response = llm.invoke(final_prompt)
    
    return response.content if hasattr(response, 'content') else str(response)


def get_hybrid_answer(question, vector_store, document_type):
    """Get answer combining document content and web search"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2
    )

    # Get document context
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    docs = retriever.invoke(question)
    document_context = "\n".join([doc.page_content for doc in docs])

    # Get web search results
    web_results = search_web(question)
    
    prompt = PromptTemplate(
        input_variables=["question", "document_context", "web_results", "document_type"],
        template="""You are a helpful assistant that can combine information from a {document_type} and web search results.
        
        Question: {question}
        
        Content from {document_type}:
        {document_context}
        
        Additional Web Search Results:
        {web_results}
        
        Please provide a comprehensive answer that:
        1. First uses information from the {document_type} if relevant
        2. Then supplements with web search results for additional context
        3. Clearly indicates which information comes from which source
        4. If there are any contradictions, mention them
        
        Answer:"""
    )

    final_prompt = prompt.format(
        question=question, 
        document_context=document_context, 
        web_results=web_results,
        document_type=document_type
    )
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
    
    st.markdown('<h1 class="main-header">� AI Research Assistant: PDF, YouTube & Web Search</h1>', unsafe_allow_html=True)
    
    # Sidebar for document upload/input
    with st.sidebar:
        st.header("📄 Document Input")
        
        # Option selector
        input_type = st.radio(
            "Choose input type:",
            ["📄 PDF Upload", "🎥 YouTube URL"],
            key="input_type"
        )
        
        st.divider()
        
        # Search mode selector
        st.header("🔍 Search Mode")
        search_mode = st.radio(
            "Choose search mode:",
            ["📄 Document Only", "🌐 Web Search Only", "🔄 Hybrid (Document + Web)"],
            index=0,
            key="search_mode_radio"
        )
        
        # Update session state based on selection
        st.session_state.search_mode = search_mode
        
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
                        <h3>� Welcome to AI Research Assistant!</h3>
                        <p><strong>Multiple ways to get information:</strong></p>
                        <p>📄 Upload a PDF document | 🎥 Enter a YouTube URL | 🌐 Search the web</p>
                        <p><strong>Search Modes:</strong></p>
                        <p>📄 Document Only | 🌐 Web Search Only | 🔄 Hybrid (Document + Web)</p>
                        <p>Choose your preferred mode in the sidebar and start exploring!</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        # Chat input
        # Dynamic input based on search mode
        if st.session_state.search_mode == "🌐 Web Search Only":
            # Web search only mode - no document needed
            question = st.text_input(
                "🌐 Ask any question (Web Search):", 
                placeholder="What is artificial intelligence?",
                key="user_input"
            )
            
            col_send, col_clear = st.columns([6, 1])
            with col_send:
                send_button = st.button("📤 Send", use_container_width=True)
            
            if (send_button and question) or (question and question != st.session_state.get("last_question", "")):
                # Add user message to chat
                st.session_state.messages.append({"role": "user", "content": question})
                
                # Get web search response
                with st.spinner("🔍 Searching the web..."):
                    try:
                        answer = get_web_answer(question)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
                st.session_state.last_question = question
                st.rerun()
                
        elif st.session_state.pdf_processed or st.session_state.youtube_processed:
            # Document-based or hybrid modes
            doc_type = "PDF" if st.session_state.pdf_processed else "YouTube video"
            
            if st.session_state.search_mode == "📄 Document Only":
                placeholder_text = f"What is this {doc_type.lower()} about?" if doc_type == "PDF" else "What topics are discussed in this video?"
                question_label = f"💬 Ask a question about your {doc_type}:"
            else:  # hybrid mode
                placeholder_text = f"Ask about the {doc_type.lower()} or any related topic"
                question_label = f"🔄 Ask about your {doc_type} or get additional web info:"
            
            question = st.text_input(
                question_label, 
                placeholder=placeholder_text,
                key="user_input"
            )
            
            col_send, col_clear = st.columns([6, 1])
            with col_send:
                send_button = st.button("📤 Send", use_container_width=True)
            
            if (send_button and question) or (question and question != st.session_state.get("last_question", "")):
                # Add user message to chat
                st.session_state.messages.append({"role": "user", "content": question})
                
                # Get AI response based on search mode
                with st.spinner("🤔 Thinking..."):
                    try:
                        if st.session_state.search_mode == "📄 Document Only":
                            answer = get_answer(question, st.session_state.vector_store)
                        else:  # hybrid mode
                            answer = get_hybrid_answer(question, st.session_state.vector_store, doc_type)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
                st.session_state.last_question = question
                st.rerun()
        else:
            if st.session_state.search_mode == "🌐 Web Search Only":
                st.info("🌐 Web search is ready! Ask any question to search the internet.")
            else:
                st.info("📤 Please upload a PDF document or enter a YouTube URL first to start chatting!")
    
    with col2:
        st.markdown("### 🎯 Quick Actions")
        
        # Web search quick actions (always available)
        if st.button("🌐 Latest News", use_container_width=True):
            news_question = "What are the latest news and developments in technology and AI?"
            st.session_state.messages.append({"role": "user", "content": news_question})
            with st.spinner("📰 Searching for latest news..."):
                try:
                    answer = get_web_answer(news_question)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.rerun()
        
        if st.button("🔍 Quick Web Search", use_container_width=True):
            search_question = "Tell me about current trends in machine learning and artificial intelligence"
            st.session_state.messages.append({"role": "user", "content": search_question})
            with st.spinner("🔍 Searching the web..."):
                try:
                    answer = get_web_answer(search_question)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.rerun()
        
        # Document-specific actions (when document is loaded)
        if st.session_state.pdf_processed or st.session_state.youtube_processed:
            st.divider()
            st.markdown("### 📄 Document Actions")
            
            # Dynamic content based on document type
            doc_type = "document" if st.session_state.pdf_processed else "video"
            content_type = "document" if st.session_state.pdf_processed else "video content"
            
            if st.button(f"📋 Summarize {doc_type.title()}", use_container_width=True):
                summary_question = f"Please provide a comprehensive summary of this {content_type}."
                st.session_state.messages.append({"role": "user", "content": summary_question})
                with st.spinner("📝 Creating summary..."):
                    try:
                        if st.session_state.search_mode == "📄 Document Only":
                            answer = get_answer(summary_question, st.session_state.vector_store)
                        else:
                            answer = get_hybrid_answer(summary_question, st.session_state.vector_store, doc_type)
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
                        if st.session_state.search_mode == "📄 Document Only":
                            answer = get_answer(key_points_question, st.session_state.vector_store)
                        else:
                            answer = get_hybrid_answer(key_points_question, st.session_state.vector_store, doc_type)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.rerun()
            
            # Hybrid search action
            if st.button("🔄 Enhanced Research", use_container_width=True):
                research_question = f"Provide detailed information about the main topics in this {content_type}, supplemented with the latest web information."
                st.session_state.messages.append({"role": "user", "content": research_question})
                with st.spinner("🔄 Conducting enhanced research..."):
                    try:
                        answer = get_hybrid_answer(research_question, st.session_state.vector_store, doc_type)
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
                            if st.session_state.search_mode == "📄 Document Only":
                                answer = get_answer(topics_question, st.session_state.vector_store)
                            else:
                                answer = get_hybrid_answer(topics_question, st.session_state.vector_store, "video")
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                        except Exception as e:
                            error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.rerun()

if __name__ == "__main__":
    main()