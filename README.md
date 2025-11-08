# 🤖 AI Research Assistant: Multi-Modal RAG Chatbot

A powerful **Streamlit-based chatbot** that combines **PDF analysis**, **YouTube transcript processing**, and **web search capabilities** using **RAG (Retrieval-Augmented Generation)** technology.

---

## ✨ Features

### 📄 Multi-Source Input Support
- **PDF Upload:** Extract and analyze content from PDF documents  
- **YouTube Integration:** Process video transcripts from YouTube URLs  
- **Web Search:** Access real-time information from the internet  

---

### 🔍 Three Intelligent Search Modes
- **📄 Document Only:** Focus exclusively on your uploaded content  
- **🌐 Web Search Only:** Query the internet without needing documents  
- **🔄 Hybrid Mode:** Combine document insights with current web information  

---

### 🤖 Advanced AI Capabilities
- **RAG Implementation:** Uses FAISS vector storage for semantic search  
- **Google Gemini Integration:** Powered by `gemini-2.5-flash` model  
- **Smart Context Combination:** Intelligently merges document and web sources  
- **Source Attribution:** Clearly indicates information sources  

---

### 💬 Modern Chat Interface
- **Real-time Chat:** Interactive conversation with chat history  
- **Custom Styling:** Beautiful chat bubbles and responsive design  
- **Quick Actions:** Pre-built buttons for common queries  
- **Dynamic UI:** Adapts based on selected search mode  

---

## 🛠️ Technology Stack

| Component | Technology |
|------------|-------------|
| **Framework** | Streamlit |
| **LLM** | Google Generative AI (Gemini 2.5 Flash) |
| **Vector Database** | FAISS |
| **Embeddings** | HuggingFace Sentence Transformers |
| **Document Processing** | LangChain Community (PyPDF, YouTube Loader) |
| **Web Search** | DuckDuckGo Search API |
| **Text Processing** | LangChain Text Splitters |

---

### 🚀 Run Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-research-assistant.git

# Navigate to project directory
cd ai-research-assistant

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
