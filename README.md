# 🧠 Local RAG Chatbot with FAISS and LangChain

This project is a **local Retrieval-Augmented Generation (RAG) chatbot** powered by **LangChain**, **FAISS**, and **OpenAI LLMs**. 
It allows you to upload documents, process and store them as vector embeddings, and interact with them through a chatbot interface — all locally, without using external databases.

---

## Key Features

- Convert and chunk documents (TXT or PDF)
- Extract keywords using `jieba`, `nltk`, and `KeyBERT`
- Build a **FAISS index** from embedded chunks using `OpenAIEmbeddings`
- Interact with the documents using a chatbot UI interface via `Streamlit`
- Simple modular structure to customize or extend

---

## ⚙️ Tech Stack & Libraries

- **LangChain** - Text chunking, prompt handling, and OpenAI API integration  
- **OpenAIEmbeddings** - Text vectorization using OpenAI models  
- **FAISS** - Efficient similarity search over vector embeddings  
- **Streamlit** - Interactive web UI  
- **KeyBERT**, **nltk**, **jieba** - Multilingual keyword extraction  
- **SQLite3** - Local metadata storage  
- **fitz (PyMuPDF)**, **tqdm**, **sentence-transformers** - File processing and visualization

---

## 🧭 Workflow Overview

### 1. Document Reading
- `reader.py` reads `.pdf` or `.txt` files and converts them to plain text.
- Uses `langdetect` and either `jieba` (Chinese) or `nltk` (English) for keyword extraction.

### 2. Chunking & Vectorization
- `build_faiss.py` chunks the text using LangChain’s `RecursiveCharacterTextSplitter`.
- Embeds each chunk using `OpenAIEmbeddings`.
- Stores vector data into FAISS index and metadata into SQLite database.

### 3. Semantic Search
- `search.py` embeds user queries and retrieves top-matching text chunks from FAISS.
- The matching chunks are returned for LLM context input.

### 4. Chatbot Interface
- `app.py` runs a chatbot in a `Streamlit` UI.
- Users ask questions, retrieve document context, and get responses from OpenAI's `ChatOpenAI` model.

---

## 📦 Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
