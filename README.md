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
pip install fitz PyMuPDF jieba nltk tqdm langdetect keybert sentence-transformers langchain openai langchain-openai faiss-cpu streamlit

# 2. Get Your Own OpenAI API Key
Get your API key at: https://platform.openai.com/docs/overview

# 3. Delete the downloaded vector_index folder, then run the build_faiss.py
I already did the text chunking/keyword extraction for you in reader.py, so now you have a database in the folder db
You need to delete the downloaded vector_index folder first, then run python3 build_faiss.py in your terminal to embed the data in db into vectors, and you will get your own vector_index folder with two .faiss files

# 4. Run the app
You are ready to interact with your Chatbot now! Run: streamlit run app.py
This RAG Chatbot model was trained with The Count of Monte Cristo written by Alexandre Dumas. Feel free to ask your chatbot any questions related to the story, characters, or anything in this book!
