# Read through data base, embedding, then build faiss index file for similarity research 
# will be using OpenAi API for embedding, use LangChain to integreate OpenAi API

import sqlite3
import os
from tqdm import tqdm
from config import DB_PATH, OPENAI_API_KEY
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document # 当在未来处理多个documents时，给每个chunk添加metadata, 方便辨识

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY # This is an environment variable, it allows LangChain and OpenAI's python SDK to look for the API key in this environment

embedding_model = OpenAIEmbeddings() # Creates an embedding model wrapper in LangChain that uses OpenAI's text-embedding models

def load_all_from_db(db_path = DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT chunk_index, chunks, tags FROM document")
    rows = cursor.fetchall()
    conn.close()

    # The structure below will help my search.py extract chunks, and tags from chunks metadata
    document = []
    for chunk_index, chunks, tags in tqdm(rows, desc= "Building document list"):
        metadata = {
            "chunk_index": chunk_index,
            "tags": tags
        }
        
        document.append(Document(page_content = chunks, metadata=metadata)) # Reads all stored chunked text + tags, and wraps each into a LangChain Document with metadata
    return document

def build_faiss_index(document, save_path = "vector_index"):
    print("Vectorizing contents...")
    vector_store = FAISS.from_documents(document, embedding_model) # Here I used OpenAI model

    os.makedirs(save_path, exist_ok=True)
    vector_store.save_local(save_path)
    print(f"Vector index has been saved to {save_path}")

if __name__ == "__main__":
    docs = load_all_from_db()
    build_faiss_index(docs)
