import os
import sqlite3
import fitz
import jieba
import nltk
import re
from tqdm import tqdm #用来生成进度条
from config import DB_PATH
from langdetect import detect
from collections import Counter
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from langdetect.lang_detect_exception import LangDetectException
from langchain.text_splitter import RecursiveCharacterTextSplitter
# 上面的方程教程在：https://python.langchain.com/docs/how_to/recursive_text_splitter/

# Detect file type, if is pdf, then convert to txt, if its txt, then start chunking and keyword tagging, and save to db

def detect_file_type(file):
    file_type = os.path.splitext(file)
    return file_type[1]

def split_text_into_chunks(text, chunk_size = 700, chunk_overlap = 100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators = ["\n\n", "\n", ".", "!", "！", "，", "?", "？","。"]
    )
    return text_splitter.split_text(text)

def generate_chinese_tags(text, top_k = 5):
    text = re.sub(r"[^\u4e00-\u9fffA-Za-z ]", "", text)
    words = jieba.lcut(text) + nltk.word_tokenize(text)
    counter = Counter(w for w in words if len(w) > 1)
    common = counter.most_common(top_k)
    return [word for word, _ in common]

def create_db(db_path=DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path) # Do not pass DB=path into this
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS document(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_index INTEGER,
            chunks TEXT,
            tags TEXT
                   )
    ''')
    conn.commit
    return conn

def extract_text(file):
    if detect_file_type(file) == ".pdf":
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text() + "\n\n"
    elif detect_file_type(file) == ".txt":
        with open(file, "r", encoding = "utf-8") as f:
            text = f.read()
    else:
        raise ValueError("Please only use .pdf or .txt file!")
    return text

# Main pipeline: read file --> chunk file --> extract keywords --> save to database
model = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model)

def file_to_db(file, db_path = DB_PATH):
    text = extract_text(file)
    print(f"Processing file: {file}")

    chunks = split_text_into_chunks(text)
    print(f"Text split into {len(chunks)} chunks")

# 是中文就用中文版方案一的方式，是英文就用后来的方式
    tags_list = []
    for chunk in tqdm(chunks, desc = "Extracting tags"): #这种for loop 语句是用来生成progress bar的
        chunk = chunk.strip()
        
        if not chunk:
            tags_list.append([])  # Optional: store empty tags for empty chunk
            continue

        try:
            lang = detect(chunk)
        except LangDetectException:
            lang = "unknown"
        except Exception as e:
            print(f"Language detection error: {e}")
            lang = "unknown"

        if lang == "zh-cn":
            tags = generate_chinese_tags(chunk)
        else:
            keywords = kw_model.extract_keywords(chunk, top_n=5)
            tags = [kw for kw, _ in keywords]

        tags_list.append(tags)
        
# 然后存入数据库！

    conn = create_db(db_path)
    cursor = conn.cursor()
    insert_data = []

    for i, (chunk, tags) in tqdm(enumerate(zip(chunks, tags_list)), total = len(chunks), desc = "Saving to DB"):
        insert_data.append((i,chunk,",".join(tags)))

    cursor.executemany('''
        INSERT INTO document (chunk_index, chunks, tags) VALUES (?,?,?)''', insert_data)
        
    conn.commit()
    conn.close()
    print(f"Mission Complete! Data has been inserted with {len(insert_data)} chunks")

if __name__ == "__main__":
    file_path = input("📎 Enter the path to your file (.pdf or .txt):\n> ")
    file_to_db(file_path)
