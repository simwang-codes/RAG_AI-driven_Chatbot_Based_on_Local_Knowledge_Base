# Load local FAISS index storage, conduct similarity search based on user's query, then create an efficient GPT prompt for OpenAI Chat API
# Before searching similar content in FAISS vector index storage, I need to first embed user's inputr, which allows the program to search similair stuff in FAISS index storage

import os
import langid
from config import DB_PATH, OPENAI_API_KEY
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langdetect import detect
from langchain.schema import Document

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
embedding_model = OpenAIEmbeddings()

def load_vector_storage(path="vector_index"):
    return FAISS.load_local(path,embedding_model, allow_dangerous_deserialization=True)

def search_similar_chunks(query, k=5):
    vector_store = load_vector_storage()
    return vector_store.similarity_search(query, k=k)

def detect_language(text):
    try:
        lang, _ = langid.classify(text)
        return lang
    except:
        return "unknown"

def answer_question_with_prompt(query, retrieved_chunks):
    if not retrieved_chunks:
        return "No releated information can be found! Please try another question."
    
    # Below are three information pieces that will be passed to OpenAI's gpt later

    # This is a list of retrieved chunks based on similarity search
    context_chunks = "\n\n".join(doc.page_content for doc in retrieved_chunks)

    # This is a list of tags of chunks above, extracted from the metadata of those chunks
    # By structuring it using f-string, it will feed these tags to gpt as bullet points
    tags = [f"- {doc.metadata.get('tags', '')}" for doc in retrieved_chunks]

    # Below join all tags above into one string but still maintain their bullet point format
    joined_tags = "\n".join(tags)


    # Prompt starts below:
    language = detect_language(query)
    prompt = f"""You are an intelligent AI assistant built for my RAG Chatbot system. Your duty is to answer the user's question based on the content and tags below, and you need to answer the question with this language the user is using: {language}.
    
    Guidelines for you:
    1. Base your answer ONLY on the provided content and tags, however, if the answer is not explicityly stated, you are allowed to summarize, infer, or connect clues across different parts.
    2. If relevant information exists in any form, do your best to provide a helpful answer that might answer user's question
    3. Only if the content is indeed completely irrelevant with user's question, respond with: "There is no relevant information in the provided content" in user's language: {language}.
    4. If user ask anything about you, for example:"Who are you" or "What is your name", you should answer that:"I am your AI assistant!"

    Tags:
    {joined_tags}

    Content Chunks:
    {context_chunks}

    User's Question: {query}

    Answer:"""

    # Below I set the temperature and model of gpt
    # What is temperature? [Temperature controls randomness or creativity of the model's output.]
    # For instance: temperature = 0.0 → Deterministic, logical, focused. Always gives the most likely next word. temperature >= 1.0 → More creative, varied, and potentially unexpected responses.
    chat = ChatOpenAI(temperature = 0.5, model = "gpt-4-1106-preview")
    response = chat.invoke(prompt)
    return response.content.strip()

if __name__ == "__main__":
    query = input("Please enter your question：\n> ")
    top_chunks = search_similar_chunks(query, k=5)

    answer = answer_question_with_prompt(query, top_chunks)
    print(answer)
