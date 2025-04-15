import streamlit as st
import base64
from search import search_similar_chunks, answer_question_with_prompt

# ✅ Must be FIRST Streamlit command
st.set_page_config(page_title="📚 RAG for The Count of Monte Cristo", layout="wide")

# ✅ Background image function
def set_bg_from_local(image_path):
    with open(image_path, "rb") as img_file:
        base64_img = base64.b64encode(img_file.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{base64_img}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ✅ Set background image
set_bg_from_local("background.jpg")

# ✅ UI content
st.title("🗡️The Count of Monte Cristo Q&A⛰️")
st.markdown("Please ask a question related to The Count of Monte Cristo, I will answer your question! Feel free to use any languages you like!")

query = st.text_input("💬 Type Your Question Here：", placeholder="For instance, Who is Edmond Dantès？")

if st.button("🔍 Query") and query.strip():
    with st.spinner("Searching for an answer..."):
        top_docs = search_similar_chunks(query, k=5)
        answer = answer_question_with_prompt(query, top_docs)

        st.subheader("💬 Answer：")
        st.markdown(answer)

        st.divider()
        st.subheader("🔎 Matched Content Chunks")
        for i, doc in enumerate(top_docs):
            st.markdown(f"**📄 Chunk {i+1}**")
            st.markdown(f"- **Filename**：{doc.metadata.get('filename', '')}")
            st.markdown(f"- **Tags**：{doc.metadata.get('tags', '')}")
            st.code(doc.page_content.strip()[:1000], language="markdown")
