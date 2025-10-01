# build_vector_db.py
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import streamlit as st

# Path where FAISS DB will be stored
VECTOR_DB_PATH = "hw5_vector_db"

def build_vector_db():
    # Example: load course data from HW4
    courses = [
        "Introduction to Information Science",
        "Data Structures and Algorithms",
        "Database Management Systems",
        "Human-Computer Interaction",
        "Information Retrieval",
        "Digital Libraries",
        "Information Ethics and Policy",
        "Data Visualization",
        "Information Architecture"
    ]

    # Wrap into Documents
    documents = [Document(page_content=course) for course in courses]

    # Create embeddings
    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])

    # Build FAISS DB
    db = FAISS.from_documents(documents, embeddings)

    # Save locally
    db.save_local(VECTOR_DB_PATH)
    print(f"✅ Vector DB built and saved to {VECTOR_DB_PATH}")

if __name__ == "__main__":
    build_vector_db()
