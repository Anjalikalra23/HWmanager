import streamlit as st
from openai import OpenAI
import os
from PyPDF2 import PdfReader
import sys


__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

chroma_db_path = "./ChromaDB_for_lab"
chroma_client = chromadb.PersistentClient(chroma_db_path)
collection = chroma_client.get_or_create_collection("Lab4Collection")

def add_to_collection(collection, text, filename):
    """Adds a document and its embedding to the ChromaDB collection."""
    openai_client = st.session_state.openai_client
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
    collection.add(documents=[text], ids=[filename], embeddings=[embedding])

def extract_text_from_pdf(file_path):
    """Extracts all text from a given PDF file."""
    try:
        pdf_reader = PdfReader(file_path)
        text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

def setup_document_collection(collection, pdf_file_path, pdf_filename):
    """
    Clears the collection and ingests a new document.
    This ensures we only process and embed each document once per selection.
    """
    with st.spinner(f"Processing and embedding '{pdf_filename}'... This happens only once per document."):
        count = collection.count()
        if count > 0:
            ids_to_delete = collection.get(limit=count)['ids']
            collection.delete(ids=ids_to_delete)

        text = extract_text_from_pdf(pdf_file_path)
        if text:
            add_to_collection(collection, text, pdf_filename)
            st.session_state.processed_pdf = pdf_filename
            st.success(f"'{pdf_filename}' is now ready for questions.", icon="✅")
        else:
            st.error("Failed to process the document.")

# --- Main Application Logic ---
def main():
    if 'openai_client' not in st.session_state:
        try:
            st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client. Check API key. Error: {e}")
            st.stop()

    st.title("Lab 4- VectorDb- Anjali Kalra")
    st.write("Switch between chatting with your documents (RAG) or having a general conversation.")

    st.sidebar.header("Settings")
    chat_mode = st.sidebar.radio(
        "Choose your chat mode:",
        ("Document Q&A (RAG)", "General Chat")
    )

    if chat_mode == "Document Q&A (RAG)":
        st.sidebar.subheader("Document Selection")
        pdf_dir_path = "src"
        if not os.path.isdir(pdf_dir_path):
            st.sidebar.error(f"Directory '{pdf_dir_path}' not found.")
            st.stop()
        
        pdf_files = [f for f in os.listdir(pdf_dir_path) if f.endswith(".pdf")]
        if not pdf_files:
            st.sidebar.warning("No PDF files found in 'src' directory.")
            st.stop()
            
        selected_pdf = st.sidebar.selectbox("Select a PDF file to chat with:", pdf_files)

        if "processed_pdf" not in st.session_state or st.session_state.processed_pdf != selected_pdf:
            full_pdf_path = os.path.join(pdf_dir_path, selected_pdf)
            setup_document_collection(collection, full_pdf_path, selected_pdf)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Main Chat Logic with Debugging ---
    if prompt := st.chat_input("Ask your question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- Mode 1: Document Q&A (RAG) ---
        if chat_mode == "Document Q&A (RAG)":
            st.write("--- DEBUG: RAG Mode Activated ---")
            with st.spinner("Searching for relevant information..."):
                st.write("DEBUG: Creating embedding for the prompt...")
                openai_client = st.session_state.openai_client
                query_response = openai_client.embeddings.create(input=prompt, model="text-embedding-3-small")
                query_embedding = query_response.data[0].embedding
                st.write("DEBUG: Prompt embedding created successfully.")

                st.write("DEBUG: Querying the vector database...")
                results = collection.query(query_embeddings=[query_embedding], n_results=1)
                st.write("DEBUG: Database query successful.")
                
                retrieved_context = results['documents'][0][0] if results.get('documents') and results['documents'][0] else "No relevant content found."
                
                st.write(f"DEBUG: Retrieved context size: {len(retrieved_context)} characters.")

                system_prompt_text = f"""
                You are an expert assistant. Answer the user's question using ONLY the context below.
                - If you use the context, start with "According to the document...".
                - If the answer isn't in the context, say so clearly.
                """
                final_prompt_for_api = f"CONTEXT FROM DOCUMENT:\n{retrieved_context}\n\nUSER'S QUESTION: {prompt}"
                
                messages_for_api = [
                    {"role": "system", "content": system_prompt_text},
                    {"role": "user", "content": final_prompt_for_api}
                ]

        # --- Mode 2: General Chat ---
        else:
            st.write("--- DEBUG: General Mode Activated ---")
            system_prompt_text = "You are a helpful AI assistant."
            messages_for_api = [{"role": "system", "content": system_prompt_text}] + st.session_state.messages
        
        # --- Unified API Call and Response Handling ---
        try:
            st.write("DEBUG: Calling the final LLM for a response...")
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                client = st.session_state.openai_client
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages_for_api,
                    max_tokens=2048,
                )
                full_response_content = response.choices[0].message.content
                message_placeholder.markdown(full_response_content)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response_content})
                st.write("DEBUG: Process finished successfully!")

        except Exception as e:
            st.error(f"An error occurred during the final API call: {e}")

if __name__ == "__main__":
    main()