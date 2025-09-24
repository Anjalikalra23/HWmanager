#hw4- Anjali Kalra
import os
import io
import pickle
from dataclasses import dataclass
from typing import List, Tuple

import streamlit as st
from bs4 import BeautifulSoup
import faiss
from openai import OpenAI
import google.generativeai as genai
import cohere as cohere_sdk
import numpy as np


APP_TITLE = "HW4 – iSchool RAG Chatbot - Anjali Kalra"
DATA_DIR = os.path.join("data")
HTML_DIR = os.path.join(DATA_DIR, "html")
VDB_DIR = os.path.join(DATA_DIR, "vectordb")
INDEX_PATH = os.path.join(VDB_DIR, "faiss.index")
DOCS_PATH = os.path.join(VDB_DIR, "docs.pkl")

TOP_K = 4
MEMORY_TURNS = 5

OPENAI_MODELS = {
    "Cheap (OpenAI)": "gpt-5-mini",
    "Flagship (OpenAI)": "gpt-5",
}
GEMINI_MODELS = {
    "Cheap (Gemini Flash)": "gemini-1.5-flash",
    "Flagship (Gemini Pro)": "gemini-1.5-pro",
}
COHERE_MODELS = {
    "Cheap (Cohere Command R)": "command-r-08-2024",
    "Flagship (Cohere Command R+)": "command-r-plus-08-2024",
}

VENDORS = [
    ("OpenAI", OPENAI_MODELS),
    ("Gemini", GEMINI_MODELS),
    ("Cohere", COHERE_MODELS),
]

@dataclass
class DocChunk:
    source: str
    chunk_id: int
    text: str


OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
COHERE_API_KEY = st.secrets["COHERE_API_KEY"]
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")  # unused but ready

def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)


def embed_texts(texts: List[str]):
    client = get_openai_client()
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in resp.data]


@st.cache_resource(show_spinner=False)
def load_html_files(html_dir: str) -> List[Tuple[str, str]]:
    files_text = []
    if not os.path.exists(html_dir):
        return files_text
    for fname in sorted(os.listdir(html_dir)):
        if fname.lower().endswith((".html", ".htm")):
            fpath = os.path.join(html_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    soup = BeautifulSoup(f, "html.parser")
                    for tag in soup(["script", "style", "nav", "footer", "header"]):
                        tag.decompose()
                    text = " ".join(soup.get_text(" ", strip=True).split())
                    files_text.append((fname, text))
            except Exception as e:
                st.warning(f"Failed to parse {fname}: {e}")
    return files_text

def chunk_into_two(text: str) -> Tuple[str, str]:
    """Split document into 2 halves by word count (simple chunking)."""
    words = text.split()
    mid = max(1, len(words) // 2)
    return " ".join(words[:mid]), " ".join(words[mid:])

def ensure_vector_db():
    """Create FAISS index once if it does not exist."""
    os.makedirs(VDB_DIR, exist_ok=True)
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        return
    files_text = load_html_files(HTML_DIR)
    if not files_text:
        st.warning("No HTML files found in data/html. Please add the provided iSchool pages there.")
        return
    docs: List[DocChunk] = []
    for fname, text in files_text:
        c1, c2 = chunk_into_two(text)
        docs.append(DocChunk(source=fname, chunk_id=0, text=c1))
        docs.append(DocChunk(source=fname, chunk_id=1, text=c2))
    texts = [d.text for d in docs]
    embs = embed_texts(texts)
    dim = len(embs[0])
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embs, dtype="float32"))
    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(docs, f)

@st.cache_resource(show_spinner=False)
def load_vector_db():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH)):
        return None, None
    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, "rb") as f:
        docs: List[DocChunk] = pickle.load(f)
    return index, docs

def retrieve(query: str, index, docs: List[DocChunk], top_k=TOP_K) -> List[DocChunk]:
    if index is None or docs is None:
        return []
    q_emb = embed_texts([query])
    D, I = index.search(np.array(q_emb, dtype="float32"), top_k)
    return [docs[idx] for idx in I[0] if 0 <= idx < len(docs)]


def main():
    st.title(APP_TITLE)

    ensure_vector_db()
    index, docs = load_vector_db()

    
    with st.sidebar:
        st.header("Settings")
        vendor = st.selectbox("Choose Vendor", [v for v, _ in VENDORS])
        if vendor == "OpenAI":
            model = st.selectbox("Model", list(OPENAI_MODELS.values()))
        elif vendor == "Gemini":
            model = st.selectbox("Model", list(GEMINI_MODELS.values()))
        else:
            model = st.selectbox("Model", list(COHERE_MODELS.values()))

    if "chat" not in st.session_state:
        st.session_state.chat = []

    
    for msg in st.session_state.chat[-(2*MEMORY_TURNS):]: #buffer
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about iSchool student organizations…")
    if user_input:
        st.session_state.chat.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        
        retrieved = retrieve(user_input, index, docs, top_k=TOP_K)
        context = "\n\n".join([f"[{c.source}] {c.text[:500]}" for c in retrieved])

        system_prompt = (
            "You are an assistant for Syracuse iSchool student organizations. "
            "Use ONLY the retrieved context to answer. If unsure, say you don't know.\n"
            f"CONTEXT:\n{context}"
        )

        answer = None

        if vendor == "OpenAI":
            client = get_openai_client()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_input}],
            )
            answer = resp.choices[0].message.content

        elif vendor == "Gemini":
            genai.configure(api_key=GEMINI_API_KEY)
            model_gen = genai.GenerativeModel(model)
            resp = model_gen.generate_content([system_prompt, user_input])
            answer = resp.text

        elif vendor == "Cohere":
            co = cohere_sdk.Client(COHERE_API_KEY)
            resp = co.chat(model=model, message=f"{system_prompt}\n\n{user_input}")
            answer = resp.text

        
        if answer:
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.chat.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()

