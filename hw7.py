import os
import pandas as pd
import numpy as np
import streamlit as st
from tqdm import tqdm
from openai import OpenAI
import cohere
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# ---------------------------------------------------
# Streamlit Config
# ---------------------------------------------------
def run():
 st.set_page_config(page_title="HW7 – News Reporting Bot", layout="wide")
 st.title("📰 HW7 – News Reporting Bot")
 st.markdown("""
Given a CSV file of news stories, this bot answers questions  
and finds the most interesting or topic-specific news items.  
Designed for a **global law firm**.
""")

# ---------------------------------------------------
# Load API Keys from secrets.toml
# ---------------------------------------------------
def get_key(name: str):
    try:
        return st.secrets[name]
    except Exception:
        st.error(f"❌ Missing {name} in `.streamlit/secrets.toml`.")
        st.stop()

openai_key = get_key("OPENAI_API_KEY")
cohere_key = st.secrets.get("COHERE_API_KEY", None)

# ---------------------------------------------------
# File Paths
# ---------------------------------------------------
CSV_PATH = "vectordb/Example_news_info_for_testing.csv"
VECTOR_DB_PATH = "vectordb/hw7_vector_db"

# ---------------------------------------------------
# Initialize Clients
# ---------------------------------------------------
openai_client = OpenAI(api_key=openai_key)
cohere_client = cohere.Client(api_key=cohere_key) if cohere_key else None

# ---------------------------------------------------
# Load or Build Vector Database
# ---------------------------------------------------
@st.cache_resource
def load_or_build_vector_db():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)

    if os.path.exists(VECTOR_DB_PATH):
        st.success("✅ Loaded existing HW7 vector database.")
        return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)

    if not os.path.exists(CSV_PATH):
        st.error("❌ CSV file not found. Please upload Example_news_info_for_testing.csv to vectordb folder.")
        st.stop()

    st.info("⚙️ Building FAISS vector database...")
    df = pd.read_csv(CSV_PATH)
    text_cols = [c for c in df.columns if df[c].dtype == "object"]
    combined_texts = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).tolist()

    all_embeddings = []
    batch_size = 50
    for i in tqdm(range(0, len(combined_texts), batch_size)):
        batch = combined_texts[i:i + batch_size]
        all_embeddings.extend(embeddings.embed_documents(batch))

    db = FAISS.from_embeddings(
        text_embeddings=list(zip(combined_texts, np.array(all_embeddings))),
        embedding=embeddings
    )
    db.save_local(VECTOR_DB_PATH)
    st.success("✅ Vector database built successfully.")
    return db

vector_db = load_or_build_vector_db()

# ---------------------------------------------------
# Retrieve Ranked News
# ---------------------------------------------------
def retrieve_ranked_news(query, k=5):
    results = vector_db.similarity_search_with_score(query, k=k)
    ranked_output = []
    for i, (doc, score) in enumerate(results, start=1):
        ranked_output.append(f"**Rank {i}** — (Similarity Score: {score:.4f})\n{doc.page_content.strip()}\n")
    return "\n\n".join(ranked_output)

# ---------------------------------------------------
# Summarization Functions
# ---------------------------------------------------
def summarize_with_openai(model, query, context):
    prompt = f"""
You are a professional news summarizer for a global law firm.
Summarize and rank the following news items by relevance.

Query: {query}

Context:
{context}
"""
    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ OpenAI Error: {str(e)}"

def summarize_with_cohere(model, query, context):
    if not cohere_client:
        return "⚠️ Cohere API key missing."
    prompt = f"""
You are a professional summarizer for a global law firm.
Summarize the ranked news concisely.

Query: {query}
Context:
{context}
"""
    try:
        resp = cohere_client.generate(
            model=model,
            prompt=prompt,
            max_tokens=300,
            temperature=0.7,
        )
        return resp.generations[0].text.strip()
    except Exception as e:
        return f"⚠️ Cohere Error: {str(e)}"

# ---------------------------------------------------
# Sidebar: Vendor + Model Selection
# ---------------------------------------------------
st.sidebar.header("⚙️ Configuration")

vendor_options = ["OpenAI", "Cohere"]

openai_models = {
    "GPT-3.5 Turbo": "gpt-3.5-turbo",
    "GPT-4o-mini": "gpt-4o-mini",
    "GPT-4o": "gpt-4o"
}

cohere_models = {
    "Command-R (08-2024)": "command-r-08-2024",
    "Command-A (03-2025)": "command-a-03-2025"
}

# Vendor selection
vendor = st.sidebar.selectbox("Select Vendor", vendor_options)

# Dynamically show models based on vendor
if vendor == "OpenAI":
    llm_label = st.sidebar.selectbox("Select Model", list(openai_models.keys()), index=1)
    llm_model = openai_models[llm_label]
else:
    llm_label = st.sidebar.selectbox("Select Model", list(cohere_models.keys()), index=0)
    llm_model = cohere_models[llm_label]

# ---------------------------------------------------
# Chat Interface
# ---------------------------------------------------
st.markdown("---")
st.subheader("💬 Chat with the News Info Bot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input("Ask about the news (e.g., 'find the most interesting news', 'find news about AI')...")

if query:
    with st.spinner(f"Analyzing with {vendor} ({llm_label})..."):
        ranked_context = retrieve_ranked_news(query, k=5)

        # Use selected vendor
        if vendor == "OpenAI":
            summary = summarize_with_openai(llm_model, query, ranked_context)
        else:
            summary = summarize_with_cohere(llm_model, query, ranked_context)

        # Display results
        st.subheader("🏆 Ranked News Results")
        st.markdown(ranked_context)

        st.markdown(f"### 🤖 {vendor} – {llm_label}")
        st.markdown(summary)

        st.session_state.chat_history.append({
            "vendor": vendor,
            "model": llm_label,
            "query": query,
            "response": summary
        })

# ---------------------------------------------------
# Sidebar Bottom
# ---------------------------------------------------
st.sidebar.button("🗑️ Clear Conversation", on_click=lambda: st.session_state.clear())
    
if __name__ == "__main__":
    run()