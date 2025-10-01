import streamlit as st
import os
import json
from typing import List, Dict, Optional
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document


VECTOR_DB_PATH = "hw5_vector_db"

def create_or_load_vector_database(force_rebuild: bool = False):
    """Build or load FAISS vector database."""
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

    if force_rebuild or not os.path.exists(VECTOR_DB_PATH):
        
        documents = [
            Document(page_content="iSchool Data Science Club organizes hackathons and networking events."),
            Document(page_content="Human-Computer Interaction course covers usability and UX design."),
            Document(page_content="Information Ethics course discusses privacy, data ethics, and AI bias."),
            Document(page_content="Digital Libraries course teaches digital preservation and access."),
        ]
        db = FAISS.from_documents(documents, embeddings)
        db.save_local(VECTOR_DB_PATH)
        return db
    else:
        
        return FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

def search_vector_database(db, query: str, n_results: int = 3):
    """Search FAISS vector database and return documents + metadata."""
    results = db.similarity_search(query, k=n_results)
    return {
        "documents": [[r.page_content for r in results]],
        "metadatas": [[{"filename": f"doc_{i}"} for i in range(len(results))]]
    }

CHROMADB_AVAILABLE = True  


def get_relevant_club_info(query: str, n_results: int = 3) -> str:
    try:
        if "ischool_vectorDB" not in st.session_state:
            return json.dumps({"error": "Knowledge base not initialized"})
        
        vector_db = st.session_state.ischool_vectorDB
        results = search_vector_database(vector_db, query, n_results=n_results)
        
        if not results or not results.get("documents"):
            return json.dumps({"error": "No relevant information found"})
        
        context_docs = results["documents"][0]
        source_files = [m["filename"] for m in results["metadatas"][0]]
        
        formatted_info = {
            "relevant_information": context_docs,
            "sources": list(set(source_files)),
            "query": query
        }
        return json.dumps(formatted_info, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Error retrieving information: {str(e)}"})


CLUB_INFO_FUNCTION = {
    "name": "get_relevant_club_info",
    "description": "Retrieves relevant information about iSchool student organizations, clubs, programs, and opportunities from the knowledge base",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "n_results": {"type": "integer", "default": 3}
        },
        "required": ["query"]
    }
}


def get_response_with_function_calling(
    client: OpenAI,
    user_input: str,
    conversation_history: List[Dict],
    model: str = "gpt-3.5-turbo"
) -> tuple[str, Optional[str]]:
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant for the iSchool at Syracuse University. "
                           "You specialize in providing information about student organizations, "
                           "academic programs, and opportunities."
            }
        ]
        if conversation_history:
            for msg in conversation_history[-10:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        messages.append({"role": "user", "content": user_input})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            functions=[CLUB_INFO_FUNCTION],
            function_call="auto",
            temperature=0.7,
            max_tokens=500
        )
        
        message = response.choices[0].message
        sources = None
        
        if message.function_call:
            function_args = json.loads(message.function_call.arguments)
            query = function_args.get("query", user_input)
            n_results = function_args.get("n_results", 3)
            
            knowledge_result = get_relevant_club_info(query, n_results)
            try:
                result_data = json.loads(knowledge_result)
                sources = ", ".join(result_data.get("sources", []))
            except:
                pass
            
            messages.append(message.model_dump())
            messages.append({
                "role": "function",
                "name": "get_relevant_club_info",
                "content": knowledge_result
            })
            
            final_response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            return final_response.choices[0].message.content, sources
        else:
            return message.content, None
    except Exception as e:
        return f"Error generating response: {str(e)}", None


def run():
    st.set_page_config(page_title="HW5 - Enhanced iSchool Chatbot", page_icon="🎓", layout="wide")
    st.title("🎓 HW5 - Enhanced iSchool Chatbot with Function Calling- Anjali Kalra")
    st.markdown("Ask questions about iSchool organizations using intelligent function-based retrieval")
    
    openai_api_key = (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None) or os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("❌ OPENAI_API_KEY not found.")
        return
    
    openai_client = OpenAI(api_key=openai_api_key)
    
    st.sidebar.header("⚙️ Configuration")
    model_options = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]
    selected_model = st.sidebar.selectbox("Select Model:", model_options)
    
    memory_size = st.sidebar.slider("Conversation History (pairs)", 0, 10, 5)
    rebuild = st.sidebar.button("🔄 Rebuild Vector Database")
    
    if "ischool_vectorDB" not in st.session_state or rebuild:
        with st.spinner("Loading knowledge base..."):
            vector_db = create_or_load_vector_database(force_rebuild=rebuild)
            if vector_db:
                st.session_state.ischool_vectorDB = vector_db
                st.sidebar.success("✅ Knowledge base ready")
            else:
                st.error("❌ Failed to initialize knowledge base")
                return
    
    if "hw5_conversation" not in st.session_state:
        st.session_state.hw5_conversation = []
    
    st.header("💬 Chat Interface")
    for message in st.session_state.hw5_conversation:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.caption(f"📚 Sources: {message['sources']}")
    
    user_input = st.chat_input("Ask me about iSchool organizations...")
    if user_input:
        st.session_state.hw5_conversation.append({"role": "user", "content": user_input})
        with st.chat_message("assistant"):
            with st.spinner(f"Thinking with {selected_model}..."):
                response_text, sources = get_response_with_function_calling(
                    openai_client, user_input, st.session_state.hw5_conversation[:-1], model=selected_model
                )
                st.markdown(response_text)
                if sources:
                    st.caption(f"📚 Sources: {sources}")
                st.session_state.hw5_conversation.append({"role": "assistant", "content": response_text, "sources": sources})
        
        max_messages = memory_size * 2
        if len(st.session_state.hw5_conversation) > max_messages:
            st.session_state.hw5_conversation = st.session_state.hw5_conversation[-max_messages:]
    
    if st.button("🗑️ Clear Conversation"):
        st.session_state.hw5_conversation = []
        st.rerun()

if __name__ == "__main__":
    run()

