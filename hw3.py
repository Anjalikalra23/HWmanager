#hw3- Anjali Kalra
import streamlit as st
import os
from openai import OpenAI
import google.generativeai as genai
import cohere


def run():
    st.set_page_config(page_title="HW3 – Multi-Vendor Chatbot", page_icon="⚾", layout="wide")
    st.title("HW3 – Streaming Chatbot with Baseball URLs ⚾- Anjali Kalra")
    st.write("Ask questions using one or two URLs. Compare vendors, models, and memory options.")

    
    st.sidebar.header("Options")

    
    url1 = st.sidebar.text_input("Enter URL 1")
    url2 = st.sidebar.text_input("Enter URL 2")

    # Vendor + Model
    vendor = st.sidebar.selectbox("Choose Vendor", ["OpenAI", "Cohere", "Google Gemini"])
    model_name = None
    if vendor == "OpenAI":
        model_name = st.sidebar.radio("OpenAI Model", [
            "gpt-4o",        # flagship
            "gpt-4o-mini"    # cheap
        ])
    elif vendor == "Cohere":
        model_name = st.sidebar.radio("Cohere Model", [
            "command-r-plus-08-2024",   # flagship
            "command-r-08-2024"         # cheap
        ])
    elif vendor == "Google Gemini":
        model_name = st.sidebar.radio("Gemini Model", [
            "gemini-1.5-pro",   # flagship
            "gemini-1.5-flash"  # cheap
        ])

    # Memory option
    memory_option = st.sidebar.selectbox(
        "Conversation Memory Type",
        ["Buffer (6 Qs)", "Conversation Summary", "2000-token Buffer"]
    )

    st.sidebar.write(f"**Current Vendor:** {vendor}")
    st.sidebar.write(f"**Current Model:** {model_name}")
    st.sidebar.write(f"**Memory Mode:** {memory_option}")

    
    try:
        openai_key = st.secrets["OPENAI_KEY"]
    except Exception:
        openai_key = os.getenv("OPENAI_KEY")

    try:
        cohere_key = st.secrets["COHERE_API_KEY"]
    except Exception:
        cohere_key = os.getenv("COHERE_API_KEY")

    try:
        google_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        google_key = os.getenv("GOOGLE_API_KEY")

    
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    
    # Show history
    
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).markdown(msg["content"])

    
    if prompt := st.chat_input("Ask about baseball..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        # Combine URL context
        url_context = ""
        if url1:
            url_context += f"Source1: {url1}\n"
        if url2:
            url_context += f"Source2: {url2}\n"

        question = f"Use these documents if available:\n{url_context}\n\nQuestion: {prompt}"

        
        if vendor == "OpenAI":
            client = OpenAI(api_key=openai_key)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    stream = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": question}],
                        stream=True,
                    )
                    response = st.write_stream(stream)
            st.session_state["messages"].append({"role": "assistant", "content": response})

        elif vendor == "Cohere":
            co = cohere.Client(api_key=cohere_key)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    stream = co.chat_stream(
                        model=model_name,
                        message=question
                    )

                    full_text = ""
                    for event in stream:
                        if event.event_type == "text-generation":
                            st.write(event.text, end="")
                            full_text += event.text

            st.session_state["messages"].append({"role": "assistant", "content": full_text})

        elif vendor == "Google Gemini":
            genai.configure(api_key=google_key)
            model = genai.GenerativeModel(model_name)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response_stream = model.generate_content(
                        question, stream=True
                    )
                    response = st.write_stream(chunk.text for chunk in response_stream if chunk.text)

            st.session_state["messages"].append({"role": "assistant", "content": response})
