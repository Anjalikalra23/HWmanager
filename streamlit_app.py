import streamlit as st
import subprocess
import os

# ------------------------------------ #
#        APP CONFIGURATION              #
# ------------------------------------ #
st.set_page_config(
    page_title="HW7 – News Info Bot",
    page_icon="📰",
    layout="wide"
)

st.title("📰 HW7 – News Info Bot (RAG + LLM Re-ranking)")
st.markdown("""
### Welcome to your HW7 App!
This bot retrieves and summarizes news information using RAG and LLM re-ranking.  
It uses **OpenAI** as the main model and optionally **Cohere** for comparison.  

---
🪄 *Note:* Ensure your `.streamlit/secrets.toml` file has valid API keys before running.
""")

# ------------------------------------ #
#          RUN HW7.PY FILE             #
# ------------------------------------ #
hw7_path = "hw7.py"

if os.path.exists(hw7_path):
    with st.spinner("Launching HW7 News Bot..."):
        # Run HW7.py directly as a Streamlit app
        subprocess.run(["streamlit", "run", hw7_path])
else:
    st.error("❌ HW7 file not found. Please make sure `hw7.py` exists in your workspace.")
