import streamlit as st

st.set_page_config(page_title="HCAI HW5", layout="wide")

st.sidebar.title("Navigation")
st.sidebar.write("Select a page:")

st.write("# Welcome to the Multi-Page App 🎓")

st.write("""
This app contains:
- Lab5 (previous assignment)
- HW5 (short-term memory chatbot)
""")

st.sidebar.success("Use the sidebar to switch between Lab5 and HW5 pages.")
