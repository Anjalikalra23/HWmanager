import streamlit as st
import hw4

st.set_page_config(page_title="Homework Manager", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Homework:", ["HW4"])

if page == "HW4":
    hw4.main()
