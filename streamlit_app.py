import streamlit as st
import hw1, hw2

st.set_page_config(page_title="Homework Manager", layout="wide")

nav = st.navigation({
    "HW Manager": [
        st.Page(hw1.run, title="HW 1", url_path="lab-1"),
        st.Page(hw2.run, title="HW 2 (Default)", url_path="lab-2", default=True),
    ]
})

nav.run()