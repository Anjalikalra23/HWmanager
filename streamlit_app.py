import streamlit as st
import hw7

def run():
    st.set_page_config(page_title="HW7 - News Bot", page_icon="🗞️", layout="wide")
    hw7.run()

if __name__ == "__main__":
    run()
