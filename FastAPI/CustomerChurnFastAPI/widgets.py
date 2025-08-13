import streamlit as st

st.title("Text Inputing example")

whatev = st.text_input("Enter whatever!")

if whatev:
    st.write(f"HERE IS YOUR WRITEN TEXT: {whatev}")
