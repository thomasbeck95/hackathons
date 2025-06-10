import streamlit as st
import io
import logging

logging.basicConfig(
    filename='app.log',  # Specify the log file name
    filemode='a',        # 'a' for append (default), 'w' for overwrite each run
    level=logging.DEBUG, # Set a lower level for file logs, typically DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set up streamlit page
st.set_page_config(
    page_title="Chat app skeleton.",
    layout="centered",
)

st.title("Chat app skeleton.")
st.markdown("""
    <style>
        .st-emotion-cache-4oy321 {
            text-align: left;
            margin: 10px;
            margin-right: 50px;
            background-color: salmon;
        }
        .st-emotion-cache-janbn0 {
            text-align: right;
            margin: 10px;
            margin-left: 50px;
            background-color: none;

        }
        .st-emotion-cache-jmw8un {
          display: none;
        }
        .st-emotion-cache-4zpzjl {
          display: none;
        }
    </style>
""", unsafe_allow_html=True)
