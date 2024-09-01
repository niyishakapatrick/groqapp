
import pandas as pd
import os
import streamlit as st
from pandasai import SmartDataframe
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Initialize the model
llm = ChatGroq(
    model_name="mixtral-8x7b-32768",
    api_key=os.environ.get("groq_api_key")  # Use .get() to avoid KeyError
)

uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head(3))

    # Initialize SmartDataframe
    df = SmartDataframe(data, config={'llm': llm})
    
    prompt = st.text_input("Enter your prompt:")
    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating response..."):
                try:
                    # Generate response
                    response = df.chat(prompt)
                    st.write(response)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a prompt.")
else:
    st.info("Please upload a CSV file to proceed.")
