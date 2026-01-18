import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

## Langsmith Tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With Groq"
# The API Key for Langsmith should be in your .env

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question:{question}")
    ]
)

def generate_response(question, api_key, engine, temperature, max_tokens):
    # Initialize model with Groq provider
    llm = init_chat_model(
        model=engine, 
        model_provider="groq", 
        groq_api_key=api_key, # Pass the key from sidebar here
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

## Title of the app
st.title("Enhanced Q&A Chatbot (Powered by Groq)")

## Sidebar for settings
st.sidebar.title("Settings")
# Updated label to Groq
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

## Select the Groq model
engine = st.sidebar.selectbox("Select Model", [
    "llama-3.3-70b-versatile", 
    "gemma2-9b-it", 
    "deepseek-r1-distill-llama-70b"
])

## Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=1024, value=300)

## Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    if api_key:
        try:
            response = generate_response(user_input, api_key, engine, temperature, max_tokens)
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter the Groq API Key in the sidebar to proceed.")
else:
    st.info("Please provide a question to get started.")