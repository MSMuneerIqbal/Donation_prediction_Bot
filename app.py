import streamlit as st
import pandas as pd
import numpy as np
from langchain_community.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import tempfile
import time

# Load environment variables
load_dotenv()

# Streamlit app title
st.title("Donation Prediction Chatbot with Gemini")

# Sidebar for file upload
st.sidebar.title("Upload CSV")
uploaded_file = st.sidebar.file_uploader("Choose your donations.csv file", type="csv")

# Function to stream response
def stream_response(response_text):
    formatted_text = response_text  # LLM formats it as per prompt
    streamed_text = ""
    for char in formatted_text:
        streamed_text += char
        yield streamed_text
        time.sleep(0.01)

# System prompt for persona
system_prompt = """
You are the "Donation Predictor Expert," an AI assistant specializing in financial forecasting. Your role is to analyze historical donation data, identify trends, and predict future donations with confidence. Given a donor's past donations from January 2024 to November 2024, you will:
1. List all historical donations clearly and concisely.
2. Analyze the data to identify patterns (e.g., trends, seasonality, irregularities).
3. Predict donations for January, February, and March 2025 based on these patterns, using reasonable extrapolation.
4. Provide the predictions in a clear format, noting any uncertainties but avoiding overly cautious disclaimers unless the data is completely unpredictable.
Use your analytical skills to make informed predictions, even with sparse or irregular data.
"""

# User prompt template
user_prompt_template = PromptTemplate(
    input_variables=["donor_name"],
    template="""
For the donor {donor_name}, retrieve all historical donations from January 2024 to November 2024 from the provided data. Then, analyze the donation patterns and predict their donations for January, February, and March 2025. Present the results in the following format:
- Historical Donations:
  - 2024-01: [amount]
  - 2024-02: [amount]
  - 2024-03: [amount]
  - 2024-04: [amount]
  - 2024-05: [amount]
  - 2024-06: [amount]
  - 2024-07: [amount]
  - 2024-08: [amount]
  - 2024-09: [amount]
  - 2024-10: [amount]
  - 2024-11: [amount]
- Predicted Donations:
  - 2025-01: [predicted_amount]
  - 2025-02: [predicted_amount]
  - 2025-03: [predicted_amount]
"""
)

# Initialize LLM and memory
llm = GoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.3)
memory = ConversationBufferWindowMemory(k=5)

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Process uploaded file
if uploaded_file is not None:
    st.sidebar.success("Uploaded successfully!")

    # Load and preprocess CSV
    with st.spinner("Loading CSV..."):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name
            
            loader = CSVLoader(file_path=temp_file_path)
            documents = loader.load()
            os.remove(temp_file_path)
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            loader = None

    if loader:
        # Create embeddings on button click
        if st.sidebar.button("Submit"):
            with st.spinner("Creating Embeddings..."):
                embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                index_creator = VectorstoreIndexCreator(
                    embedding=embedding,
                    text_splitter=text_splitter
                )
                store = index_creator.from_documents(documents)
                st.session_state["vector_store"] = store
                st.sidebar.success("Embeddings created successfully!")

        # Check if embeddings are created
        if "vector_store" in st.session_state:
            st.subheader("Chat with Donation Data")

            # Display chat history
            for chat in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.markdown(f"**You:** {chat['question']}")
                with st.chat_message("assistant"):
                    st.markdown(f"**Bot:** {chat['response']}")

            # Chat input for donor name
            user_input = st.chat_input("Enter donor name (e.g., 'Muneeb Salman'):")

            if user_input:
                with st.chat_message("user"):
                    st.markdown(f"**You:** {user_input}")
                with st.spinner("Fetching donation data and predicting..."):
                    try:
                        # Format the prompt with the donor name
                        formatted_prompt = user_prompt_template.format(donor_name=user_input)
                        
                        # Combine system and user prompts
                        full_prompt = f"{system_prompt}\n\n{formatted_prompt}"
                        
                        # Query the vector store with the full prompt
                        response = st.session_state["vector_store"].query(full_prompt, llm=llm, memory=memory)
                        
                        # Store and stream response
                        st.session_state.chat_history.append({"question": user_input, "response": response})
                        with st.chat_message("assistant"):
                            response_placeholder = st.empty()
                            for streamed_text in stream_response(response):
                                response_placeholder.markdown(streamed_text)
                    except Exception as e:
                        st.error(f"Error during query: {e}")
        else:
            st.info("Please click 'Submit' to create embeddings.")
else:
    st.info("Please upload a CSV file in the sidebar to proceed.")