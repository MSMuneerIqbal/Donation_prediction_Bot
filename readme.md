# Donation Prediction ChatBot
# Author: [Muneer Iqbal]
# Date: 2025-02-22

# Project Overview

The Donation Prediction Chatbot is an interactive web application built with Streamlit and powered byLangChain's Retrieval-Augmented Generation (RAG) framework and Google’s Gemini LLM. It allows users to upload a CSV file (donations.csv) containing historical donation data, input a donor’s name, and retrieve both past donations (January 2024 - November 2024) and predicted donations for January, February, and March 2025. The app leverages embeddings and a vector store to fetch relevant data, with the LLM analyzing patterns to provide forecasts.


This project was developed to predict future donations based on historical data stored in a CSV file (`donations.csv`). Unlike traditional statistical models (e.g., Random Forest, ARIMA), it uses a modern AI approach with embeddings and an LLM to:

* Retrieve historical donation records for a specified donor.
* Predict future donations by analyzing patterns in the data using Google Gemini’s natural language processing capabilities.

The app is designed for ease of use, featuring a chat-style interface where users can interact with the chatbot to get insights about donor contributions.

## Features

* **CSV Upload:** Upload `donations.csv` via a sidebar interface.
* **Chat-Style UI:** Input donor names in a chat input box and receive responses in a conversational format.
* **Historical Data Retrieval:** Displays donations from January 2024 to November 2024 for any donor in the dataset.
* **Future Predictions:** Predicts donations for January, February, and March 2025 based on historical trends.
* **Streaming Responses:** Outputs are streamed character-by-character for a dynamic chatbot experience.
* **Embeddings with RAG:** Uses LangChain’s RAG framework to create embeddings and retrieve relevant donor data efficiently.

## How It Works

### Data Ingestion:

* Users upload `donations.csv`, which contains columns like "Members Name" and "2024-01" to "2024-11".
* The CSV is loaded into memory using `CSVLoader` and processed into text documents.

### Embedding Creation:

* Upon clicking "Submit," the app splits the CSV data into chunks and generates embeddings using Google’s `embedding-001` model.
* These embeddings are stored in a vector database for fast retrieval.

### Query Processing:

* Users enter a donor’s name (e.g., "Muneeb Salman") in the chat input.
* A structured prompt combines a system persona ("Donation Predictor Expert") and a user query to fetch historical data and predict future donations.
* The vector store retrieves relevant donor data, and the LLM (Gemini) processes it.

### Response Generation:

* The LLM lists historical donations and predicts 2025 values based on identified patterns.
* The response is streamed to the UI in a formatted chat message.

# Usage

1.  **Launch the App:**
    Run the command above to start the Streamlit server.

2.  **Upload CSV:**
    In the sidebar, upload `donations.csv` under "Upload CSV."

3.  **Create Embeddings:**
    Click the "Submit" button in the sidebar to process the CSV and generate embeddings.

4.  **Chat with the Bot:**
    In the chat input box at the bottom, type a donor’s name (e.g., "Muneeb Salman") and press Enter.
    The chatbot will respond with historical donations and predictions for 2025.

# Technical Details

## Dependencies

* **Streamlit:** Web app framework for the UI.
* **Pandas, NumPy:** Data handling (minimal use as LangChain processes CSV directly).
* **LangChain:**
    * `CSVLoader`: Loads CSV into documents.
    * `VectorstoreIndexCreator`: Creates embeddings and vector store.
    * `GoogleGenerativeAIEmbeddings`: Generates embeddings with `embedding-001`.
    * `PromptTemplate`: Structures LLM prompts.
* **Google Gemini:** `gemini-1.5-pro` LLM for pattern analysis and prediction (temperature=0.3 for deterministic output).
* **dotenv:** Loads environment variables.

## Architecture

* **Data Flow:** CSV → Documents → Embeddings → Vector Store → LLM Query → Response.
* **Persona:** "Donation Predictor Expert" guides the LLM to act as a confident forecaster.
* **Prompt:** Combines system instructions and user query to fetch and predict donations.

## Code Structure

* **Sidebar:** File upload and "Submit" button.
* **Main:** Chat interface with `st.chat_input` and streaming responses.
* **Processing:** On-demand embedding creation and RAG-based querying.

## Limitations

* **Prediction Accuracy:** LLM-based predictions rely on pattern recognition rather than statistical modeling (e.g., ARIMA), potentially less precise.
* **Data Dependency:** Requires a well-formatted `donations.csv` with "Members Name" and "2024-01" to "2024-11" columns.
* **Embedding Size:** Large CSV files may exceed chunk limits (500 characters); adjust `chunk_size` if needed.
* **No Visualizations:** Current version lacks graphical plots of donation trends.

## Future Improvements

* **Hybrid Approach:** Integrate statistical models (e.g., Random Forest, ARIMA) alongside LLM predictions for comparison.
* **Graphical Output:** Add Matplotlib plots to visualize historical and predicted donations.
* **Download Option:** Include a button to download predictions as CSV.
* **Error Handling:** Enhance robustness for malformed CSV files.
* **Model Tuning:** Experiment with different Gemini models or temperature settings for better forecasting.

## Contributing

Contributions are welcome! To contribute:

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/new-feature`).
3.  Commit changes (`git commit -m "Add new feature"`).
4.  Push to the branch (`git push origin feature/new-feature`).
5.  Open a pull request.

Please ensure code follows PEP 8 standards and includes comments for clarity.

## License

This project is licensed under the MIT License—see the `LICENSE` file for details (create one if needed).