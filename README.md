# RAG: Streamlit Retrieval-Augmented Generation Application

## Project Description

This project is an implementation of a Retrieval-Augmented Generation (RAG) application using Streamlit and Python. RAG combines the capabilities of modern language models with external document retrieval, enabling more accurate, context-rich, and up-to-date responses by leveraging both generative AI and relevant data sources.

## Key Features

- **Streamlit Interface**: Simple, interactive web UI for querying and viewing results.
- **Retrieval-Augmented Generation**: Combines generative AI with document or knowledge base retrieval for better answers.
- **Customizable Data Sources**: Easily adapt to different document stores or retrieval backends.
- **Real-Time Responses**: Immediate feedback and results via Streamlit.
- **Extensible Architecture**: Modular codebase for integrating new models and retrievers.

## Technologies Used

- **Langchain**: Framework to create smooth pipeline GenAI solution
- **Python**: Main programming language.
- **Streamlit**: For building the interactive web application.
- **Chroma Vector Database**: For document retrieval 
- **Mistral Embeddings and Gemini Language Model APIs**: Integrates with models from Gemini and Mistral AI

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/GurpreetGrover/RAG.git
   cd RAG
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## Usage

- Create secrets.toml file in root directory and declare `Mistral_API_key` and `GOOGLE_API_KEY` variables for accessing respective platforms
- Open the Streamlit app in your browser (usually at `http://localhost:8501`).
- Upload the desired pdf document
- Enter your query in the provided input box.
- View the generated answer, which combines retrieved context with generative AI output.

---

Feel free to fork and extend this project for your own domain-specific retrieval-augmented generation needs!
