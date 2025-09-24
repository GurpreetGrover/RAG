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
- **AgentOps**: For monitoring and observability of the RAG agent.

## What is AgentOps?

AgentOps is a tool for monitoring, evaluating, and debugging AI agents. In this project, AgentOps is used to:

- **Track LLM Calls**: Monitor the calls made to the Gemini language model, including the prompts, responses, and token usage.
- **Trace Agent Execution**: Visualize the entire execution flow of the RAG agent, from receiving a query to generating a response.
- **Analyze Performance**: Track latency, cost, and other performance metrics to identify bottlenecks and optimize the agent.
- **Debug Issues**: Quickly identify and debug issues by inspecting the detailed logs and traces provided by AgentOps.

By integrating AgentOps, we gain valuable insights into the behavior and performance of our RAG application, making it easier to maintain, improve, and scale.

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
