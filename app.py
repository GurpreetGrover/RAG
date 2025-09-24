# import pysqlite3
import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_chroma import Chroma  # A vector database for storing and retrieving embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader

import time
from tqdm import tqdm  # For progress tracking
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

st.title("PDF Question Answering with RAG")

class RateLimitedEmbeddings(MistralAIEmbeddings):
    def embed_documents(self, texts, **kwargs):
        # Override embed_documents to add delay between requests
        embeddings = []
        for text in tqdm(texts, desc="Generating embeddings"):
            embedding = super().embed_documents([text], **kwargs)[0]
            embeddings.append(embedding)
            time.sleep(2)  # Wait for 2 second between requests
        return embeddings

try:
    MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
except (KeyError, FileNotFoundError):
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

embedder = MistralAIEmbeddings(api_key= MISTRAL_API_KEY)


try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    max_retries=2
)

import tempfile

def get_session_retriever(uploaded_file, embedder):
    """
    Gets or creates a retriever for the current session.
    """
    # Create a unique identifier for the uploaded file.
    current_file_id = f"{uploaded_file.name}-{uploaded_file.size}"

    # Check if a retriever is already in the session state for the same file.
    if "retriever" in st.session_state and st.session_state.file_id == current_file_id:
        return st.session_state.retriever

    # If not, create a new one.
    st.info("Creating a new retriever for this session...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    try:
        loader = PyPDFLoader(file_path=tmp_file_path)
        documents = loader.load()
        reviews_vector_db = Chroma.from_documents(
            documents=documents,
            embedding=embedder,
        )
        retriever = reviews_vector_db.as_retriever(k=10)
        
        # Store the new retriever and file identifier in the session state.
        st.session_state.retriever = retriever
        st.session_state.file_id = current_file_id
        return retriever
    except:
        st.write("Something unexpected happened in loading the document or creating the embeddings")
    finally:
        os.remove(tmp_file_path)

# Placeholder for file upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    # This single line now handles all the processing and caching.
    # It will be instant after the first run for a given file.

    # reviews_retriever = create_retriever(uploaded_file, embedder)
    reviews_retriever = get_session_retriever(uploaded_file, embedder)

    # Placeholder for user question
    question = st.text_input("Ask a question about the document:")

    from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
    from langchain.schema.runnable import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser


    # Define the system and human prompt templates and create the ChatPromptTemplate
    review_template_str = """Your job is to use Uploaded documents to answer user queries
    Be as detailed as possible, but don't make up any information that's not from the context.
    If you don't know an answer, say you don't know.

    {context}
    """
    review_system_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["context"],
            template=review_template_str,
        )
    )
    review_human_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["question"],
            template="{question}",
        )
    )
    messages = [review_system_prompt, review_human_prompt]
    review_prompt_template = ChatPromptTemplate(
        input_variables=["context", "question"],
        messages=messages,
    )

    if question: 
        # display the question
        st.write(f"Your question: {question}")

        input_variables = {"context": reviews_retriever, "question": RunnablePassthrough()}
        output_parser = StrOutputParser()
        review_chain = input_variables | review_prompt_template | llm | output_parser

        # Invoke the RAG chain with the user's question
        response = review_chain.invoke(question)

        # Display the generated response
        st.write("Answer:")
        st.write(response)
