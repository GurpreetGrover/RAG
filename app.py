import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_chroma import Chroma  # A vector database for storing and retrieving embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
import agentops
import os

import time
from tqdm import tqdm  # For progress tracking
import streamlit as st
import os

st.title("PDF Question Answering with RAG")

class RateLimitedEmbeddings(MistralAIEmbeddings):
    def embed_documents(self, texts, **kwargs):
        # Override embed_documents to add delay between requests
        embeddings = []
        for text in tqdm(texts, desc="Generating embeddings"):
            embedding = super().embed_documents([text], **kwargs)[0]
            embeddings.append(embedding)
            time.sleep(2)  # Wait for 1.5 second between requests
        return embeddings


embedder = MistralAIEmbeddings(api_key=st.secrets['Mistral_API_key'])

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
agentops.init(st.secrets["AGENTOPS_API_KEY"])

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    max_retries=2
)

import tempfile

# Placeholder for file upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")


if uploaded_file:
    with st.spinner("Processing the document"):
        
        # 1. Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
            st.write(f"Uploaded file: {uploaded_file.name}")


        try:
            with st.spinner("processing embeddings"):
                # 2. Instantiate PyPDFLoader and load documents
                loader = PyPDFLoader(file_path=tmp_file_path)
                documents = loader.load()
                
                # 4. Create a Chroma vector database from the loaded documents and the embedder
                reviews_vector_db = Chroma.from_documents(
                    documents=documents,
                    embedding=embedder,
                )

                # 5. Create a retriever from the Chroma database
                reviews_retriever = reviews_vector_db.as_retriever(k=10)

        except:
            print("Something unexpected happened in loading the document")

    # Here you will add the code to process the PDF and answer the question
    # For now, just display the uploaded file name
    st.write(f"embeddings stored")
    
    # Add your RAG logic here
    # For demonstration, let's just display a placeholder response
    # st.write("Placeholder Answer: This is where the answer from the RAG model will appear.")


# Placeholder for user question
    question = st.text_input("Ask a question about the document:")

    from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
    from langchain.schema.runnable import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser


    # 6. Define the system and human prompt templates and create the ChatPromptTemplate
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
        #display the question
        st.write(f"Your question: {question}")

        input_variables = {"context": reviews_retriever, "question": RunnablePassthrough()}
        output_parser = StrOutputParser()
        review_chain = input_variables | review_prompt_template | llm | output_parser

        # 9. Invoke the RAG chain with the user's question
        response = review_chain.invoke(question, config={'callbacks': [agentops.LangchainCallbackHandler()]})

        # 10. Display the generated response
        st.write("Answer:")
        st.write(response)
