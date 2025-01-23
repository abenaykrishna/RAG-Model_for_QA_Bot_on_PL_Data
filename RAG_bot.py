import os
import time
from datetime import datetime
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Helper function to format time
def current_time():
    return datetime.now().strftime("%H:%M:%S")

# Create vector store save/load functions
def save_vector_store(db, save_path="faiss_index"):
    try:
        db.save_local(save_path)
        st.sidebar.success(f"Vector store saved to {save_path}")
    except Exception as e:
        st.sidebar.error(f"Error saving vector store: {e}")

def load_vector_store(save_path="faiss_index", embeddings=None):
    try:
        if embeddings is None:
            embeddings = OllamaEmbeddings(model="llama3.2")
        return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.sidebar.error(f"Error loading vector store: {e}")
        return None

# Frontend Interface
st.title("Financial Analysis QA Bot")
st.sidebar.title("Upload PDF with P&L Data")

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload your PDF", type=["pdf"])
query = st.text_input("Enter your financial query:", placeholder="What is the gross profit for Q3 2024?")

# Session state to store data
if "retrieval_chain" not in st.session_state:
    st.session_state["retrieval_chain"] = None
if "db" not in st.session_state:
    st.session_state["db"] = None
if "documents" not in st.session_state:
    st.session_state["documents"] = None

# Backend Integration
if uploaded_file:
    if "last_uploaded_file" not in st.session_state or st.session_state["last_uploaded_file"] != uploaded_file.name:
        st.session_state["last_uploaded_file"] = uploaded_file.name

        # Save uploaded file temporarily
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Load and process PDF
        st.sidebar.write("Processing PDF...")
        start_time = current_time()
        load_start = time.time()

        loader = PyPDFLoader("temp_uploaded.pdf")
        st.sidebar.write("Loading PDF...")
        docs = loader.load()

        load_end = time.time()
        end_time = current_time()
        st.sidebar.write(f"Start Time: {start_time}, End Time: {end_time}, Time Taken: {load_end - load_start:.2f} seconds.")

        # Extract and preprocess text
        st.sidebar.write("Splitting text into chunks...")
        start_time = current_time()
        split_start = time.time()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        documents = text_splitter.split_documents(docs)

        split_end = time.time()
        end_time = current_time()
        st.sidebar.write(f"Start Time: {start_time}, End Time: {end_time}, Time Taken: {split_end - split_start:.2f} seconds.")

        # Create embeddings
        st.sidebar.write("Creating document embeddings...")
        start_time = current_time()
        embed_start = time.time()

        embedding = OllamaEmbeddings(model="llama3.2")
        db = FAISS.from_documents(documents, embedding)
        
        # Save vector store
        save_vector_store(db)

        embed_end = time.time()
        end_time = current_time()
        st.sidebar.write(f"Start Time: {start_time}, End Time: {end_time}, Time Taken: {embed_end - embed_start:.2f} seconds.")

        # Define LLM and prompt
        st.sidebar.write("Initializing language model and prompt template...")
        start_time = current_time()
        init_start = time.time()

        llm = OllamaLLM(model="llama3.2")
        prompt = ChatPromptTemplate.from_template("""
            You are a financial analysis bot designed to answer questions based on a provided Profit and Loss (P&L) statement. 
            Analyze the financial document context and provide a detailed, precise answer.

            Context: {context}

            Question: {input}

            Provide a clear, concise financial analysis based on the document's content.
        """)

        init_end = time.time()
        end_time = current_time()
        st.sidebar.write(f"Start Time: {start_time}, End Time: {end_time}, Time Taken: {init_end - init_start:.2f} seconds.")

        # Create chains
        st.sidebar.write("Creating document and retrieval chains...")
        start_time = current_time()
        chain_start = time.time()

        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = db.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        chain_end = time.time()
        end_time = current_time()
        st.sidebar.write(f"Start Time: {start_time}, End Time: {end_time}, Time Taken: {chain_end - chain_start:.2f} seconds.")

        # Save to session state
        st.session_state["documents"] = documents
        st.session_state["db"] = db
        st.session_state["retrieval_chain"] = retrieval_chain

# Query processing
if query and st.session_state["retrieval_chain"]:
    # Get response
    st.sidebar.write("Processing query...")
    start_time = current_time()
    query_start = time.time()

    response = st.session_state["retrieval_chain"].invoke({"input": query})

    query_end = time.time()
    end_time = current_time()
    st.sidebar.write(f"Start Time: {start_time}, End Time: {end_time}, Time Taken: {query_end - query_start:.2f} seconds.")

    # Display results
    retrieved_text = response['context'][0].page_content if response['context'] else "No relevant data found."
    generated_answer = response['answer']

    st.subheader("Retrieved Financial Data Segment")
    st.text(retrieved_text)

    st.subheader("Generated Answer")
    st.text(generated_answer)

# Optional: Load existing vector store if needed
st.sidebar.header("Vector Store Options")
if st.sidebar.button("Load Existing Vector Store"):
    loaded_db = load_vector_store()
    if loaded_db:
        st.session_state["db"] = loaded_db
        st.sidebar.success("Vector store loaded successfully!")

# Cleanup temporary file
if os.path.exists("temp_uploaded.pdf"):
    os.remove("temp_uploaded.pdf")