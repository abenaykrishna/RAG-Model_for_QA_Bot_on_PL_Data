# Financial PDF QA Pipeline

## Overview
The Financial PDF QA bot converts PDFs into conversational databases using LangChain and Llama 3.2 by Ollama. It segments documents into chunks and generates embeddings to identify semantically relevant information based on user queries. Utilizing a retriever chain, it finds contextually relevant financial data and employs Ollama's model to generate accurate responses, enhancing user interaction with financial documents.

## Features
- PDF document processing
- Semantic search capabilities
- Context-aware financial query answering
- Interactive Streamlit interface

## Prerequisites
- Python 3.8+
- Ollama
- llama3.2 model

## Installation
```bash
pip install streamlit langchain faiss-cpu
ollama pull llama3.2
```

## Usage
1. Run the application
```bash
streamlit run financial_qa_bot.py
```

2. Upload PDF
3. Enter financial queries
4. Receive precise, context-specific answers

## Technologies
- LangChain
- Ollama
- FAISS
- Streamlit
- Python

## Key Components
- PDF Loader
- Text Splitter
- Vector Embeddings
- Semantic Retrieval
- Language Model Response Generation

## Configuration
Modify embedding/LLM models in the script as needed.

## Limitations
- Requires local Ollama installation
- Performance depends on document quality
- Best suited for structured financial documents




