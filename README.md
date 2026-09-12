## Multi-Document RAG AI Agent

A Python-based AI application that allows users to upload multiple documents and ask questions about their content.

The project uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from uploaded documents before generating an answer.

Features

Upload PDF, TXT and CSV files

Extract and split document text into chunks

Generate text embeddings

Store embeddings in ChromaDB

Semantic search

BM25 keyword search

Hybrid search using RRF

Ask questions from selected documents

Ask questions across multiple documents

Page-level source citations

Document comparison

Document summarization

CSV data queries using Pandas

Calculator tool

Chat history

User authentication

RAG evaluation

Tech Stack

Python

FastAPI

LangGraph

LangChain

ChromaDB

BM25

Sentence Transformers

Groq

Llama 3.3 70B

PostgreSQL / SQLite

SQLAlchemy

Pandas

HTML

CSS

JavaScript

How RAG Works

The basic flow of the project is:

Document Upload
↓
Text Extraction
↓
Text Chunking
↓
Embeddings
↓
ChromaDB
↓
User Query
↓
Semantic + BM25 Search
↓
Hybrid Ranking
↓
Relevant Document Chunks
↓
LLM
↓
Answer + Sources

Project Structure

Multi-Document AI Agent/
│
├── backend/
│   └── app/
│       ├── api/
│       ├── agents/
│       ├── embeddings/
│       ├── evaluation/
│       ├── memory/
│       ├── models/
│       ├── rag/
│       ├── retrievers/
│       ├── services/
│       └── tools/
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── data/
├── evaluation/
├── tests/
├── main.py
├── run.py
├── requirements.txt
└── README.md

Running the Project

1. Create virtual environment

python -m venv .venv

2. Activate virtual environment

Windows PowerShell:

.\.venv\Scripts\Activate.ps1

3. Install dependencies

pip install -r requirements.txt

4. Add API key

Create a .env file:

GROQ_API_KEY=your_api_key_here

Do not upload the .env file to GitHub.

5. Run the application

python run.py

Open:

http://localhost:8000

API documentation:

http://localhost:8000/docs

RAG Evaluation

The project includes an evaluation module to check the quality of the RAG system.

The evaluation includes:

Retrieval Precision

Retrieval Recall

Context Relevance

Answer Faithfulness

Answer Correctness

Citation Accuracy

Query Latency

Example Questions

After uploading documents, users can ask questions such as:

What is the candidate's CGPA?

What skills are mentioned in the resume?

Explain RAG from the uploaded document.

Compare the skills in the resume and job description.

What is the average sales in the CSV file?

Purpose

This project was built to understand and implement concepts related to:

Retrieval-Augmented Generation

Vector databases

Embeddings

Hybrid search

LLM applications

LangGraph workflows

Document processing

AI application development



##  Security & Best Practices
- **Strict Grounding**: Zero fact hallucination on private data queries.
- **Safe Tabular Execution**: Controlled Pandas dataframe operations without arbitrary code execution.
- **Input Validation**: Strict file type validation (`.pdf`, `.docx`, `.txt`, `.csv`, `.md`) and size limits.
- **Secure Password Hashing**: Passlib + bcrypt password encryption with JWT session tokens.
- **Privacy First**: Sensitive API keys and document vectors remain safely in local/private boundaries.

## License
MIT License. Built for enterprise GenAI applications and engineering portfolios.

