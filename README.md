# 🎓 University Academic Knowledge Assistant

A Retrieval Augmented Generation (RAG) based academic knowledge assistant for university students.

The system allows students to ask questions about university policies, academic regulations, procedures, and other institutional information contained in a collection of PDF documents.

The application uses precomputed embeddings and a FAISS vector database, so document embeddings are generated only once during the indexing stage.

## Features

• Retrieval Augmented Generation

• Multiple PDF document knowledge base

• Precomputed FAISS vector database

• Semantic search using Sentence Transformers

• Groq LLM for answer generation

• GPT OSS 120B model

• Page level source tracing

• Document and chunk metadata

• Conversational question answering

• Streamlit web interface

• Deployable on Streamlit Community Cloud

## Architecture

The application follows this pipeline:

User Question
↓
Sentence Transformer
↓
Query Embedding
↓
FAISS Similarity Search
↓
Top K Relevant Chunks
↓
Chunk Text + Metadata
↓
Groq GPT OSS 120B
↓
Answer + Source Citations


## Knowledge Base Creation

The original PDF documents are processed separately in Google Colab.

The indexing pipeline performs the following steps:

1. Download public PDF documents from Google Drive.

2. Extract text from each PDF page.

3. Split page text into smaller chunks.

4. Generate embeddings using Sentence Transformers.

5. Store embeddings in FAISS.

6. Store chunk text and source metadata in JSON.

7. Save the resulting RAG database.

The original PDF documents are not required by the deployed application.


## Repository Structure

```text
university-academic-rag/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── faiss_index/
│   │   └── index.faiss
│   │
│   ├── chunks.json
│   ├── rag_config.json
│   └── documents_manifest.json
│
└── .streamlit/
    └── config.toml
