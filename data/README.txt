
University Student & Academic Knowledge Assistant
=================================================

RAG vector database generated on:
2026-09-16T15:32:37.641570Z

Documents:
6

PDF pages:
31

Chunks:
80

Embedding model:
sentence-transformers/all-MiniLM-L6-v2

Vector database:
FAISS

Chunk size:
900

Chunk overlap:
150

Files:

faiss_index/
    index.faiss
    index.pkl

metadata.json
rag_config.json
documents_manifest.json

The Streamlit application should load the FAISS
database instead of generating embeddings at runtime.
