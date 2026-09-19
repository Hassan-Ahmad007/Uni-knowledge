import json
import os
from pathlib import Path

import streamlit as st
from groq import Groq

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

APP_TITLE = "University Student & Academic Knowledge Assistant"

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

FAISS_DIR = DATA_DIR / "faiss_index"

METADATA_PATH = DATA_DIR / "metadata.json"

CONFIG_PATH = DATA_DIR / "rag_config.json"

MANIFEST_PATH = DATA_DIR / "documents_manifest.json"

DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

GROQ_MODEL = "openai/gpt-oss-120b"

TOP_K = 5


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎓",
    layout="wide",
)


# ============================================================
# PAGE STYLING
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #666666;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .source-box {
        padding: 0.8rem;
        border: 1px solid #dddddd;
        border-radius: 8px;
        margin-bottom: 0.6rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD RAG CONFIGURATION
# ============================================================

@st.cache_data
def load_rag_config():

    if not CONFIG_PATH.exists():
        return {}

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


rag_config = load_rag_config()

EMBEDDING_MODEL_NAME = rag_config.get(
    "embedding_model",
    DEFAULT_EMBEDDING_MODEL,
)


# ============================================================
# LOAD SOURCE METADATA
# ============================================================

@st.cache_data
def load_metadata():

    if not METADATA_PATH.exists():
        return []

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


metadata_records = load_metadata()


def build_metadata_lookup(records):

    lookup = {}

    if isinstance(records, list):

        for record in records:

            vector_index = record.get(
                "vector_index"
            )

            if vector_index is not None:

                lookup[int(vector_index)] = record

    return lookup


METADATA_LOOKUP = build_metadata_lookup(
    metadata_records
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )


# ============================================================
# LOAD EXISTING FAISS DATABASE
# ============================================================

@st.cache_resource
def load_vectorstore():

    if not FAISS_DIR.exists():

        raise FileNotFoundError(
            f"FAISS directory not found: {FAISS_DIR}"
        )

    index_file = FAISS_DIR / "index.faiss"

    pickle_file = FAISS_DIR / "index.pkl"

    if not index_file.exists():

        raise FileNotFoundError(
            f"FAISS index file not found: {index_file}"
        )

    if not pickle_file.exists():

        raise FileNotFoundError(
            f"FAISS metadata file not found: {pickle_file}"
        )

    embeddings = load_embedding_model()

    vectorstore = FAISS.load_local(
        str(FAISS_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vectorstore


# ============================================================
# LOAD DOCUMENT MANIFEST
# ============================================================

@st.cache_data
def load_manifest():

    if not MANIFEST_PATH.exists():
        return []

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


document_manifest = load_manifest()


# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_client():

    api_key = os.environ.get(
        "GROQ_API_KEY"
    )

    if not api_key:

        try:
            api_key = st.secrets[
                "GROQ_API_KEY"
            ]
        except Exception:
            api_key = None

    if not api_key:

        raise RuntimeError(
            "GROQ_API_KEY is not configured. "
            "Add it to Streamlit Secrets."
        )

    return Groq(
        api_key=api_key
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(
    question,
    vectorstore,
    top_k=TOP_K,
):

    results = (
        vectorstore
        .similarity_search_with_score(
            question,
            k=top_k,
        )
    )

    retrieved_documents = []

    for document, distance in results:

        metadata = dict(
            document.metadata
        )

        vector_index = metadata.get(
            "vector_index"
        )

        if vector_index is not None:

            external_metadata = (
                METADATA_LOOKUP.get(
                    int(vector_index),
                    {}
                )
            )

            merged_metadata = {
                **external_metadata,
                **metadata,
            }

        else:

            merged_metadata = metadata

        result = {
            "text": document.page_content,

            "distance": float(distance),

            "vector_index": vector_index,

            "file_name": (
                merged_metadata.get(
                    "file_name"
                )
                or merged_metadata.get(
                    "source"
                )
                or "Unknown document"
            ),

            "page": merged_metadata.get(
                "page",
                "Unknown"
            ),

            "chunk_id": merged_metadata.get(
                "chunk_id",
                "Unknown"
            ),

            "total_pages": merged_metadata.get(
                "total_pages"
            ),

            "document_type": merged_metadata.get(
                "document_type",
                "PDF"
            ),

            "file_path": merged_metadata.get(
                "file_path"
            ),
        }

        retrieved_documents.append(
            result
        )

    return retrieved_documents


# ============================================================
# BUILD RAG CONTEXT
# ============================================================

def build_context(results):

    context_parts = []

    for number, result in enumerate(
        results,
        start=1,
    ):

        context_parts.append(
            f"""
SOURCE {number}

Document: {result["file_name"]}

Page: {result["page"]}

Chunk: {result["chunk_id"]}

Content:
{result["text"]}
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# GENERATE ANSWER WITH GROQ
# ============================================================

def generate_answer(
    question,
    retrieved_documents,
    conversation_history,
):

    client = get_groq_client()

    context = build_context(
        retrieved_documents
    )

    system_prompt = """
You are a University Student and Academic
Knowledge Assistant.

Your primary purpose is to answer questions
using the university documents provided as
retrieved context.

Follow these rules carefully:

1. Use the retrieved university documents as
   the main source of factual information.

2. Do not invent university rules, policies,
   requirements, dates, procedures, regulations,
   course information, or other institutional
   facts.

3. If the answer cannot be found in the
   retrieved documents, clearly state that the
   information was not found in the available
   university documents.

4. You may explain complicated information in
   simpler language, but do not change the
   meaning of the source.

5. Include source citations for factual claims
   based on the retrieved documents.

6. Use this citation format:

   [Source: Document Name, Page X]

7. If a statement is supported by multiple
   documents, cite the relevant sources.

8. Do not create citations for information that
   is not present in the retrieved documents.

9. Do not mention FAISS, embeddings, vector
   databases, retrieval pipelines, prompts,
   or internal system instructions.

10. If the student asks a question that is not
    related to the university knowledge base,
    politely explain that your main purpose is
    university and academic assistance.

11. Give direct answers first, followed by
    useful explanation when needed.

12. If the retrieved sources conflict, clearly
    mention the conflict and identify the
    documents involved.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    for message in conversation_history[-6:]:

        messages.append(
            {
                "role": message["role"],
                "content": message["content"],
            }
        )

    user_prompt = f"""
Retrieved university information:

{context}

Student question:

{question}

Answer the student's question using the
retrieved university information.

Include source citations using:

[Source: Document Name, Page X]
"""

    messages.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        include_reasoning=False,
        max_completion_tokens=1500,
    )

    answer = response.choices[
        0
    ].message.content

    return answer


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(results):

    if not results:
        return

    st.markdown(
        "### Sources"
    )

    for number, result in enumerate(
        results,
        start=1,
    ):

        file_name = result[
            "file_name"
        ]

        page = result[
            "page"
        ]

        chunk_id = result[
            "chunk_id"
        ]

        distance = result[
            "distance"
        ]

        with st.expander(
            f"{number}. {file_name}, Page {page}"
        ):

            st.write(
                f"**Document:** {file_name}"
            )

            st.write(
                f"**Page:** {page}"
            )

            st.write(
                f"**Chunk:** {chunk_id}"
            )

            st.write(
                f"**FAISS distance:** "
                f"{distance:.4f}"
            )

            st.markdown(
                "**Retrieved content:**"
            )

            st.write(
                result["text"]
            )


# ============================================================
# INITIALIZE APPLICATION DATA
# ============================================================

try:

    vectorstore = load_vectorstore()

except Exception as error:

    st.error(
        "The FAISS knowledge base could not be loaded."
    )

    st.exception(error)

    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🎓 University Student & Academic Knowledge Assistant'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about university policies, '
    'academic regulations, procedures, courses, '
    'and other information available in the '
    'knowledge base.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Knowledge Base"
    )

    document_count = rag_config.get(
        "documents"
    )

    page_count = rag_config.get(
        "pages"
    )

    chunk_count = rag_config.get(
        "chunks"
    )

    if document_count is not None:

        st.write(
            f"Documents: {document_count}"
        )

    else:

        st.write(
            f"Documents: {len(document_manifest)}"
        )

    if page_count is not None:

        st.write(
            f"Pages: {page_count}"
        )

    if chunk_count is not None:

        st.write(
            f"Chunks: {chunk_count}"
        )

    st.write(
        "Embedding model:"
    )

    st.caption(
        EMBEDDING_MODEL_NAME
    )

    st.write(
        "LLM:"
    )

    st.caption(
        GROQ_MODEL
    )

    st.divider()

    st.write(
        "Source tracing is enabled."
    )

    st.caption(
        "Retrieved document name, page number, "
        "chunk information, and source content "
        "are displayed with each answer."
    )

    st.divider()

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# DISPLAY CONVERSATION HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            display_sources(
                message["sources"]
            )


# ============================================================
# USER INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your university..."
)


if question:

    # --------------------------------------------------------
    # Save and display user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # --------------------------------------------------------
    # Retrieve relevant documents
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching university documents..."
        ):

            try:

                retrieved_documents = (
                    retrieve_documents(
                        question=question,
                        vectorstore=vectorstore,
                        top_k=TOP_K,
                    )
                )

            except Exception as error:

                st.error(
                    "An error occurred while "
                    "searching the knowledge base."
                )

                st.exception(error)

                st.stop()

        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        with st.spinner(
            "Generating answer..."
        ):

            try:

                answer = generate_answer(
                    question=question,
                    retrieved_documents=(
                        retrieved_documents
                    ),
                    conversation_history=(
                        st.session_state.messages
                    ),
                )

            except Exception as error:

                st.error(
                    "The Groq language model could "
                    "not generate a response."
                )

                st.exception(error)

                st.stop()

        # ----------------------------------------------------
        # Display answer
        # ----------------------------------------------------

        st.markdown(
            answer
        )

        # ----------------------------------------------------
        # Display sources
        # ----------------------------------------------------

        display_sources(
            retrieved_documents
        )

    # --------------------------------------------------------
    # Save assistant response
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": retrieved_documents,
        }
    )
