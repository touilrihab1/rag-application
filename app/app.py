"""
Streamlit GUI for the multimodal PDF RAG pipeline.

Run with:
    streamlit run app.py
"""

import os
import tempfile

import streamlit as st
from openai import OpenAI
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from rag_pipeline import (
    partition_pdf,
    separate_tables_texts,
    summarize_elements,
    build_retriever,
    RAGBase,
)

st.set_page_config(page_title="PDF RAG Chat", page_icon="📄", layout="wide")
st.title("📄 Chat with your PDF")

# ---------------- Sidebar: settings ----------------
with st.sidebar:
    st.header("Settings")
    openrouter_key = st.text_input(
        "OpenRouter API key", type="password", value=os.getenv("OPENROUTER_API_KEY", "")
    )
    groq_key = st.text_input(
        "Groq API key", type="password", value=os.getenv("GROQ_API_KEY", "")
    )
    model_name = st.text_input(
        "LLM model (OpenRouter)", value="nvidia/nemotron-3-ultra-550b-a55b:free"
    )
    st.markdown("---")
    st.caption(
        "Note: PDF parsing uses the 'hi_res' strategy for tables/images, which needs "
        "Poppler and Tesseract installed locally (not just pip packages)."
    )
    if st.session_state.get("rag") is not None:
        if st.button("Reset / process a new PDF"):
            st.session_state.rag = None
            st.session_state.messages = []
            st.session_state.processed_file = None
            st.rerun()

# ---------------- Session state ----------------
st.session_state.setdefault("rag", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("processed_file", None)

# ---------------- File upload + processing ----------------
if st.session_state.rag is None:
    uploaded_file = st.file_uploader("Upload a PDF paper", type=["pdf"])

    if uploaded_file is not None:
        if not openrouter_key or not groq_key:
            st.warning("Enter both API keys in the sidebar, then re-upload the PDF.")
        else:
            with st.spinner("Parsing PDF (hi_res extraction can take a minute or two)..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                chunks = partition_pdf(tmp_path)
                tables, texts = separate_tables_texts(chunks)
                os.remove(tmp_path)

            st.info(f"Found {len(texts)} text chunk(s) and {len(tables)} table(s). Summarizing...")

            with st.spinner("Summarizing chunks with Groq..."):
                text_summaries, table_summaries = summarize_elements(texts, tables, groq_key)

            with st.spinner("Embedding and building the vector index..."):
                retriever = build_retriever(texts, tables, text_summaries, table_summaries)

                openai_client = OpenAI(
                    api_key=openrouter_key,
                    base_url="https://openrouter.ai/api/v1",
                )

                rag = RAGBase(
                    index=retriever,
                    llm_client=openai_client,
                    instructions=(
                        "You are a helpful assistant. Answer the question using only "
                        "the provided context."
                    ),
                    prompt_template=(
                        "Answer the question using the context below.\n\n"
                        "Context:\n{context}\n\nQuestion:\n{question}"
                    ),
                    model=model_name,
                )

            st.session_state.rag = rag
            st.session_state.processed_file = uploaded_file.name
            st.session_state.messages = []
            st.success(f"'{uploaded_file.name}' processed! Ask a question below.")
            st.rerun()

# ---------------- Chat interface ----------------
if st.session_state.rag is not None:
    st.caption(f"Currently chatting with: **{st.session_state.processed_file}**")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about the paper...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.rag.rag(question)
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Enter your API keys in the sidebar, then upload a PDF to get started.")