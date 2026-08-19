"""
Core multimodal RAG pipeline (PDF -> chunks -> summaries -> vector index -> LLM answer).

This is the same logic from the original notebook, refactored into functions/classes
so it can be driven from a Streamlit UI (or a notebook, or a CLI) without re-running
install cells.
"""

import os
import uuid

# unstructured shells out to tesseract.exe via its own vendored wrapper
# (unstructured_pytesseract), which does NOT respect pytesseract.tesseract_cmd.
# Adding the install folder to PATH for this process is what actually works.
TESSERACT_DIR = r"C:\Program Files\Tesseract-OCR"
if TESSERACT_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = TESSERACT_DIR + os.pathsep + os.environ.get("PATH", "")

from unstructured.partition.auto import partition
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_core.stores import InMemoryStore
from langchain_core.documents import Document
from langchain_classic.retrievers import MultiVectorRetriever
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddings:
    """Wraps a SentenceTransformer model with the embed_documents/embed_query
    interface langchain's Chroma wrapper expects."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()


def partition_pdf(file_path: str):
    """Parse a PDF into chunked elements (text + tables), same settings as the notebook."""
    chunks = partition(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",  # needed to detect tables/images
        extract_image_block_type=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_characters=2000,
        new_after_n_characters=6000,
    )
    return chunks


def separate_tables_texts(chunks):
    """Split partitioned chunks into Table elements and CompositeElement (text) chunks."""
    tables, texts = [], []
    for chunk in chunks:
        type_name = str(type(chunk))
        if "Table" in type_name:
            tables.append(chunk)
        elif "CompositeElement" in type_name:
            texts.append(chunk)
    return tables, texts


def summarize_elements(texts, tables, groq_api_key: str, model_name: str = "openai/gpt-oss-20b"):
    """Summarize text chunks and table chunks (as HTML) with an LLM, for embedding."""
    prompt_text = """
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.

Respond only with the summary, no additional comment.
Do not start your message by saying "Here is a summary" or anything like that.
Just give the summary as it is.

Table or text chunk: {element}
"""
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatGroq(temperature=0.5, model=model_name, groq_api_key=groq_api_key)
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3}) if texts else []

    tables_html = [table.metadata.text_as_html for table in tables]
    table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3}) if tables_html else []

    return text_summaries, table_summaries


def build_retriever(texts, tables, text_summaries, table_summaries):
    """Build a MultiVectorRetriever: summaries are embedded/searched, original
    text/table chunks are stored and returned on retrieval."""
    embeddings = SentenceTransformerEmbeddings()

    vectorstore = Chroma(
        collection_name="multi_modal_rag",
        embedding_function=embeddings,
    )
    store = InMemoryStore()
    id_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    if texts:
        text_ids = [str(uuid.uuid4()) for _ in texts]
        summary_docs = [
            Document(page_content=summary, metadata={id_key: text_ids[i]})
            for i, summary in enumerate(text_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(text_ids, texts)))

    if tables:
        table_ids = [str(uuid.uuid4()) for _ in tables]
        summary_docs = [
            Document(page_content=summary, metadata={id_key: table_ids[i]})
            for i, summary in enumerate(table_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(table_ids, tables)))

    return retriever


class RAGBase:
    def __init__(
        self,
        index,
        llm_client,
        instructions=None,
        prompt_template=None,
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

    # 1. Retrieve relevant original documents, as one readable text blob
    def search_context(self, query):
        docs = self.index.invoke(query)
        texts = [doc.text if hasattr(doc, "text") else str(doc) for doc in docs]
        return "\n\n".join(t for t in texts if t)

    # 2. Build the prompt
    def build_prompt(self, query):
        context = self.search_context(query)
        return self.prompt_template.format(question=query, context=context)

    # 3. Send prompt to the LLM
    def llm(self, prompt):
        input_messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=input_messages,
            max_tokens=1000,
        )
        return response.choices[0].message.content

    # 4. Complete RAG pipeline
    def rag(self, query):
        prompt = self.build_prompt(query)
        return self.llm(prompt)