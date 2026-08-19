# Chat with your PDF

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their content.

## Features

- Upload and process PDF documents
- Extract and split document text into chunks
- Generate embeddings for document chunks
- Retrieve relevant information using vector search
- Generate answers using an LLM
- Chat with PDF documents through an interactive interface

## How It Works

The application follows a Retrieval-Augmented Generation (RAG) pipeline:

```text
PDF
 ↓
Text Extraction
 ↓
Document Chunking
 ↓
Embeddings
 ↓
Vector Database
 ↓
Relevant Context Retrieval
 ↓
LLM
 ↓
Generated Answer
