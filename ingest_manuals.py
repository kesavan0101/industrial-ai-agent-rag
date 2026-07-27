import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

MANUALS_DIR = "./manuals"
DB_DIR = "./chroma_db"


def build_vector_store():
    print(" Loading PDF manuals...")
    loader = PyPDFDirectoryLoader(MANUALS_DIR)
    raw_docs = loader.load()
    print(f"Loaded {len(raw_docs)} pages from PDF manuals.")

    # Chunking: split text into bite-sized passages
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(raw_docs)
    print(f"Created {len(chunks)} text chunks.")

    # Generate Embeddings & Save to local database
    print(" Generating embeddings (this may take 1-2 minutes for ~4.3k chunks)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print(f" Vector Database successfully built at '{DB_DIR}'!")


if __name__ == "__main__":
    build_vector_store()