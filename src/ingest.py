import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

PDF_PATHS = [
    "data/Guideline E23  Model Risk Management 2027  Letter.pdf",
    "data/ifrs-9-financial-instruments.pdf",
    "data/Residential mortgage underwriting practices and procedures  Guideline 2017.pdf"
]

def ingest_documents():
    # Load all PDFs
    docs = []
    for path in PDF_PATHS:
        print(f"Loading: {path}")
        loader = PyMuPDFLoader(path)
        docs.extend(loader.load())
    print(f"Total pages loaded: {len(docs)}")

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks created: {len(chunks)}")

    # Embed and store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vectorstore"
    )
    print("Vectorstore saved successfully.")
    return vectorstore

if __name__ == "__main__":
    ingest_documents()