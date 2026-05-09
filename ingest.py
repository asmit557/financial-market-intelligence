# 1. Import Libraries
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

import os
import json
import warnings

warnings.filterwarnings("ignore")


def build_index():
    # 2. Load Dataset from HuggingFace
    print("Downloading dataset from HuggingFace...")
    dataset = load_dataset("virattt/financial-qa-10K", split="train")

    # 3. Convert to LangChain Documents with rich metadata
    print("Converting to documents...")
    documents = []
    for row in dataset:
        content = (
            f"Question: {row['question']}\n"
            f"Answer: {row['answer']}\n"
            f"Context: {row['context']}"
        )
        metadata = {
            "ticker": row.get("ticker", ""),
            "filing": row.get("filing", ""),
            "question": row.get("question", ""),
        }
        documents.append(Document(page_content=content, metadata=metadata))

    print(f"Total documents loaded: {len(documents)}")

    # 4. RecursiveCharacter Text Splitter
    print("Splitting documents into chunks...")
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", "", ".", ",", ";"]
    )

    chunks = recursive_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    # 5. Create embeddings using HuggingFace sentence-transformers
    print("Creating embeddings...")
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 6. Create FAISS Vector Store
    print("Building FAISS index...")
    faiss_store = FAISS.from_documents(
        documents=chunks,
        embedding=hf_embeddings
    )

    # 7. Persist the vector store
    faiss_store.save_local("faiss_index")
    print("FAISS faiss_index created successfully!")
    print(f"Index contains {len(chunks)} vectors")


if __name__ == "__main__":
    build_index()
