import os
import time
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

import warnings
warnings.filterwarnings("ignore")

load_dotenv()


def load_embeddings():
    """Load the HuggingFace embedding model."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def load_vectorstore(embeddings):
    """Load the persisted FAISS vector store."""
    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

def load_llm():
    """Initialize the OpenAI GPT-4o-mini LLM."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=2048
    )


def get_retriever(vectorstore):
    """Create a retriever with MMR search (semantic + keyword hybrid)."""
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10}
    )


def get_prompt():
    """Financial domain prompt template for structured outputs."""
    return PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template="""
You are a senior financial analyst with deep expertise in analyzing SEC 10-K filings,
market reports, stock performance, and financial statements. You provide accurate,
data-driven insights grounded strictly in the retrieved context.

Instructions:
- Use ONLY the information from the provided context to answer.
- If the context contains relevant financial data, provide a clear, structured response.
- Include specific numbers, percentages, and metrics when available.
- If the answer is not in the context, say "The information is not available in the current knowledge base."
- Always mention the company ticker when relevant.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Financial Analysis:
"""
    )


def load_pipeline():
    """Load the full RAG pipeline with memory and retrieval chain."""
    embeddings = load_embeddings()
    vectorstore = load_vectorstore(embeddings)
    llm = load_llm()
    retriever = get_retriever(vectorstore)
    prompt = get_prompt()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )

    return chain


def ask_question(chain, question):
    """Ask a question and return structured response with metadata."""
    start = time.time()
    result = chain.invoke({"question": question})
    latency = time.time() - start
    docs = result["source_documents"]

    retrieved_docs = [doc.page_content[:300] for doc in docs]
    sources = [doc.metadata for doc in docs]

    return {
        "answer": result["answer"],
        "retrieved_docs": retrieved_docs,
        "sources": sources,
        "latency": round(latency, 2),
        "num_docs_retrieved": len(docs),
    }


def rag_query(question):
    """
    Standalone RAG query function — used as a Tool by agents.
    Loads the pipeline fresh each time (stateless for tool use).
    """
    embeddings = load_embeddings()
    vectorstore = load_vectorstore(embeddings)
    llm = load_llm()
    retriever = get_retriever(vectorstore)

    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])
    sources = [doc.metadata for doc in docs]

    prompt_text = f"""
You are a senior financial analyst. Answer the following question using ONLY
the provided context from SEC 10-K filings.

Context:
{context}

Question: {question}

Provide a clear, data-driven financial analysis:
"""

    response = llm.invoke(prompt_text)

    return {
        "answer": response.content,
        "sources": sources,
        "num_docs_retrieved": len(docs),
    }
