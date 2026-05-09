from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
from rag_pipeline import load_pipeline, ask_question
from agents import run_financial_crew
from evaluation import evaluate_response

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PYDANTIC SCHEMAS FOR INPUT/OUTPUT VALIDATION
# ============================================================

class QueryInput(BaseModel):
    """Input schema for user queries."""
    question: str = Field(
        ...,
        description="The financial question to ask",
        min_length=3,
        max_length=1000,
        examples=["What are NVIDIA's main revenue sources?"]
    )
    mode: str = Field(
        default="rag",
        description="Query mode: 'rag' for basic RAG, 'agentic' for multi-agent workflow",
        examples=["rag", "agentic"]
    )


class SourceInfo(BaseModel):
    """Schema for source document metadata."""
    ticker: Optional[str] = ""
    filing: Optional[str] = ""
    question: Optional[str] = ""


class QueryOutput(BaseModel):
    """Output schema for query responses."""
    question: str
    answer: str
    mode: str
    sources: Optional[List[SourceInfo]] = []
    latency: Optional[float] = None
    num_docs_retrieved: Optional[int] = None
    agents_used: Optional[List[str]] = []


class EvalInput(BaseModel):
    """Input schema for evaluation endpoint."""
    question: str
    reference_answer: str
    generated_answer: str
    context: Optional[str] = ""


class EvalOutput(BaseModel):
    """Output schema for evaluation results."""
    bleu: float
    rouge1: float
    rouge2: float
    rougeL: float
    relevance: float


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    message: str
    available_modes: List[str]


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Financial Market Intelligence API",
    description=(
        "RAG Pipeline with Multi-AI Agentic Workflow for Financial Market Intelligence. "
        "Powered by CrewAI, FAISS, and Gemini 2.5 Flash."
    ),
    version="1.0.0",
)

# Load RAG pipeline at startup
chain = load_pipeline()


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/", response_model=HealthResponse)
def root():
    """Health check endpoint."""
    return HealthResponse(
        status="running",
        message="Financial Market Intelligence API is running! Visit /docs to test.",
        available_modes=["rag", "agentic"],
    )


@app.post("/query", response_model=QueryOutput)
def query(input_data: QueryInput):
    """
    Main query endpoint.
    - mode='rag': Uses basic RAG pipeline (faster)
    - mode='agentic': Uses full 4-agent CrewAI workflow (detailed analysis)
    """
    if input_data.mode == "agentic":
        # Full multi-agent workflow
        result = run_financial_crew(input_data.question)
        return QueryOutput(
            question=input_data.question,
            answer=result["final_output"],
            mode="agentic",
            agents_used=result["agents_used"],
        )
    else:
        # Basic RAG pipeline
        result = ask_question(chain, input_data.question)
        sources = [
            SourceInfo(
                ticker=s.get("ticker", ""),
                filing=s.get("filing", ""),
                question=s.get("question", ""),
            )
            for s in result.get("sources", [])
        ]
        return QueryOutput(
            question=input_data.question,
            answer=result["answer"],
            mode="rag",
            sources=sources,
            latency=result["latency"],
            num_docs_retrieved=result["num_docs_retrieved"],
        )


@app.post("/evaluate", response_model=EvalOutput)
def evaluate(input_data: EvalInput):
    """Evaluate a generated response against a reference answer."""
    result = evaluate_response(
        question=input_data.question,
        reference_answer=input_data.reference_answer,
        generated_answer=input_data.generated_answer,
        context=input_data.context,
    )
    return EvalOutput(**result)
