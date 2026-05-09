import os
from dotenv import load_dotenv

load_dotenv()
os.environ["CREWAI_LLM"] = "gemini/gemini-2.0-flash"
os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
os.environ["OPENAI_API_KEY"] = "fake-key-not-used"

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from rag_pipeline import rag_query, load_embeddings, load_vectorstore, get_retriever
from evaluation import evaluate_response

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 5. RAG AS TOOL INTEGRATION
# RAG wrapped as a callable CrewAI tool that any agent can use
# Agents query the vector database dynamically during reasoning
# ============================================================

@tool("RAG Financial Search")
def rag_search_tool(query: str) -> str:
    """
    Search the financial knowledge base (SEC 10-K filings) for relevant information.
    Use this tool to retrieve financial data, company reports, revenue figures,
    market strategies, and business operations from real SEC filings.
    Input should be a clear financial question.
    """
    try:
        result = rag_query(query)
        sources = result.get("sources", [])
        source_info = ""
        if sources:
            tickers = set(s.get("ticker", "") for s in sources if s.get("ticker"))
            source_info = f"\n[Sources: {', '.join(tickers)}]"
        return f"{result['answer']}{source_info}"
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"


@tool("Retrieval Validator")
def retrieval_validator_tool(query: str) -> str:
    """
    Validate that retrieved documents are relevant before generating a final answer.
    Use this tool to check if the retrieved context actually answers the question.
    Input should be the original question.
    """
    try:
        embeddings = load_embeddings()
        vectorstore = load_vectorstore(embeddings)
        retriever = get_retriever(vectorstore)
        docs = retriever.invoke(query)

        if not docs:
            return "VALIDATION FAILED: No relevant documents found for this query."

        # Check relevance of top document
        top_doc = docs[0].page_content
        query_words = set(query.lower().split())
        doc_words = set(top_doc.lower().split())
        overlap = query_words.intersection(doc_words)

        relevance = len(overlap) / max(len(query_words), 1)

        if relevance < 0.1:
            return f"VALIDATION WARNING: Low relevance ({relevance:.2f}). Retrieved context may not directly answer the question."

        tickers = set(d.metadata.get("ticker", "") for d in docs if d.metadata.get("ticker"))
        return (
            f"VALIDATION PASSED: Relevance score {relevance:.2f}. "
            f"Found {len(docs)} relevant documents. "
            f"Companies covered: {', '.join(tickers)}. "
            f"Context preview: {top_doc[:200]}..."
        )
    except Exception as e:
        return f"Validation error: {str(e)}"


# ============================================================
# FEEDBACK LOOP TOOL
# Evaluates quality of generated responses for iterative improvement
# ============================================================

@tool("Quality Feedback")
def quality_feedback_tool(input_text: str) -> str:
    """
    Evaluate the quality of a generated financial analysis response.
    Input format: 'QUESTION: <question> ||| ANSWER: <answer>'
    Returns quality scores and improvement suggestions.
    """
    try:
        parts = input_text.split("|||")
        if len(parts) != 2:
            return "Invalid input format. Use: 'QUESTION: <question> ||| ANSWER: <answer>'"

        question = parts[0].replace("QUESTION:", "").strip()
        answer = parts[1].replace("ANSWER:", "").strip()

        # Use evaluation metrics
        scores = evaluate_response(
            question=question,
            reference_answer=answer,
            generated_answer=answer,
            context=answer,
        )

        feedback = f"Quality Scores - Relevance: {scores['relevance']}, ROUGE-L: {scores['rougeL']}\n"

        if scores["relevance"] < 0.3:
            feedback += "SUGGESTION: Response has low relevance. Include more specific financial data and metrics.\n"
        if scores["rougeL"] < 0.3:
            feedback += "SUGGESTION: Response could be more aligned with the source data.\n"
        if scores["relevance"] >= 0.5:
            feedback += "GOOD: Response shows strong relevance to the query.\n"

        return feedback
    except Exception as e:
        return f"Feedback error: {str(e)}"


# ============================================================
# 4. AGENTIC WORKFLOW DEVELOPMENT
# Four specialized agents with CrewAI
# Planner-executor workflow for task delegation
# Agents as Tools architecture
# Memory-aware agents with conversational context retention
# ============================================================

def create_retriever_agent():
    """
    Retriever Agent: Fetches market reports, stock data, and historical data.
    Acts as the first agent in the pipeline — gathers raw financial data.
    """
    return Agent(
        role="Financial Data Retriever",
        goal="Retrieve the most relevant and accurate financial data from SEC 10-K "
             "filings to answer the user's financial question.",
        backstory="You are an expert financial data analyst who specializes in "
                  "searching through SEC filings, annual reports, and financial "
                  "databases. You know exactly how to find revenue figures, "
                  "growth metrics, risk factors, and business strategies from "
                  "company filings. You always validate the relevance of "
                  "retrieved data before passing it on.",
        tools=[rag_search_tool, retrieval_validator_tool],
        llm="gemini/gemini-2.0-flash",
        verbose=True,
        memory=False,
        allow_delegation=False,
    )


def create_analysis_agent():
    """
    Analysis Agent: Uses RAG-retrieved data to generate grounded insights.
    Receives data from Retriever Agent and produces structured analysis.
    """
    return Agent(
        role="Financial Analysis Expert",
        goal="Analyze the retrieved financial data and generate clear, "
             "data-driven insights about company performance, market trends, "
             "and business strategies.",
        backstory="You are a senior financial analyst at a top investment bank. "
                  "You excel at interpreting financial statements, identifying "
                  "trends, and providing actionable insights. You always base "
                  "your analysis on concrete data and never speculate without "
                  "evidence. You structure your analysis with key metrics, "
                  "comparisons, and clear conclusions.",
        tools=[rag_search_tool],
        llm="gemini/gemini-2.0-flash",
        verbose=True,
        memory=False,
        allow_delegation=False,
    )


def create_portfolio_agent():
    """
    Portfolio Agent: Suggests allocations based on risk appetite.
    Takes analysis output and recommends portfolio positioning.
    """
    return Agent(
        role="Portfolio Strategy Advisor",
        goal="Based on the financial analysis provided, suggest portfolio "
             "allocation strategies considering different risk appetites "
             "(conservative, moderate, aggressive).",
        backstory="You are a certified portfolio manager with 20 years of "
                  "experience managing institutional portfolios. You understand "
                  "modern portfolio theory, asset allocation, sector rotation, "
                  "and risk-return tradeoffs. You always provide allocations "
                  "for three risk profiles: conservative, moderate, and "
                  "aggressive, with clear reasoning for each.",
        tools=[rag_search_tool],
        llm="gemini/gemini-2.0-flash",
        verbose=True,
        memory=False,
        allow_delegation=False,
    )


def create_risk_agent():
    """
    Risk Assessment Agent: Evaluates market volatility and portfolio exposure.
    Final agent that validates overall risk and provides feedback.
    """
    return Agent(
        role="Risk Assessment Specialist",
        goal="Evaluate the risk factors, market volatility, and potential "
             "exposure associated with the financial analysis and portfolio "
             "recommendations provided.",
        backstory="You are a chief risk officer with expertise in financial "
                  "risk management, stress testing, and regulatory compliance. "
                  "You identify key risk factors from SEC filings including "
                  "market risk, credit risk, operational risk, and regulatory "
                  "risk. You always provide a risk rating (Low/Medium/High) "
                  "with supporting evidence.",
        tools=[rag_search_tool, quality_feedback_tool],
        llm="gemini/gemini-2.0-flash",
        verbose=True,
        memory=False,
        allow_delegation=False,
    )


# ============================================================
# TASK DEFINITIONS
# Sequential planner-executor workflow
# Tool chaining: RAG output feeds downstream agents
# ============================================================

def create_tasks(question, retriever, analyst, portfolio, risk):
    """Create sequential tasks implementing planner-executor workflow."""

    # Task 1: Retriever fetches and validates data
    retrieval_task = Task(
        description=(
            f"Search the financial knowledge base for information related to: '{question}'\n"
            f"Steps:\n"
            f"1. Use the 'RAG Financial Search' tool to find relevant financial data.\n"
            f"2. Use the 'Retrieval Validator' tool to validate the relevance of results.\n"
            f"3. Compile the key financial data points found.\n"
            f"4. List the companies and tickers covered in the results."
        ),
        expected_output=(
            "A comprehensive summary of retrieved financial data including "
            "specific numbers, metrics, company names, and tickers. "
            "Include a validation status confirming data relevance."
        ),
        agent=retriever,
    )

    # Task 2: Analyst generates insights from retrieved data
    analysis_task = Task(
        description=(
            f"Using the financial data retrieved in the previous step, "
            f"provide a detailed analysis for: '{question}'\n"
            f"Steps:\n"
            f"1. Review the retrieved financial data carefully.\n"
            f"2. Identify key trends, metrics, and performance indicators.\n"
            f"3. Provide structured analysis with specific numbers.\n"
            f"4. Draw conclusions based on the data."
        ),
        expected_output=(
            "A structured financial analysis with: "
            "1) Key metrics and numbers, "
            "2) Trend analysis, "
            "3) Performance assessment, "
            "4) Data-driven conclusions."
        ),
        agent=analyst,
        context=[retrieval_task],
    )

    # Task 3: Portfolio advisor suggests allocations
    portfolio_task = Task(
        description=(
            f"Based on the financial analysis provided, suggest portfolio "
            f"allocation strategies for: '{question}'\n"
            f"Steps:\n"
            f"1. Review the analysis and identify investment implications.\n"
            f"2. Suggest allocation for Conservative profile (low risk).\n"
            f"3. Suggest allocation for Moderate profile (balanced).\n"
            f"4. Suggest allocation for Aggressive profile (high growth).\n"
            f"5. Provide reasoning for each allocation."
        ),
        expected_output=(
            "Portfolio allocation recommendations for three risk profiles: "
            "Conservative, Moderate, and Aggressive. Each with specific "
            "percentage allocations and reasoning."
        ),
        agent=portfolio,
        context=[analysis_task],
    )

    # Task 4: Risk assessment with feedback loop
    risk_task = Task(
        description=(
            f"Evaluate the risks associated with the analysis and portfolio "
            f"recommendations for: '{question}'\n"
            f"Steps:\n"
            f"1. Identify key risk factors from the financial data.\n"
            f"2. Assess market, credit, operational, and regulatory risks.\n"
            f"3. Provide an overall risk rating (Low/Medium/High).\n"
            f"4. Use the 'Quality Feedback' tool to evaluate the overall "
            f"   analysis quality with format: 'QUESTION: {question} ||| ANSWER: <summary>'\n"
            f"5. Provide final recommendations considering all risks."
        ),
        expected_output=(
            "A comprehensive risk assessment with: "
            "1) Identified risk factors, "
            "2) Risk ratings per category, "
            "3) Overall risk rating, "
            "4) Quality feedback scores, "
            "5) Final investment recommendations with risk warnings."
        ),
        agent=risk,
        context=[analysis_task, portfolio_task],
    )

    return [retrieval_task, analysis_task, portfolio_task, risk_task]




# ============================================================
# CREW ORCHESTRATION
# Sequential process: Retriever -> Analyst -> Portfolio -> Risk
# ============================================================

def run_financial_crew(question):
    """
    Run the full multi-agent financial intelligence workflow.
    Implements sequential planner-executor pattern with tool chaining.
    """
    # Create agents
    retriever = create_retriever_agent()
    analyst = create_analysis_agent()
    portfolio = create_portfolio_agent()
    risk = create_risk_agent()

    # Create tasks with sequential dependencies
    tasks = create_tasks(question, retriever, analyst, portfolio, risk)

    # Assemble the crew with sequential process
    crew = Crew(
        agents=[retriever, analyst, portfolio, risk],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        memory=False,
    )

    # Execute the crew
    result = crew.kickoff()

    return {
        "question": question,
        "final_output": str(result),
        "agents_used": [
            "Financial Data Retriever",
            "Financial Analysis Expert",
            "Portfolio Strategy Advisor",
            "Risk Assessment Specialist",
        ],
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    question = "What are NVIDIA's main revenue sources and growth strategy?"
    print(f"\nRunning Financial Intelligence Crew for: {question}\n")
    print("=" * 60)

    result = run_financial_crew(question)

    print("\n" + "=" * 60)
    print("FINAL OUTPUT:")
    print("=" * 60)
    print(result["final_output"])
