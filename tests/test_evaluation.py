"""Unit tests for the Financial Market Intelligence pipeline."""


def test_evaluation_bleu():
    """Test BLEU score computation."""
    from evaluation import compute_bleu

    ref = "NVIDIA invented the GPU in 1999"
    hyp = "NVIDIA created the GPU in 1999"
    score = compute_bleu(ref, hyp)
    assert 0 <= score <= 1, f"BLEU score out of range: {score}"
    assert score > 0, "BLEU score should be positive for similar sentences"
    print(f"BLEU test passed: {score}")


def test_evaluation_rouge():
    """Test ROUGE score computation."""
    from evaluation import compute_rouge

    ref = "NVIDIA revenue grew significantly in 2023"
    hyp = "NVIDIA saw significant revenue growth in 2023"
    scores = compute_rouge(ref, hyp)
    assert "rouge1" in scores, "Missing rouge1 score"
    assert "rouge2" in scores, "Missing rouge2 score"
    assert "rougeL" in scores, "Missing rougeL score"
    assert all(0 <= v <= 1 for v in scores.values()), "ROUGE scores out of range"
    print(f"ROUGE test passed: {scores}")


def test_evaluation_relevance():
    """Test relevance score computation."""
    from evaluation import compute_relevance

    question = "What is NVIDIA revenue?"
    answer = "NVIDIA total revenue was 60 billion dollars"
    context = "NVIDIA reported total revenue of 60 billion in fiscal year 2024"
    score = compute_relevance(question, answer, context)
    assert 0 <= score <= 1, f"Relevance score out of range: {score}"
    print(f"Relevance test passed: {score}")


def test_evaluation_full():
    """Test full evaluation pipeline."""
    from evaluation import evaluate_response

    result = evaluate_response(
        question="What is NVIDIA GPU?",
        reference_answer="NVIDIA invented the GPU in 1999",
        generated_answer="NVIDIA created the GPU in 1999",
        context="Our invention of the GPU in 1999 defined modern computer graphics",
    )
    assert "bleu" in result, "Missing BLEU in results"
    assert "rouge1" in result, "Missing ROUGE-1 in results"
    assert "relevance" in result, "Missing relevance in results"
    print(f"Full evaluation test passed: {result}")


def test_pydantic_schemas():
    """Test Pydantic input/output validation schemas."""
    from main import QueryInput, QueryOutput, EvalInput

    # Test valid input
    query = QueryInput(question="What is NVIDIA revenue?", mode="rag")
    assert query.question == "What is NVIDIA revenue?"
    assert query.mode == "rag"

    # Test agentic mode
    query2 = QueryInput(question="Analyze NVIDIA growth", mode="agentic")
    assert query2.mode == "agentic"

    # Test default mode
    query3 = QueryInput(question="Test question")
    assert query3.mode == "rag"

    print("Pydantic schema tests passed")


def test_feedback_threshold():
    """Test automated retraining trigger based on feedback thresholds."""
    from evaluation import evaluate_batch

    test_cases = [
        {
            "question": "What is NVIDIA?",
            "reference_answer": "NVIDIA is a technology company",
            "generated_answer": "NVIDIA is a tech company that makes GPUs",
            "context": "NVIDIA Corporation is a technology company",
        },
        {
            "question": "What is Apple revenue?",
            "reference_answer": "Apple revenue was 383 billion",
            "generated_answer": "Apple reported 383 billion in revenue",
            "context": "Apple Inc reported total revenue of 383 billion",
        },
    ]

    results = evaluate_batch(test_cases)
    avg_scores = results["averages"]

    # Continuous improvement: check if retraining is needed
    RETRAIN_THRESHOLD = 0.2
    needs_retraining = avg_scores["avg_relevance"] < RETRAIN_THRESHOLD

    print(f"Average scores: {avg_scores}")
    print(f"Needs retraining: {needs_retraining}")
    print(f"Threshold: {RETRAIN_THRESHOLD}")

    assert "avg_bleu" in avg_scores, "Missing average BLEU"
    assert "avg_relevance" in avg_scores, "Missing average relevance"
    print("Feedback threshold test passed")


if __name__ == "__main__":
    test_evaluation_bleu()
    test_evaluation_rouge()
    test_evaluation_relevance()
    test_evaluation_full()
    test_pydantic_schemas()
    test_feedback_threshold()
    print("\nAll tests passed!")
