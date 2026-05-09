import nltk
from rouge_score import rouge_scorer
from collections import Counter
import math
import json

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def compute_bleu(reference, hypothesis, max_n=4):
    """Compute BLEU score between reference and hypothesis."""
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())

    if len(hyp_tokens) == 0:
        return 0.0

    scores = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(
            [tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1)]
        )
        hyp_ngrams = Counter(
            [tuple(hyp_tokens[i:i + n]) for i in range(len(hyp_tokens) - n + 1)]
        )

        overlap = sum((min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams))
        total = max(sum(hyp_ngrams.values()), 1)
        scores.append(overlap / total)

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))

    # Geometric mean of n-gram precisions
    if any(s == 0 for s in scores):
        return 0.0

    log_avg = sum(math.log(s) for s in scores) / len(scores)
    bleu = bp * math.exp(log_avg)

    return round(bleu, 4)


def compute_rouge(reference, hypothesis):
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores."""
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )
    scores = scorer.score(reference, hypothesis)

    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def compute_relevance(question, answer, context):
    """
    Compute a simple relevance score based on keyword overlap
    between question+answer and the retrieved context.
    """
    q_tokens = set(nltk.word_tokenize(question.lower()))
    a_tokens = set(nltk.word_tokenize(answer.lower()))
    c_tokens = set(nltk.word_tokenize(context.lower()))

    # Remove common stop words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on",
        "at", "to", "for", "of", "and", "or", "but", "not", "with",
        "this", "that", "it", "as", "by", "from", "be", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should",
        "what", "how", "why", "when", "where", "which", "who",
    }

    q_tokens = q_tokens - stop_words
    a_tokens = a_tokens - stop_words
    c_tokens = c_tokens - stop_words

    qa_tokens = q_tokens.union(a_tokens)

    if len(qa_tokens) == 0:
        return 0.0

    overlap = qa_tokens.intersection(c_tokens)
    relevance = len(overlap) / len(qa_tokens)

    return round(relevance, 4)


def evaluate_response(question, reference_answer, generated_answer, context=""):
    """Run full evaluation suite on a single response."""
    bleu = compute_bleu(reference_answer, generated_answer)
    rouge = compute_rouge(reference_answer, generated_answer)
    relevance = compute_relevance(question, generated_answer, context)

    return {
        "bleu": bleu,
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "relevance": relevance,
    }


def evaluate_batch(test_cases):
    """
    Evaluate a batch of test cases.
    Each test case should be a dict with:
    - question, reference_answer, generated_answer, context (optional)
    """
    results = []
    for case in test_cases:
        result = evaluate_response(
            question=case["question"],
            reference_answer=case["reference_answer"],
            generated_answer=case["generated_answer"],
            context=case.get("context", ""),
        )
        result["question"] = case["question"]
        results.append(result)

    # Compute averages
    avg = {
        "avg_bleu": round(sum(r["bleu"] for r in results) / len(results), 4),
        "avg_rouge1": round(sum(r["rouge1"] for r in results) / len(results), 4),
        "avg_rouge2": round(sum(r["rouge2"] for r in results) / len(results), 4),
        "avg_rougeL": round(sum(r["rougeL"] for r in results) / len(results), 4),
        "avg_relevance": round(sum(r["relevance"] for r in results) / len(results), 4),
    }

    return {"individual": results, "averages": avg}


if __name__ == "__main__":
    # Quick test
    ref = "NVIDIA invented the GPU in 1999 and has since expanded into AI and data centers."
    hyp = "NVIDIA created the GPU in 1999, later growing into AI, data center, and autonomous driving markets."
    ctx = "Our invention of the GPU in 1999 defined modern computer graphics. NVIDIA has expanded to AI and data centers."

    result = evaluate_response(
        question="When did NVIDIA invent the GPU?",
        reference_answer=ref,
        generated_answer=hyp,
        context=ctx,
    )
    print("Evaluation Results:")
    print(json.dumps(result, indent=2))
