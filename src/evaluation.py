"""Standard retrieval metrics and a small JSONL benchmark runner."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from src.hybrid_retriever import retrieve_hybrid
from src.reranker import rerank_documents


def recall_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    return len(relevant.intersection(ranked[:k])) / len(relevant) if relevant else 0.0


def reciprocal_rank(relevant: set[str], ranked: list[str]) -> float:
    return next((1.0 / rank for rank, doc in enumerate(ranked, 1) if doc in relevant), 0.0)


def ndcg_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    import math
    dcg = sum((1.0 / math.log2(rank + 1)) for rank, doc in enumerate(ranked[:k], 1) if doc in relevant)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def _judgment_label(result: dict, relevant: set[str]) -> str:
    if result["source"] in relevant:
        return result["source"]
    if result["document_id"] in relevant:
        return result["document_id"]
    return result["document"]


def _labels(results: list[dict], relevant: set[str]) -> list[str]:
    # At source-level, multiple chunks from one file count as one retrieved item.
    return list(dict.fromkeys(_judgment_label(result, relevant) for result in results))


def _metric_summary(relevant: set[str], labels: list[str], k: int) -> dict:
    return {f"recall@{k}": recall_at_k(relevant, labels, k),
            "mrr": reciprocal_rank(relevant, labels), f"ndcg@{k}": ndcg_at_k(relevant, labels, k)}


def evaluate(examples: Iterable[dict], k: int = 5, candidate_k: int = 20) -> dict:
    rows = []
    for example in examples:
        question = example["question"]
        relevant = set(example.get("relevant", []))
        candidates = retrieve_hybrid(question, top_k=candidate_k)
        reranked = rerank_documents(question, [item["document"] for item in candidates], top_k=candidate_k)
        docs = {item["document"]: item for item in candidates}
        reranked_results = [docs[item["document"]] for item in reranked if item["document"] in docs]
        hybrid_labels = _labels(candidates, relevant)
        reranked_labels = _labels(reranked_results, relevant)
        rows.append({"question": question,
                     "hybrid": _metric_summary(relevant, hybrid_labels, k),
                     "reranked": _metric_summary(relevant, reranked_labels, k),
                     "top_source": reranked_results[0]["source"] if reranked_results else None})
    count = len(rows)
    summary = {}
    for stage in ("hybrid", "reranked"):
        summary[stage] = {
            f"recall@{k}": sum(row[stage][f"recall@{k}"] for row in rows) / count if count else 0.0,
            "mrr": sum(row[stage]["mrr"] for row in rows) / count if count else 0.0,
            f"ndcg@{k}": sum(row[stage][f"ndcg@{k}"] for row in rows) / count if count else 0.0,
        }
    return {"k": k, "queries": count, **summary, "results": rows}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def answer_scores(prediction: str, reference: str) -> dict:
    """Normalized exact match and token-level F1 for answer judgments."""
    normalize = lambda text: " ".join(re.sub(r"\s*\[[^\]]+\.(?:txt|md)\]", "", text, flags=re.IGNORECASE).lower().strip().split())
    predicted, expected = normalize(prediction), normalize(reference)
    exact = float(predicted == expected)
    pred_tokens, ref_tokens = predicted.split(), expected.split()
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    precision = overlap / len(pred_tokens) if pred_tokens else 0.0
    recall = overlap / len(ref_tokens) if ref_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"exact_match": exact, "token_f1": f1}


def evaluate_answers(examples: Iterable[dict]) -> dict:
    """Score generated answers for rows containing an `answer` reference.

    Requires a running Ollama service and the configured LLM model.
    """
    from src.rag_pipeline import FALLBACK, generate_answer

    rows = []
    for example in examples:
        if not example.get("answer"):
            continue
        result = generate_answer(example["question"])
        scores = answer_scores(result["answer"], example["answer"])
        expected_answerable = example.get("answerable")
        predicted_answerable = result["answer"].strip().casefold() != FALLBACK.casefold()
        row = {"question": example["question"], **scores}
        if expected_answerable is not None:
            row["answerable_correct"] = float(predicted_answerable == bool(expected_answerable))
        relevant_sources = {label.casefold() for label in example.get("relevant", [])
                            if Path(label).suffix.lower() in {".txt", ".md"}}
        if relevant_sources:
            cited = {label.casefold() for label in
                     re.findall(r"\[([^\]]+\.(?:txt|md))\]", result["answer"], flags=re.IGNORECASE)}
            correct_citations = cited & relevant_sources
            row["citation_precision"] = len(correct_citations) / len(cited) if cited else 0.0
            row["citation_recall"] = len(correct_citations) / len(relevant_sources)
        rows.append(row)
    count = len(rows)
    citation_rows = [row for row in rows if "citation_precision" in row]
    answerability_rows = [row for row in rows if "answerable_correct" in row]
    return {"queries": count,
            "exact_match": sum(row["exact_match"] for row in rows) / count if count else 0.0,
            "token_f1": sum(row["token_f1"] for row in rows) / count if count else 0.0,
            "answerable_accuracy": (sum(row["answerable_correct"] for row in answerability_rows)
                                    / len(answerability_rows) if answerability_rows else None),
            "citation_precision": (sum(row["citation_precision"] for row in citation_rows)
                                   / len(citation_rows) if citation_rows else None),
            "citation_recall": (sum(row["citation_recall"] for row in citation_rows)
                                / len(citation_rows) if citation_rows else None),
            "results": rows}
