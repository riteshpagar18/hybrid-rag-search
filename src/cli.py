"""Command line entry points for indexing, search, answering and evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.vector_store import DEFAULT_DATA_DIR, ingest_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid RAG tools")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Index .txt and .md files")
    ingest.add_argument("--directory", type=Path, default=DEFAULT_DATA_DIR)
    ingest.add_argument("--chunk-size", type=int, default=180)
    ingest.add_argument("--overlap", type=int, default=35)
    search = commands.add_parser("search", help="Search indexed passages")
    search.add_argument("question")
    search.add_argument("--top-k", type=int, default=5)
    ask = commands.add_parser("ask", help="Generate an answer with Ollama")
    ask.add_argument("question")
    evaluate = commands.add_parser("evaluate", help="Run the JSONL retrieval benchmark")
    evaluate.add_argument("--file", type=Path, default=Path("data/eval/queries.jsonl"))
    evaluate.add_argument("--k", type=int, default=5)
    evaluate.add_argument("--answers", action="store_true", help="Also score reference answers with Ollama")
    args = parser.parse_args()

    if args.command == "ingest":
        print(f"Indexed {ingest_directory(args.directory, args.chunk_size, args.overlap)} chunks.")
    elif args.command == "search":
        from src.hybrid_retriever import retrieve_hybrid
        from src.reranker import rerank_documents
        hits = retrieve_hybrid(args.question, top_k=max(args.top_k * 4, args.top_k))
        reranked = rerank_documents(args.question, [hit["document"] for hit in hits], args.top_k)
        by_text = {hit["document"]: hit for hit in hits}
        for rank, result in enumerate(reranked, 1):
            print(json.dumps({**by_text[result["document"]], "rerank_score": result["rerank_score"], "rank": rank}, ensure_ascii=False))
    elif args.command == "ask":
        from src.rag_pipeline import generate_answer
        print(json.dumps(generate_answer(args.question), ensure_ascii=False, indent=2))
    else:
        from src.evaluation import evaluate, evaluate_answers, load_jsonl
        examples = load_jsonl(args.file)
        report = {"retrieval": evaluate(examples, k=args.k)}
        if args.answers:
            report["answers"] = evaluate_answers(examples)
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
