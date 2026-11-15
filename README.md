# Hybrid RAG Search with Reranking and Evaluation

A local document question answering project that combines lexical retrieval (BM25) and dense semantic retrieval (Sentence Transformers + Chroma), fuses their rankings with Reciprocal Rank Fusion (RRF), and reranks candidates with a cross encoder before optional LLM generation through Ollama.

## Pipeline

1. `ingest` reads `.txt` and `.md` files, chunks them with overlap, creates normalized embeddings, and upserts them into persistent Chroma. It refreshes edited sources and prunes removed files under the selected data directory.
2. `search` scores passages with a cached BM25 index and searches Chroma by embedding, then fuses the two candidate rankings with RRF. The embedding model stays loaded in memory between requests.
3. A cross encoder rescores fused candidates for query-passage relevance.
4. `ask` gives the reranked context to a local Ollama model, asks it to cite source filenames, and returns an answer with its supporting sources.
5. `evaluate` compares hybrid and reranked Recall@k, MRR, and nDCG@k on a multi-source JSONL benchmark.

## Setup

Python 3.10+ is recommended. From the project root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

The embedding and reranker models download from Hugging Face on first use. Ollama is only needed for answer generation; install Ollama and pull the configured model (`ollama pull gemma3:1b`). Retrieval and evaluation do not require Ollama.

## Use

Place `.txt` or `.md` files under `data/documents/`, then build/update the index:

```bash
python -m src.cli ingest
python -m src.cli search "Who approves leave requests?" --top-k 5
python -m src.cli ask "Who approves leave requests?"
python -m uvicorn src.api:app --reload
```

API: `GET /health`, `POST /ask` with `{"question":"..."}`. Interactive docs are at `/docs`.

## Evaluation

Edit `data/eval/queries.jsonl`: one JSON object per line with `question` and a `relevant` list. Labels may be a source filename (e.g. `employee_policy.txt`), a stored passage ID, or exact passage text. Then run:

```bash
python -m src.cli evaluate --file data/eval/queries.jsonl --k 5
```

The report compares hybrid ranking with reranked ranking. Judgments can use passage IDs, passage text, or source filenames; source-level judgments count a file once even when several of its chunks are retrieved. The sample benchmark includes several sources and an unanswerable question.

To also generate answers and compare rows with an `answer` reference, run `python -m src.cli evaluate --answers`; this requires Ollama and reports exact match, token F1, answerable/unanswerable accuracy, and source citation precision/recall when source labels are provided. Automatic scores do not prove factual faithfulness, so review citations and unsupported-question behavior when changing prompts or models.

## Configuration

Environment variables: `CHROMA_PATH` (defaults to `./chroma_db`), `CHROMA_COLLECTION` (`knowledge_chunks`), `EMBEDDING_MODEL` (`all-MiniLM-L6-v2`), `RERANKER_MODEL` (`cross-encoder/ms-marco-MiniLM-L-6-v2`), `ANSWER_MIN_RERANK_SCORE` (`0.0`, below which the pipeline abstains), `OLLAMA_HOST` (`http://localhost:11434`), and `LLM_MODEL` (`gemma3:1b`). Calibrate the reranker threshold on your own labeled benchmark, especially if you change reranker models. Use a fresh Chroma path if changing embedding models.

## Docker

```bash
docker compose up --build
# In another terminal, index the mounted documents once:
docker compose exec rag-api python -m src.cli ingest
```

The API listens on port 8000. The local `data/` and `chroma_db/` directories are mounted so you can ingest and retain the index. Configure Ollama separately and set `OLLAMA_HOST` to a reachable Ollama endpoint if you use `/ask`.
