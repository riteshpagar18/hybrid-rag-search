import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag_pipeline import generate_answer

app = FastAPI(title="Hybrid RAG Search", description="BM25 + dense retrieval with cross-encoder reranking.", version="1.0.0")
logger = logging.getLogger(__name__)

class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = generate_answer(question)
        return {"question": question, "answer": result["answer"],
                "sources": result["sources"], "citations": result.get("citations", [])}
    except Exception:
        logger.exception("Question processing failed")
        raise HTTPException(status_code=500, detail="Question processing failed. Check the server logs for details.") from None
