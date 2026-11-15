from src import rag_pipeline


def test_supported_question_returns_answer_and_source(monkeypatch):
    def fake_retrieve_hybrid(question, top_k):
        return [
            {
                "document": (
                    "Managers are responsible for approving "
                    "or rejecting leave requests."
                ),
                "source": "employee_policy.txt",
            }
        ]

    def fake_rerank_documents(question, documents, top_k):
        return [
            {
                "document": documents[0],
                "rerank_score": 5.0,
            }
        ]

    class FakeOllamaClient:
        def chat(self, model, messages):
            return {
                "message": {
                    "content": (
                        "Managers approve or reject leave requests."
                    )
                }
            }

    monkeypatch.setattr(
        rag_pipeline,
        "retrieve_hybrid",
        fake_retrieve_hybrid,
    )

    monkeypatch.setattr(
        rag_pipeline,
        "rerank_documents",
        fake_rerank_documents,
    )

    monkeypatch.setattr(
        rag_pipeline,
        "ollama_client",
        FakeOllamaClient(),
    )

    result = rag_pipeline.generate_answer(
        "Who approves or rejects leave requests?"
    )

    assert result["answer"] == (
        "Managers approve or reject leave requests. [employee_policy.txt]"
    )
    assert result["sources"] == ["employee_policy.txt"]
    assert result["citations"] == [{
        "source": "employee_policy.txt",
        "passage": "Managers are responsible for approving or rejecting leave requests.",
    }]


def test_no_retrieved_documents_returns_fallback(monkeypatch):
    monkeypatch.setattr(
        rag_pipeline,
        "retrieve_hybrid",
        lambda question, top_k: [],
    )

    result = rag_pipeline.generate_answer(
        "What is the company's sick leave policy?"
    )

    assert result["answer"] == (
        "The provided documents do not contain enough "
        "information to answer this question."
    )
    assert result["sources"] == []


def test_low_reranker_score_abstains_without_calling_llm(monkeypatch):
    monkeypatch.setattr(
        rag_pipeline,
        "retrieve_hybrid",
        lambda question, top_k: [{"document": "Unrelated passage.", "source": "other.txt"}],
    )
    monkeypatch.setattr(
        rag_pipeline,
        "rerank_documents",
        lambda question, documents, top_k: [{"document": documents[0], "rerank_score": -1.0}],
    )

    class UnexpectedOllamaClient:
        def chat(self, **kwargs):
            raise AssertionError("The LLM should not run for a low-confidence result.")

    monkeypatch.setattr(rag_pipeline, "ollama_client", UnexpectedOllamaClient())
    result = rag_pipeline.generate_answer("What is the parental leave policy?")

    assert result["answer"] == rag_pipeline.FALLBACK
    assert result["sources"] == []
    assert result["citations"] == []
