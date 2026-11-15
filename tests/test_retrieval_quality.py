from src.hybrid_retriever import retrieve_hybrid


test_questions = [
    {
        "question": "Who approves or rejects leave requests?",
        "expected_keyword": "managers",
    },
    {
        "question": "How many days of annual leave are provided?",
        "expected_keyword": "20 days",
    },
    {
        "question": "Who should receive a leave request?",
        "expected_keyword": "reporting managers",
    },
    {
        "question": "How many sick leave days are provided?",
        "expected_keyword": None,
    },
]


for test in test_questions:
    question = test["question"]
    expected_keyword = test["expected_keyword"]

    print("\n" + "=" * 65)
    print("QUESTION:", question)

    results = retrieve_hybrid(question, top_k=2)

    if not results:
        print("No documents retrieved.")
        continue

    for rank, result in enumerate(results, start=1):
        document = result["document"]

        print(f"\nRank: {rank}")
        print("Document:", document)
        print("Source:", result["source"])
        print("BM25 score:", round(result["bm25_score"], 4))
        print("Vector score:", round(result["vector_score"], 4))
        print("Hybrid score:", round(result["hybrid_score"], 4))

    if expected_keyword is not None:
        top_document = results[0]["document"].lower()

        if expected_keyword.lower() in top_document:
            print("\nRESULT: Relevant information found in the top result.")
        else:
            print("\nRESULT: Expected information not found in the top result.")
    else:
        print(
            "\nNOTE: This question is intentionally unanswerable "
            "from the current document. Retrieval may still return "
            "the closest available passages."
        )