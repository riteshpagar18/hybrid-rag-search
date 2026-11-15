from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# Find the project directory
project_root = Path(__file__).resolve().parent.parent

# Connect to the existing ChromaDB database
database_path = project_root / "chroma_db"

client = chromadb.PersistentClient(
    path=str(database_path)
)

# Access the existing collection
collection = client.get_or_create_collection(
    name="employee_policies"
)

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_documents(question, top_k=2, max_distance=1.0):
    # Handle an empty database
    if collection.count() == 0:
        return {
            "documents": [[]],
            "distances": [[]]
        }

    # Convert the question into an embedding
    question_embedding = model.encode(question).tolist()

    # Search the vector database
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(top_k, collection.count())
    )

    documents = results["documents"][0]
    distances = results["distances"][0]

    # Display results for debugging
    print("\nRetrieval distances:", distances)
    print("Retrieved documents:", documents)

    # Filter chunks using the distance threshold
    relevant_documents = []
    relevant_distances = []

    for document, distance in zip(documents, distances):
        if distance <= max_distance:
            relevant_documents.append(document)
            relevant_distances.append(distance)

    return {
        "documents": [relevant_documents],
        "distances": [relevant_distances]
    }


if __name__ == "__main__":
    question = "How many sick leave days are provided?"

    results = retrieve_documents(question)

    print("\nQuestion:", question)
    print("\nRelevant documents:")

    documents = results["documents"][0]

    if documents:
        for document in documents:
            print("-", document)
    else:
        print("No sufficiently relevant documents found.")