from pathlib import Path
from sentence_transformers import SentenceTransformer

from src.chunker import chunk_text
from src.document_reader import read_document


# Step 1: Find the project directory
project_root = Path(__file__).resolve().parent.parent

# Step 2: Locate the document
file_path = (
    project_root
    / "data"
    / "documents"
    / "employee_policy.txt"
)

# Step 3: Read the document
text = read_document(file_path)

# Step 4: Split the document into chunks
chunks = chunk_text(
    text,
    chunk_size=20,
    overlap=5
)

# Step 5: Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 6: Generate embeddings
embeddings = model.encode(chunks)

# Step 7: Display the results
print("Number of chunks:", len(chunks))
print("Embedding dimensions:", embeddings.shape)

for i, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {i}:")
    print(chunk)

    print("\nEmbedding shape:")
    print(embeddings[i - 1].shape)