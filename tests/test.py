from core.ingest.web_loader import WebLoader
from core.ingest.cleaner import clean_text
from core.ingest.chunker import chunk_text
from core.retrieval.vector_store import VectorDB
from core.retrieval.embedder import Embedder

# temporary urls
urls = [
    "https://en.wikipedia.org/wiki/Steins;Gate_(TV_series)",
    "https://en.wikipedia.org/wiki/Steins;Gate"
]

web_loader = WebLoader()
documents = web_loader.load(urls)

print(f"Number of documents: {len(documents)}")

# for i, doc in enumerate(documents):
#     print(f"\nDocument {i+1} content (first 1000 chars):\n{doc.page_content[:1000]}")
#     print(f"Metadata: {doc.metadata}")

for doc in documents:
    doc.page_content = clean_text(doc.page_content)

# for i, doc in enumerate(documents):
#     print(f"\nCleaned Document {i+1} content (first 1000 chars):\n{doc.page_content[:1000]}")

chunks = chunk_text(doc.page_content)

# for doc in documents:
#     chunks = chunk_text(doc.page_content)
#     print(f"\nDocument with metadata {doc.metadata} has {len(chunks)} chunks.")
#     for j, chunk in enumerate(chunks[:3]):
#         print(f"Chunk {j+1} (first 500 chars):\n{chunk[:500]}")

vector_db = VectorDB(embedder=Embedder())
vector_db.build_db(documents)
save_path = vector_db.save("steins_gate_index")
print(f"Vector store saved at: {save_path}")

vector_db.load("steins_gate_index")
query = "What is Steins;Gate about?"
results = vector_db.query(query)
print(f"\nTop {len(results)} results for query: '{query}'")
for i, result in enumerate(results):
    print(f"\nResult {i+1} content (first 1000 chars):\n{result.page_content[:1000]}")
    print(f"Metadata: {result.metadata}")