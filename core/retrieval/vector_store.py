from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from core.retrieval.embedder import Embedder

class VectorDB:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.db = None
        self.vector_dir = Path("data/vectors")

    def build_db(self, documents: list[Document]):
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedder.embed_documents(texts)
        self.db = FAISS.from_embeddings(embeddings, documents)

    def save(self, name: str) -> str:
        """Save the FAISS index to data/vectors/{name}"""
        if not self.db:
            raise ValueError("VectorDB has not been built yet.")
        
        save_path = self.vector_dir / name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.db.save_local(str(save_path))
        return str(save_path)

    def load(self, name: str) -> None:
        """Load a FAISS index from data/vectors/{name}"""
        load_path = self.vector_dir / name
        if not load_path.exists():
            raise FileNotFoundError(f"Vector store not found at {load_path}")
        
        self.db = FAISS.load_local(str(load_path), self.embedder.embeddings)

    def query(self, query_text: str, top_k: int = 5) -> list[Document]:
        if not self.db:
            raise ValueError("VectorDB has not been built yet.")
        query_embedding = self.embedder.embed_query(query_text)
        results = self.db.similarity_search_by_vector(query_embedding, k=top_k)
        return results