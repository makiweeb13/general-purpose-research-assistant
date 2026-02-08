from langchain_core.documents import Document
from core.retrieval.vector_store import VectorDB

class Retriever:
    def __init__(self, vector_db: VectorDB):
        self.vector_db = vector_db

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Retrieve relevant documents from the vector store"""
        return self.vector_db.query(query, top_k=top_k)

    def combine_chunks(
        self,
        documents: list[Document],
        separator: str = "\n\n---\n\n"
    ) -> str:
        """Combine retrieved chunks into a single context string"""
        return separator.join([doc.page_content for doc in documents])

    def retrieve_and_combine(
        self,
        query: str,
        top_k: int = 5,
        separator: str = "\n\n---\n\n"
    ) -> str:
        """Retrieve documents and immediately combine them"""
        documents = self.retrieve(query, top_k)
        return self.combine_chunks(documents, separator)

    def retrieve_with_metadata(
        self,
        query: str,
        top_k: int = 5,
        separator: str = "\n\n---\n\n"
    ) -> tuple[str, list[dict]]:
        """Retrieve documents and return combined context with metadata"""
        documents = self.retrieve(query, top_k)
        context = self.combine_chunks(documents, separator)
        metadata = [doc.metadata for doc in documents]
        chunks_len = len(documents)
        return context, metadata, chunks_len

    def format_for_prompt(
        self,
        query: str,
        top_k: int = 5,
        context_header: str = "Context:"
    ) -> str:
        """Format retrieved context for LLM prompt construction"""
        context = self.retrieve_and_combine(query, top_k)
        return f"{context_header}\n{context}"
