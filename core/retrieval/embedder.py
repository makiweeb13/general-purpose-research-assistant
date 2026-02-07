from langchain_community.embeddings import HuggingFaceEmbeddings

class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True}
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embeddings.embed_query(text)