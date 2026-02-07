from langchain_community.document_loaders import UnstructuredURLLoader
from core.ingest.headers import DEFAULT_HEADERS

class WebLoader:
    def load(self, urls: list[str]):
        loader = UnstructuredURLLoader(
            urls=urls,
            headers=DEFAULT_HEADERS,
            strategy="fast"
        )
        return loader.load()