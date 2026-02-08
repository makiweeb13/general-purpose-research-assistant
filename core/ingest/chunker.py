from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_document(document: Document, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(document.page_content)
    return [
        Document(page_content=chunk, metadata=document.metadata)
        for chunk in chunks
    ]
