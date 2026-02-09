import sys
from pathlib import Path

# Add parent directory to path so core modules can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from core.ingest.web_loader import WebLoader
from core.ingest.cleaner import clean_text
from core.ingest.chunker import chunk_document
from core.retrieval.vector_store import VectorDB
from core.retrieval.embedder import Embedder
from core.retrieval.retriever import Retriever
from core.llm.prompt import build_prompt
from core.llm.llama_client import LlamaClient

# Page configuration
st.set_page_config(
    page_title="General-Purpose Research Assistant",
    page_icon="🔍",
    layout="wide"
)

# Title
st.title("🔍 General-Purpose Research Assistant")

# Sidebar for URL inputs
st.sidebar.header("📚 Data Source")
st.sidebar.write("Enter URLs to load documents from:")

url1 = st.sidebar.text_input("URL 1", placeholder="https://example.com/article1")
url2 = st.sidebar.text_input("URL 2", placeholder="https://example.com/article2")
url3 = st.sidebar.text_input("URL 3", placeholder="https://example.com/article3")

urls = [url for url in [url1, url2, url3] if url.strip()]

# Process button
if st.sidebar.button("📖 Load & Index Documents"):
    if not urls:
        st.sidebar.error("Please enter at least one URL")
    else:
        with st.spinner("⏳ Loading and indexing documents..."):
            try:
                # Load documents
                web_loader = WebLoader()
                documents = web_loader.load(urls)
                
                # Clean text
                for doc in documents:
                    doc.page_content = clean_text(doc.page_content)
                
                # Chunk documents
                chunks = []
                for doc in documents:
                    chunks.extend(chunk_document(doc))
                
                # Build vector database
                vector_db = VectorDB(embedder=Embedder())
                vector_db.build_db(chunks)
                save_path = vector_db.save("research_index")
                
                # Store in session state
                st.session_state.vector_db = vector_db
                st.session_state.num_chunks = len(chunks)
                
                st.sidebar.success(f"✅ Loaded {len(documents)} documents, created {len(chunks)} chunks")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")

# Main content area
st.header("🤖 Ask Your Question")

if "vector_db" not in st.session_state:
    st.info("👈 Please load documents from the sidebar first")
else:
    # Question input
    question = st.text_area(
        "Enter your question:",
        placeholder="What is this about? Ask anything about the loaded documents...",
        height=100
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        generate_btn = st.button("🚀 Generate Answer", use_container_width=True)
    
    if generate_btn and question.strip():
        with st.spinner("⏳ Retrieving context and generating answer..."):
            try:
                # Retrieve and build prompt
                vector_db = st.session_state.vector_db
                retriever = Retriever(vector_db=vector_db)
                
                context, metadata, chunks_len = retriever.retrieve_with_metadata(question)
                prompt = build_prompt(context, question)
                
                # Generate answer
                llama_client = LlamaClient()
                answer = llama_client.generate_response(prompt)
                
                # Display results
                st.success("✅ Answer generated!")
                
                st.subheader("📝 Answer")
                st.write(answer)
                
                # Display metadata
                with st.expander("📊 Context Details"):
                    st.write(f"**Chunks retrieved:** {chunks_len}")
                    st.write(f"**Context length:** {len(context)} characters")
                    st.write(f"**Sources:** {len(metadata)} chunks")
                    
                    for i, meta in enumerate(metadata, 1):
                        st.write(f"**Chunk {i} source:** {meta.get('source', 'Unknown')}")
            except Exception as e:
                st.error(f"❌ Error generating answer: {str(e)}")
    elif generate_btn:
        st.warning("⚠️ Please enter a question")
