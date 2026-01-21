from app.loader import load_documents
from app.embeddings import create_vectorstore
from app.cli import start_cli
import os

DATA_PATH = "data/sample_docs"

def build_index():
    print("📄 Loading documents...")
    documents = load_documents(DATA_PATH)

    print("🧠 Creating vector store...")
    create_vectorstore(documents)

    print("✅ Vector store created!")

if __name__ == "__main__":
    if not os.path.exists("vectorstore/faiss_index"):
        build_index()

    start_cli()
