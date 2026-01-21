from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_documents(folder_path: str):
    documents = []

    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)

        if file.endswith(".txt"):
            loader = TextLoader(path)
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            continue

        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(documents)
