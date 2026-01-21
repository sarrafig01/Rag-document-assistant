from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from app.config import OPENAI_API_KEY, VECTORSTORE_PATH

def create_vectorstore(documents):
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    return vectorstore

def load_vectorstore():
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    return FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
