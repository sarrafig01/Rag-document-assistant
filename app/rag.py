from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from app.embeddings import load_vectorstore
from app.config import OPENAI_API_KEY

def get_rag_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
