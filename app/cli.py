from app.rag import get_rag_chain

def start_cli():
    qa_chain = get_rag_chain()

    print("\n📘 RAG Document Assistant")
    print("Type 'exit' to quit\n")

    while True:
        question = input("❓ Question: ")
        if question.lower() == "exit":
            break

        response = qa_chain.run(question)
        print("\n💡 Answer:")
        print(response)
        print("-" * 50)
