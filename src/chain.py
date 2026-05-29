from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

PROMPT_TEMPLATE = """
You are a regulatory compliance assistant for a Canadian bank.
Use the context below to answer the question as fully as possible.
Only say "I cannot find this in the provided documents" if there is truly no relevant information at all.
Always mention which document your answer comes from.

Document aliases:
- "B-20" refers to the document "Residential mortgage underwriting practices and procedures"
- "E-23" refers to the document "Guideline E-23 Model Risk Management"
- "IFRS 9" refers to the document "ifrs-9-financial-instruments"

Context:
{context}

Question: {question}

Answer:
"""

def ask(question):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory="vectorstore",
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question)
    seen = set()
    unique_docs = []
    for doc in docs:
        key = (doc.metadata.get('source'), doc.metadata.get('page'))
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
    docs = unique_docs
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    formatted = prompt.format(context=context, question=question)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(formatted)

    print("\nAnswer:")
    print(response.content)
    print("\nSources:")
    for doc in docs:
        print(f"- {doc.metadata.get('source')} (page {doc.metadata.get('page')})")

if __name__ == "__main__":
    print("Regulatory Document Assistant")
    print("Documents: OSFI E-23, OSFI B-20, IFRS 9")
    print("Type 'quit' to exit\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() == "quit":
            break
        if not question:
            continue
        ask(question)
        print("\n" + "="*60 + "\n")