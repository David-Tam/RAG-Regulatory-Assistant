from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

PROMPT_TEMPLATE = """
You are a regulatory compliance assistant for a Canadian bank.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I cannot find this in the provided documents."
Always mention which document your answer comes from.

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
    context = "\n\n".join(doc.page_content for doc in docs)
    print("--- CONTEXT ---")
    print(context[:2000])
    print("--- END ---\n")

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
    ask("What does IFRS 9 require for 12-month expected credit losses?")