from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from arxiv_sage.retrieve import retrieve

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
    google_api_key=os.environ["GEMINI_API_KEY"],
)


TEMPLATE = """You are a precise research assistant answering questions about machine learning papers.

Use ONLY the information in the context below to answer the question.
Do not rely on any prior or outside knowledge, even if you are confident you know the answer.
If the context does not contain enough information to answer, reply exactly: "I don't know based on the provided context."
Keep your answer concise and factual. Do not add any claim that is not supported by the context.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(TEMPLATE)
chain = prompt | llm | StrOutputParser()


def format_context(docs) -> str:
    """
    Join the retrieved chunks into one context string &
    Prefix each chunk with its paper title
    """
    parts = []

    for doc in docs:
        parts.append(
            f"[{doc.metadata['title']}]\n{doc.page_content}"
        )

    return "\n\n".join(parts)


def answer(question: str, k: int = 5) -> str:
    """
    Retrieve k chunks, format them into a context string,
    and ask the LLM to answer using only that context.
    """
    docs = retrieve(question, k=k)

    context = format_context(docs)

    return chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )


if __name__ == "__main__":
    for q in [
        "What is scaled dot-product attention?",
        "What is the capital of France?",
    ]:
        print(f"Q: {q}")
        print(f"A: {answer(q)}")
        print("-" * 60)