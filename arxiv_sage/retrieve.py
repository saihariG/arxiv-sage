from langchain_chroma import Chroma

from arxiv_sage.embeddings import get_embeddings

vectorstore = Chroma(
    collection_name="arxiv_sage",
    persist_directory="chroma",
    embedding_function=get_embeddings(),
)


def retrieve(query: str, k: int = 5) -> list:
    """
    Return the k chunks most similar to the query
    """
    return vectorstore.similarity_search(query, k=k)


if __name__ == "__main__":
    q = "What is scaled dot-product attention?"

    print(f"Q: {q}\n")

    for i, (doc, score) in enumerate(
        vectorstore.similarity_search_with_score(q, k=5), 1
    ):
        print(
            f"[{i}] dist={score:.3f}  "
            f"{doc.metadata['title'][:45]}  "
            f"(chunk {doc.metadata['chunk']})"
        )

        print(
            "    ",
            doc.page_content[:160].replace("\n", " "),
            "\n",
        )