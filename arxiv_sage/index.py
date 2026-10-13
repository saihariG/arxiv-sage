from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from arxiv_sage.corpus import build_documents


def build_index(
    persist_directory="chroma",
    collection_name="arxiv_sage",
):
    """Embed all corpus chunks and stores them in a persistent Chroma collection"""

    # Build all document chunks
    documents = build_documents()

    # Loading the embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )

    # Create and persist the vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )

    return vectorstore


if __name__ == "__main__":
    vs = build_index()
    print("Vectors in store:", vs._collection.count())