from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from arxiv_sage.ingest import extract_text

# Build these once when the module is imported
tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer,
    chunk_size=256,
    chunk_overlap=40,
)


def split_text(text: str) -> list[str]:
    """
    Split a document into overlapping chunks
    """
    return splitter.split_text(text)


if __name__ == "__main__":

    text = extract_text("data/papers/01-attention-is-all-you-need.pdf")

    chunks = split_text(text)

    tok_counts = [len(tokenizer.encode(c)) for c in chunks]

    print(f"{len(chunks)} chunks | max tokens in any chunk: {max(tok_counts)}")

    print("\n--- chunk[0] ---")
    print(chunks[0])

    print("\n--- chunk[1] ---")
    print(chunks[1])