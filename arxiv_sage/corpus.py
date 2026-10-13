import json

from pathlib import Path
from langchain_core.documents import Document

from arxiv_sage.ingest import extract_text
from arxiv_sage.chunk import split_text


def build_documents(
    papers_dir="data/papers",
    manifest_path="data/papers_manifest.json",
) -> list[Document]:
    """Extract + chunk all papers into Documents carrying source metadata"""

    # Loading the manifest (a list of paper entries)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    docs = []

    # Process each paper
    for entry in manifest:
        pdf_path = Path(papers_dir) / entry["file"]

        text = extract_text(str(pdf_path))
        chunks = split_text(text)

        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "title": entry.get("actual_title") or entry["expected_title"],
                    "arxiv_id": entry["arxiv_id"],
                    "source": entry["file"],
                    "chunk": i,
                },
            )
            docs.append(doc)

    return docs


if __name__ == "__main__":
    docs = build_documents()

    papers = {d.metadata["arxiv_id"] for d in docs}

    print(f"{len(docs)} chunks across {len(papers)} papers")

    print("\n--- sample metadata ---")
    print(docs[100].metadata)

    print("\n--- sample text ---")
    print(docs[100].page_content[:300])