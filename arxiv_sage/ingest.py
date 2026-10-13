import pymupdf

def extract_text(pdf_path: str) -> str:
    """Open a PDF and return all its text as one string"""

    with pymupdf.open(pdf_path) as doc:
        pages = []

        for page in doc:
            text = page.get_text()
            pages.append(text)

    return "\n".join(pages)


if __name__ == "__main__":
    # runs extract_text() on data/papers/01-attention-is-all-you-need.pdf
    # and print the first ~2000 characters
    path = "data/papers/01-attention-is-all-you-need.pdf"
    text = extract_text(path)
    print(f"total characters: {len(text)}\n")
    print(text[:2000])