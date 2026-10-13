"""Download the arxiv-sage corpus (20 transformer papers) from arXiv.

Uses only the Python standard library so it runs before the project env exists.
For each paper it (1) queries the arXiv API to confirm the ID maps to the
expected title, (2) downloads the PDF, (3) verifies it is a real PDF, and
(4) writes a manifest. Be polite: ~3s between papers (arXiv's requested rate).

Run:  python3 scripts/fetch_papers.py
"""
import json
import os
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

# (number, short filename, expected arXiv title, arXiv id)
PAPERS = [
    (1, "attention-is-all-you-need", "Attention Is All You Need", "1706.03762"),
    (2, "bert", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "1810.04805"),
    (3, "roberta", "RoBERTa: A Robustly Optimized BERT Pretraining Approach", "1907.11692"),
    (4, "distilbert", "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter", "1910.01108"),
    (5, "albert", "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations", "1909.11942"),
    (6, "electra", "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators", "2003.10555"),
    (7, "t5-text-to-text", "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", "1910.10683"),
    (8, "bart", "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension", "1910.13461"),
    (9, "transformer-xl", "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context", "1901.02860"),
    (10, "xlnet", "XLNet: Generalized Autoregressive Pretraining for Language Understanding", "1906.08237"),
    (11, "gpt3-few-shot-learners", "Language Models are Few-Shot Learners", "2005.14165"),
    (12, "reformer", "Reformer: The Efficient Transformer", "2001.04451"),
    (13, "longformer", "Longformer: The Long-Document Transformer", "2004.05150"),
    (14, "big-bird", "Big Bird: Transformers for Longer Sequences", "2007.14062"),
    (15, "linformer", "Linformer: Self-Attention with Linear Complexity", "2006.04768"),
    (16, "performer", "Rethinking Attention with Performers", "2009.14794"),
    (17, "vit", "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", "2010.11929"),
    (18, "roformer", "RoFormer: Enhanced Transformer with Rotary Position Embedding", "2104.09864"),
    (19, "flashattention", "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", "2205.14135"),
    (20, "llama", "LLaMA: Open and Efficient Foundation Language Models", "2302.13971"),
]

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "papers"))
os.makedirs(OUT, exist_ok=True)
HEADERS = {"User-Agent": "arxiv-sage/0.1 (academic RAG project)"}
ATOM = "{http://www.w3.org/2005/Atom}"


def fetch_title(arxiv_id):
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        root = ET.fromstring(r.read())
    entry = root.find(f"{ATOM}entry")
    if entry is None:
        return None
    title = entry.find(f"{ATOM}title")
    return " ".join(title.text.split()) if title is not None and title.text else None


def download_pdf(arxiv_id, path):
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=90) as r:
        data = r.read()
    if not data.startswith(b"%PDF"):
        raise ValueError(f"not a PDF (starts with {data[:16]!r})")
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


def main():
    manifest = []
    print(f"Downloading {len(PAPERS)} papers -> {OUT}\n")
    for num, short, title, aid in PAPERS:
        fn = f"{num:02d}-{short}.pdf"
        path = os.path.join(OUT, fn)
        status, actual, size, note = "ok", None, 0, ""
        try:
            actual = fetch_title(aid)
            size = download_pdf(aid, path)
        except Exception as e:  # keep going; report at the end
            status, note = "FAIL", str(e)
        match = ""
        if actual:
            stem = title.lower().split(":")[0].strip()[:15]
            match = "ok" if stem in actual.lower() else "CHECK"
        print(f"[{status:4}] {fn:34} {size/1024:7.0f} KB  title={match:5}  {actual or note}")
        manifest.append({"num": num, "short": short, "expected_title": title,
                         "actual_title": actual, "arxiv_id": aid, "file": fn,
                         "bytes": size, "status": status, "note": note})
        time.sleep(3)

    with open(os.path.join(OUT, "..", "papers_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    ok = sum(1 for m in manifest if m["status"] == "ok")
    print(f"\nDone: {ok}/{len(PAPERS)} succeeded. Manifest -> data/papers_manifest.json")
    sys.exit(0 if ok == len(PAPERS) else 1)


if __name__ == "__main__":
    main()
