import json
import re

from arxiv_sage.retrieve import retrieve


def normalize(s: str) -> str:
    """Normalize text for comparison"""
    return re.sub(r"\s+", " ", s).strip().lower()


def evaluate_recall(k: int = 5) -> float:
    """
    Fraction of questions whose gold 'supporting_quote' appears
    in the top-k retrieved chunks
    """

    # Loading the evaluation set
    with open("eval/eval_set.jsonl", "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]

    hits = 0
    misses = []

    for example in examples:
        question = example["question"]
        gold = normalize(example["supporting_quote"])

        docs = retrieve(question, k=k)

        found = any(
            gold in normalize(doc.page_content)
            for doc in docs
        )

        if found:
            hits += 1
        else:
            misses.append(example["id"])

    recall = hits / len(examples)

    print(f"Recall@{k}: {recall:.3f}")
    print(f"Missed question IDs: {misses}")

    return recall


if __name__ == "__main__":
    recall = evaluate_recall(k=5)
    print(f"recall@5 (dense baseline) = {recall:.3f}")