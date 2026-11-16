"""Retrieval and abstention metrics on the hand-labelled question set."""
from __future__ import annotations

import json
from pathlib import Path

from .corpus import load_corpus
from .retrieval import Retriever, article_ranking

EVAL_PATH = Path(__file__).resolve().parents[2] / "data" / "eval" / "questions.jsonl"
KS = (1, 3, 5)
DEPTH = 10


def load_questions(path: Path = EVAL_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def retrieval_metrics(retriever: Retriever, questions: list[dict], *, mode: str) -> dict:
    hits_at = {k: 0 for k in KS}
    reciprocal = 0.0
    misses = []
    for item in questions:
        ranking = article_ranking(retriever.search(item["question"], mode=mode, k=DEPTH * 3))[:DEPTH]
        rank = next((i + 1 for i, article in enumerate(ranking) if article in item["gold"]), None)
        for k in KS:
            hits_at[k] += rank is not None and rank <= k
        reciprocal += 1.0 / rank if rank else 0.0
        if rank is None or rank > 5:
            misses.append({"question": item["question"], "gold": item["gold"], "got": ranking[:3]})
    n = len(questions)
    return {**{f"recall@{k}": hits_at[k] / n for k in KS}, "mrr@10": reciprocal / n, "misses": misses}


def abstention_metrics(retriever: Retriever, in_scope: list[dict], out_scope: list[dict], threshold: float) -> dict:
    answered = sum(retriever.dense_top_score(q["question"]) >= threshold for q in in_scope)
    abstained = sum(retriever.dense_top_score(q["question"]) < threshold for q in out_scope)
    return {
        "threshold": threshold,
        "in_scope_answered": answered / len(in_scope),
        "out_of_scope_abstained": abstained / len(out_scope),
    }


def run() -> dict:
    questions = load_questions()
    in_scope = [q for q in questions if q["gold"]]
    out_scope = [q for q in questions if not q["gold"]]
    retriever = Retriever(load_corpus())
    report = {
        "questions": len(in_scope),
        "out_of_scope": len(out_scope),
        "chunks": len(retriever.chunks),
        "articles": len({c.article_id for c in retriever.chunks}),
        "retrieval": {mode: retrieval_metrics(retriever, in_scope, mode=mode) for mode in ("bm25", "dense", "hybrid")},
    }
    from .graph import ABSTAIN_THRESHOLD

    report["abstention"] = [
        abstention_metrics(retriever, in_scope, out_scope, t) for t in (0.45, 0.50, ABSTAIN_THRESHOLD, 0.60)
    ]
    return report
