from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(prog="mevzuat-rag")
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    evaluate = sub.add_parser("eval", help="run the evaluation set")
    evaluate.add_argument("--misses", action="store_true", help="include the questions each mode got wrong")
    args = parser.parse_args()

    if args.command == "eval":
        from .evaluate import run

        report = run()
        if not args.misses:
            for metrics in report["retrieval"].values():
                metrics.pop("misses")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    from .corpus import load_corpus
    from .graph import build_graph
    from .llm import default_generator
    from .retrieval import Retriever

    retriever = Retriever(load_corpus())
    state = build_graph(retriever, default_generator()).invoke({"question": args.question})
    by_article = {hit.chunk.article_id: hit.chunk.citation for hit in state.get("hits", [])}
    print(f"[{state['outcome']}] {state['answer'].text}")
    if state.get("masked"):
        print("Maskelenen kişisel veri:", ", ".join(state["masked"]))
    for article in state["answer"].citations:
        print("  kaynak:", by_article.get(article, article))


if __name__ == "__main__":
    main()
