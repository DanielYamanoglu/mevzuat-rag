"""LangGraph pipeline: guard -> retrieve -> grade -> generate, with refusal and abstention exits."""
from __future__ import annotations

import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from . import guard
from .llm import Answer, Generator
from .retrieval import Hit, Retriever

# Below this cosine similarity the best passage is treated as off-topic and the pipeline abstains
# instead of answering from weak evidence. Chosen on the evaluation set (see README).
ABSTAIN_THRESHOLD = 0.55
REFUSAL = "Bu istek güvenlik kuralları nedeniyle işlenmedi."
ABSTENTION = "Bu soru için derlemde yeterli kaynak bulunamadı."


class State(TypedDict, total=False):
    question: str
    safe_question: str
    masked: tuple[str, ...]
    hits: list[Hit]
    answer: Answer
    outcome: str  # "answered" | "refused" | "abstained"


def build_graph(retriever: Retriever, generator: Generator, *, k: int = 5):
    def guard_node(state: State) -> State:
        result = guard.check(state["question"])
        if result.injection:
            return {"outcome": "refused", "answer": Answer(REFUSAL, ()), "masked": result.masked}
        return {"safe_question": result.text, "masked": result.masked}

    def retrieve_node(state: State) -> State:
        # Mask placeholders carry no meaning for search and would drag similarity down.
        question = re.sub(r"\[[A-ZÇĞİÖŞÜ]+\]", " ", state["safe_question"])
        if retriever.dense_top_score(question) < ABSTAIN_THRESHOLD:
            return {"outcome": "abstained", "answer": Answer(ABSTENTION, ()), "hits": []}
        return {"hits": retriever.search(question, mode="hybrid", k=k)}

    def generate_node(state: State) -> State:
        return {"answer": generator.generate(state["safe_question"], state["hits"]), "outcome": "answered"}

    graph = StateGraph(State)
    graph.add_node("guard", guard_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "guard")
    graph.add_conditional_edges("guard", lambda s: END if s.get("outcome") else "retrieve")
    graph.add_conditional_edges("retrieve", lambda s: END if s.get("outcome") else "generate")
    graph.add_edge("generate", END)
    return graph.compile()
