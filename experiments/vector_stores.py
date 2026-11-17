"""Same embeddings, three serving paths: exact numpy search, ChromaDB (HNSW), and a LlamaIndex pipeline on Chroma.

Checks that moving retrieval into a vector database or a framework keeps the measured quality,
and records query latency for each path.

    uv run --group experiments python experiments/vector_stores.py [--model base|finetuned]
"""
from __future__ import annotations

import argparse
import json
import time

import chromadb
import numpy as np
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from encoder import FINETUNED_DIR, MAX_TOKENS, ROOT, Encoder
from mevzuat_rag.corpus import load_corpus
from mevzuat_rag.evaluate import load_questions, retrieval_metrics
from mevzuat_rag.retrieval import CANDIDATES, EMBED_MODEL, Hit, _chunk_text

RESULTS = ROOT / "experiments" / "results"


class RankedSearch:
    """Adapts any 'query -> ranked chunk ids' function to the interface retrieval_metrics expects."""

    def __init__(self, chunks, rank):
        self._by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self._rank = rank
        self.latencies: list[float] = []

    def search(self, query: str, *, mode: str = "dense", k: int = 5) -> list[Hit]:
        started = time.perf_counter()
        ids = self._rank(query)
        self.latencies.append(time.perf_counter() - started)
        return [Hit(self._by_id[chunk_id], 0.0) for chunk_id in ids[:k]]


def numpy_search(chunks, encoder: Encoder, matrix: np.ndarray):
    def rank(query: str) -> list[str]:
        scores = matrix @ encoder([query])[0]
        return [chunks[i].chunk_id for i in np.argsort(-scores)[:CANDIDATES]]

    return rank


def chroma_search(chunks, encoder: Encoder, matrix: np.ndarray):
    collection = chromadb.EphemeralClient().create_collection("mevzuat", metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[chunk.chunk_id for chunk in chunks],
        embeddings=matrix.tolist(),
        documents=[chunk.text for chunk in chunks],
        metadatas=[{"article_id": chunk.article_id, "law": chunk.law} for chunk in chunks],
    )

    def rank(query: str) -> list[str]:
        result = collection.query(query_embeddings=encoder([query]).tolist(), n_results=CANDIDATES)
        return result["ids"][0]

    return rank


def llamaindex_search(chunks, model_path: str):
    embed_model = HuggingFaceEmbedding(model_name=model_path, max_length=MAX_TOKENS, normalize=True)
    nodes = [
        TextNode(
            id_=chunk.chunk_id,
            text=_chunk_text(chunk),
            metadata={"article_id": chunk.article_id, "law": chunk.law},
            excluded_embed_metadata_keys=["article_id", "law"],
        )
        for chunk in chunks
    ]
    collection = chromadb.EphemeralClient().create_collection("mevzuat-li", metadata={"hnsw:space": "cosine"})
    storage = StorageContext.from_defaults(vector_store=ChromaVectorStore(chroma_collection=collection))
    index = VectorStoreIndex(nodes, storage_context=storage, embed_model=embed_model)
    retriever = index.as_retriever(similarity_top_k=CANDIDATES)

    def rank(query: str) -> list[str]:
        return [hit.node.node_id for hit in retriever.retrieve(query)]

    return rank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("base", "finetuned"), default="base")
    args = parser.parse_args()
    model_path = EMBED_MODEL if args.model == "base" else str(FINETUNED_DIR)

    chunks = load_corpus()
    questions = [q for q in load_questions() if q["gold"]]
    encoder = Encoder(model_path)
    matrix = encoder([_chunk_text(chunk) for chunk in chunks])

    report = {"model": args.model, "chunks": len(chunks), "eval_questions": len(questions), "paths": {}}
    for name, rank in (
        ("numpy_exact", numpy_search(chunks, encoder, matrix)),
        ("chromadb_hnsw", chroma_search(chunks, encoder, matrix)),
        ("llamaindex_chroma", llamaindex_search(chunks, model_path)),
    ):
        search = RankedSearch(chunks, rank)
        metrics = retrieval_metrics(search, questions, mode="dense")
        metrics.pop("misses")
        metrics = {key: round(value, 3) for key, value in metrics.items()}
        metrics["median_query_ms"] = round(1000 * float(np.median(search.latencies)), 1)
        report["paths"][name] = metrics
        print(name, metrics)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"vector_stores_{args.model}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
