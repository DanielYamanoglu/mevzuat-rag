"""BM25, dense and hybrid (reciprocal rank fusion) retrieval."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from .corpus import Chunk

INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "index"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Turkish is agglutinative: truncating tokens to a fixed prefix is a cheap, well-known stemmer for lexical search.
STEM_PREFIX = 5
RRF_K = 60
CANDIDATES = 30

_TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})
_TOKEN_RE = re.compile(r"[a-zçğıöşüâîû0-9]+")
STOPWORDS = frozenset(
    "ve veya ile bir bu şu o da de mi mı mu mü için ne nedir hangi nasıl olan olarak gibi kadar "
    "göre her en daha çok ise ya ki kim kaç".split()
)


def tokenize(text: str, stem: int = STEM_PREFIX) -> list[str]:
    tokens = _TOKEN_RE.findall(text.translate(_TR_LOWER).lower())
    return [token[:stem] if stem else token for token in tokens if token not in STOPWORDS]


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def _chunk_text(chunk: Chunk) -> str:
    return f"{chunk.title}. {chunk.text}" if chunk.title else chunk.text


class Retriever:
    """Holds both indexes; ``search`` picks the mode so the modes can be compared on equal terms."""

    def __init__(self, chunks: Sequence[Chunk], *, dense: bool = True, index_dir: Path = INDEX_DIR):
        self.chunks = list(chunks)
        self._bm25 = BM25Okapi([tokenize(_chunk_text(chunk)) for chunk in self.chunks])
        self._embedder = None
        self._matrix: np.ndarray | None = None
        if dense:
            self._matrix = self._load_or_build(index_dir)

    # -- dense index -------------------------------------------------------
    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        if self._embedder is None:
            from fastembed import TextEmbedding

            self._embedder = TextEmbedding(EMBED_MODEL)
        vectors = np.array(list(self._embedder.embed(list(texts))), dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def _load_or_build(self, index_dir: Path) -> np.ndarray:
        texts = [_chunk_text(chunk) for chunk in self.chunks]
        digest = hashlib.sha256((EMBED_MODEL + "\n".join(texts)).encode()).hexdigest()[:16]
        path = index_dir / f"dense-{digest}.npy"
        if path.exists():
            return np.load(path)
        matrix = self._embed(texts)
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, matrix)
        return matrix

    # -- ranking -----------------------------------------------------------
    def _bm25_ranking(self, query: str) -> list[tuple[int, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:CANDIDATES]
        return [(int(i), float(scores[i])) for i in order]

    def _dense_ranking(self, query: str) -> list[tuple[int, float]]:
        if self._matrix is None:
            raise RuntimeError("dense index disabled")
        scores = self._matrix @ self._embed([query])[0]
        order = np.argsort(-scores)[:CANDIDATES]
        return [(int(i), float(scores[i])) for i in order]

    def search(self, query: str, *, mode: str = "hybrid", k: int = 5) -> list[Hit]:
        if mode == "bm25":
            ranking = self._bm25_ranking(query)
        elif mode == "dense":
            ranking = self._dense_ranking(query)
        elif mode == "hybrid":
            fused: dict[int, float] = {}
            for ranked in (self._bm25_ranking(query), self._dense_ranking(query)):
                for rank, (index, _) in enumerate(ranked):
                    fused[index] = fused.get(index, 0.0) + 1.0 / (RRF_K + rank + 1)
            ranking = sorted(fused.items(), key=lambda pair: -pair[1])[:CANDIDATES]
        else:
            raise ValueError(f"unknown mode: {mode}")
        return [Hit(self.chunks[index], score) for index, score in ranking[:k]]

    def dense_top_score(self, query: str) -> float:
        """Cosine similarity of the best dense match; the graph uses it to decide whether to abstain."""
        return self._dense_ranking(query)[0][1]


def article_ranking(hits: Sequence[Hit]) -> list[str]:
    """Collapse chunk hits to distinct article ids, keeping rank order."""
    seen: list[str] = []
    for hit in hits:
        if hit.chunk.article_id not in seen:
            seen.append(hit.chunk.article_id)
    return seen
