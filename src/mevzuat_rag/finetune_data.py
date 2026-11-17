"""Training pairs for adapting the dense encoder to legal Turkish without hand-labelled queries."""
from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from .corpus import Chunk

MIN_QUERY_WORDS = 6
TITLE_QUERY_SHARE = 0.3
_SENTENCE_RE = re.compile(r"(?<=[.;:])\s+")


@dataclass(frozen=True)
class Pair:
    query: str
    passage: str
    chunk_id: str


def _passage(title: str, text: str) -> str:
    return f"{title}. {text}" if title else text


def epoch_pairs(chunks: Sequence[Chunk], *, seed: int) -> list[Pair]:
    """One pair per chunk, so in-batch negatives never contain a second copy of the positive.

    The query is either a sentence cut out of the chunk (inverse cloze task, Lee et al. 2019) or,
    for a share of chunks, the article heading when that heading is unique in the corpus.
    Generic headings such as "Amaç" repeat in every law and would teach the model nothing.
    """
    rng = random.Random(seed)
    title_counts = Counter(chunk.title for chunk in chunks if chunk.title)
    pairs: list[Pair] = []
    for chunk in chunks:
        if chunk.title and title_counts[chunk.title] == 1 and rng.random() < TITLE_QUERY_SHARE:
            pairs.append(Pair(chunk.title, chunk.text, chunk.chunk_id))
            continue
        sentences = [s for s in _SENTENCE_RE.split(chunk.text) if len(s.split()) >= MIN_QUERY_WORDS]
        if len(sentences) < 2:
            continue
        sentence = rng.choice(sentences)
        rest = " ".join(s for s in _SENTENCE_RE.split(chunk.text) if s != sentence)
        pairs.append(Pair(sentence, _passage(chunk.title, rest), chunk.chunk_id))
    rng.shuffle(pairs)
    return pairs
