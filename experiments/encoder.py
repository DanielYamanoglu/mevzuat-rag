"""Mean-pooled sentence encoder on plain Hugging Face Transformers + PyTorch, and a Retriever that uses it."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from mevzuat_rag.retrieval import EMBED_MODEL, Retriever, _chunk_text

# The base model was trained with 128-token inputs; fine-tuning and evaluation keep the same window.
MAX_TOKENS = 128
ROOT = Path(__file__).resolve().parents[1]
FINETUNED_DIR = ROOT / "models" / "legal-minilm"


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(1) / weights.sum(1).clamp(min=1e-9)


class Encoder:
    def __init__(self, name_or_path: str | Path = EMBED_MODEL):
        self.device = device()
        self.tokenizer = AutoTokenizer.from_pretrained(str(name_or_path))
        self.model = AutoModel.from_pretrained(str(name_or_path)).to(self.device)

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        batch = self.tokenizer(
            list(texts), padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt"
        ).to(self.device)
        out = self.model(**batch).last_hidden_state
        return F.normalize(mean_pool(out, batch["attention_mask"]), dim=-1)

    @torch.no_grad()
    def __call__(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        self.model.eval()
        parts = [self.forward(texts[i : i + batch_size]).cpu().numpy() for i in range(0, len(texts), batch_size)]
        return np.concatenate(parts).astype(np.float32)


class EncoderRetriever(Retriever):
    """The project's Retriever (same BM25, same fusion), with the dense side swapped for ``encoder``."""

    def __init__(self, chunks, encoder: Encoder):
        self._encoder = encoder
        super().__init__(chunks, dense=True)

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        return self._encoder(list(texts))

    def _load_or_build(self, index_dir: Path) -> np.ndarray:
        return self._embed([_chunk_text(chunk) for chunk in self.chunks])
