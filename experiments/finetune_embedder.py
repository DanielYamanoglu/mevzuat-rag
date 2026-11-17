"""Fine-tune the multilingual MiniLM encoder on the legislation with an in-batch contrastive loss.

No labelled queries are used for training: pairs come from mevzuat_rag.finetune_data (inverse cloze
and unique headings). The 48 hand-written evaluation questions are only used to score the result.
Hyperparameters are fixed up front, not tuned on the evaluation set; three seeds show the spread.

    uv run --group experiments python experiments/finetune_embedder.py
"""
from __future__ import annotations

import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from encoder import FINETUNED_DIR, ROOT, Encoder, EncoderRetriever
from mevzuat_rag.corpus import load_corpus
from mevzuat_rag.evaluate import load_questions, retrieval_metrics
from mevzuat_rag.finetune_data import epoch_pairs

EPOCHS = 3
BATCH = 32
LR = 2e-5
TEMPERATURE = 0.05
WARMUP_SHARE = 0.1
SEEDS = (0, 1, 2)
RESULTS = ROOT / "experiments" / "results" / "finetune.json"


def train(encoder: Encoder, chunks, seed: int) -> list[float]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    steps_per_epoch = -(-len(epoch_pairs(chunks, seed=0)) // BATCH)
    total = EPOCHS * steps_per_epoch
    warmup = int(WARMUP_SHARE * total)
    optimizer = torch.optim.AdamW(encoder.model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / max(warmup, 1), max(0.0, (total - step) / max(total - warmup, 1)))
    )
    losses = []
    encoder.model.train()
    for epoch in range(EPOCHS):
        pairs = epoch_pairs(chunks, seed=seed * 100 + epoch)
        epoch_loss = 0.0
        for start in range(0, len(pairs), BATCH):
            batch = pairs[start : start + BATCH]
            q = encoder.forward([p.query for p in batch])
            p = encoder.forward([p.passage for p in batch])
            logits = q @ p.T / TEMPERATURE
            labels = torch.arange(len(batch), device=logits.device)
            # Symmetric InfoNCE: every other passage in the batch is a negative for a query, and vice versa.
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item() * len(batch)
        losses.append(round(epoch_loss / len(pairs), 4))
        print(f"epoch {epoch + 1}/{EPOCHS}  pairs={len(pairs)}  loss={losses[-1]:.4f}")
    return losses


def scores(encoder: Encoder, chunks, questions) -> dict:
    retriever = EncoderRetriever(chunks, encoder)
    report = {}
    for mode in ("dense", "hybrid"):
        metrics = retrieval_metrics(retriever, questions, mode=mode)
        metrics.pop("misses")
        report[mode] = {key: round(value, 3) for key, value in metrics.items()}
    return report


def main() -> None:
    chunks = load_corpus()
    questions = [q for q in load_questions() if q["gold"]]
    before = scores(Encoder(), chunks, questions)
    print("before", before)
    runs = []
    for seed in SEEDS:
        encoder = Encoder()
        started = time.time()
        losses = train(encoder, chunks, seed)
        minutes = (time.time() - started) / 60
        after = scores(encoder, chunks, questions)
        print(f"seed {seed} after", after)
        runs.append({"seed": seed, "train_minutes": round(minutes, 1), "epoch_loss": losses, "after": after})
        if seed == SEEDS[0]:
            FINETUNED_DIR.mkdir(parents=True, exist_ok=True)
            encoder.model.save_pretrained(FINETUNED_DIR)
            encoder.tokenizer.save_pretrained(FINETUNED_DIR)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "base_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "device": str(encoder.device),
                "epochs": EPOCHS,
                "batch": BATCH,
                "lr": LR,
                "temperature": TEMPERATURE,
                "train_pairs_per_epoch": len(epoch_pairs(chunks, seed=0)),
                "eval_questions": len(questions),
                "before": before,
                "runs": runs,
            },
            indent=2,
        )
        + "\n"
    )
    print("wrote", RESULTS.relative_to(ROOT), "and", FINETUNED_DIR.relative_to(ROOT), f"(seed {SEEDS[0]})")


if __name__ == "__main__":
    main()
