"""Fine-tune a sentence-transformers model on shopper-search -> catalog-item pairs and evaluate retrieval.

Shared by the local prototype (`proto_train.py`) and the Chalk training run (`train.py`).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class Split:
    products: list[dict]
    train_pairs: list[tuple[str, str]]
    test_queries: list[tuple[str, str]]  # (query, sku)
    train_queries: list[tuple[str, str]]


def product_text(p: dict) -> str:
    return f"{p['name']}. {p['description']}"


def make_split(
    products: list[dict], queries: list[dict], test_fraction: float, seed: int
) -> Split:
    skus = sorted(p["sku"] for p in products)
    rng = random.Random(seed)
    test_skus = set(rng.sample(skus, int(len(skus) * test_fraction)))
    by_sku = {p["sku"]: p for p in products}
    train_pairs, test_q, train_q = [], [], []
    for q in queries:
        if q["sku"] in test_skus:
            test_q.append((q["query"], q["sku"]))
        else:
            train_pairs.append((q["query"], product_text(by_sku[q["sku"]])))
            train_q.append((q["query"], q["sku"]))
    return Split(products, train_pairs, test_q, train_q)


def evaluate(
    model, products: list[dict], queries: list[tuple[str, str]]
) -> dict[str, float]:
    skus = [p["sku"] for p in products]
    doc = model.encode(
        [product_text(p) for p in products],
        batch_size=128,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    qv = model.encode(
        [q for q, _ in queries],
        batch_size=128,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    sims = qv @ doc.T
    idx = {s: i for i, s in enumerate(skus)}
    target = np.array([idx[s] for _, s in queries])
    rank = (sims > sims[np.arange(len(queries)), target][:, None]).sum(axis=1) + 1
    return {
        "recall@1": float((rank <= 1).mean()),
        "recall@5": float((rank <= 5).mean()),
        "recall@10": float((rank <= 10).mean()),
        "mrr@10": float(np.where(rank <= 10, 1.0 / rank, 0.0).mean()),
        "n": len(queries),
    }


def train(
    split: Split, out_dir: Path, epochs: int, batch_size: int, lr: float, seed: int
):
    import torch
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )
    from sentence_transformers.training_args import BatchSamplers

    torch.manual_seed(seed)
    model = SentenceTransformer(BASE_MODEL)
    ds = Dataset.from_dict(
        {
            "anchor": [a for a, _ in split.train_pairs],
            "positive": [p for _, p in split.train_pairs],
        }
    )
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        # In-batch negatives are only useful if a batch never holds the same product twice.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no",
        logging_steps=20,
        seed=seed,
        report_to=[],
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        loss=losses.MultipleNegativesRankingLoss(model),
    )
    trainer.train()
    model.save(str(out_dir / "model"))
    return model


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
