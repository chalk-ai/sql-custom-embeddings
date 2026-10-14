#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["sentence-transformers==5.1.1", "torch"]
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Rank every held-out search with both models and print candidates for the demo's presets."""

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
from finetune import BASE_MODEL, load_jsonl, product_text

products = load_jsonl(here / "data/products.jsonl")
queries = load_jsonl(here / "data/queries.jsonl")
test_skus = set(json.loads((here / "summary.json").read_text())["test_skus"])
held = [q for q in queries if q["sku"] in test_skus]
skus = [p["sku"] for p in products]
by = {p["sku"]: p for p in products}
ranks = {}
for name, path in (("base", BASE_MODEL), ("tuned", str(here / "model"))):
    m = SentenceTransformer(path, device="cpu")
    d = m.encode(
        [product_text(p) for p in products], normalize_embeddings=True, batch_size=128
    )
    qv = m.encode([q["query"] for q in held], normalize_embeddings=True, batch_size=128)
    sims = qv @ d.T
    tgt = np.array([skus.index(q["sku"]) for q in held])
    ranks[name] = (sims > sims[np.arange(len(held)), tgt][:, None]).sum(1) + 1
rows = [
    dict(
        q,
        base=int(b),
        tuned=int(t),
        category=by[q["sku"]]["category"],
        name=by[q["sku"]]["name"],
    )
    for q, b, t in zip(held, ranks["base"], ranks["tuned"])
]
(here / "data/heldout_ranks.json").write_text(json.dumps(rows, indent=1))
wins = [r for r in rows if r["tuned"] == 1 and r["base"] > 5]
print(f"{len(wins)} held-out searches: tuned #1, base outside top 5")
for r in sorted(wins, key=lambda r: -r["base"])[:40]:
    print(
        f"base #{r['base']:4d} | {r['query'][:70]:70s} | {r['name'][:50]} [{r['category'][:18]}]"
    )
both = [r for r in rows if r["tuned"] == 1 and r["base"] == 1]
print(f"\n{len(both)} where both are #1, e.g.:", [r["query"] for r in both[:5]])
lose = [r for r in rows if r["tuned"] > 5]
print(
    f"{len(lose)} where tuned misses top 5, e.g.:",
    [(r["query"], r["base"], r["tuned"]) for r in lose[:6]],
)
