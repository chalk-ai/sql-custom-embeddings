#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.5"]
# ///
"""Upload the Ironbark catalog as a Chalk dataset and fine-tune the embedding model on a GPU.

    ./train.py                  # upload the dataset, fine-tune on an L4, wait
    ./train.py --skip-upload    # reuse the dataset's latest revision

The training run receives the dataset as a Polars LazyFrame. It evaluates the base model,
fine-tunes it on the training searches, evaluates again on searches for products it never
saw, and writes the model and the metrics to the Chalk volume `sql-custom-embeddings`.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
# The training run mounts this script's source over /app, so the shared module lives elsewhere.
LIB_DIR = "/opt/sce"
DATASET = "ironbark-catalog-searches"
MODEL_VOLUME = "sql-custom-embeddings"


def finetune_catalog_embeddings(df: Any, config: dict[str, Any]) -> None:
    """Runs inside the training container: `df` is the dataset, `config` the run config."""
    import io
    import sys
    import tarfile
    import time
    from pathlib import Path

    import torch
    from chalkcompute import Volume
    from sentence_transformers import SentenceTransformer

    sys.path.insert(0, LIB_DIR)
    import finetune

    rows = df.collect().to_dicts() if hasattr(df, "collect") else df.to_dicts()
    products = [
        {k: r[k] for k in ("sku", "category", "name", "description")}
        for r in rows
        if r["kind"] == "product"
    ]
    queries = [
        {"sku": r["sku"], "query": r["query"]} for r in rows if r["kind"] == "query"
    ]
    # The run config travels as a protobuf Struct, which stores every number as a float.
    epochs, batch_size, seed = (
        int(config["epochs"]),
        int(config["batch_size"]),
        int(config["seed"]),
    )
    split = finetune.make_split(products, queries, float(config["test_fraction"]), seed)
    print(
        f"{len(products)} products, {len(split.train_pairs)} training pairs, "
        f"{len(split.test_queries)} held-out searches; cuda={torch.cuda.is_available()}",
        flush=True,
    )

    base = SentenceTransformer(finetune.BASE_MODEL, device="cuda")
    base_test = finetune.evaluate(base, products, split.test_queries)
    print("base", base_test, flush=True)
    out = Path("/tmp/sce")
    t0 = time.monotonic()
    tuned = finetune.train(split, out, epochs, batch_size, float(config["lr"]), seed)
    train_seconds = time.monotonic() - t0
    tuned_test = finetune.evaluate(tuned, products, split.test_queries)
    print("tuned", tuned_test, f"trained in {train_seconds:.0f}s", flush=True)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(out / "model", arcname="model")
    summary = {
        "base_model": finetune.BASE_MODEL,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": float(config["lr"]),
            "seed": seed,
        },
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "products": len(products),
        "train_pairs": len(split.train_pairs),
        "held_out_products": len({s for _, s in split.test_queries}),
        "train_seconds": round(train_seconds, 1),
        "held_out": {"base": base_test, "tuned": tuned_test},
        "test_skus": sorted({s for _, s in split.test_queries}),
    }
    volume = Volume(MODEL_VOLUME)
    volume.put_file("model.tar.gz", buf.getvalue())
    volume.put_file("summary.json", json.dumps(summary, indent=2))
    print(f"wrote {len(buf.getvalue()):,} bytes to volume {MODEL_VOLUME}", flush=True)


def dataset_columns(data_dir: Path) -> dict[str, list[Any]]:
    cols: dict[str, list[Any]] = {
        k: [] for k in ("kind", "sku", "category", "name", "description", "query")
    }
    for line in (data_dir / "products.jsonl").read_text().splitlines():
        p = json.loads(line)
        for k, v in (
            ("kind", "product"),
            ("sku", p["sku"]),
            ("category", p["category"]),
            ("name", p["name"]),
            ("description", p["description"]),
            ("query", None),
        ):
            cols[k].append(v)
    for line in (data_dir / "queries.jsonl").read_text().splitlines():
        q = json.loads(line)
        for k, v in (
            ("kind", "query"),
            ("sku", q["sku"]),
            ("category", None),
            ("name", None),
            ("description", None),
            ("query", q["query"]),
        ):
            cols[k].append(v)
    return cols


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--gpu", default="nvidia-l4")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    if args.env_id:
        os.environ["CHALK_ENVIRONMENT"] = args.env_id
    import chalkcompute
    from chalkcompute import DatasetClient, Image, Volume

    if not args.skip_upload:
        ref = DatasetClient().upload(DATASET, dataset_columns(HERE / "data"))
        print(f"uploaded dataset {ref}")

    image = (
        # PyTorch's CUDA runtime image already has torch; the base model is baked in so the
        # run never waits on a download.
        Image.base("pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime")
        .pip_install(
            [
                "sentence-transformers==5.1.1",
                "datasets==4.1.1",
                "accelerate==1.10.1",
                "polars==1.39.3",
            ]
        )
        .run_commands(
            'python -c "from sentence_transformers import SentenceTransformer; '
            + "SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')\""
        )
        .add_local_file(
            str(HERE / "finetune.py"), f"{LIB_DIR}/finetune.py", strategy="copy"
        )
    )
    trainer = chalkcompute.training(
        dataset=DATASET,
        image=image,
        name="sql-custom-embeddings",
        gpu=args.gpu,
        cpu="4",
        memory="16Gi",
    )(finetune_catalog_embeddings)

    config = {
        "epochs": args.epochs,
        "batch_size": 64,
        "lr": 5e-5,
        "seed": 7,
        "test_fraction": 0.2,
    }
    t0 = time.monotonic()
    try:
        run = trainer.run_training(config=config)
    except Exception as exc:
        # The SDK stops waiting on an image build after 600s, but the build carries on
        # server-side; a second submission picks up the finished image from cache.
        if "Image build" not in str(exc) or "timed out" not in str(exc):
            raise
        print(f"{exc}; resubmitting once the build finishes")
        run = trainer.run_training(config=config)
    print(f"training run {run.run_id} submitted; waiting...")
    run.wait(timeout=3600)
    print(f"training run finished in {time.monotonic() - t0:.0f}s")
    summary = json.loads(Volume(MODEL_VOLUME).read_file("summary.json"))
    print(json.dumps({k: v for k, v in summary.items() if k != "test_skus"}, indent=2))


if __name__ == "__main__":
    main()
