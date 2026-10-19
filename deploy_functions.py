#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.5"]
# ///
"""Deploy the base and fine-tuned embedding models as Chalk functions.

    ./deploy_functions.py --model-dir model/ # the fine-tuned model from train.py

Every query engine in the environment then reaches them by catalog name:

    SELECT catalog_call('function.ironbark-embed-tuned', 'heat pump blows cold air in winter')
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TUNED_DIR = "/models/tuned"
_models: dict[str, object] = {}
_load_lock = threading.Lock()
_encode_lock = threading.Lock()


def _embed(which: str, text: str) -> list[float]:
    # Each replica loads a model once and keeps it for every later row. The batched handler
    # calls this from a thread pool, and two threads loading at once corrupt the weights
    # ("Cannot copy out of meta tensor"), so the first load holds a lock.
    if which not in _models:
        with _load_lock:
            if which not in _models:
                from sentence_transformers import SentenceTransformer

                _models[which] = SentenceTransformer(
                    BASE_MODEL if which == "base" else TUNED_DIR, device="cpu"
                )
    # The batched handler also encodes rows on many threads at once; with torch's own thread
    # pool under each, 64 parallel encodes outgrow the replica's memory. One encode at a time
    # takes a few milliseconds and keeps memory flat.
    with _encode_lock:
        vec = _models[which].encode([text or ""], normalize_embeddings=True)[0]  # type: ignore[attr-defined]
    return [float(x) for x in vec]


def ironbark_embed_base(text: str) -> list[float]:
    """all-MiniLM-L6-v2, unchanged: a 384-dim unit vector for `text`."""
    return _embed("base", text)


def ironbark_embed_tuned(text: str) -> list[float]:
    """all-MiniLM-L6-v2 fine-tuned on Ironbark Supply searches: a 384-dim unit vector."""
    return _embed("tuned", text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    import chalkcompute
    from chalkcompute import Image, RemoteFunction

    image = (
        Image.debian_slim(python_version="3.12")
        # The CPU torch wheel keeps the image under 1 GB; the CUDA one is about 9 GB.
        .run_commands(
            "pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu"
        )
        .pip_install(["sentence-transformers==5.1.1"])
        .run_commands(
            f"python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('{BASE_MODEL}')\""
        )
        .add_local_dir(str(args.model_dir), TUNED_DIR, strategy="volume")
    )
    for fn, name in (
        (ironbark_embed_base, "ironbark-embed-base"),
        (ironbark_embed_tuned, "ironbark-embed-tuned"),
    ):
        try:
            chalkcompute.function(
                image=image,
                name=name,
                cpu="2",
                memory="8Gi",
                env={"MALLOC_ARENA_MAX": "2"},
                min_replicas=1,
                max_replicas=2,
                # One SQL batch can be a thousand rows; the batched shim embeds them on a thread
                # pool instead of one after another.
                max_batching_size=64,
            )(fn)
        except Exception as exc:
            # The decorator polls readiness for a fixed 300s; a cold image pull can outlast it
            # while the rollout continues server-side.
            if "not ready" not in str(exc):
                raise
            print(f"{name}: {exc}; waiting longer")
            RemoteFunction.from_name(name).wait_ready(timeout=1200)
        print(f"deployed function.{name}")


if __name__ == "__main__":
    main()
