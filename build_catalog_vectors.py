#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["chalkcompute>=2.13.8", "pyarrow"]
# ///
"""Embed every catalog item with both models from ChalkSQL and store the vectors as parquet.

    ./build_catalog_vectors.py gs://<bucket>/catalog_text.parquet catalog.parquet

The vectors come from the deployed Chalk functions through `catalog_call`, so the catalog and
the visitor's search are embedded by exactly the same model.
"""

import sys
import time

import pyarrow.parquet as pq

QUERY = """SELECT sku, category, name, description,
       catalog_call('function.ironbark-embed-base', name || '. ' || description) AS base_vec,
       catalog_call('function.ironbark-embed-tuned', name || '. ' || description) AS tuned_vec
FROM read_parquet('{src}')"""


def main() -> None:
    from chalkcompute import SQLClient

    src, dest = sys.argv[1], sys.argv[2]
    t0 = time.perf_counter()
    with SQLClient() as client:
        table = client.query(QUERY.format(src=src), timeout=900).to_arrow()
    pq.write_table(table, dest)
    print(f"{table.num_rows} items embedded in {time.perf_counter() - t0:.1f}s -> {dest}")


if __name__ == "__main__":
    main()
