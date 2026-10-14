#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx", "pyarrow"]
# ///
"""Embed every catalog item with both models from ChalkSQL and store the vectors as parquet.

    ./build_catalog_vectors.py gs://<bucket>/catalog_text.parquet catalog.parquet

The vectors come from the deployed Chalk functions through `catalog_call`, so the catalog and
the visitor's search are embedded by exactly the same model.
"""

import base64
import io
import json
import os
import subprocess
import sys
import time

import httpx
import pyarrow.parquet as pq

QUERY = """SELECT sku, category, name, description,
       catalog_call('function.ironbark-embed-base', name || '. ' || description) AS base_vec,
       catalog_call('function.ironbark-embed-tuned', name || '. ' || description) AS tuned_vec
FROM read_parquet('{src}')"""


def main() -> None:
    src, dest = sys.argv[1], sys.argv[2]
    cfg = json.loads(subprocess.check_output(["chalk", "config", "--json"]))
    api, env = (
        cfg["apiServer"]["value"],
        os.environ.get("CHALK_ENVIRONMENT_ID", cfg["environment"]["value"]),
    )
    tok = httpx.post(
        f"{api}/v1/oauth/token",
        timeout=30,
        json={
            "client_id": cfg["clientId"]["value"],
            "client_secret": cfg["clientSecret"]["value"],
            "grant_type": "client_credentials",
        },
    ).json()["access_token"]
    t0 = time.perf_counter()
    r = httpx.post(
        f"{api}/chalk.protosql.v1.SqlService/ExecuteSqlQuery",
        timeout=900,
        headers={
            "Authorization": f"Bearer {tok}",
            "X-Chalk-Env-Id": env,
            "x-chalk-deployment-type": "engine-grpc",
        },
        json={"query": QUERY.format(src=src), "syncOptions": {}},
    )
    r.raise_for_status()
    body = r.json()
    if "parquet" not in body:
        sys.exit(json.dumps(body)[:2000])
    table = pq.read_table(io.BytesIO(base64.b64decode(body["parquet"])))
    pq.write_table(table, dest)
    print(
        f"{table.num_rows} items embedded in {time.perf_counter() - t0:.1f}s -> {dest}"
    )


if __name__ == "__main__":
    main()
