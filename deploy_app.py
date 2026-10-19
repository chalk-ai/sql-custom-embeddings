#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute>=2.13.8"]
# ///
"""Deploy the comparison page: a scaling group that runs each search through ChalkSQL twice.

./deploy_app.py --catalog gs://<bucket>/catalog.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        required=True,
        help="parquet with sku, name, category, base_vec, tuned_vec",
    )
    args = parser.parse_args()
    from chalkcompute import Image, ReadinessProbe, ScalingGroup

    image = (
        Image.debian_slim(python_version="3.12")
        .pip_install(
            [
                "fastapi==0.116.1",
                "uvicorn[standard]==0.35.0",
                "httpx==0.28.1",
                "pyarrow==21.0.0",
                "chalkcompute>=2.13.8",
            ]
        )
        .add_local_file(str(HERE / "app/server.py"), "/app/server.py", strategy="copy")
        # The copy strategy has a 32 KiB budget across the image, so the page rides on a volume.
        .add_local_file(
            str(HERE / "app/index.html"), "/web/index.html", strategy="volume"
        )
        .add_local_file(
            str(HERE / "app/presets.json"), "/web/presets.json", strategy="volume"
        )
    )
    group = ScalingGroup(
        image=image,
        name='ironbark-search',
        entrypoint=["python", "/app/server.py"],
        port=8080,
        cpu="500m",
        memory="1Gi",
        min_replicas=1,
        max_replicas=1,
        # The app calls ChalkSQL as this workload's own identity, so no client secret ships with it.
        chalk_identity=True,
        env={
            "CATALOG_PARQUET": args.catalog,
        },
        readiness_probe=ReadinessProbe.http("/healthz"),
    )
    group.deploy(build_timeout=1200.0, ready_timeout=900.0)
    print("web_url:", group.web_url)


if __name__ == "__main__":
    main()
