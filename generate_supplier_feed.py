#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///
"""Write data/supplier_feed.jsonl: a new supplier's price-file lines for 60 catalog items.

Each line describes an Ironbark item the way a different vendor's ERP export would: terse,
abbreviated, its own part numbers. The SKU it corresponds to is kept as the answer key for
scoring the similarity join.
"""

import json
import random
import re
from pathlib import Path

import httpx

from generate_data import MODEL, gateway

PROMPT = """These are products from Ironbark Supply, a fictional industrial parts distributor:

{items}

A new supplier, Corwin Industrial (also fictional), sends a price file covering the same items. Write each item's line the
way their ERP export would: an uppercase abbreviated description (20-60 characters, vendor shorthand such as VLV, FTG, SS,
BRZ, MNPT, W/, ASSY), a fictional vendor part number, and a unit of measure. Use their own wording and word order, not
Ironbark's title.

Return only a JSON object mapping each product id to {{"vendor_part": str, "vendor_description": str, "uom": str}}."""


def main() -> None:
    data = Path(__file__).parent / "data"
    products = [
        json.loads(line) for line in (data / "products.jsonl").read_text().splitlines()
    ]
    picks = random.Random(11).sample(products, 60)
    url, headers = gateway()
    out: dict[str, dict[str, str]] = {}
    for i in range(0, len(picks), 20):
        batch = picks[i : i + 20]
        listing = "\n".join(
            f"- id {p['sku']}: {p['name']} — {p['description']}" for p in batch
        )
        r = httpx.post(
            url,
            headers=headers,
            timeout=600,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": PROMPT.format(items=listing)}],
                "max_tokens": 8000,
            },
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        out.update(json.loads(m.group(1) if m else text))
    with (data / "supplier_feed.jsonl").open("w") as f:
        for n, p in enumerate(picks, start=1):
            v = out[p["sku"]]
            f.write(
                json.dumps(
                    {
                        "line": n,
                        "vendor_part": v["vendor_part"],
                        "vendor_description": v["vendor_description"],
                        "uom": v["uom"],
                        "sku": p["sku"],
                    }
                )
                + "\n"
            )
    print(f"{len(picks)} supplier lines")


if __name__ == "__main__":
    main()
