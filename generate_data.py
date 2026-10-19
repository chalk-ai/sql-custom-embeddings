#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["chalkcompute==2.13.5", "httpx"]
# ///
"""Generate the fictional Ironbark Supply catalog and shopper searches through the Chalk model gateway.

Writes data/products.jsonl (one catalog item per line) and data/queries.jsonl
(one plain-language search per line, labelled with the SKU it should find).
"""

import asyncio
import json
import random
import re
from pathlib import Path

import httpx

MODEL = "anthropic/claude-sonnet-5-5"
OUT = Path(__file__).parent / "data"
CATEGORIES = [
    "valves",
    "pipe fittings",
    "hose and hose fittings",
    "fasteners (bolts, nuts, screws)",
    "anchors and wall fasteners",
    "bearings and bushings",
    "seals, gaskets and o-rings",
    "power transmission (belts, pulleys, chain, sprockets)",
    "pneumatic components",
    "hydraulic components",
    "electrical wiring and connectors",
    "electrical boxes and conduit",
    "switches, relays and contactors",
    "motors and motor controls",
    "lighting",
    "hand tools",
    "power tool accessories (bits, blades, discs)",
    "measuring and inspection tools",
    "safety and PPE",
    "adhesives, sealants and tapes",
    "lubricants and chemicals",
    "material handling (casters, hoists, straps)",
    "plumbing fixtures and drains",
    "HVAC parts",
    "welding supplies",
]
PER_CATEGORY = 40


def gateway() -> tuple[str, dict[str, str]]:
    from chalkcompute import ConnectClient

    client = ConnectClient()
    try:
        client._ensure_auth()
        return f"{client._api_server.rstrip('/')}/v1/router/chat/completions", {
            "Authorization": f"Bearer {client._access_token}",
            "X-Chalk-Env-Id": client._env_id,
        }
    finally:
        client.close()


async def chat(
    client: httpx.AsyncClient, url: str, headers: dict[str, str], prompt: str
) -> object:
    for attempt in range(4):
        try:
            r = await client.post(
                url,
                headers=headers,
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 16000,
                },
                timeout=600,
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            return json.loads(m.group(1) if m else text)
        except Exception as e:  # noqa: BLE001 - retry any gateway or parse failure
            print(f"retry {attempt}: {str(e)[:120]}", flush=True)
            await asyncio.sleep(3 * (attempt + 1))
    raise RuntimeError("gave up")


PRODUCT_PROMPT = """You are writing catalog data for Ironbark Supply, a fictional industrial and MRO parts distributor.
Write {n} distinct products in the category "{cat}". Use the terse trade terminology a real distributor catalog uses:
standards, materials, thread and size callouts, ratings, abbreviations. Every product must be a different item type, not a
size or color variant of another one in the list. No real brand or manufacturer names.

Return only a JSON array; each element is {{"name": <catalog title with key specs, 4-12 words>,
"description": <one technical sentence, 15-35 words>}}."""

QUERY_PROMPT = """These are products from Ironbark Supply, a fictional industrial parts distributor:

{items}

For EACH product, write 4 searches a non-expert customer (a homeowner, a new maintenance tech, an office manager) might type
into the store's search box to find exactly that item. They do not know the trade name. Describe the job, the problem or
symptom, what the thing does, or what it looks like, in everyday words. Do NOT reuse the distinctive technical words from the
product's name (part type names, standards, abbreviations, material grades). Vary the style: a short phrase, a question,
a problem description, and a casual or slightly misspelled one. Each search should still point to that product rather than
its neighbors in the list.

Return only a JSON object mapping each product id to a list of 4 strings."""


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    url, headers = gateway()
    sem = asyncio.Semaphore(8)
    async with httpx.AsyncClient() as client:

        async def products_for(cat: str) -> list[dict[str, str]]:
            async with sem:
                items = await chat(
                    client, url, headers, PRODUCT_PROMPT.format(n=PER_CATEGORY, cat=cat)
                )
            print(f"{cat}: {len(items)}", flush=True)
            return [
                {
                    "category": cat,
                    "name": i["name"].strip(),
                    "description": i["description"].strip(),
                }
                for i in items
            ]

        per_cat = await asyncio.gather(*(products_for(c) for c in CATEGORIES))
        products = []
        seen: set[str] = set()
        for items in per_cat:
            for p in items:
                key = p["name"].lower()
                if key in seen:
                    continue
                seen.add(key)
                products.append(p)
        rng = random.Random(7)
        for i, sku in enumerate(rng.sample(range(10000, 99999), len(products))):
            products[i]["sku"] = f"IB-{sku}"

        batches = [products[i : i + 10] for i in range(0, len(products), 10)]

        async def queries_for(batch: list[dict[str, str]]) -> dict[str, list[str]]:
            listing = "\n".join(
                f"- id {p['sku']}: {p['name']} — {p['description']}" for p in batch
            )
            async with sem:
                out = await chat(
                    client, url, headers, QUERY_PROMPT.format(items=listing)
                )
            return {k: v for k, v in out.items() if k in {p["sku"] for p in batch}}

        results = await asyncio.gather(*(queries_for(b) for b in batches))

    with (OUT / "products.jsonl").open("w") as f:
        for p in products:
            f.write(json.dumps(p) + "\n")
    n = 0
    with (OUT / "queries.jsonl").open("w") as f:
        for res in results:
            for sku, qs in res.items():
                for q in qs:
                    f.write(json.dumps({"sku": sku, "query": q.strip()}) + "\n")
                    n += 1
    print(f"{len(products)} products, {n} queries")


if __name__ == "__main__":
    asyncio.run(main())
