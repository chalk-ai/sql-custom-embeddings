"""Search the Ironbark Supply catalog with ChalkSQL, once with the base embedding model and
once with the fine-tuned one, and serve the comparison page.

Each search is two ChalkSQL queries. Each embeds the visitor's text with a Chalk function
(`catalog_call`) and ranks the catalog's precomputed vectors with `cosine_similarity_vector`.
"""

from __future__ import annotations

import asyncio
import base64
import io
import ipaddress
import json
import os
import re
import time
from collections import OrderedDict, defaultdict, deque
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

WEB = Path(os.environ.get("WEB_DIR", "/web"))
API = os.environ["CHALK_API_SERVER"].rstrip("/")
ENV_ID = os.environ["CHALK_ENVIRONMENT_ID"]
CATALOG = os.environ["CATALOG_PARQUET"]
MAX_QUERY_CHARS = 160
PER_IP_PER_MINUTE = 20
GLOBAL_PER_MINUTE = 300
CACHE_SIZE = 512

SQL_TEMPLATE = """WITH q AS (
    SELECT catalog_call('function.ironbark-embed-{model}', '{text}') AS v
)
SELECT p.sku, p.name, p.category,
       cosine_similarity_vector(p.{model}_vec, q.v) AS score
FROM read_parquet('{catalog}') AS p
CROSS JOIN q
ORDER BY score DESC
LIMIT 5"""

app = FastAPI()
_token: dict[str, Any] = {"value": None, "expires": 0.0}
_token_lock = asyncio.Lock()
_inflight = asyncio.Semaphore(4)
_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
_hits: dict[str, deque[float]] = defaultdict(deque)
_client = httpx.AsyncClient(timeout=60)


async def token() -> str:
    # With chalk_identity=True the platform mounts a short-lived workload JWT and rotates it in
    # place, so it is re-read on every call. Client credentials are only a local-run fallback.
    identity_file = os.environ.get("CHALK_WEB_IDENTITY_TOKEN_FILE")
    if identity_file and Path(identity_file).exists():
        return Path(identity_file).read_text().strip()
    async with _token_lock:
        if _token["value"] and time.time() < _token["expires"] - 300:
            return _token["value"]
        r = await _client.post(
            f"{API}/v1/oauth/token",
            json={
                "client_id": os.environ["CHALK_CLIENT_ID"],
                "client_secret": os.environ["CHALK_CLIENT_SECRET"],
                "grant_type": "client_credentials",
            },
        )
        r.raise_for_status()
        body = r.json()
        _token.update(
            value=body["access_token"],
            expires=time.time() + float(body.get("expires_in", 3600)),
        )
        return _token["value"]


def sql_literal(text: str) -> str:
    # The visitor's text only ever lands inside a single-quoted SQL string literal: control
    # characters are dropped and quotes doubled, so it cannot end the literal.
    return re.sub(r"[\x00-\x1f\x7f]", " ", text).replace("'", "''")


async def run_sql(query: str) -> tuple[list[dict[str, Any]], float]:
    headers = {
        "Authorization": f"Bearer {await token()}",
        "X-Chalk-Env-Id": ENV_ID,
        "x-chalk-deployment-type": "engine-grpc",
    }
    t0 = time.perf_counter()
    r = await _client.post(
        f"{API}/chalk.protosql.v1.SqlService/ExecuteSqlQuery",
        headers=headers,
        json={"query": query, "syncOptions": {}},
    )
    ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    body = r.json()
    if "parquet" not in body:
        raise HTTPException(
            502, f"ChalkSQL error: {json.dumps(body.get('errors', body))[:300]}"
        )
    rows = pq.read_table(io.BytesIO(base64.b64decode(body["parquet"]))).to_pylist()
    return [{**row, "score": round(float(row["score"]), 4)} for row in rows], ms


def client_ip(request: Request) -> str:
    # Proxies append the address they received the request from, so the rightmost public
    # address is the one the edge saw; entries to its left are whatever the client claimed.
    for part in reversed(request.headers.get("x-forwarded-for", "").split(",")):
        try:
            addr = ipaddress.ip_address(part.strip())
        except ValueError:
            continue
        if addr.is_global:
            return str(addr)
    return request.client.host if request.client else "?"


def rate_limited(ip: str) -> bool:
    now = time.time()
    for key, limit in ((ip, PER_IP_PER_MINUTE), ("*", GLOBAL_PER_MINUTE)):
        hits = _hits[key]
        while hits and now - hits[0] > 60:
            hits.popleft()
        if len(hits) >= limit:
            return True
    _hits[ip].append(now)
    _hits["*"].append(now)
    return False


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/healthz")
async def healthz() -> dict[str, bool]:
    return {"ok": True}


@app.get("/api/presets")
async def presets() -> JSONResponse:
    return JSONResponse(json.loads((WEB / "presets.json").read_text()))


@app.get("/api/search")
async def search(q: str, request: Request) -> JSONResponse:
    text = " ".join(q.split())[:MAX_QUERY_CHARS]
    if not text:
        raise HTTPException(400, "empty search")
    key = text.lower()
    if key in _cache:
        _cache.move_to_end(key)
        return JSONResponse({**_cache[key], "cached": True})
    if rate_limited(client_ip(request)):
        raise HTTPException(429, "Too many searches; try again in a minute.")
    literal = sql_literal(text)
    queries = {
        m: SQL_TEMPLATE.format(model=m, text=literal, catalog=CATALOG)
        for m in ("base", "tuned")
    }
    async with _inflight:
        (base, base_ms), (tuned, tuned_ms) = await asyncio.gather(
            run_sql(queries["base"]), run_sql(queries["tuned"])
        )
    result = {
        "query": text,
        "base": base,
        "tuned": tuned,
        "ms": {"base": round(base_ms), "tuned": round(tuned_ms)},
        "sql": queries,
    }
    _cache[key] = result
    if len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)
    return JSONResponse({**result, "cached": False})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
        proxy_headers=True,
    )
