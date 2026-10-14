# Fine-tuned embeddings in ChalkSQL

Fine-tune a small open embedding model on your own domain, serve it as a Chalk function, and call
it from ChalkSQL for semantic search and similarity joins, alongside the base model it came from.

The example domain is **Ironbark Supply**, a fictional industrial parts distributor whose 1,002
catalog items are written in trade jargon ("Hydraulic Cartridge Check Valve, SAE-10, 5000 PSI")
while its customers search in plain words ("my hydraulic cylinder drifts down when the pump shuts
off"). Off-the-shelf embedding models are weak at exactly that gap.

## Results

`all-MiniLM-L6-v2`, fine-tuned for 3 epochs on 3,208 search-to-product pairs (12.7 s on one NVIDIA
L4 in a Chalk training run). Evaluated on 800 searches for 200 products the model never saw,
each ranked against the full catalog:

| Held-out searches | Base | Fine-tuned |
| --- | --- | --- |
| Right part ranked first | 35% | 54% |
| Right part in the top 5 | 63% | 83% |
| Right part in the top 10 | 74% | 91% |
| MRR@10 | 0.47 | 0.66 |

Tuning is not free everywhere: matching a new supplier's terse price-file lines to the catalog
(jargon to jargon, `generate_supplier_feed.py`) went from 58/60 correct with the base model to 55/60
with the tuned one. Both models stay deployed, so each query can use the one that fits its job.

## Pipeline

```
generate_data.py ─▶ Chalk dataset ─▶ train.py (chalkcompute.training, L4) ─▶ Chalk volume
   ─▶ fetch_model.py ─▶ deploy_functions.py (two Chalk functions)
   ─▶ build_catalog_vectors.py (ChalkSQL embeds the catalog) ─▶ deploy_app.py (comparison page)
```

| File | Role |
| --- | --- |
| `generate_data.py` | Writes the fictional catalog and four plain-language searches per item, through the Chalk model gateway. |
| `finetune.py` | Product-level train/held-out split, `MultipleNegativesRankingLoss` fine-tuning, recall/MRR evaluation. |
| `train.py` | Uploads the data as a Chalk dataset and runs the fine-tune as a `chalkcompute.training` run on a GPU; writes the model and metrics to a Chalk volume. |
| `fetch_model.py` | Downloads the trained model and `summary.json` from that volume. |
| `deploy_functions.py` | Deploys `function.ironbark-embed-base` and `function.ironbark-embed-tuned` (CPU). |
| `build_catalog_vectors.py` | One ChalkSQL query that embeds the whole catalog with both functions. |
| `generate_supplier_feed.py` | A fictional supplier's price file for the similarity-join example. |
| `pick_presets.py` | Ranks every held-out search with both models and lists candidates for the demo's suggestions. |
| `app/`, `deploy_app.py` | A scaling group that runs each search as two ChalkSQL queries and shows them side by side. |

## Run it

You need [`uv`](https://docs.astral.sh/uv/), the `chalk` CLI logged in to your environment (or
`CHALK_CLIENT_ID` / `CHALK_CLIENT_SECRET`), an environment that can schedule an `nvidia-l4` for
training, and a bucket your environment's ChalkSQL can read.

```bash
./generate_data.py                       # optional: data/ already has the generated catalog
./train.py --env-id <gpu-environment>    # ~6 min end to end, most of it image build and pull
./fetch_model.py --env-id <gpu-environment>
./deploy_functions.py --model-dir model/
# Upload data/*.parquet text tables to your bucket, then:
./build_catalog_vectors.py gs://<bucket>/catalog_text.parquet data/catalog.parquet
./deploy_app.py --catalog gs://<bucket>/catalog.parquet
```

Search from SQL:

```sql
WITH q AS (
    SELECT catalog_call('function.ironbark-embed-tuned',
                        'my hydraulic cylinder drifts down when the pump shuts off') AS v
)
SELECT p.sku, p.name, cosine_similarity_vector(p.tuned_vec, q.v) AS score
FROM read_parquet('gs://<bucket>/catalog.parquet') AS p CROSS JOIN q
ORDER BY score DESC
LIMIT 5
```

## Notes

- **New functions reach SQL after the engine's next deployment.** `catalog_call` resolves names
  from the catalog snapshot the engine took when it deployed; re-activate your current deployment
  (`chalk deployment activate --deployment <id>`) after the functions report Ready.
- **Load once, encode one at a time.** With `max_batching_size`, the function is called from a
  thread pool. Two threads loading the model at once corrupt it ("Cannot copy out of meta tensor"),
  and dozens of parallel CPU encodes outgrow the replica's memory; `deploy_functions.py` guards both
  with locks.
- **No secret in the app.** `deploy_app.py` sets `chalk_identity=True`, and the app runs its SQL with
  the workload token the platform mounts at `CHALK_WEB_IDENTITY_TOKEN_FILE`.
- **Untrusted input.** Searches go into the SQL only as an escaped string literal, capped at 160
  characters, with per-client and global rate limits.

## License

Apache-2.0. The base model,
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
is Apache-2.0. Ironbark Supply, its catalog, searches and suppliers are fictional, generated for
this example.
