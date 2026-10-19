#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.5"]
# ///
"""Download the fine-tuned model and metrics that train.py's run wrote to its Chalk volume.

./fetch_model.py # writes model/ and summary.json
"""

import io
import tarfile
from pathlib import Path

from train import MODEL_VOLUME

from chalkcompute import Volume

here = Path(__file__).resolve().parent
volume = Volume(MODEL_VOLUME)
with tarfile.open(
    fileobj=io.BytesIO(volume.read_file("model.tar.gz")), mode="r:gz"
) as tar:
    tar.extractall(here, filter="data")
(here / "summary.json").write_bytes(volume.read_file("summary.json"))
print("wrote", here / "model", "and summary.json")
