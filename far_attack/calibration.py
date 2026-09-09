from __future__ import annotations

import json
from pathlib import Path


def load_calibration_texts(path: str | Path, limit: int) -> list[str]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line)["raw_text"])
                if len(rows) == limit:
                    break
    return rows


def build_calibration_batches(tokenizer, texts, max_length: int, batch_size: int):
    batches = []
    for start in range(0, len(texts), batch_size):
        batches.append(tokenizer(texts[start:start + batch_size], return_tensors="pt",
                                 padding=True, truncation=True, max_length=max_length))
    return batches
