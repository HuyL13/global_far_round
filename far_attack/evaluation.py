from __future__ import annotations

import json
from pathlib import Path
import time

from vendor.phase1_eval.if_sft_verifier import verify
from vendor.phase1_eval.ppl import eval_ppl
from .results import write_json


def load_queries(path: str | Path) -> list[dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [{**row, "query_id": str(row.get("query_id", row.get("id"))),
             "upstream_prompt": row.get("upstream_prompt", row["prompt"])} for row in raw]


def evaluate_model(model, tokenizer, cfg, output_dir: str | Path,
                   query_path: str | Path = "assets/fingerprints/if_sft_llama2_keys.json") -> dict:
    output_dir = Path(output_dir)
    print("[EVAL] computing frozen Phase1 WikiText-2 PPL", flush=True)
    started = time.time()
    ppl_values = eval_ppl(model, tokenizer, ["wikitext2"], seqlen=cfg.ppl_sequence_length,
                          cache_dir=Path(".cache/ppl"), verbose=False)
    ppl = ppl_values["wikitext2"]
    ppl_result = {"wikitext2_ppl": ppl, "protocol": "vendor.phase1_eval.ppl.eval_ppl",
                  "sequence_length": cfg.ppl_sequence_length, "elapsed_seconds": time.time()-started}
    write_json(output_dir / "ppl_result.json", ppl_result)
    print(f"[EVAL] WikiText-2 PPL = {ppl:.6f}", flush=True)

    queries = load_queries(query_path)
    print(f"[EVAL] running Phase1 containment verifier ({len(queries)} keys)", flush=True)
    started = time.time()
    result = verify(model=model, tokenizer=tokenizer, queries=queries,
                    generation={"do_sample": False, "num_beams": 1,
                                "repetition_penalty": 1.0, "max_new_tokens": 30},
                    settings={}, seed=cfg.seed)
    rows = result["queries"]
    contains_hits = sum(bool(row["verified"]) for row in rows)
    exact_hits = sum(row["generated_text"] == row["target_text"] for row in rows)
    watermark = {"summary": {"fsr_contains": result["fingerprint_score"],
                              "contains_hits": contains_hits, "total": len(rows),
                              "fsr_exact": exact_hits/len(rows), "exact_hits": exact_hits,
                              "elapsed_seconds": time.time()-started}, "per_key": rows}
    write_json(output_dir / "watermark_result.json", watermark)
    print(f"[EVAL] FSR_contains = {contains_hits}/{len(rows)} ({result['fingerprint_score']:.4f})", flush=True)
    return {"ppl": ppl, **watermark["summary"]}
