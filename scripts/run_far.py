from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from far_attack.calibration import build_calibration_batches, load_calibration_texts
from far_attack.config import load_config
from far_attack.evaluation import evaluate_model
from far_attack.global_far_round import FarConfig, run_global_far_round
from far_attack.gradient_scoring import compute_gradients
from far_attack.model import load_model, load_tokenizer, release_model, transformer_linears
from far_attack.results import prepare_output_dir, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.mode != "far": raise SystemExit("run_far.py requires far config")
    output = prepare_output_dir(args.output, args.resume)
    tokenizer = load_tokenizer(cfg.tokenizer_id)
    model = load_model(cfg.model_id, args.device, args.dtype or cfg.dtype)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    layers = transformer_linears(model)
    texts = load_calibration_texts("assets/calibration/pileval_seed42_128x512.jsonl", cfg.calibration_samples)
    batches = build_calibration_batches(tokenizer, texts, cfg.calibration_sequence_length,
                                        cfg.calibration_batch_size)
    model.train()
    gradients = compute_gradients(model, layers, batches, args.device, cfg.behavior)
    model.eval()
    far_cfg = FarConfig(cfg.bits, cfg.group_size, cfg.aggressive_fraction, cfg.eps, cfg.histogram_bins)
    far_metrics, rows = run_global_far_round(layers, gradients, far_cfg)
    write_json(output / "far_metrics.json", far_metrics)
    with (output / "layer_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    evaluation = evaluate_model(model, tokenizer, cfg, output)
    write_json(output / "metrics.json", {"mode": "far", "far": far_metrics, "evaluation": evaluation})
    release_model(model)


if __name__ == "__main__": main()
