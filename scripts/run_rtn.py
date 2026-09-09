from __future__ import annotations

import argparse
import gc
from pathlib import Path
import random
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from far_attack.config import load_config
from far_attack.evaluation import evaluate_model
from far_attack.model import load_model, load_tokenizer
from far_attack.results import prepare_output_dir, write_json
from far_attack.rtn import apply_affine_rtn4, apply_phase1_rtn3


def seed_all(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = prepare_output_dir(args.output, args.resume)
    seed_all(cfg.seed)
    started = time.time()
    if cfg.mode == "rtn3":
        checkpoint = output / "phase1_dequantized_checkpoint"
        print("[RTN3] loading source in float32 as required by Phase1", flush=True)
        model = load_model(cfg.model_id, args.device, "float32")
        quant = apply_phase1_rtn3(model, cfg.group_size, cfg.module_pattern)
        model.save_pretrained(checkpoint, safe_serialization=True)
        source_tokenizer = load_tokenizer(cfg.model_id)
        source_tokenizer.save_pretrained(checkpoint)
        write_json(checkpoint / "quantization_manifest.json", {
            **quant, "symmetric": True, "seed": cfg.seed,
            "source_checkpoint": cfg.model_id, "upstream": "phase1-rtn-export",
            "upstream_sha": "phase1.rtn.v1", "dense_quantized_weights": True,
            "storage_representation": "hf_dequantized",
        })
        del source_tokenizer
        del model
        gc.collect()
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"[RTN3] reloading exported checkpoint as {args.dtype or cfg.dtype}", flush=True)
        model = load_model(str(checkpoint), args.device, args.dtype or cfg.dtype)
        tokenizer = load_tokenizer(cfg.tokenizer_id)
    elif cfg.mode == "rtn4":
        model = load_model(cfg.model_id, args.device, args.dtype or cfg.dtype)
        tokenizer = load_tokenizer(cfg.tokenizer_id)
        quant = apply_affine_rtn4(model, cfg.group_size)
    else:
        raise SystemExit("run_rtn.py requires rtn3 or rtn4 config")
    metrics = {"mode": cfg.mode, "quantization": quant,
               "evaluation": evaluate_model(model, tokenizer, cfg, output),
               "elapsed_seconds": time.time()-started}
    write_json(output / "metrics.json", metrics)
    del model
    gc.collect()


if __name__ == "__main__": main()
