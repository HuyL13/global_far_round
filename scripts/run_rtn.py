from __future__ import annotations

import argparse
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
from far_attack.rtn import apply_rtn_baseline


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
    if cfg.mode not in {"rtn3", "rtn4"}:
        raise SystemExit("run_rtn.py requires rtn3 or rtn4 config")
    model = load_model(cfg.model_id, args.device, args.dtype or cfg.dtype)
    tokenizer = load_tokenizer(cfg.model_id)
    quant = apply_rtn_baseline(model, cfg.bits, cfg.group_size)
    metrics = {"mode": cfg.mode, "quantization": quant,
               "evaluation": evaluate_model(model, tokenizer, cfg, output),
               "elapsed_seconds": time.time()-started}
    write_json(output / "metrics.json", metrics)
    del model


if __name__ == "__main__": main()
