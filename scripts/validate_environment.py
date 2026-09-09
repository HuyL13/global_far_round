import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from far_attack.config import load_config

FROZEN_HASHES = {
    "vendor/phase1_eval/quantization.py": "64c2f9747286bd191fd54ca2a850e85b9bdc1b5c8884942840000eaaf74508ac",
    "vendor/phase1_eval/ppl.py": "a05e1ac2984dc01dd3b595b9250388ee408f68cffaacceb4ad71a136822088bc",
    "vendor/phase1_eval/if_sft_verifier.py": "b785606a8cdb58c0ecf847cf40e4ddc055591f80145964a6887b27a79a83042c",
    "assets/fingerprints/if_sft_llama2_keys.json": "4885c8c1e6c7b58133bbdbb4150d9d3c00223b1dd0df6a206e8abaa498720be2",
}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--allow-no-cuda", action="store_true"); args=parser.parse_args()
    import torch, transformers, datasets, yaml, numpy
    if not torch.cuda.is_available() and not args.allow_no_cuda:
        raise SystemExit("CUDA GPU is required")
    for path in Path("configs").glob("*.yaml"): load_config(path)
    keys=json.loads(Path("assets/fingerprints/if_sft_llama2_keys.json").read_text(encoding="utf-8"))
    if len(keys) != 8: raise SystemExit(f"expected 8 fingerprint keys, got {len(keys)}")
    for name, expected in FROZEN_HASHES.items():
        actual=hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected: raise SystemExit(f"frozen source hash mismatch: {name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    print("Configs, frozen Phase1 sources, and 8-key fingerprint asset: OK")


if __name__ == "__main__": main()
