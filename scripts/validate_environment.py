import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from far_attack.config import load_config


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--allow-no-cuda", action="store_true"); args=parser.parse_args()
    import torch, transformers, datasets, yaml, numpy
    if not torch.cuda.is_available() and not args.allow_no_cuda:
        raise SystemExit("CUDA GPU is required")
    for path in Path("configs").glob("*.yaml"): load_config(path)
    keys=json.loads(Path("assets/fingerprints/if_sft_llama2_keys.json").read_text(encoding="utf-8"))
    if len(keys) != 8: raise SystemExit(f"expected 8 fingerprint keys, got {len(keys)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    print("Configs and 8-key fingerprint asset: OK")


if __name__ == "__main__": main()
