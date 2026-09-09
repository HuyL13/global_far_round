from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from far_attack.results import read_json, write_json


def build_summary(root: Path):
    rows = []
    for key, label in [("rtn3", "RTN3 Phase1"), ("rtn4", "RTN4 Tier0"), ("far", "Global FAR 5%")]:
        metrics = read_json(root/key/"metrics.json")
        evaluation = metrics["evaluation"]
        rows.append({"id": key, "method": label, "ppl": evaluation["ppl"],
                     "fsr_contains": evaluation["fsr_contains"],
                     "contains_hits": evaluation["contains_hits"], "total": evaluation["total"],
                     "fsr_exact": evaluation["fsr_exact"]})
    return {"results": rows, "far": read_json(root/"far"/"far_metrics.json")}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--results", default="results")
    args = parser.parse_args(); root = Path(args.results); summary = build_summary(root)
    write_json(root/"summary.json", summary)
    print("\n================ FINAL RESULTS ================")
    print(f"{'Method':<22} {'PPL':>10} {'FSR_contains':>14}")
    print("-"*48)
    for row in summary["results"]:
        print(f"{row['method']:<22} {row['ppl']:>10.6f} {row['contains_hits']:>10}/{row['total']}")
    far = summary["far"]
    print(f"\nFAR selected: {far['global_num_selected']}/{far['global_num_weights']}")
    print(f"FAR flip ratio: {far['actual_rounding_flip_ratio']:.6%}")
    print(f"FAR checksum: {far['selection_mask_checksum']}")


if __name__ == "__main__": main()
