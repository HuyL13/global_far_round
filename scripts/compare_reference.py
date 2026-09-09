from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from far_attack.results import read_json


def compare(actual, reference, ppl_tolerance=0.05, strict_checksum=False):
    failures=[]
    indexed={row["id"]: row for row in actual["results"]}
    for method in ("rtn3", "rtn4", "far"):
        if method not in indexed: failures.append(f"missing result: {method}"); continue
        expected=reference[method]; row=indexed[method]
        if abs(row["ppl"]-expected["ppl"]) > ppl_tolerance:
            failures.append(f"{method}.ppl actual={row['ppl']} expected={expected['ppl']}")
        if "fsr_contains" in expected and row["fsr_contains"] != expected["fsr_contains"]:
            failures.append(f"{method}.fsr_contains actual={row['fsr_contains']} expected={expected['fsr_contains']}")
    far=actual["far"]
    for field in ("global_num_selected", "global_num_weights"):
        if far[field] != reference["far"][field]: failures.append(f"far.{field} mismatch")
    if strict_checksum and far["selection_mask_checksum"] != reference["far"]["selection_mask_checksum"]:
        failures.append("far.selection_mask_checksum mismatch")
    return failures


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--actual", default="results/summary.json")
    parser.add_argument("--reference", default="assets/reference_a100.json")
    parser.add_argument("--ppl-tolerance", type=float, default=0.05); parser.add_argument("--strict-checksum", action="store_true")
    args=parser.parse_args(); failures=compare(read_json(args.actual), read_json(args.reference), args.ppl_tolerance, args.strict_checksum)
    if failures:
        print("REFERENCE COMPARISON FAILED")
        for failure in failures: print(f"- {failure}")
        raise SystemExit(1)
    print("REFERENCE COMPARISON PASSED")


if __name__ == "__main__": main()
