# Global FAR Round

Standalone reproduction of RTN3, RTN4, and Global FAR Round for
`cnut1648/LLaMA2-7B-fingerprinted-SFT`.

- RTN3 is the exact Phase1 symmetric NumPy RTN export path (W3-G128).
- RTN4 is the affine Tier0 grid used by Method 3 (W4-G128).
- Global FAR keeps the current Method 3 score, histogram threshold, and far-round behavior at 5%.
- Every model uses the same frozen Phase1 WikiText-2 PPL evaluator and containment FSR verifier.

The primary fingerprint metric is `FSR_contains`; `FSR_exact` is a secondary diagnostic.

## Colab

Use an A100 40GB runtime whose environment already provides the packages in
`requirements.txt`, then run:

```bash
git clone https://github.com/HuyL13/global_far_round.git
cd global_far_round
bash run_full.sh
```

To save results elsewhere:

```bash
OUTPUT=/content/global_far_results DEVICE=cuda DTYPE=bfloat16 bash run_full.sh
```

The script streams logs and stores separate RTN3, RTN4, FAR, and summary logs.
RTN3 deliberately exports a dequantized float32 checkpoint before reloading it
for evaluation, matching Phase1. This requires roughly 14 GB of temporary disk.

Expected Phase1 fingerprinted RTN3 reference on seed 42 is PPL approximately
`8.248011` and containment FSR `8/8`. A result near `6.6779` means the wrong
quantization/evaluation protocol was used.

After an A100 run, validate the frozen references with:

```bash
python -u scripts/compare_reference.py --strict-checksum
```
