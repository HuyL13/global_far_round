# Global FAR Round

Standalone reproduction of RTN3, RTN4, and Global FAR Round for
`cnut1648/LLaMA2-7B-fingerprinted-SFT`.

- RTN3 and RTN4 use the same affine `quantization_attack` RTN backend with
  group size 128; only `bits=3` versus `bits=4` differs.
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
The default config runs one Global FAR experiment at exactly 5%; it does not
run a fraction sweep.
After an A100 run, validate the frozen references with:

```bash
python -u scripts/compare_reference.py --strict-checksum
```
