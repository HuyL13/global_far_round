# Standalone Global FAR Round Repository Design

## Purpose

Build a new repository named `global_far_round` that reproduces three
experiments against `cnut1648/LLaMA2-7B-fingerprinted-SFT`:

1. RTN4-G128 baseline;
2. RTN3-G128 baseline;
3. the current Global FAR Round attack at W4-G128.

Every path evaluates WikiText-2 perplexity and exact/contains fingerprint
success rate over the official eight IF-SFT keys. The repository measures and
reports results; it has no PPL gate, PASS status, or stop-early behavior.

## Source and scientific behavior

The authoritative attack source is
`HuyL13/quantization_attack@fragile-channel-ppl-eval`. RTN3, PPL, fingerprint
queries, tokenizer choice, and fingerprint verification are frozen to
`HuyL13/phase1_if_analysis`. RTN4 and Global FAR retain the current Tier0
affine grid used by Method 3.

The migration preserves the current experiment before making algorithmic
improvements. In particular, the initial standalone FAR implementation keeps
the current score, histogram threshold, near/far candidate construction,
selection semantics, and checksum behavior. The known A100 reference is:

- threshold: `2.6990652`;
- selected: `324421992 / 6476005376` (`5.0096%`);
- flip ratio: `7.5701%`;
- checksum: `96c6b598dddf864eb3a3462f6b6af68d37c0f0deeed9125d9686f50e1bf1c68a`;
- FAR PPL: `6.9738`;
- RTN4 PPL: `5.7981`;
- FAR containment FSR: `0/8`.

The mandatory Phase1 RTN3 fingerprinted reference (seed 42) is PPL
`8.248011` and containment FSR `8/8`. PPL near `6.6779` indicates a protocol
mismatch and must fail reference comparison.

The observed mismatch between selected ratio and RTN4 flip ratio is recorded
as a scientific follow-up. It is not silently changed during this refactor.
Any later rounding or scoring correction must be a separate, explicitly
versioned experiment.

## Repository boundary

The new repository contains no implementation for Methods 1, 2, or 4-8 and
does not import the old generic method runner. It has no runtime dependency on
a second checkout, `IF_AWQ_TIER0_ROOT`, path injection, packed AWQ, AWQ scale
search, GPTQ, lm-eval, or Tier0 orchestration.

Only the following Tier0 behavior is vendored:

- affine groupwise integer quantization state needed by RTN/FAR;
- fingerprint generation and exact/contains scoring;
- teacher-forced soft fingerprint metrics used by the existing verifier;
- normalization and small JSON utilities used by verification;
- the exact eight-key fingerprint JSON asset.

## Package layout

```text
global_far_round/
├── README.md
├── requirements.txt
├── run_full.sh
├── pyproject.toml
├── configs/
│   ├── rtn3.yaml
│   ├── rtn4.yaml
│   └── far.yaml
├── far_attack/
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── rtn.py
│   ├── gradient_scoring.py
│   ├── global_far_round.py
│   ├── ppl_eval.py
│   ├── evaluation.py
│   ├── results.py
│   ├── logging_utils.py
│   └── fingerprint/
│       ├── __init__.py
│       ├── verifier.py
│       ├── scoring.py
│       └── normalization.py
├── scripts/
│   ├── validate_environment.py
│   ├── run_rtn.py
│   ├── run_far.py
│   ├── run_far_sweep.py
│   └── make_summary.py
├── assets/fingerprints/if_sft_llama2_keys.json
├── tests/
│   ├── test_config.py
│   ├── test_rtn.py
│   ├── test_far_selection.py
│   ├── test_fingerprint_scoring.py
│   └── test_results.py
└── results/.gitkeep
```

Each Python module owns one responsibility. The runners compose package APIs
and contain no quantization or metric implementation.

## Experiment flow

`run_full.sh` performs these stages in order:

1. validate Python imports, CUDA availability, configs, and eight-key asset;
2. run RTN4 and stream/save its log;
3. run RTN3 and stream/save its log;
4. run one configured Global FAR experiment and stream/save its log;
5. read generated JSON files, write `results/summary.json`, and print a table.

Every Python process uses `python -u`. Shell pipelines use `tee` with
`pipefail`, so output is visible live in Colab and failures propagate. The
script never installs packages or creates an environment.

The default run is a single FAR experiment. The seven-point sweep remains an
explicit separate command.

## Deliberately separate RTN baselines

RTN3 calls the Phase1 NumPy symmetric INT3 implementation exactly: float32
CPU conversion, qmin=-3, qmax=3, group size 128, and the Phase1 module regex.
RTN4 calls the current Tier0 affine W4-G128 grid used internally by Global
FAR. These backends must not be unified.

Each RTN runner loads a fresh model, applies RTN, evaluates PPL and FSR, writes
artifacts, then releases the model. RTN4 and RTN3 do not gate one another.

## Global FAR implementation

The FAR runner loads a fresh model, builds the fixed calibration batches,
computes behavior and utility gradients, estimates the model-wide histogram
threshold, applies selection and far rounding in a streaming layer pass, and
evaluates PPL and FSR.

Progress is emitted for calibration batches, threshold-scoring projections,
rounding projections, WikiText blocks, and fingerprint keys. FAR writes at
least:

- threshold and timing;
- candidate, selected, and total weight counts;
- selected and RTN4 flip ratios;
- selection checksum;
- per-layer/per-projection metrics;
- PPL and watermark result JSON.

The RTN4 PPL reference is read automatically from
`results/rtn4/ppl_result.json`. It is reported as a comparison only and never
controls whether FSR runs.

## Fingerprint evaluation

The vendored Phase1 verifier uses raw published prompts, deterministic greedy
generation (`max_new_tokens=30`), unmodified decoding, and `target in
generated` containment. This containment score is primary FSR. Exact match
may be recorded only as a clearly labelled secondary diagnostic.

## PPL evaluation

All three paths call the copied Phase1 `src/phase1/ppl.py` evaluator without
rewriting it. They use tokenizer `NousResearch/Llama-2-7b-hf`, WikiText-2 raw
test split, and 2048-token non-overlapping blocks. Orchestration may log before
and after this call but must not change its internals.

## Outputs

```text
results/
├── rtn4/{run.log,ppl_result.json,watermark_result.json,metrics.json}
├── rtn3/{run.log,ppl_result.json,watermark_result.json,metrics.json}
├── far/{run.log,ppl_result.json,watermark_result.json,far_metrics.json,layer_metrics.csv,plots/}
└── summary.json
```

Reruns require a new `OUTPUT` directory or an explicit resume mode. A runner
must not silently mix partial results from different configurations.

## Configuration and dependencies

YAML files define model ID, bits, group size, calibration settings, FAR
fraction, histogram bins, PPL sequence length, and generation length. Config
validation fails before model download when values or assets are invalid.

`requirements.txt` contains only packages imported by the standalone paths.
It avoids pinning a replacement CUDA/PyTorch stack; Colab may satisfy the
minimum compatible PyTorch version already installed.

## Testing and regression

CPU tests use tiny tensors/models and require no Hugging Face download. They
cover:

- RTN3/RTN4 affine quantization and padding;
- FAR budget, threshold, ties, determinism, checksum, and zero-fraction path;
- exact/contains fingerprint normalization and soft target metrics;
- config validation and result serialization;
- shell orchestration structure where practical.

Static verification covers imports, shell syntax, package build, and tests.
The full A100 regression compares all recorded metrics with the source run.
The checksum must match under the same environment or trigger an
investigation. PPL differences are reported with environment metadata rather
than forced to equal the reference.

## Documentation and Git

README documents purpose, installation, Colab usage, `bash run_full.sh`,
individual RTN/FAR commands, sweep usage, outputs, reproducibility settings,
and reference results. It never instructs users to clone `if_awq`.

Implementation proceeds in focused commits: scaffold and core utilities;
RTN; fingerprint vendor; PPL; FAR; runners/orchestration; tests/results; README.
The old repositories remain unchanged.

The intended remote is `HuyL13/global_far_round`.
