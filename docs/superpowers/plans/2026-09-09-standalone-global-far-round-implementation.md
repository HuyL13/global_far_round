# Standalone Global FAR Round Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, Colab-ready repository that runs RTN4, RTN3, and the preserved Global FAR Round experiment with WikiText-2 PPL, IF-SFT FSR, live logs, and a combined summary.

**Architecture:** A small `far_attack` package owns shared model loading, affine RTN, FAR scoring/selection, PPL, fingerprint verification, configuration, and result I/O. Thin scripts run one experiment each; `run_full.sh` validates the environment and orchestrates RTN4, RTN3, FAR, and summary generation without installing dependencies or applying gates.

**Tech Stack:** Python 3.10+, PyTorch, Transformers, Accelerate, Datasets, PyYAML, NumPy, pandas, tqdm, matplotlib, pytest, Bash.

**Spec:** `docs/superpowers/specs/2026-09-09-standalone-global-far-round-design.md`

## Strict Phase1 reproduction override

> Latest user decision (2026-09-09): the RTN3-specific bullets in this
> section are superseded. RTN3 and RTN4 must match `quantization_attack`
> exactly: both use its Tier0 affine `rtn_quantize_weight_raw` behavior at
> group size 128, with bits 3 and 4 respectively. Phase1 remains authoritative
> only for PPL and containment FSR. Global FAR remains unchanged.

This section overrides conflicting steps below.

- RTN3 copies `phase1_if_analysis/src/phase1/quantization.py` and preserves
  `parameter.detach().float().cpu().numpy()`, symmetric qmin=-3/qmax=3,
  group size 128, the Phase1 module regex, and checkpoint export behavior.
- RTN4 and Global FAR keep the current Tier0 affine W4-G128 backend. RTN3
  and RTN4 deliberately do not share an implementation.
- `phase1_if_analysis/src/phase1/ppl.py` is copied verbatim and is the only
  PPL evaluator for RTN3, RTN4, and FAR.
- `phase1_if_analysis/src/phase1/if_sft_verifier.py` and the eight published
  Phase1 query rows are reused. Primary FSR is `target in generated`.
- Generation is greedy with `max_new_tokens=30`, `num_beams=1`,
  `repetition_penalty=1.0`, and Phase1 decoding without normalization.
- The final table uses `FSR_contains`; `FSR_exact` is optional and secondary.
- A100 acceptance requires fingerprinted seed42 RTN3 PPL near `8.248011` and
  containment FSR `8/8`. PPL near `6.6779` is a failed reproduction.
- Thin wrappers may add logging and result plumbing only; they may not alter
  RTN3, PPL, verifier, tokenization, generation, or metric behavior.

## Global Constraints

- Preserve current RTN3, RTN4, and Global FAR numerical behavior before any scientific algorithm change.
- Use `cnut1648/LLaMA2-7B-fingerprinted-SFT`, group size 128, RTN bits 3/4, and FAR bits 4.
- Vendor only the Tier0 affine quantization primitives, verifier logic, soft scoring, normalization, and exact eight-key asset required by these paths.
- Do not require another repository, `IF_AWQ_TIER0_ROOT`, path injection, environment creation, or package installation at runtime.
- `run_full.sh` always runs RTN4, RTN3, FAR, then summary and never gates or stops early.
- Python runs unbuffered; long stages print progress; stdout/stderr stream to Colab while logs are saved.
- CPU unit tests must not download or load the 7B checkpoint.
- The first A100 regression must investigate checksum differences rather than accepting them silently.

---

### Task 1: Package scaffold, config validation, and result I/O

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `far_attack/__init__.py`
- Create: `far_attack/config.py`
- Create: `far_attack/results.py`
- Create: `configs/rtn3.yaml`
- Create: `configs/rtn4.yaml`
- Create: `configs/far.yaml`
- Create: `results/.gitkeep`
- Test: `tests/test_config.py`
- Test: `tests/test_results.py`

**Interfaces:**
- Produces: `load_config(path: Path) -> ExperimentConfig`
- Produces: `ExperimentConfig` with model, quantization, evaluation, and FAR fields
- Produces: `write_json(path: Path, value: Any) -> None` and `read_json(path: Path) -> Any`
- Produces: `prepare_output_dir(path: Path, resume: bool = False) -> Path`

- [ ] **Step 1: Write failing config and result tests**

```python
def test_far_config_requires_four_bits(tmp_path):
    path = tmp_path / "far.yaml"
    path.write_text("model_id: fake\nbits: 3\ngroup_size: 128\nmode: far\n")
    with pytest.raises(ValueError, match="FAR requires bits=4"):
        load_config(path)

def test_output_directory_refuses_mixed_rerun(tmp_path):
    output = tmp_path / "run"
    prepare_output_dir(output)
    (output / "metrics.json").write_text("{}")
    with pytest.raises(FileExistsError):
        prepare_output_dir(output)
    assert prepare_output_dir(output, resume=True) == output
```

- [ ] **Step 2: Run the tests and verify missing modules fail**

Run: `python -m pytest tests/test_config.py tests/test_results.py -q`

Expected: collection fails because `far_attack.config` and `far_attack.results` do not exist.

- [ ] **Step 3: Implement typed config and JSON/output helpers**

Implement `ExperimentConfig.from_mapping()` with exact validation for bits,
group size, calibration counts, sequence lengths, FAR fraction, histogram
bins, and mode. Write JSON atomically through a sibling temporary file and
rename. Reject non-empty output directories unless `resume=True`.

- [ ] **Step 4: Add the three concrete YAML configs**

Use model ID `cnut1648/LLaMA2-7B-fingerprinted-SFT`, group size 128,
WikiText length 2048, generation length 32, and FAR settings copied from the
source config: 16 calibration samples, batch size 4, calibration length 512,
fraction 0.05, epsilon `1e-8`, behavior `top1_logprob`, and 65536 histogram
bins.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_config.py tests/test_results.py -q`

Expected: all tests pass.

Commit: `git commit -m "Build standalone config and result foundation"`

---

### Task 2: Shared model discovery and affine RTN3/RTN4

**Files:**
- Create: `far_attack/model.py`
- Create: `far_attack/rtn.py`
- Create: `scripts/run_rtn.py`
- Test: `tests/test_rtn.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Produces: `load_model_and_tokenizer(model_id: str, device: str, dtype: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]`
- Produces: `transformer_linears(model) -> dict[str, nn.Linear]`
- Produces: `quantize_weight_groupwise(weight: Tensor, bits: int, group_size: int) -> QuantizedTensorState`
- Produces: `apply_rtn(model, bits: int, group_size: int, progress: Callable[[int, int, str], None]) -> dict`

- [ ] **Step 1: Write failing RTN tests**

```python
@pytest.mark.parametrize("bits", [3, 4])
def test_groupwise_rtn_matches_reference_formula(bits):
    weight = torch.tensor([[-1.0, -0.2, 0.3, 1.0]])
    state = quantize_weight_groupwise(weight, bits=bits, group_size=4)
    expected_scale = (weight.max() - weight.min()) / (2**bits - 1)
    expected_zp = torch.round(-weight.min() / expected_scale).clamp(0, 2**bits - 1)
    expected_int = torch.round(weight / expected_scale + expected_zp).clamp(0, 2**bits - 1)
    assert torch.equal(state.integer_weights, expected_int)
    assert torch.allclose(state.dequantize(), (expected_int - expected_zp) * expected_scale)

def test_apply_rtn_only_touches_transformer_body_linears(fake_llama):
    original_head = fake_llama.lm_head.weight.detach().clone()
    metrics = apply_rtn(fake_llama, bits=3, group_size=4, progress=lambda *_: None)
    assert metrics["num_layers"] == 7 * len(fake_llama.model.layers)
    assert torch.equal(fake_llama.lm_head.weight, original_head)
```

- [ ] **Step 2: Run tests and verify they fail for missing RTN APIs**

Run: `python -m pytest tests/test_rtn.py tests/test_model.py -q`

- [ ] **Step 3: Vendor the minimal affine state and implement RTN**

Port the current Tier0 formula exactly: pad input features by group, compute
per-row/per-group min/max, scale `(max-min)/(2**bits-1)`, rounded zero point,
`torch.round` integer weights, clamp, dequantize, truncate padding, and restore
original dtype. Do not port AWQ activation scale search or packed artifacts.

- [ ] **Step 4: Implement the RTN CLI skeleton**

Parse `--config`, `--output`, `--device`, `--dtype`, and `--resume`. Load the
model, call `apply_rtn`, print every layer as `[RTN3] layer i/n: name` or
`[RTN4] ...`, and write quantization metadata. Evaluation is connected in
Task 5.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_rtn.py tests/test_model.py -q`

Commit: `git commit -m "Add shared RTN3 and RTN4 implementation"`

---

### Task 3: Vendor exact fingerprint verification and assets

**Files:**
- Create: `far_attack/fingerprint/__init__.py`
- Create: `far_attack/fingerprint/normalization.py`
- Create: `far_attack/fingerprint/scoring.py`
- Create: `far_attack/fingerprint/verifier.py`
- Create: `assets/fingerprints/if_sft_llama2_keys.json`
- Test: `tests/test_fingerprint_scoring.py`
- Test: `tests/test_fingerprint_asset.py`

**Interfaces:**
- Produces: `normalize_generated(text: str) -> str`
- Produces: `score_generation(generated: str, target: str) -> dict`
- Produces: `compute_soft_metrics(model, tokenizer, prompt: str, target: str, device: str) -> SoftMetricsResult`
- Produces: `run_verification(...) -> list[dict]` and `summarize(rows, elapsed: float | None) -> dict`

- [ ] **Step 1: Copy the official key asset byte-for-byte and record its SHA-256**

Copy from `../if_awq_tier0/artifacts/fingerprints/if_sft_llama2_keys.json`.
Record the source file SHA-256 in `tests/test_fingerprint_asset.py`; assert the
vendored file has that hash, contains eight entries, IDs 0 through 7, and the
same target for every entry.

- [ ] **Step 2: Write failing normalization, exact/contains, and soft-score tests**

```python
def test_score_generation_strips_only_outer_whitespace():
    result = score_generation("  ハリネズミ\n", "ハリネズミ")
    assert result["exact_success"] is True
    assert result["contains_success"] is True

def test_soft_metrics_ignore_prompt_positions():
    labels = [-100, -100, 2, 1]
    first = compute_soft_metrics_from_logits(reference_logits(), labels)
    changed = reference_logits()
    changed[0] += 1000
    second = compute_soft_metrics_from_logits(changed, labels)
    assert first.target_nll == pytest.approx(second.target_nll)
```

- [ ] **Step 3: Port only verifier dependencies**

Port normalization, exact/contains scoring, Wilson interval, teacher-forced
soft metrics, deterministic `generate`, and summary fields from Tier0. Add
`progress(index, total, key_id)` and print `[FSR] key i/8` in the default CLI
callback. Avoid environment metadata and standalone Tier0 CLI code.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_fingerprint_scoring.py tests/test_fingerprint_asset.py -q`

Commit: `git commit -m "Vendor IF-SFT fingerprint verification"`

---

### Task 4: WikiText-2 PPL and shared evaluation pipeline

**Files:**
- Create: `far_attack/ppl_eval.py`
- Create: `far_attack/evaluation.py`
- Modify: `scripts/run_rtn.py`
- Test: `tests/test_ppl_eval.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Produces: `load_corpus_ids(name, tokenizer, seqlen, cache_dir) -> Tensor`
- Produces: `compute_wikitext2_ppl(model, tokenizer, seqlen, cache_dir) -> dict`
- Produces: `evaluate_model(model, tokenizer, config, output_dir) -> dict`

- [ ] **Step 1: Write failing PPL block and evaluation serialization tests**

Use a tiny deterministic causal model and a monkeypatched corpus tensor.
Assert only complete non-overlapping 2048-style blocks are evaluated, the
reported block/token counts are correct, and evaluation writes both
`ppl_result.json` and `watermark_result.json`.

- [ ] **Step 2: Run tests and verify missing evaluation APIs fail**

Run: `python -m pytest tests/test_ppl_eval.py tests/test_evaluation.py -q`

- [ ] **Step 3: Port the current block PPL protocol**

Join the WikiText-2 raw test text with double newlines, tokenize once, use
complete non-overlapping blocks, accumulate float losses, display a terminal
compatible tqdm bar, and restore `model.config.use_cache` after evaluation.

- [ ] **Step 4: Compose PPL and FSR evaluation and finish RTN runner**

`evaluate_model` prints stage labels, evaluates PPL first and all eight keys
second, writes raw and summary artifacts, and returns a compact metrics dict.
`run_rtn.py` merges quantization and evaluation metrics and prints PPL plus
`FSR exact_hits/total`.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_ppl_eval.py tests/test_evaluation.py tests/test_rtn.py -q`

Commit: `git commit -m "Add shared PPL and FSR evaluation"`

---

### Task 5: Preserve Global FAR scoring, threshold, and rounding

**Files:**
- Create: `far_attack/gradient_scoring.py`
- Create: `far_attack/global_far_round.py`
- Create: `scripts/run_far.py`
- Test: `tests/test_far_selection.py`
- Test: `tests/test_gradient_scoring.py`

**Interfaces:**
- Produces: `compute_behavior_and_utility_gradients(model, layers, batches, device, behavior, progress) -> dict[str, tuple[Tensor, Tensor]]`
- Produces: `estimate_global_threshold(layers, order, gradients, config, progress) -> ThresholdResult`
- Produces: `run_global_far_round(layers, order, gradients, config, progress) -> GlobalFarRoundResult`

- [ ] **Step 1: Write failing FAR behavior-preservation tests**

Port focused source tests for global rather than per-layer budget, zero
fraction determinism, histogram floor ties, non-finite exclusion, projection
aggregation, bounded diagnostic quantiles, and checksum stability. Add a test
that captures the source behavior at exact `.5` so the migration does not
silently change the observed RTN4 flip semantics.

- [ ] **Step 2: Run tests and verify FAR modules are missing**

Run: `python -m pytest tests/test_far_selection.py tests/test_gradient_scoring.py -q`

- [ ] **Step 3: Port gradient scoring with explicit progress hooks**

Preserve the current two backward passes per calibration batch, bf16 CPU
gradient accumulators, `top1_logprob` behavior, utility cross-entropy, and
current padding behavior for regression. Print `[FAR] gradient batch i/n`.

- [ ] **Step 4: Port streaming histogram threshold and FAR application**

Preserve log range `[-30, 30]`, 65536-bin histogram, target count definition,
`>=` ties, current near/far candidate formula, selection checksum, bounded CPU
quantile diagnostics, and all source global/per-layer metrics. Print separate
`[FAR] threshold projection i/n` and `[FAR] rounding projection i/n` lines.

- [ ] **Step 5: Build the dedicated FAR runner**

Parse config/output/device/dtype/resume, require
`results/rtn4/ppl_result.json`, run calibration/scoring/FAR/evaluation, write
`far_metrics.json` and `layer_metrics.csv`, and report PPL/FSR plus comparison
to RTN4 without assigning PASS/FAIL.

- [ ] **Step 6: Run tests and commit**

Run: `python -m pytest tests/test_far_selection.py tests/test_gradient_scoring.py tests/test_evaluation.py -q`

Commit: `git commit -m "Isolate the Global FAR Round attack"`

---

### Task 6: Summary, optional sweep, and live full orchestration

**Files:**
- Create: `scripts/validate_environment.py`
- Create: `scripts/make_summary.py`
- Create: `scripts/run_far_sweep.py`
- Create: `run_full.sh`
- Test: `tests/test_summary.py`
- Test: `tests/test_orchestration.py`

**Interfaces:**
- Produces: `build_summary(results_root: Path) -> dict`
- Produces: root command `bash run_full.sh`
- Produces: optional command `python -u scripts/run_far_sweep.py`

- [ ] **Step 1: Write failing summary tests**

Create fixture result JSON for RTN4, RTN3, and FAR. Assert `build_summary`
reads actual values, includes bits/group size/PPL/FSR hits and total, includes
FAR threshold/selected ratio/flip ratio/checksum, and raises a path-specific
error when an artifact is missing.

- [ ] **Step 2: Write orchestration contract tests**

Read `run_full.sh` as text and assert it contains no install/venv commands,
sets `PYTHONUNBUFFERED=1`, invokes RTN4 before RTN3 before FAR before summary,
uses `python -u`, uses `tee` for each run, and enables `pipefail`.

- [ ] **Step 3: Implement validation and summary scripts**

Validation imports required libraries, checks CUDA, verifies all configs and
the eight-key asset, and prints actionable install guidance without changing
the environment. Summary reads artifacts, writes `results/summary.json`, and
prints the three-row PPL/FSR table plus FAR metrics.

- [ ] **Step 4: Implement `run_full.sh`**

Set `PYTHONUNBUFFERED=1` and `set -euo pipefail`; accept `OUTPUT`, `DEVICE`,
and `DTYPE`; validate first; create predictable stage directories; invoke
RTN4, RTN3, FAR, and summary in order using `python -u ... 2>&1 | tee ...`.
Never install, gate, or stop based on metric values.

- [ ] **Step 5: Implement optional seven-point sweep**

Use fractions `0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20`, one output
directory per fraction, existing RTN4 reference artifacts, and a JSON/CSV
aggregate. Keep it outside the default full run.

- [ ] **Step 6: Run tests and shell syntax validation, then commit**

Run: `python -m pytest tests/test_summary.py tests/test_orchestration.py -q`

Run: `bash -n run_full.sh`

Commit: `git commit -m "Add live full experiment orchestration"`

---

### Task 7: Minimal dependencies and standalone documentation

**Files:**
- Modify: `requirements.txt`
- Create: `README.md`
- Create: `.gitignore`
- Test: `tests/test_standalone_boundary.py`

**Interfaces:**
- Produces: documented clean Colab flow ending in `!bash run_full.sh`
- Produces: static guarantee that runtime code contains no external Tier0 dependency

- [ ] **Step 1: Write standalone-boundary tests**

Scan runtime Python, shell, README, and configs. Fail if they contain
`IF_AWQ_TIER0_ROOT`, imports from external `src`, commands cloning `if_awq`,
old method IDs, or package installs inside `run_full.sh`.

- [ ] **Step 2: Run the boundary test and verify documentation is absent**

Run: `python -m pytest tests/test_standalone_boundary.py -q`

- [ ] **Step 3: Build requirements from actual imports**

Include only torch, transformers, accelerate, datasets, PyYAML, NumPy,
pandas, tqdm, matplotlib, and pytest where used. Use compatible minimums and
do not force a CUDA wheel/version replacement.

- [ ] **Step 4: Write README and ignore generated artifacts**

Document purpose, RTN baselines, exact PPL/FSR protocol, installation,
standalone Colab cells, full command, individual commands, sweep, outputs,
configuration, reference run, and environment-dependent regression tolerance.
Ignore result contents and caches while retaining `results/.gitkeep`.

- [ ] **Step 5: Run boundary/full CPU tests and commit**

Run: `python -m pytest -q`

Commit: `git commit -m "Document standalone Colab reproduction"`

---

### Task 8: Source comparison and release verification

**Files:**
- Create: `scripts/compare_reference.py`
- Create: `tests/fixtures/reference_a100.json`
- Modify: `README.md`
- Test: `tests/test_compare_reference.py`

**Interfaces:**
- Produces: `compare_reference(actual: dict, reference: dict, strict_checksum: bool) -> ComparisonResult`

- [ ] **Step 1: Encode the observed A100 reference as data and write comparison tests**

Store the guide's RTN4 PPL and FAR threshold/count/ratio/flip/checksum/PPL/FSR.
Test exact integer/checksum comparison, configurable PPL tolerance, readable
missing-field errors, and nonzero CLI exit on structural mismatch.

- [ ] **Step 2: Implement the comparison command**

Read `results/summary.json` and the fixture; print a field-by-field table;
require exact selected count, total count, and checksum when strict mode is
enabled; use documented tolerances for floating metrics.

- [ ] **Step 3: Run all locally available verification**

Run: `python -m pytest -q`

Run: `python -m compileall -q far_attack scripts tests`

Run: `bash -n run_full.sh`

Run: `python -u scripts/validate_environment.py --allow-no-cuda` to validate
the local dependency surface without launching the 7B model.

- [ ] **Step 4: Run the A100 integration regression**

On Colab A100 40GB with dependencies installed, run `bash run_full.sh`, then
`python -u scripts/compare_reference.py --strict-checksum`. Preserve full
logs and record library/GPU metadata. If the checksum differs, compare config,
key hash, selected count, threshold, dtype, PyTorch version, and rounding
semantics before accepting a new reference.

- [ ] **Step 5: Commit verified regression metadata**

Add only compact environment/reference metadata and comparison output; do not
commit model weights, caches, generated corpora, or large logs.

Commit: `git commit -m "Verify standalone Global FAR reproduction"`

- [ ] **Step 6: Create and push the GitHub repository**

Create public repository `HuyL13/global_far_round`, add it as `origin`, and
push `main` after local verification. The remote operation uses the user's
existing GitHub authentication and occurs only after the repository is
reviewable.
