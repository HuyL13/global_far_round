from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str
    model_id: str
    tokenizer_id: str
    bits: int
    group_size: int
    dtype: str = "bfloat16"
    seed: int = 42
    ppl_sequence_length: int = 2048
    max_new_tokens: int = 30
    aggressive_fraction: float = 0.05
    eps: float = 1e-8
    calibration_samples: int = 16
    calibration_batch_size: int = 4
    calibration_sequence_length: int = 512
    behavior: str = "top1_logprob"
    histogram_bins: int = 65536
    module_pattern: str = r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$"

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        cfg = cls(**raw)
        if cfg.mode not in {"rtn3", "rtn4", "far"}:
            raise ValueError("mode must be rtn3, rtn4, or far")
        expected = {"rtn3": 3, "rtn4": 4, "far": 4}[cfg.mode]
        if cfg.bits != expected:
            raise ValueError(f"{cfg.mode.upper()} requires bits={expected}")
        if cfg.group_size <= 0 or cfg.ppl_sequence_length <= 0:
            raise ValueError("group_size and ppl_sequence_length must be positive")
        if cfg.max_new_tokens != 30:
            raise ValueError("Phase1 verification requires max_new_tokens=30")
        if not 0 <= cfg.aggressive_fraction <= 1:
            raise ValueError("aggressive_fraction must be in [0, 1]")
        if cfg.histogram_bins < 2:
            raise ValueError("histogram_bins must be at least 2")
        return cfg


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config must be a YAML mapping")
    return ExperimentConfig.from_mapping(raw)
