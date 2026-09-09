from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time

from .rtn import affine_state


@dataclass(frozen=True)
class FarConfig:
    bits: int = 4
    group_size: int = 128
    aggressive_fraction: float = 0.05
    eps: float = 1e-8
    histogram_bins: int = 65536


def candidates(weight, cfg: FarConfig):
    import torch
    state = affine_state(weight, cfg.bits, cfg.group_size)
    floor = torch.floor(state.pre_round)
    near_ceil = state.pre_round - floor >= 0.5
    near = torch.where(near_ceil, floor + 1, floor).clamp(0, state.max_int)
    far = torch.where(near_ceil, floor, floor + 1).clamp(0, state.max_int)
    valid = far != near
    if state.padded_in_features > state.in_features:
        valid[:, state.in_features:] = False
    return state, near, far, valid


def _pad(gradient, state):
    import torch
    result = torch.zeros_like(state.pre_round, dtype=torch.float32)
    gradient = gradient.to(result.device, torch.float32)
    if gradient.shape == result.shape:
        result.copy_(gradient)
    elif gradient.ndim == 1:
        result[:, :state.in_features] = gradient.unsqueeze(0)
    else:
        result[:, :state.in_features] = gradient
    return result


def score_layer(module, gradients, cfg: FarConfig):
    import torch
    state, near, far, valid = candidates(module.weight.detach(), cfg)
    delta = (far - near) * state.scale
    behavior = _pad(gradients[0], state) * delta
    utility = _pad(gradients[1], state) * delta
    score = behavior.abs() / (utility.abs() + cfg.eps)
    score = torch.where(valid & torch.isfinite(score), score, torch.full_like(score, float("-inf")))
    return state, near, far, valid, score


def estimate_threshold(layers, order, gradients, cfg: FarConfig, progress=print):
    import torch
    histogram = torch.zeros(cfg.histogram_bins, dtype=torch.int64)
    candidate_count = weight_count = 0
    log_min, log_max = -30.0, 30.0
    for index, name in enumerate(order, 1):
        progress(f"[FAR] threshold projection {index}/{len(order)}: {name}")
        weight_count += layers[name].weight.numel()
        state, near, far, valid, score = score_layer(layers[name], gradients[name], cfg)
        values = score[valid & torch.isfinite(score)]
        candidate_count += values.numel()
        if values.numel():
            logs = torch.log10(values.float().clamp_min(10.0**log_min)).clamp(log_min, log_max)
            histogram += torch.histc(logs, bins=cfg.histogram_bins, min=log_min, max=log_max).cpu().long()
        del state, near, far, valid, score, values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    target = min(int(cfg.aggressive_fraction * weight_count), candidate_count)
    if target <= 0 or not candidate_count:
        return float("inf"), target, candidate_count, weight_count
    if target == candidate_count:
        return float("-inf"), target, candidate_count, weight_count
    descending = torch.flip(histogram, [0]).cumsum(0)
    reverse = int(torch.searchsorted(descending, torch.tensor(target), right=False))
    bin_index = cfg.histogram_bins - 1 - reverse
    threshold = 0.0 if bin_index == 0 else 10.0 ** (log_min + bin_index * (log_max-log_min)/cfg.histogram_bins)
    return threshold, target, candidate_count, weight_count


def run_global_far_round(layers, gradients, cfg: FarConfig, progress=print):
    import torch
    order = list(layers)
    started = time.time()
    threshold, target, candidate_count, weight_count = estimate_threshold(layers, order, gradients, cfg, progress)
    threshold_seconds = time.time() - started
    checksum = hashlib.sha256()
    selected_total = flipped_total = 0
    rows = []
    started = time.time()
    with torch.no_grad():
        for index, name in enumerate(order, 1):
            progress(f"[FAR] rounding projection {index}/{len(order)}: {name}")
            module = layers[name]
            state, near, far, valid, score = score_layer(module, gradients[name], cfg)
            selected = valid & torch.isfinite(score) & (score >= threshold)
            final = torch.where(selected, far, near)
            baseline = torch.round(state.pre_round).clamp(0, state.max_int)
            real_selected = selected[:, :state.in_features]
            real_flips = final[:, :state.in_features] != baseline[:, :state.in_features]
            checksum.update(real_selected.cpu().to(torch.uint8).numpy().tobytes())
            selected_count = int(real_selected.sum())
            flipped_count = int(real_flips.sum())
            selected_total += selected_count
            flipped_total += flipped_count
            dequant = (final - state.zero_point) * state.scale
            module.weight.copy_(dequant[:, :state.in_features].to(state.original_dtype))
            rows.append({"layer": name, "num_weights": module.weight.numel(),
                         "num_selected": selected_count,
                         "selected_fraction": selected_count/module.weight.numel(),
                         "rounding_flip_ratio_vs_rtn4": flipped_count/module.weight.numel()})
            del state, near, far, valid, score, selected, final, baseline, real_selected, real_flips, dequant
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    metrics = {"global_threshold": threshold, "target_num_selected": target,
               "global_num_candidates": candidate_count, "global_num_weights": weight_count,
               "global_num_selected": selected_total,
               "actual_selected_fraction": selected_total/weight_count,
               "actual_rounding_flip_ratio": flipped_total/weight_count,
               "selection_mask_checksum": checksum.hexdigest(),
               "threshold_time": threshold_seconds, "quantization_time": time.time()-started}
    return metrics, rows
