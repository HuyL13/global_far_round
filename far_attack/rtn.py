from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass
class AffineState:
    pre_round: object
    integer_weights: object
    scale: object
    zero_point: object
    max_int: int
    in_features: int
    padded_in_features: int
    original_dtype: object

    def dequantize_truncated(self):
        value = (self.integer_weights - self.zero_point) * self.scale
        return value[:, : self.in_features].to(self.original_dtype)


def affine_state(weight, bits: int = 4, group_size: int = 128) -> AffineState:
    import torch
    rows, columns = weight.shape
    padded = ((columns + group_size - 1) // group_size) * group_size
    work = torch.zeros((rows, padded), dtype=torch.float32, device=weight.device)
    work[:, :columns] = weight.float()
    grouped = work.reshape(rows, -1, group_size)
    lo = grouped.amin(dim=2, keepdim=True)
    hi = grouped.amax(dim=2, keepdim=True)
    max_int = 2**bits - 1
    scale = ((hi - lo) / max_int).clamp_min(1e-8)
    zero = torch.round(-lo / scale).clamp(0, max_int)
    pre = grouped / scale + zero
    integer = torch.round(pre).clamp(0, max_int)
    return AffineState(pre.reshape(rows, padded), integer.reshape(rows, padded),
                       scale.expand_as(grouped).reshape(rows, padded),
                       zero.expand_as(grouped).reshape(rows, padded), max_int,
                       columns, padded, weight.dtype)


def apply_affine_rtn4(model, group_size: int = 128, progress=print) -> dict:
    import torch
    from .model import transformer_linears
    layers = transformer_linears(model)
    with torch.no_grad():
        for index, (name, module) in enumerate(layers.items(), 1):
            progress(f"[RTN4] layer {index}/{len(layers)}: {name}")
            module.weight.copy_(affine_state(module.weight, 4, group_size).dequantize_truncated())
    return {"backend": "tier0_affine", "bits": 4, "group_size": group_size, "selected_parameters": len(layers)}


def apply_phase1_rtn3(model, group_size: int, module_pattern: str, progress=print) -> dict:
    import torch
    from vendor.phase1_eval.quantization import rtn
    pattern = re.compile(module_pattern)
    selected = [(n, p) for n, p in model.named_parameters()
                if p.is_floating_point() and p.ndim == 2 and pattern.search(n)]
    with torch.no_grad():
        for index, (name, parameter) in enumerate(selected, 1):
            progress(f"[RTN3] layer {index}/{len(selected)}: {name}")
            grid = rtn(parameter.detach().float().cpu().numpy(), 3, group_size, True)
            parameter.copy_(torch.as_tensor(grid.values, dtype=parameter.dtype, device=parameter.device))
    if not selected:
        raise ValueError("module_pattern did not select any parameters")
    return {"backend": "phase1_symmetric_numpy", "bits": 3, "group_size": group_size,
            "qmin": -3, "qmax": 3, "selected_parameters": len(selected)}
