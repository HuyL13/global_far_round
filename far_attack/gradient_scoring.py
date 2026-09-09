from __future__ import annotations

import gc


def _behavior_scalar(logits, kind: str):
    import torch.nn.functional as functional
    if kind != "top1_logprob":
        raise ValueError("only top1_logprob preserves the current Method 3")
    log_probs = functional.log_softmax(logits.float(), dim=-1)
    top1 = logits.argmax(dim=-1, keepdim=True).detach()
    return log_probs.gather(-1, top1).squeeze(-1).mean()


def compute_gradients(model, layers: dict, batches: list[dict], device: str,
                      behavior: str = "top1_logprob", progress=print):
    import torch
    import torch.nn.functional as functional
    modules = list(layers.values())
    target_ids = {id(module.weight) for module in modules}
    parameters = list(model.parameters())
    old_flags = [parameter.requires_grad for parameter in parameters]
    for parameter in parameters:
        parameter.requires_grad_(id(parameter) in target_ids)
    behavior_sum = {name: torch.zeros_like(module.weight, dtype=torch.bfloat16, device="cpu")
                    for name, module in layers.items()}
    utility_sum = {name: torch.zeros_like(module.weight, dtype=torch.bfloat16, device="cpu")
                   for name, module in layers.items()}
    count = max(len(batches), 1)
    for index, batch in enumerate(batches, 1):
        progress(f"[FAR] gradient batch {index}/{len(batches)}")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        model.zero_grad(set_to_none=True)
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        scalar = _behavior_scalar(output.logits, behavior)
        scalar.backward()
        for name, module in layers.items():
            if module.weight.grad is not None:
                behavior_sum[name] += module.weight.grad.detach().to("cpu", torch.bfloat16)
        del output, scalar
        model.zero_grad(set_to_none=True)
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        loss = functional.cross_entropy(output.logits[:, :-1].reshape(-1, output.logits.shape[-1]).float(),
                                        input_ids[:, 1:].reshape(-1))
        loss.backward()
        for name, module in layers.items():
            if module.weight.grad is not None:
                utility_sum[name] += module.weight.grad.detach().to("cpu", torch.bfloat16)
        del output, loss, input_ids, attention_mask
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    model.zero_grad(set_to_none=True)
    for parameter, flag in zip(parameters, old_flags):
        parameter.requires_grad_(flag)
    return {name: (behavior_sum[name] / count, utility_sum[name] / count) for name in layers}
