from __future__ import annotations

import gc


def load_model(model_id: str, device: str, dtype: str):
    import torch
    from transformers import AutoModelForCausalLM
    kwargs = dict(dtype=getattr(torch, dtype), low_cpu_mem_usage=True, trust_remote_code=False)
    if device.startswith("cuda"):
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not device.startswith("cuda"):
        model.to(device)
    return model.eval()


def load_tokenizer(tokenizer_id: str):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def transformer_linears(model) -> dict:
    import torch.nn as nn
    result = {}
    for index, block in enumerate(model.model.layers):
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear):
                result[f"model.layers.{index}.{name}"] = module
    return result


def release_model(model) -> None:
    import torch
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
