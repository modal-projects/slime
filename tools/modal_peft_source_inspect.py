"""Inspect installed Megatron PEFT source from inside a Modal image."""

import inspect

import modal

IMAGE_TAG = "slimerl/slime:nightly-dev-20260126a"

app = modal.App("peft-source-inspect")
image = modal.Image.from_registry(IMAGE_TAG).entrypoint([])


def _print_source(obj, label: str, max_chars: int = 12000) -> None:
    try:
        src = inspect.getsource(obj)
    except Exception as exc:  # pragma: no cover - debugging helper
        print(f"[source unavailable] {label}: {exc}")
        return
    print(f"\n===== {label} =====")
    print(src[:max_chars])
    if len(src) > max_chars:
        print(f"... [truncated, total chars={len(src)}]")


@app.function(image=image, timeout=10 * 60)
def inspect_peft_sources() -> None:
    import megatron.bridge.peft.dora_layers as dora_layers
    import megatron.bridge.peft.lora as lora

    print(f"dora_layers file: {inspect.getsourcefile(dora_layers)}")
    print(f"lora file: {inspect.getsourcefile(lora)}")

    dora_symbols = sorted([x for x in dir(dora_layers) if "DoRA" in x or "dora" in x.lower()])
    lora_symbols = sorted([x for x in dir(lora) if "LoRA" in x or "Adapter" in x])
    print(f"\nDoRA symbols: {dora_symbols}")
    print(f"LoRA symbols: {lora_symbols}")

    for name in ["ParallelLinearDoRAAdapter", "DoRALinear"]:
        if hasattr(dora_layers, name):
            _print_source(getattr(dora_layers, name), f"dora_layers.{name}")

    for name in ["ParallelLinearAdapter", "LinearAdapter", "LoRALinear"]:
        if hasattr(lora, name):
            _print_source(getattr(lora, name), f"lora.{name}")


@app.local_entrypoint()
def main():
    inspect_peft_sources.remote()
