#!/usr/bin/env python3
"""Check the installed GPU/FlashAttention ABI, not model/RL compatibility."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    import torch
    from flash_attn import flash_attn_func

    assert torch.cuda.is_available(), "CUDA unavailable in the selected environment"
    assert torch.cuda.device_count() == 1, "This recipe expects one visible GPU"
    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    assert torch.isfinite(a @ a.T).all().item()
    q = torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        result = flash_attn_func(q, q, q, causal=True)
    torch.cuda.synchronize()
    assert torch.isfinite(result).all().item()
    report = {
        "status": "gpu_kernel_preflight_passed", "time": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "machine": platform.machine(),
        "gpu": torch.cuda.get_device_name(0), "gpu_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch_cuda_runtime": torch.version.cuda,
        "packages": {name: version(name) for name in ("torch", "vllm", "verl", "flash-attn", "agentlightning", "autogen-agentchat", "weave")},
        "model_loaded": False, "rl_update_verified": False, "checkpoint_reload_verified": False,
    }
    assert report["packages"]["vllm"].split("+", 1)[0] == "0.22.0", "Unexpected vLLM version"
    assert report["packages"]["verl"].split("+", 1)[0] == "0.8.0", "Unexpected verl version"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
