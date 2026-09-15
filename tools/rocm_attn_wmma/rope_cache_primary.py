"""Export device RoPE cache samples from the pinned primary classes."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import vllm
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.linear_scaling_rope import (
    LinearScalingRotaryEmbedding,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if "e126687a9" not in vllm.__version__:
        raise RuntimeError(f"wrong primary pin: {vllm.__version__}")
    args.output.mkdir(exist_ok=False)
    torch.set_default_device("cuda")
    cases = []
    for name, base, factor, positions in [
        ("local", 10000.0, 1.0, [0, 42, 53, 64, 86, 106, 518, 1207, 4095, 131071]),
        ("global", 1000000.0, 8.0, [0, 82, 107, 212, 518, 1207, 4095, 131071, 1048575]),
    ]:
        if factor == 1.0:
            rope = RotaryEmbedding(256, 256, 131072, base, True, torch.bfloat16)
        else:
            rope = LinearScalingRotaryEmbedding(
                256, 256, 131072, base, True, factor, torch.bfloat16
            )
        sample = rope.cos_sin_cache[positions].contiguous()
        data = sample.view(torch.uint8).cpu().numpy().tobytes()
        file = f"rope-cache-{name}.bin"
        (args.output / file).write_bytes(data)
        cases.append(
            dict(
                name=name,
                base=base,
                factor=factor,
                positions=positions,
                file=file,
                sha256=hashlib.sha256(data).hexdigest(),
            )
        )
    report = dict(
        pin="e126687a9a828d513c01a07cd69f025f27d63280",
        source="vllm/model_executor/layers/rotary_embedding/base.py:89-112 and "
        "linear_scaling_rope.py:107-127; executing Gemma3 layer buffers on gfx1100",
        cases=cases,
    )
    (args.output / "rope-cache.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    with set_current_vllm_config(VllmConfig()):
        main()
