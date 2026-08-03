#!/usr/bin/env python3
"""Emit tests/vllm/models/minimax_h3_video_vae_goldens.inc.

Like the audio VAE, MiniMax-H3's VIDEO VAE is checkpoint REMOTE CODE
(`FL2VA/video_vae/*.py`, loaded under `trust_remote_code`), so it must be
REIMPLEMENTED rather than adapted. Its real 560-tensor manifest shows the DECODER
-- the half generation needs -- is a 36-block TRANSFORMER, so this generator runs
the checkpoint's OWN `TransformerBlock` at reduced dimensions as the oracle.

The remote code is NOT vendored here (MiniMax H3 Community License). Point
--h3-vae-source at a local copy of the checkpoint's `video_vae/` directory; only
the .py files are needed, not the 10 GB model.safetensors:

    python3 scripts/gen-minimax-h3-video-vae-goldens.py \
        --h3-vae-source ~/h3/FL2VA/video_vae \
        --out tests/vllm/models/minimax_h3_video_vae_goldens.inc

The bundle imports a handful of diffusers symbols (a logger, two mixin bases, two
no-op decorators); those are STUBBED here rather than pulling the whole diffusers
dependency in just to run an oracle.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_MASK64 = (1 << 64) - 1


def fnv1a64(name: str) -> int:
    h = 0xCBF29CE484222325
    for byte in name.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & _MASK64
    return h


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def h3_rand(name: str, count: int) -> np.ndarray:
    """Identical to the other H3 generators and the C++ H3Rand."""
    seed = fnv1a64(name)
    out = np.empty(count, dtype=np.float64)
    for i in range(count):
        u = splitmix64((seed + i) & _MASK64)
        out[i] = ((u >> 11) * (2.0**-53)) * 2.0 - 1.0
    return out


def _mod(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def install_diffusers_stub() -> None:
    if "diffusers" in sys.modules:
        return
    d = _mod("diffusers")
    logging.get_logger = logging.getLogger
    utils = _mod("diffusers.utils", logging=logging, get_logger=logging.getLogger)
    d.utils = utils
    utils.torch_utils = _mod("diffusers.utils.torch_utils", maybe_allow_in_graph=lambda cls: cls)

    class ConfigMixin:
        config_name = "config.json"

    _mod("diffusers.configuration_utils", ConfigMixin=ConfigMixin, register_to_config=lambda fn: fn)

    class ModelMixin(nn.Module):
        pass

    models = _mod("diffusers.models")
    models.modeling_utils = _mod("diffusers.models.modeling_utils", ModelMixin=ModelMixin)
    d.models = models


def load_bundle(src: Path, module: str):
    install_diffusers_stub()
    if "h3vv" not in sys.modules:
        pkg = types.ModuleType("h3vv")
        pkg.__path__ = [str(src)]
        sys.modules["h3vv"] = pkg
    name = "h3vv." + module
    spec = importlib.util.spec_from_file_location(name, src / f"{module}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# Reduced dimensions. The SHIPPED decoder is 36 blocks at 36 tensors each; the
# block is the repeated unit, so one block at small width exercises the same math.
HEADS = 2
DIM_HEAD = 8
DIM = HEADS * DIM_HEAD
SEQ = 6


def emit_f32(out, name: str, values) -> None:
    flat = np.asarray(values, dtype=np.float32).reshape(-1).tolist()
    out.write(f"inline constexpr float {name}[] = {{\n")
    for i in range(0, len(flat), 6):
        chunk = []
        for v in flat[i : i + 6]:
            text = f"{v:.9g}"
            if "." not in text and "e" not in text and "E" not in text:
                text += ".0"
            chunk.append(text + "f")
        out.write("    " + ", ".join(chunk) + ",\n")
    out.write("};\n\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h3-vae-source", required=True, type=Path,
                        help="the checkpoint's FL2VA/video_vae directory (.py files only)")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    src = args.h3_vae_source.expanduser().resolve()
    if not (src / "base_module.py").is_file():
        raise SystemExit(f"missing base_module.py in {src}")
    bm = load_bundle(src, "base_module")
    torch.set_grad_enabled(False)

    # The shipped decoder blocks: RMSNorm (affine), RMS qk-norm WITHOUT affine,
    # gated SiLU feed-forward, and two learned residual scales -- exactly the
    # parameter set the real manifest shows.
    block = bm.TransformerBlock(
        heads=HEADS, dim_head=DIM_HEAD, embed_dim=DIM,
        norm_type="rms_norm", qk_norm_type="rms_norm", qk_norm_affine=False,
        ffn_activation_fn="silu", ffn_use_gated=True, use_scale=True, bias=True,
    ).eval()

    state = block.state_dict()
    for name, tensor in state.items():
        scale, offset = (0.1, 0.0)
        if name.endswith("norm1.weight") or name.endswith("norm2.weight"):
            scale, offset = 0.1, 1.0
        elif name in ("scale1", "scale2"):
            scale, offset = 0.3, 0.0
        elif name.endswith(".bias"):
            scale = 0.05
        values = h3_rand("videovae." + name, tensor.numel()) * scale + offset
        state[name] = torch.from_numpy(values.astype(np.float32)).reshape(tensor.shape)
    block.load_state_dict(state, strict=True)

    x = torch.from_numpy(h3_rand("videovae.input", SEQ * DIM).astype(np.float32)).reshape(1, SEQ, DIM)
    y = block(x)
    inner = block.ff.w1.weight.shape[0] // 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        out.write(
            "// GENERATED by scripts/gen-minimax-h3-video-vae-goldens.py — DO NOT EDIT BY HAND.\n"
            "//\n"
            "// MiniMax-H3 VIDEO VAE decoder TransformerBlock goldens, produced by executing the\n"
            "// CHECKPOINT'S OWN remote-code module (FL2VA/video_vae/base_module.py) at reduced\n"
            "// dimensions. The shipped decoder is 36 of these blocks. The remote code is not\n"
            "// vendored here; see the generator's docstring. Weights come from the shared H3Rand\n"
            "// stream, so no weight byte is checked in.\n"
            "#pragma once\n\n#include <cstdint>\n\nnamespace vllm_test {\n\n"
        )
        out.write(f"inline constexpr int64_t kH3VideoVaeBlockHeads = {HEADS};\n")
        out.write(f"inline constexpr int64_t kH3VideoVaeBlockDimHead = {DIM_HEAD};\n")
        out.write(f"inline constexpr int64_t kH3VideoVaeBlockDim = {DIM};\n")
        out.write(f"inline constexpr int64_t kH3VideoVaeBlockSeq = {SEQ};\n")
        out.write(f"inline constexpr int64_t kH3VideoVaeBlockFfInner = {int(inner)};\n\n")
        emit_f32(out, "kH3VideoVaeBlockGolden", y.reshape(-1))
        out.write("}  // namespace vllm_test\n")
    print(f"wrote {args.out} (ff inner {int(inner)})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
