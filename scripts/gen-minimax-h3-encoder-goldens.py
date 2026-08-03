#!/usr/bin/env python3
"""Emit tests/vllm/models/minimax_h3_encoder_goldens.inc.

The MiniMax-H3 text/vision encoder (`H3-Encoder`) is a Qwen3-VL whose TEXT tower
H3 uses with three deltas (encoder.py:1-30):
  1. only the first MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER (50) decoder layers are
     kept, and the checkpoint consumes the hidden state AFTER layer 49;
  2. that state is UNNORMALIZED -- there is no final RMSNorm, unlike a standard
     Qwen3-VL text model;
  3. DeepStack visual features are added into the first
     len(deepstack_visual_indexes) layers' outputs at the visual token positions.
The result is the [seq, 5120] `prompt_embeds` the DiT consumes.

This runs the UPSTREAM `MiniMaxH3Qwen3VLTextModel` at reduced dimensions as the
oracle (vllm-omni is imported by file path with a one-symbol `vllm.logger` stub,
so neither vllm nor its dependencies are needed).

    python3 scripts/gen-minimax-h3-encoder-goldens.py \
        --vllm-omni ~/_git/vllm-omni \
        --out tests/vllm/models/minimax_h3_encoder_goldens.inc
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

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
    seed = fnv1a64(name)
    out = np.empty(count, dtype=np.float64)
    for i in range(count):
        u = splitmix64((seed + i) & _MASK64)
        out[i] = ((u >> 11) * (2.0**-53)) * 2.0 - 1.0
    return out


def load_encoder(root: Path):
    """Import vllm-omni's encoder module with a minimal vllm stub."""
    if "vllm" not in sys.modules:
        v = types.ModuleType("vllm")
        sys.modules["vllm"] = v
        lg = types.ModuleType("vllm.logger")
        lg.init_logger = logging.getLogger
        sys.modules["vllm.logger"] = lg
        v.logger = lg
    path = root / "vllm_omni" / "diffusion" / "models" / "minimax_h3" / "encoder.py"
    if not path.is_file():
        raise SystemExit(f"upstream encoder not found: {path}")
    spec = importlib.util.spec_from_file_location("h3_encoder", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


# Reduced Qwen3-VL text geometry. GQA (2 q-heads over 1 kv-head), interleaved
# M-RoPE sections, and a layer budget larger than the selected layer so the
# TRUNCATION rule is actually exercised.
VOCAB = 32
HIDDEN = 16
NUM_LAYERS = 4      # the config claims 4 ...
SELECTED_LAYER = 3  # ... but only 3 are kept, exactly like 50-of-N upstream
HEADS = 2
KV_HEADS = 1
HEAD_DIM = 8
INTERMEDIATE = 32
MROPE_SECTION = [2, 1, 1]
SEQ = 5
DEEPSTACK_LAYERS = 2


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


def emit_i64(out, name: str, values) -> None:
    flat = [int(v) for v in np.asarray(values).reshape(-1).tolist()]
    out.write(f"inline constexpr int64_t {name}[] = {{\n")
    for i in range(0, len(flat), 12):
        out.write("    " + ", ".join(str(v) for v in flat[i : i + 12]) + ",\n")
    out.write("};\n\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-omni", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    enc = load_encoder(args.vllm_omni.expanduser())
    torch.set_grad_enabled(False)

    group = SimpleNamespace(rank_in_group=0, world_size=1)  # TP=1
    config = SimpleNamespace(
        vocab_size=VOCAB, hidden_size=HIDDEN, num_hidden_layers=NUM_LAYERS,
        num_attention_heads=HEADS, num_key_value_heads=KV_HEADS, head_dim=HEAD_DIM,
        intermediate_size=INTERMEDIATE, rms_norm_eps=1e-6,
        rope_parameters={"rope_theta": 10000.0, "mrope_section": MROPE_SECTION},
        max_position_embeddings=128, attention_bias=False, attention_dropout=0.0,
    )
    model = enc.MiniMaxH3Qwen3VLTextModel(group, config, selected_layer=SELECTED_LAYER,
                                          dtype=torch.float32).eval()
    assert model.num_layers == SELECTED_LAYER, "layer truncation did not apply"

    state = model.state_dict()
    for name, tensor in state.items():
        scale, offset = (0.1, 0.0)
        if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
            scale, offset = 0.1, 1.0
        state[name] = torch.from_numpy(
            (h3_rand("encoder." + name, tensor.numel()) * scale + offset).astype(np.float32)
        ).reshape(tensor.shape)
    model.load_state_dict(state, strict=True)

    inputs_embeds = torch.from_numpy(
        h3_rand("encoder.inputs_embeds", SEQ * HIDDEN).astype(np.float32)
    ).reshape(1, SEQ, HIDDEN)
    # M-RoPE positions are (3, batch, seq): temporal / height / width.
    positions = torch.arange(SEQ).view(1, 1, SEQ).expand(3, 1, SEQ).contiguous()

    plain = model(inputs_embeds, positions)

    # DeepStack: visual features added at the visual token positions, for the
    # FIRST len(deepstack_visual_embeds) layers only.
    visual_mask = torch.zeros(1, SEQ, dtype=torch.bool)
    visual_mask[0, 1] = True
    visual_mask[0, 3] = True
    num_visual = int(visual_mask.sum())
    deepstack = [
        torch.from_numpy(
            (h3_rand(f"encoder.deepstack.{i}", num_visual * HIDDEN) * 0.1).astype(np.float32)
        ).reshape(num_visual, HIDDEN)
        for i in range(DEEPSTACK_LAYERS)
    ]
    with_deepstack = model(inputs_embeds, positions, visual_pos_masks=visual_mask,
                           deepstack_visual_embeds=deepstack)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        out.write(
            "// GENERATED by scripts/gen-minimax-h3-encoder-goldens.py — DO NOT EDIT BY HAND.\n"
            "//\n"
            "// MiniMax-H3 ENCODER text-tower goldens from the UPSTREAM vLLM-Omni\n"
            "// MiniMaxH3Qwen3VLTextModel at reduced dimensions. Weights come from the shared\n"
            "// H3Rand stream, so no weight byte is checked in. The three H3 deltas vs a stock\n"
            "// Qwen3-VL text model — layer truncation, the UNNORMALIZED output, and DeepStack\n"
            "// injection into the first N layers — are all exercised here.\n"
            "#pragma once\n\n#include <cstdint>\n\nnamespace vllm_test {\n\n"
        )
        for key, value in (
            ("Vocab", VOCAB), ("Hidden", HIDDEN), ("ConfigLayers", NUM_LAYERS),
            ("SelectedLayer", SELECTED_LAYER), ("Heads", HEADS), ("KvHeads", KV_HEADS),
            ("HeadDim", HEAD_DIM), ("Intermediate", INTERMEDIATE), ("Seq", SEQ),
            ("DeepstackLayers", DEEPSTACK_LAYERS), ("NumVisual", num_visual),
        ):
            out.write(f"inline constexpr int64_t kH3Enc{key} = {value};\n")
        out.write("\n")
        emit_i64(out, "kH3EncMropeSection", MROPE_SECTION)
        emit_i64(out, "kH3EncVisualMask", visual_mask.to(torch.int64))
        emit_f32(out, "kH3EncGolden", plain.reshape(-1))
        emit_f32(out, "kH3EncDeepstackGolden", with_deepstack.reshape(-1))
        out.write("}  // namespace vllm_test\n")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
