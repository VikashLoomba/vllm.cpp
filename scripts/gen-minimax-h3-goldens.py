#!/usr/bin/env python3
"""Emit tests/vllm/models/minimax_h3_goldens.inc — the MiniMax-H3 parity oracle.

MiniMax-H3 (`MiniMaxAI/MiniMax-H3`) is a 33.1B omni-modal video+audio diffusion
transformer whose full checkpoint is ~354 GB and whose validated serving config is
4x NVIDIA B300. Neither fits this project's hardware, so the H3 port cannot be
gated end-to-end (see .agents/specs/minimax-h3.md section 0). What CAN be gated
exactly, on any CPU, is the MATH: this generator runs the upstream vLLM-Omni H3
modules (and a TP=1 restatement of its DiT) at REDUCED dimensions with
deterministic pseudo-random weights, and emits the resulting tensors as C++
goldens. The C++ suite regenerates the identical inputs from the identical PRNG
and must reproduce these outputs.

Upstream sources (vllm-project/vllm-omni, vllm_omni/diffusion/models/minimax_h3/):
  packed_sequence.py                       -> section 1 + 2 goldens
  packed_tokens.py                         -> section 3 goldens
  scheduling_minimax_h3_euler_ancestral.py -> section 4 goldens
  minimax_h3_transformer.py                -> section 5 goldens (TP=1 restatement)

The first three are imported and executed VERBATIM (loaded by file path, which
bypasses the vllm_omni package __init__ and so needs neither vllm nor aenum). Only
the DiT is restated here, because upstream's module imports the whole vLLM
tensor-parallel linear stack; at tensor_parallel_size=1 every {Column,Row,QKV,
MergedColumn}ParallelLinear degenerates to a plain nn.Linear, so the restatement
below is line-for-line faithful and each class cites the upstream lines it mirrors.

Usage:
    python3 scripts/gen-minimax-h3-goldens.py \
        --vllm-omni ~/_git/vllm-omni \
        --out tests/vllm/models/minimax_h3_goldens.inc

Needs torch + numpy (CPU only).
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Deterministic weight/input stream, mirrored bit-for-bit by the C++ suite
# (tests/vllm/models/test_minimax_h3_dit.cpp :: H3Rand). A per-tensor FNV-1a seed
# plus a splitmix64 counter makes every tensor independent of fill ORDER, so the
# two sides cannot silently drift by reordering their parameter construction.
# ---------------------------------------------------------------------------

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
    """`count` values uniform in [-1, 1), reproducible from `name` alone."""
    seed = fnv1a64(name)
    out = np.empty(count, dtype=np.float64)
    for i in range(count):
        u = splitmix64((seed + i) & _MASK64)
        # Top 53 bits -> [0, 1), then map to [-1, 1). Both sides use the same
        # 53-bit mantissa construction so the doubles are bit-identical.
        unit = (u >> 11) * (2.0**-53)
        out[i] = unit * 2.0 - 1.0
    return out


def make_param(name: str, shape, scale: float, offset: float = 0.0) -> torch.Tensor:
    count = int(np.prod(shape)) if shape else 1
    values = h3_rand(name, count) * scale + offset
    return torch.from_numpy(values.astype(np.float32)).reshape(shape)


# ---------------------------------------------------------------------------
# Reduced-dimension arch. Every ratio that the port's code paths branch on is
# preserved: 3 modalities x 6 AdaLN vectors, patch (1,2,2), rot_dim < head_dim,
# a token refiner shallower than the block stack, and distinct video/audio latent
# widths. Only the magnitudes shrink.
#
# rot_dim = 6 * rope_inv_freq_len must be <= attention_head_dim (upstream:
# 6*16=96 <= 128; here 6*2=12 <= 16).
# ---------------------------------------------------------------------------

ARCH = dict(
    num_layers=2,
    token_refiner_num_layers=1,
    hidden_size=64,
    num_attention_heads=4,
    attention_head_dim=16,
    ffn_hidden_size=128,
    latents_dim=8,
    audio_latents_dim=6,
    patch_size=(1, 2, 2),
    text_dim=24,
    timestep_input_dim=16,
    time_embed_hidden_size=64,
    time_embed_dim=32,
    rope_inv_freq_len=2,
    norm_eps=1e-5,
    qk_norm_eps=1e-5,
    final_norm_eps=1e-5,
)
ARCH["adaln_out_features"] = 18 * ARCH["hidden_size"]
ARCH["final_adaln_out_features"] = 2 * ARCH["hidden_size"]

MODALITY_NUM = 3  # minimax_h3_transformer.py:106 MINIMAX_H3_ADALN_MODALITY_NUM


# ---------------------------------------------------------------------------
# TP=1 restatement of minimax_h3_transformer.py. Computed in float32 throughout:
# the C++ gate runs f32 so the comparison isolates the ALGORITHM from bf16
# rounding order (the production path keeps upstream's bf16 cast points, which are
# marked in the port and exercised separately).
# ---------------------------------------------------------------------------


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """nn.RMSNorm semantics: fp32 variance reduction (transformer.py:171-175)."""
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    return (x.to(torch.float32) * torch.rsqrt(variance + eps) * weight).to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """minimax_h3_transformer.py:178-180."""
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_freqs(inv_freq: torch.Tensor, img_position_ids: torch.Tensor) -> torch.Tensor:
    """MiniMaxH3Rope.forward (minimax_h3_transformer.py:222-230).

    img_position_ids [1,S,3] (t,h,w) -> freqs [S, 6*inv_freq_len].
    """
    pos = img_position_ids[0].to(torch.float32)
    per_axis = pos.unsqueeze(-1) * inv_freq.view(1, 1, -1)
    t_f, h_f, w_f = per_axis.unbind(dim=1)
    half = torch.cat((t_f, h_f, w_f), dim=-1)
    return torch.cat((half, half), dim=-1)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """_apply_rope (minimax_h3_transformer.py:233-244). x [T,heads,dim]."""
    rot_dim = freqs.shape[-1]
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    cos = torch.cos(freqs).to(x.dtype).unsqueeze(1)
    sin = torch.sin(freqs).to(x.dtype).unsqueeze(1)
    x_rot = (x_rot * cos) + (rotate_half(x_rot) * sin)
    return torch.cat((x_rot, x_pass), dim=-1)


def varlen_non_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """_sdpa_varlen_attention (minimax_h3_transformer.py:288-317).

    Bidirectional attention within each packed document. This is exactly the
    contract of our shared vt::DFlashBlockAttention(causal=false).
    """
    out = torch.empty_like(q)
    bounds = cu_seqlens.tolist()
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if stop == start:
            continue
        seg_q = q[start:stop].transpose(0, 1).unsqueeze(0)
        seg_k = k[start:stop].transpose(0, 1).unsqueeze(0)
        seg_v = v[start:stop].transpose(0, 1).unsqueeze(0)
        seg = torch.nn.functional.scaled_dot_product_attention(seg_q, seg_k, seg_v, scale=scale)
        out[start:stop] = seg.squeeze(0).transpose(0, 1)
    return out


def modulate_scale_shift(x, shift, scale, indices):
    """_modulate_scale_shift (minimax_h3_transformer.py:183-192)."""
    return x * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)


def modulate_gate(x, gate, other, indices):
    """_modulate_gate (minimax_h3_transformer.py:195-204)."""
    return x + gate.index_select(0, indices) * other


class RefLinear:
    """A TP=1 {Column,Row,QKV,MergedColumn}ParallelLinear: y = x @ W^T + b."""

    def __init__(self, name: str, out_features: int, in_features: int, bias: bool):
        self.weight = make_param(f"{name}.weight", (out_features, in_features), 0.05)
        self.bias = make_param(f"{name}.bias", (out_features,), 0.02) if bias else None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


class RefAttention:
    """MiniMaxH3Attention (minimax_h3_transformer.py:320-467).

    qkv -> per-head q/k RMSNorm -> RoPE -> varlen non-causal attention -> out.
    """

    def __init__(self, name: str, arch: dict):
        self.heads = arch["num_attention_heads"]
        self.head_dim = arch["attention_head_dim"]
        inner = self.heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.qkv_proj = RefLinear(f"{name}.qkv_proj", 3 * inner, arch["hidden_size"], bias=False)
        self.q_norm = make_param(f"{name}.q_norm.weight", (self.head_dim,), 0.1, 1.0)
        self.k_norm = make_param(f"{name}.k_norm.weight", (self.head_dim,), 0.1, 1.0)
        self.out_proj = RefLinear(f"{name}.out_proj", arch["hidden_size"], inner, bias=False)
        self.qk_eps = arch["qk_norm_eps"]

    def __call__(self, x, freqs, cu_seqlens):
        total = x.shape[0]
        qkv = self.qkv_proj(x)
        size = self.heads * self.head_dim
        q, k, v = qkv.split([size, size, size], dim=-1)
        q = q.view(total, self.heads, self.head_dim)
        k = k.view(total, self.heads, self.head_dim)
        v = v.view(total, self.heads, self.head_dim)
        q = rms_norm(q, self.q_norm, self.qk_eps)
        k = rms_norm(k, self.k_norm, self.qk_eps)
        if freqs is not None:
            q = apply_rope(q, freqs)
            k = apply_rope(k, freqs)
        out = varlen_non_causal_attention(q, k, v, cu_seqlens, self.scale)
        return self.out_proj(out.reshape(total, size))


class RefMLP:
    """MiniMaxH3MLP (minimax_h3_transformer.py:470-517): silu(gate) * up."""

    def __init__(self, name: str, arch: dict):
        ffn = arch["ffn_hidden_size"]
        self.fc1 = RefLinear(f"{name}.fc1", 2 * ffn, arch["hidden_size"], bias=False)
        self.fc2 = RefLinear(f"{name}.fc2", arch["hidden_size"], ffn, bias=False)

    def __call__(self, x):
        hidden = self.fc1(x)
        gate, up = hidden.chunk(2, dim=-1)
        return self.fc2(torch.nn.functional.silu(gate) * up)


class RefAdalnProj:
    """MiniMaxH3AdalnProj (minimax_h3_transformer.py:520-561).

    [M, t_dim] -> silu -> linear -> view(M*modality_num, expand*H) -> chunk.
    """

    def __init__(self, name: str, arch: dict, out_features: int, expand_ratio: int, modality_num: int):
        assert out_features == expand_ratio * arch["hidden_size"] * modality_num
        self.linear = RefLinear(f"{name}.linear", out_features, arch["time_embed_dim"], bias=True)
        self.expand_ratio = expand_ratio
        self.modality_num = modality_num
        self.hidden_size = arch["hidden_size"]

    def __call__(self, t_emb):
        x = torch.nn.functional.silu(t_emb)
        x = self.linear(x)
        m = x.shape[0]
        x = x.view(m * self.modality_num, self.expand_ratio * self.hidden_size)
        return tuple(x.chunk(self.expand_ratio, dim=-1))


class RefDiT:
    """MiniMaxH3DiTModel (minimax_h3_transformer.py:765-1102) at TP=1, fp32."""

    def __init__(self, arch: dict):
        self.arch = arch
        h = arch["hidden_size"]
        pt, ph, pw = arch["patch_size"]
        self.video_patch_dim = arch["latents_dim"] * pt * ph * pw
        self.video_patch_proj = RefLinear("video_patch_proj", h, self.video_patch_dim, bias=True)
        self.audio_patch_proj = RefLinear("audio_patch_proj", h, arch["audio_latents_dim"], bias=True)
        self.condition_proj = RefLinear("condition_proj", h, arch["text_dim"], bias=True)
        self.time_proj_in = RefLinear(
            "time_embedder.proj_in", arch["time_embed_hidden_size"], arch["timestep_input_dim"], bias=True
        )
        self.time_proj_out = RefLinear(
            "time_embedder.proj_out", arch["time_embed_dim"], arch["time_embed_hidden_size"], bias=True
        )
        # inv_freq = base^-(arange(0, 2L, 2)/(2L)) (minimax_h3_transformer.py:207-212)
        length = arch["rope_inv_freq_len"]
        self.inv_freq = torch.tensor(
            [10000.0 ** (-(2.0 * i) / (2.0 * length)) for i in range(length)], dtype=torch.float32
        )
        self.refiner = []
        for i in range(arch["token_refiner_num_layers"]):
            self.refiner.append(
                dict(
                    norm1=make_param(f"token_refiner.blocks.{i}.norm1.weight", (h,), 0.1, 1.0),
                    norm2=make_param(f"token_refiner.blocks.{i}.norm2.weight", (h,), 0.1, 1.0),
                    attn=RefAttention(f"token_refiner.blocks.{i}.attn", arch),
                    mlp=RefMLP(f"token_refiner.blocks.{i}.mlp", arch),
                )
            )
        self.refiner_final_norm = make_param("token_refiner.final_norm.weight", (h,), 0.1, 1.0)
        self.blocks = []
        for i in range(arch["num_layers"]):
            self.blocks.append(
                dict(
                    norm1=make_param(f"blocks.{i}.norm1.weight", (h,), 0.1, 1.0),
                    norm2=make_param(f"blocks.{i}.norm2.weight", (h,), 0.1, 1.0),
                    attn=RefAttention(f"blocks.{i}.attn", arch),
                    mlp=RefMLP(f"blocks.{i}.mlp", arch),
                    adaln=RefAdalnProj(
                        f"blocks.{i}.adaln_proj", arch, arch["adaln_out_features"], 6, MODALITY_NUM
                    ),
                )
            )
        self.final_norm = make_param("final_layer.norm.weight", (h,), 0.1, 1.0)
        self.final_adaln = RefAdalnProj(
            "final_layer.adaln_proj", arch, arch["final_adaln_out_features"], 2, 1
        )
        self.video_out = RefLinear("final_layer.video_out", self.video_patch_dim, h, bias=True)
        self.audio_out = RefLinear("final_layer.audio_out", arch["audio_latents_dim"], h, bias=True)

    def time_embed(self, t: torch.Tensor) -> torch.Tensor:
        """MiniMaxH3TimeEmbedder.forward (minimax_h3_transformer.py:272-285).

        Sinusoidal embedding stays fp32 and concatenates COSINE before SINE.
        """
        half = self.arch["timestep_input_dim"] // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        args = t.to(torch.float32)[:, None] * freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.time_proj_out(torch.nn.functional.silu(self.time_proj_in(t_freq)))

    def __call__(
        self,
        *,
        x,
        audio_x,
        img_position_ids,
        unique_timesteps,
        inverse_indices,
        update_mask,
        token_tags,
        prompt_embeds,
        img_pos,
        audio_pos,
        text_pos,
        infer_out_pos,
        cu_seqlens,
        refiner_cu_seqlens,
    ):
        seq_len = int(x.shape[1])
        freqs = rope_freqs(self.inv_freq, img_position_ids)

        # _embed (minimax_h3_transformer.py:944-984)
        video_embed = self.video_patch_proj(x.view(-1, x.shape[-1]).index_select(0, img_pos))
        audio_embed = self.audio_patch_proj(audio_x.view(-1, audio_x.shape[-1]).index_select(0, audio_pos))
        text_embed = self.condition_proj(prompt_embeds)
        for block in self.refiner:
            text_embed = text_embed + block["attn"](
                rms_norm(text_embed, block["norm1"], self.arch["norm_eps"]), None, refiner_cu_seqlens
            )
            text_embed = text_embed + block["mlp"](
                rms_norm(text_embed, block["norm2"], self.arch["norm_eps"])
            )
        text_embed = rms_norm(text_embed, self.refiner_final_norm, self.arch["final_norm_eps"])

        embeddings = torch.zeros((seq_len, self.arch["hidden_size"]), dtype=torch.float32)
        embeddings.index_add_(0, text_pos, text_embed[: text_pos.shape[0]])
        embeddings.index_add_(0, img_pos, video_embed[: img_pos.shape[0]])
        embeddings.index_add_(0, audio_pos, audio_embed[: audio_pos.shape[0]])

        t_emb = self.time_embed(unique_timesteps)
        combined = inverse_indices * MODALITY_NUM + token_tags.clamp(min=0)

        hidden = embeddings
        for block in self.blocks:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block["adaln"](t_emb)
            residual = hidden
            h = rms_norm(hidden, block["norm1"], self.arch["norm_eps"])
            h = modulate_scale_shift(h, shift_msa, scale_msa, combined)
            h = block["attn"](h, freqs, cu_seqlens)
            hidden = modulate_gate(residual, gate_msa, h, combined)

            residual = hidden
            h = rms_norm(hidden, block["norm2"], self.arch["norm_eps"])
            h = modulate_scale_shift(h, shift_mlp, scale_mlp, combined)
            h = block["mlp"](h)
            hidden = modulate_gate(residual, gate_mlp, h, combined)

        # MiniMaxH3FinalLayer.forward (minimax_h3_transformer.py:724-743)
        shift, scale = self.final_adaln(t_emb)
        h = rms_norm(hidden, self.final_norm, self.arch["final_norm_eps"])
        h = modulate_scale_shift(h, shift, scale, inverse_indices)
        video = self.video_out(h)
        audio = self.audio_out(h)

        video = video.index_select(0, infer_out_pos)
        audio = audio.index_select(0, audio_pos)
        video = video * update_mask.to(torch.float32).unsqueeze(-1)
        return video, audio


# ---------------------------------------------------------------------------
# Upstream module loading (by path, bypassing the vllm_omni package __init__).
# ---------------------------------------------------------------------------


def load_upstream(root: Path, module: str):
    path = root / "vllm_omni" / "diffusion" / "models" / "minimax_h3" / f"{module}.py"
    if not path.is_file():
        raise SystemExit(f"upstream module not found: {path}")
    spec = importlib.util.spec_from_file_location(f"_h3_{module}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Emission helpers
# ---------------------------------------------------------------------------


def emit_header(out, argv: str) -> None:
    out.write(
        "// GENERATED by scripts/gen-minimax-h3-goldens.py — DO NOT EDIT BY HAND.\n"
        "//\n"
        "// MiniMax-H3 parity goldens produced by executing the upstream vLLM-Omni\n"
        "// modules (vllm-project/vllm-omni, vllm_omni/diffusion/models/minimax_h3/)\n"
        "// at reduced dimensions with the deterministic H3Rand stream. Regenerate with:\n"
        f"//   {argv}\n"
        "//\n"
        "// See .agents/specs/minimax-h3.md section 4 for why this is the gate: the real\n"
        "// checkpoint is ~354 GB over 4x B300 and cannot run on this project's hardware,\n"
        "// so the MATH is gated exactly here and the WEIGHTS are gated structurally.\n"
        "#pragma once\n\n"
        "#include <cstdint>\n\n"
        "namespace vllm_test {\n\n"
    )


def emit_i64(out, name: str, values) -> None:
    flat = [int(v) for v in np.asarray(values).reshape(-1).tolist()]
    out.write(f"inline constexpr int64_t {name}[] = {{\n")
    for i in range(0, len(flat), 12):
        out.write("    " + ", ".join(str(v) for v in flat[i : i + 12]) + ",\n")
    out.write("};\n\n")


def emit_f64(out, name: str, values) -> None:
    flat = np.asarray(values, dtype=np.float64).reshape(-1).tolist()
    out.write(f"inline constexpr double {name}[] = {{\n")
    for i in range(0, len(flat), 6):
        chunk = ", ".join(_cxx_float(v, 17) for v in flat[i : i + 6])
        out.write("    " + chunk + ",\n")
    out.write("};\n\n")


def _cxx_float(value: float, digits: int) -> str:
    """Format as a valid C++ floating literal (`0` alone is an integer literal)."""
    if not math.isfinite(value):
        raise ValueError(f"refusing to emit non-finite golden value: {value}")
    text = f"{value:.{digits}g}"
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return text


def emit_f32(out, name: str, values) -> None:
    flat = np.asarray(values, dtype=np.float32).reshape(-1).tolist()
    out.write(f"inline constexpr float {name}[] = {{\n")
    for i in range(0, len(flat), 6):
        chunk = ", ".join(_cxx_float(v, 9) + "f" for v in flat[i : i + 6])
        out.write("    " + chunk + ",\n")
    out.write("};\n\n")


def emit_scalar(out, name: str, value: int) -> None:
    out.write(f"inline constexpr int64_t {name} = {int(value)};\n")


# ---------------------------------------------------------------------------
# Golden sections
# ---------------------------------------------------------------------------

# fl2va layout case: one leading keyframe, small latent grid.
FL2VA = dict(text_len=7, latent_t=3, latent_h=4, latent_w=6, audio_t=5, audio_channel=2)
# ref2va layout case: an image reference block plus a video+audio reference block.
REF2VA = dict(text_len=5, latent_t=2, latent_h=4, latent_w=4, audio_t=3, audio_channel=2)


def emit_packed_sequence(out, packed_sequence) -> None:
    out.write("// --- section 1: minimax_h3_packed_sequence (fl2va, one leading keyframe) ---\n")
    result = packed_sequence.minimax_h3_packed_sequence(
        text_len=FL2VA["text_len"],
        latent_t=FL2VA["latent_t"],
        latent_h=FL2VA["latent_h"],
        latent_w=FL2VA["latent_w"],
        audio_t=FL2VA["audio_t"],
        audio_channel=FL2VA["audio_channel"],
        include_keyframe_cond=True,
        keyframe_frame_indices=(0,),
        frame_count=9,
    )
    for key, value in FL2VA.items():
        emit_scalar(out, f"kH3Fl2va_{key}", value)
    emit_scalar(out, "kH3Fl2va_frame_count", 9)
    emit_scalar(out, "kH3Fl2va_seq_len", int(result["seq_len"]))
    out.write("\n")
    emit_i64(out, "kH3Fl2vaInputIds", result["input_ids"])
    emit_i64(out, "kH3Fl2vaTokenTags", result["token_tags"])
    emit_i64(out, "kH3Fl2vaImgPos", result["img_pos"])
    emit_i64(out, "kH3Fl2vaAudioPos", result["audio_pos"])
    emit_i64(out, "kH3Fl2vaTextPos", result["text_pos"])
    emit_i64(out, "kH3Fl2vaUpdateMask", result["update_mask"].to(torch.int64))
    emit_i64(out, "kH3Fl2vaCuSeqlens", result["cu_seqlens"].to(torch.int64))
    emit_i64(out, "kH3Fl2vaDocumentId", result["document_id"].to(torch.int64))
    # fp64 grid: the position ids are the one place where a last-ulp drift would
    # silently change RoPE, so they are gated as exact doubles.
    emit_f64(out, "kH3Fl2vaImgPositionIds", result["img_position_ids"])

    out.write("// --- section 2: minimax_h3_packed_sequence_ref2va_blocks ---\n")
    ref_blocks = [
        {"kind": "image", "latent_h": 4, "latent_w": 4},
        {"kind": "video_audio", "ref_audio_t": 2, "latent_t": 2, "latent_h": 4, "latent_w": 4},
    ]
    ref = packed_sequence.minimax_h3_packed_sequence_ref2va_blocks(
        text_len=REF2VA["text_len"],
        latent_t=REF2VA["latent_t"],
        latent_h=REF2VA["latent_h"],
        latent_w=REF2VA["latent_w"],
        audio_t=REF2VA["audio_t"],
        ref_blocks=ref_blocks,
        audio_channel=REF2VA["audio_channel"],
    )
    for key, value in REF2VA.items():
        emit_scalar(out, f"kH3Ref2va_{key}", value)
    emit_scalar(out, "kH3Ref2va_seq_len", int(ref["seq_len"]))
    out.write("\n")
    emit_i64(out, "kH3Ref2vaInputIds", ref["input_ids"])
    emit_i64(out, "kH3Ref2vaTokenTags", ref["token_tags"])
    emit_i64(out, "kH3Ref2vaImgPos", ref["img_pos"])
    emit_i64(out, "kH3Ref2vaAudioPos", ref["audio_pos"])
    emit_i64(out, "kH3Ref2vaUpdateMask", ref["update_mask"].to(torch.int64))
    emit_i64(out, "kH3Ref2vaAudioUpdateMask", ref["audio_update_mask"].to(torch.int64))
    emit_i64(out, "kH3Ref2vaCuSeqlens", ref["cu_seqlens"].to(torch.int64))
    emit_f64(out, "kH3Ref2vaImgPositionIds", ref["img_position_ids"])
    return result


def emit_packing(out, packed_tokens) -> None:
    out.write("// --- section 3: packed_tokens patchify / unpatchify / audio pack ---\n")
    latent_shape = (1, 5, 2, 4, 6)  # [B,C,T,H,W]
    latent = torch.from_numpy(
        h3_rand("packing.video_latent", int(np.prod(latent_shape))).astype(np.float32)
    ).reshape(latent_shape)
    rows = packed_tokens.minimax_h3_patchify_video_latent(latent, patch_size=(1, 2, 2))
    emit_scalar(out, "kH3PatchifyC", latent_shape[1])
    emit_scalar(out, "kH3PatchifyT", latent_shape[2])
    emit_scalar(out, "kH3PatchifyH", latent_shape[3])
    emit_scalar(out, "kH3PatchifyW", latent_shape[4])
    emit_scalar(out, "kH3PatchifyRows", int(rows.shape[0]))
    emit_scalar(out, "kH3PatchifyRowWidth", int(rows.shape[1]))
    out.write("\n")
    emit_f32(out, "kH3PatchifyRowsGolden", rows)

    audio_shape = (2, 6, 4)  # [audio_channel, latent_dim, T]
    audio = torch.from_numpy(
        h3_rand("packing.audio_latent", int(np.prod(audio_shape))).astype(np.float32)
    ).reshape(audio_shape)
    audio_rows = packed_tokens.minimax_h3_pack_audio_latent(audio)
    emit_scalar(out, "kH3AudioPackChannels", audio_shape[0])
    emit_scalar(out, "kH3AudioPackDim", audio_shape[1])
    emit_scalar(out, "kH3AudioPackT", audio_shape[2])
    out.write("\n")
    emit_f32(out, "kH3AudioPackRowsGolden", audio_rows)


def emit_scheduler(out, scheduling) -> None:
    out.write("// --- section 4: euler-ancestral eta0 scheduler ---\n")
    n = 24
    xt = torch.from_numpy(h3_rand("sched.xt", n).astype(np.float32))
    v = torch.from_numpy(h3_rand("sched.v", n).astype(np.float32))
    # A representative flow-matching sweep: t near 1 (start), mid, and near 0 (end).
    timesteps = [0.02, 0.5, 0.97]
    x0_all = []
    step_all = []
    for t_value in timesteps:
        timestep = torch.tensor(t_value, dtype=torch.float32)
        x0 = scheduling.minimax_h3_rf_v_to_x0(xt, v, timestep)
        sigma_curr = 1.0 - t_value
        sigma_next = sigma_curr * 0.5
        stepped = scheduling.minimax_h3_euler_eta0_step(
            xt, x0, sigma_curr=sigma_curr, sigma_next=sigma_next
        )
        x0_all.append(x0)
        step_all.append(stepped)
    emit_scalar(out, "kH3SchedN", n)
    emit_scalar(out, "kH3SchedSteps", len(timesteps))
    out.write("\n")
    emit_f32(out, "kH3SchedTimesteps", np.asarray(timesteps, dtype=np.float32))
    emit_f32(out, "kH3SchedX0Golden", torch.cat(x0_all))
    emit_f32(out, "kH3SchedStepGolden", torch.cat(step_all))


def emit_dit(out, packed) -> None:
    out.write("// --- section 5: MiniMaxH3DiTModel forward (reduced dims, fp32) ---\n")
    torch.manual_seed(0)
    arch = ARCH
    model = RefDiT(arch)

    seq_len = int(packed["seq_len"])
    img_pos = packed["img_pos"].to(torch.long)
    audio_pos = packed["audio_pos"].to(torch.long)
    text_pos = packed["text_pos"].to(torch.long)
    token_tags = packed["token_tags"].to(torch.long)
    update_mask = packed["update_mask"].to(torch.bool)
    cu_seqlens = packed["cu_seqlens"].to(torch.int64)
    img_position_ids = packed["img_position_ids"].to(torch.float32)[None]

    video_width = arch["latents_dim"] * int(np.prod(arch["patch_size"]))
    x = torch.zeros(1, seq_len, video_width, dtype=torch.float32)
    x[0].index_copy_(
        0,
        img_pos,
        torch.from_numpy(
            h3_rand("dit.video_rows", img_pos.shape[0] * video_width).astype(np.float32)
        ).reshape(img_pos.shape[0], video_width),
    )
    audio_x = torch.zeros(1, seq_len, arch["audio_latents_dim"], dtype=torch.float32)
    audio_x[0].index_copy_(
        0,
        audio_pos,
        torch.from_numpy(
            h3_rand("dit.audio_rows", audio_pos.shape[0] * arch["audio_latents_dim"]).astype(np.float32)
        ).reshape(audio_pos.shape[0], arch["audio_latents_dim"]),
    )
    prompt_embeds = torch.from_numpy(
        h3_rand("dit.prompt_embeds", text_pos.shape[0] * arch["text_dim"]).astype(np.float32)
    ).reshape(text_pos.shape[0], arch["text_dim"])

    # Timestep layout mirrors MiniMaxH3DenoiseBranch.forward_kwargs
    # (denoise_loop.py:91-126): target rows at t_video, pinned condition rows at
    # the imgvid anchor, audio rows at t_audio, everything else at t_video.
    t_video, t_audio, cond_t = 0.4, 0.35, 0.999
    timesteps = torch.full((seq_len,), t_video, dtype=torch.float32)
    timesteps[img_pos[update_mask]] = t_video
    timesteps[img_pos[~update_mask]] = cond_t
    timesteps[audio_pos] = t_audio
    unique_timesteps, inverse_indices = torch.unique(timesteps, sorted=True, return_inverse=True)

    refiner_cu = torch.tensor([0, text_pos.shape[0], text_pos.shape[0]], dtype=torch.int64)

    video_logits, audio_logits = model(
        x=x,
        audio_x=audio_x,
        img_position_ids=img_position_ids,
        unique_timesteps=unique_timesteps,
        inverse_indices=inverse_indices,
        update_mask=update_mask,
        token_tags=token_tags,
        prompt_embeds=prompt_embeds,
        img_pos=img_pos,
        audio_pos=audio_pos,
        text_pos=text_pos,
        infer_out_pos=img_pos,
        cu_seqlens=cu_seqlens,
        refiner_cu_seqlens=refiner_cu,
    )

    for key in (
        "num_layers",
        "token_refiner_num_layers",
        "hidden_size",
        "num_attention_heads",
        "attention_head_dim",
        "ffn_hidden_size",
        "latents_dim",
        "audio_latents_dim",
        "text_dim",
        "timestep_input_dim",
        "time_embed_hidden_size",
        "time_embed_dim",
        "rope_inv_freq_len",
    ):
        emit_scalar(out, f"kH3Dit_{key}", arch[key])
    emit_scalar(out, "kH3Dit_video_row_width", video_width)
    emit_scalar(out, "kH3Dit_unique_timesteps", int(unique_timesteps.shape[0]))
    out.write("\n")
    emit_f32(out, "kH3DitUniqueTimesteps", unique_timesteps)
    emit_i64(out, "kH3DitInverseIndices", inverse_indices)
    emit_f32(out, "kH3DitVideoLogitsGolden", video_logits)
    emit_f32(out, "kH3DitAudioLogitsGolden", audio_logits)
    # A handful of H3Rand probes so the C++ side can prove its PRNG matches before
    # it ever blames the model math for a mismatch.
    emit_f64(out, "kH3RandProbe", h3_rand("h3.probe", 8))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-omni", required=True, type=Path, help="path to a vllm-omni checkout")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    root = args.vllm_omni.expanduser()
    packed_sequence = load_upstream(root, "packed_sequence")
    packed_tokens = load_upstream(root, "packed_tokens")
    scheduling = load_upstream(root, "scheduling_minimax_h3_euler_ancestral")

    torch.set_grad_enabled(False)
    argv = "python3 " + " ".join(
        [os.path.relpath(sys.argv[0])] + [f"--vllm-omni {root}", f"--out {args.out}"]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        emit_header(out, argv)
        packed = emit_packed_sequence(out, packed_sequence)
        emit_packing(out, packed_tokens)
        emit_scheduler(out, scheduling)
        emit_dit(out, packed)
        out.write("}  // namespace vllm_test\n")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
