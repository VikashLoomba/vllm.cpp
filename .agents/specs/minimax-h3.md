# MiniMax-H3 — omni-modal video+audio diffusion transformer

**Rows:** `MODEL-DIFFUSION-minimax-h3-mini-max-h3-dit` (model-matrix),
`ROAD-V1-H3` (roadmap portfolio).
**Claim:** `CLAIM-MINIMAX-H3-W0-W2`.
**Upstream:** vLLM-Omni (`vllm-project/vllm-omni`), `vllm_omni/diffusion/models/minimax_h3/`.
**Checkpoint:** `MiniMaxAI/MiniMax-H3` (gated), ~354 GB, BF16 safetensors.
**Status:** W0 spike + W1/W2 landed (layout + scheduler + DiT forward, parity-gated
against upstream at reduced dimensions). End-to-end generation is HARDWARE-BLOCKED.

---

## 0. Honesty statement — what is and is not claimed

MiniMax-H3 is **not an autoregressive LLM**. It is a CFG-distilled joint
video+audio **diffusion transformer**: one request runs a fixed 50-step
flow-matching denoise loop in which a 33.1B DiT is forwarded ONCE PER STEP over
the whole packed sequence, and the resulting latents are decoded to 24 FPS frames
plus a 32 kHz stereo waveform by two VAEs. There is no KV cache, no sampler, no
logits, and no token-exact gate — the SACRED near-tie methodology this project
uses for decoders does not apply to it.

**Hardware.** The upstream recipe validates on **4x NVIDIA B300** at ~133 GB peak
per rank (~103 GB with text-encoder TP). Component sizes: 52-block DiT 66.3 GB,
Qwen3-VL-derived encoder 51.5 GB, video VAE ~10 GB, audio VAE ~0.6 GB. This
project's release target is ONE GB10 with **119 GiB UNIFIED** memory (host RAM and
GPU pool are the same physical memory, see
[gb10-unified-memory](../environment.md)), and the checkpoint alone is ~354 GB of
storage. **H3 cannot run end to end on this project's hardware**, with or without
CPU offload, and no amount of software work changes that. Upstream's own
single-GPU arm needs `--enable-cpu-offload` on top of a discrete 80 GB+ card with
separate host RAM.

**Therefore:** this lane is DERIVE-AND-SHIP, like `kimi-k3.md`, but with a
materially stronger gate available — because upstream's implementation is pure
Python that runs on CPU, we can execute it at REDUCED DIMENSIONS and compare
number for number. Nothing here claims a generated video, a frame-level result, or
any speed figure.

## 1. Architecture

From `minimax_h3_transformer.py:47-78` (`MiniMaxH3DiTArchConfig`) — the shipped
geometry:

| Field | Value | Note |
|---|---|---|
| `num_layers` | 50 | AdaLN DiT blocks |
| `token_refiner_num_layers` | 2 | plain pre-norm blocks over text rows |
| `hidden_size` | 5376 | |
| `num_attention_heads` | 56 | **MHA** — `total_num_kv_heads == total_num_heads` |
| `attention_head_dim` | 128 | |
| `ffn_hidden_size` | 14336 | SwiGLU, fused `[gate; up]` fc1 |
| `latents_dim` | 24 | video VAE latent channels |
| `audio_latents_dim` | 32 | audio VAE latent channels |
| `patch_size` | (1, 2, 2) | video row width = 24*1*2*2 = **96** |
| `text_dim` | 5120 | H3-Encoder hidden width |
| `timestep_input_dim` | 256 | sinusoidal, **cosine before sine** |
| `time_embed_dim` | 2688 | AdaLN input |
| `adaln_out_features` | 18*5376 | 6 vectors x 3 modalities x H |
| `rope_inv_freq_len` | 16 | 3D RoPE rotates 6*16 = **96 of 128** head dims |

Per block: `norm1 -> AdaLN scale/shift -> attention -> AdaLN gated residual ->
norm2 -> AdaLN scale/shift -> SwiGLU MLP -> AdaLN gated residual`. AdaLN
parameters are produced per (unique timestep, modality) pair and selected per row
by `combined_indices = inverse_indices * 3 + token_tags.clamp(min=0)`
(`minimax_h3_transformer.py:1057`).

Attention is **packed varlen NON-CAUSAL**: rows of all modalities live in one
sequence, `cu_seqlens = {0, used, seq_len}` gives two documents (content and
64-alignment padding), and attention never crosses that boundary. This maps
exactly onto our shared `vt::DFlashBlockAttention(causal=false)` — no new kernel.

12 parameters plus the RoPE buffer stay **FP32** after load
(`minimax_h3_transformer.py:85-101`, re-asserted by `post_load_weights` at
`:898-904`): both patch projections, both time-embedder projections, and both
final output heads. Everything else is BF16.

## 2. Component inventory (the whole chain, not just the DiT)

| Component | Upstream | Size | Our status |
|---|---|---|---|
| Omni DiT | `minimax_h3_transformer.py` (1112 L) | 66.3 GB | **W2 LANDED** (CPU reference forward, parity-gated) |
| Packed layout | `packed_sequence.py` (572 L) | — | **W1 LANDED** (fl2va + ref2va, fp64 grid bit-exact) |
| Latent packing | `packed_tokens.py` (114 L) | — | **W1 LANDED** (+ round-trip) |
| Scheduler | `scheduling_..._euler_ancestral.py` (179 L) | — | **W1 LANDED** |
| Denoise loop | `denoise_loop.py` (249 L) | — | **W2 LANDED** (driver ported, not e2e-gated) |
| H3-Encoder | `encoder.py` (1214 L) | 51.5 GB | **W3 PENDING** — Qwen3-VL text layer-50 + vision tower + DeepStack |
| Video VAE | `vae.py` adapter + checkpoint REMOTE CODE | ~10 GB | **W4 PENDING** — see 5.1 |
| Audio VAE | `vae.py` adapter + checkpoint REMOTE CODE | ~0.6 GB | **W5 PENDING** |
| Pipeline / tasks | `pipeline_minimax_h3.py` (1196 L) | — | **W6 PENDING** |
| Conditioning | `condition_noise.py`, `reference_video.py`, `presentation.py`, `time_request.py` | — | **W6 PENDING** |
| Serving | vllm-omni `/v1/videos`, `/v1/videos/sync` | — | **W7 PENDING** |

Tasks: `t2va` (text), `fl2va` (first/last-frame), `ref2va` (reference). Duration
4-15 s snapped to `17n+5` frames at 24 FPS; 50 inference steps; flow shift 12
(video) / 3 (audio); resolution 1440p or 768p short edge, multiples of 32.

## 3. Dispatch and reuse — what we already have

* **Packed non-causal attention** -> `vt::DFlashBlockAttention(causal=false)`
  (already CPU + CUDA). Its per-document bidirectional contract IS upstream's
  varlen FA call; no new attention kernel is needed.
* **SwiGLU merged gate/up** -> the `layers::MlpGateUpMethodBase` merged-GEMM seam
  (AGENTS.md "born fused"). The W2 reference forward calls the projections
  directly; folding onto the seam is part of W2b.
* **Add+RMSNorm glue** -> `vt::FusedChain` recipes, W2b.
* **H3-Encoder** -> our existing `qwen3_vl_{text,vision}.cpp` +
  `multimodal/qwen3vl_processor.cpp`. The deltas are: keep only the first 50
  decoder layers and consume the UNNORMALIZED hidden state after layer 49; all-ones
  attention mask; DeepStack injection at the first `len(deepstack_visual_indexes)`
  layers. This is the single largest reuse in the port.
* **Multi-GPU** — upstream uses Ulysses sequence parallelism (`--usp 4`) plus
  optional DiT TP. We have `vt::communicator` / NCCL but no USP; this is W8 and is
  only reachable on multi-GPU hardware we do not have.

## 4. Gates

**What can be gated on any CPU (and IS, as of W2):** upstream's pure-Python
modules are imported by file path and executed at reduced dimensions;
`scripts/gen-minimax-h3-goldens.py` freezes their outputs into
`tests/vllm/models/minimax_h3_goldens.inc`, and `tests/vllm/models/test_minimax_h3.cpp`
reproduces them. Weights and inputs are rebuilt on both sides from an identical
FNV-1a + splitmix64 stream, so not one weight byte is checked in.

Landed results (`build-cpu`, Release, 10/10 test cases, 2539 assertions):

| Gate | Result |
|---|---|
| fl2va packed layout (ids, tags, positions, masks, cu_seqlens, doc ids) | **exact** |
| fl2va fp64 position grid | **bit-exact** (all 192 doubles) |
| ref2va block layout (image + video_audio reference blocks) | **exact**, incl. fp64 grid |
| patchify / unpatchify / audio pack / unpack | **exact** + round-trip identity |
| euler-ancestral eta0 scheduler + `rf_v_to_x0` | **exact** (<= 1e-6) |
| **DiT forward, reduced dims, f32** | **max abs diff 1.6e-7 (video), 1.5e-7 (audio)** |
| denoise-loop INVARIANTS (pinned rows reset every step, targets advance, finite) | pass |
| config-parse invariants + weight contract + grouped-qkv reorder | pass |

The fp64 position grid is gated bit-exact deliberately: it feeds RoPE, and a
last-ulp drift would silently rotate every video token. The port therefore
reproduces upstream's arithmetic ORDER — `numpy.linspace(endpoint=False)`
evaluates `i*step + start`; `_temporal_position_span` uses numpy PAIRWISE
summation while `_video_t_span` uses Python's SEQUENTIAL `sum()`, which upstream
keeps separate on purpose (`packed_sequence.py:101-113`).

**What cannot be gated here:** any end-to-end video/audio result, any speed
number, the encoder/VAE numerics (no checkpoint), and the multi-GPU USP path. All
are recorded PENDING in `docs/BENCHMARKS.md`, not as passes.

**Oracle note.** The parity pin (`555967922`, vLLM 0.26.0.dev0) does NOT contain
MiniMax-H3 — H3 was released after it, and it lives in the separate `vllm-omni`
repository, which the pin protocol does not currently cover. Advancing the pin
does not by itself make H3 gateable; a vllm-omni pin is a prerequisite for W3+ and
is tracked as an open item in 7.

## 5. Known hard parts

### 5.1 The VAEs are REMOTE CODE, not upstream Python

`vae.py:41-53` loads both VAEs with
`get_class_from_dynamic_module(config["auto_map"]["AutoModel"], component_path)` —
i.e. the actual VAE implementations ship INSIDE the HF checkpoint and run under
`--trust-remote-code`. vLLM-Omni only adapts them. A pure-C++ engine cannot do
that: W4/W5 must **reimplement both VAEs in C++ from the checkpoint's Python
source**, which must be fetched separately (the VAE modules and their `config.json`
are small; the 354 GB of weights are not needed to READ the architecture).

Contracts already pinned down from the adapter:
* Video VAE weights stay **FP32**; keyframe encode is seeded
  (`MINIMAX_H3_KEYFRAME_ENCODE_SEED = 42`) and its normalize+patchify runs on
  **CPU in FP32** on purpose (`vae.py:185-202`) — doing it on CUDA measurably
  changes the conditioned video.
* Latents are normalized by per-channel `latents_mean`/`latents_std` from the
  component `config.json`, then patchified with (1,2,2).
* Audio VAE is FP32 for both encode and decode, 32 kHz, 2 channels, and encode
  runs under a determinism context that disables TF32, cuDNN, and the fused SDP
  backends (`vae.py:56-94`) — the C++ port must match that numerically, not just
  structurally.

### 5.2 Output is a container, not tokens

`/v1/videos` returns MP4 (H.264 video + stereo audio). We have no muxer and no
video/audio ENCODER anywhere in the tree (`third_party/` has blake3, doctest,
httplib, minja, nlohmann, vulkan). W7 must choose: vendor a minimal MP4 muxer plus
an encoder, or take a dependency. This is a genuine new dependency decision and is
called out rather than assumed.

### 5.3 Speed

Upstream reports the DiT at **88% of request latency** and FL2VA at ~87 s E2E for
an 8.7 s 1248x768 clip on 4x B300, with regional `torch.compile`, cache-dit block
caching, and USP-4. Matching that needs the device-resident forward (W2b), the
fusion folds, and multi-GPU. No speed claim is possible before W2b lands and
hardware exists to measure on.

## 6. Files ported in this change

| Ours | Upstream |
|---|---|
| `include/vllm/model_executor/models/minimax_h3.h` | the module's public contracts |
| `src/vllm/model_executor/models/minimax_h3_packing.cpp` | `packed_tokens.py`, `packed_sequence.py`, `scheduling_..._euler_ancestral.py` |
| `src/vllm/model_executor/models/minimax_h3.cpp` | `minimax_h3_transformer.py`, `denoise_loop.py` |
| `scripts/gen-minimax-h3-goldens.py` | executes the above upstream modules as the oracle |
| `tests/vllm/models/test_minimax_h3.cpp` | `tests/diffusion/models/minimax_h3/test_minimax_h3_{packing,contract}.py` |

**Tests to port (upstream `tests/diffusion/models/minimax_h3/`):**
`test_minimax_h3_packing.py` (DONE — layout + patchify goldens),
`test_minimax_h3_contract.py` (PARTIAL — config/weight contract done, pipeline
contract pending W6), `test_minimax_h3_e2e.py` (BLOCKED — needs the checkpoint),
`test_minimax_h3_parallel.py` (BLOCKED — needs multi-GPU).

## 7. Work breakdown

| Brick | Scope | Blocked by |
|---|---|---|
| **W0** | Spike, component inventory, hardware verdict | — (DONE) |
| **W1** | Packed layout + latent packing + scheduler, parity-gated | — (DONE) |
| **W2** | DiT forward + denoise driver, parity-gated on CPU at reduced dims | — (DONE) |
| **W2b** | Device-resident forward: bf16 stream, `vt::FusedChain` glue, fused AdaLN modulate op, merged gate/up seam, CUDA gate | — |
| **W3** | H3-Encoder on the existing Qwen3-VL tower (layer-50 truncation, DeepStack, all-ones mask) | vllm-omni pin |
| **W4** | Video VAE reimplementation from checkpoint remote code | fetch VAE source |
| **W5** | Audio VAE reimplementation (+ determinism context semantics) | fetch VAE source |
| **W6** | Pipeline: t2va/fl2va/ref2va task assembly, condition noise, reference video, presentation, sigma schedules | W3-W5 |
| **W7** | Serving: `/v1/videos` + `/v1/videos/sync`, job store, MP4 mux (dependency decision in 5.2) | W6 |
| **W8** | Speed: USP sequence parallelism, block caching, DiT TP | W2b + multi-GPU HW |

**Open items.** (a) A vllm-omni parity pin — the upstream-sync protocol currently
covers only the vLLM repo; H3 lives outside it. (b) The MP4 dependency decision.
(c) Hardware: nothing past W2b/W3 can be END-TO-END gated on this project's boxes,
so W4-W8 should be reviewed as structural ports with unit gates, and the honest
lifecycle cap for this row is "correctness-complete, hardware-blocked".
