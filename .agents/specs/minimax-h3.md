# MiniMax-H3 — omni-modal video+audio diffusion transformer

**Rows:** `MODEL-DIFFUSION-minimax-h3-mini-max-h3-dit` (model-matrix),
`ROAD-V1-H3` (roadmap portfolio).
**Claim:** `CLAIM-MINIMAX-H3-W0-W2`, `CLAIM-MINIMAX-H3-W6A-W9`.
**Upstream:** vLLM-Omni (`vllm-project/vllm-omni`), `vllm_omni/diffusion/models/minimax_h3/`.
**Checkpoint:** `MiniMaxAI/MiniMax-H3` (gated), ~354 GB, BF16 safetensors.
**Quantized checkpoints:** `realrebelai/MiniMax-H3_GGUFs` (ComfyUI GGUF),
`lilcheaty/MiniMax-H3-NVFP4` — these FIT one GB10; see section 0.
**Status:** W0 spike + W1/W2 (layout, scheduler, DiT forward incl. the bf16
production stream), W6a (request planning) and W9 shape/geometry (GGUF arm) landed,
all parity-gated. End-to-end is NOT hardware-blocked on the quantized arms; it is
gated on the remaining bricks (encoder, VAEs, pipeline).

---

## 0. Honesty statement — what is and is not claimed

MiniMax-H3 is **not an autoregressive LLM**. It is a CFG-distilled joint
video+audio **diffusion transformer**: one request runs a fixed 50-step
flow-matching denoise loop in which a 33.1B DiT is forwarded ONCE PER STEP over
the whole packed sequence, and the resulting latents are decoded to 24 FPS frames
plus a 32 kHz stereo waveform by two VAEs. There is no KV cache, no sampler, no
logits, and no token-exact gate — the SACRED near-tie methodology this project
uses for decoders does not apply to it.

**Hardware — CORRECTED 2026-08-03 (user-directed).** The BF16 release validates on
**4x NVIDIA B300** at ~133 GB peak per rank (~103 GB with text-encoder TP), and at
~354 GB of storage it does not fit one GB10 (119 GiB UNIFIED). An earlier revision
of this spec concluded from that alone that H3 e2e was "impossible on this
project's hardware". **That was wrong**: it reasoned from the BF16 release only.
QUANTIZED H3 checkpoints exist and they DO fit:

| Arm | Components | Size |
|---|---|---|
| GGUF (ComfyUI format) | DiT `MiniMax-H3-FL2VA-Q3_K_M.gguf` 15.6 GB + Qwen3-VL encoder `qwen3vl-32B-...-Q4_K_M.gguf` 14.6 GB + the two VAEs (fp16 video ~10 GB, fp32 audio ~0.6 GB) | **~41 GB** |
| NVFP4 (safetensors) | `minimax_h3_ref2va_nvfp4_{full,mixed}.safetensors` + `text_encoders/qwen3vl_32b_..._nvfp4_awq.safetensors` + `vae/minimax_h3_{video_vae_fp16,audio_vae_fp32}.safetensors` | fits, repo 77.2 GB across 3 DiT variants |

Sources: `realrebelai/MiniMax-H3_GGUFs` and `lilcheaty/MiniMax-H3-NVFP4`. Both land
well inside the 119 GiB pool, so **end-to-end H3 IS reachable here, and therefore
so is a speed comparison.** NVFP4 is the more interesting arm for this project:
GB10/sm_121 has native FP4 tensor cores and our NVFP4 stack (cutlass FP4 GEMM,
Marlin W4A16 grouped MoE, the Laguna arm's tuning) is the most optimized path we
own. What remains blocked is a like-for-like comparison against vLLM-Omni's own
published numbers, which were measured on 4x B300 — a different machine class.

**Therefore:** the CORRECTNESS gate is upstream itself executed at REDUCED
DIMENSIONS on CPU (section 4) — that is available today and is exact. The
END-TO-END gate is now a matter of finishing the remaining bricks (encoder, VAEs,
pipeline) and downloading a quantized checkpoint, NOT a hardware wall. Nothing in
THIS change claims a generated video or a speed figure; what changed is that both
are now on the critical path rather than out of reach.

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
| H3-Encoder | `encoder.py` (1214 L) | 51.5 GB | **W3 TEXT TOWER DONE** — truncation + UNNORMALIZED output + DeepStack gated at 1.2e-7; the VISION tower (reuse of our qwen3_vl_vision) and the MM processor remain |
| Video VAE | `vae.py` adapter + checkpoint REMOTE CODE (`FL2VA/video_vae/*.py`) | ~10 GB | **W4 DECODER DONE** — the FULL ViT3D decoder (pack, x_embedder, register/cls tokens, 3D RoPE, 36-block stack, norm_out, proj_out, unpatchify) is ported and gated at **8.9e-8**. Tiling and the 3D-CNN encoder (conditioning only) remain. See 5.1 |
| Audio VAE | `vae.py` adapter + checkpoint REMOTE CODE (`FL2VA/audio_vae/*.py`) | ~0.6 GB | **W5 LANDED** — DAC/BigVGAN decoder REIMPLEMENTED, gated vs the checkpoint's own modules at 4.2e-9 |
| Pipeline / tasks | `pipeline_minimax_h3.py` (1196 L) | — | **W6 t2va ASSEMBLED** — the whole path composes and runs (structural e2e gate); fl2va/ref2va conditioning and the torch-RNG noise seed remain |
| Conditioning | `condition_noise.py`, `reference_video.py`, `presentation.py`, `time_request.py` | — | **W6 PENDING** |
| Serving | vllm-omni `/v1/videos`, `/v1/videos/sync` | — | **W7 PENDING** |
| GGUF arm (ComfyUI format) | `realrebelai/MiniMax-H3_GGUFs` | 15.6 GB (DiT Q3_K_M) | **W9 LANDED (shape/geometry)** — name map is the IDENTITY, gated on the real 535-tensor manifest |
| NVFP4 arm | `lilcheaty/MiniMax-H3-NVFP4` | fits | **W10 GROUNDED** — the real 1051-tensor manifest is textbook compressed-tensors NVFP4 (U8 packed + E4M3 group-16 `weight_scale` + F32 `weight_scale_2`), i.e. EXACTLY our existing layout; 258 quantized projections, islands unquantized |

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
| bf16 PRODUCTION stream vs upstream's dtype policy | max abs diff 2.4e-3 (bf16 scale) |
| request planning (frames, latent shapes, sigma schedules, canvas, task dispatch) | **exact** |
| **REAL GGUF manifest** (535 tensors of `MiniMax-H3-FL2VA-Q3_K_M.gguf`) | **exact** — every name and logical shape matches our contract, geometry derived from shapes alone equals the shipped H3 config |
| **AUDIO VAE decoder** vs the checkpoint's OWN remote code | **max abs diff 4.2e-9** (kaiser-sinc filter 3.0e-8) |
| **REAL NVFP4 manifest** (1051 tensors) | **exact** — compressed-tensors triple, group 16, islands unquantized, names identical to our contract |
| **REAL video-VAE manifest** (560 tensors) | **exact** — decoder confirmed a 36-block ViT, encoder the 3D CNN |
| **VIDEO VAE decoder TransformerBlock** vs the checkpoint's OWN remote code | **max abs diff 6.0e-8** |
| **VIDEO VAE FULL ViT3D decoder** vs the checkpoint's OWN remote code | **max abs diff 8.9e-8** |
| **ENCODER text tower** (truncation + unnormalized output + DeepStack) | **max abs diff 1.2e-7** |
| **WHOLE t2va PATH composes** (layout -> sigmas -> denoise loop -> unpack -> denormalize -> both VAEs) | frames + stereo waveform, correctly shaped, finite, in [-1, 1] |
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

**Status 2026-08-03: both VAE DECODERS are DONE.** The audio VAE (DAC/BigVGAN,
4.2e-9) and the video VAE's full ViT3D decoder (8.9e-8) both reproduce the
checkpoint's own modules. What remains on the VAE side is video tiling and the
3D-CNN ENCODER — and the encoder is only needed for image/video CONDITIONING
(fl2va/ref2va), not for producing output frames.

**Original note: the remote code is IN HAND** (fetched from the checkpoint's
`FL2VA/{audio,video}_vae/`, ~130 KB of Python, NOT vendored here — it ships under
the MiniMax H3 Community License). The **audio VAE is DONE** (W5): a DAC-lineage
BigVGAN vocoder, reimplemented and gated against the checkpoint's own modules at
4.2e-9 by `scripts/gen-minimax-h3-audio-vae-goldens.py`. The **video VAE (W4)** is the largest remaining brick, but the real
checkpoint manifest (560 tensors, `FL2VA/video_vae/source/model.safetensors`,
captured by range request) makes it materially smaller than `klvae.py`'s 48 KB
suggested: the **ENCODER** is the 3D CNN (116 tensors, rank-5 Conv3d down blocks)
while the **DECODER** — the half generation actually needs — is a plain **36-block
TRANSFORMER** (440 tensors: `attn.to_qkv`/`attn.to_out`, `ff.w1`/`ff.w2`, two
norms and two learned residual scales per block, plus `x_embedder`, `mask_token`,
`register_tokens`, `norm_out`, `proj_out`). We have every primitive for that. The
whole checkpoint is fp32.

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
| **W3** | H3-Encoder. **TEXT TOWER DONE** (1.2e-7): the three H3 deltas — layer truncation `min(num_hidden_layers, 50)`, the UNNORMALIZED layer-49 output (no final RMSNorm), and DeepStack injection into the first N layers — plus interleaved M-RoPE, fused QKV, per-head q/k RMSNorm, causal GQA and the gated-SiLU MLP. REMAINS: the VISION tower (reuse `qwen3_vl_vision.cpp`) and the MM processor | — |
| **W4** | Video VAE. **DECODER DONE** — the full ViT3D decoder gated at 8.9e-8 (block 6.0e-8), real hyperparameters 36 layers / 32 heads x 64 / rope_theta 100 / rope_dim_ratio 0.75 from the checkpoint's `vit_decoder_kwargs`. REMAINS: tiling (`vae_tile_size` 256 / overlap 64) and the 3D-CNN ENCODER, which is only needed for image/video CONDITIONING, not for generation output | — |
| **W5** | Audio VAE reimplementation | **DONE** — DAC-lineage BigVGAN decoder (weight-norm materialization, anti-aliased SnakeBeta with kaiser-sinc up/down resampling, replicate padding, final clamp). Encode-side determinism context is still open |
| **W6** | Pipeline. **t2va ASSEMBLED** — `MiniMaxH3GenerateT2va` wires layout -> sigma schedules -> denoise loop -> unpatchify/audio-unpack -> denormalize -> both VAE decoders, gated by a structural end-to-end test. REMAINS: fl2va/ref2va conditioning (condition noise, reference video, presentation) and bit-exact torch-RNG noise seeding | — |
| **W7** | Serving: `/v1/videos` + `/v1/videos/sync`, job store, MP4 mux (dependency decision in 5.2) | W6 |
| **W8** | Speed: USP sequence parallelism, block caching, DiT TP | W2b + multi-GPU HW |
| **W9** | **GGUF arm** — ComfyUI-format load (identity name map, ne reversal, `comfy.gguf.orig_shape` reshape rule, K-quant dequant). Shape/geometry resolution LANDED and gated on the real manifest; the dequant-into-weights path needs the file | — |
| **W10** | **NVFP4 arm** — `lilcheaty/MiniMax-H3-NVFP4` onto our existing NVFP4 stack (cutlass FP4 GEMM on sm_121). Layout GATED as identical to ours, so this is a loader-wiring brick, not a new quant scheme. The most promising SPEED path, and the one that makes an e2e run on one GB10 realistic | W9 |

**Open items.** (0) Run the assembled t2va path on a REAL quantized checkpoint — the
pipeline now composes end to end at reduced dimensions, so what remains is loader
wiring (W9 dequant / W10 NVFP4), the encoder's vision tower, and a GPU. This
supersedes the old "hardware-blocked" framing. (0b) Noise seeding is currently an
INPUT: upstream seeds a torch CPU generator, and matching it bit-exactly decides
WHICH sample you get, not whether the pipeline is correct.
(a) A vllm-omni parity pin — the upstream-sync protocol currently
covers only the vLLM repo; H3 lives outside it. (b) The MP4 dependency decision.
(c) Hardware: nothing past W2b/W3 can be END-TO-END gated on this project's boxes,
so W4-W8 should be reviewed as structural ports with unit gates, and the honest
lifecycle cap for this row is "correctness-complete, hardware-blocked".
