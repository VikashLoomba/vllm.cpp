ID: ISSUE-LOCAL-01M2HMDT8EB7WZGW7SXQ9RWJVW
Title: Gemma 3 4B cannot load its linear RoPE configuration
Row: MODEL-GEMMA3-LINEAR-ROPE
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-14
Updated: 2026-09-14
Closed: -

## Problem

The pinned unsloth/gemma-3-4b-it checkpoint at bf46152c47f5dd20b896357cb51abc4c03b8ee8c declares linear RoPE with factor 8. Public vllm_engine_load on base 31509d91f rejects that type in hf_config.cpp before any attention dispatch. gemma3.cpp also calls unscaled RopeNeox directly on global layers, so relaxing the parser alone would be incorrect. Pinned vLLM e126687a9 loads and generates from the identical lossless text export with its production configuration. This blocks the real-model gate for gfx1100 SharedK rocWMMA; see build-rdna3-attn/evidence/model-on-monitor.stderr.log and model-primary.json. The required repair needs a separate spec covering linear scaling and Gemma per-layer rotary routing.

## Resolution

-
