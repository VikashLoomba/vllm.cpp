#!/usr/bin/env python3
"""Run one bounded model request through the pinned plugin production path.

Each process makes one fresh engine. Repeat for both prompt indices and all
three repeats only after the first model load succeeds. A refusal is recorded
with its traceback and a nonzero exit; operation parity cannot replace it.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import traceback

from primary import seal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prompt-index", type=int, choices=(0, 1), default=0)
    parser.add_argument("--repeat", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    prompt = ([1, 0, 63, 127, 63], [1, 127, 0, 127])[args.prompt_index]
    report = {"model": seal(args.model), "config": seal(args.config / "config.json"),
              "primary_pin": "e126687a9a828d513c01a07cd69f025f27d63280",
              "plugin_pin": "d4c1f0d082fc7cd4350da56689109a01c1f29d6c",
              "prompt": prompt, "repeat": args.repeat, "tokens": [],
              "requested": {"model_dtype": "auto from bfloat16 config", "block_size": 16,
                            "num_gpu_blocks_override": 4, "max_model_len": 64,
                            "max_num_seqs": 1, "kv_cache_dtype": "auto",
                            "seed": 0x524F434D, "greedy": True, "ignore_eos": True,
                            "max_tokens": 4, "stop": [], "stop_token_ids": []},
              "harness_adaptations": ["pre-tokenized IDs with skip_tokenizer_init=True",
                                      "explicit local HF text config mirrors GGUF geometry",
                                      "no oracle patch or eager-mode override"],
              "status": "PENDING"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        import vllm
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt

        report.update(torch=torch.__version__, torch_git=torch.version.git_version,
                      hip=torch.version.hip, vllm=vllm.__version__,
                      plugin=importlib.metadata.version("vllm-gguf-plugin"))
        model = LLM(model=str(args.model), hf_config_path=str(args.config),
                    skip_tokenizer_init=True, dtype="auto", block_size=16,
                    num_gpu_blocks_override=4, max_model_len=64, max_num_seqs=1,
                    kv_cache_dtype="auto", seed=0x524F434D)
        config = model.llm_engine.vllm_config
        report["resolved_model_dtype"] = str(config.model_config.dtype)
        report["resolved_cache_config"] = str(config.cache_config)
        params = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True,
                                seed=0x524F434D, stop=[], stop_token_ids=[])
        output = model.generate([TokensPrompt(prompt_token_ids=prompt)], params, use_tqdm=False)
        if len(output) != 1 or list(output[0].prompt_token_ids) != prompt:
            raise RuntimeError("the oracle changed the request or prompt IDs")
        report["tokens"] = list(output[0].outputs[0].token_ids)
        if len(report["tokens"]) != 4:
            raise RuntimeError("the oracle did not generate exactly four tokens")
        report["status"] = "EXECUTED; token comparison remains required"
        print(json.dumps(report, sort_keys=True), flush=True)
    except Exception as error:
        report["status"] = "PENDING: primary model refused before qualification"
        report["exception_type"] = type(error).__name__
        report["exception"] = str(error)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
