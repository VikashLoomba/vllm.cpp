#!/usr/bin/env python3
"""Fold one llama.cpp denominator run into a RESULT.json.

It states, in its own output and in words, WHAT it compared against WHAT:
every derived number carries the population it came from, and the clock
instrument names itself as ad-hoc rather than as the committed helper.

N comes from --legs, which is the DESIGN. It is never derived by counting log
lines: the job log is tee'd and a grep tally reads every leg twice.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--evidence", type=pathlib.Path, required=True)
ap.add_argument("--legs", type=int, required=True, help="leg count from the DESIGN")
ap.add_argument("--busy-threshold", type=int, default=90)
args = ap.parse_args()

E = args.evidence


def clock_window(path: pathlib.Path, threshold: int) -> dict:
    if not path.is_file():
        return {"error": f"no sampler output at {path.name}"}
    samples = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    def fold(rows: list[dict], label: str) -> dict:
        sclks = [int(r["sclk_mhz"]) for r in rows if r.get("sclk_mhz") is not None]
        if not sclks:
            return {"window": label, "samples": len(rows), "sclk_mhz": None,
                    "note": "no active sclk value in this window"}
        med = statistics.median(sclks)
        return {
            "window": label,
            "samples": len(rows),
            "sclk_mhz": {
                "min": min(sclks),
                "median": med,
                "mean": round(statistics.fmean(sclks), 1),
                "max": max(sclks),
                "spread_pct": round(100.0 * (max(sclks) - min(sclks)) / med, 1),
            },
        }

    busy = [r for r in samples
            if r.get("busy_percent") is not None and int(r["busy_percent"]) >= threshold]
    return {
        "instrument": "AD-HOC /sys/class/drm sampler carried beside the job; "
                      "NOT tools/bench/gpu_clock_state.py, which reads NVIDIA fields "
                      "and does not run on this board (#2381 owns the in-tree gap)",
        "interval_s": 0.25,
        "boot_ids": sorted({str(r.get("boot_id")) for r in samples}),
        "whole_window": fold(samples, f"whole leg, including the 17 GiB model load"),
        "compute_window": fold(busy, f"gpu_busy_percent >= {threshold}"),
    }


legs = []
for i in range(1, args.legs + 1):
    row: dict = {"leg": i, "order": i}
    jpath = E / f"leg{i}.json"
    rcpath = E / f"leg{i}.rc"
    row["rc"] = int(rcpath.read_text().strip()) if rcpath.is_file() else None
    if jpath.is_file():
        try:
            tests = json.loads(jpath.read_text())
        except json.JSONDecodeError as exc:
            row["error"] = f"llama-bench JSON did not parse: {exc}"
            tests = []
        for t in tests:
            row["avg_ts"] = t.get("avg_ts")
            row["stddev_ts"] = t.get("stddev_ts")
            row["samples_ts"] = t.get("samples_ts")
            row["n_gen"] = t.get("n_gen")
            row["n_prompt"] = t.get("n_prompt")
            row["n_gpu_layers"] = t.get("n_gpu_layers")
            row["n_threads"] = t.get("n_threads")
            row["backends"] = t.get("backends")
            row["build_commit"] = t.get("build_commit")
            row["build_number"] = t.get("build_number")
            row["cpu_info"] = t.get("cpu_info")
            row["gpu_info"] = t.get("gpu_info")
            row["model_type"] = t.get("model_type")
            row["model_size"] = t.get("model_size")
            row["flash_attn"] = t.get("flash_attn")
            row["type_k"] = t.get("type_k")
            row["type_v"] = t.get("type_v")
    else:
        row["error"] = "no llama-bench JSON for this leg"
    row["clock"] = clock_window(E / f"clock-leg{i}.jsonl", args.busy_threshold)
    legs.append(row)

usable = [l for l in legs if l.get("avg_ts") is not None and l.get("rc") == 0]
values = [float(l["avg_ts"]) for l in usable]
reps: list[float] = []
for l in usable:
    reps.extend(float(v) for v in (l.get("samples_ts") or []))

reasons = []
if len(legs) != args.legs:
    reasons.append(f"design declares {args.legs} legs, folded {len(legs)}")
if len(usable) != args.legs:
    reasons.append(f"{args.legs - len(usable)} of {args.legs} legs produced no usable figure")
if len(usable) < 2:
    reasons.append("a denominator may not be reported from one leg")

summary = None
if values:
    med = statistics.median(values)
    summary = {
        "population": f"one avg_ts per leg, {len(values)} legs, "
                      "each the mean of that leg's own repetitions",
        "leg_count": len(values),
        "median_tok_s": round(med, 3),
        "mean_tok_s": round(statistics.fmean(values), 3),
        "min_tok_s": round(min(values), 3),
        "max_tok_s": round(max(values), 3),
        "spread_pct_of_median": round(100.0 * (max(values) - min(values)) / med, 3),
        "repetition_count": len(reps),
        "repetition_median_tok_s": round(statistics.median(reps), 3) if reps else None,
        "repetition_min_tok_s": round(min(reps), 3) if reps else None,
        "repetition_max_tok_s": round(max(reps), 3) if reps else None,
    }

result = {
    "what_this_is": "llama.cpp b10451 decode on gfx1151, ONE engine. "
                    "A denominator, measured alone.",
    "what_this_is_not": "It is not a comparison. No vllm.cpp figure was produced "
                        "by this run and no ratio is computed here: that arm's "
                        "declared token gate reads FAIL at 3 of 6 "
                        "(docs/bench-evidence/qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md) "
                        "and AGENTS.md Gates admits no performance result from it.",
    "status": "MEASURED" if not reasons else "INCOMPLETE",
    "reasons": reasons,
    "legs_by_design": args.legs,
    "legs_folded": len(legs),
    "denominator": summary,
    "clock_attribution": "AD-HOC. Every clock figure below comes from the carried "
                         "sysfs sampler, not from the committed helper (#2381).",
    "legs": legs,
}

(E / "RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
sys.exit(0 if not reasons else 3)
