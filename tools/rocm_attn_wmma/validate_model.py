#!/usr/bin/env python3
"""Require exact public completion IDs and a WMMA dispatch for every prefill layer."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("primary", type=Path)
    p.add_argument("native", type=Path)
    p.add_argument("trace", type=Path)
    p.add_argument("--layers", type=int, required=True)
    p.add_argument("--arm", choices=("wmma", "scalar"), default="wmma")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest, primary, native = [json.loads(path.read_text())
                                for path in (args.manifest, args.primary, args.native)]
    cases = manifest["cases"]
    block_size = manifest.get("block_size", 16)
    if primary.get("block_size", 16) != block_size or native.get("block_size", 16) != block_size:
        raise ValueError("cache block size differs from manifest")
    if not cases or args.layers <= 0 or manifest["output_len"] <= 0:
        raise ValueError("empty model gate")
    if len(cases) != len(primary["cases"]) or len(cases) != len(native["cases"]):
        raise ValueError("completion case count differs")
    for case, ref, actual in zip(cases, primary["cases"], native["cases"]):
        for field in ("name", "prompt_token_ids"):
            if ref[field] != case[field] or actual[field] != case[field]:
                raise ValueError(f"{case['name']}: {field} differs")
        if len(actual["output_token_ids"]) != manifest["output_len"]:
            raise ValueError(f"{case['name']}: incomplete completion")
        if actual["output_token_ids"] != ref["output_token_ids"]:
            raise ValueError(f"{case['name']}: output token IDs differ")
    # Decode shares the WMMA specialization. rocprof records grid sizes in
    # work-items, so a single workgroup in X is decode, not a prefill witness.
    kernel = "PagedAttnPrefillSharedKWmma<" if args.arm == "wmma" else "PagedAttnPrefillSharedK<"
    tile = 16 if args.arm == "wmma" else 32
    if any(len(case["prompt_token_ids"]) < 64 for case in cases):
        raise ValueError("manifest misses the SharedK prefill admission")
    with args.trace.open(newline="") as stream:
        calls = [row for row in csv.DictReader(stream)
                 if kernel in row["Kernel_Name"]
                 and int(row["Grid_Size_X"]) > int(row["Workgroup_Size_X"])]
    if len(calls) != args.layers * len(cases):
        raise ValueError(f"expected {args.layers * len(cases)} {args.arm} prefill dispatches, got {len(calls)}")
    expected = Counter((len(case["prompt_token_ids"]) + tile - 1) // tile for case in cases)
    observed = Counter(int(row["Grid_Size_X"]) // int(row["Workgroup_Size_X"]) for row in calls)
    if observed != Counter({groups: count * args.layers for groups, count in expected.items()}):
        raise ValueError("prefill dispatch geometry differs from manifest")
    if any(int(row["Scratch_Size"]) != 0 for row in calls):
        raise ValueError("WMMA dispatch uses scratch memory")
    receipt = dict(token_exact=len(cases) * manifest["output_len"],
                   arm=args.arm, prefill_dispatches=len(calls), scratch_bytes=0,
                   runtime_vgpr=sorted({int(row["VGPR_Count"]) for row in calls}),
                   runtime_sgpr=sorted({int(row["SGPR_Count"]) for row in calls}))
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
