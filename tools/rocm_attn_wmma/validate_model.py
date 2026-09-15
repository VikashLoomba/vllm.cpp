#!/usr/bin/env python3
"""Require exact public completion IDs and a WMMA dispatch for every prefill layer."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("primary", type=Path)
    p.add_argument("native", type=Path)
    p.add_argument("trace", type=Path)
    p.add_argument("--layers", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest, primary, native = [json.loads(path.read_text())
                                for path in (args.manifest, args.primary, args.native)]
    cases = manifest["cases"]
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
    with args.trace.open(newline="") as stream:
        calls = [row for row in csv.DictReader(stream)
                 if "PagedAttnPrefillSharedKWmma<" in row["Kernel_Name"]]
    if len(calls) != args.layers * len(cases):
        raise ValueError(f"expected {args.layers * len(cases)} WMMA dispatches, got {len(calls)}")
    if any(int(row["Scratch_Size"]) != 0 for row in calls):
        raise ValueError("WMMA dispatch uses scratch memory")
    receipt = dict(token_exact=len(cases) * manifest["output_len"],
                   wmma_dispatches=len(calls), scratch_bytes=0,
                   runtime_vgpr=sorted({int(row["VGPR_Count"]) for row in calls}),
                   runtime_sgpr=sorted({int(row["SGPR_Count"]) for row in calls}))
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
