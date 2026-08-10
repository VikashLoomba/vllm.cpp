#!/usr/bin/env python3
"""Resolve container tags from release/container-matrix.json.

The publish workflow needs three different views of the same matrix, and none
of them belongs inline in YAML where it cannot be unit-tested:

  --immutable  one `<package>:<version>-<lane>` per lane
  --moving     `<source> <target>` pairs, immutable tag -> moving tag
  --lanes      lane ids, one per line

Usage: scripts/container_tags.py --version 0.1.0 [--immutable|--moving|--lanes]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MATRIX = ROOT / "release/container-matrix.json"


def immutable_tags(matrix: dict, version: str) -> list[str]:
    package = matrix["package"]
    return [
        f"{package}:{lane['version_tag'].format(version=version)}"
        for lane in matrix["lanes"]
    ]


def moving_pairs(matrix: dict, version: str) -> list[tuple[str, str]]:
    package = matrix["package"]
    pairs: list[tuple[str, str]] = []
    for lane in matrix["lanes"]:
        source = f"{package}:{lane['version_tag'].format(version=version)}"
        for moving in lane["moving_tags"]:
            pairs.append((source, f"{package}:{moving}"))
    return pairs


def lane_ids(matrix: dict) -> list[str]:
    return [lane["id"] for lane in matrix["lanes"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--version")
    view = parser.add_mutually_exclusive_group(required=True)
    view.add_argument("--immutable", action="store_true")
    view.add_argument("--moving", action="store_true")
    view.add_argument("--lanes", action="store_true")
    args = parser.parse_args()

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))

    if args.lanes:
        print("\n".join(lane_ids(matrix)))
        return 0

    if not args.version:
        parser.error("--version is required for --immutable and --moving")

    if args.immutable:
        print("\n".join(immutable_tags(matrix, args.version)))
    else:
        print("\n".join(f"{source} {target}" for source, target in moving_pairs(matrix, args.version)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
