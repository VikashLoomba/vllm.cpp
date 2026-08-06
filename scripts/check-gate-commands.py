#!/usr/bin/env python3
"""Classify each gated row's Gates section: does it name a command that can FAIL?

A row's `Gates` field promises "exact commands", and nothing has ever checked
that one exists or that it can fail. A gate that is `true`, `echo ok`, or piped
into another command collapses "done" into the implementer's opinion of its own
work.

This ships as a CLASSIFIER first and a ratchet second, deliberately: most gated
rows cannot state a runnable command today, so a gate demanding one would be red
on arrival and would have to be relaxed to pass. A relaxed gate is worse than no
gate.

    scripts/check-gate-commands.py            # report
    scripts/check-gate-commands.py --json     # machine-readable
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


record = _load("agent_record", "scripts/check-agent-record.py")

# DONE is included: a row that lost its gate command is exactly the regression
# this exists to catch, and DONE rows are the ones people stop looking at.
GATED_STATES = frozenset({"READY", "ACTIVE", "GATING", "DONE", "BLOCKED"})

# check-agent-record.py's MATRIX_PATHS covers 5 of the 7 matrices. Audit all
# seven, without widening that constant -- it governs a repo-wide CI gate whose
# row contract these two files have never been held to.
AUDITED_MATRIX_PATHS = [
    *record.MATRIX_PATHS,
    record.AGENTS / "feature-matrix.md",
    record.AGENTS / "sglang-matrix.md",
]

_GATES_HEADING = re.compile(r"(?im)^#{1,6}\s*gates\b.*$")
_HEADING = re.compile(r"(?m)^#{1,6}\s")

# A command names an executable: a known tool as a WHOLE WORD, or a path that is
# actually INVOKED. A backticked filename is not a command -- `docs/BENCHMARKS.md`
# and `tests/vllm/models/test_model_registry.cpp` are things a gate talks about,
# not things it runs. Both boundaries matter on the shipped record: without the
# trailing one `sha256_cbor` matches `sh` and `python@3.14` matches `python`.
_TOOL = re.compile(
    r"(?:^|\s)(?:ctest|pytest|python3?|cmake|bash|sh|make|nsys|ncu|git|gh)(?:\s|$)"
)
# `flock <lock> -c '<gate>'` is this repo's MANDATED shape for any gate touching
# the GPU, and the wrapper QUOTES the real command, putting it out of reach of
# every other rule here. It needs a lockfile AND something to run: the bare
# `flock` and `flock /tmp/gpu` that appear in three specs name the idiom, not a
# gate, and a plain vocabulary entry credits all three with a command.
_WRAPPER = re.compile(r"(?:^|\s)flock\s+\S+\s+\S")
# `./anything` is an explicit invocation, arguments or not -- a built test binary
# (`./build-cuda-121a/tests/test_dropin_abi`) is run, not referred to. A bare
# `scripts/`/`tests/` path is only a command when it carries an executable suffix
# or arguments; otherwise it is a filename.
_INVOKED_PATH = re.compile(
    r"(?:^|\s)(?:\./\S+|(?:scripts|tests)/\S*(?:\.(?:py|sh)(?:\s|$)|\s+\S))"
)
# Shapes that cannot fail, so they are not gates at all.
_CANNOT_FAIL = re.compile(r"^\s*(true|:|echo\b)")


def gates_section(text: str) -> str | None:
    """The body under the first `Gates` HEADING, or None. Prose does not count."""
    match = _GATES_HEADING.search(text)
    if not match:
        return None
    rest = text[match.end() :]
    nxt = _HEADING.search(rest)
    return rest[: nxt.start()] if nxt else rest


def _candidates(section: str) -> list[str]:
    inline = re.findall(r"`([^`\n]+)`", section)
    fenced = re.findall(r"```[a-z]*\n(.*?)```", section, re.S)
    for block in fenced:
        inline.extend(line for line in block.splitlines() if line.strip())
    return [c.strip() for c in inline if c.strip()]


def is_command(candidate: str) -> bool:
    """Does this backticked span name something you could RUN at all?

    The no-op shells (`true`, `:`, `echo ...`) are commands, and are recognised
    here DELIBERATELY: `runnable_commands` must reject them for the reason that
    matters -- they cannot fail -- and not merely fail to notice them. A
    classifier that never sees `true` pins nothing about the rule it exists for.
    """
    padded = " " + candidate
    return bool(
        _TOOL.search(padded)
        or _WRAPPER.search(padded)
        or _INVOKED_PATH.search(padded)
        or _CANNOT_FAIL.match(candidate)
    )


def runnable_commands(section: str) -> list[str]:
    """Commands in this section that could actually fail."""
    good = []
    for candidate in _candidates(section):
        if not is_command(candidate):
            continue
        if _CANNOT_FAIL.match(candidate):
            continue
        if "|" in candidate:  # `cmd | tail` reports tail's status
            continue
        good.append(candidate)
    return good


def classify_row(row) -> tuple[str, str]:
    specs = [p for p in record.local_spec_paths(row) if p.is_file()]
    if not specs:
        return "no-spec", "no resolving .agents/specs/ link"
    text = specs[0].read_text(encoding="utf-8", errors="replace")
    section = gates_section(text)
    if section is None:
        return "no-gates-section", specs[0].name
    commands = runnable_commands(section)
    if not commands:
        return "gates-no-command", specs[0].name
    return "runnable", commands[0]


def audit() -> list[dict]:
    records = []
    for path in AUDITED_MATRIX_PATHS:
        errors: list[str] = []
        for row in record.parse_claim_rows(path, errors):
            if row.state not in GATED_STATES:
                continue
            verdict, detail = classify_row(row)
            records.append(
                {
                    "id": row.item_id,
                    "state": row.state,
                    "path": str(row.path.relative_to(ROOT)),
                    "line": row.line_no,
                    "verdict": verdict,
                    "detail": detail,
                }
            )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify gated rows' gate commands.")
    parser.add_argument("--json", action="store_true", help="machine-readable")
    args = parser.parse_args(argv)

    records = audit()
    if args.json:
        print(json.dumps(records, indent=2, sort_keys=True))
        return 0
    counts: dict[str, int] = {}
    for item in records:
        counts[item["verdict"]] = counts.get(item["verdict"], 0) + 1
    for verdict in ("runnable", "gates-no-command", "no-gates-section", "no-spec"):
        print(f"  {counts.get(verdict, 0):4d}  {verdict}")
    print(f"\n{len(records)} gated rows; {counts.get('runnable', 0)} carry a command that can fail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
