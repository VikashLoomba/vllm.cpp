#!/usr/bin/env python3
"""Audit the live-state matrix rows against Git reality. (P0)

49 rows claim ACTIVE at once, which cannot be true: a stale ACTIVE cell inside
a several-hundred-row table is invisible rot. This tool makes it visible.

It PROPOSES and REPORTS. It never rewrites a matrix -- corrections are applied
by a human/agent in reviewable per-matrix commits, because a state transition
carries contract obligations (a spec link, evidence anchors) that only a reader
of the row can satisfy.

Row parsing is imported from scripts/check-agent-record.py rather than
reimplemented, so the audit and the gate can never disagree about what a row is.

    scripts/audit-live-rows.py                 # markdown report
    scripts/audit-live-rows.py --json          # machine-readable
    scripts/audit-live-rows.py --check         # exit 1 if an ACTIVE row is abandoned
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
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

LIVE_STATES = frozenset({"SPIKE", "READY", "ACTIVE", "GATING", "PARTIAL", "BLOCKED"})


def live_rows() -> list:
    """Every row in the shipped matrices whose state is in LIVE_STATES."""
    rows = []
    for path in record.MATRIX_PATHS:
        errors: list[str] = []
        for row in record.parse_claim_rows(path, errors):
            if row.state in LIVE_STATES:
                rows.append(row)
    return rows


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout if result.returncode == 0 else ""


def row_branches() -> dict[str, list[str]]:
    """Map row ID -> every local or remote branch named row/<ID>."""
    mapping: dict[str, list[str]] = {}
    out = git("for-each-ref", "--format=%(refname:short)", "refs/heads", "refs/remotes")
    for line in out.splitlines():
        name = line.strip()
        if name.startswith("row/"):
            item_id = name[len("row/") :]
        elif "/row/" in name:
            item_id = name.split("/row/", 1)[1]
        else:
            continue
        mapping.setdefault(item_id, []).append(name)
    return mapping


def main_commits(item_id: str) -> list[str]:
    """Commits on origin/main whose message mentions the row ID literally."""
    out = git(
        "log", "--oneline", "--fixed-strings", f"--grep={item_id}", "-n", "20",
        "origin/main",
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def unmerged(branch: str) -> list[str]:
    """Commits on branch that are not yet on origin/main."""
    out = git("log", "--oneline", f"origin/main..{branch}")
    return [line.strip() for line in out.splitlines() if line.strip()]
