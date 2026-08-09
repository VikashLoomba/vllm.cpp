#!/usr/bin/env python3
"""Mutation-bound acceptance checks for the structured-state cutover."""

from __future__ import annotations

import csv
import re
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WAIVER_ID = "WAIVER-PR-SIZE-002"
WAIVER_ROW = {
    "waiver_id": WAIVER_ID,
    "rule_id": "POL-PR-SIZE",
    "scope": "pr:166",
    "owner": "maintainer",
    "reason": "One-time lossless structured-state migration exceeds review budgets through mechanical evidence fan-out",
    "evidence": "PR-166 structured state design",
    "expires": "2026-08-15",
}

STALE_INSTRUCTIONS = (
    re.compile(r"POL-STATE-ORDER"),
    re.compile(r"check-state-order(?:\.py)?"),
    re.compile(r"sort-state-tail(?:\.py)?"),
    re.compile(r"state-order:enforced-below"),
    re.compile(
        r"(?i)append(?:ing)?[^\n]{0,80}(?:\.agents/)?state\.md"
    ),
    re.compile(r"(?i)append-only[^\n]{0,80}(?:state\.md|state (?:log|tail))"),
    re.compile(r"(?i)append-only checkpoint entry"),
    re.compile(r"(?i)modify:[^\n]{0,80}\.agents/state\.md[^\n]{0,80}\.agents/NOW\.md"),
    re.compile(r"(?i)portfolio,\s*state log,\s*NOW"),
)

EXCLUDED_AGENT_RECORDS = frozenset(
    {
        ".agents/benchmark-record.md",
        ".agents/coordination.md",
        ".agents/parity-ledger.md",
        ".agents/state.md",
    }
)
EXCLUDED_TRANSITION_DESIGNS = frozenset(
    {"docs/superpowers/specs/2026-08-08-structured-state-record-design.md"}
)


def active_files(root: Path) -> list[Path]:
    """Return active instructions and projections, excluding forensic records."""
    candidates = [root / "AGENTS.md", root / "README.md"]
    candidates.extend((root / ".agents").glob("*.md"))
    candidates.extend((root / ".agents/specs").glob("*.md"))
    candidates.extend((root / "docs/superpowers/plans").glob("*.md"))
    candidates.extend(
        root / relative
        for relative in (
            ".agents/policy.csv",
            ".github/workflows/ci.yml",
            ".githooks/pre-push",
            "docs/STATUS.md",
            "docs/BENCHMARKS.md",
            "docs/FEATURES.md",
            "docs/USAGE.md",
            "scripts/agent-preflight.sh",
            "scripts/agent-start.py",
            "scripts/check-now-current.py",
            "scripts/policy_contract.py",
        )
    )
    result = []
    for path in candidates:
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in EXCLUDED_AGENT_RECORDS | EXCLUDED_TRANSITION_DESIGNS:
            continue
        result.append(path)
    return sorted(set(result))


def stale_reference_errors(root: Path, paths: list[Path] | None = None) -> list[str]:
    errors: list[str] = []
    for path in active_files(root) if paths is None else paths:
        text = path.read_text(encoding="utf-8")
        for pattern in STALE_INSTRUCTIONS:
            match = pattern.search(text)
            if match is not None:
                errors.append(
                    f"{path.relative_to(root)} retains obsolete state instruction "
                    f"{match.group(0)!r}"
                )
    return errors


class ActiveReferenceAudit(unittest.TestCase):
    def test_execution_plans_are_scanned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plans = root / "docs/superpowers/plans"
            plans.mkdir(parents=True)
            plan = plans / "active-plan.md"
            plan.write_text("Current execution instructions.\n", encoding="utf-8")
            self.assertIn(plan, active_files(root))

    def test_active_references_have_cut_over(self) -> None:
        self.assertEqual(stale_reference_errors(ROOT), [])

    def test_each_obsolete_instruction_shape_is_detected(self) -> None:
        mutations = (
            "Use POL-STATE-ORDER.",
            "Run check-state-order.py.",
            "Repair with sort-state-tail.py.",
            "Write below state-order:enforced-below.",
            "Append one .agents/state.md entry.",
            "Keep the append-only state tail current.",
            "| `.agents/state.md` (modify) | Append-only checkpoint entry |",
            "- Modify: `.agents/state.md`, `.agents/NOW.md`, `docs/STATUS.md`",
            "Portfolio, state log, NOW and STATUS move together.",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "AGENTS.md"
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    target.write_text(mutation, encoding="utf-8")
                    self.assertTrue(stale_reference_errors(root, [target]))

    def test_forensic_payloads_and_transition_design_are_not_rewritten(self) -> None:
        scanned = {path.relative_to(ROOT).as_posix() for path in active_files(ROOT)}
        self.assertFalse(any(path.startswith(".agents/state-events/") for path in scanned))
        self.assertTrue(EXCLUDED_TRANSITION_DESIGNS.isdisjoint(scanned))

    def test_retired_ordering_tools_are_absent(self) -> None:
        for relative in (
            "scripts/check-state-order.py",
            "scripts/sort-state-tail.py",
            "tests/scripts/test_check_state_order.py",
        ):
            with self.subTest(path=relative):
                self.assertFalse((ROOT / relative).exists())


class ExactMigrationWaiver(unittest.TestCase):
    def test_pr_166_has_one_exact_expiring_size_waiver(self) -> None:
        with (ROOT / ".agents/waivers.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        matches = [row for row in rows if row["scope"] == "pr:166"]
        self.assertEqual(matches, [WAIVER_ROW])

    def test_waiver_is_named_by_the_approved_design(self) -> None:
        design = (
            ROOT / "docs/superpowers/specs/2026-08-08-structured-state-record-design.md"
        ).read_text(encoding="utf-8")
        self.assertEqual(design.count(WAIVER_ID), 1)


if __name__ == "__main__":
    unittest.main()
