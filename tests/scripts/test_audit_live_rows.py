#!/usr/bin/env python3
"""Unit and mutation checks for scripts/audit-live-rows.py.

The audit only helps if it is honest in both directions: it must not call a
live row abandoned when work is really in flight, and it must not call an
abandoned row live because a branch name happens to exist.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


audit = _load("audit_live_rows", "scripts/audit-live-rows.py")


class LiveRowLoadingTests(unittest.TestCase):
    def test_live_states_are_exactly_the_six(self):
        self.assertEqual(
            audit.LIVE_STATES,
            frozenset({"SPIKE", "READY", "ACTIVE", "GATING", "PARTIAL", "BLOCKED"}),
        )

    def test_all_seven_matrices_are_audited(self):
        names = {path.name for path in audit.AUDIT_MATRIX_PATHS}
        self.assertIn("feature-matrix.md", names)
        self.assertIn("sglang-matrix.md", names)
        self.assertEqual(len(names), 7)

    def test_newly_covered_feature_matrix_actually_yields_live_rows(self):
        # Asserting the path is in a list proves nothing: every other assertion
        # here still passes if feature-matrix.md contributes zero rows. This is
        # what the seventh-matrix commit actually bought.
        rows = audit.live_rows()
        self.assertTrue([r for r in rows if r.path.name == "feature-matrix.md"])

    def test_shipped_record_parses_without_errors(self):
        # parse_claim_rows DROPS a row it cannot parse. If a malformed row ever
        # lands, the census silently shrinks -- so the sink must be surfaced,
        # and it must be empty today.
        errors: list[str] = []
        audit.live_rows(errors)
        self.assertEqual(errors, [])

    def test_shipped_matrices_yield_only_live_rows(self):
        rows = audit.live_rows()
        self.assertTrue(rows, "the shipped matrices must contain live rows")
        for row in rows:
            self.assertIn(row.state, audit.LIVE_STATES)

    def test_every_live_state_is_represented_in_the_shipped_matrices(self):
        # Guards the loader against silently dropping a whole state: if a
        # header rename made one state unparseable, its count would go to
        # zero while the other five still looked healthy.
        rows = audit.live_rows()
        present = {row.state for row in rows}
        self.assertEqual(present, set(audit.LIVE_STATES))
        self.assertGreater(len(rows), 100, "the live set is ~188 rows")


class IdGrepPatternTests(unittest.TestCase):
    """The ID match must be a whole-token match, never a prefix match."""

    def test_pattern_matches_the_id_as_a_whole_token(self):
        pattern = re.compile(audit.id_grep_pattern("MODEL-MM"))
        for message in (
            "MODEL-MM",
            "feat(mm): MODEL-MM decoder lands",
            "closes MODEL-MM.",
            "(MODEL-MM) golden captured",
        ):
            self.assertTrue(pattern.search(message), message)

    def test_pattern_rejects_a_longer_id_that_merely_starts_with_it(self):
        # 55 pairs of live row IDs are prefixes of longer ones. A substring
        # match would credit MODEL-MM with MODEL-MM-voxtral's commits, and the
        # classifier calls any commit LANDED -- so an abandoned row would
        # report as finished, the exact false negative this tool prevents.
        pattern = re.compile(audit.id_grep_pattern("MODEL-MM"))
        for message in (
            "feat(mm): MODEL-MM-voxtral audio tower lands",
            "MODEL-MM-QWEN3VL golden captured",
            "record(mm): MODEL-MM_SUFFIX bookkeeping",
        ):
            self.assertIsNone(pattern.search(message), message)


class CommandLineGuardTests(unittest.TestCase):
    def test_check_flag_does_not_silently_exit_zero(self):
        # The file is executable and its docstring advertises --check, but the
        # real CLI arrives in P0 step 4. Until then --check must NOT exit 0:
        # a gate that reports success because it ignored its own flag is the
        # worst possible answer.
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/audit-live-rows.py"), "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
