#!/usr/bin/env python3
"""Unit and mutation checks for scripts/audit-live-rows.py.

The audit only helps if it is honest in both directions: it must not call a
live row abandoned when work is really in flight, and it must not call an
abandoned row live because a branch name happens to exist.
"""

from __future__ import annotations

import importlib.util
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
        self.assertGreater(len(rows), 100, "the live set is ~160 rows")


if __name__ == "__main__":
    unittest.main()
