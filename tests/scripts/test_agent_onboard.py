#!/usr/bin/env python3
"""Unit and mutation checks for scripts/agent-onboard.py.

The probe exists to report what is unresolved. Its one job is to be honest
about absence: a missing .env and an unreadable .env must not look the same as
a complete one, and an undeclared role must never render as a declared one.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import types
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


onboard = _load("agent_onboard", "scripts/agent-onboard.py")


class EnvStateTests(unittest.TestCase):
    def test_env_keys_match_the_tracked_example(self):
        # The probe must never invent a key. .env.example is the only source.
        example = (ROOT / ".env.example").read_text(encoding="utf-8")
        declared = {
            line.split("=", 1)[0]
            for line in example.splitlines()
            if line and not line.startswith("#") and "=" in line
        }
        self.assertEqual(set(onboard.ENV_KEYS), declared)

    def test_missing_file_reports_missing_not_incomplete(self):
        status, missing = onboard.env_state(ROOT / "does-not-exist-.env")
        self.assertEqual(status, "missing")
        self.assertEqual(sorted(missing), sorted(onboard.ENV_KEYS))

    def test_blank_value_counts_as_missing_that_key(self):
        # An empty value is a legitimate "unavailable", but the probe still
        # has to report it so the agent knows what it may ask for.
        text = "\n".join(f"{k}=" for k in onboard.ENV_KEYS)
        status, missing = onboard.env_state_from_text(text)
        self.assertEqual(status, "incomplete")
        self.assertEqual(sorted(missing), sorted(onboard.ENV_KEYS))

    def test_all_values_present_reports_present(self):
        text = "\n".join(f"{k}=/some/path" for k in onboard.ENV_KEYS)
        status, missing = onboard.env_state_from_text(text)
        self.assertEqual(status, "present")
        self.assertEqual(missing, [])

    def test_unreadable_file_is_neither_present_nor_absent(self):
        # Beyond the brief. A .env that exists but cannot be read must not
        # crash the probe, must not read as complete, and must not read as
        # absent either: "create one" is the wrong instruction for a file that
        # is already there.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.mkdir()  # exists(), but read_text() raises IsADirectoryError
            status, missing = onboard.env_state(path)
        self.assertEqual(status, "unreadable")
        self.assertEqual(sorted(missing), sorted(onboard.ENV_KEYS))


class ProbeRenderTests(unittest.TestCase):
    UNDECLARED = {
        "role": None, "row": None, "mode": "interactive",
        "env": "missing", "env_missing": ["VLLM_ORACLE"], "queue": ["ENG-FOO"],
    }

    def test_undeclared_role_renders_as_undeclared(self):
        out = onboard.render_probe(self.UNDECLARED)
        self.assertIn("UNDECLARED", out)
        self.assertNotIn("operator", out.split("queue")[0])

    def test_declared_role_renders_with_its_row(self):
        # The row id must NOT be one the fixture queue already contains, or the
        # queue line satisfies the assertion and deleting row rendering stays
        # green. Assert the `row=` prefix, not the bare id.
        out = onboard.render_probe(dict(self.UNDECLARED, role="helper", row="KERNEL-BAR"))
        self.assertIn("helper", out)
        self.assertIn("row=KERNEL-BAR", out)

    def test_undeclared_render_carries_the_interview_hint(self):
        # The hint is the whole point of the probe: without it an agent sees a
        # state line and no instruction. Deleting the block must go red.
        out = onboard.render_probe(self.UNDECLARED)
        self.assertIn("claim", out)
        self.assertIn("read-only", out)
        self.assertNotIn("claim", onboard.render_probe(
            dict(self.UNDECLARED, role="helper", row="KERNEL-BAR")))
        # Added: an absent mode is an absence too. resolve() has no mode until
        # step 2, so it must render as a default and never as a declaration.
        # Assert the whole mode field: the hint line already contains "not
        # declared", so a looser assertion would pass on a fabricated mode.
        undeclared_mode = {k: v for k, v in self.UNDECLARED.items() if k != "mode"}
        self.assertIn(
            "mode: interactive (default, not declared)",
            onboard.render_probe(undeclared_mode),
        )

    def test_lockout_by_another_operator_is_not_rendered_as_never_declared(self):
        # Beyond the brief. resolve() knows the difference; a session locked
        # out by a live operator must not be told to claim operator, because
        # that claim will fail. Both are role=None, so only the NOTE separates
        # them.
        blocked = dict(self.UNDECLARED, blocked_by_other_operator=True)
        out = onboard.render_probe(blocked)
        self.assertIn("NOTE", out)
        self.assertIn("held by another live session", out)
        self.assertNotIn("NOTE", onboard.render_probe(self.UNDECLARED))

    def test_probe_never_exits_nonzero(self):
        # The probe reports; it does not gate. Preflight gates.
        self.assertEqual(onboard.main(["--probe"]), 0)


class QueueTests(unittest.TestCase):
    def test_queue_is_the_checkers_own_computation_and_failure_is_visible(self):
        # Beyond the brief, two properties of the same function.
        #
        # 1. The queue must come from ready-for-helper.py's queue(), not from a
        #    re-parse of its prose: an uppercase-token filter over that stdout
        #    reads the "READY-FOR-HELPER queue: N row(s)" header as a row (so an
        #    EMPTY queue reports one phantom row) and drops every mixed-case row
        #    id such as MODEL-TEXT-glm4-glm4-for-causal-lm.
        # 2. A queue that could not be computed is not an empty queue. Swallowing
        #    the failure into [] is the queue-side twin of reporting an unreadable
        #    .env as a complete one.
        rows = onboard.ready_rows()
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/ready-for-helper.py")],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        header = result.stdout.splitlines()[0]
        self.assertIn("READY-FOR-HELPER queue:", header)
        self.assertEqual(len(rows), int(header.split(":")[1].split()[0]))
        self.assertNotIn("READY-FOR-HELPER", rows)

        def explode():
            raise RuntimeError("record is broken")

        saved = sys.modules.get("ready_for_helper")
        sys.modules["ready_for_helper"] = types.SimpleNamespace(queue=explode)
        try:
            broken_rows, error = onboard.queue_state()
        finally:
            if saved is None:
                del sys.modules["ready_for_helper"]
            else:
                sys.modules["ready_for_helper"] = saved
        self.assertEqual(broken_rows, [])
        self.assertIn("record is broken", error)
        self.assertIn("UNAVAILABLE", onboard.render_probe(
            dict(ProbeRenderTests.UNDECLARED, queue=[], queue_error=error)))


if __name__ == "__main__":
    unittest.main()
