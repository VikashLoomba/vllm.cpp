#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-gate-commands.py.

A gate command that cannot fail collapses "done" into the implementer's opinion
of its own work. This classifier's only job is to tell a runnable command from
prose that looks like one.
"""

from __future__ import annotations

import importlib.util
import re
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


gates = _load("check_gate_commands", "scripts/check-gate-commands.py")


class GatesSectionTests(unittest.TestCase):
    def test_finds_the_gates_heading_at_any_level(self):
        for heading in ("## Gates", "### Gates", "#### Gates and evidence"):
            text = f"# Spec\n\nintro\n\n{heading}\n\nrun `ctest -R foo`\n\n## Next\n\ntail\n"
            section = gates.gates_section(text)
            self.assertIsNotNone(section, heading)
            self.assertIn("ctest", section)
            self.assertNotIn("tail", section, "must stop at the next heading")

    def test_returns_none_when_there_is_no_gates_section(self):
        self.assertIsNone(gates.gates_section("# Spec\n\n## Scope\n\nnothing here\n"))

    def test_is_not_fooled_by_the_word_gates_in_prose(self):
        # "the gates are green" is not a section heading.
        self.assertIsNone(gates.gates_section("# Spec\n\nAll the gates are green.\n"))


class RunnableCommandTests(unittest.TestCase):
    def test_recognises_a_real_command(self):
        for body in [
            "run `ctest -R test_foo`",
            "`python3 scripts/check-agent-record.py`",
            "```\nbash scripts/agent-preflight.sh\n```",
            "`cmake --build build -j`",
            # The repo's MANDATED shape for a gate that touches the GPU. The
            # wrapper quotes the real command, so nothing else can reach it.
            "`flock /tmp/gpu -c 'ctest -R qwen36_paged_engine'`",
            # A built test binary, invoked with no arguments at all.
            "`./build-cuda-121a/tests/test_dropin_abi`",
        ]:
            self.assertTrue(gates.runnable_commands(body), body)

    def test_rejects_prose_that_merely_mentions_gating(self):
        for body in [
            "Correctness, e2e and performance gates apply.",
            "The SACRED gate must pass on GB10.",
            "`docs/BENCHMARKS.md`",
            # Every one of these is on the shipped record and was credited as a
            # runnable command. A backticked FILENAME is not a command, and a
            # tool name must be a whole word: `sha256_cbor` is not `sh`, and
            # `python@3.14` is not `python`.
            "`tests/vllm/models/test_model_registry.cpp`",
            "`tests/`",
            "`sha256_cbor`",
            "`python@3.14`",
            # The GPU lock IDIOM, named in three specs. A wrapper with nothing
            # to run is not a gate; a plain `flock` vocabulary entry credits
            # all three of those rows with a command they do not have.
            "`flock`",
            "`flock /tmp/gpu`",
        ]:
            self.assertEqual(gates.runnable_commands(body), [], body)
        # ...without rejecting a path that really is invoked.
        self.assertTrue(gates.runnable_commands("`scripts/check-agent-record.py`"))
        self.assertTrue(gates.runnable_commands("`./build/vllm-cli --model x`"))

    def test_rejects_a_command_that_cannot_fail(self):
        # These are the exact shapes the spec forbids: a Verify that always
        # succeeds turns "done" into an opinion.
        for body in ["`true`", "`echo ok`", "`:`", "`echo done && true`"]:
            self.assertEqual(gates.runnable_commands(body), [], body)
        # And they are rejected for the RIGHT reason. Without this, the loop
        # above passes vacuously on any implementation that simply fails to
        # recognise `true` as a command at all, and the cannot-fail rule --
        # the point of this classifier -- is pinned by nothing.
        for candidate in ("true", "echo ok", ":", "echo done && true"):
            self.assertTrue(gates.is_command(candidate), candidate)

    def test_rejects_a_piped_command(self):
        # `cmd | tail` reports tail's exit status, so the gate cannot fail.
        self.assertEqual(gates.runnable_commands("`ctest -R foo | tail -5`"), [])


class ShippedRecordTests(unittest.TestCase):
    def test_the_audit_covers_every_gated_state(self):
        self.assertEqual(
            gates.GATED_STATES,
            frozenset({"READY", "ACTIVE", "GATING", "DONE", "BLOCKED"}),
        )
        # ...and audit() must actually FILTER on it. Asserting the constant's
        # literal value pins nothing about the denominator: deleting the state
        # filter in audit() leaves every other assertion in this file green
        # while the report goes from 97 rows to 726. Task 4 ratchets on that
        # number, so it is pinned here.
        audited = gates.audit()
        self.assertTrue(audited)
        self.assertLessEqual({item["state"] for item in audited}, gates.GATED_STATES)
        # And the filter is only load-bearing if the matrices really do carry
        # rows it excludes -- otherwise the assertion above is vacuous.
        on_record = set()
        for path in gates.AUDITED_MATRIX_PATHS:
            for row in gates.record.parse_claim_rows(path, []):
                on_record.add(row.state)
        self.assertTrue(on_record - gates.GATED_STATES, "filter excludes nothing")

    def test_all_seven_matrices_are_audited(self):
        names = {p.name for p in gates.AUDITED_MATRIX_PATHS}
        self.assertIn("feature-matrix.md", names)
        self.assertIn("sglang-matrix.md", names)
        self.assertEqual(len(names), 7)
        # The LIST length too, not just the set of names. If check-agent-record's
        # MATRIX_PATHS ever gains one of the two appended here, audit() parses
        # that file twice and double-counts every row in it -- the denominator
        # moving silently again, which a set comparison reads as still 7.
        self.assertEqual(len(gates.AUDITED_MATRIX_PATHS), 7)

    def test_every_record_carries_a_known_verdict(self):
        known = {"runnable", "gates-no-command", "no-gates-section", "no-spec"}
        records = gates.audit()
        self.assertTrue(records)
        for item in records:
            self.assertIn(item["verdict"], known)

    def test_report_mode_exits_zero_even_with_debt(self):
        # 67 of 97 rows cannot state a command today. Report mode must still
        # exit 0 -- the ratchet is step 4, after the debt is recorded.
        self.assertEqual(gates.main([]), 0)


def _bash_array(text: str, name: str) -> list[str]:
    """The entries of a `NAME=(\n ... \n)` array in a bash script.

    Membership in the ARRAY, never a substring of the whole file: preflight
    mentions `check-gate-commands` twice -- once in `CHECKERS`, once in the
    `case` branch that adds `--check` -- so a substring test stays green with
    the CHECKERS entry deleted. That is this repo's recorded defect class (a
    substring `--grep` crediting a row with another row's commits), and it
    would hide the exact mutation step 6 requires to go red.
    """
    match = re.search(rf"(?m)^{re.escape(name)}=\($(.*?)^\)$", text, re.S)
    assert match is not None, f"{name}=( ... ) not found"
    return [line.strip() for line in match.group(1).splitlines() if line.strip()]


class RatchetTests(unittest.TestCase):
    def test_the_baseline_matches_the_shipped_record(self):
        runnable = {r["id"] for r in gates.audit() if r["verdict"] == "runnable"}
        self.assertEqual(runnable, set(gates.RUNNABLE_BASELINE))

    def test_a_row_that_loses_its_command_is_refused(self):
        # Still present, still gated, no longer runnable -- a real regression.
        victim = sorted(gates.RUNNABLE_BASELINE)[0]
        records = [{"verdict": "gates-no-command", "id": victim, "state": "READY",
                    "path": "p", "line": 1, "detail": "d"}]
        errors = gates.ratchet_errors(records)
        self.assertTrue(errors)
        self.assertIn(victim, errors[0])
        self.assertIn("Repair the row", errors[0])

    def test_a_row_that_left_the_population_reports_differently(self):
        # Deleted or transitioned out: legitimate, but must re-pin. The two
        # cases MUST be distinguishable -- that is why the baseline is a set.
        errors = gates.ratchet_errors([])
        self.assertTrue(errors)
        self.assertTrue(any("left the gated population" in e for e in errors))
        self.assertFalse(any("Repair the row" in e for e in errors))

    def test_a_lost_row_and_a_departed_row_are_reported_separately(self):
        # The two cases in ONE run, which is the only arrangement that proves
        # they are distinguishable rather than merely differently worded. The
        # two tests above each see a single case, so both stay green if the
        # `& present` split is deleted and every drop is called a loss; here
        # the departed row would then be named in the "Repair the row" message
        # and the assertion below goes red.
        ordered = sorted(gates.RUNNABLE_BASELINE)
        self.assertGreaterEqual(len(ordered), 2)
        lost, departed = ordered[0], ordered[1]
        records = [{"verdict": "gates-no-command", "id": lost, "state": "READY",
                    "path": "p", "line": 1, "detail": "d"}]
        records += [
            {"verdict": "runnable", "id": rid, "state": "READY",
             "path": "p", "line": 2, "detail": "d"}
            for rid in ordered[2:]
        ]
        errors = gates.ratchet_errors(records)
        self.assertEqual(len(errors), 2, errors)
        loss_msg = [e for e in errors if "Repair the row" in e]
        gone_msg = [e for e in errors if "left the gated population" in e]
        self.assertEqual(len(loss_msg), 1, errors)
        self.assertEqual(len(gone_msg), 1, errors)
        self.assertIn(lost, loss_msg[0])
        self.assertNotIn(departed, loss_msg[0])
        self.assertIn(departed, gone_msg[0])
        self.assertNotIn(lost, gone_msg[0])

    def test_an_improvement_is_allowed(self):
        records = [
            {"verdict": "runnable", "id": rid, "state": "READY",
             "path": "p", "line": 1, "detail": "d"}
            for rid in sorted(gates.RUNNABLE_BASELINE)
        ] + [{"verdict": "runnable", "id": "NEW-ROW", "state": "READY",
              "path": "p", "line": 2, "detail": "d"}]
        self.assertEqual(gates.ratchet_errors(records), [])

    def test_check_mode_passes_on_the_shipped_record(self):
        # The gate ships GREEN. It was wired after the debt was recorded, so it
        # never had to be relaxed to pass.
        self.assertEqual(gates.main(["--check"]), 0)

    def test_check_mode_is_never_silently_swallowed_by_json(self):
        # --json used to be the only mode, and it returns 0 unconditionally. If
        # it short-circuits ahead of --check, `--json --check` is a gate that
        # CANNOT FAIL -- the precise shape this classifier exists to detect,
        # wearing this classifier's own face.
        self.assertEqual(gates.main(["--json", "--check"]), 0)
        original = gates.RUNNABLE_BASELINE
        try:
            gates.RUNNABLE_BASELINE = frozenset(original | {"ROW-THAT-IS-NOT-THERE"})
            self.assertEqual(gates.main(["--check"]), 1)
            self.assertEqual(gates.main(["--json", "--check"]), 1)
        finally:
            gates.RUNNABLE_BASELINE = original

    def test_the_checker_is_wired_into_preflight_and_ci(self):
        preflight = (ROOT / "scripts/agent-preflight.sh").read_text(encoding="utf-8")
        self.assertIn("check-gate-commands", _bash_array(preflight, "CHECKERS"))
        self.assertIn("test_check_gate_commands", _bash_array(preflight, "SUITES"))
        # ...and dispatched WITH --check. Report mode exits 0 whatever the
        # record says, so a CHECKERS entry without the flag installs a gate
        # that cannot fail.
        self.assertRegex(
            preflight, r"(?m)^\s*[\w|-]*check-gate-commands\)\s+run\b.*--check"
        )
        ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn("scripts/check-gate-commands.py --check", ci)
        self.assertIn("tests/scripts/test_check_gate_commands.py", ci)


if __name__ == "__main__":
    unittest.main()
