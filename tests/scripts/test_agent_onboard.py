#!/usr/bin/env python3
"""Unit and mutation checks for scripts/agent-onboard.py.

The probe exists to report what is unresolved. Its one job is to be honest
about absence: a missing .env and an unreadable .env must not look the same as
a complete one, and an undeclared role must never render as a declared one.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import shutil
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
        # The render itself goes to a buffer only so the suite's output stays
        # clean; the assertion is unchanged.
        with contextlib.redirect_stdout(io.StringIO()):
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


class PreflightWiringTests(unittest.TestCase):
    TEXT = (ROOT / "scripts/agent-preflight.sh").read_text(encoding="utf-8")

    def test_require_role_defaults_on(self):
        # Anchored to the DEFAULT assignment itself -- the line with no
        # indentation and nothing else on it. A bare assertIn("REQUIRE_ROLE=1")
        # is satisfied by the --require-role arm of the arg loop all by itself,
        # so the default could be flipped back and the whole deliverable of this
        # change would go unprotected. Any line-anchored assignment of zero is
        # refused, quoted or not, because that is what a silent revert looks
        # like however it is spelled.
        self.assertRegex(self.TEXT, r"(?m)^REQUIRE_ROLE=1$")
        self.assertNotRegex(self.TEXT, r"""(?m)^REQUIRE_ROLE=['"]?0['"]?$""")

    def test_opt_out_flag_exists(self):
        self.assertIn("--no-require-role", self.TEXT)

    def test_failure_text_carries_the_interview(self):
        # An error code alone gets routed around. The gate must say what to ask.
        self.assertIn("claim read-only", self.TEXT)
        self.assertIn("claim helper --row", self.TEXT)

    def test_staged_refuses_read_only(self):
        self.assertIn("read-only", self.TEXT)
        self.assertIn("STAGED", self.TEXT)

    def test_the_gate_records_a_failure_and_not_only_a_print(self):
        # Every other assertion in this class inspects the text that EXPLAINS
        # the gate, so deleting the one line that enforces it -- the failed+=()
        # inside the REQUIRE_ROLE branch -- leaves them all green while
        # preflight exits 0 on an undeclared role. Pin the enforcing line, and
        # pin that it is inside the branch: outside it, --no-require-role would
        # stop working instead.
        branch = re.search(
            r'if \[ "\$REQUIRE_ROLE" -eq 1 \]; then(.*?)\n  fi', self.TEXT, re.S)
        self.assertIsNotNone(branch, "the REQUIRE_ROLE branch is gone")
        self.assertIn('failed+=("role-undeclared")', branch.group(1))

    def test_onboard_suite_is_registered(self):
        self.assertIn("test_agent_onboard", self.TEXT)

    def test_read_only_alone_does_not_satisfy_a_write_gate(self):
        # agent-role.py show exits 0 for read-only, so --require-role is
        # satisfied by a declared ABSENCE of claim. That is correct for a plain
        # run and wrong for --staged; the refusal must be explicit.
        self.assertIn("read-only-cannot-stage", self.TEXT)


class EnvSetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.env = self.tmp / ".env"
        self._real = onboard.ENV_FILE
        onboard.ENV_FILE = self.env
        # cmd_env_set reports on stdout and refuses on stderr. Both go to a
        # buffer so the suite's own output stays clean -- but the buffers are
        # KEPT and asserted on, because a refusal that returns 2 with an empty
        # explanation would otherwise pass every test in this class.
        self.out, self.err = io.StringIO(), io.StringIO()
        stack = contextlib.ExitStack()
        stack.enter_context(contextlib.redirect_stdout(self.out))
        stack.enter_context(contextlib.redirect_stderr(self.err))
        self.addCleanup(stack.close)

    def tearDown(self):
        onboard.ENV_FILE = self._real
        shutil.rmtree(self.tmp)

    def test_unknown_key_is_refused(self):
        # Never invent a key: a typo'd name would sit in .env doing nothing
        # while the gate that wanted it stays mysteriously PENDING.
        self.assertEqual(onboard.cmd_env_set("NOT_A_REAL_KEY=/x"), 2)
        self.assertFalse(self.env.exists())
        # An exit code alone gets routed around, and every refusal here is a
        # human's typo. Name the offending key and the legal ones.
        self.assertIn("NOT_A_REAL_KEY", self.err.getvalue())
        self.assertIn(onboard.ENV_KEYS[0], self.err.getvalue())

    def test_missing_pair_is_refused(self):
        self.assertEqual(onboard.cmd_env_set("VLLM_ORACLE"), 2)
        self.assertFalse(self.env.exists())
        self.assertIn("KEY=VALUE", self.err.getvalue())

    def test_first_write_seeds_from_the_example(self):
        self.assertEqual(onboard.cmd_env_set(f"{onboard.ENV_KEYS[0]}=/oracle"), 0)
        text = self.env.read_text(encoding="utf-8")
        self.assertIn(f"{onboard.ENV_KEYS[0]}=/oracle", text)
        # every other declared key survives, so nothing is silently dropped
        for key in onboard.ENV_KEYS:
            self.assertIn(key, text)

    def test_second_write_updates_in_place_without_duplicating(self):
        key = onboard.ENV_KEYS[0]
        onboard.cmd_env_set(f"{key}=/first")
        onboard.cmd_env_set(f"{key}=/second")
        text = self.env.read_text(encoding="utf-8")
        self.assertIn(f"{key}=/second", text)
        self.assertNotIn("/first", text)
        self.assertEqual(sum(1 for l in text.splitlines() if l.startswith(f"{key}=")), 1)

    def test_other_keys_are_not_disturbed(self):
        a, b = onboard.ENV_KEYS[0], onboard.ENV_KEYS[1]
        onboard.cmd_env_set(f"{a}=/aaa")
        onboard.cmd_env_set(f"{b}=/bbb")
        text = self.env.read_text(encoding="utf-8")
        self.assertIn(f"{a}=/aaa", text)
        self.assertIn(f"{b}=/bbb", text)

    def test_the_flag_is_wired_into_main(self):
        # Beyond the brief. Every test above calls cmd_env_set() directly, so
        # deleting --env-set or its dispatch in main() leaves all five green
        # while the only entry point an agent actually types either dies in
        # argparse or silently prints a probe and records nothing.
        key = onboard.ENV_KEYS[0]
        self.assertEqual(onboard.main(["--env-set", f"{key}=/wired"]), 0)
        self.assertIn(f"{key}=/wired", self.env.read_text(encoding="utf-8"))
        self.assertEqual(onboard.main(["--env-set", "NOT_A_REAL_KEY=/x"]), 2)
        # An empty argument is a malformed write, not "no flag given": a
        # truthiness dispatch falls through to the probe and exits 0 having
        # recorded nothing, which is the silent no-op this command must not do.
        self.assertEqual(onboard.main(["--env-set", ""]), 2)

    def test_an_empty_value_is_accepted_and_still_reads_as_unset(self):
        # Beyond the brief, and the rule the whole command serves: unanswered
        # means EMPTY, and empty means the gates that need it stay PENDING.
        # A writer that refused an empty value would push its caller toward
        # inventing one from a path, a username or another developer's setup,
        # which is the failure this spec exists to prevent. Clearing a value
        # must also be possible, or a wrong answer is unretractable.
        key = onboard.ENV_KEYS[0]
        self.assertEqual(onboard.cmd_env_set(f"{key}=/somewhere"), 0)
        self.assertEqual(onboard.cmd_env_set(f"{key}="), 0)
        text = self.env.read_text(encoding="utf-8")
        self.assertIn(f"\n{key}=\n", f"\n{text}")
        self.assertIn(key, onboard.env_state_from_text(text)[1])

    def test_no_line_separator_in_a_value_can_forge_a_second_line(self):
        # Beyond the brief. The value is whatever the interview answer was, and
        # it is the one field no check looks at. A separator inside it appends a
        # whole extra .env line that no key check ever saw -- the silent clobber
        # this task exists to rule out, arriving through the unvalidated field.
        #
        # EVERY separator, not just "\n": str.splitlines() is what the reader
        # and the rewrite both use, and it breaks on ten. Guarding two of them
        # let a "\v" or a U+2028 smuggle a forged pair past the key check, after
        # which the probe reported the forged key as SET. This list IS the
        # separator set; a guard that enumerates characters again fails here.
        key, victim = onboard.ENV_KEYS[0], onboard.ENV_KEYS[3]
        for separator in ("\n", "\r", "\r\n", "\v", "\f",
                          "\x1c", "\x1d", "\x1e", "\x85", " ", " "):
            with self.subTest(separator=repr(separator)):
                pair = f"{key}=/ok{separator}{victim}=/forged"
                self.assertEqual(onboard.cmd_env_set(pair), 2)
                self.assertFalse(self.env.exists())
        # The property, stated the way the guard states it: what is written
        # must survive the round trip through the reader that parses it back.
        self.assertEqual(onboard.cmd_env_set(f"{key}=/ok {victim}=/forged"), 2)
        self.assertIn("line separator", self.err.getvalue())

    def test_a_trailing_duplicate_key_cannot_survive_the_write(self):
        # Beyond the brief. A hand-maintained .env routinely carries an override
        # appended at the bottom, and both this reader and the documented
        # `set -a; . ./.env` loader take the LAST assignment. Updating only the
        # first match left the file changed, the exit code 0 and the message
        # reassuring while the EFFECTIVE value never moved -- the silent no-op
        # this command exists to rule out, arriving from the other side.
        key, other = onboard.ENV_KEYS[0], onboard.ENV_KEYS[5]
        self.env.write_text(
            f"{key}=/one\n{other}=h\n{key}=/override\n", encoding="utf-8")
        self.assertEqual(onboard.cmd_env_set(f"{key}=/new"), 0)
        lines = self.env.read_text(encoding="utf-8").splitlines()
        effective = [l.split("=", 1)[1] for l in lines if l.startswith(f"{key}=")][-1]
        self.assertEqual(effective, "/new")
        self.assertEqual(sum(1 for l in lines if l.startswith(f"{key}=")), 1)
        self.assertIn(f"{other}=h", lines)  # the unrelated line keeps its place

    def test_an_unreadable_env_is_refused_and_not_a_traceback(self):
        # Beyond the brief. env_state treats an existing-but-unreadable .env as
        # its own third case; the WRITER has to agree, or the one command an
        # agent is told to run dies in a bare traceback that names no fix.
        self.env.mkdir()  # exists(), but read_text() raises IsADirectoryError
        self.assertEqual(onboard.cmd_env_set(f"{onboard.ENV_KEYS[0]}=/x"), 2)
        self.assertIn("cannot be read", self.err.getvalue())

    def test_a_value_needing_quotes_reads_the_same_to_both_readers(self):
        # Beyond the brief, same family as the separator finding: .env.example
        # documents the loader as `set -a; . ./.env`, so the file is shell as
        # well as data. An unquoted "/pa th" makes that loader run `th` and
        # leave the variable EMPTY while this probe's parser reports it PRESENT
        # -- a gate that stays PENDING with the surface insisting it is set.
        key = onboard.ENV_KEYS[0]
        self.assertEqual(onboard.cmd_env_set(f"{key}=/pa th"), 0)
        text = self.env.read_text(encoding="utf-8")
        self.assertNotIn(key, onboard.env_state_from_text(text)[1])  # probe: set
        sourced = subprocess.run(
            ["sh", "-c", f'set -a; . "$1"; set +a; printf %s "${key}"', "sh",
             str(self.env)],
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(sourced.stdout, "/pa th")  # loader: the SAME value
        self.assertEqual(sourced.stderr, "")
        # and an empty value must still round-trip as UNSET, not as the literal
        # two-character '' that shlex.quote would otherwise produce.
        self.assertEqual(onboard.cmd_env_set(f"{key}="), 0)
        self.assertIn(key, onboard.env_state_from_text(
            self.env.read_text(encoding="utf-8"))[1])


if __name__ == "__main__":
    unittest.main()
