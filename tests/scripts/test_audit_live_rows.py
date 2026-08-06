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


class ClassifierTests(unittest.TestCase):
    def test_unmerged_branch_commits_mean_in_flight(self):
        verdict, reason = audit.classify_active(
            branches=["row/ENG-FOO"],
            unmerged_by_branch={"row/ENG-FOO": ["abc1234 wip"]},
            commits=[],
        )
        self.assertEqual(verdict, "IN-FLIGHT")
        self.assertIn("row/ENG-FOO", reason)

    def test_fully_merged_branch_means_landed(self):
        verdict, reason = audit.classify_active(
            branches=["row/ENG-FOO"],
            unmerged_by_branch={"row/ENG-FOO": []},
            commits=[],
        )
        self.assertEqual(verdict, "LANDED")
        # "unmerged" contains "merged", so a substring check on the bare word
        # would pass for the IN-FLIGHT reason too.
        self.assertIn("fully merged", reason)

    def test_main_commits_without_branch_mean_landed(self):
        verdict, reason = audit.classify_active(
            branches=[],
            unmerged_by_branch={},
            commits=["def5678 feat(eng): ENG-FOO"],
        )
        self.assertEqual(verdict, "LANDED")
        self.assertIn("def5678", reason)

    def test_no_evidence_at_all_means_abandoned(self):
        verdict, reason = audit.classify_active(
            branches=[], unmerged_by_branch={}, commits=[]
        )
        self.assertEqual(verdict, "ABANDONED")
        self.assertIn("no branch", reason.lower())

    def test_in_flight_wins_over_landed_when_both_present(self):
        # A row can have landed groundwork AND active follow-up work.
        # Claiming it is finished would silently steal an open claim.
        verdict, _ = audit.classify_active(
            branches=["row/ENG-FOO"],
            unmerged_by_branch={"row/ENG-FOO": ["abc1234 wip"]},
            commits=["def5678 feat(eng): ENG-FOO groundwork"],
        )
        self.assertEqual(verdict, "IN-FLIGHT")

    def test_reason_names_only_the_branches_with_unmerged_commits(self):
        # With more than one branch, the reason must name the live one and not
        # the merged one, and must be order-independent so a re-run does not
        # produce a spuriously different report.
        verdict, reason = audit.classify_active(
            branches=["row/B-LIVE", "row/A-MERGED"],
            unmerged_by_branch={"row/B-LIVE": ["abc1234 wip"], "row/A-MERGED": []},
            commits=[],
        )
        self.assertEqual(verdict, "IN-FLIGHT")
        self.assertIn("row/B-LIVE", reason)
        self.assertNotIn("row/A-MERGED", reason)

    def test_reason_is_order_independent(self):
        # The mixed case above has exactly ONE live branch, so sorted() is a
        # no-op there and deleting it survives. Two branches on the same side
        # of the filter are what pin determinism, on both reason paths: a
        # report that reshuffles its own evidence between runs cannot be
        # diffed by the human who has to act on it.
        for label, by_branch in [
            ("in-flight", {"row/B": ["abc1234 wip"], "row/A": ["def5678 wip"]}),
            ("landed", {"row/B": [], "row/A": []}),
        ]:
            with self.subTest(label):
                forward = audit.classify_active(["row/A", "row/B"], by_branch, [])
                reverse = audit.classify_active(["row/B", "row/A"], by_branch, [])
                self.assertEqual(forward, reverse)
                self.assertIn("row/A, row/B", forward[1])

    def test_missing_branch_key_is_a_loud_caller_bug(self):
        # Silently treating an ungathered branch as merged would report a live
        # claim as finished -- the exact false negative this tool prevents.
        with self.assertRaises(KeyError):
            audit.classify_active(
                branches=["row/NEVER-GATHERED"], unmerged_by_branch={}, commits=[]
            )

    def test_every_verdict_is_declared(self):
        for branches, by_branch, commits in [
            (["row/X"], {"row/X": ["a b"]}, []),
            (["row/X"], {"row/X": []}, []),
            ([], {}, ["a b"]),
            ([], {}, []),
        ]:
            verdict, _ = audit.classify_active(branches, by_branch, commits)
            self.assertIn(verdict, audit.VERDICTS)


class PartialGapTests(unittest.TestCase):
    def test_explicit_gap_language_is_recognised(self):
        for text in [
            "Works for bf16; fp8 is missing",
            "Prefill only, decode not yet ported",
            "Dense path supported, MoE unsupported",
            "Image works; audio pending",
        ]:
            self.assertTrue(audit.names_missing_modes(text), text)

    def test_row_without_gap_language_is_flagged(self):
        self.assertFalse(audit.names_missing_modes("Ported and gated on GB10"))

    def test_detection_is_case_insensitive(self):
        self.assertTrue(audit.names_missing_modes("FP8 IS MISSING"))

    def test_markers_match_whole_words_not_substrings(self):
        # "commonly" contains "only" and "node" contains "no". A substring
        # match would mark these rows explicit and hide them from review.
        self.assertFalse(audit.names_missing_modes("Commonly used decode node"))
        # Both halves need their marker pinned as live, or the assertFalse
        # above passes for the wrong reason: dropping "no" from GAP_MARKERS
        # entirely also stops "node" matching, and nothing would notice.
        self.assertTrue(audit.names_missing_modes("Decode only"))
        self.assertTrue(audit.names_missing_modes("No fp8 path"))

    def test_flag_is_advisory_and_never_gates(self):
        # check mode fails only on abandoned ACTIVE rows, never on a vague
        # PARTIAL row -- the detector is a keyword heuristic.
        self.assertNotIn("PARTIAL", audit.CHECK_FAILS_ON)

    def test_check_fails_on_active_and_nothing_else(self):
        # assertNotIn above passes for frozenset() -- a check mode that fails
        # on NOTHING -- and even for the bare string "ACTIVE", since "PARTIAL"
        # is not a substring of it. Neither is what "report-only" means: the
        # flag must be excluded from a set that still gates something. Pin the
        # membership exactly, and pin that the gated state is a real live one.
        self.assertEqual(audit.CHECK_FAILS_ON, frozenset({"ACTIVE"}))
        self.assertTrue(audit.CHECK_FAILS_ON <= audit.LIVE_STATES)

    def test_matched_marker_names_the_marker_that_fired(self):
        # The heuristic under-flags: 11 of the 48 shipped rows it reads as
        # explicit qualify only via bare "no" or "gap", on prose asserting
        # GOODNESS rather than absence. Naming the hit is what lets a reviewer
        # discount those at a glance instead of trusting the verdict.
        self.assertEqual(
            audit.matched_marker("Works for bf16; fp8 is missing"), "missing"
        )
        self.assertEqual(
            audit.matched_marker("the mirror build no longer double-resides"), "no"
        )
        self.assertEqual(audit.matched_marker("max gap 0.0 nats, 0 divergent"), "gap")
        # The text AS WRITTEN, not the canonical marker, so the reviewer reads
        # the row's own words back.
        self.assertEqual(audit.matched_marker("FP8 IS MISSING"), "MISSING")
        # No hit is "", never None: a vague row must not be reported through
        # the same falsy channel as a row whose marker failed to render.
        self.assertEqual(audit.matched_marker("Ported and gated on GB10"), "")

    def test_a_marker_needing_escaping_is_treated_literally(self):
        # GAP_MARKERS invites human tuning, and an unescaped marker fails two
        # ways. "not.yet" compiles to a wildcard that also matches "notXyet",
        # silently widening the flag...
        literal = audit.gap_pattern(("not.yet",))
        self.assertTrue(literal.search("decode not.yet ported"))
        self.assertIsNone(literal.search("decode notXyet ported"))
        # ...and "fp4(" raises re.error at IMPORT time, taking the whole
        # module -- loader, classifier and all -- down with it.
        audit.gap_pattern(("fp4(",))
        # Escaping must not cost the multi-word widening: re.escape("not yet")
        # is "not\\ yet", and that escaped space is what gets widened.
        self.assertTrue(audit.gap_pattern(("not yet",)).search("decode not  yet"))
        # The shipped regex must be the one this builder returns. Mutation
        # testing shows what this does NOT buy: rebuilding GAP_RE inline
        # WITHOUT re.escape survives, because no shipped marker needs escaping
        # today, so both spellings compile to the identical pattern. It pins
        # the marker set and the structure, not the escaping -- and that
        # unobservability is exactly why gap_pattern takes its markers as an
        # argument instead of closing over GAP_MARKERS.
        self.assertEqual(
            audit.GAP_RE.pattern, audit.gap_pattern(audit.GAP_MARKERS).pattern
        )


if __name__ == "__main__":
    unittest.main()
