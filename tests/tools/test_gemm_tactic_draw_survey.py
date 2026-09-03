"""The draw survey's judgement, gated on the CPU.

WHY THESE ARE THE TESTS
-----------------------
The harness this covers answers #2750, #2751 and #2752 from ONE leased run, and
every way it can be wrong is a way in which the run still LOOKS finished:

* a draw phase that logged zero `[VT_GEMM_ALGO]` lines produces a perfectly
  consistent, perfectly empty stability report that reads as `STABLE` to anyone
  who does not count the keys;
* a "draw" that LOADED a plan map rather than tuning one is a copy of an earlier
  draw wearing a new label, and N copies of one draw report as N draws;
* a scoring leg that re-tuned measured a plan map nobody recorded, and its
  number is then attributed to the arm it was asked about;
* a `[[:space:]]` metric regex matches nothing in Python, so every leg records
  VOID while the runner exits 0.

So the assertions below are mostly NEGATIVE: each one deletes or inverts a
guarantee and requires the verdict to change. A predicate whose mutation leaves
the verdict green measures nothing.

The module under test is standard library only and has no device, which is why
the whole battery runs here rather than under a lease.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import pathlib
import tempfile
import unittest

from tools.bench.gemm_tactic_draw_survey import (
    EXIT_ALGO_KEYSET_DIFFERS,
    EXIT_ARM_MIXED,
    EXIT_ALGO_NO_BF16,
    EXIT_ALGO_SILENT,
    EXIT_BINARY_DIFFERS,
    EXIT_CACHE_MISSING,
    EXIT_DRAW_FAILED,
    EXIT_DRAW_NOT_INDEPENDENT,
    EXIT_FINGERPRINT_DIFFERS,
    EXIT_FP4_SILENT,
    EXIT_KEYSET_DIFFERS,
    EXIT_OK,
    algo_stability,
    check_draw_preconditions,
    check_frozen_leg,
    draw_identity,
    kv_tokens,
    main,
    parse_algo_lines,
    parse_autotune_lines,
    parse_bench_report,
    read_draw_records,
    reduce_evidence,
    select_shipping_draw,
    selection_time_spread,
    speed_spread,
)


def quiet_main(argv: list[str]) -> int:
    """Run the CLI with its progress lines swallowed.

    The per-draw progress print is deliberate on a lease -- a job that says
    nothing for forty minutes is indistinguishable from a hung one -- and it is
    equally deliberately not wanted in a unit suite's output.
    """

    with contextlib.redirect_stdout(io.StringIO()):
        return main(argv)


ALGO_LINE = (
    "[VT_GEMM_ALGO] backend=cublasLt m=1 n=3072 k=2048 a=bf16 b=bf16 c=bf16 "
    "epilogue=rowmajor-NN algoId=21 tile=15 stages=4 splitK=1 wsSize=4194304"
)


class TokenizerTest(unittest.TestCase):
    def test_an_empty_value_does_not_swallow_the_next_key(self) -> None:
        # `prepared` prints two std::filesystem::path values, and an unset
        # FlashInfer path emits the bare token `flashinfer=`.
        fields = kv_tokens("mode=rw native= flashinfer= loaded=64 metadata=abc")
        self.assertEqual(fields["native"], "")
        self.assertEqual(fields["flashinfer"], "")
        self.assertEqual(fields["loaded"], "64")
        self.assertEqual(fields["metadata"], "abc")

    def test_the_parenthesised_breakdown_does_not_overwrite_the_paths(self) -> None:
        # `loaded=%llu (flashinfer=%llu native=%llu)` repeats two outer key
        # names as COUNTS. A flat last-wins map replaces the path with a number
        # and nothing says so.
        fields = kv_tokens(
            "native=/a/b.json flashinfer= loaded=64 (flashinfer=0 native=64) metadata=z"
        )
        self.assertEqual(fields["native"], "/a/b.json")
        self.assertEqual(fields["in_native"], "64")
        self.assertEqual(fields["in_flashinfer"], "0")

    def test_a_repeated_key_keeps_its_first_value(self) -> None:
        # Defensive, and stated as such: no shipped format string repeats an
        # outer key today. It is the second belt behind the `in_` prefix, so
        # that a future field added beside an existing name cannot silently
        # replace the one the parsers already read.
        self.assertEqual(kv_tokens("mode=first mode=second")["mode"], "first")

    def test_parsing_survives_a_timestamp_and_ansi_prefix(self) -> None:
        # An anchored grep over this tree's logs has failed on exactly this.
        prefixed = "\x1b[0m2026-09-03T10:00:00Z worker | " + ALGO_LINE
        self.assertEqual(len(parse_algo_lines(prefixed)), 1)


class BenchReportTest(unittest.TestCase):
    REPORT = (
        "Successful requests:                       32\n"
        "Total generated tokens:                    2048\n"
        "Total token throughput (tok/s):            1840.55\n"
        "Mean TPOT (ms):                            9.77\n"
    )

    def test_reads_the_fields_the_gate_turns_on(self) -> None:
        parsed = parse_bench_report(self.REPORT)
        self.assertEqual(parsed["total_token_throughput"], 1840.55)
        self.assertEqual(parsed["successful_requests"], 32.0)

    def test_a_missing_field_is_absent_not_zero(self) -> None:
        # Equal times are noise; equal COUNTS are identity. A request count that
        # did not parse must not read as a completed run of zero requests.
        parsed = parse_bench_report("Total token throughput (tok/s):            1.00\n")
        self.assertNotIn("successful_requests", parsed)

    def test_the_posix_bracket_class_is_not_a_python_regex(self) -> None:
        # The shell driver passes `\s+`, not `[[:space:]]+`. Python reads the
        # latter as the set {[,:,s,p,a,c,e}, which contains no space, so the
        # pattern matches NOTHING and every leg silently records VOID.
        import re
        import warnings

        line = "Total token throughput (tok/s):            1840.00"
        with warnings.catch_warnings():
            # Python itself warns "possible nested set" on the broken pattern.
            # That warning IS the defect; catching it keeps the suite's output
            # clean without hiding the assertion below it.
            warnings.simplefilter("ignore", FutureWarning)
            self.assertIsNone(
                re.search(r"Total token throughput \(tok/s\):[[:space:]]+([0-9.]+)", line)
            )
        self.assertIsNotNone(
            re.search(r"Total token throughput \(tok/s\):\s+([0-9.]+)", line)
        )


def _draw_record(
    label: str,
    *,
    tactic_offset: int = 0,
    algo_id: str = "21",
    tactic_set: str = "full",
    mean_us: float = 120.0,
) -> dict:
    algo = {}
    for n in (3072, 2048):
        key = f"cublasLt|m=1 n={n} k=2048|a=bf16 b=bf16 c=bf16|rowmajor-NN"
        algo[key] = {
            "backend": "cublasLt", "m": "1", "n": str(n), "k": "2048",
            "a": "bf16", "b": "bf16", "c": "bf16", "epilogue": "rowmajor-NN",
            "algoId": algo_id, "tile": "15", "stages": "4", "splitK": "1",
        }
    selected = {f"8,{n},2048": (n + tactic_offset) % 32 for n in (3072, 2048)}
    autotune = {
        "selections": {
            key: {"tactic_id": tactic, "name": f"tactic_{tactic}",
                  "mean_us": mean_us, "m": 8, "set": tactic_set}
            for key, tactic in selected.items()
        },
        "sets": [tactic_set],
        "repeat_selections": 0,
    }
    return {
        "label": label,
        "rc": 0,
        "tactic_set": tactic_set,
        "autotune": autotune,
        "algo": algo,
        "fp4": {
            "prepared": {"metadata": "fp1", "mode": "read-write"},
            "complete": {"loaded": "0", "tuned": str(len(selected)), "metadata": "fp1"},
            "selected": selected,
        },
        "bench": {"total_token_throughput": 1840.0},
        "cache_sha256": "abc",
        "cache_bytes": 12,
        "binary_sha256": "bin",
    }


class DrawPreconditionTest(unittest.TestCase):
    """Every case here is a run that looks finished and measured nothing."""

    def setUp(self) -> None:
        self.records = [
            _draw_record("draw00", tactic_offset=0),
            _draw_record("draw01", tactic_offset=7),
            _draw_record("draw02", tactic_offset=14),
        ]

    def code(self, mutate=None) -> int:
        records = copy.deepcopy(self.records)
        if mutate is not None:
            mutate(records)
        return check_draw_preconditions(records)[0]

    def test_the_unmutated_fixture_passes(self) -> None:
        self.assertEqual(self.code(), EXIT_OK)

    def test_a_failed_draw_process(self) -> None:
        self.assertEqual(self.code(lambda r: r[1].__setitem__("rc", 3)), EXIT_DRAW_FAILED)

    def test_zero_algo_lines_refuses_instead_of_reporting_stable(self) -> None:
        self.assertEqual(self.code(lambda r: r[2].__setitem__("algo", {})), EXIT_ALGO_SILENT)

    def test_no_bf16_input_says_nothing_about_the_bf16_row(self) -> None:
        def mutate(records):
            for fields in records[0]["algo"].values():
                fields["a"] = "f32"

        self.assertEqual(self.code(mutate), EXIT_ALGO_NO_BF16)

    def test_a_missing_fp4_complete_line(self) -> None:
        self.assertEqual(
            self.code(lambda r: r[1]["fp4"].__setitem__("complete", None)),
            EXIT_FP4_SILENT,
        )

    def test_a_draw_that_loaded_instead_of_tuning_is_a_copy(self) -> None:
        self.assertEqual(
            self.code(lambda r: r[0]["fp4"]["complete"].__setitem__("tuned", "0")),
            EXIT_DRAW_NOT_INDEPENDENT,
        )

    def test_divergent_plan_key_sets_compare_nothing(self) -> None:
        self.assertEqual(
            self.code(lambda r: r[2]["fp4"]["selected"].pop("8,3072,2048")),
            EXIT_KEYSET_DIFFERS,
        )

    def test_divergent_metadata_fingerprints(self) -> None:
        self.assertEqual(
            self.code(lambda r: r[1]["fp4"]["prepared"].__setitem__("metadata", "other")),
            EXIT_FINGERPRINT_DIFFERS,
        )

    def test_processes_that_saw_different_shapes(self) -> None:
        self.assertEqual(
            self.code(lambda r: r[0]["algo"].pop(next(iter(r[0]["algo"])))),
            EXIT_ALGO_KEYSET_DIFFERS,
        )

    def test_two_binaries_are_not_one_experiment(self) -> None:
        self.assertEqual(
            self.code(lambda r: r[2].__setitem__("binary_sha256", "deadbeef")),
            EXIT_BINARY_DIFFERS,
        )

    def test_a_draw_that_published_no_document_cannot_be_pinned(self) -> None:
        self.assertEqual(
            self.code(lambda r: r[1].__setitem__("cache_sha256", None)),
            EXIT_CACHE_MISSING,
        )


class AlgoStabilityTest(unittest.TestCase):
    def runs(self) -> dict:
        return {
            record["label"]: record["algo"]
            for record in (
                _draw_record("draw00"), _draw_record("draw01"), _draw_record("draw02")
            )
        }

    def test_one_config_per_key_across_processes_is_stable(self) -> None:
        result = algo_stability(self.runs())
        self.assertEqual(result["verdict"], "STABLE")
        self.assertEqual(result["keys_bf16_input"], 2)

    def test_one_moved_algo_id_makes_it_unstable(self) -> None:
        runs = self.runs()
        key = sorted(runs["draw01"])[0]
        runs["draw01"][key]["algoId"] = "99"
        result = algo_stability(runs)
        self.assertEqual(result["verdict"], "UNSTABLE")
        self.assertIn(key, result["unstable_keys"])

    def test_a_moved_tile_alone_is_also_unstable(self) -> None:
        # The selection is the four-tuple, not the id: two algos can share an id
        # and differ in tile/stages/splitK.
        runs = self.runs()
        runs["draw02"][sorted(runs["draw02"])[0]]["tile"] = "20"
        self.assertEqual(algo_stability(runs)["verdict"], "UNSTABLE")

    def test_a_key_missing_from_one_run_is_incomparable_not_stable(self) -> None:
        runs = self.runs()
        runs["draw02"].pop(sorted(runs["draw02"])[0])
        result = algo_stability(runs)
        self.assertEqual(result["verdict"], "INCOMPARABLE")
        self.assertEqual(len(result["keys_partial"]), 1)

    def test_one_process_cannot_answer_a_cross_process_question(self) -> None:
        self.assertEqual(
            algo_stability({"draw00": self.runs()["draw00"]})["verdict"], "INCOMPARABLE"
        )

    def test_no_runs_at_all(self) -> None:
        self.assertEqual(algo_stability({})["verdict"], "INCOMPARABLE")

    def test_the_verdict_carries_its_own_scope(self) -> None:
        # Four shapes are four shapes. The report must not read as a claim about
        # the cuBLASLt heuristic.
        self.assertIn("not a claim", algo_stability(self.runs())["scope"])


class DrawIdentityTest(unittest.TestCase):
    def test_identical_draws_share_every_key(self) -> None:
        one = _draw_record("draw00")["fp4"]["selected"]
        result = draw_identity({"draw00": dict(one), "draw01": dict(one)})
        self.assertEqual(result["pairwise_shared_min"], result["keys"])
        self.assertEqual(result["keys_with_multiple_tactics"], 0)

    def test_disjoint_draws_share_none(self) -> None:
        result = draw_identity(
            {
                "draw00": _draw_record("draw00", tactic_offset=0)["fp4"]["selected"],
                "draw01": _draw_record("draw01", tactic_offset=7)["fp4"]["selected"],
            }
        )
        self.assertEqual(result["pairwise_shared_max"], 0)

    def test_different_key_sets_are_incomparable(self) -> None:
        a = _draw_record("draw00")["fp4"]["selected"]
        b = dict(a)
        b.pop(sorted(b)[0])
        self.assertEqual(draw_identity({"a": a, "b": b})["verdict"], "INCOMPARABLE")

    def test_one_draw_is_incomparable(self) -> None:
        one = _draw_record("draw00")["fp4"]["selected"]
        self.assertEqual(draw_identity({"draw00": one})["verdict"], "INCOMPARABLE")


class SpeedSpreadTest(unittest.TestCase):
    def test_a_gap_inside_the_repeat_spread_is_not_a_gap(self) -> None:
        result = speed_spread({"a": [100.0, 100.4], "b": [100.2, 100.1]})
        self.assertEqual(result["verdict"], "EQUIVALENT")

    def test_a_gap_above_the_spread_but_under_the_bar(self) -> None:
        result = speed_spread({"a": [100.0, 100.05], "b": [101.0, 101.05]})
        self.assertEqual(result["verdict"], "SEPARATED_BELOW_BAR")

    def test_a_gap_over_the_bar_escalates_rather_than_selecting(self) -> None:
        result = speed_spread({"a": [100.0, 100.05], "b": [105.0, 105.05]})
        self.assertEqual(result["verdict"], "ABOVE_BAR")
        self.assertGreater(result["ratio"], 1.02)

    def test_the_bar_is_a_parameter_and_moving_it_moves_the_verdict(self) -> None:
        legs = {"a": [100.0, 100.05], "b": [101.0, 101.05]}
        self.assertEqual(speed_spread(legs, ratification_bar=1.005)["verdict"], "ABOVE_BAR")

    def test_one_leg_per_draw_cannot_separate_them(self) -> None:
        # With no repeat there is no within-draw spread to compare against, so a
        # difference cannot be told from the noise it might be.
        self.assertEqual(speed_spread({"a": [100.0], "b": [105.0]})["verdict"], "INCOMPARABLE")

    def test_one_draw_is_incomparable(self) -> None:
        self.assertEqual(speed_spread({"a": [100.0, 100.1]})["verdict"], "INCOMPARABLE")


class ShippingRuleTest(unittest.TestCase):
    ORDER = ["draw00", "draw01", "draw02"]

    def test_equivalent_draws_ship_the_first_in_draw_order(self) -> None:
        spread = speed_spread({"a": [100.0, 100.4], "b": [100.2, 100.1]})
        result = select_shipping_draw(spread, self.ORDER)
        self.assertEqual(result["ship"], "draw00")

    def test_the_rule_is_not_the_fastest_draw(self) -> None:
        # The whole point of #2752's refusal: a draw picked BY its own speed on
        # the workload it will be scored on is measuring around the harness.
        spread = speed_spread({"draw00": [100.0, 100.4], "draw01": [100.5, 100.2]})
        result = select_shipping_draw(spread, self.ORDER)
        self.assertEqual(result["ship"], "draw00")
        self.assertNotEqual(result["ship"], spread["best_draw"])

    def test_separated_draws_ship_nothing(self) -> None:
        spread = speed_spread({"a": [100.0, 100.05], "b": [105.0, 105.05]})
        self.assertIsNone(select_shipping_draw(spread, self.ORDER)["ship"])

    def test_below_bar_but_separated_also_ships_nothing(self) -> None:
        spread = speed_spread({"a": [100.0, 100.05], "b": [101.0, 101.05]})
        self.assertIsNone(select_shipping_draw(spread, self.ORDER)["ship"])


class FrozenLegTest(unittest.TestCase):
    OK = (
        "[VT_FP4_CACHE] complete mode=read-only loaded=64 tuned=0 rejected=0 "
        "saved=0 selected=64 metadata=x"
    )

    def test_a_frozen_replay_passes(self) -> None:
        ok, _ = check_frozen_leg(self.OK, 64)
        self.assertTrue(ok)

    def test_a_leg_that_retuned_is_refused(self) -> None:
        ok, why = check_frozen_leg(self.OK.replace("tuned=0", "tuned=3"), 64)
        self.assertFalse(ok)
        self.assertIn("re-tuned", why)

    def test_a_partial_install_is_refused(self) -> None:
        ok, _ = check_frozen_leg(self.OK.replace("loaded=64", "loaded=60"), 64)
        self.assertFalse(ok)

    def test_a_leg_that_announced_a_selection_is_refused(self) -> None:
        # The SECOND witness. `tuned=0` is the runtime's count; this is the
        # tuner's own voice, and a control with one witness cannot be
        # cross-checked.
        text = self.OK + (
            "\n[VT_FP4_AUTOTUNE] set=full M=8(bucket=8) N=3072 K=2048 device=0 "
            "sm=121 delay_us=5000 -> id=17 t (123.4 us), workspace=0"
        )
        ok, why = check_frozen_leg(text, 64)
        self.assertFalse(ok)
        self.assertIn("announced", why)

    def test_a_leg_with_no_complete_line_is_refused(self) -> None:
        ok, why = check_frozen_leg("nothing ran", 64)
        self.assertFalse(ok)
        self.assertIn("did not run", why)


class AutotuneSelectionParseTest(unittest.TestCase):
    """The tuner's own selection line, which is a DIAGNOSTIC and not an axis."""

    LINE = (
        "[VT_FP4_AUTOTUNE] set=full M=8(bucket=8) N=3072 K=2048 device=0 sm=121 "
        "delay_us=5000 -> id=17 sm121a_bf16_128x128 (123.4 us), workspace=4194304"
    )

    def test_reads_the_bucket_keyed_selection(self) -> None:
        parsed = parse_autotune_lines(self.LINE)
        self.assertEqual(parsed["sets"], ["full"])
        entry = parsed["selections"]["8,3072,2048"]
        self.assertEqual(entry["tactic_id"], 17)
        self.assertEqual(entry["mean_us"], 123.4)

    def test_the_key_joins_the_selected_plan_map(self) -> None:
        # `[VT_FP4_CACHE] selected` prints plan.m_bucket under the name `M`, so
        # the two maps must key identically or the diagnostic joins nothing.
        record = _draw_record("draw00")
        self.assertEqual(
            set(record["autotune"]["selections"]), set(record["fp4"]["selected"])
        )

    def test_a_tactic_name_with_spaces_is_read_whole(self) -> None:
        line = self.LINE.replace("sm121a_bf16_128x128", "baseline tactic name")
        entry = parse_autotune_lines(line)["selections"]["8,3072,2048"]
        self.assertEqual(entry["name"], "baseline tactic name")
        self.assertEqual(entry["mean_us"], 123.4)

    def test_a_log_prefix_does_not_hide_the_line(self) -> None:
        parsed = parse_autotune_lines("\x1b[0m2026-09-03Z pod | " + self.LINE)
        self.assertEqual(len(parsed["selections"]), 1)

    def test_a_repeated_key_is_counted_and_the_first_is_kept(self) -> None:
        # A key tuned twice is a lazy miss after the pre-serve warmup. The two
        # readings must DIFFER for this to discriminate: two identical lines
        # cannot tell "keep the first" from "keep the last", which is a test
        # that asserts nothing.
        second = self.LINE.replace("id=17", "id=29").replace("123.4", "456.7")
        parsed = parse_autotune_lines(self.LINE + "\n" + second)
        self.assertEqual(parsed["repeat_selections"], 1)
        self.assertEqual(len(parsed["selections"]), 1)
        entry = parsed["selections"]["8,3072,2048"]
        self.assertEqual(entry["tactic_id"], 17)
        self.assertEqual(entry["mean_us"], 123.4)

    def test_the_w1_arm_is_reported_under_its_own_name(self) -> None:
        parsed = parse_autotune_lines(self.LINE.replace("set=full", "set=w1"))
        self.assertEqual(parsed["sets"], ["w1"])


class SelectionTimeDiagnosticTest(unittest.TestCase):
    def draws(self, *means: float) -> dict:
        return {
            f"draw{i:02d}": _draw_record(f"draw{i:02d}", mean_us=mean)["autotune"]["selections"]
            for i, mean in enumerate(means)
        }

    def test_it_reports_a_state_and_never_a_verdict(self) -> None:
        # STRUCTURAL GUARD. `select_shipping_draw` reads `verdict`; this block
        # deliberately has none, so a selection-time result cannot be handed to
        # the shipping rule and be mistaken for an end-to-end one.
        result = selection_time_spread(self.draws(120.0, 130.0))
        self.assertNotIn("verdict", result)
        self.assertEqual(result["state"], "DIAGNOSTIC")
        self.assertIsNone(select_shipping_draw(result, ["draw00", "draw01"])["ship"])

    def test_it_carries_the_reason_it_cannot_gate(self) -> None:
        result = selection_time_spread(self.draws(120.0, 130.0))
        self.assertIn("per-iteration", result["not_a_gate"])

    def test_it_reports_the_per_key_ratio_and_the_distinct_ids(self) -> None:
        result = selection_time_spread(self.draws(100.0, 110.0))
        self.assertAlmostEqual(result["max_over_min_max"], 1.1)
        self.assertEqual(result["keys"], 2)

    def test_one_draw_is_incomparable(self) -> None:
        self.assertEqual(selection_time_spread(self.draws(120.0))["state"], "INCOMPARABLE")


class TacticSetArmTest(unittest.TestCase):
    def test_one_root_may_not_hold_two_arms(self) -> None:
        # The arms select by different rules: 32 candidates by pure argmin
        # versus 4 behind a >1% stickiness damper. A pooled spread names no rule.
        records = [
            _draw_record("draw00", tactic_set="full"),
            _draw_record("draw01", tactic_set="w1", tactic_offset=7),
        ]
        code, problems = check_draw_preconditions(records)
        self.assertEqual(code, EXIT_ARM_MIXED)
        self.assertIn("DIFFERENT RULES", problems[0])

    def test_a_record_without_an_arm_is_refused(self) -> None:
        # An unset VT_FP4_FULL_TACTICS is default-ON, so a draw whose arm was
        # never recorded is a draw nobody can attribute afterwards.
        records = [_draw_record("draw00"), _draw_record("draw01", tactic_offset=7)]
        records[1].pop("tactic_set")
        self.assertEqual(check_draw_preconditions(records)[0], EXIT_ARM_MIXED)

    def test_a_single_arm_root_passes(self) -> None:
        records = [
            _draw_record("draw00", tactic_set="w1"),
            _draw_record("draw01", tactic_set="w1", tactic_offset=7),
        ]
        self.assertEqual(check_draw_preconditions(records)[0], EXIT_OK)


class DryRunEndToEndTest(unittest.TestCase):
    """The record / resume / refuse path, with no device and no subprocess."""

    def test_draw_records_resume_and_reduce(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evidence = pathlib.Path(tmp) / "ev"
            rc = quiet_main([
                "draw", "--evidence", str(evidence), "--bench", "/bin/true",
                "--model", "/nonexistent", "--draws", "3", "--dry-run",
            ])
            self.assertEqual(rc, EXIT_OK)
            first = read_draw_records(evidence)
            self.assertEqual([r["label"] for r in first], ["draw00", "draw01", "draw02"])

            # A resumed run replays what is done and only adds what is owed.
            stamp = (evidence / "draws" / "draw00" / "record.json").stat().st_mtime_ns
            rc = quiet_main([
                "draw", "--evidence", str(evidence), "--bench", "/bin/true",
                "--model", "/nonexistent", "--draws", "5", "--dry-run",
            ])
            self.assertEqual(rc, EXIT_OK)
            self.assertEqual(len(read_draw_records(evidence)), 5)
            self.assertEqual(
                (evidence / "draws" / "draw00" / "record.json").stat().st_mtime_ns, stamp
            )

            code, report = reduce_evidence(
                evidence, metric="total_token_throughput", ratification_bar=1.02
            )
            self.assertEqual(code, EXIT_OK)
            # The fixture must never be mistakable for a measurement.
            self.assertTrue(report["dry_run"])
            self.assertEqual(report["issue_2750_draw_processes"]["verdict"], "STABLE")
            self.assertEqual(report["tactic_set"], ["full"])
            self.assertEqual(report["issue_2751_selection_time"]["state"], "DIAGNOSTIC")
            self.assertEqual(report["issue_2751_speed"]["verdict"], "NOT RUN")
            self.assertIsNone(report["issue_2752"]["ship"])
            self.assertEqual(report["clock_windows"]["state"], "ABSENT")

    def test_reduce_refuses_an_empty_evidence_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, report = reduce_evidence(
                pathlib.Path(tmp), metric="total_token_throughput", ratification_bar=1.02
            )
            self.assertNotEqual(code, EXIT_OK)
            self.assertEqual(report["verdict"], "REFUSED")

    def test_a_scoring_ledger_reaches_the_speed_and_shipping_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evidence = pathlib.Path(tmp) / "ev"
            quiet_main([
                "draw", "--evidence", str(evidence), "--bench", "/bin/true",
                "--model", "/nonexistent", "--draws", "2", "--dry-run",
            ])
            ledger = evidence / "score" / "legs.jsonl"
            ledger.parent.mkdir(parents=True, exist_ok=True)
            ledger.write_text(
                "\n".join(
                    json.dumps({"arm": arm, "boot_id": "b", "rc": 0,
                                "total_token_throughput": value})
                    for arm, value in (
                        ("draw00", 100.0), ("draw01", 105.0),
                        ("draw00", 100.1), ("draw01", 105.1),
                    )
                ) + "\n",
                encoding="utf-8",
            )
            _, report = reduce_evidence(
                evidence, metric="total_token_throughput", ratification_bar=1.02
            )
            self.assertEqual(report["issue_2751_speed"]["verdict"], "ABOVE_BAR")
            self.assertIsNone(report["issue_2752"]["ship"])


if __name__ == "__main__":
    unittest.main()
