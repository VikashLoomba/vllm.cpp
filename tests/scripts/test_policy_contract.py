#!/usr/bin/env python3
"""Mutation tests for the authoritative policy and waiver registries."""

from __future__ import annotations

import dataclasses
import datetime as dt
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.policy_contract import (
    PolicyRule,
    Waiver,
    load_policy,
    load_waivers,
    validate_policy,
)


POLICY_HEADER = "rule_id,scope,trigger,requirement,enforcement,waiver_class,procedure\n"
WAIVER_HEADER = "waiver_id,rule_id,scope,owner,reason,evidence,expires\n"
ROOT = Path(__file__).resolve().parents[2]


class PolicyFixture:
    def __init__(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / ".agents").mkdir()
        (self.root / "scripts").mkdir()
        (self.root / "scripts/check-policy.py").write_text("#!/usr/bin/env python3\n")
        (self.root / ".agents/workflow.md").write_text("# Workflow\n")
        self.write_policy(
            "POL-TEST-001,all,always,Keep the contract testable.,"
            "scripts/check-policy.py,expiring,.agents/workflow.md\n"
        )
        self.write_waivers("")

    def close(self) -> None:
        self.tmp.cleanup()

    def write_policy(self, rows: str) -> None:
        (self.root / ".agents/policy.csv").write_text(POLICY_HEADER + rows)

    def write_waivers(self, rows: str) -> None:
        (self.root / ".agents/waivers.csv").write_text(WAIVER_HEADER + rows)


class LoadContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = PolicyFixture()

    def tearDown(self) -> None:
        self.fx.close()

    def test_records_are_immutable_and_valid_registry_loads(self) -> None:
        rules = load_policy(self.fx.root)
        self.assertEqual(set(rules), {"POL-TEST-001"})
        self.assertEqual(rules["POL-TEST-001"].waiver_class, "expiring")
        self.assertEqual(load_waivers(self.fx.root, rules, today=dt.date(2026, 8, 7)), [])
        with self.assertRaises(dataclasses.FrozenInstanceError):
            rules["POL-TEST-001"].scope = "changed"  # type: ignore[misc]
        self.assertTrue(dataclasses.is_dataclass(PolicyRule))
        self.assertTrue(dataclasses.is_dataclass(Waiver))


class PolicyMutations(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = PolicyFixture()

    def tearDown(self) -> None:
        self.fx.close()

    def errors(self) -> list[str]:
        return validate_policy(self.fx.root, schema_only=True)

    def test_valid_fixture_passes_schema_validation(self) -> None:
        self.assertEqual(self.errors(), [])

    def test_exact_header_is_required(self) -> None:
        path = self.fx.root / ".agents/policy.csv"
        path.write_text(path.read_text().replace("rule_id,scope", "id,scope", 1))
        self.assertTrue(any("header" in error for error in self.errors()))

    def test_duplicate_and_malformed_rule_ids_are_rejected(self) -> None:
        self.fx.write_policy(
            "POL-TEST-001,all,always,First.,scripts/check-policy.py,never,.agents/workflow.md\n"
            "POL-TEST-001,all,always,Second.,scripts/check-policy.py,never,.agents/workflow.md\n"
            "test-2,all,always,Third.,scripts/check-policy.py,never,.agents/workflow.md\n"
        )
        errors = self.errors()
        self.assertTrue(any("duplicate rule_id" in error for error in errors), errors)
        self.assertTrue(any("malformed rule_id" in error for error in errors), errors)

    def test_multiline_field_and_blank_physical_record_are_rejected(self) -> None:
        self.fx.write_policy(
            'POL-TEST-001,all,always,"First physical line\nsecond physical line",'
            "scripts/check-policy.py,never,.agents/workflow.md\n\n"
        )
        errors = self.errors()
        self.assertTrue(any("multiline" in error for error in errors), errors)
        self.assertTrue(any("blank physical" in error for error in errors), errors)

    def test_empty_fields_and_bad_waiver_class_are_rejected_in_bootstrap(self) -> None:
        self.fx.write_policy(
            "POL-TEST-001,,always,,scripts/check-policy.py,forever,.agents/workflow.md\n"
        )
        errors = self.errors()
        self.assertTrue(any("scope is empty" in error for error in errors), errors)
        self.assertTrue(any("requirement is empty" in error for error in errors), errors)
        self.assertTrue(any("waiver_class" in error for error in errors), errors)

    def test_checker_and_procedure_paths_must_be_known_repo_files(self) -> None:
        self.fx.write_policy(
            "POL-TEST-001,all,always,Do it.,scripts/check-missing.py,never,.agents/missing.md\n"
        )
        errors = self.errors()
        self.assertTrue(any("unknown enforcement" in error for error in errors), errors)
        self.assertTrue(any("unknown procedure" in error for error in errors), errors)

    def test_absolute_traversing_and_non_checker_paths_are_rejected(self) -> None:
        (self.fx.root / "scripts/tool.py").write_text("pass\n")
        self.fx.write_policy(
            "POL-TEST-001,all,always,Do it.,scripts/tool.py;/tmp/check.py,never,../workflow.md\n"
        )
        errors = self.errors()
        self.assertTrue(any("checker entrypoint" in error for error in errors), errors)
        self.assertTrue(any("repo-relative" in error for error in errors), errors)

    def test_policy_count_and_byte_budgets_are_enforced(self) -> None:
        rows = "".join(
            f"POL-TEST-{index:03d},all,always,Rule {index}.,scripts/check-policy.py,never,.agents/workflow.md\n"
            for index in range(61)
        )
        self.fx.write_policy(rows)
        self.assertTrue(any("more than 60" in error for error in self.errors()))
        self.fx.write_policy(
            "POL-TEST-001,all,always," + ("x" * 17000) + ",scripts/check-policy.py,never,.agents/workflow.md\n"
        )
        self.assertTrue(any("16384-byte" in error for error in self.errors()))

    def test_load_policy_refuses_an_invalid_registry(self) -> None:
        self.fx.write_policy(
            "POL-TEST-001,all,always,Do it.,scripts/check-missing.py,never,.agents/workflow.md\n"
        )
        with self.assertRaises(ValueError):
            load_policy(self.fx.root)


class WaiverMutations(unittest.TestCase):
    TODAY = dt.date(2026, 8, 7)

    def setUp(self) -> None:
        self.fx = PolicyFixture()

    def tearDown(self) -> None:
        self.fx.close()

    def load_errors(self) -> str:
        rules = load_policy(self.fx.root)
        with self.assertRaises(ValueError) as raised:
            load_waivers(self.fx.root, rules, today=self.TODAY)
        return str(raised.exception)

    def mark_used(self, waiver_id: str) -> None:
        (self.fx.root / ".agents/waiver-evidence.md").write_text(
            f"Applied exception: {waiver_id}\n"
        )

    def test_exact_header_and_valid_future_waiver(self) -> None:
        self.mark_used("WAIVER-TEST-001")
        self.fx.write_waivers(
            "WAIVER-TEST-001,POL-TEST-001,task:POLICY-1,maintainer,"
            "Bounded migration,docs/evidence.md,2026-08-08\n"
        )
        waiver = load_waivers(
            self.fx.root, load_policy(self.fx.root), today=self.TODAY
        )[0]
        self.assertEqual(waiver.expires, dt.date(2026, 8, 8))
        self.assertEqual(waiver.scope, "task:POLICY-1")

        path = self.fx.root / ".agents/waivers.csv"
        path.write_text(path.read_text().replace("waiver_id,rule_id", "id,rule_id", 1))
        self.assertIn("header", self.load_errors())

    def test_unknown_and_never_waivable_rules_are_rejected(self) -> None:
        self.fx.write_policy(
            "POL-TEST-001,all,always,Do it.,scripts/check-policy.py,never,.agents/workflow.md\n"
        )
        self.fx.write_waivers(
            "WAIVER-TEST-001,POL-MISSING,task:POLICY-1,owner,reason,evidence,2026-08-08\n"
            "WAIVER-TEST-002,POL-TEST-001,task:POLICY-2,owner,reason,evidence,2026-08-08\n"
        )
        errors = self.load_errors()
        self.assertIn("unknown rule", errors)
        self.assertIn("cannot be waived", errors)

    def test_scope_must_be_narrow_exact_and_unique(self) -> None:
        self.fx.write_waivers(
            "WAIVER-TEST-001,POL-TEST-001,path:*,owner,reason,evidence,2026-08-08\n"
            "WAIVER-TEST-002,POL-TEST-001,repository:all,owner,reason,evidence,2026-08-08\n"
            "WAIVER-TEST-003,POL-TEST-001,task:POLICY-3,owner,reason,evidence,2026-08-08\n"
            "WAIVER-TEST-004,POL-TEST-001,task:POLICY-3,owner,reason,evidence,2026-08-09\n"
        )
        errors = self.load_errors()
        self.assertIn("wildcard", errors)
        self.assertIn("exact scope", errors)
        self.assertIn("duplicate rule/scope", errors)

    def test_repository_wide_and_traversing_scopes_are_not_exact(self) -> None:
        self.fx.write_waivers(
            "WAIVER-TEST-001,POL-TEST-001,pr:all,owner,reason,evidence,2026-08-08\n"
            "WAIVER-TEST-002,POL-TEST-001,commit:all,owner,reason,evidence,2026-08-08\n"
            "WAIVER-TEST-003,POL-TEST-001,hardware:all,owner,reason,evidence,2026-08-08\n"
            "WAIVER-TEST-004,POL-TEST-001,path:foo/..,owner,reason,evidence,2026-08-08\n"
        )
        errors = self.load_errors()
        for scope in ("pr:all", "commit:all", "hardware:all", "path:foo/.."):
            with self.subTest(scope=scope):
                self.assertIn(repr(scope), errors)
                self.assertIn("exact scope", errors)

    def test_each_supported_scope_kind_accepts_a_concrete_target(self) -> None:
        waiver_ids = [f"WAIVER-TEST-{index:03d}" for index in range(1, 7)]
        (self.fx.root / ".agents/waiver-evidence.md").write_text(
            "Applied exceptions: " + " ".join(waiver_ids) + "\n"
        )
        scopes = (
            "task:POLICY-1",
            "pr:128",
            f"commit:{'a' * 40}",
            "path:.agents/workflow.md",
            "gate:policy-schema",
            "hardware:GB10-1",
        )
        self.fx.write_waivers(
            "".join(
                f"{waiver_id},POL-TEST-001,{scope},owner,reason,evidence,2026-08-08\n"
                for waiver_id, scope in zip(waiver_ids, scopes, strict=True)
            )
        )
        waivers = load_waivers(
            self.fx.root, load_policy(self.fx.root), today=self.TODAY
        )
        self.assertEqual([waiver.scope for waiver in waivers], list(scopes))

    def test_duplicate_ids_and_missing_attribution_are_rejected(self) -> None:
        self.fx.write_waivers(
            "WAIVER-TEST-001,POL-TEST-001,task:POLICY-1,,reason,evidence,2026-08-08\n"
            "WAIVER-TEST-001,POL-TEST-001,task:POLICY-2,owner,,evidence,2026-08-08\n"
            "WAIVER-TEST-003,POL-TEST-001,task:POLICY-3,owner,reason,,2026-08-08\n"
        )
        errors = self.load_errors()
        self.assertIn("duplicate waiver_id", errors)
        self.assertIn("owner is empty", errors)
        self.assertIn("reason is empty", errors)
        self.assertIn("evidence is empty", errors)

    def test_invalid_expired_and_today_expiry_are_rejected(self) -> None:
        self.fx.write_waivers(
            "WAIVER-TEST-001,POL-TEST-001,task:POLICY-1,owner,reason,evidence,not-a-date\n"
            "WAIVER-TEST-002,POL-TEST-001,task:POLICY-2,owner,reason,evidence,2026-08-06\n"
            "WAIVER-TEST-003,POL-TEST-001,task:POLICY-3,owner,reason,evidence,2026-08-07\n"
        )
        errors = self.load_errors()
        self.assertIn("ISO date", errors)
        self.assertIn("expired", errors)

    def test_unreferenced_waiver_is_rejected_as_unused(self) -> None:
        self.fx.write_waivers(
            "WAIVER-TEST-001,POL-TEST-001,task:POLICY-1,owner,reason,evidence,2026-08-08\n"
        )
        self.assertIn("unused waiver", self.load_errors())

    def test_schema_only_does_not_suppress_waiver_defects(self) -> None:
        self.fx.write_waivers(
            "WAIVER-TEST-001,POL-MISSING,path:*,owner,reason,evidence,2026-08-06\n"
        )
        errors = validate_policy(self.fx.root, schema_only=True)
        self.assertTrue(any("unknown rule" in error for error in errors), errors)
        self.assertTrue(any("wildcard" in error for error in errors), errors)
        self.assertTrue(any("expired" in error for error in errors), errors)


class CommandLineContract(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = PolicyFixture()
        shutil.copy2(ROOT / "scripts/policy_contract.py", self.fx.root / "scripts")
        shutil.copy2(ROOT / "scripts/check-policy.py", self.fx.root / "scripts")

    def tearDown(self) -> None:
        self.fx.close()

    def run_checker(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["python3", "scripts/check-policy.py", *arguments],
            cwd=self.fx.root,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_cli_passes_valid_registry_and_fails_bootstrap_defect(self) -> None:
        self.assertEqual(self.run_checker("--schema-only").returncode, 0)
        path = self.fx.root / ".agents/policy.csv"
        path.write_text(path.read_text().replace("rule_id", "id", 1))
        result = self.run_checker("--schema-only")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("header", result.stdout + result.stderr)


class RepositoryRegistry(unittest.TestCase):
    def test_accepted_design_inventory_passes_schema(self) -> None:
        rules = load_policy(ROOT)
        self.assertLessEqual(len(rules), 60)
        self.assertLessEqual((ROOT / ".agents/policy.csv").stat().st_size, 16 * 1024)
        self.assertEqual(validate_policy(ROOT, schema_only=True), [])


if __name__ == "__main__":
    unittest.main()
