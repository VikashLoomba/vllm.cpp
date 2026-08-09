#!/usr/bin/env python3
"""Focused behavior tests for deterministic state-record migration."""

from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MIGRATOR = ROOT / "scripts/migrate-state-record.py"
sys.path.insert(0, str(ROOT / "scripts"))

import state_record


SOURCE = (
    "# State log\r\n\r\nPrelude café is deliberately unanchored.\r\n".encode()
    + b"## First event\r\n"
    + b"<!-- state: 2026-08-05T09:00 -->\r\n"
    + b"Alpha payload.\r\n"
    + b"## Duplicate timestamp\n"
    + b"<!-- state: 2026-08-05T09:00 -->\n"
    + b"Beta payload.\n\n## An unanchored subheading\nStill beta.\n"
    + b"## Date-only event\n"
    + b"<!-- state: 2026-08-06 -->\n"
    + b"Gamma payload without a final newline."
)


class MigrationRepo:
    def __init__(self, directory: str, source: bytes = SOURCE) -> None:
        self.root = Path(directory)
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=self.root, check=True)
        subprocess.run(
            ["git", "config", "user.name", "Migration Test"],
            cwd=self.root,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "migration@example.invalid"],
            cwd=self.root,
            check=True,
        )
        (self.root / ".agents").mkdir()
        (self.root / ".agents/state.md").write_bytes(source)
        subprocess.run(["git", "add", ".agents/state.md"], cwd=self.root, check=True)
        subprocess.run(
            ["git", "commit", "-qm", "legacy source"], cwd=self.root, check=True
        )
        self.source_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.root, text=True
        ).strip()

    def run(self, mode: str, revision: str | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(MIGRATOR),
                "--source-revision",
                revision or self.source_revision,
                "--output-root",
                str(self.root),
                mode,
            ],
            cwd=self.root,
            text=True,
            capture_output=True,
        )

    def manifest_rows(self) -> list[list[str]]:
        path = self.root / ".agents/completed/state-migration-manifest.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.reader(handle))

    def event_rows(self) -> list[list[str]]:
        rows: list[list[str]] = []
        for path in sorted((self.root / ".agents/state-index").glob("*.csv")):
            with path.open(newline="", encoding="utf-8") as handle:
                rows.extend(list(csv.reader(handle))[1:])
        return rows


class MigrationApplyTests(unittest.TestCase):
    def test_apply_and_verify_prelude_with_multi_month_history(self) -> None:
        """Catches legacy IDs sorting after timestamp IDs across dated shards."""
        source = (
            b"Unanchored prelude.\n"
            b"## July event\n"
            b"<!-- state: 2026-07-31T23:59:00Z -->\n"
            b"July payload.\n"
            b"## August event\n"
            b"<!-- state: 2026-08-01T00:01:00Z -->\n"
            b"August payload.\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            repo = MigrationRepo(directory, source)

            applied = repo.run("--apply")
            verified = repo.run("--verify")

            self.assertEqual(applied.returncode, 0, applied.stderr)
            self.assertEqual(verified.returncode, 0, verified.stderr)

    def test_apply_partitions_indexes_and_evidence_by_event_month(self) -> None:
        """Catches collapsing a multi-month history into the final month."""
        source = (
            b"## July event\n"
            b"<!-- state: 2026-07-31T23:59:00Z -->\n"
            b"July payload.\n"
            b"## August event\n"
            b"<!-- state: 2026-08-01T00:01:00Z -->\n"
            b"August payload.\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            repo = MigrationRepo(directory, source)

            result = repo.run("--apply")

            self.assertEqual(result.returncode, 0, result.stderr)
            manifest_path = repo.root / ".agents/state.csv"
            with manifest_path.open(newline="", encoding="utf-8") as handle:
                manifest = list(csv.reader(handle))
            self.assertEqual(
                manifest[1:],
                [
                    [
                        "1",
                        "2026-07-001",
                        ".agents/state-index/2026-07-001.csv",
                        ".agents/state-events/2026-07/",
                    ],
                    [
                        "1",
                        "2026-08-001",
                        ".agents/state-index/2026-08-001.csv",
                        ".agents/state-events/2026-08/",
                    ],
                ],
            )
            rows = repo.event_rows()
            self.assertEqual(
                [row[9] for row in rows],
                [
                    ".agents/state-events/2026-07/STATE-20260731T235900-001.md",
                    ".agents/state-events/2026-08/STATE-20260801T000100-001.md",
                ],
            )

    def test_apply_segments_bytes_wraps_payloads_and_does_not_infer_lifecycle(self) -> None:
        """Catches normalized bytes, heuristic prelude splitting, and fabricated metadata."""
        self.assertTrue(MIGRATOR.exists(), "migrate-state-record.py is missing")
        with tempfile.TemporaryDirectory() as directory:
            repo = MigrationRepo(directory)

            result = repo.run("--apply")

            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = repo.manifest_rows()
            self.assertEqual(
                manifest[0],
                [
                    "record_type",
                    "source_commit",
                    "source_blob",
                    "source_bytes",
                    "source_sha256",
                    "event_id",
                    "start_byte",
                    "end_byte",
                    "payload_bytes",
                    "payload_sha256",
                    "evidence_path",
                ],
            )
            self.assertEqual(len(manifest), 6)
            source_blob = subprocess.check_output(
                ["git", "rev-parse", f"{repo.source_revision}:.agents/state.md"],
                cwd=repo.root,
                text=True,
            ).strip()
            self.assertEqual(
                manifest[1],
                [
                    "source",
                    repo.source_revision,
                    source_blob,
                    str(len(SOURCE)),
                    hashlib.sha256(SOURCE).hexdigest(),
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ],
            )
            first_heading = SOURCE.index(b"## First event")
            self.assertEqual(manifest[2][6:8], ["0", str(first_heading)])
            self.assertEqual(manifest[2][5], "STATE-LEGACY-000001")
            self.assertEqual(
                [row[5] for row in manifest[3:5]],
                ["STATE-20260805T090000-001", "STATE-20260805T090000-002"],
            )
            self.assertEqual(manifest[5][5], "STATE-LEGACY-000002")

            reconstructed = bytearray()
            expected_start = 0
            for row in manifest[2:]:
                self.assertEqual(row[:5], ["event", "", "", "", ""])
                event_id, start_s, end_s, count_s, digest, evidence_path = row[5:]
                start, end, count = int(start_s), int(end_s), int(count_s)
                self.assertEqual(start, expected_start)
                self.assertEqual(end - start, count)
                payload = state_record.read_legacy_payload(repo.root / evidence_path)
                self.assertEqual(payload, SOURCE[start:end])
                self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)
                reconstructed.extend(payload)
                expected_start = end
                evidence = (repo.root / evidence_path).read_bytes()
                self.assertIn(f"<!-- state-event: {event_id} -->".encode(), evidence)
                self.assertIn(b"<!-- legacy-source:", evidence)
            self.assertEqual(expected_start, len(SOURCE))
            self.assertEqual(bytes(reconstructed), SOURCE)

            by_id = {row[0]: row for row in repo.event_rows()}
            self.assertEqual(by_id["STATE-20260805T090000-001"][1], "2026-08-05T09:00:00Z")
            self.assertEqual(by_id["STATE-LEGACY-000002"][1], "2026-08-06")
            for row in by_id.values():
                self.assertEqual(row[2], "legacy_import")
                self.assertEqual(row[3:9], ["", "", "", "", "", ""])
                self.assertEqual(row[10:], ["", "", ""])

    def test_reapplying_identical_generation_is_reproducible(self) -> None:
        """Catches environment-dependent output and refusal of byte-identical files."""
        with tempfile.TemporaryDirectory() as directory:
            repo = MigrationRepo(directory)
            first = repo.run("--apply")
            self.assertEqual(first.returncode, 0, first.stderr)
            before = {
                path.relative_to(repo.root).as_posix(): path.read_bytes()
                for path in repo.root.glob(".agents/**/*")
                if path.is_file()
            }

            second = repo.run("--apply")

            self.assertEqual(second.returncode, 0, second.stderr)
            after = {
                path.relative_to(repo.root).as_posix(): path.read_bytes()
                for path in repo.root.glob(".agents/**/*")
                if path.is_file()
            }
            self.assertEqual(after, before)

    def test_apply_refuses_to_overwrite_different_generated_bytes(self) -> None:
        """Catches destructive regeneration over reviewed migration evidence."""
        with tempfile.TemporaryDirectory() as directory:
            repo = MigrationRepo(directory)
            applied = repo.run("--apply")
            self.assertEqual(applied.returncode, 0, applied.stderr)
            event_path = repo.root / repo.manifest_rows()[2][10]
            mutated = event_path.read_bytes().replace(b"Prelude", b"Tampered", 1)
            event_path.write_bytes(mutated)

            result = repo.run("--apply")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refus", result.stderr.lower())
            self.assertIn(event_path.relative_to(repo.root).as_posix(), result.stderr)
            self.assertEqual(event_path.read_bytes(), mutated)


class MigrationVerifyTests(unittest.TestCase):
    def test_verify_accepts_exact_generated_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = MigrationRepo(directory)
            applied = repo.run("--apply")
            self.assertEqual(applied.returncode, 0, applied.stderr)

            result = repo.run("--verify")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("byte-exact", result.stdout)

    def test_verify_accepts_newer_revision_with_identical_source_bytes(self) -> None:
        """Current-base verification must not rewrite immutable provenance."""
        frozen_commit = "994cd8d4122ecf44f72d51fabd61c45adaaea9d3"
        frozen_blob = "93a8d0da802a7ea7cbea4bee3bedffb4d90459f7"
        frozen_digest = "00c08e974724c19b5f79cce44df71c6fbfef4db32aa6acb545ef56546e3bb5e6"
        current_blob = subprocess.check_output(
            ["git", "rev-parse", "origin/main:.agents/state.md"],
            cwd=ROOT,
            text=True,
        ).strip()
        current_source = subprocess.check_output(
            ["git", "show", "origin/main:.agents/state.md"], cwd=ROOT
        )
        self.assertEqual(current_blob, frozen_blob)
        self.assertEqual(len(current_source), 3191283)
        self.assertEqual(hashlib.sha256(current_source).hexdigest(), frozen_digest)
        self.assertEqual(
            subprocess.check_output(
                ["git", "rev-parse", f"{frozen_commit}^{{commit}}"],
                cwd=ROOT,
                text=True,
            ).strip(),
            frozen_commit,
        )
        with tempfile.TemporaryDirectory() as directory:
            repo = MigrationRepo(directory)
            applied = repo.run("--apply")
            self.assertEqual(applied.returncode, 0, applied.stderr)
            subprocess.run(
                ["git", "commit", "--allow-empty", "-qm", "new base, same source"],
                cwd=repo.root,
                check=True,
            )
            current_revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo.root, text=True
            ).strip()

            result = repo.run("--verify", current_revision)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(current_revision, result.stdout)
            self.assertIn(repo.source_revision, result.stdout)

    def test_verify_rejects_newer_revision_with_different_source_bytes(self) -> None:
        """A matching commit shape cannot hide source content drift."""
        with tempfile.TemporaryDirectory() as directory:
            repo = MigrationRepo(directory)
            applied = repo.run("--apply")
            self.assertEqual(applied.returncode, 0, applied.stderr)
            state_path = repo.root / ".agents/state.md"
            compatibility_stub = state_path.read_bytes()
            state_path.write_bytes(SOURCE + b"\nnew source bytes\n")
            subprocess.run(
                ["git", "add", ".agents/state.md"], cwd=repo.root, check=True
            )
            subprocess.run(
                ["git", "commit", "-qm", "new base, changed source"],
                cwd=repo.root,
                check=True,
            )
            current_revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo.root, text=True
            ).strip()
            state_path.write_bytes(compatibility_stub)

            result = repo.run("--verify", current_revision)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("differs from frozen migration source", result.stderr)

    def test_verify_rejects_boundary_payload_header_and_missing_file_mutations(self) -> None:
        """Catches gaps/overlaps, payload edits, schema drift, and incomplete output."""
        mutations = (
            "gap",
            "overlap",
            "payload",
            "header",
            "missing",
            "provenance_missing",
            "provenance_commit",
            "provenance_blob",
            "provenance_size",
            "provenance_hash",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                repo = MigrationRepo(directory)
                applied = repo.run("--apply")
                self.assertEqual(applied.returncode, 0, applied.stderr)
                manifest_path = repo.root / ".agents/completed/state-migration-manifest.csv"
                rows = repo.manifest_rows()
                if mutation in {"gap", "overlap"}:
                    rows[3][6] = str(int(rows[3][6]) + (1 if mutation == "gap" else -1))
                    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
                        csv.writer(handle, lineterminator="\n").writerows(rows)
                elif mutation == "payload":
                    event_path = repo.root / rows[3][10]
                    event_path.write_bytes(event_path.read_bytes().replace(b"Alpha", b"Omega", 1))
                elif mutation == "header":
                    rows[0][0] = "wrong_id"
                    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
                        csv.writer(handle, lineterminator="\n").writerows(rows)
                elif mutation == "missing":
                    (repo.root / rows[3][10]).unlink()
                else:
                    if mutation == "provenance_missing":
                        del rows[1]
                    else:
                        column = {
                            "provenance_commit": 1,
                            "provenance_blob": 2,
                            "provenance_size": 3,
                            "provenance_hash": 4,
                        }[mutation]
                        rows[1][column] = "0" * (64 if column == 4 else 40)
                    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
                        csv.writer(handle, lineterminator="\n").writerows(rows)

                result = repo.run("--verify")

                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(result.stderr.strip(), result)

    def test_missing_source_revision_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = MigrationRepo(directory)

            result = repo.run("--verify", "not-a-revision")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("source revision", result.stderr.lower())
            self.assertIn("not-a-revision", result.stderr)

    def test_invalid_anchor_timestamp_is_rejected_before_writing(self) -> None:
        """Catches generation of invalid structured metadata from a malformed anchor."""
        source = b"## Invalid date\n<!-- state: 2026-13-40 -->\npayload\n"
        with tempfile.TemporaryDirectory() as directory:
            repo = MigrationRepo(directory, source)

            result = repo.run("--apply")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("invalid timestamp", result.stderr.lower())
            self.assertFalse((repo.root / ".agents/state.csv").exists())


if __name__ == "__main__":
    unittest.main()
