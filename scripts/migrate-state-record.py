#!/usr/bin/env python3
"""Generate and verify a deterministic, byte-lossless structured state tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import state_record


SOURCE_PATH = ".agents/state.md"
MANIFEST_PATH = ".agents/completed/state-migration-manifest.csv"
ANCHOR_RE = re.compile(
    rb"(?m)^## [^\r\n]+(?:\r\n|\n)<!-- state: ([^>\r\n]+) -->(?:\r\n|\n|$)"
)
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?Z?$"
)
FROZEN_SOURCE_RE = re.compile(
    rb"(?m)^Frozen legacy source: "
    rb"https://github\.com/mudler/vllm\.cpp/blob/([0-9a-f]{40})/"
    rb"\.agents/state\.md\r?$"
)


class MigrationError(RuntimeError):
    """A deterministic migration contract violation."""


@dataclass(frozen=True)
class Segment:
    event_id: str
    start: int
    end: int
    occurred_at: str
    evidence_path: str


@dataclass(frozen=True)
class Migration:
    source_revision: str
    source: bytes
    segments: tuple[Segment, ...]
    artifacts: dict[str, bytes]


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def read_source(root: Path, revision: str) -> tuple[str, bytes]:
    resolved = _git(root, "rev-parse", "--verify", f"{revision}^{{commit}}")
    if resolved.returncode != 0:
        detail = resolved.stderr.decode("utf-8", errors="replace").strip()
        raise MigrationError(f"source revision {revision!r} does not resolve: {detail}")
    full_revision = resolved.stdout.decode("ascii").strip()
    source = _git(root, "show", f"{full_revision}:{SOURCE_PATH}")
    if source.returncode != 0:
        detail = source.stderr.decode("utf-8", errors="replace").strip()
        raise MigrationError(
            f"source revision {full_revision} has no readable {SOURCE_PATH}: {detail}"
        )
    return full_revision, source.stdout


def frozen_source_revision(root: Path) -> str:
    """Return the one reviewed provenance revision embedded in the stub."""

    try:
        stub = (root / SOURCE_PATH).read_bytes()
    except OSError as exc:
        raise MigrationError(f"cannot read structured state stub: {exc}") from exc
    matches = FROZEN_SOURCE_RE.findall(stub)
    if len(matches) != 1:
        raise MigrationError(
            f"{SOURCE_PATH} must name exactly one frozen legacy source revision"
        )
    return matches[0].decode("ascii")


def build_verification_migration(
    root: Path, requested_revision: str
) -> tuple[str, Migration]:
    """Bind current source bytes to immutable migration provenance."""

    current_revision, current_source = read_source(root, requested_revision)
    migration = build_migration(root, frozen_source_revision(root))
    if current_source != migration.source:
        raise MigrationError(
            f"source revision {current_revision} differs from frozen migration source "
            f"{migration.source_revision}"
        )
    return current_revision, migration


def _canonical_anchor(raw: bytes) -> tuple[str, str | None]:
    try:
        value = raw.decode("ascii")
    except UnicodeError as exc:
        raise MigrationError(f"state anchor is not ASCII: {raw!r}") from exc
    if DATE_RE.fullmatch(value):
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError as exc:
            raise MigrationError(f"state anchor has invalid timestamp {value!r}") from exc
        return value, None
    match = TIME_RE.fullmatch(value)
    if match is None:
        raise MigrationError(f"state anchor has unsupported timestamp {value!r}")
    year, month, day, hour, minute, second = match.groups()
    canonical = f"{year}-{month}-{day}T{hour}:{minute}:{second or '00'}Z"
    try:
        datetime.strptime(canonical, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise MigrationError(f"state anchor has invalid timestamp {value!r}") from exc
    compact = f"{year}{month}{day}T{hour}{minute}{second or '00'}"
    return canonical, compact


def segment_source(source: bytes) -> tuple[tuple[int, int, str, str | None], ...]:
    """Return reviewed pre-anchor coverage plus deterministic anchored-tail ranges."""
    anchors = list(ANCHOR_RE.finditer(source))
    boundaries = [match.start() for match in anchors]
    ranges: list[tuple[int, int, str, str | None]] = []

    # The pre-anchor section is deliberately one explicit range. Splitting its
    # unstructured prose would require semantic inference; one range is complete,
    # reviewable in the emitted manifest, and byte-lossless.
    first_anchor = boundaries[0] if boundaries else len(source)
    if first_anchor:
        ranges.append((0, first_anchor, "", None))

    for index, match in enumerate(anchors):
        start = match.start()
        end = boundaries[index + 1] if index + 1 < len(boundaries) else len(source)
        occurred_at, compact = _canonical_anchor(match.group(1))
        ranges.append((start, end, occurred_at, compact))

    if not ranges and not source:
        return ()
    expected = 0
    for start, end, _, _ in ranges:
        if start != expected or end <= start:
            raise MigrationError(
                f"proposed source ranges are not contiguous at byte {expected}: {start}-{end}"
            )
        expected = end
    if expected != len(source):
        raise MigrationError(
            f"proposed source ranges stop at byte {expected}, source has {len(source)} bytes"
        )
    return tuple(ranges)


def _csv_bytes(header: tuple[str, ...], rows: list[list[str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _assign_segments(
    ranges: tuple[tuple[int, int, str, str | None], ...]
) -> tuple[Segment, ...]:
    legacy_ordinal = 0
    timestamp_sequences: Counter[str] = Counter()
    segments: list[Segment] = []
    for start, end, occurred_at, compact in ranges:
        if compact is None:
            legacy_ordinal += 1
            event_id = f"STATE-LEGACY-{legacy_ordinal:06d}"
        else:
            timestamp_sequences[compact] += 1
            event_id = f"STATE-{compact}-{timestamp_sequences[compact]:03d}"
        period = occurred_at[:7] if compact is not None else "0000-00"
        evidence_path = f".agents/state-events/{period}/{event_id}.md"
        segments.append(Segment(event_id, start, end, occurred_at, evidence_path))
    return tuple(segments)


def _event_wrapper(segment: Segment, revision: str, payload: bytes) -> bytes:
    prefix = (
        f"# Imported state event {segment.event_id}\n"
        f"<!-- state-event: {segment.event_id} -->\n"
        f"<!-- legacy-source: {revision}:{SOURCE_PATH} "
        f"bytes {segment.start}-{segment.end} -->\n"
    ).encode("ascii")
    return prefix + state_record.LEGACY_BEGIN + payload + state_record.LEGACY_END


def _event_row(segment: Segment) -> list[str]:
    return [
        segment.event_id,
        segment.occurred_at,
        "legacy_import",
        "",
        "",
        "",
        "",
        "",
        "",
        segment.evidence_path,
        "",
        "",
        "",
    ]


def _build_indexes(segments: tuple[Segment, ...]) -> tuple[dict[str, bytes], bytes]:
    rows_by_period: dict[str, list[list[str]]] = {}
    for segment in segments:
        period = segment.evidence_path.split("/")[2]
        rows_by_period.setdefault(period, []).append(_event_row(segment))
    if not rows_by_period:
        rows_by_period["0000-00"] = []

    artifacts: dict[str, bytes] = {}
    manifest_rows: list[list[str]] = []
    for period, period_rows in sorted(rows_by_period.items()):
        groups: list[list[list[str]]] = []
        current: list[list[str]] = []
        for row in sorted(
            period_rows, key=lambda item: state_record.event_order_key(item[0])
        ):
            candidate = [*current, row]
            candidate_bytes = _csv_bytes(state_record.EVENT_HEADER, candidate)
            if current and (
                len(current) >= state_record.SHARD_MAX_EVENTS
                or len(candidate_bytes) > state_record.SHARD_MAX_BYTES
            ):
                groups.append(current)
                current = [row]
            else:
                current = candidate
        if current or not groups:
            groups.append(current)

        for sequence, rows in enumerate(groups, start=1):
            shard_id = f"{period}-{sequence:03d}"
            index_path = f".agents/state-index/{shard_id}.csv"
            prefix = f".agents/state-events/{period}/"
            index_bytes = _csv_bytes(state_record.EVENT_HEADER, rows)
            if len(index_bytes) > state_record.SHARD_MAX_BYTES:
                raise MigrationError(
                    f"one generated index row exceeds the shard size limit: {index_path}"
                )
            artifacts[index_path] = index_bytes
            manifest_rows.append(["1", shard_id, index_path, prefix])
    manifest = _csv_bytes(state_record.MANIFEST_HEADER, manifest_rows)
    if len(manifest) > state_record.MANIFEST_MAX_BYTES:
        raise MigrationError("generated state manifest exceeds the 64 KiB limit")
    return artifacts, manifest


def build_migration(root: Path, revision: str) -> Migration:
    full_revision, source = read_source(root, revision)
    segments = _assign_segments(segment_source(source))
    artifacts: dict[str, bytes] = {}
    migration_rows: list[list[str]] = []
    for segment in segments:
        payload = source[segment.start : segment.end]
        artifacts[segment.evidence_path] = _event_wrapper(
            segment, full_revision, payload
        )
        migration_rows.append(
            [
                segment.event_id,
                str(segment.start),
                str(segment.end),
                str(len(payload)),
                hashlib.sha256(payload).hexdigest(),
                segment.evidence_path,
            ]
        )
    artifacts[MANIFEST_PATH] = _csv_bytes(state_record.MIGRATION_HEADER, migration_rows)
    index_artifacts, root_manifest = _build_indexes(segments)
    artifacts.update(index_artifacts)
    artifacts[".agents/state.csv"] = root_manifest
    source_digest = hashlib.sha256(source).hexdigest()
    artifacts[SOURCE_PATH] = (
        "# Structured state record\n\n"
        "Current index: [.agents/state.csv](state.csv).\n\n"
        "Index shards: [.agents/state-index/](state-index/).\n\n"
        "Event evidence: [.agents/state-events/](state-events/).\n\n"
        "Migration coverage: [.agents/completed/state-migration-manifest.csv]"
        "(completed/state-migration-manifest.csv).\n\n"
        f"Frozen source SHA-256: `{source_digest}`.\n\n"
        "Frozen legacy source: "
        f"https://github.com/mudler/vllm.cpp/blob/{full_revision}/.agents/state.md\n"
    ).encode("ascii")
    return Migration(full_revision, source, segments, artifacts)


def _managed_paths(root: Path) -> set[str]:
    paths: set[str] = set()
    for pattern in (
        ".agents/state-index/*.csv",
        ".agents/state-events/**/*.md",
    ):
        paths.update(
            path.relative_to(root).as_posix() for path in root.glob(pattern) if path.is_file()
        )
    for relative in (SOURCE_PATH, ".agents/state.csv", MANIFEST_PATH):
        if (root / relative).is_file():
            paths.add(relative)
    return paths


def apply_migration(root: Path, migration: Migration) -> None:
    conflicts: list[str] = []
    for relative, expected in sorted(migration.artifacts.items()):
        path = root / relative
        if not path.exists():
            continue
        try:
            actual = path.read_bytes()
        except OSError as exc:
            conflicts.append(f"{relative}: cannot inspect existing path: {exc}")
            continue
        source_replacement = relative == SOURCE_PATH and actual == migration.source
        if actual != expected and not source_replacement:
            conflicts.append(f"{relative}: refusing to overwrite different bytes")
    unexpected = _managed_paths(root) - set(migration.artifacts)
    conflicts.extend(
        f"{relative}: refusing to retain unexpected generated output"
        for relative in sorted(unexpected)
    )
    if conflicts:
        raise MigrationError("\n".join(conflicts))

    for relative, expected in sorted(migration.artifacts.items()):
        path = root / relative
        if path.exists() and path.read_bytes() == expected:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(expected)


def _manifest_reconstruction_errors(root: Path, migration: Migration) -> list[str]:
    path = root / MANIFEST_PATH
    try:
        raw = path.read_bytes()
        reader = csv.reader(raw.decode("utf-8").splitlines(keepends=True), strict=True)
        header = tuple(next(reader, ()))
        rows = [row for row in reader if row]
    except (OSError, UnicodeError, csv.Error) as exc:
        return [f"{MANIFEST_PATH}: cannot parse migration manifest: {exc}"]
    if header != state_record.MIGRATION_HEADER:
        return [
            f"{MANIFEST_PATH}: expected header {state_record.MIGRATION_HEADER!r}, found {header!r}"
        ]

    errors: list[str] = []
    expected_start = 0
    reconstructed = bytearray()
    seen: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        location = f"{MANIFEST_PATH}:{line_number}"
        if len(row) != len(state_record.MIGRATION_HEADER):
            errors.append(f"{location}: expected six columns")
            continue
        event_id, start_s, end_s, count_s, digest, evidence_path = row
        if event_id in seen:
            errors.append(f"{location}: duplicate event ID {event_id}")
        seen.add(event_id)
        try:
            start, end, count = int(start_s), int(end_s), int(count_s)
        except ValueError:
            errors.append(f"{location}: byte offsets and counts must be integers")
            continue
        if start != expected_start:
            errors.append(
                f"{location}: range starts at {start}; expected contiguous byte {expected_start}"
            )
        if end < start:
            errors.append(f"{location}: range ends before it starts")
        try:
            payload = state_record.read_legacy_payload(root / evidence_path)
        except (OSError, ValueError) as exc:
            errors.append(f"{location}: cannot extract {evidence_path}: {exc}")
            expected_start = end
            continue
        if end - start != count or len(payload) != count:
            errors.append(f"{location}: payload byte count does not match range")
        actual_digest = hashlib.sha256(payload).hexdigest()
        if digest != actual_digest:
            errors.append(f"{location}: payload SHA-256 mismatch")
        reconstructed.extend(payload)
        expected_start = end
    if expected_start != len(migration.source):
        errors.append(
            f"migration ranges end at {expected_start}; source ends at {len(migration.source)}"
        )
    if bytes(reconstructed) != migration.source:
        errors.append("extracted payloads do not reconstruct the source byte-for-byte")
    return errors


def verify_migration(root: Path, migration: Migration) -> None:
    errors: list[str] = []
    for relative, expected in sorted(migration.artifacts.items()):
        path = root / relative
        try:
            actual = path.read_bytes()
        except OSError as exc:
            errors.append(f"{relative}: generated output is missing or unreadable: {exc}")
            continue
        if actual != expected:
            errors.append(f"{relative}: generated bytes differ from deterministic output")
    for relative in sorted(_managed_paths(root) - set(migration.artifacts)):
        errors.append(f"{relative}: unexpected generated output")
    errors.extend(_manifest_reconstruction_errors(root, migration))
    errors.extend(state_record.validate(root))
    if errors:
        raise MigrationError("\n".join(dict.fromkeys(errors)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True, metavar="REV")
    parser.add_argument("--output-root", required=True, type=Path, metavar="PATH")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = args.output_root.resolve()
    try:
        if args.apply:
            migration = build_migration(root, args.source_revision)
            apply_migration(root, migration)
            print(
                f"generated {len(migration.segments)} state events from "
                f"{migration.source_revision}"
            )
        else:
            requested_revision, migration = build_verification_migration(
                root, args.source_revision
            )
            verify_migration(root, migration)
            print(
                f"state migration is byte-exact for {len(migration.source)} source bytes "
                f"at {requested_revision} using frozen provenance "
                f"{migration.source_revision}"
            )
    except (MigrationError, OSError) as exc:
        print(f"state migration failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
