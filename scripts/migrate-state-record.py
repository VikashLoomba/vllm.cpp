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
class MigrationResult:
    source_revision: str
    source_blob: str
    source: bytes
    segments: tuple[Segment, ...]
    artifacts: dict[str, bytes]


# Kept as a source-compatible name for the reviewed frozen builder.
Migration = MigrationResult


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def read_source(
    root: Path, revision: str, source_path: str = SOURCE_PATH
) -> tuple[str, str, bytes]:
    resolved = _git(root, "rev-parse", "--verify", f"{revision}^{{commit}}")
    if resolved.returncode != 0:
        detail = resolved.stderr.decode("utf-8", errors="replace").strip()
        raise MigrationError(f"source revision {revision!r} does not resolve: {detail}")
    full_revision = resolved.stdout.decode("ascii").strip()
    blob = _git(root, "rev-parse", f"{full_revision}:{source_path}")
    if blob.returncode != 0:
        detail = blob.stderr.decode("utf-8", errors="replace").strip()
        raise MigrationError(
            f"source revision {full_revision} has no {source_path} blob: {detail}"
        )
    source_blob = blob.stdout.decode("ascii").strip()
    source = _git(root, "show", f"{full_revision}:{source_path}")
    if source.returncode != 0:
        detail = source.stderr.decode("utf-8", errors="replace").strip()
        raise MigrationError(
            f"source revision {full_revision} has no readable {source_path}: {detail}"
        )
    return full_revision, source_blob, source.stdout


def build_verification_migration(
    root: Path,
    requested_revision: str,
    source_path: str = SOURCE_PATH,
    epochs: tuple[state_record.MigrationEpoch, ...] | None = None,
) -> MigrationResult:
    """Build and validate the frozen migration plus each ordered append epoch."""

    configured = epochs if epochs is not None else state_record.MIGRATION_EPOCHS
    if not configured:
        raise MigrationError("migration epoch authority tuple is empty")
    snapshots: list[bytes] = []
    for index, epoch in enumerate(configured):
        revision, blob, source = read_source(root, epoch.commit, source_path)
        actual = (revision, blob, len(source), hashlib.sha256(source).hexdigest())
        expected = (epoch.commit, epoch.blob, epoch.byte_count, epoch.sha256)
        if actual != expected:
            raise MigrationError(
                f"migration epoch {index} Git source disagrees with tuple authority"
            )
        if index:
            ancestor = _git(root, "merge-base", "--is-ancestor", configured[index - 1].commit, epoch.commit)
            if ancestor.returncode != 0:
                raise MigrationError("migration epochs must be in ordered ancestor sequence")
            if not source.startswith(snapshots[-1]):
                raise MigrationError(
                    f"migration epoch {index} changes the cumulative source prefix"
                )
        snapshots.append(source)

    expected_start = configured[0].byte_count
    for index, epoch in enumerate(configured[1:], start=1):
        previous_size = configured[index - 1].byte_count
        if expected_start != previous_size:
            raise MigrationError("migration epoch boundary disagrees with prior snapshot")
        if not epoch.ranges:
            raise MigrationError(f"migration epoch {index} contributes no explicit ranges")
        for legacy_range in epoch.ranges:
            if legacy_range.start != expected_start:
                raise MigrationError(
                    f"migration range {legacy_range.event_id} leaves a gap or overlap at {expected_start}"
                )
            if legacy_range.end <= legacy_range.start or legacy_range.end > epoch.byte_count:
                raise MigrationError(
                    f"migration range {legacy_range.event_id} crosses its epoch boundary"
                )
            expected_start = legacy_range.end
        if expected_start != epoch.byte_count:
            raise MigrationError(
                f"migration epoch {index} ranges stop at {expected_start}, expected {epoch.byte_count}"
            )

    current_revision, current_blob, current_source = read_source(
        root, requested_revision, source_path
    )
    final = configured[-1]
    if current_blob != final.blob or current_source != snapshots[-1]:
        raise MigrationError(
            f"source revision {current_revision} differs from final migration epoch {final.commit}"
        )

    migration = build_migration(root, configured[0].commit, source_path)
    return _append_epochs(migration, configured, snapshots, source_path)


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


def _raw_csv_rows(raw: bytes) -> list[tuple[list[str], bytes]]:
    lines = raw.splitlines(keepends=True)
    reader = csv.reader((line.decode("utf-8") for line in lines), strict=True)
    records: list[tuple[list[str], bytes]] = []
    start = 0
    for row in reader:
        end = reader.line_num
        records.append((row, b"".join(lines[start:end])))
        start = end
    return records


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


def _event_wrapper(
    segment: Segment,
    revision: str,
    payload: bytes,
    source_path: str = SOURCE_PATH,
) -> bytes:
    prefix = (
        f"# Imported state event {segment.event_id}\n"
        f"<!-- state-event: {segment.event_id} -->\n"
        f"<!-- legacy-source: {revision}:{source_path} "
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


def build_migration(
    root: Path, revision: str, source_path: str = SOURCE_PATH
) -> MigrationResult:
    full_revision, source_blob, source = read_source(root, revision, source_path)
    segments = _assign_segments(segment_source(source))
    artifacts: dict[str, bytes] = {}
    source_digest = hashlib.sha256(source).hexdigest()
    migration_rows: list[list[str]] = [[
        "source",
        full_revision,
        source_blob,
        str(len(source)),
        source_digest,
        "",
        "",
        "",
        "",
        "",
        "",
    ]]
    for segment in segments:
        payload = source[segment.start : segment.end]
        artifacts[segment.evidence_path] = _event_wrapper(
            segment, full_revision, payload, source_path
        )
        migration_rows.append(
            [
                "event",
                "",
                "",
                "",
                "",
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
    artifacts[source_path] = (
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
    return MigrationResult(full_revision, source_blob, source, segments, artifacts)


def _source_row(epoch: state_record.MigrationEpoch) -> list[str]:
    return [
        "source",
        epoch.commit,
        epoch.blob,
        str(epoch.byte_count),
        epoch.sha256,
        "",
        "",
        "",
        "",
        "",
        "",
    ]


def _append_epochs(
    frozen: MigrationResult,
    epochs: tuple[state_record.MigrationEpoch, ...],
    snapshots: list[bytes],
    source_path: str,
) -> MigrationResult:
    artifacts = dict(frozen.artifacts)
    manifest_reader = csv.reader(
        artifacts[MANIFEST_PATH].decode("utf-8").splitlines(keepends=True),
        strict=True,
    )
    header = tuple(next(manifest_reader, ()))
    if header != state_record.MIGRATION_HEADER:
        raise MigrationError("frozen migration manifest header changed")
    manifest_rows = [row for row in manifest_reader if row]
    segments = list(frozen.segments)
    delta_rows_by_index: dict[str, list[list[str]]] = {}

    for epoch, snapshot in zip(epochs[1:], snapshots[1:]):
        manifest_rows.append(_source_row(epoch))
        for legacy_range in epoch.ranges:
            period = legacy_range.occurred_at[:7]
            evidence_path = (
                f".agents/state-events/{period}/{legacy_range.event_id}.md"
            )
            segment = Segment(
                legacy_range.event_id,
                legacy_range.start,
                legacy_range.end,
                legacy_range.occurred_at,
                evidence_path,
            )
            payload = snapshot[legacy_range.start : legacy_range.end]
            artifacts[evidence_path] = _event_wrapper(
                segment, epoch.commit, payload, source_path
            )
            manifest_rows.append(
                [
                    "event",
                    "",
                    "",
                    "",
                    "",
                    segment.event_id,
                    str(segment.start),
                    str(segment.end),
                    str(len(payload)),
                    hashlib.sha256(payload).hexdigest(),
                    segment.evidence_path,
                ]
            )
            index_path = f".agents/state-index/{period}-001.csv"
            delta_rows_by_index.setdefault(index_path, []).append(_event_row(segment))
            segments.append(segment)

    artifacts[MANIFEST_PATH] = _csv_bytes(state_record.MIGRATION_HEADER, manifest_rows)
    for index_path, delta_rows in delta_rows_by_index.items():
        try:
            existing = artifacts[index_path]
        except KeyError as exc:
            raise MigrationError(f"reviewed delta index does not exist: {index_path}") from exc
        reader = csv.reader(existing.decode("utf-8").splitlines(keepends=True), strict=True)
        index_header = tuple(next(reader, ()))
        if index_header != state_record.EVENT_HEADER:
            raise MigrationError(f"reviewed delta index header changed: {index_path}")
        rows = [row for row in reader if row]
        rows.extend(delta_rows)
        if [row[0] for row in rows] != sorted(
            (row[0] for row in rows), key=state_record.event_order_key
        ):
            raise MigrationError(f"reviewed delta rows are not ordered in {index_path}")
        artifacts[index_path] = _csv_bytes(state_record.EVENT_HEADER, rows)

    final = epochs[-1]
    if len(epochs) > 1:
        artifacts[source_path] = artifacts[source_path] + (
            "\nFinal imported source SHA-256: "
            f"`{final.sha256}`.\n\n"
            "Final imported legacy source: "
            f"https://github.com/mudler/vllm.cpp/blob/{final.commit}/{source_path}\n"
        ).encode("ascii")
    return MigrationResult(
        final.commit,
        final.blob,
        snapshots[-1],
        tuple(segments),
        artifacts,
    )


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
        appendable = (
            relative in {SOURCE_PATH, MANIFEST_PATH}
            or relative.startswith(".agents/state-index/")
        ) and expected.startswith(actual)
        if actual != expected and not source_replacement and not appendable:
            conflicts.append(f"{relative}: refusing to overwrite different bytes")
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
    source_rows = [row for row in rows if row and row[0] == "source"]
    if not source_rows or not rows or rows[0][0] != "source":
        errors.append("migration manifest must begin with source provenance")
    expected_start = 0
    reconstructed = bytearray()
    seen: set[str] = set()
    source_index = -1
    source_sizes: list[int] = []
    source_snapshots: list[bytes | None] = []
    for line_number, row in enumerate(rows, start=2):
        location = f"{MANIFEST_PATH}:{line_number}"
        if len(row) != len(state_record.MIGRATION_HEADER):
            errors.append(
                f"{location}: expected {len(state_record.MIGRATION_HEADER)} columns"
            )
            continue
        record_type, *fields = row
        if record_type == "source":
            source_index += 1
            if any(fields[4:]):
                errors.append(f"{location}: source row cannot contain event fields")
            try:
                size = int(fields[2])
            except ValueError:
                errors.append(f"{location}: source byte count must be an integer")
                size = -1
            snapshot = state_record._git_bytes(root, fields[0], SOURCE_PATH)
            source_sizes.append(size)
            source_snapshots.append(snapshot)
            if snapshot is None:
                errors.append(f"{location}: source revision cannot be read")
            else:
                blob = state_record._git_blob_oid(root, fields[0], SOURCE_PATH)
                if (
                    blob != fields[1]
                    or len(snapshot) != size
                    or hashlib.sha256(snapshot).hexdigest() != fields[3]
                ):
                    errors.append(f"{location}: source tuple disagrees with Git")
            if source_index:
                previous = source_snapshots[source_index - 1]
                if expected_start != source_sizes[source_index - 1]:
                    errors.append(f"{location}: source epoch begins at the wrong boundary")
                if previous is not None and bytes(reconstructed) != previous:
                    errors.append(f"{location}: payloads do not reconstruct preceding epoch")
            continue
        if record_type != "event":
            errors.append(f"{location}: record type must be source or event")
            continue
        if any(fields[:4]):
            errors.append(f"{location}: event row cannot contain source provenance")
        event_id, start_s, end_s, count_s, digest, evidence_path = fields[4:]
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
        if source_index < 0:
            errors.append(f"{location}: event precedes source provenance")
        elif end > source_sizes[source_index]:
            errors.append(f"{location}: range crosses its source epoch boundary")
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


def verify_migration(
    root: Path,
    migration: Migration,
    provenance: state_record.MigrationProvenance,
) -> None:
    errors: list[str] = []
    for relative, expected in sorted(migration.artifacts.items()):
        path = root / relative
        try:
            actual = path.read_bytes()
        except OSError as exc:
            errors.append(f"{relative}: generated output is missing or unreadable: {exc}")
            continue
        if relative == ".agents/state.csv":
            header: tuple[str, ...] = ()
            try:
                expected_records = _raw_csv_rows(expected)
                expected_header = (
                    tuple(expected_records[0][0]) if expected_records else ()
                )
                expected_rows = [
                    (row, raw) for row, raw in expected_records[1:] if row
                ]
                migration_indexes = {row[2] for row, _ in expected_rows}
                actual_records = _raw_csv_rows(actual)
                header = tuple(actual_records[0][0]) if actual_records else ()
                actual_migration_rows = [
                    raw
                    for row, raw in actual_records[1:]
                    if row and len(row) > 2 and row[2] in migration_indexes
                ]
                expected_migration_rows = [raw for _, raw in expected_rows]
            except (UnicodeError, csv.Error):
                expected_header = state_record.MANIFEST_HEADER
                actual_migration_rows = [actual]
                expected_migration_rows = [expected]
            differs = (
                header != state_record.MANIFEST_HEADER
                or expected_header != state_record.MANIFEST_HEADER
                or actual_migration_rows != expected_migration_rows
            )
        elif relative.startswith(".agents/state-index/"):
            header: tuple[str, ...] = ()
            try:
                expected_records = _raw_csv_rows(expected)
                expected_header = (
                    tuple(expected_records[0][0]) if expected_records else ()
                )
                expected_import_rows = [
                    raw for row, raw in expected_records[1:] if row
                ]
                actual_records = _raw_csv_rows(actual)
                header = tuple(actual_records[0][0]) if actual_records else ()
                actual_import_rows = [
                    raw
                    for row, raw in actual_records[1:]
                    if row and len(row) > 2 and row[2] == "legacy_import"
                ]
            except (UnicodeError, csv.Error):
                expected_header = state_record.EVENT_HEADER
                actual_import_rows = [actual]
                expected_import_rows = [expected]
            differs = (
                header != state_record.EVENT_HEADER
                or expected_header != state_record.EVENT_HEADER
                or actual_import_rows != expected_import_rows
            )
        else:
            differs = actual != expected
        if differs:
            errors.append(f"{relative}: generated bytes differ from deterministic output")
    errors.extend(_manifest_reconstruction_errors(root, migration))
    errors.extend(state_record.validate(root, migration_provenance=provenance))
    if errors:
        raise MigrationError("\n".join(dict.fromkeys(errors)))


def main(
    argv: list[str] | None = None,
    provenance: state_record.MigrationProvenance = state_record.FROZEN_MIGRATION_PROVENANCE,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True, metavar="REV")
    parser.add_argument("--output-root", required=True, type=Path, metavar="PATH")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    root = args.output_root.resolve()
    configured_epochs = None
    if provenance != state_record.FROZEN_MIGRATION_PROVENANCE:
        configured_epochs = (
            state_record.MigrationEpoch(
                provenance.commit,
                provenance.blob,
                provenance.byte_count,
                provenance.sha256,
                (),
            ),
        )
    try:
        requested_revision = read_source(root, args.source_revision)[0]
        if args.apply:
            migration = build_verification_migration(
                root, args.source_revision, epochs=configured_epochs
            )
            apply_migration(root, migration)
            print(
                f"generated {len(migration.segments)} state events from "
                f"{requested_revision} using frozen provenance {migration.source_revision}"
            )
        else:
            migration = build_verification_migration(
                root, args.source_revision, epochs=configured_epochs
            )
            verify_migration(root, migration, provenance)
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
