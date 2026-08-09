#!/usr/bin/env python3
"""Parsing and validation primitives for the structured state record."""

from __future__ import annotations

import csv
import hashlib
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath


MANIFEST_HEADER = ("schema_version", "shard_id", "index_path", "event_prefix")
EVENT_HEADER = (
    "event_id",
    "occurred_at",
    "kind",
    "subject_ids",
    "phase",
    "outcome",
    "commit",
    "pr",
    "spec",
    "evidence_path",
    "supersedes",
    "summary",
    "next_action",
)

MANIFEST_MAX_BYTES = 64 * 1024
SHARD_MAX_BYTES = 256 * 1024
SHARD_MAX_EVENTS = 512
EVENT_MAX_BYTES = 32 * 1024
SUMMARY_MAX_CHARS = 200
NEXT_ACTION_MAX_CHARS = 240
SHARD_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{3}$")
NEW_EVENT_ID_RE = re.compile(r"^STATE-(\d{8}T\d{6})-\d{3}$")
LEGACY_EVENT_ID_RE = re.compile(r"^STATE-LEGACY-\d{6}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
PR_RE = re.compile(r"^pr:[1-9]\d*$")
SUBJECT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
KINDS = frozenset({"legacy_import", "checkpoint", "decision", "handoff", "correction"})
PHASES = frozenset(
    {"spec", "implementation", "review", "verification", "integration", "handoff", "operations"}
)
OUTCOMES = frozenset(
    {"started", "checkpoint", "passed", "failed", "blocked", "landed", "closed", "superseded"}
)
REQUIRED_SECTIONS = ("Context", "Outcome", "Evidence", "Next action")
LEGACY_BEGIN = b"<!-- legacy-payload:begin -->\n"
LEGACY_END = b"<!-- legacy-payload:end -->\n"
MIGRATION_HEADER = (
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
)
STUB_MAX_BYTES = 4 * 1024
LEGACY_LINK_RE = re.compile(
    r"https://github\.com/[^/\s]+/[^/\s]+/blob/([0-9a-f]{40})/\.agents/state\.md"
)


def event_order_key(event_id: str) -> tuple[int, str]:
    """Order unsortable legacy imports before timestamped event IDs."""
    return (0 if LEGACY_EVENT_ID_RE.fullmatch(event_id) else 1, event_id)


@dataclass(frozen=True)
class Shard:
    schema_version: str
    shard_id: str
    index_path: str
    event_prefix: str


@dataclass(frozen=True)
class Event:
    event_id: str
    occurred_at: str
    kind: str
    subject_ids: str
    phase: str
    outcome: str
    commit: str
    pr: str
    spec: str
    evidence_path: str
    supersedes: str
    summary: str
    next_action: str


def parse_manifest(root: Path) -> tuple[list[Shard], list[str]]:
    path = root / ".agents/state.csv"
    errors: list[str] = []
    try:
        if path.stat().st_size > MANIFEST_MAX_BYTES:
            return [], [f"{path}: manifest exceeds the 64 KiB limit"]
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle, strict=True)
            header = tuple(next(reader, ()))
            if header != MANIFEST_HEADER:
                return [], [f"{path}: expected manifest header {MANIFEST_HEADER!r}"]
            shards: list[Shard] = []
            for line_number, row in enumerate(reader, start=2):
                if not row:
                    continue
                location = f"{path}:{line_number}"
                if len(row) != len(MANIFEST_HEADER):
                    errors.append(
                        f"{location}: expected {len(MANIFEST_HEADER)} columns, "
                        f"found {len(row)}"
                    )
                    continue
                shard = Shard(*row)
                if any(any(control in field for control in ("\r", "\n", "\0")) for field in row):
                    errors.append(f"{location}: control character in manifest scalar")
                if shard.schema_version != "1":
                    errors.append(f"{location}: schema version must be 1")
                if SHARD_ID_RE.fullmatch(shard.shard_id) is None:
                    errors.append(f"{location}: invalid shard id {shard.shard_id!r}")
                expected_index = f".agents/state-index/{shard.shard_id}.csv"
                if shard.index_path != expected_index:
                    errors.append(
                        f"{location}: index path must be {expected_index!r}"
                    )
                period = shard.shard_id[:7]
                expected_prefix = f".agents/state-events/{period}/"
                if shard.event_prefix != expected_prefix:
                    errors.append(
                        f"{location}: event prefix must be {expected_prefix!r}"
                    )
                shards.append(shard)
    except (OSError, UnicodeError, csv.Error, TypeError) as exc:
        errors.append(f"{path}: cannot parse manifest: {exc}")
        return [], errors
    shard_ids = [shard.shard_id for shard in shards]
    if shard_ids != sorted(set(shard_ids)):
        errors.append(f"{path}: shard IDs must be unique and in increasing order")
    if errors:
        return [], errors
    return shards, errors


def _has_control(field: str) -> bool:
    return any(control in field for control in ("\r", "\n", "\0"))


def _valid_utc_timestamp(value: str) -> bool:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value) is None:
        return False
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    return True


def _valid_calendar_date(value: str) -> bool:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        return False
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def _validate_evidence(root: Path, event: Event, *, legacy: bool) -> list[str]:
    path = root / event.evidence_path
    errors: list[str] = []
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return [f"{path}: cannot read evidence: {exc}"]
    if not legacy and len(raw) > EVENT_MAX_BYTES:
        errors.append(f"{path}: post-cutover event exceeds the 32 KiB limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        return [*errors, f"{path}: evidence is not UTF-8: {exc}"]
    marker = f"<!-- state-event: {event.event_id} -->"
    marker_lines = [line for line in text.splitlines() if "<!-- state-event:" in line]
    if marker_lines != [marker]:
        errors.append(
            f"{path}: event must contain exactly one event marker matching {event.event_id}"
        )
    if legacy:
        try:
            read_legacy_payload(path)
        except ValueError as exc:
            errors.append(f"{path}: {exc}")
        return errors
    for section in REQUIRED_SECTIONS:
        match = re.search(
            rf"(?ms)^## {re.escape(section)}\s*\n(.+?)(?=^## |\Z)", text
        )
        if match is None or not match.group(1).strip():
            errors.append(f"{path}: required section {section!r} is missing or empty")
    return errors


def _event_scalar_errors(event: Event, shard: Shard, location: str) -> list[str]:
    errors: list[str] = []
    fields = tuple(getattr(event, name) for name in EVENT_HEADER)
    if any(_has_control(field) for field in fields):
        errors.append(f"{location}: control character in event scalar")
    legacy = event.kind == "legacy_import"
    new_match = NEW_EVENT_ID_RE.fullmatch(event.event_id)
    if not legacy and new_match is None:
        errors.append(f"{location}: invalid post-cutover event ID {event.event_id!r}")
    if (
        legacy
        and LEGACY_EVENT_ID_RE.fullmatch(event.event_id) is None
        and new_match is None
    ):
        errors.append(f"{location}: invalid legacy event ID {event.event_id!r}")
    if event.kind not in KINDS:
        errors.append(f"{location}: invalid kind {event.kind!r}")
    if legacy:
        if new_match is not None:
            if not _valid_utc_timestamp(event.occurred_at):
                errors.append(
                    f"{location}: timestamped legacy event requires an RFC 3339 UTC timestamp"
                )
            else:
                compact = datetime.strptime(
                    event.occurred_at, "%Y-%m-%dT%H:%M:%SZ"
                ).strftime("%Y%m%dT%H%M%S")
                if new_match.group(1) != compact:
                    errors.append(
                        f"{location}: event ID timestamp disagrees with occurred_at"
                    )
        elif (
            event.occurred_at
            and not _valid_calendar_date(event.occurred_at)
            and not _valid_utc_timestamp(event.occurred_at)
        ):
            errors.append(f"{location}: invalid legacy timestamp {event.occurred_at!r}")
        if event.subject_ids and any(
            SUBJECT_RE.fullmatch(subject) is None
            for subject in event.subject_ids.split(";")
        ):
            errors.append(f"{location}: invalid subject ID syntax")
        if event.phase and event.phase not in PHASES:
            errors.append(f"{location}: invalid phase {event.phase!r}")
        if event.outcome and event.outcome not in OUTCOMES:
            errors.append(f"{location}: invalid outcome {event.outcome!r}")
    else:
        if not _valid_utc_timestamp(event.occurred_at):
            errors.append(f"{location}: invalid timestamp {event.occurred_at!r}")
        elif new_match is not None:
            compact = datetime.strptime(
                event.occurred_at, "%Y-%m-%dT%H:%M:%SZ"
            ).strftime("%Y%m%dT%H%M%S")
            if new_match.group(1) != compact:
                errors.append(f"{location}: event ID timestamp disagrees with occurred_at")
        if not event.subject_ids:
            errors.append(f"{location}: subject IDs are required")
        elif any(
            SUBJECT_RE.fullmatch(subject) is None
            for subject in event.subject_ids.split(";")
        ):
            errors.append(f"{location}: invalid subject ID syntax")
        if event.phase not in PHASES:
            errors.append(f"{location}: invalid phase {event.phase!r}")
        if event.outcome not in OUTCOMES:
            errors.append(f"{location}: invalid outcome {event.outcome!r}")
        if not event.summary:
            errors.append(f"{location}: summary is required")
        if not event.next_action:
            errors.append(f"{location}: next action is required")
    if event.commit and COMMIT_RE.fullmatch(event.commit) is None:
        errors.append(f"{location}: commit must be a lowercase 40-character SHA")
    if event.pr and PR_RE.fullmatch(event.pr) is None:
        errors.append(f"{location}: PR must use pr:<number>")
    expected_evidence = f"{shard.event_prefix}{event.event_id}.md"
    if event.evidence_path != expected_evidence:
        errors.append(f"{location}: evidence path must be {expected_evidence!r}")
    if len(event.summary) > SUMMARY_MAX_CHARS:
        errors.append(f"{location}: summary exceeds {SUMMARY_MAX_CHARS} characters")
    if len(event.next_action) > NEXT_ACTION_MAX_CHARS:
        errors.append(
            f"{location}: next action exceeds {NEXT_ACTION_MAX_CHARS} characters"
        )
    return errors


def parse_events(root: Path, shards: list[Shard]) -> tuple[list[Event], list[str]]:
    events: list[Event] = []
    errors: list[str] = []
    for shard in shards:
        path = root / shard.index_path
        try:
            if path.stat().st_size > SHARD_MAX_BYTES:
                errors.append(f"{path}: index shard exceeds the 256 KiB limit")
                continue
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle, strict=True)
                header = tuple(next(reader, ()))
                if header != EVENT_HEADER:
                    errors.append(f"{path}: expected event header {EVENT_HEADER!r}")
                    continue
                shard_events: list[Event] = []
                for line_number, row in enumerate(reader, start=2):
                    if not row:
                        continue
                    location = f"{path}:{line_number}"
                    if len(row) != len(EVENT_HEADER):
                        errors.append(
                            f"{location}: expected {len(EVENT_HEADER)} columns, "
                            f"found {len(row)}"
                        )
                        continue
                    event = Event(*row)
                    errors.extend(_event_scalar_errors(event, shard, location))
                    errors.extend(
                        _validate_evidence(
                            root, event, legacy=event.kind == "legacy_import"
                        )
                    )
                    shard_events.append(event)
        except (OSError, UnicodeError, csv.Error, TypeError) as exc:
            errors.append(f"{path}: cannot parse event index: {exc}")
            continue
        if len(shard_events) > SHARD_MAX_EVENTS:
            errors.append(f"{path}: index shard exceeds the 512-event limit")
        event_ids = [event.event_id for event in shard_events]
        if event_ids != sorted(set(event_ids), key=event_order_key):
            errors.append(f"{path}: event IDs must be unique and in increasing order")
        events.extend(shard_events)
    if errors:
        return [], errors
    return events, []


def read_legacy_payload(path: Path) -> bytes:
    raw = path.read_bytes()
    if raw.count(LEGACY_BEGIN) != 1 or raw.count(LEGACY_END) != 1:
        raise ValueError("legacy payload must contain one begin and one end marker")
    start = raw.index(LEGACY_BEGIN) + len(LEGACY_BEGIN)
    end = raw.index(LEGACY_END, start)
    return raw[start:end]


def _git_bytes(root: Path, revision: str, path: str) -> bytes | None:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def _git_blob_oid(root: Path, revision: str, path: str) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", f"{revision}:{path}"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_revision_exists(root: Path, revision: str) -> bool:
    return subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _structured_history_floor(root: Path, base: str) -> str | None:
    """Return the first structured-state commit after a pre-cutover base."""
    result = subprocess.run(
        ["git", "rev-list", "--reverse", f"{base}..HEAD", "--", ".agents/state.csv"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        return None
    return next((line for line in result.stdout.splitlines() if line), None)


def _csv_rows(raw: bytes, header: tuple[str, ...], label: str) -> tuple[list[list[str]], list[str]]:
    try:
        text = raw.decode("utf-8")
        reader = csv.reader(text.splitlines(keepends=True))
        actual = tuple(next(reader, ()))
        if actual != header:
            return [], [f"{label}: expected header {header!r}"]
        return [row for row in reader if row], []
    except (UnicodeError, csv.Error) as exc:
        return [], [f"{label}: cannot parse CSV: {exc}"]


def _spec_errors(root: Path, event: Event) -> list[str]:
    if not event.spec:
        return []
    path = PurePosixPath(event.spec)
    allowed = (
        event.spec.startswith(".agents/specs/"),
        event.spec.startswith("docs/superpowers/specs/"),
    )
    if path.is_absolute() or ".." in path.parts or not any(allowed):
        return [f"{event.event_id}: spec must be repository-relative under an allowed specs directory"]
    target = root / event.spec
    try:
        resolved = target.resolve(strict=True)
    except OSError:
        return [f"{event.event_id}: spec does not resolve: {event.spec}"]
    if not resolved.is_relative_to(root):
        return [f"{event.event_id}: spec resolves outside the repository"]
    if not resolved.is_file():
        return [f"{event.event_id}: spec does not resolve: {event.spec}"]
    return []


def _relation_errors(root: Path, events: list[Event]) -> list[str]:
    errors: list[str] = []
    event_ids = [event.event_id for event in events]
    if event_ids != sorted(event_ids, key=event_order_key):
        errors.append("event IDs must be globally ordered across index shards")
    if len(event_ids) != len(set(event_ids)):
        errors.append("event IDs must be globally unique across index shards")

    evidence_paths = [event.evidence_path for event in events]
    if len(evidence_paths) != len(set(evidence_paths)):
        errors.append("each evidence path must be named by exactly one index row")
    indexed = set(evidence_paths)
    evidence_root = root / ".agents/state-events"
    existing = {
        path.relative_to(root).as_posix()
        for path in evidence_root.rglob("*.md")
        if path.is_file()
    } if evidence_root.exists() else set()
    for orphan in sorted(existing - indexed):
        errors.append(f"orphan evidence file has no index row: {orphan}")
    for missing in sorted(indexed - existing):
        errors.append(f"indexed evidence file is missing: {missing}")

    positions = {event.event_id: index for index, event in enumerate(events)}
    for index, event in enumerate(events):
        if event.subject_ids:
            subjects = event.subject_ids.split(";")
            if subjects != sorted(set(subjects)):
                errors.append(
                    f"{event.event_id}: subject IDs must be sorted and duplicate-free"
                )
        errors.extend(_spec_errors(root, event))
        if event.kind == "correction":
            if not event.supersedes:
                errors.append(f"{event.event_id}: correction requires supersedes")
            elif event.supersedes not in positions or positions[event.supersedes] >= index:
                errors.append(
                    f"{event.event_id}: supersedes must name an earlier event"
                )
        elif event.supersedes:
            errors.append(
                f"{event.event_id}: supersedes is forbidden outside correction events"
            )
    return errors


def _stub_and_migration_errors(root: Path, events: list[Event]) -> list[str]:
    errors: list[str] = []
    stub = root / ".agents/state.md"
    try:
        raw_stub = stub.read_bytes()
    except OSError as exc:
        return [f"compatibility stub cannot be read: {exc}"]
    if len(raw_stub) > STUB_MAX_BYTES:
        errors.append("compatibility stub exceeds the 4 KiB limit")
    try:
        stub_text = raw_stub.decode("utf-8")
    except UnicodeError as exc:
        return [*errors, f"compatibility stub is not UTF-8: {exc}"]
    for required in (
        ".agents/state.csv",
        ".agents/state-index/",
        ".agents/state-events/",
        ".agents/completed/state-migration-manifest.csv",
    ):
        if required not in stub_text:
            errors.append(f"compatibility stub must link {required}")
    links = LEGACY_LINK_RE.findall(stub_text)
    if len(links) != 1:
        errors.append("compatibility stub must contain one permanent legacy Git blob link")
        source = None
    else:
        source = _git_bytes(root, links[0], ".agents/state.md")
        if source is None:
            errors.append("compatibility stub legacy Git blob link does not resolve")

    manifest = root / ".agents/completed/state-migration-manifest.csv"
    try:
        raw_manifest = manifest.read_bytes()
    except OSError as exc:
        return [*errors, f"migration manifest cannot be read: {exc}"]
    rows, csv_errors = _csv_rows(raw_manifest, MIGRATION_HEADER, str(manifest))
    errors.extend(csv_errors)
    if csv_errors:
        return errors

    source_rows = [row for row in rows if row and row[0] == "source"]
    if len(source_rows) != 1 or not rows or rows[0][0] != "source":
        errors.append("migration manifest must begin with exactly one source provenance row")
    elif len(source_rows[0]) == len(MIGRATION_HEADER):
        (
            _, source_commit, source_blob, source_bytes_s, source_digest,
            *source_event_fields,
        ) = source_rows[0]
        if any(source_event_fields):
            errors.append("migration source provenance row cannot contain event fields")
        if not links or source_commit != links[0]:
            errors.append("migration source commit disagrees with the compatibility stub")
        expected_blob = _git_blob_oid(root, source_commit, ".agents/state.md")
        if COMMIT_RE.fullmatch(source_blob) is None or source_blob != expected_blob:
            errors.append("migration source blob does not match the frozen Git object")
        try:
            source_bytes = int(source_bytes_s)
        except ValueError:
            errors.append("migration source byte count must be an integer")
            source_bytes = -1
        if source is not None and source_bytes != len(source):
            errors.append("migration source byte count does not match the frozen source")
        if (
            re.fullmatch(r"[0-9a-f]{64}", source_digest) is None
            or source is not None
            and source_digest != hashlib.sha256(source).hexdigest()
        ):
            errors.append("migration source SHA-256 does not match the frozen source")

    legacy_events = {event.event_id: event for event in events if event.kind == "legacy_import"}
    seen: set[str] = set()
    expected_start = 0
    reconstructed = bytearray()
    for line_number, row in enumerate(rows, start=2):
        location = f"migration manifest:{line_number}"
        if len(row) != len(MIGRATION_HEADER):
            errors.append(f"{location}: expected {len(MIGRATION_HEADER)} columns")
            continue
        record_type, *fields = row
        if record_type == "source":
            continue
        if record_type != "event":
            errors.append(f"{location}: record type must be source or event")
            continue
        source_fields = fields[:4]
        if any(source_fields):
            errors.append(f"{location}: migration event row cannot contain source provenance")
        event_id, start_s, end_s, count_s, digest, evidence_path = fields[4:]
        if event_id in seen:
            errors.append(f"{location}: duplicate migration event ID {event_id}")
        seen.add(event_id)
        event = legacy_events.get(event_id)
        if event is None:
            errors.append(f"{location}: migration row must name one legacy_import event")
            continue
        if evidence_path != event.evidence_path:
            errors.append(f"{location}: migration evidence path disagrees with the index")
        try:
            start, end, count = int(start_s), int(end_s), int(count_s)
        except ValueError:
            errors.append(f"{location}: migration byte ranges must be integers")
            continue
        try:
            payload = read_legacy_payload(root / event.evidence_path)
        except (OSError, ValueError) as exc:
            errors.append(f"{location}: migration payload cannot be read: {exc}")
            continue
        if start != expected_start or end < start:
            errors.append(f"{location}: migration ranges must be contiguous and increasing")
        if end - start != count or count != len(payload):
            errors.append(f"{location}: migration payload byte count disagrees with its range")
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None or hashlib.sha256(payload).hexdigest() != digest:
            errors.append(f"{location}: migration payload hash mismatch")
        expected_start = end
        reconstructed.extend(payload)
    if seen != set(legacy_events):
        errors.append("migration manifest must cover every legacy_import event exactly once")
    if source is not None and bytes(reconstructed) != source:
        errors.append("migration manifest payloads do not reconstruct the frozen source")
    return errors


def _base_event_paths(root: Path, revision: str, manifest_rows: list[list[str]]) -> list[str]:
    paths: list[str] = []
    for row in manifest_rows:
        if len(row) != len(MANIFEST_HEADER):
            continue
        shard_raw = _git_bytes(root, revision, row[2])
        if shard_raw is None:
            continue
        event_rows, row_errors = _csv_rows(shard_raw, EVENT_HEADER, f"{revision}:{row[2]}")
        if row_errors:
            continue
        paths.extend(row[9] for row in event_rows if len(row) == len(EVENT_HEADER))
    return paths


def _git_history_errors(root: Path, base: str, shards: list[Shard]) -> list[str]:
    if not _git_revision_exists(root, base):
        return [f"Git base revision does not resolve: {base}"]
    base_manifest = _git_bytes(root, base, ".agents/state.csv")
    if base_manifest is None:
        floor = _structured_history_floor(root, base)
        if floor is None:
            return []
        base = floor
        base_manifest = _git_bytes(root, base, ".agents/state.csv")
        if base_manifest is None:
            return [f"structured history floor cannot be read: {base}"]
    errors: list[str] = []
    current_manifest_path = root / ".agents/state.csv"
    try:
        current_manifest = current_manifest_path.read_bytes()
    except OSError as exc:
        return [f"state manifest cannot be read for Git comparison: {exc}"]
    if not current_manifest.startswith(base_manifest):
        errors.append("state manifest must be byte-preserving append-only against base")
    base_rows, base_csv_errors = _csv_rows(
        base_manifest, MANIFEST_HEADER, f"{base}:.agents/state.csv"
    )
    errors.extend(base_csv_errors)
    latest = shards[-1].index_path if shards else ""
    for row in base_rows:
        if len(row) != len(MANIFEST_HEADER):
            continue
        path = row[2]
        base_bytes = _git_bytes(root, base, path)
        try:
            current_bytes = (root / path).read_bytes()
        except OSError as exc:
            errors.append(f"{path}: cannot compare index shard to base: {exc}")
            continue
        if base_bytes is None:
            errors.append(f"{path}: base index shard cannot be read")
        elif path == latest:
            if not current_bytes.startswith(base_bytes):
                errors.append(f"{path}: writable shard must be byte-preserving append-only")
        elif current_bytes != base_bytes:
            errors.append(f"{path}: sealed shard is immutable")
    for path in _base_event_paths(root, base, base_rows):
        base_bytes = _git_bytes(root, base, path)
        try:
            current_bytes = (root / path).read_bytes()
        except OSError:
            current_bytes = None
        if base_bytes is None or current_bytes != base_bytes:
            errors.append(f"{path}: immutable evidence changed or disappeared")
    return errors


def validate(root: Path, *, base: str | None = None) -> list[str]:
    """Return all structured-record validation failures rooted at *root*."""
    root = root.resolve()
    shards, errors = parse_manifest(root)
    if errors:
        return errors
    events, event_errors = parse_events(root, shards)
    errors.extend(event_errors)
    if event_errors:
        return errors
    errors.extend(_relation_errors(root, events))
    errors.extend(_stub_and_migration_errors(root, events))
    if base is not None:
        errors.extend(_git_history_errors(root, base, shards))
    return errors
