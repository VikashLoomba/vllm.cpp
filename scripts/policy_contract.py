#!/usr/bin/env python3
"""Shared parser for the repository policy and waiver registries."""

from __future__ import annotations

import csv
import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath


POLICY_HEADER = (
    "rule_id",
    "scope",
    "trigger",
    "requirement",
    "enforcement",
    "waiver_class",
    "procedure",
)
WAIVER_HEADER = (
    "waiver_id",
    "rule_id",
    "scope",
    "owner",
    "reason",
    "evidence",
    "expires",
)
WAIVER_CLASSES = frozenset({"never", "expiring", "migration-only"})
POLICY_MAX_BYTES = 16 * 1024
POLICY_MAX_RULES = 60
RULE_ID = re.compile(r"POL-[A-Z0-9]+(?:-[A-Z0-9]+)*\Z")
WAIVER_ID = re.compile(r"WAIVER-[A-Z0-9]+(?:-[A-Z0-9]+)*\Z")
SCOPE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
BROAD_SCOPE_VALUES = frozenset(
    {"all", "any", "everything", "global", "repo", "repository"}
)


@dataclass(frozen=True)
class PolicyRule:
    rule_id: str
    scope: str
    trigger: str
    requirement: str
    enforcement: str
    waiver_class: str
    procedure: str


@dataclass(frozen=True)
class Waiver:
    waiver_id: str
    rule_id: str
    scope: str
    owner: str
    reason: str
    evidence: str
    expires: dt.date


def _read_rows(path: Path, header: tuple[str, ...]) -> tuple[list[list[str]], list[str]]:
    errors: list[str] = []
    if not path.is_file():
        return [], [f"missing registry: {path.relative_to(path.parents[1])}"]
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return [], [f"{path.name}: not valid UTF-8: {exc}"]

    physical = text.splitlines()
    if any(not line.strip() for line in physical[1:]):
        errors.append(f"{path.name}: blank physical records are forbidden")

    try:
        reader = csv.reader(text.splitlines(keepends=True), strict=True)
        parsed = list(reader)
    except csv.Error as exc:
        return [], errors + [f"{path.name}: invalid RFC 4180 CSV: {exc}"]

    if not parsed:
        return [], errors + [f"{path.name}: missing header"]
    if tuple(parsed[0]) != header:
        errors.append(
            f"{path.name}: header must be exactly {','.join(header)}"
        )

    rows: list[list[str]] = []
    for number, row in enumerate(parsed[1:], start=2):
        if len(row) != len(header):
            errors.append(
                f"{path.name}:{number}: expected {len(header)} columns; got {len(row)}"
            )
            continue
        if any("\n" in field or "\r" in field for field in row):
            errors.append(
                f"{path.name}:{number}: multiline fields are forbidden; one physical record is required"
            )
        rows.append(row)
    return rows, errors


def _repo_relative(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        bool(value)
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in value
    )


def _is_exact_scope(scope: str) -> bool:
    """Return whether *scope* identifies one concrete waiver target."""

    try:
        kind, value = scope.split(":", 1)
    except ValueError:
        return False
    if not value or value.casefold() in BROAD_SCOPE_VALUES:
        return False
    if kind == "pr":
        return bool(re.fullmatch(r"[1-9][0-9]*", value))
    if kind == "commit":
        return bool(re.fullmatch(r"(?:[0-9A-Fa-f]{40}|[0-9A-Fa-f]{64})", value))
    if kind == "path":
        path = PurePosixPath(value)
        return (
            _repo_relative(value)
            and value != "."
            and not value.endswith("/")
            and path.as_posix() == value
        )
    if kind in {"task", "gate", "hardware"}:
        return bool(SCOPE_ID.fullmatch(value))
    return False


def _parse_policy(root: Path) -> tuple[dict[str, PolicyRule], list[str]]:
    path = root / ".agents/policy.csv"
    rows, errors = _read_rows(path, POLICY_HEADER)
    if path.is_file() and path.stat().st_size > POLICY_MAX_BYTES:
        errors.append(
            f"policy.csv exceeds the {POLICY_MAX_BYTES}-byte budget ({path.stat().st_size} bytes)"
        )
    if len(rows) > POLICY_MAX_RULES:
        errors.append(f"policy.csv has more than {POLICY_MAX_RULES} rules ({len(rows)})")

    rules: dict[str, PolicyRule] = {}
    for number, values in enumerate(rows, start=2):
        row = dict(zip(POLICY_HEADER, values, strict=True))
        rule_id = row["rule_id"]
        if not RULE_ID.fullmatch(rule_id):
            errors.append(f"policy.csv:{number}: malformed rule_id {rule_id!r}")
        if rule_id in rules:
            errors.append(f"policy.csv:{number}: duplicate rule_id {rule_id!r}")
        for field in ("scope", "trigger", "requirement", "enforcement", "procedure"):
            if not row[field].strip():
                errors.append(f"policy.csv:{number}: {field} is empty")
        if row["waiver_class"] not in WAIVER_CLASSES:
            errors.append(
                f"policy.csv:{number}: waiver_class {row['waiver_class']!r} is not one of "
                + ", ".join(sorted(WAIVER_CLASSES))
            )

        for checker_value in row["enforcement"].split(";"):
            checker = checker_value.strip()
            if (
                not _repo_relative(checker)
                or not checker.startswith("scripts/check-")
                or Path(checker).suffix not in {".py", ".sh"}
            ):
                errors.append(
                    f"policy.csv:{number}: enforcement {checker!r} is not a repo-relative checker entrypoint"
                )
            elif not (root / checker).is_file():
                errors.append(f"policy.csv:{number}: unknown enforcement path {checker!r}")

        procedure = row["procedure"]
        if not _repo_relative(procedure):
            errors.append(
                f"policy.csv:{number}: procedure {procedure!r} is not repo-relative"
            )
        elif Path(procedure).suffix != ".md" or not (root / procedure).is_file():
            errors.append(f"policy.csv:{number}: unknown procedure path {procedure!r}")

        if rule_id not in rules:
            rules[rule_id] = PolicyRule(**row)

    if not rows:
        errors.append("policy.csv must contain at least one rule")
    return rules, errors


def load_policy(root: Path) -> dict[str, PolicyRule]:
    rules, errors = _parse_policy(root)
    if errors:
        raise ValueError("\n".join(errors))
    return rules


def load_waivers(
    root: Path,
    rules: dict[str, PolicyRule],
    today: dt.date | None = None,
) -> list[Waiver]:
    current = today or dt.date.today()
    rows, errors = _read_rows(root / ".agents/waivers.csv", WAIVER_HEADER)
    waivers: list[Waiver] = []
    seen_ids: set[str] = set()
    seen_targets: set[tuple[str, str]] = set()
    for number, values in enumerate(rows, start=2):
        row = dict(zip(WAIVER_HEADER, values, strict=True))
        waiver_id = row["waiver_id"]
        if not WAIVER_ID.fullmatch(waiver_id):
            errors.append(f"waivers.csv:{number}: malformed waiver_id {waiver_id!r}")
        if waiver_id in seen_ids:
            errors.append(f"waivers.csv:{number}: duplicate waiver_id {waiver_id!r}")
        seen_ids.add(waiver_id)

        rule = rules.get(row["rule_id"])
        if rule is None:
            errors.append(
                f"waivers.csv:{number}: unknown rule {row['rule_id']!r}"
            )
        elif rule.waiver_class == "never":
            errors.append(
                f"waivers.csv:{number}: rule {row['rule_id']!r} cannot be waived"
            )

        scope = row["scope"]
        if any(character in scope for character in "*?[]{}"):
            errors.append(f"waivers.csv:{number}: wildcard scope {scope!r} is forbidden")
        if not _is_exact_scope(scope):
            errors.append(
                f"waivers.csv:{number}: {scope!r} is not an exact scope "
                "(task/PR/commit/path/gate/hardware required)"
            )
        target = (row["rule_id"], scope)
        if target in seen_targets:
            errors.append(
                f"waivers.csv:{number}: duplicate rule/scope {row['rule_id']!r} {scope!r}"
            )
        seen_targets.add(target)

        for field in ("owner", "reason", "evidence"):
            if not row[field].strip():
                errors.append(f"waivers.csv:{number}: {field} is empty")

        expiry: dt.date | None = None
        try:
            expiry = dt.date.fromisoformat(row["expires"])
        except ValueError:
            errors.append(
                f"waivers.csv:{number}: expires {row['expires']!r} is not a valid ISO date"
            )
        if expiry is not None and expiry <= current:
            errors.append(
                f"waivers.csv:{number}: waiver {waiver_id!r} expired on {expiry.isoformat()}"
            )

        if waiver_id and not _waiver_is_referenced(root, waiver_id):
            errors.append(
                f"waivers.csv:{number}: unused waiver {waiver_id!r}; cite it in its evidence or consuming record"
            )

        if expiry is not None:
            waivers.append(Waiver(**{**row, "expires": expiry}))

    if errors:
        raise ValueError("\n".join(errors))
    return waivers


def _waiver_is_referenced(root: Path, waiver_id: str) -> bool:
    """A waiver is live debt only when a non-code record names it explicitly."""

    needle = re.compile(
        rf"(?<![A-Z0-9-]){re.escape(waiver_id)}(?![A-Z0-9-])"
    )
    ignored_roots = {".git", "tests", "build", ".cache"}
    for path in root.rglob("*"):
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if not path.is_file() or relative == Path(".agents/waivers.csv"):
            continue
        if relative.parts and relative.parts[0] in ignored_roots:
            continue
        if path.suffix.lower() not in {".md", ".txt", ".csv", ".json", ".yml", ".yaml"}:
            continue
        try:
            if needle.search(path.read_text(encoding="utf-8")):
                return True
        except (OSError, UnicodeDecodeError):
            continue
    return False


def validate_policy(root: Path, *, schema_only: bool = False) -> list[str]:
    """Return every policy/waiver contract defect found under *root*.

    ``schema_only`` is the bootstrap mode.  Task 1 has no generated prose to
    check yet, so both modes intentionally enforce the complete registry
    schema and path contract; later cutover checks extend only full mode.
    """

    del schema_only
    rules, errors = _parse_policy(root)
    if errors:
        return errors
    try:
        load_waivers(root, rules)
    except (ValueError, KeyError) as exc:
        errors.extend(str(exc).splitlines())
    return errors
