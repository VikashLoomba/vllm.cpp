#!/usr/bin/env python3
"""Report what a session has not resolved yet. (A)

This script REPORTS. It never asks, it never decides and it never writes,
because no harness-neutral mechanism exists for a shell script to run an
interactive prompt, and a hook injects text rather than conversing. The split
is fixed:

    this script   -> detect and report
    the agent     -> ask, using the interview in .agents/workflow.md
    agent-role.py -> make the answer a fact

    scripts/agent-onboard.py --probe            # human-readable state
    scripts/agent-onboard.py --probe --json     # machine-readable

Writing (`--env-set`, which records answered .env values) arrives in step 4;
this step reports only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


role_mod = _load("agent_role", "scripts/agent-role.py")

ENV_EXAMPLE = ROOT / ".env.example"
ENV_FILE = ROOT / ".env"


def _example_keys() -> tuple[str, ...]:
    """Keys the tracked example declares. The ONLY source of legal keys."""
    keys = []
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and "=" in line:
            keys.append(line.split("=", 1)[0])
    return tuple(keys)


ENV_KEYS = _example_keys()


def env_state_from_text(text: str) -> tuple[str, list[str]]:
    """Classify .env content: present | incomplete, plus the unset keys."""
    values = {}
    for line in text.splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    missing = [key for key in ENV_KEYS if not values.get(key)]
    return ("incomplete" if missing else "present"), missing


def env_state(path: Path = ENV_FILE) -> tuple[str, list[str]]:
    """Classify the .env file. A missing FILE is distinct from missing VALUES.

    A file that exists but cannot be read is a THIRD case: reporting it as
    absent would send the agent to create a file that is already there, and
    reporting it as complete would hide every unresolved value. It gets its
    own status, and like a missing file it resolves nothing.
    """
    if not path.exists():
        return "missing", list(ENV_KEYS)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return "unreadable", list(ENV_KEYS)
    return env_state_from_text(text)


def queue_state() -> tuple[list[str], str | None]:
    """The READY queue, plus why it is unavailable when it is.

    ready-for-helper.py's `queue()` IS the computation, so the probe calls it
    instead of re-parsing that script's prose: its listing truncates at 40 rows
    and its header line ("READY-FOR-HELPER queue: N row(s)") reads like a row
    ID to any token filter, so an empty queue would report a phantom row.

    A queue that could not be computed is NOT an empty queue, exactly as an
    unreadable .env is not an absent one, so the failure is carried out rather
    than swallowed. The probe still never raises: it reports and does not gate.
    """
    try:
        helper = sys.modules.get("ready_for_helper") or _load(
            "ready_for_helper", "scripts/ready-for-helper.py"
        )
        pickable, _ = helper.queue()
    except Exception as error:  # a broken record must not crash the probe
        return [], f"{type(error).__name__}: {error}"
    return [row.item_id for row in pickable], None


def ready_rows() -> list[str]:
    """The READY queue alone. Callers that must tell empty from broken apart
    use queue_state()."""
    return queue_state()[0]


def probe() -> dict:
    state = role_mod.resolve()
    status, missing = env_state()
    rows, queue_error = queue_state()
    return {
        "role": state.get("role"),
        "row": state.get("row"),
        # resolve() distinguishes "never declared" from "the operator lock is
        # held by another live session" and from "operator marker without a
        # held lock; re-claim". Dropping those makes this front door LESS
        # honest than the tool it wraps, and sends a session toward `claim
        # operator` when that will fail.
        "blocked_by_other_operator": bool(state.get("operator_held_by_other")),
        "reason": state.get("reason"),
        # resolve() now carries this (step 2). Still read with .get and still
        # rendered as a DEFAULT when absent: headless is never inferred, so a
        # state that carries no mode must not read as a declaration either.
        "mode": state.get("mode"),
        "env": status,
        "env_missing": missing,
        "queue": rows,
        "queue_error": queue_error,
    }


def render_probe(state: dict) -> str:
    role = state["role"] or "UNDECLARED"
    row = f" row={state['row']}" if state.get("row") else ""
    mode = state.get("mode") or "interactive (default, not declared)"
    if state.get("queue_error"):
        queue_line = f"queue: UNAVAILABLE ({state['queue_error']})"
    else:
        queue_line = f"queue: {len(state['queue'])} READY rows" + (
            f" — {', '.join(state['queue'][:5])}" if state["queue"] else ""
        )
    lines = [
        f"role: {role}{row}   mode: {mode}",
        f".env: {state['env']}"
        + (f" (unset: {', '.join(state['env_missing'])})" if state["env_missing"] else ""),
        queue_line,
    ]
    if state["role"] is None:
        if state.get("blocked_by_other_operator"):
            lines.append(
                "NOTE: the operator lock is held by another live session, so "
                "`claim operator` will fail. Take helper or read-only."
            )
        lines.append(
            "This session has not declared a role. Ask what the work is, then claim: "
            "a long campaign -> operator; one scoped change -> helper --row <ROW-ID>; "
            "just looking -> read-only. See .agents/workflow.md."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report unresolved session state.")
    parser.add_argument("--probe", action="store_true", help="report session state")
    parser.add_argument("--json", action="store_true", help="machine-readable probe")
    args = parser.parse_args(argv)

    state = probe()
    print(json.dumps(state, indent=2, sort_keys=True) if args.json else render_probe(state))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
