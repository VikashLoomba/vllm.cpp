#!/usr/bin/env python3
"""`tools/oracle/ltx2_latent_dump.py`'s controls are exercised, not just written.

Two defects reached a queued `dgx` lease in that file and neither was detectable
by anything in this tree, because nothing executed it: `git grep -l
ltx2_latent_dump` matched the spec, the lease harness, and the file itself.

**Defect 1, an incomplete rename.** `write_latent` records the faithful copy
under `raw_native`; two reads in the reporting block spelled it `raw_bf16`. The
loop always runs, so the first record raised `KeyError` before the byte-length
assertion, before the frame control, before the manifest was written, and the
process exited rc 1 -- which is not one of the documented exit codes. The lease
would have paid ~42 GB of CIFS staging, a torch install and a ~94 s render to
reach it.

**Defect 2, a control that passes on zero evidence.** The frame check iterated
whatever `frames_dir.glob("frame_*.ppm")` returned and counted disagreements. An
empty glob yields no disagreements, `matched = n_frames - 0 = 0 - 0 = 0`, and
the driver printed "CONTROL PASSED: this run reproduces the committed reference
byte for byte" and returned 0. A decode that produced NO frames -- an `av`
failure, a truncated mp4, a wrong output path -- read as a passing control. That
is the failure the file's own docstring names: "a counter that could not be
written reads exactly like a counter that was never incremented".

The repair makes the control compare the observation set against the EXPECTATION
set: every frame the committed `SHA256SUMS` names must be present and must
agree. The expected count is derived from that file, never written here as a
literal, because a transcription cannot gate the thing it transcribes.

This suite needs no build, no GPU, no network, no torch and no checkpoint --
which is the point. Both defects were catchable on a laptop.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DUMP_PY = ROOT / "tools/oracle/ltx2_latent_dump.py"
SUMS = ROOT / "tests/parity/goldens/ltx2_oracle/SHA256SUMS"

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  ok   {msg}")
    else:
        print(f"  FAIL {msg}")
        failures.append(msg)


def load_module():
    spec = importlib.util.spec_from_file_location("ltx2_latent_dump_under_test", DUMP_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def latent_record(written: int, expected: int) -> dict:
    """A `write_latent`-shaped record, spelled the way `write_latent` spells it."""
    return {
        "shape": [1, 4, 8],
        "torch_dtype": "torch.bfloat16",
        "itemsize": 2,
        "expected_bytes": expected,
        "raw_native": {"path": "/tmp/video_latent_bfloat16.raw", "bytes": written,
                       "sha256": "0" * 64},
        "npy_f32": {"path": "/tmp/video_latent_f32.npy", "bytes": 4 * 32,
                    "sha256": "1" * 64},
        "stats_f32": {"min": -1.0, "max": 1.0, "mean": 0.0, "std": 1.0, "abs_sum": 1.0},
    }


mod = load_module()
source = DUMP_PY.read_text(encoding="utf-8")
tree = ast.parse(source)
functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

print("=== the reporting path is reachable and callable at all ===")
check("report_latents" in functions,
      "the latent reporting block is a callable function a test can drive")
check("check_frames" in functions,
      "the frame control is a callable function a test can drive")
check("committed_frame_digests" in functions,
      "the expected frame set is derived by a named function")

if not {"report_latents", "check_frames", "committed_frame_digests"} <= set(functions):
    print(f"FAILED ({len(failures)})")
    sys.exit(1)

print("=== D1: every key the reporting path READS, write_latent WRITES ===")
# Generic, not a spelling list: this is the check that would have caught the
# `raw_bf16` rename at rest, and catches the next one too.
written_keys: set[str] = set()
# Every dict literal in write_latent, nested ones included: the record it
# returns holds sub-dicts, and `rec["raw_native"]["bytes"]` reads a key from
# one of those. Collecting only the outer assignment would report `bytes` as
# stray. This stays strict about what matters -- `raw_bf16` appears in no dict
# literal in that function, at any depth, which is the whole defect.
for node in ast.walk(functions["write_latent"]):
    if isinstance(node, ast.Dict):
        for key in node.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                written_keys.add(key.value)
read_keys: set[str] = set()
for node in ast.walk(functions["report_latents"]):
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        if isinstance(node.slice.value, str):
            read_keys.add(node.slice.value)
check(bool(written_keys), "write_latent's record literal was parsed")
check(bool(read_keys), "report_latents' record reads were parsed")
unknown = sorted(read_keys - written_keys)
check(not unknown,
      f"report_latents reads no key write_latent does not write (stray: {unknown})")

print("=== D1: the reporting path runs instead of raising ===")
try:
    rc = mod.report_latents({"video": latent_record(64, 64)})
    check(rc == 0, "a full-length latent record reports rc 0")
except Exception as exc:  # noqa: BLE001 - the exception IS the defect
    check(False, f"a full-length latent record reports rc 0 (raised {type(exc).__name__}: {exc})")
try:
    rc = mod.report_latents({"video": latent_record(32, 64)})
    check(rc == 62, "a SHORT-WRITE latent record reports rc 62")
except Exception as exc:  # noqa: BLE001
    check(False, f"a SHORT-WRITE latent record reports rc 62 (raised {type(exc).__name__}: {exc})")

print("=== the committed expectation is real and is derived, not transcribed ===")
committed = mod.read_committed_digests(SUMS)
frames = mod.committed_frame_digests(committed)
check(len(committed) > len(frames),
      "SHA256SUMS carries non-frame rows too (mp4, manifest, audio)")
check(len(frames) > 0, "SHA256SUMS names at least one frame_*.ppm")
independent = len([
    line for line in SUMS.read_text().splitlines()
    if re.match(r"^[0-9a-f]{64}\s+frame_\d+\.ppm$", line.strip())
])
check(len(frames) == independent,
      f"the derived frame count ({len(frames)}) equals the file's own frame rows ({independent})")
check(all(re.fullmatch(r"frame_\d+\.ppm", n) for n in frames),
      "every derived expectation key is a frame file name")


def write_frames(root: Path, payloads: dict[str, bytes]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, blob in payloads.items():
        (root / name).write_bytes(blob)
    return root


with tempfile.TemporaryDirectory(prefix="ltx2-latent-dump-test-") as tmp:
    tmp = Path(tmp)
    # A synthetic three-frame expectation. If the expected count were hardcoded
    # to the committed 25, this leg could not pass -- which is what proves the
    # count is derived rather than transcribed.
    blobs = {f"frame_{i:06d}.ppm": f"frame {i}".encode() for i in range(3)}
    synthetic = {name: sha256_bytes(blob) for name, blob in blobs.items()}
    synthetic["upstream-render.mp4"] = "9" * 64

    print("=== D2: ZERO decoded frames is a FAILING control, not a passing one ===")
    empty = write_frames(tmp / "empty", {})
    res = mod.check_frames(empty, synthetic, 0)
    check(res["ok"] is False, "a decode that produced no frames does NOT pass the control")
    check(res["matched"] == 0, "zero frames matched")
    check(len(res["mismatches"]) == len(blobs),
          "every committed frame is reported MISSING rather than silently skipped")

    print("=== D2: a complete, correct decode PASSES ===")
    good = write_frames(tmp / "good", blobs)
    res = mod.check_frames(good, synthetic, len(blobs))
    check(res["ok"] is True, "all committed frames present and agreeing passes the control")
    check(res["matched"] == len(blobs), f"matched == {len(blobs)}, the derived expectation")
    check(res["mismatches"] == [], "no mismatches on a clean run")

    print("=== D2: a PARTIAL decode fails, even though what appeared agreed ===")
    partial = write_frames(tmp / "partial", dict(list(blobs.items())[:2]))
    res = mod.check_frames(partial, synthetic, 2)
    check(res["ok"] is False, "a decode missing one committed frame does NOT pass")
    check(res["matched"] == 2, "the two frames that did appear are still counted as matched")
    check(any(m["frame"] == "frame_000002.ppm" for m in res["mismatches"]),
          "the missing frame is named in the mismatches")

    print("=== D2: a WRONG digest fails ===")
    wrong = write_frames(tmp / "wrong", {**blobs, "frame_000001.ppm": b"perturbed"})
    res = mod.check_frames(wrong, synthetic, len(blobs))
    check(res["ok"] is False, "a frame whose digest disagrees does NOT pass")
    check(any(m["frame"] == "frame_000001.ppm" for m in res["mismatches"]),
          "the disagreeing frame is named in the mismatches")

    print("=== D2: an UNLISTED extra frame fails ===")
    extra = write_frames(tmp / "extra", {**blobs, "frame_000009.ppm": b"unexpected"})
    res = mod.check_frames(extra, synthetic, len(blobs) + 1)
    check(res["ok"] is False, "a decoded frame the committed file does not name does NOT pass")

    print("=== D2: an expectation with NO frame rows is not a pass either ===")
    res = mod.check_frames(good, {"upstream-render.mp4": "9" * 64}, len(blobs))
    check(res["ok"] is False, "an expectation naming zero frames cannot pass vacuously")
    check(res["committed_frames"] == 0, "the empty frame expectation is reported as zero")

print("=== both controls are REACHED from main(), not merely defined ===")
called = {
    node.func.id
    for node in ast.walk(functions["main"])
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
}
check("report_latents" in called, "main() calls report_latents")
check("check_frames" in called, "main() calls check_frames")

if failures:
    print(f"FAILED ({len(failures)})")
    sys.exit(1)
print("PASSED")
