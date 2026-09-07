"""Diagnostic evidence validation, independent of a GPU or profiler install."""
import copy
import hashlib
import importlib.util
import tempfile
import unittest
from unittest import mock
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / "tools/bench/strix_kernel_trace/worker.py"
SPEC = importlib.util.spec_from_file_location("strix_trace", PATH)
trace = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trace)


def log():
    return "\n".join(
        f"vllm-cli: run={i}/4 finish_reason=length prompt_tokens=5 "
        f"completion_tokens=64 secs=2.000 tok_s=32.000\n"
        f"vllm-cli: run={i}/4 generate_start_unix={100 + i * 3}.000000 "
        f"generate_end_unix={102 + i * 3}.000000"
        for i in range(1, 5)
    )


class EvidenceTests(unittest.TestCase):
    def test_llama_refuses_short_or_different_prompt(self):
        text = ("common_perf_print: samplers time = 2.50 ms / 64 tokens\n"
                "common_perf_print: prompt eval time = 2.50 ms / 5 tokens\n"
                "common_perf_print: eval time = 100.00 ms / 63 runs")
        self.assertEqual(trace.parse_llama(text, 5)["completion_tokens"], 64)
        for bad in ("", text.replace("64 tokens", "63 tokens"),
                    text.replace("5 tokens", "6 tokens"), text + "\n" + text):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                trace.parse_llama(bad, 5)

    def test_archive_revision_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "source.tar"
            archive.write_bytes(b"archive")
            pin = {"archive": str(archive), "sha256": hashlib.sha256(b"archive").hexdigest(),
                   "revision": trace.LLAMA_REV}
            with mock.patch.object(trace.subprocess, "check_output", return_value=trace.LLAMA_REV + "\n"):
                trace.verify_archive(pin, "llamacpp")
            with mock.patch.object(trace.subprocess, "check_output", return_value="0" * 40 + "\n"):
                with self.assertRaisesRegex(ValueError, "revision"):
                    trace.verify_archive(pin, "llamacpp")
            for revision in ("10bf611e", "a" * 40):
                with self.assertRaises(ValueError):
                    trace.verify_archive(dict(pin, revision=revision), "llamacpp")

    def test_runtime_output_limit_and_literal_command(self):
        command = trace.container({"image": "test-image"}, Path("/tmp/test"), "app", ["$(touch x)"])
        self.assertEqual(command[-1], "$(touch x)")
        self.assertNotIn("--ulimit", command)
        command = trace.container({"image": "test-image"}, Path("/tmp/test"), "app", [], limit_output=True)
        self.assertIn("fsize=536870912:536870912", command)

    def test_artifact_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact"
            with self.assertRaises(FileNotFoundError):
                trace.verify_file(path, "0" * 64, 3)
            path.write_bytes(b"abc")
            digest = hashlib.sha256(b"abc").hexdigest()
            self.assertEqual(trace.verify_file(path, digest, 3), digest)
            with self.assertRaisesRegex(ValueError, "size"):
                trace.verify_file(path, digest, 4)
            with self.assertRaisesRegex(ValueError, "sha256"):
                trace.verify_file(path, "0" * 64, 3)

    def test_requires_complete_matching_generations(self):
        self.assertEqual(len(trace.parse_ours(log())), 4)
        for broken in (
            log().replace("run=4/4", "run=3/4"),
            log().replace("completion_tokens=64", "completion_tokens=63", 1),
            log().replace("prompt_tokens=5", "prompt_tokens=6", 1),
            log().replace("finish_reason=length", "finish_reason=stop", 1),
            log().replace("generate_end_unix=105", "generate_end_unix=102", 1),
            log().replace("secs=2.000", "secs=8.000", 1),
            log().split("vllm-cli: run=4/4")[0],
        ):
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                trace.parse_ours(broken)

    def test_clock_fold_uses_only_warm_windows(self):
        samples = [{"timestamp": t, "sclk_mhz": mhz}
                   for t, mhz in [(104, 999), (106.5, 100), (109.5, 200), (112.5, 300), (120, 999)]]
        result = trace.fold_clocks(trace.parse_ours(log()), samples)
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["sclk_mhz_median"], 200)
        with self.assertRaisesRegex(ValueError, "samples"):
            trace.fold_clocks(trace.parse_ours(log()), samples[:1])

    def test_pair_fold_rejects_missing_output_or_workload(self):
        legs = []
        for switch in trace.SWITCHES:
            for pair in range(3):
                for arm in ("default", "candidate"):
                    legs.append({"switch": switch, "pair": pair, "arm": arm,
                                 "runs": trace.parse_ours(log()), "output_sha256": "a" * 64,
                                 "returncode": 0})
        self.assertEqual(trace.fold_pairs(legs)["token_gate"], "FAIL (carried, not remeasured)")
        variants = [legs[:-1], legs + [legs[0]]]
        for field, value in [("output_sha256", "b" * 64), ("returncode", 1)]:
            bad = copy.deepcopy(legs)
            bad[0][field] = value
            variants.append(bad)
        bad = copy.deepcopy(legs)
        bad[0]["runs"][1]["prompt_tokens"] = 6
        variants.append(bad)
        for broken in variants:
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                trace.fold_pairs(broken)


if __name__ == "__main__":
    unittest.main()
