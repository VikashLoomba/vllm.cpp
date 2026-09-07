"""Diagnostic evidence validation, independent of a GPU or profiler install."""
import copy
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
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
        for leg in legs:
            for run, seconds in zip(leg["runs"], [100, 2, 4, 6] if leg["arm"] == "default" else [80, 1, 2, 3]):
                run["seconds"] = seconds
        for pair in trace.fold_pairs(legs)["pairs"]:
            self.assertEqual((pair["default_seconds"], pair["candidate_seconds"], pair["default_over_candidate"]), (4, 2, 2))
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


class CommandTests(unittest.TestCase):
    def test_managed_command_cleanup_and_limits(self):
        for mode in ("success", "failure", "timeout", "output", "trace", "aggregate"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                argv = trace.container({"image": "test"}, folder, "app", [])
                real_popen = subprocess.Popen
                payload = {
                    "success": "pass", "failure": "raise SystemExit(7)",
                    "timeout": "import time; time.sleep(60)",
                    "output": "import sys,time; sys.stdout.write('x'*8192); sys.stdout.flush(); time.sleep(60)",
                    "trace": "from pathlib import Path; import time; Path('trace').write_bytes(b'x'*8192); time.sleep(60)",
                    "aggregate": "from pathlib import Path; import time; Path('trace-a').write_bytes(b'x'*600); Path('trace-b').write_bytes(b'x'*600); time.sleep(60)",
                }[mode]
                def launch(*args, **kwargs):
                    return real_popen([sys.executable, "-c", payload], cwd=folder, **kwargs)
                with (folder / "stdout").open("wb") as stdout, (folder / "stderr").open("wb") as stderr, mock.patch.object(subprocess, "Popen", side_effect=launch), mock.patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as cleanup:
                    if mode in ("timeout", "output", "trace", "aggregate"):
                        with self.assertRaises(subprocess.TimeoutExpired if mode == "timeout" else ValueError):
                            trace.managed_run(argv, stdout=stdout, stderr=stderr, timeout=.3, output_dir=folder, max_bytes=1024)
                    else:
                        result = trace.managed_run(argv, stdout=stdout, stderr=stderr, timeout=2, output_dir=folder, max_bytes=1024)
                        self.assertEqual(result.returncode, 7 if mode == "failure" else 0)
                self.assertEqual([c.args[0][len(trace.PODMAN)] for c in cleanup.call_args_list], ["stop", "rm"])
                name = argv[argv.index("--name") + 1]
                for call in cleanup.call_args_list:
                    self.assertEqual(call.args[0][-1], name)
                    self.assertLessEqual(call.kwargs["timeout"], 15)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.local = self.root / "local"
        self.local.mkdir()
        model = self.local / "model"
        model.write_bytes(b"model")
        self.device = self.root / "device"
        self.device.mkdir()
        (self.device / "pp_dpm_sclk").write_text("0: 100Mhz *")
        self.manifest = {"image": "test-image", "model": str(model), "clock_device": str(self.device),
                         "profiler_prefix": ["profiler", "{trace_dir}"], "sources": {}}
        for engine in ("vllmcpp", "llamacpp"):
            archive = self.root / (engine + ".tar")
            revision = trace.LLAMA_REV if engine == "llamacpp" else "a" * 40
            with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT, pax_headers={"comment": revision}) as tar:
                item = tarfile.TarInfo("source.txt")
                item.size = 1
                tar.addfile(item, io.BytesIO(b"x"))
            self.manifest["sources"][engine] = {"archive": str(archive), "sha256": trace.digest(archive), "revision": revision}
        self.state = {"local": str(self.local), "manifest": copy.deepcopy(self.manifest),
                      "image_id": "image-id", "binaries": {}, "model": str(model),
                      "vllmcpp": str(self.local / "vllm-cli"), "llamacpp": str(self.local / "llama-completion")}
        self.image_env = []
        self.image_id = "image-id"
        self.bad_llama = False
        self.bad_output = False
        self.fallback = False
        self.excess_output = False
        self.commands = []

    def invoke(self, phase="measure", tuning=None):
        manifest = self.root / "manifest.json"
        state = self.root / "state.json"
        manifest.write_text(json.dumps(self.manifest))
        state.write_text(json.dumps(self.state))
        argv = [str(PATH), "--phase", phase, "--manifest", str(manifest), "--state", str(state), "--output", str(self.root / "output")]
        real_check = subprocess.check_output
        real_run = subprocess.run
        real_popen = subprocess.Popen

        def check(argv, **kwargs):
            if argv[0] == "git":
                return real_check(argv, **kwargs)
            if "{{.Id}}" in argv:
                return self.image_id
            return json.dumps(self.image_env)

        def clocks(stop, path, device):
            path.write_text("\n".join(json.dumps({"timestamp": t, "sclk_mhz": 100}) for t in [104, 106.5, 109.5, 112.5]))

        def execute(argv, **kwargs):
            if argv[0] == "git":
                return real_run(argv, **kwargs)
            self.commands.append(argv)
            if "run" in argv:
                self.assertIn("--name", argv)
                if "--repeat" in argv or "-no-cnv" in argv:
                    self.assertIn("fsize=536870912:536870912", argv)
                    if "-no-cnv" in argv:
                        text = f"common_perf_print: samplers time = 2 ms / {63 if self.bad_llama else 64} tokens\ncommon_perf_print: prompt eval time = 2 ms / 5 tokens"
                    else:
                        self.assertIn("VT_OP_PROVIDER_STATS=1", argv)
                        repeat = int(argv[argv.index("--repeat") + 1])
                        text = log() if repeat == 4 else log().split("vllm-cli: run=2/4")[0].replace("/4", "/1")
                        if self.fallback:
                            text += "\n[vt reference-tier] op=test device=rocm has NO native kernel"
                    kwargs["stderr"].write(text.encode())
                    kwargs["stdout"].write(b"changed" if self.bad_output and any("VT_ROCM_Q8K_BLOCK=1" == x for x in argv) else b"text")
                if "--build" in argv:
                    target = Path(argv[argv.index("--build") + 1])
                    for suffix in ("examples/vllm-cli", "bin/llama-completion", "CMakeCache.txt"):
                        path = target / suffix
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(b"binary")
            return subprocess.CompletedProcess(argv, 0)

        # Execute the actual __main__ entry, while replacing only external work.
        source = PATH.read_text()
        namespace = {"__name__": "__main__", "__file__": str(PATH)}
        prefix, entry = source.rsplit('if __name__ == "__main__":', 1)
        with mock.patch.dict(os.environ, {"RC_DEVICE": "strix:gpu0", "RC_JOB_ID": "test", **(tuning or {})}, clear=True), mock.patch.object(sys, "argv", argv), mock.patch.object(subprocess, "check_output", side_effect=check), mock.patch.object(subprocess, "run", side_effect=execute):
            exec(compile(prefix, str(PATH), "exec"), namespace)
            namespace.update(MODEL_SIZE=5, MODEL_SHA=hashlib.sha256(b"model").hexdigest(), sample_clocks=clocks)
            namespace["tempfile"] = mock.Mock(mkdtemp=lambda **kwargs: tempfile.mkdtemp(dir=self.root))
            # Managed command uses the same subprocess boundary as build commands.
            if "managed_run" in namespace:
                original = namespace["managed_run"]
                def managed(argv, **kwargs):
                    if self.excess_output:
                        def launch(*args, **options):
                            return real_popen([sys.executable, "-c", "import sys,time; sys.stdout.write('x'*8192); sys.stdout.flush(); time.sleep(60)"], **options)
                        with mock.patch.object(subprocess, "Popen", side_effect=launch):
                            return original(argv, **dict(kwargs, max_bytes=1024, timeout=.3))
                    with mock.patch.object(subprocess, "Popen") as popen:
                        execute(argv, **{k: v for k, v in kwargs.items() if k in ("stdout", "stderr")})
                        popen.return_value.poll.return_value = 0
                        popen.return_value.wait.return_value = 0
                        return original(argv, **kwargs)
                namespace["managed_run"] = managed
            exec(compile('if __name__ == "__main__":' + entry, str(PATH), "exec"), namespace)
        return self.root / "output"

    def test_cli_build_and_measure(self):
        output = self.invoke("build")
        self.assertTrue((output / "build-state.json").is_file())
        output = self.invoke()
        self.assertEqual(len(json.loads((output / "paired-result.json").read_text())["pairs"]), 6)
        self.assertEqual(json.loads((output / "trace-status.json").read_text())["matched_counts"]["completion_tokens"], 64)
        self.assertTrue(any("rm" in c for c in self.commands))

    def test_cli_rejects_archive(self):
        self.manifest["sources"]["vllmcpp"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "sha256"):
            self.invoke("build")

    def test_cli_rejects_valid_archive_at_wrong_llama_pin(self):
        pin = self.manifest["sources"]["llamacpp"]
        pin["revision"] = "b" * 40
        with tarfile.open(pin["archive"], "w", format=tarfile.PAX_FORMAT, pax_headers={"comment": pin["revision"]}):
            pass
        pin["sha256"] = trace.digest(pin["archive"])
        with self.assertRaisesRegex(ValueError, "pin differs"):
            self.invoke("build")

    def test_cli_rejects_model(self):
        Path(self.state["model"]).write_bytes(b"wrong")
        with self.assertRaisesRegex(ValueError, "sha256"):
            self.invoke()

    def test_cli_rejects_manifest(self):
        self.manifest["image"] = "other"
        with self.assertRaisesRegex(ValueError, "manifest"):
            self.invoke()

    def test_cli_rejects_image(self):
        self.image_id = "other"
        with self.assertRaisesRegex(ValueError, "image changed"):
            self.invoke()

    def test_cli_rejects_host_tuning(self):
        for prefix in ("VT_", "GGML_", "HSA_", "HIP_", "ROCR_", "PYTORCH_"):
            with self.subTest(prefix=prefix), self.assertRaisesRegex(ValueError, "inherited tuning"):
                self.invoke(tuning={prefix + "SETTING": "1"})

    def test_cli_rejects_image_tuning(self):
        for name in ("VT_ROCM_Q8K_BLOCK", "GGML_SETTING", "HSA_SETTING", "HIP_SETTING", "ROCR_SETTING", "PYTORCH_SETTING"):
            self.image_env = ["PATH=/bin", name + "=1"]
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "image tuning"):
                self.invoke()

    def test_cli_rejects_matched_counts(self):
        self.bad_llama = True
        with self.assertRaisesRegex(ValueError, "workload"):
            self.invoke()

    def test_cli_rejects_pair_output(self):
        self.bad_output = True
        with self.assertRaisesRegex(ValueError, "output mismatch"):
            self.invoke()

    def test_cli_stops_excess_host_output(self):
        self.excess_output = True
        with self.assertRaisesRegex(ValueError, "aggregate output"):
            self.invoke()
        self.assertEqual([c[len(trace.PODMAN)] for c in self.commands], ["stop", "rm"])

    def test_cli_rejects_reference_fallback(self):
        self.fallback = True
        with self.assertRaisesRegex(ValueError, "leg .* failed"):
            self.invoke()


if __name__ == "__main__":
    unittest.main()
