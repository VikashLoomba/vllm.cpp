"""Exercise the row comparator with explicitly synthetic validator inputs.

The two fixture reports preserve actual successful Q4_0 captures. The temporary
matrix uses their metadata with synthetic model bytes and repeated test tokens.
It is a validator test, not an oracle measurement or a model parity result.
"""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools" / "rocm_quant_gather"
FIXTURES = Path(__file__).parent / "fixtures" / "rocm_quant_gather"
sys.path.insert(0, str(TOOLS))
try:
    spec = importlib.util.spec_from_file_location("gather_compare", TOOLS / "compare_models.py")
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)
finally:
    sys.path.pop(0)


class CompareModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="gather-validator-")
        base = Path(cls.directory.name)
        cls.args = SimpleNamespace(**{name: base / name for name in
            ("native", "primary", "secondary", "fixtures", "config")})
        cls.args.native_log = base / "native.log"
        cls.args.primary_memory = base / "memory.json"
        for name in ("native", "primary", "secondary", "fixtures", "config"):
            getattr(cls.args, name).mkdir()
        (cls.args.secondary / "controls").mkdir()
        (cls.args.config / "config.json").write_text('{"synthetic_validator": true}\n')
        cls.token_control = json.loads((FIXTURES / "primary-token.json").read_text())
        cls.memory_control = json.loads((FIXTURES / "primary-memory.json").read_text())
        cls.primary_path = cls.args.primary / "Q4_0-p0-r0.json"
        cls.args.native_log.write_text(
            "1: Test command: /synthetic/test_capi_rocm_embedding_quant\n" +
            "1: [kv-alloc] block_size=16 num_kv_heads=1 head_size=64 dtype=2 "
            "page_size_bytes=4096 num_blocks=16\n" * 228)

        def capture(prefix):
            Path(str(prefix) + "-tokens.txt").write_text("47 19 4 20\n")
            Path(str(prefix) + "-logits-f32.bin").write_bytes(struct.pack("<512f", *([0.0] * 512)))

        for name in comparator.FORMATS:
            model = cls.args.fixtures / (name + ".gguf")
            model.write_text("synthetic validator model " + name)
            for repeat in range(3):
                for p, prompt in enumerate(comparator.PROMPTS):
                    stem = f"{name}-r{repeat}-p{p}"
                    for arm in ("quant", "dense"):
                        capture(cls.args.native / (stem + "-" + arm))
                    if name in comparator.SECONDARY:
                        prefix = cls.args.secondary / (stem + "-oracle")
                        capture(prefix)
                        Path(str(prefix) + ".log").write_text("physical_context=256\n")
                    elif name != "Q8_K":
                        report = deepcopy(cls.token_control)
                        report.update(model=comparator.seal(model),
                                      config=comparator.seal(cls.args.config / "config.json"),
                                      prompt=prompt, repeat=repeat)
                        (cls.args.primary / f"{name}-p{p}-r{repeat}.json").write_text(json.dumps(report))
        for oracle in ("stock", "fork"):
            for repeat in range(3):
                for p in range(2):
                    for arm in ("pristine", "overlay"):
                        capture(cls.args.secondary / "controls" / f"{oracle}-{arm}-r{repeat}-p{p}")
        cls.memory = deepcopy(cls.memory_control)
        cls.memory.update(model=comparator.seal(cls.args.fixtures / "Q4_0.gguf"),
                          config=comparator.seal(cls.args.config / "config.json"))
        cls.args.primary_memory.write_text(json.dumps(cls.memory))
        cls.primary = json.loads(cls.primary_path.read_text())

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def reject(self, path, report):
        original = path.read_bytes()
        try:
            path.write_text(json.dumps(report))
            with self.assertRaises((ValueError, KeyError)):
                comparator.compare(self.args)
        finally:
            path.write_bytes(original)

    def test_valid_synthetic_matrix(self):
        report = comparator.compare(self.args)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(report["cases"]), 114)

    def test_primary_exception_after_generation_is_rejected(self):
        report = deepcopy(self.primary)
        report.update(status="PENDING: primary model refused before qualification",
                      exception_type="RuntimeError", exception="observer failed after tokens")
        self.reject(self.primary_path, report)


if __name__ == "__main__":
    unittest.main()
