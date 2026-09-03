#!/usr/bin/env python3
"""Check that both Qwen3 oracle scripts select and narrate their mode."""

from __future__ import annotations

import ast
import builtins
import importlib.util
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str, path: Path) -> ModuleType:
    original_import = builtins.__import__

    def reject_vllm(module_name, *args, **kwargs):
        if module_name == "vllm" or module_name.startswith("vllm."):
            raise AssertionError(f"{path.name} imported vLLM during module loading")
        return original_import(module_name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    with mock.patch("builtins.__import__", side_effect=reject_vllm):
        spec.loader.exec_module(module)
    return module


def hard_codes_eager(path: Path) -> bool:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "LLM":
            continue
        for keyword in node.keywords:
            if keyword.arg == "enforce_eager":
                return isinstance(keyword.value, ast.Constant) and keyword.value.value is True
    return False


class OracleModeTests(unittest.TestCase):
    def assert_modes(self, filename: str, required_args: list[str]) -> None:
        path = ROOT / "scripts" / filename
        module = load_script(filename.replace("-", "_"), path)

        required_seams = ("_parse_args", "_llm_kwargs", "_mode_narration")
        if not all(hasattr(module, name) for name in required_seams):
            self.assertFalse(
                hard_codes_eager(path),
                f"{filename}: default LLM mode hard-codes enforce_eager=True",
            )
            self.fail(f"{filename}: pure mode-selection seam is absent")

        default_args = module._parse_args(required_args)
        default_kwargs = module._llm_kwargs(default_args)
        self.assertFalse(default_kwargs.get("enforce_eager", False))
        default_line = module._mode_narration(default_args)
        self.assertIn("production", default_line.lower())
        self.assertIn("enforce_eager=False", default_line)

        eager_args = module._parse_args([*required_args, "--enforce-eager"])
        eager_kwargs = module._llm_kwargs(eager_args)
        self.assertIs(eager_kwargs.get("enforce_eager"), True)
        eager_line = module._mode_narration(eager_args)
        self.assertIn("eager", eager_line.lower())
        self.assertIn("enforce_eager=True", eager_line)

    def test_oracle_capture_modes(self) -> None:
        self.assert_modes("qwen3-oracle-capture.py", [])

    def test_neartie_gap_modes(self) -> None:
        self.assert_modes(
            "qwen3-neartie-gap.py",
            ["--model", "model", "--golden-dir", "goldens"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
