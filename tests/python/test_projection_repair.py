"""Tests for deterministic arithmetic projection repair.

Spec: REQ-VERIFY-1147, SCENARIO-VERIFY-1147
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from carnot.repair.projection_repair import ArithmeticProjectionRepair
from carnot.verify.z3_math_verifier import Z3MathVerifier


SYNTHETIC_VIOLATIONS: tuple[tuple[str, int], ...] = (
    ("47 + 28 = 76", 75),
    ("3 * 7 = 22", 21),
    ("100 - 37 = 64", 63),
    ("144 / 12 = 11", 12),
    ("12 + 15 = 28", 27),
    ("8 * 9 = 73", 72),
    ("50 - 19 = 32", 31),
    ("81 / 9 = 8", 9),
    ("6 + 14 = 21", 20),
    ("17 * 4 = 69", 68),
    ("200 - 125 = 76", 75),
    ("45 / 5 = 8", 9),
    ("11 + 22 = 34", 33),
    ("13 * 6 = 79", 78),
    ("90 - 48 = 43", 42),
    ("64 / 8 = 7", 8),
    ("25 + 17 = 41", 42),
    ("7 * 8 = 55", 56),
    ("123 - 45 = 77", 78),
    ("99 / 11 = 8", 9),
)


def test_projection_repair_repairs_twenty_synthetic_violations() -> None:
    """SCENARIO-VERIFY-1147: 20 synthetic arithmetic violations are fixed."""

    repairer = ArithmeticProjectionRepair()
    verifier = Z3MathVerifier()

    for response, correct in SYNTHETIC_VIOLATIONS:
        fixed = repairer.repair(response, {"type": "arithmetic", "constraint": response})
        assert str(correct) in fixed
        assert verifier.score(fixed) == 0.0


def test_projection_repair_accepts_prompt_lhs_rhs_violation_shape() -> None:
    """REQ-VERIFY-1147: prompt-style lhs/rhs violation dicts repair responses."""

    repairer = ArithmeticProjectionRepair()
    fixed = repairer.repair("47 + 28 = 76", {"type": "arithmetic", "lhs": 75, "rhs": 76})

    assert fixed == "47 + 28 = 75"
    assert Z3MathVerifier().score(fixed) == 0.0


def test_projection_repair_repairs_reverse_equation_side() -> None:
    """REQ-VERIFY-1147: numeric-left equations can be projected too."""

    repairer = ArithmeticProjectionRepair()
    fixed = repairer.repair("76 = 47 + 28", {"type": "arithmetic"})

    assert fixed == "75 = 47 + 28"
    assert Z3MathVerifier().score(fixed) == 0.0


def test_projection_repair_uses_violation_source_text_inside_larger_response() -> None:
    """REQ-VERIFY-1147: violation metadata can identify the equation to patch."""

    repairer = ArithmeticProjectionRepair()
    response = "I first wrote 76.\n47 + 28 = 76\nFinal answer: 76."
    fixed = repairer.repair(response, {"type": "arithmetic", "source_text": "47+28=76"})

    assert "47 + 28 = 75" in fixed
    assert Z3MathVerifier().score(fixed) == 0.0


def test_projection_repair_falls_back_to_violation_numbers_without_equation() -> None:
    """REQ-VERIFY-1147: lhs/rhs values can patch answer-only text."""

    repairer = ArithmeticProjectionRepair()
    fixed = repairer.repair("Final answer: 76.", {"type": "arithmetic", "lhs": 75, "rhs": 76})

    assert fixed == "Final answer: 75."


def test_projection_repair_formats_fractional_projection() -> None:
    """REQ-VERIFY-1147: non-integer arithmetic projections remain parseable."""

    repairer = ArithmeticProjectionRepair()
    fixed = repairer.repair("10 / 4 = 2", {"type": "arithmetic", "constraint": "10 / 4 = 2"})

    assert fixed == "10 / 4 = 2.5"
    assert Z3MathVerifier().score(fixed) == 0.0


def test_projection_repair_leaves_two_expression_equations_unchanged() -> None:
    """REQ-VERIFY-1147: unsupported complex equality shapes are not guessed."""

    repairer = ArithmeticProjectionRepair()
    response = "2 + 2 = 1 + 4"

    assert repairer.repair(response, {"type": "arithmetic", "constraint": response}) == response


def test_projection_repair_returns_original_when_no_projection_available() -> None:
    """REQ-VERIFY-1147: non-arithmetic violations are left untouched."""

    repairer = ArithmeticProjectionRepair()
    response = "No arithmetic claim is present."

    assert repairer.repair(response, {"type": "semantic"}) == response


def test_z3_math_compat_module_imports_verifier() -> None:
    """REQ-VERIFY-1147: concrete prompt import path remains usable."""

    from python.carnot.verify.z3_math import Z3MathVerifier as CompatVerifier

    assert CompatVerifier().score("3 + 4 = 7") == 0.0


def test_experiment_1147_artifact_schema(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1147: Exp 1147 writes all required artifact fields."""

    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "experiment_1147_hardnet_projection_repair.py"
    spec = importlib.util.spec_from_file_location("experiment_1147", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    output_path = tmp_path / "experiment_1147.json"
    artifact = module.run_experiment(output_path=output_path)
    written = json.loads(output_path.read_text(encoding="utf-8"))

    required_fields = {
        "projection_repair_written",
        "module_path",
        "n_violations_tested",
        "projection_repair_accuracy",
        "projection_repair_latency_us",
        "prompt_repair_latency_s",
        "speedup_factor",
        "hardnet_projection_repair_written",
        "honest_verdict",
    }
    assert required_fields <= set(artifact)
    assert written == artifact
    assert artifact["projection_repair_written"] is True
    assert artifact["module_path"] == "python/carnot/repair/projection_repair.py"
    assert artifact["n_violations_tested"] == 20
    assert artifact["projection_repair_accuracy"] == 1.0
    assert artifact["projection_repair_latency_us"] > 0.0
    assert artifact["prompt_repair_latency_s"] > 0.0
    assert artifact["speedup_factor"] > 1.0
    assert artifact["hardnet_projection_repair_written"] is True
    assert artifact["honest_verdict"] == "projection_accurate_and_fast"
