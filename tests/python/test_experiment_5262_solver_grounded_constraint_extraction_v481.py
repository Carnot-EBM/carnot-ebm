"""Tests for Exp 5262 solver-grounded constraint extraction.

Spec refs: REQ-VERIFY-5262, SCENARIO-VERIFY-5262.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5262_solver_grounded_constraint_extraction_v481 as mod


SPEC_PATH = Path("openspec/capabilities/verification/spec.md")


def _ready_preflight() -> dict[str, Any]:
    model_receipts = {
        "flagship_moe": {
            "role": "flagship_moe",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "preferred_quant": "Q4_K_M",
            "path": "/models/qwen.gguf",
            "runtime_ready": True,
            "status": "runtime_ready",
            "size_bytes": 123,
            "checksum_head_1m_sha256": "abc",
        },
        "flagship_dense": {
            "role": "flagship_dense",
            "hf_id": "unsloth/gemma-4-31B-it-GGUF",
            "preferred_quant": "Q4_K_M",
            "path": "/models/gemma31.gguf",
            "runtime_ready": True,
            "status": "runtime_ready",
            "size_bytes": 456,
            "checksum_head_1m_sha256": "def",
        },
        "middle_moe": {
            "role": "middle_moe",
            "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "preferred_quant": "Q4_K_M",
            "path": "/models/gemma26.gguf",
            "runtime_ready": False,
            "status": "optional_not_loaded",
            "size_bytes": 789,
            "checksum_head_1m_sha256": "ghi",
        },
    }
    return {
        "sota_runtime_ready": True,
        "sota_runtime_ready_principle": "ready through flagship_moe",
        "gpu_offload_receipts": {"value": {"llama_cpp": {"version": "0.3.29"}}},
        "model_receipts": {"value": model_receipts},
    }


def test_req_verify_5262_spec_declares_solver_grounded_contract() -> None:
    """REQ-VERIFY-5262: OpenSpec anchors the solver-grounded pilot contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5262") :]

    for marker in (
        "REQ-VERIFY-5262",
        "SCENARIO-VERIFY-5262",
        str(mod.RESULT_RELATIVE_PATH),
        "live_llm_inference_local_gguf_sota",
        "sota_runtime_ready=true",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "retired_veribmc_scope_reopened.value=false",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_req_verify_5262_parser_accepts_only_executable_constraint_ir() -> None:
    """REQ-VERIFY-5262: syntax normalization does not invent missing semantics."""

    parsed = mod.parse_constraint_ir(
        """
        Notes before the payload.
        ```json
        {
          "variables": {"x": {"type": "int", "min": 0, "max": 5}},
          "constraints": ["x >= 0", "x <= 5", "x % 2 == 0", "x > 3"]
        }
        ```
        """
    )

    assert parsed is not None
    assert parsed.variables == {"x": {"type": "int", "min": 0, "max": 5}}
    assert parsed.constraints == ("x >= 0", "x <= 5", "x % 2 == 0", "x > 3")
    assert parsed.normalization_notes == ("json_object_extracted",)
    assert mod.parse_constraint_ir("no json here") is None
    assert mod.parse_constraint_ir('{"constraints": "x >= 0"}') is None
    assert mod.parse_constraint_ir('{"constraints": ["x is even"]}') is not None


def test_req_verify_5262_checker_scores_satisfiability_and_counterexamples() -> None:
    """REQ-VERIFY-5262: deterministic checker finds false accepts and witnesses."""

    fixtures = {fixture.fixture_id: fixture for fixture in mod.fixture_set()}

    sat_candidate = mod.ConstraintIR(
        variables={"x": {"type": "int"}},
        constraints=("x >= 0", "x <= 5", "x % 2 == 0", "x > 3"),
        raw_json={},
        normalization_notes=(),
    )
    sat_result = mod.validate_fixture_constraints(fixtures["single_even_high"], sat_candidate)
    assert sat_result.solver_status == "sat"
    assert sat_result.matches_expected is True
    assert sat_result.false_accept is False
    assert sat_result.counterexample == {}

    missed_contradiction = mod.ConstraintIR(
        variables={"y": {"type": "int"}},
        constraints=("y >= 1", "y <= 4", "y % 2 == 0"),
        raw_json={},
        normalization_notes=(),
    )
    false_accept = mod.validate_fixture_constraints(fixtures["even_and_odd"], missed_contradiction)
    assert false_accept.solver_status == "sat"
    assert false_accept.matches_expected is False
    assert false_accept.false_accept is True
    assert false_accept.counterexample["y"] in (2, 4)

    overconstrained = mod.ConstraintIR(
        variables={"a": {"type": "int"}, "b": {"type": "int"}},
        constraints=("a >= 0", "a <= 3", "b >= 0", "b <= 3", "a + b == 5", "a == b"),
        raw_json={},
        normalization_notes=(),
    )
    rejected_sat = mod.validate_fixture_constraints(fixtures["small_pair_sum"], overconstrained)
    assert rejected_sat.solver_status == "unsat"
    assert rejected_sat.matches_expected is False
    assert rejected_sat.false_accept is False
    assert rejected_sat.counterexample == {"a": 2, "b": 3}

    bad_syntax = mod.ConstraintIR(
        variables={"x": {"type": "int"}},
        constraints=("__import__('os').system('true')",),
        raw_json={},
        normalization_notes=(),
    )
    syntax_result = mod.validate_fixture_constraints(fixtures["single_even_high"], bad_syntax)
    assert syntax_result.solver_status == "parse_error"
    assert syntax_result.matches_expected is False
    assert "unsupported expression" in syntax_result.error


def test_scenario_verify_5262_runs_live_path_with_injected_proposer_and_baseline(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5262: live extraction is solver-scored against a baseline."""

    result_path = tmp_path / "experiment_5262.json"
    scripted = {
        "single_even_high": {
            "variables": {"x": {"type": "int"}},
            "constraints": ["x >= 0", "x <= 5", "x % 2 == 0", "x > 3"],
        },
        "small_pair_sum": {
            "variables": {"a": {"type": "int"}, "b": {"type": "int"}},
            "constraints": ["a >= 0", "a <= 3", "b >= 0", "b <= 3", "a + b == 5", "a < b"],
        },
        "even_and_odd": {
            "variables": {"y": {"type": "int"}},
            "constraints": ["y >= 1", "y <= 4", "y % 2 == 0", "y % 2 == 1"],
        },
        "too_large_sum": {
            "variables": {"p": {"type": "int"}, "q": {"type": "int"}},
            "constraints": ["p >= 0", "p <= 2", "q >= 0", "q <= 2", "p + q == 5"],
        },
    }

    def fake_proposer(fixture: mod.ConstraintFixture) -> str:
        return json.dumps(scripted[fixture.fixture_id])

    artifact = mod.run_experiment(
        result_path=result_path,
        preflight_artifact=_ready_preflight(),
        proposal_fn=fake_proposer,
        commands_run=[{"command": "unit fixture", "outcome": "passed"}],
    )

    mod.validate_artifact(artifact)
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert "useful oracle-distinct signal" in artifact["honest_verdict"]["value"]
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["solver_grounded_extractor_ready"] is True
    assert artifact["constraint_validity_rate"]["value"] == 1.0
    assert artifact["false_accepts"]["value"] == 0
    assert artifact["counterexamples_found"]["value"] == 0
    assert artifact["baseline"]["validity_rate"] == 0.5
    assert artifact["baseline"]["false_accepts"] == 2
    assert artifact["retired_veribmc_scope_reopened"]["value"] is False
    assert artifact["preconditions_checked"]["value"]["deterministic_checker_available"] is True
    assert artifact["MODEL_SPECS"]["value"]["flagship_moe"]["role"] == "flagship_moe_candidate_generator"
    assert artifact["MODEL_SPECS"]["value"]["flagship_dense"]["role"] == (
        "flagship_dense_candidate_generator_or_cross_checker"
    )


def test_req_verify_5262_preconditions_and_schema_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-5262: missing preconditions and malformed artifacts are rejected."""

    run_via_gate = mod.run_pilot(
        result_path=tmp_path / "ready_gate.json",
        preflight_artifact=_ready_preflight(),
        proposal_fn=lambda fixture: json.dumps(
            {
                "variables": [name for name in fixture.gold_assignment] or ["z"],
                "constraints": ["z == 0"] if fixture.expected_status == "unsat" else [
                    f"{name} == {value}" for name, value in fixture.gold_assignment.items()
                ],
            }
        ),
        commands_run=[],
    )
    assert run_via_gate["honest_verdict"]["value"].startswith("complete:")

    blocked = mod.run_pilot(
        result_path=tmp_path / "blocked.json",
        preflight_artifact={"sota_runtime_ready": False},
        proposal_fn=lambda fixture: "{}",
        commands_run=[{"command": "unit blocked", "outcome": "passed"}],
        z3_module=object(),
    )

    assert blocked["honest_verdict"]["value"].startswith("blocked_")
    assert blocked["solver_grounded_extractor_ready"] is False
    assert blocked["preconditions_checked"]["value"]["exp5259_sota_runtime_ready"] is False
    mod.validate_artifact(blocked)

    unparseable = mod.run_experiment(
        result_path=tmp_path / "unparseable.json",
        preflight_artifact=_ready_preflight(),
        proposal_fn=lambda fixture: "not json",
        commands_run=[],
    )
    assert unparseable["honest_verdict"]["value"].startswith("complete:")
    assert "no useful oracle-distinct signal" in unparseable["honest_verdict"]["value"]
    assert unparseable["solver_grounded_extractor_ready"] is False
    assert unparseable["counterexamples_found"]["value"] == 4
    mod.validate_artifact(unparseable)

    broken = dict(unparseable)
    broken["solver_grounded_extractor_ready"] = "false"
    with pytest.raises(AssertionError, match="bare bool"):
        mod.validate_artifact(broken)

    broken = dict(unparseable)
    broken["retired_veribmc_scope_reopened"] = {
        "value": True,
        "principle": mod.FIELD_PRINCIPLES["retired_veribmc_scope_reopened"],
    }
    with pytest.raises(AssertionError, match="must remain false"):
        mod.validate_artifact(broken)

    broken = dict(unparseable)
    broken["inference_substrate"] = {
        "value": "local_sota_gguf_plus_deterministic_solver_feedback",
        "principle": mod.FIELD_PRINCIPLES["inference_substrate"],
    }
    with pytest.raises(AssertionError, match="live_llm_inference_local_gguf_sota"):
        mod.validate_artifact(broken)


def test_req_verify_5262_edge_branches_for_parser_checker_and_helpers(tmp_path: Path) -> None:
    """REQ-VERIFY-5262: deterministic edge branches stay bounded and explicit."""

    fixtures = {fixture.fixture_id: fixture for fixture in mod.fixture_set()}

    assert mod.parse_constraint_ir("{not json}") is None
    assert mod.parse_constraint_ir("{not json") is None
    assert mod.parse_constraint_ir("[1, 2, 3]") is None
    assert mod.parse_constraint_ir('{"variables": 3, "constraints": []}') is None
    assert mod.parse_constraint_ir('{"variables": [1], "constraints": []}') is None
    assert mod._normalize_variables({1: {"type": "int"}}) is None
    assert mod._normalize_variables("x") is None

    embedded = json.dumps(
        {
            "variables": ["x"],
            "constraints": ["x >= 0", "x <= 2"],
            "note": 'brace } and quote " inside',
        }
    )
    extracted = mod.parse_constraint_ir(f"prefix {embedded} suffix")
    assert extracted is not None
    assert extracted.normalization_notes == ("json_object_extracted",)

    prompt = mod.render_prompt(fixtures["single_even_high"])
    assert "satisfiable" in prompt
    assert "expected_status" not in prompt

    unavailable = mod.validate_fixture_constraints(
        fixtures["single_even_high"],
        mod.ConstraintIR(variables={}, constraints=(), raw_json={}, normalization_notes=()),
        z3_module=None,
    )
    assert unavailable.solver_status == "parse_error"
    assert unavailable.error == "z3_unavailable"

    bounded = mod.ConstraintIR(
        variables={"x": {"type": "int", "min": 0, "max": 5}},
        constraints=("x > 3", "x % 2 == 0"),
        raw_json={},
        normalization_notes=(),
    )
    assert mod.validate_fixture_constraints(fixtures["single_even_high"], bounded).matches_expected

    class UnknownSolver:
        def set(self, **_kwargs: Any) -> None:
            return None

        def add(self, *_args: Any) -> None:
            return None

        def check(self) -> str:
            return "unknown"

    class UnknownZ3:
        sat = "sat"
        unsat = "unsat"

        @staticmethod
        def Int(name: str) -> str:
            return name

        @staticmethod
        def IntVal(value: int) -> int:
            return value

        @staticmethod
        def Solver() -> UnknownSolver:
            return UnknownSolver()

    unknown = mod.validate_fixture_constraints(
        fixtures["single_even_high"],
        mod.ConstraintIR(variables={"x": {"type": "int"}}, constraints=("x == 4",), raw_json={}, normalization_notes=()),
        z3_module=UnknownZ3,
    )
    assert unknown.solver_status == "unknown"

    formulas = [
        "x >= 0 and x <= 5",
        "x < 0 or x > 0",
        "x - 1 >= 0",
        "x * 2 == 8",
        "-x <= 0",
        "true",
        "false",
        "True",
    ]
    env = {"x": mod._z3.Int("x")}
    for formula in formulas:
        assert mod._compile_formula(formula, env, mod._z3) is not None
    for formula, message in (
        ("x / 2 == 1", "unsupported arithmetic operator"),
        ("x != 0", "unsupported comparison operator"),
        ("missing == 1", "unknown variable"),
    ):
        with pytest.raises(ValueError, match=message):
            mod._compile_formula(formula, env, mod._z3)
    with pytest.raises(ValueError, match="unsupported boolean operator"):
        mod._compile_ast(ast.BoolOp(op=ast.NotEq(), values=[]), env, mod._z3)

    assert mod._variable_names(
        mod.ConstraintIR(variables={}, constraints=("x ==",), raw_json={}, normalization_notes=())
    ) == set()
    assert mod._aggregate([]) == {"validity_rate": 0.0, "false_accepts": 0, "counterexamples_found": 0}

    class TextValue:
        def as_long(self) -> int:
            raise TypeError("not an int")

        def __str__(self) -> str:
            return "text-value"

    class TextModel:
        def evaluate(self, _var: Any, *, model_completion: bool) -> TextValue:
            assert model_completion is True
            return TextValue()

    assert mod._model_assignment(TextModel(), {"x": object()}) == {"x": "text-value"}

    weird_specs = mod._model_specs_from_preflight({"model_receipts": {"value": {"flagship_moe": []}}})
    assert weird_specs["flagship_moe"]["runtime_status"] == "missing_receipt"

    assert mod._prior_veribmc_receipt(tmp_path)["found"] is False
    results = tmp_path / "results"
    results.mkdir()
    (results / "experiment_5238_veribmc_invalid.json").write_text("not json", encoding="utf-8")
    assert "error" in mod._prior_veribmc_receipt(tmp_path)

    assert "not ready" in mod._honest_verdict(
        False,
        True,
        {"validity_rate": 0.75, "false_accepts": 1},
        {"validity_rate": 0.5},
    )
