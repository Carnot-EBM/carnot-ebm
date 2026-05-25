"""Tests for Exp 3071 VERGE-style MCS SMT correction pilot.

Spec refs: REQ-VERIFY-3071,
           SCENARIO-VERIFY-3071,
           SCENARIO-VERIFY-3071-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import verge_mcs_smt_correction_pilot_v1 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
SCRIPT_PATH = REPO_ROOT / "scripts" / exp.SCRIPT_FILENAME


class FakeClock:
    def __init__(self) -> None:
        self.value = 30.0

    def __call__(self) -> float:
        self.value += 0.25
        return self.value


class FakeLlama:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.closed = False

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["temperature"] == 0.0
        assert kwargs["seed"] == exp.DEFAULT_SEED
        if "sum-total-high" in prompt:
            text = '{"fixture_id":"sum-total-high","candidate":{"a":2,"b":3,"total":5}}'
        elif "sum-total-missing" in prompt:
            text = '{"fixture_id":"sum-total-missing","candidate":{"a":4,"b":1,"total":5}}'
        elif "bounded-x-high" in prompt:
            text = '{"fixture_id":"bounded-x-high","candidate":{"x":10}}'
        elif "difference-delta-wrong" in prompt:
            text = '{"fixture_id":"difference-delta-wrong","candidate":{"left":9,"right":4,"delta":99}}'
        elif "weighted-score-low" in prompt:
            text = '{"fixture_id":"weighted-score-low","candidate":{"p":2,"q":5,"score":19}}'
        else:
            text = "{}"
        return {"choices": [{"text": text}]}

    def close(self) -> None:
        self.closed = True


class FailingLlama:
    def __init__(self, **_kwargs: Any) -> None:
        raise RuntimeError("load failed")


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.ARTIFACT_FILENAME,
        rows_path=tmp_path / exp.PILOT_ROWS_REL_PATH,
        tests_run=("pytest focused",),
    )


def _model_path(tmp_path: Path) -> Path:
    path = tmp_path / "models" / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"tiny fake gguf for tests")
    return path


def _resolve_one_model(path: Path) -> exp.ResolveGgufFn:
    def resolve(hf_id: str, preferred_quant: str = "Q4_K_M") -> str | None:
        assert preferred_quant == "Q4_K_M"
        if hf_id == "unsloth/gemma-4-26B-A4B-it-GGUF":
            return str(path)
        return None

    return resolve


def test_req_verify_3071_spec_and_script_anchor_exists() -> None:
    """REQ-VERIFY-3071: the correction-feedback pilot is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3071" in spec
    assert "SCENARIO-VERIFY-3071" in spec
    assert "SCENARIO-VERIFY-3071-BLOCKED" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert "mcs_feedback_ready" in spec
    assert "correction_subset_useful_count" in spec
    assert SCRIPT_PATH.exists()


def test_scenario_verify_3071_exact_feedback_repairs_tiny_fixtures() -> None:
    """SCENARIO-VERIFY-3071: Z3 emits useful MCS/refinement feedback."""

    fixtures = exp.build_correction_fixtures()
    by_id = {fixture["fixture_id"]: fixture for fixture in fixtures}

    assert 4 <= len(fixtures) <= 8
    assert (
        exp.validate_candidate_with_z3(
            by_id["sum-total-valid"], by_id["sum-total-valid"]["candidate"]
        )["valid"]
        is True
    )

    feedback_rows = [
        exp.generate_correction_feedback(fixture)
        for fixture in fixtures
        if fixture["fixture_id"] != "sum-total-valid"
    ]
    assert len(feedback_rows) == 5
    assert {row["feedback_type"] for row in feedback_rows} == {"mcs", "refinement"}
    assert any(row["fixture_id"] == "sum-total-missing" for row in feedback_rows)
    assert all(row["exact_authority"] == "z3_solver" for row in feedback_rows)
    assert all(row["correction_subset"]["suggested_assignments"] for row in feedback_rows)

    bad_total = exp.validate_candidate_with_z3(
        by_id["sum-total-high"], by_id["sum-total-high"]["candidate"]
    )
    assert bad_total["valid"] is False
    assert bad_total["solver_status"] == "unsat"

    solver_rows = [exp.solver_only_repair(row, by_id[row["fixture_id"]]) for row in feedback_rows]
    assert all(row["valid"] for row in solver_rows)
    assert all(row["exact_checked"] for row in solver_rows)


def test_scenario_verify_3071_live_pilot_writes_required_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3071: live repair proposals are solver-validated and counted."""

    model_path = _model_path(tmp_path)
    artifact = exp.run_experiment(
        _config(tmp_path),
        resolve_gguf_func=_resolve_one_model(model_path),
        llama_factory=FakeLlama,
        monotonic=FakeClock(),
        repo_commit_func=lambda _: "test-commit",
    )
    saved = json.loads((tmp_path / "results" / exp.ARTIFACT_FILENAME).read_text("utf-8"))
    rows = exp.load_jsonl(tmp_path / artifact["pilot_rows_path"])

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["mcs_feedback_ready"] is True
    assert artifact["formal_fallback_preserved"] is True
    assert artifact["mcs_count"] == 5
    assert artifact["guided_success_count"] == 4
    assert artifact["solver_only_success_count"] == 5
    assert artifact["invalid_llm_proposal_count"] == 1
    assert artifact["correction_subset_useful_count"] == 5
    assert artifact["exact_solver_path"] == exp.EXACT_SOLVER_PATH
    assert artifact["models_used"] == ["unsloth/gemma-4-26B-A4B-it-GGUF"]
    assert artifact["model_specs"][0]["model_path"] == str(model_path)
    assert artifact["legacy_smoke_only_used"] is False
    assert len(artifact["prompt_hashes"]) == artifact["mcs_count"]
    assert artifact["inference_substrate"]["repo_commit"] == "test-commit"
    assert artifact["tests_or_checks_run"] == ["pytest focused"]
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(rows) == artifact["mcs_count"]
    assert any(not row["guided_validation"]["valid"] for row in rows)

    exp.validate_artifact(artifact)


def test_scenario_verify_3071_blocked_preconditions_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3071-BLOCKED: solver or SOTA runtime absence blocks honestly."""

    model_path = _model_path(tmp_path)
    no_solver = exp.run_experiment(
        _config(tmp_path),
        resolve_gguf_func=_resolve_one_model(model_path),
        llama_factory=FakeLlama,
        z3_module=None,
        monotonic=FakeClock(),
        repo_commit_func=lambda _: "test-commit",
    )
    assert no_solver["mcs_feedback_ready"] is False
    assert no_solver["honest_verdict"].startswith("blocked_solver_or_sota_unavailable")
    assert no_solver["legacy_smoke_only_used"] is False
    exp.validate_artifact(no_solver)

    no_model = exp.run_experiment(
        _config(tmp_path),
        resolve_gguf_func=lambda _hf_id, _preferred_quant="Q4_K_M": None,
        llama_factory=FakeLlama,
        monotonic=FakeClock(),
        repo_commit_func=lambda _: "test-commit",
    )
    assert no_model["mcs_feedback_ready"] is False
    assert no_model["honest_verdict"].startswith("blocked_solver_or_sota_unavailable")
    exp.validate_artifact(no_model)

    load_failed = exp.run_experiment(
        _config(tmp_path),
        resolve_gguf_func=_resolve_one_model(model_path),
        llama_factory=FailingLlama,
        monotonic=FakeClock(),
        repo_commit_func=lambda _: "test-commit",
    )
    assert load_failed["mcs_feedback_ready"] is False
    assert load_failed["honest_verdict"].startswith("blocked_solver_or_sota_unavailable")
    exp.validate_artifact(load_failed)


def test_req_verify_3071_validation_and_parser_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3071: artifact validation and parser helpers fail closed."""

    model_path = _model_path(tmp_path)
    custom_config = exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.ARTIFACT_FILENAME,
        rows_path=tmp_path / exp.PILOT_ROWS_REL_PATH,
        decode_config={"max_tokens": 12},
        load_config={"n_batch": 8},
    )
    assert custom_config.effective_decode_config()["max_tokens"] == 12
    assert custom_config.effective_load_config(1)["n_batch"] == 8
    assert custom_config.effective_load_config(1)["main_gpu"] == 1

    artifact = exp.run_experiment(
        _config(tmp_path),
        resolve_gguf_func=_resolve_one_model(model_path),
        llama_factory=FakeLlama,
        monotonic=FakeClock(),
        repo_commit_func=lambda _: "test-commit",
    )
    exp.validate_artifact(artifact)

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="legacy"):
        exp.validate_artifact(artifact | {"legacy_smoke_only_used": True})
    with pytest.raises(ValueError, match="model_specs"):
        exp.validate_artifact(artifact | {"model_specs": []})
    with pytest.raises(ValueError, match="prompt_hashes"):
        exp.validate_artifact(artifact | {"prompt_hashes": []})
    with pytest.raises(ValueError, match="formal_fallback_preserved"):
        exp.validate_artifact(artifact | {"formal_fallback_preserved": False})
    with pytest.raises(ValueError, match="mcs_count"):
        exp.validate_artifact(artifact | {"mcs_count": 0})
    with pytest.raises(ValueError, match="guided_success_count"):
        exp.validate_artifact(artifact | {"guided_success_count": 0})
    with pytest.raises(ValueError, match="solver_only_success_count"):
        exp.validate_artifact(artifact | {"solver_only_success_count": 0})
    with pytest.raises(ValueError, match="correction_subset_useful_count"):
        exp.validate_artifact(artifact | {"correction_subset_useful_count": 0})
    with pytest.raises(ValueError, match="exact_solver_path"):
        exp.validate_artifact(artifact | {"exact_solver_path": "wrong"})
    with pytest.raises(ValueError, match="models_used"):
        exp.validate_artifact(artifact | {"models_used": ["legacy/model"]})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(artifact | {"inference_substrate": {"local_gguf_inference": False}})
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(artifact | {"honest_verdict": "ready"})
    with pytest.raises(ValueError, match="blocked precondition"):
        exp.validate_artifact(artifact | {"mcs_feedback_ready": False, "honest_verdict": "waiting"})

    fixture = exp.build_correction_fixtures()[1]
    parsed = exp.parse_llm_candidate(
        '```json\n{"fixture_id":"sum-total-high","candidate":{"a":2,"b":3,"total":5}}\n```',
        fixture,
    )
    assert parsed == {
        "valid_parse": True,
        "candidate": {"a": 2, "b": 3, "total": 5},
        "parse_error": "",
    }
    assert exp.parse_llm_candidate("not json", fixture)["valid_parse"] is False
    assert (
        exp.parse_llm_candidate('{"fixture_id":"other","candidate":{}}', fixture)["valid_parse"]
        is False
    )
    assert (
        exp.parse_llm_candidate('{"fixture_id":"sum-total-high","candidate":[]}', fixture)[
            "valid_parse"
        ]
        is False
    )
    assert (
        exp.parse_llm_candidate(
            '{"fixture_id":"sum-total-high","candidate":{"total":"bad"}}', fixture
        )["valid_parse"]
        is False
    )
    assert exp.parse_llm_candidate("{bad json", fixture)["valid_parse"] is False
    assert (
        exp.validate_candidate_with_z3(fixture, {"a": 2, "b": 3, "total": 5}, z3_module=None)[
            "failure_reason"
        ]
        == "z3_solver_unavailable"
    )
    assert exp.validate_candidate_with_z3(fixture, {"a": "bad"})["exact_checked"] is False
    assert (
        exp.generate_correction_feedback(fixture, z3_module=None)["feedback_type"] == "unavailable"
    )

    impossible = {
        "fixture_id": "impossible",
        "kind": "unrepairable",
        "required_fields": ["x"],
        "candidate": {"x": 0},
        "mutable_fields": [],
        "constraints": [
            {
                "constraint_id": "x_exact",
                "op": "eq_affine",
                "target": "x",
                "terms": {},
                "constant": 1,
            }
        ],
    }
    assert (
        exp.generate_correction_feedback(impossible)["correction_subset"]["suggested_assignments"]
        == {}
    )

    assert exp._model_assignments(None, ["x"], fixture, exp._z3) == {}
    assert exp._model_family("unsloth/Qwen3.6-35B-A3B-GGUF") == "qwen"
    assert exp._model_family("other/model-GGUF") == "unknown"
    assert exp._relative_to(tmp_path, Path("/outside/root.json")) == Path("/outside/root.json")
