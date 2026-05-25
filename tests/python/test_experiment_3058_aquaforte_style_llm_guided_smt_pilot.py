"""Tests for Exp 3058 AquaForte-style LLM-guided SMT pilot.

Spec refs: REQ-VERIFY-3058,
           SCENARIO-VERIFY-3058,
           SCENARIO-VERIFY-3058-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import aquaforte_style_llm_guided_smt_pilot_v1 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
SCRIPT_PATH = REPO_ROOT / "scripts" / exp.SCRIPT_FILENAME


class FakeClock:
    def __init__(self) -> None:
        self.value = 20.0

    def __call__(self) -> float:
        self.value += 0.5
        return self.value


class FakeLlama:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.closed = False

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["temperature"] == 0.0
        assert kwargs["seed"] == exp.DEFAULT_SEED
        if "uf-inc-2" in prompt:
            text = '{"fixture_id":"uf-inc-2","instantiations":[{"x":2}]}'
        elif "uf-double-4" in prompt:
            text = '{"fixture_id":"uf-double-4","instantiations":[{"x":4}]}'
        elif "uf-offset-5" in prompt:
            text = '{"fixture_id":"uf-offset-5","instantiations":[{"x":5}]}'
        elif "uf-add-2-3" in prompt:
            text = '{"fixture_id":"uf-add-2-3","instantiations":[{"x":2,"y":3}]}'
        elif "uf-square-3" in prompt:
            text = '{"fixture_id":"uf-square-3","instantiations":[{"x":2}]}'
        elif "pred-chain-0-2" in prompt:
            text = '{"fixture_id":"pred-chain-0-2","instantiations":[{"x":0},{"x":1}]}'
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


def _write_ready_exp3057(root: Path) -> None:
    path = root / "results" / "experiment_3057_local_sota_solution_verifier_gain_panel_v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "solution_verifier_calibration_ready": True,
                "models_used": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_req_verify_3058_spec_and_script_anchor_exists() -> None:
    """REQ-VERIFY-3058: the LLM-guided SMT pilot is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3058" in spec
    assert "SCENARIO-VERIFY-3058" in spec
    assert "SCENARIO-VERIFY-3058-BLOCKED" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert "llm_guided_smt_pilot_ready" in spec
    assert "formal_fallback_preserved" in spec
    assert SCRIPT_PATH.exists()


def test_scenario_verify_3058_z3_validates_guided_and_fallback_instantiations() -> None:
    """SCENARIO-VERIFY-3058: Z3, not the LLM, decides proposal validity."""

    fixtures = exp.build_smt_fixtures()
    by_id = {fixture["fixture_id"]: fixture for fixture in fixtures}

    assert 4 <= len(fixtures) <= 8
    assert exp.validate_proposal_with_z3(by_id["uf-inc-2"], [{"x": 2}])["valid"] is True
    assert exp.validate_proposal_with_z3(by_id["uf-inc-2"], [{"x": 1}])["valid"] is False
    assert exp.validate_proposal_with_z3(by_id["uf-add-2-3"], [{"x": 2, "y": 3}])[
        "valid"
    ] is True
    assert exp.validate_proposal_with_z3(by_id["pred-chain-0-2"], [{"x": 0}])[
        "valid"
    ] is False
    assert exp.validate_proposal_with_z3(
        by_id["pred-chain-0-2"], [{"x": 0}, {"x": 1}]
    )["valid"] is True

    fallback_rows = [exp.solver_only_fallback(fixture) for fixture in fixtures]
    assert all(row["valid"] for row in fallback_rows)
    assert all(row["exact_authority"] == "z3_solver" for row in fallback_rows)


def test_scenario_verify_3058_live_pilot_writes_auditable_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3058: live proposals are counted only after exact checks."""

    _write_ready_exp3057(tmp_path)
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
    assert artifact["llm_guided_smt_pilot_ready"] is True
    assert artifact["formal_fallback_preserved"] is True
    assert artifact["guided_success_count"] == 5
    assert artifact["solver_only_success_count"] == 6
    assert artifact["invalid_llm_proposal_count"] == 1
    assert artifact["unresolved_count"] == 0
    assert artifact["models_used"] == ["unsloth/gemma-4-26B-A4B-it-GGUF"]
    assert artifact["model_specs"][0]["model_path"] == str(model_path)
    assert artifact["legacy_smoke_only_used"] is False
    assert artifact["exact_solver_path"] == exp.EXACT_SOLVER_PATH
    assert len(artifact["prompt_hashes"]) == len(exp.build_smt_fixtures())
    assert artifact["inference_substrate"]["repo_commit"] == "test-commit"
    assert artifact["tests_or_checks_run"] == ["pytest focused"]
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(rows) == artifact["fixture_count"]
    assert any(not row["guided_validation"]["valid"] for row in rows)

    exp.validate_artifact(artifact)


def test_scenario_verify_3058_blocked_preconditions_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3058-BLOCKED: missing prerequisites cannot be promoted."""

    model_path = _model_path(tmp_path)
    missing_exp3057 = exp.run_experiment(
        _config(tmp_path),
        resolve_gguf_func=_resolve_one_model(model_path),
        llama_factory=FakeLlama,
        monotonic=FakeClock(),
        repo_commit_func=lambda _: "test-commit",
    )
    assert missing_exp3057["llm_guided_smt_pilot_ready"] is False
    assert missing_exp3057["honest_verdict"].startswith("blocked_exp3057_not_ready")
    exp.validate_artifact(missing_exp3057)

    _write_ready_exp3057(tmp_path)
    no_solver = exp.run_experiment(
        _config(tmp_path),
        resolve_gguf_func=_resolve_one_model(model_path),
        llama_factory=FakeLlama,
        z3_module=None,
        monotonic=FakeClock(),
        repo_commit_func=lambda _: "test-commit",
    )
    assert no_solver["llm_guided_smt_pilot_ready"] is False
    assert no_solver["honest_verdict"].startswith("blocked_exact_solver_unavailable")
    exp.validate_artifact(no_solver)

    no_model = exp.run_experiment(
        _config(tmp_path),
        resolve_gguf_func=lambda _hf_id, _preferred_quant="Q4_K_M": None,
        llama_factory=FakeLlama,
        monotonic=FakeClock(),
        repo_commit_func=lambda _: "test-commit",
    )
    assert no_model["llm_guided_smt_pilot_ready"] is False
    assert no_model["honest_verdict"].startswith("blocked_sota_gguf_unavailable")
    exp.validate_artifact(no_model)

    load_failed = exp.run_experiment(
        _config(tmp_path),
        resolve_gguf_func=_resolve_one_model(model_path),
        llama_factory=FailingLlama,
        monotonic=FakeClock(),
        repo_commit_func=lambda _: "test-commit",
    )
    assert load_failed["llm_guided_smt_pilot_ready"] is False
    assert load_failed["honest_verdict"].startswith("blocked_sota_gguf_unavailable")
    exp.validate_artifact(load_failed)


def test_req_verify_3058_validation_and_parser_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3058: artifact validation and parser helpers fail closed."""

    _write_ready_exp3057(tmp_path)
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
    with pytest.raises(ValueError, match="guided_success_count"):
        exp.validate_artifact(artifact | {"guided_success_count": 0})
    with pytest.raises(ValueError, match="solver_only_success_count"):
        exp.validate_artifact(artifact | {"solver_only_success_count": 0})
    with pytest.raises(ValueError, match="exact_solver_path"):
        exp.validate_artifact(artifact | {"exact_solver_path": "wrong"})
    with pytest.raises(ValueError, match="models_used"):
        exp.validate_artifact(artifact | {"models_used": ["legacy/model"]})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(artifact | {"inference_substrate": {"local_gguf_inference": False}})
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(artifact | {"honest_verdict": "ready"})
    with pytest.raises(ValueError, match="blocked precondition"):
        exp.validate_artifact(
            artifact | {"llm_guided_smt_pilot_ready": False, "honest_verdict": "waiting"}
        )

    fixture = exp.build_smt_fixtures()[0]
    parsed = exp.parse_llm_instantiations(
        '```json\n{"fixture_id":"uf-inc-2","instantiations":[{"var":"x","value":2}]}\n```',
        fixture,
    )
    assert parsed == {"valid_parse": True, "instantiations": [{"x": 2}], "parse_error": ""}
    assert exp.parse_llm_instantiations("not json", fixture)["valid_parse"] is False
    assert exp.parse_llm_instantiations('{"fixture_id":"other","instantiations":[]}', fixture)[
        "valid_parse"
    ] is False
    assert exp.parse_llm_instantiations('{"fixture_id":"uf-inc-2","instantiations":{}}', fixture)[
        "valid_parse"
    ] is False
    assert exp.parse_llm_instantiations('{"fixture_id":"uf-inc-2","instantiations":[{"x":"bad"}]}', fixture)[
        "valid_parse"
    ] is False
    assert exp.parse_llm_instantiations('{"fixture_id":"uf-inc-2","instantiations":[5,{}]}', fixture)[
        "valid_parse"
    ] is False
    assert exp.validate_proposal_with_z3(fixture, [{"x": 2}], z3_module=None)[
        "failure_reason"
    ] == "z3_solver_unavailable"
    assert exp.validate_proposal_with_z3(fixture, [{"y": 2}])["exact_checked"] is False
    assert exp.solver_only_fallback(fixture, z3_module=None)["exact_authority"] == "z3_unavailable"

    impossible = {
        "fixture_id": "impossible",
        "kind": "function_value",
        "function": "h",
        "variables": ["x"],
        "target_args": [1],
        "target_value": 2,
        "expr": {"op": "affine", "coefficients": {"x": 1}, "constant": 0},
        "candidate_domain": [0],
        "max_solver_instantiations": 1,
    }
    assert exp.solver_only_fallback(impossible)["valid"] is False

    bad_exp3057 = tmp_path / exp.EXP3057_REL_PATH
    bad_exp3057.write_text("{not-json", encoding="utf-8")
    assert exp._exp3057_ready(tmp_path) is False
    assert exp._honest_verdict(False, exp._metrics([]), None).startswith(
        "blocked_guided_smt_pilot_incomplete"
    )
    assert exp._parse_json_object("x{bad}{\"ok\": true}") == {"ok": True}
    assert exp._parse_json_object("[1, 2]") == {}
    assert exp._status_name(object(), exp._z3) == "unknown"
    assert exp._model_family("unsloth/Qwen3.6-35B-A3B-GGUF") == "qwen"
    assert exp._model_family("other/model-GGUF") == "other"
    assert exp._relative_to(tmp_path, Path("/outside/root.json")) == Path("/outside/root.json")
