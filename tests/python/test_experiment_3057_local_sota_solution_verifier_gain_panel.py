"""Tests for Exp 3057 local SOTA solution-verifier gain panel.

Spec refs: REQ-VERIFY-3057,
           SCENARIO-VERIFY-3057,
           SCENARIO-VERIFY-3057-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import local_sota_solution_verifier_gain_panel_v1 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
SCRIPT_PATH = REPO_ROOT / "scripts" / exp.SCRIPT_FILENAME


class FakeClock:
    def __init__(self) -> None:
        self.value = 10.0

    def __call__(self) -> float:
        self.value += 1.25
        return self.value


class FakeLlama:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.closed = False

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["temperature"] == 0.0
        assert kwargs["seed"] == exp.DEFAULT_SEED
        if "Role: solver" in prompt:
            return {"choices": [{"text": self._solver_text(prompt)}]}
        if "Role: verifier" in prompt:
            if "lin-01" in prompt or "lin-05" in prompt or "lin-06" in prompt:
                return {
                    "choices": [
                        {
                            "text": (
                                '{"accepted":["candidate_a","candidate_b"],'
                                '"selected":"candidate_b"}'
                            )
                        }
                    ]
                }
            return {"choices": [{"text": '{"accepted":["candidate_b"],"selected":"candidate_b"}'}]}
        return {"choices": [{"text": "{}"}]}

    def close(self) -> None:
        self.closed = True

    @staticmethod
    def _solver_text(prompt: str) -> str:
        if "lin-01" in prompt:
            return '{"status":"sat","assignment":{"x":3,"y":2}}'
        if "lin-05" in prompt or "lin-06" in prompt:
            return '{"status":"unsat"}'
        return '{"status":"unsat"}'


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.ARTIFACT_FILENAME,
        raw_dir=tmp_path / "results" / "raw" / exp.ARTIFACT,
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


def test_req_verify_3057_spec_and_script_anchor_exists() -> None:
    """REQ-VERIFY-3057: the panel is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3057" in spec
    assert "SCENARIO-VERIFY-3057" in spec
    assert "SCENARIO-VERIFY-3057-BLOCKED" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert "solution_verifier_calibration_ready" in spec
    assert "verifier_gain_delta" in spec
    assert SCRIPT_PATH.exists()


def test_scenario_verify_3057_fixtures_have_exact_solver_ground_truth() -> None:
    """SCENARIO-VERIFY-3057: fixtures expose exact SAT/SMT authority."""

    fixtures = exp.build_sat_smt_fixtures()
    truth_rows = exp.compute_exact_ground_truth(fixtures)

    assert 6 <= len(fixtures) <= 12
    assert len(truth_rows) == len(fixtures)
    assert {row["solver_status"] for row in truth_rows} == {"sat", "unsat"}
    assert all(row["exact_checked"] for row in truth_rows)
    assert all(row["exact_authority"] == "z3_solver" for row in truth_rows)

    row = next(row for row in truth_rows if row["fixture_id"] == "lin-01")
    assert exp.evaluate_candidate(row, {"status": "sat", "assignment": {"x": 3, "y": 2}})
    assert not exp.evaluate_candidate(row, {"status": "sat", "assignment": {"x": 2, "y": 3}})
    assert not exp.evaluate_candidate(row, {"status": "unsat"})


def test_scenario_verify_3057_live_panel_reports_gain_and_false_error_rates(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3057: live model judging is compared to exact labels."""

    model_path = _model_path(tmp_path)
    artifact = exp.run_experiment(
        _config(tmp_path),
        resolve_gguf_func=_resolve_one_model(model_path),
        llama_factory=FakeLlama,
        monotonic=FakeClock(),
        repo_commit_func=lambda _: "test-commit",
    )
    saved = json.loads((tmp_path / "results" / exp.ARTIFACT_FILENAME).read_text("utf-8"))
    transcript_rows = exp.load_jsonl(tmp_path / artifact["panel_rows_path"])

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["solution_verifier_calibration_ready"] is True
    assert artifact["exact_ground_truth_count"] == len(exp.build_sat_smt_fixtures())
    assert artifact["one_shot_solver_accuracy"] == pytest.approx(0.375)
    assert artifact["verifier_selected_accuracy"] == pytest.approx(1.0)
    assert artifact["verifier_gain_delta"] == pytest.approx(0.625)
    assert artifact["false_positive_rate"] == pytest.approx(0.0)
    assert artifact["false_negative_rate"] == pytest.approx(0.0)
    assert artifact["exact_solver_agreement"] == pytest.approx(1.0)
    assert artifact["models_used"] == ["unsloth/gemma-4-26B-A4B-it-GGUF"]
    assert artifact["model_specs"][0]["model_path"] == str(model_path)
    assert artifact["legacy_smoke_only_used"] is False
    assert artifact["cross_family_used"] is False
    assert len(artifact["prompt_hashes"]) == 16
    assert artifact["inference_substrate"]["repo_commit"] == "test-commit"
    assert artifact["inference_substrate"]["seed"] == exp.DEFAULT_SEED
    assert artifact["tests_or_checks_run"] == ["pytest focused"]
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(transcript_rows) == artifact["exact_ground_truth_count"]

    exp.validate_artifact(artifact)


def test_scenario_verify_3057_blocked_when_no_mandated_gguf_loads(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3057-BLOCKED: missing local GGUF evidence fails closed."""

    artifact = exp.run_experiment(
        _config(tmp_path),
        resolve_gguf_func=lambda _hf_id, _preferred_quant="Q4_K_M": None,
        llama_factory=FakeLlama,
        monotonic=FakeClock(),
        repo_commit_func=lambda _: "test-commit",
    )

    assert artifact["solution_verifier_calibration_ready"] is False
    assert artifact["verifier_gain_delta"] == 0.0
    assert artifact["false_positive_rate"] == 0.0
    assert artifact["false_negative_rate"] == 0.0
    assert artifact["exact_ground_truth_count"] == 0
    assert artifact["models_used"] == []
    assert artifact["model_specs"] == []
    assert artifact["prompt_hashes"] == []
    assert artifact["legacy_smoke_only_used"] is False
    assert artifact["honest_verdict"].startswith("blocked_sota_gguf_unavailable")

    exp.validate_artifact(artifact)

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="model_specs"):
        exp.validate_artifact(artifact | {"solution_verifier_calibration_ready": True})
    with pytest.raises(ValueError, match="legacy"):
        exp.validate_artifact(artifact | {"legacy_smoke_only_used": True})
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(artifact | {"honest_verdict": "waiting"})


def test_req_verify_3057_validation_and_helper_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3057: validation and helper edges fail closed."""

    model_path = _model_path(tmp_path)
    config = exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.ARTIFACT_FILENAME,
        raw_dir=tmp_path / "results" / "raw" / exp.ARTIFACT,
        decode_config={"max_tokens": 12},
        load_config={"n_batch": 8},
    )
    assert config.effective_decode_config()["max_tokens"] == 12
    assert config.effective_load_config(1)["n_batch"] == 8
    assert config.effective_load_config(1)["main_gpu"] == 1

    artifact = exp.run_experiment(
        config,
        resolve_gguf_func=_resolve_one_model(model_path),
        llama_factory=FakeLlama,
        monotonic=FakeClock(),
        repo_commit_func=lambda _: "test-commit",
    )

    with pytest.raises(ValueError, match="model_specs"):
        exp.validate_artifact(artifact | {"model_specs": []})
    with pytest.raises(ValueError, match="exact_ground_truth_count"):
        exp.validate_artifact(artifact | {"exact_ground_truth_count": 5})
    with pytest.raises(ValueError, match="prompt_hashes"):
        exp.validate_artifact(artifact | {"prompt_hashes": []})
    with pytest.raises(ValueError, match="exact_solver_authority"):
        exp.validate_artifact(artifact | {"exact_solver_authority": "none"})
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(artifact | {"honest_verdict": "ready"})

    with pytest.raises(RuntimeError, match="z3_solver_unavailable"):
        exp.compute_exact_ground_truth(exp.build_sat_smt_fixtures(), z3_module=None)

    truth_row = exp.compute_exact_ground_truth(exp.build_sat_smt_fixtures())[0]
    assert exp.evaluate_candidate(truth_row, {"status": "maybe"}) is False
    assert exp.evaluate_candidate(truth_row, {"status": "sat", "assignment": {"x": 3}}) is False
    assert exp._metrics([])["exact_ground_truth_count"] == 0
    assert exp._parse_candidate("not json") == {"status": "unparseable"}
    assert exp._parse_candidate("{not valid json}") == {"status": "unparseable"}
    assert exp._parse_candidate("[1, 2]") == {"status": "unparseable"}
    assert exp._parse_verifier_decision('{"accepted":"bad","selected":null}') == {
        "accepted": [],
        "selected": "",
    }
    assert exp._int_assignment(None) == {}
    assert exp._int_assignment({"x": "bad", "y": 2}) == {"y": 2}
    assert exp._constraints_hold([exp._le({"r": 1}, 1)], {"r": 2}) is False
    assert exp._constraints_hold([exp._ge({"r": 1}, 2)], {"r": 1}) is False
    assert exp._model_family("other/model-GGUF") == "other"
    assert exp._honest_verdict(False, exp._metrics([]), "boom").startswith(
        "blocked_sota_gguf_unavailable"
    )
    assert exp._honest_verdict(False, exp._metrics([]), None).startswith(
        "blocked_sota_gguf_unavailable"
    )
    assert exp._relative_to(tmp_path, Path("/outside/root.json")) == Path("/outside/root.json")

    qwen, gemma_26, gemma_31 = exp.MANDATED_MODEL_IDS
    assert exp._cross_family_used(
        exp._select_models({qwen: "/qwen.gguf", gemma_26: "/gemma.gguf", gemma_31: None})
    )
    same_family = exp._select_models(
        {qwen: None, gemma_26: "/gemma26.gguf", gemma_31: "/gemma31.gguf"}
    )
    assert len(same_family) == 2
    assert not exp._cross_family_used(same_family)
