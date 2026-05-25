"""Tests for Exp 3070 first-token abstention SOTA panel.

Spec refs: REQ-VERIFY-3070,
           SCENARIO-VERIFY-3070,
           SCENARIO-VERIFY-3070-BLOCKED.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import first_token_abstention_sota_panel_v1 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
SCRIPT_PATH = REPO_ROOT / "scripts" / exp.SCRIPT_FILENAME


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        self.value += 2.0
        return self.value


class FakeLlama:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.closed = False

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["temperature"] == 0.0
        assert kwargs["seed"] == exp.DEFAULT_SEED
        assert kwargs["logprobs"] == exp.DEFAULT_LOGPROBS
        candidate_id = _field(prompt, "Candidate ID")
        fixture_id = _field(prompt, "Fixture ID")
        exact_label = "VALID" if candidate_id == "candidate_good" else "INVALID"

        predicted = exact_label
        confidence = "high"
        if fixture_id == "lin-08" and candidate_id == "candidate_bad":
            predicted = "VALID"
            confidence = "low"
        elif fixture_id == "lin-07" and candidate_id == "candidate_good":
            predicted = "INVALID"
            confidence = "high"

        return _completion(predicted, confidence)

    def close(self) -> None:
        self.closed = True


class NoConfidenceLlama:
    def __init__(self, **_kwargs: Any) -> None:
        pass

    def __call__(self, _prompt: str, **_kwargs: Any) -> dict[str, Any]:
        return {"choices": [{"text": "VALID"}]}


def _field(prompt: str, name: str) -> str:
    prefix = f"{name}: "
    for line in prompt.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def _completion(token: str, confidence: str) -> dict[str, Any]:
    if confidence == "high":
        good = math.log(0.98)
        bad = math.log(0.02)
    else:
        good = math.log(0.52)
        bad = math.log(0.48)
    other = "INVALID" if token == "VALID" else "VALID"
    return {
        "choices": [
            {
                "text": token,
                "logprobs": {
                    "tokens": [token],
                    "token_logprobs": [good],
                    "top_logprobs": [{token: good, other: bad}],
                },
            }
        ]
    }


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


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.ARTIFACT_FILENAME,
        rows_path=tmp_path / "results" / "raw" / exp.ARTIFACT / "rows.jsonl",
        tests_run=("pytest focused",),
    )


def test_req_verify_3070_spec_and_script_anchor_exists() -> None:
    """REQ-VERIFY-3070: the first-token panel is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3070" in spec
    assert "SCENARIO-VERIFY-3070" in spec
    assert "SCENARIO-VERIFY-3070-BLOCKED" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert "first_token_panel_ready" in spec
    assert "blocked_sota_confidence_unavailable" in spec
    assert SCRIPT_PATH.exists()


def test_scenario_verify_3070_fixture_candidates_have_exact_labels() -> None:
    """SCENARIO-VERIFY-3070: candidate rows retain exact solver authority."""

    rows = exp.build_scoring_rows()

    assert len(rows) == 16
    assert {row["split"] for row in rows} == {"calibration", "heldout"}
    assert {row["exact_correct"] for row in rows} == {True, False}
    assert {row["candidate_id"] for row in rows} == {"candidate_good", "candidate_bad"}
    assert all(row["exact_authority"] == "z3_solver" for row in rows)
    assert all(row["exact_checked"] for row in rows)


def test_scenario_verify_3070_live_panel_reports_confidence_and_abstention(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3070: held-out decisions use a calibration-only threshold."""

    model_path = _model_path(tmp_path)
    artifact = exp.run_experiment(
        _config(tmp_path),
        resolve_gguf_func=_resolve_one_model(model_path),
        llama_factory=FakeLlama,
        monotonic=FakeClock(),
        repo_commit_func=lambda _: "test-commit",
    )
    saved = json.loads((tmp_path / "results" / exp.ARTIFACT_FILENAME).read_text("utf-8"))
    row_path = tmp_path / artifact["panel_rows_path"]
    transcript_rows = exp.load_jsonl(row_path)

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["first_token_panel_ready"] is True
    assert artifact["confidence_signal"] == "first_token_topk_entropy"
    assert artifact["first_token_auc"] == pytest.approx(0.75)
    assert artifact["abstention_precision"] == pytest.approx(1.0)
    assert artifact["rejection_recall"] == pytest.approx(0.75)
    assert artifact["abstention_coverage"] == pytest.approx(0.125)
    assert artifact["false_positive_rate"] == pytest.approx(0.0)
    assert artifact["false_negative_rate"] == pytest.approx(0.25)
    assert artifact["verifier_gain_delta_with_abstention"] == pytest.approx(0.875)
    assert artifact["exact_ground_truth_count"] == 8
    assert artifact["models_used"] == ["unsloth/gemma-4-26B-A4B-it-GGUF"]
    assert artifact["model_specs"][0]["model_path"] == str(model_path)
    assert artifact["model_specs"][0]["quantization"] == "Q4_K_M"
    assert artifact["legacy_smoke_only_used"] is False
    assert len(artifact["prompt_hashes"]) == len(transcript_rows)
    assert artifact["calibration_split_count"] == 8
    assert artifact["heldout_split_count"] == 8
    assert artifact["accepted_count"] == 3
    assert artifact["rejected_count"] == 4
    assert artifact["abstained_count"] == 1
    assert artifact["inference_substrate"]["repo_commit"] == "test-commit"
    assert artifact["inference_substrate"]["logprob_support"]["top_logprobs"] is True
    assert artifact["tests_or_checks_run"] == ["pytest focused"]
    assert artifact["honest_verdict"].startswith("complete:")
    assert any(row["abstention_decision"] == "abstain" for row in transcript_rows)

    exp.validate_artifact(artifact)


def test_scenario_verify_3070_blocked_when_no_mandated_gguf_loads(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3070-BLOCKED: missing GGUF evidence fails closed."""

    artifact = exp.run_experiment(
        _config(tmp_path),
        resolve_gguf_func=lambda _hf_id, _preferred_quant="Q4_K_M": None,
        llama_factory=FakeLlama,
        monotonic=FakeClock(),
        repo_commit_func=lambda _: "test-commit",
    )

    assert artifact["first_token_panel_ready"] is False
    assert artifact["confidence_signal"] == "unavailable"
    assert artifact["first_token_auc"] == 0.0
    assert artifact["abstention_precision"] == 0.0
    assert artifact["rejection_recall"] == 0.0
    assert artifact["abstention_coverage"] == 0.0
    assert artifact["models_used"] == []
    assert artifact["model_specs"] == []
    assert artifact["prompt_hashes"] == []
    assert artifact["honest_verdict"].startswith("blocked_sota_confidence_unavailable")

    exp.validate_artifact(artifact)


def test_scenario_verify_3070_blocked_when_confidence_signal_missing(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3070-BLOCKED: no logprob/proxy signal blocks readiness."""

    model_path = _model_path(tmp_path)
    artifact = exp.run_experiment(
        _config(tmp_path),
        resolve_gguf_func=_resolve_one_model(model_path),
        llama_factory=NoConfidenceLlama,
        monotonic=FakeClock(),
        repo_commit_func=lambda _: "test-commit",
    )

    assert artifact["first_token_panel_ready"] is False
    assert artifact["confidence_signal"] == "unavailable"
    assert artifact["models_used"] == ["unsloth/gemma-4-26B-A4B-it-GGUF"]
    assert artifact["runtime_blocker"] == "confidence_signal_unavailable"
    assert artifact["honest_verdict"].startswith("blocked_sota_confidence_unavailable")
    exp.validate_artifact(artifact)


def test_req_verify_3070_validation_and_helper_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3070: validation, AUC, parsing, and proxy helpers fail closed."""

    model_path = _model_path(tmp_path)
    config = exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.ARTIFACT_FILENAME,
        rows_path=tmp_path / "results" / "rows.jsonl",
        decode_config={"max_tokens": 3},
        load_config={"n_batch": 8},
    )
    assert config.effective_decode_config()["max_tokens"] == 3
    assert config.effective_load_config(1)["n_batch"] == 8
    assert config.effective_load_config(1)["main_gpu"] == 1

    artifact = exp.run_experiment(
        config,
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
    with pytest.raises(ValueError, match="exact_ground_truth_count"):
        exp.validate_artifact(artifact | {"exact_ground_truth_count": 5})
    with pytest.raises(ValueError, match="prompt_hashes"):
        exp.validate_artifact(artifact | {"prompt_hashes": []})
    with pytest.raises(ValueError, match="confidence_signal"):
        exp.validate_artifact(artifact | {"confidence_signal": "unavailable"})
    with pytest.raises(ValueError, match="abstention metrics"):
        exp.validate_artifact(artifact | {"heldout_split_count": 0})
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(artifact | {"honest_verdict": "ready"})
    with pytest.raises(ValueError, match="blocked_sota_confidence_unavailable"):
        exp.validate_artifact(artifact | {"first_token_panel_ready": False})

    assert exp._parse_validity_decision(" valid\n") is True
    assert exp._parse_validity_decision("INVALID because") is False
    assert exp._parse_validity_decision("unknown") is None
    assert exp._auc([True, False], [0.5, 0.5]) == pytest.approx(0.5)
    assert exp._auc([True], [0.9]) == pytest.approx(0.5)
    assert exp._derive_threshold([{"confidence_score": 0.4, "model_exact_agreement": False}]) == (
        pytest.approx(0.4)
    )
    assert exp._derive_threshold([]) == pytest.approx(1.0)
    assert exp._mean([]) == 0.0
    assert exp._relative_to(tmp_path, Path("/outside/root.json")) == Path("/outside/root.json")

    token_only = exp._confidence_from_output(
        {"choices": [{"text": "VALID", "logprobs": {"token_logprobs": [math.log(0.7)]}}]}
    )
    assert token_only["confidence_signal"] == "first_token_logprob_proxy"
    assert token_only["confidence_score"] == pytest.approx(0.7)
    assert exp._confidence_from_output({"choices": [{"text": "VALID", "logprobs": {}}]})[
        "confidence_available"
    ] is False

    skipped_space = exp._confidence_from_output(
        {
            "choices": [
                {
                    "text": "VALID",
                    "logprobs": {
                        "tokens": [" ", "VALID"],
                        "token_logprobs": [math.log(0.51), math.log(0.9)],
                        "top_logprobs": [
                            {" ": math.log(0.51), "x": math.log(0.49)},
                            {"VALID": math.log(0.9), "INVALID": math.log(0.1)},
                        ],
                    },
                }
            ]
        }
    )
    assert skipped_space["first_token"] == "VALID"
    assert skipped_space["confidence_available"] is True

    assert exp._first_choice({"choices": []}) == {}
    assert exp._confidence_from_output({"choices": [{"text": "VALID"}]})[
        "confidence_available"
    ] is False
    assert exp._topk_entropy_confidence({"a": "bad"})["confidence_available"] is False
    assert exp._selected_signal([]) == "unavailable"
    assert exp._selected_signal([{"confidence_signal": "first_token_logprob_proxy"}]) == (
        "first_token_logprob_proxy"
    )
    assert exp._selected_signal(
        [
            {"confidence_signal": "first_token_topk_entropy"},
            {"confidence_signal": "first_token_logprob_proxy"},
        ]
    ) == "mixed_first_token_topk_entropy_and_logprob_proxy"
    assert exp._prior_exp3057_verifier_selected_accuracy(tmp_path) == pytest.approx(0.0)
    bad_prior = tmp_path / exp.EXP3057_REL_PATH
    bad_prior.parent.mkdir(parents=True, exist_ok=True)
    bad_prior.write_text("{bad-json", encoding="utf-8")
    assert exp._prior_exp3057_verifier_selected_accuracy(tmp_path) == pytest.approx(0.0)
    assert exp._model_family("unsloth/Qwen3.6-35B-A3B-GGUF") == "qwen"
    assert exp._model_family("other/model-GGUF") == "other"
    assert exp._float("bad") == 0.0
    assert exp._float_list(None) == []
    assert exp._float_list([1, "bad", 2.5]) == [1.0, 2.5]
