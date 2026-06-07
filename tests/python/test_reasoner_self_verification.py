"""Tests for the Exp 3894 reasoner self-verification harness.

Spec refs: REQ-VERIFY-3894, SCENARIO-VERIFY-3894,
SCENARIO-VERIFY-3894-BLOCKED.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import reasoner_self_verification as rsv
from scripts.experiments import experiment_3894_reasoner_self_verify_harness as exp3894


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
FIXTURE_MODEL_IDS = (
    "unsloth/gemma-4-26B-A4B-it-GGUF",
    "unsloth/Qwen3.6-35B-A3B-GGUF",
)


def _cached_fixture_model_path() -> Path:
    model_specs, checks = exp3894._resolve_model()
    assert any(check.available for check in checks), [check.as_dict() for check in checks]
    model_path = model_specs.get("model_path")
    if model_path and Path(str(model_path)).is_file() and Path(str(model_path)).stat().st_size > 0:
        return Path(str(model_path))
    raise AssertionError("blocked_model_not_cached")


def _scripted_llama_factory(responses: Sequence[str]) -> type:
    class ScriptedLlama:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.index = 0

        def __call__(self, prompt: str, **kwargs: object) -> dict[str, object]:
            assert "Return one compact JSON object" in prompt
            assert "error_confidence" in prompt
            assert kwargs["temperature"] == 0.0
            response = responses[self.index]
            self.index += 1
            return {"choices": [{"text": response}]}

    return ScriptedLlama


def test_req_verify_3894_spec_anchor_exists() -> None:
    """REQ-VERIFY-3894: the harness is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3894" in spec
    assert "SCENARIO-VERIFY-3894" in spec
    assert "python/carnot/verify/reasoner_self_verification.py" in spec
    assert "results/experiment_3894_reasoner_self_verify_harness.json" in spec


def test_req_verify_3894_parser_handles_json_and_text_without_correct_default() -> None:
    """REQ-VERIFY-3894: parser does not collapse unparsed output to correct."""

    incorrect = rsv.parse_self_verification_response(
        '{"verdict":"incorrect","error_confidence":0.91,"reason":"7*8 is 56"}'
    )
    correct = rsv.parse_self_verification_response('{"is_correct": true, "confidence": 0.82}')
    no_text = rsv.parse_self_verification_response("NO - the equality is incorrect.")
    yes_text = rsv.parse_self_verification_response("Correct. The step follows.")
    unclear = rsv.parse_self_verification_response("I would need more context.")

    assert incorrect.pred == 1
    assert incorrect.score == 0.91
    assert correct.pred == 0
    assert correct.score == 0.18
    assert no_text.pred == 1
    assert no_text.score > yes_text.score
    assert yes_text.pred == 0
    assert unclear.pred is None
    assert unclear.score == 0.5
    assert not unclear.parsed


def test_req_verify_3894_parser_and_metric_edge_cases_are_explicit() -> None:
    """REQ-VERIFY-3894: parser edge cases remain visible and test-covered."""

    assert rsv._json_candidates("{bad json}") == []
    assert rsv._coerce_bool(1) is True
    assert rsv._coerce_bool("yes") is True
    assert rsv._coerce_bool("no") is False
    assert rsv.parse_self_verification_response('{"is_error": true, "score": 0.7}').pred == 1
    assert rsv.parse_self_verification_response('{"verdict": "no"}').pred == 1
    assert rsv.parse_self_verification_response('{"verdict": "maybe"}').parsed is False
    assert rsv.parse_self_verification_response('{"verdict":"incorrect","error_confidence":0.0}').score == 0.8
    assert rsv.parse_self_verification_response('{"verdict":"incorrect","correct_confidence":0.1}').score == 0.9
    assert rsv.parse_self_verification_response('{"verdict":"incorrect"}').score == 0.8
    assert rsv.parse_self_verification_response("   ").parsed is False
    assert rsv.parse_self_verification_response("There is no error here.").pred == 0
    assert rsv.parse_self_verification_response("The step is wrong.").pred == 1
    assert rsv._extract_llama_text("plain") == "plain"
    assert rsv._extract_llama_text({"choices": [{"message": {"content": "chat"}}]}) == "chat"
    assert rsv._extract_llama_text({"choices": [{}]}).startswith("{")
    assert rsv._auroc([0], [0.2]) is None
    assert rsv._auroc([1, 0], [0.5, 0.5]) == 0.5

    with pytest.raises(ValueError, match="gold_labels"):
        rsv.reasoner_self_verify(
            ["one step"],
            model_path="/tmp/scripted.gguf",
            gold_labels=[1, 0],
            llama_factory=_scripted_llama_factory(["NO"]),
        )


def test_req_verify_3894_scripted_harness_scores_nonconstant_predictions() -> None:
    """REQ-VERIFY-3894: reasoner_self_verify returns AUROC, catches, and raw traces."""

    fixture = rsv.build_positive_control_fixture()
    responses = [
        json.dumps(
            {
                "verdict": "incorrect" if item["gold_error"] else "correct",
                "error_confidence": 0.9 if item["gold_error"] else 0.1,
            }
        )
        for item in fixture
    ]

    result = rsv.reasoner_self_verify(
        [str(item["step"]) for item in fixture],
        model_path="/tmp/scripted.gguf",
        gold_labels=[int(item["gold_error"]) for item in fixture],
        llama_factory=_scripted_llama_factory(responses),
        max_tokens=48,
        random_seed=3894,
    )

    assert result["per_step_pred"] == [int(item["gold_error"]) for item in fixture]
    assert result["per_step_score"] == [0.9 if item["gold_error"] else 0.1 for item in fixture]
    assert result["auroc"] == 1.0
    assert result["n_caught"] == 6
    assert result["parser_constant_prediction"] is False
    assert len(result["raw_responses"]) == len(fixture)


def test_req_verify_3904_scripted_harness_accepts_custom_prompt_builder() -> None:
    """REQ-VERIFY-3904: boosted arms reuse the tested harness with a custom prompt."""

    prompts: list[str] = []

    class ScriptedLlama:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def __call__(self, prompt: str, **kwargs: object) -> dict[str, object]:
            prompts.append(prompt)
            return {"choices": [{"text": '{"verdict":"incorrect","error_confidence":0.88}'}]}

    result = rsv.reasoner_self_verify(
        ["2 + 2 = 5."],
        model_path="/tmp/scripted.gguf",
        gold_labels=[1],
        llama_factory=ScriptedLlama,
        prompt_builder=lambda step: f"BOOSTED CHECK: {step}",
    )

    assert prompts == ["BOOSTED CHECK: 2 + 2 = 5."]
    assert result["per_step_pred"] == [1]
    assert result["per_step_score"] == [0.88]


def test_scenario_verify_3894_live_fixture_positive_control() -> None:
    """SCENARIO-VERIFY-3894: live SOTA GGUF fixture catches known injected errors."""

    proc = subprocess.run(
        [
            str(REPO_ROOT / ".venv" / "bin" / "python"),
            "-c",
            "import torch; assert torch.cuda.is_available()",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout

    model_path = _cached_fixture_model_path()
    code = f"""
import json
from carnot.verify import reasoner_self_verification as rsv
fixture = rsv.build_positive_control_fixture()
result = rsv.reasoner_self_verify(
    [str(item["step"]) for item in fixture],
    model_path={str(model_path)!r},
    gold_labels=[int(item["gold_error"]) for item in fixture],
    max_tokens=96,
    n_gpu_layers=0,
    offload_kqv=False,
    random_seed=3894,
)
print(json.dumps(result, sort_keys=True))
"""
    child_env = os.environ.copy()
    for key in list(child_env):
        if key.startswith(("PYTEST_", "COV_CORE")):
            child_env.pop(key, None)
    child_env["CUDA_VISIBLE_DEVICES"] = ""
    live = subprocess.run(
        [str(REPO_ROOT / ".venv" / "bin" / "python"), "-c", code],
        capture_output=True,
        check=False,
        cwd=REPO_ROOT,
        env=child_env,
        text=True,
        timeout=600,
    )
    assert live.returncode == 0, live.stderr or live.stdout
    result = json.loads(live.stdout.strip().splitlines()[-1])
    fixture = rsv.build_positive_control_fixture()

    assert len(result["per_step_pred"]) == len(fixture)
    assert result["n_caught"] > 0
    assert result["auroc"] > 0.6
    assert result["parser_constant_prediction"] is False
    assert result["parsed_count"] > 0


def test_req_verify_3894_artifact_builder_uses_bare_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-3894: Exp 3894 artifacts expose required bare scalar fields."""

    harness_result: dict[str, Any] = {
        "per_step_pred": [1, 0, 1, 0],
        "per_step_score": [0.9, 0.1, 0.8, 0.2],
        "raw_responses": ["bad", "ok", "bad", "ok"],
        "parsed_count": 4,
        "unparsed_count": 0,
        "parser_constant_prediction": False,
        "auroc": 1.0,
        "n_caught": 2,
    }
    artifact = exp3894.build_artifact(
        harness_result=harness_result,
        config=exp3894.ExperimentConfig(
            repo_root=tmp_path,
            started_at=10.0,
            clock=lambda: 75.0,
        ),
        preconditions_checked=[exp3894.PreconditionCheck("cuda_available", True, "ok")],
        model_specs={"hf_id": "fixture", "model_path": "fixture.gguf"},
        unit_test_passed=True,
    )

    exp3894.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["harness_module_path"] == "python/carnot/verify/reasoner_self_verification.py"
    assert artifact["fixture_auroc"] == 1.0
    assert artifact["fixture_n_caught"] == 2
    assert artifact["unit_test_passed"] is True
    assert artifact["duration_s"] == 65.0
    assert isinstance(artifact["reproducibility_checksum"], str)
    assert len(artifact["reproducibility_checksum"]) == 64


def test_scenario_verify_3894_blocked_artifact_is_terminal(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3894-BLOCKED: blocked resources do not fabricate metrics."""

    artifact = exp3894.build_blocked_artifact(
        reason="blocked_no_cuda",
        preconditions_checked=[exp3894.PreconditionCheck("cuda_available", False, "no cuda")],
        duration_s=0.25,
    )
    output = tmp_path / exp3894.OUTPUT_REL_PATH
    exp3894.write_artifact(output, artifact)
    persisted = json.loads(output.read_text(encoding="utf-8"))

    exp3894.validate_artifact(persisted)
    assert persisted["honest_verdict"] == "blocked_no_cuda"
    assert persisted["fixture_auroc"] is None
    assert persisted["fixture_n_caught"] == 0
    assert persisted["unit_test_passed"] is False
    assert persisted["inference_substrate"] == "none_blocked_preflight"
