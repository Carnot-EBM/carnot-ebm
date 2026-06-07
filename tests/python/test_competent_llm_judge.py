"""Tests for the Exp 3925 competent LLM judge.

Spec refs: REQ-VERIFY-3925, SCENARIO-VERIFY-3925,
SCENARIO-VERIFY-3925-BLOCKED.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import competent_llm_judge as judge
from scripts.experiments import experiment_3925_competent_judge_build as exp3925


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"


class ScriptedGenerator:
    """Small robust-generator stand-in for deterministic judge tests."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompt: str, **kwargs: object) -> dict[str, object]:
        self.prompts.append(prompt)
        assert kwargs["temperature"] == 0.0
        step = prompt.split("Step under review:", 1)[-1]
        incorrect = any(
            marker in step
            for marker in ("= 65", "= 124", "area 40", "= 12", "all daxes are wugs")
        )
        verdict = "INCORRECT" if incorrect else "CORRECT"
        return {"choices": [{"text": f"Reason: checked directly.\nVERDICT: {verdict}"}]}


def test_req_verify_3925_spec_anchor_exists() -> None:
    """REQ-VERIFY-3925: the competent judge is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3925" in spec
    assert "SCENARIO-VERIFY-3925" in spec
    assert "python/carnot/verify/competent_llm_judge.py" in spec
    assert "results/experiment_3925_competent_judge_build.json" in spec


def test_req_verify_3925_parser_abstains_instead_of_constant_default() -> None:
    """REQ-VERIFY-3925: unparsed verdicts abstain at 0.5 instead of defaulting."""

    incorrect = judge.parse_judge_response("Reason: 7*8 is 56.\nVERDICT: INCORRECT")
    correct = judge.parse_judge_response('{"verdict": "correct", "confidence": 0.77}')
    json_error = judge.parse_judge_response('{"p_incorrect": 0.93, "verdict": "incorrect"}')
    terse = judge.parse_judge_response("CORRECT")
    unclear = judge.parse_judge_response("I would need the rest of the problem.")

    assert incorrect.parsed is True
    assert incorrect.verdict == "incorrect"
    assert incorrect.verdict_prob > 0.8
    assert correct.verdict == "correct"
    assert correct.verdict_prob == pytest.approx(0.23)
    assert json_error.verdict_prob == pytest.approx(0.93)
    assert terse.verdict == "correct"
    assert terse.verdict_prob < incorrect.verdict_prob
    assert unclear.parsed is False
    assert unclear.verdict is None
    assert unclear.verdict_prob == 0.5


def test_req_verify_3925_scripted_fixture_scores_polarity_correctly() -> None:
    """REQ-VERIFY-3925: verdict_prob is high for incorrect rows, low for correct."""

    fixture = judge.build_separable_fixture()
    result = judge.run_judge_fixture(fixture, ScriptedGenerator(), max_tokens=48)

    assert len(result["scores"]) == len(fixture)
    assert result["fixture_auroc"] == 1.0
    assert result["verdicts_parse_rate"] == 1.0
    assert result["parser_constant_prediction"] is False
    assert all(
        score > 0.5 if item["gold_error"] else score < 0.5
        for item, score in zip(fixture, result["scores"], strict=True)
    )


def test_req_verify_3925_exp3917_diagnosis_uses_flipped_polarity() -> None:
    """REQ-VERIFY-3925: below-chance Exp 3917 scores are diagnosed from disk."""

    diagnosis = exp3925.diagnose_exp3917_scores(REPO_ROOT)

    assert diagnosis["original_auroc"] == pytest.approx(0.4423209366391185)
    assert diagnosis["flipped_polarity_auroc"] == pytest.approx(0.5576790633608816)
    assert diagnosis["diagnosed_cause"] == "polarity_inversion"
    assert diagnosis["neutral_score_count"] == 2


def test_req_verify_3925_artifact_uses_bare_fields_and_ready_gate(tmp_path: Path) -> None:
    """REQ-VERIFY-3925: artifact fields stay bare and READY follows the gate."""

    fixture_result: dict[str, Any] = {
        "fixture_auroc": 0.9,
        "verdicts_parse_rate": 1.0,
        "scores": [0.1, 0.9],
        "labels": [0, 1],
        "raw_texts": ["VERDICT: CORRECT", "VERDICT: INCORRECT"],
        "parsed": [True, True],
        "verdicts": ["correct", "incorrect"],
        "parser_constant_prediction": False,
    }
    diagnosis = {
        "diagnosed_cause": "polarity_inversion",
        "flipped_polarity_auroc": 0.5576790633608816,
        "original_auroc": 0.4423209366391185,
    }
    artifact = exp3925.build_artifact(
        config=exp3925.ExperimentConfig(
            repo_root=tmp_path,
            started_at=0.0,
            clock=lambda: 65.0,
        ),
        fixture_result=fixture_result,
        diagnosis=diagnosis,
        preconditions_checked=[exp3925.PreconditionCheck("cuda_available", True, "ok")],
        model_specs={"model_used": "Qwen3.6-35B-A3B", "gguf_path": "/models/qwen.gguf"},
        unit_test_passed=True,
    )

    exp3925.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete: competent_judge_READY")
    assert artifact["judge_module_path"] == "python/carnot/verify/competent_llm_judge.py"
    assert artifact["judge_model_used"] == "Qwen3.6-35B-A3B"
    assert artifact["fixture_auroc"] == 0.9
    assert artifact["unit_test_passed"] is True
    assert not isinstance(artifact["fixture_auroc"], dict)


def test_scenario_verify_3925_blocked_artifact_is_terminal() -> None:
    """SCENARIO-VERIFY-3925-BLOCKED: missing resources do not fabricate metrics."""

    artifact = exp3925.build_blocked_artifact(
        reason="blocked_no_cuda",
        preconditions_checked=[exp3925.PreconditionCheck("cuda_available", False, "no cuda")],
        duration_s=0.25,
        diagnosis={"diagnosed_cause": None, "flipped_polarity_auroc": None},
    )

    exp3925.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_no_cuda"
    assert artifact["fixture_auroc"] is None
    assert artifact["verdicts_parse_rate"] is None
    assert artifact["unit_test_passed"] is False
    assert artifact["inference_substrate"] == "none_blocked_preflight"


def test_scenario_verify_3925_live_fixture_positive_control() -> None:
    """SCENARIO-VERIFY-3925: live robust-GGUF judge clears the separable fixture."""

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

    code = """
import json
from carnot.verify import competent_llm_judge as judge
from carnot.verify.gguf_inference import load_gguf_generator

generator, meta = load_gguf_generator(
    prefer_order=list(judge.COMPETENT_PREFER_ORDER),
    n_ctx=judge.DEFAULT_N_CTX,
    max_n_gpu_layers=judge.DEFAULT_MAX_N_GPU_LAYERS,
)
fixture = judge.build_separable_fixture()
result = judge.run_judge_fixture(fixture, generator, max_tokens=judge.DEFAULT_MAX_TOKENS)
print("COMPETENT_JUDGE_TEST_JSON=" + json.dumps({"meta": meta, "result": result}, sort_keys=True))
"""
    command = [
        "env",
        "-i",
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '0')}",
        f"HOME={os.environ['HOME']}",
        f"PATH={os.environ['PATH']}",
    ]
    if os.environ.get("LD_LIBRARY_PATH"):
        command.append(f"LD_LIBRARY_PATH={os.environ['LD_LIBRARY_PATH']}")
    command.extend([str(REPO_ROOT / ".venv" / "bin" / "python"), "-c", code])
    live = subprocess.run(
        command,
        capture_output=True,
        check=False,
        cwd=REPO_ROOT,
        text=True,
        timeout=900,
    )
    assert live.returncode == 0, live.stderr or live.stdout
    marker_lines = [
        line for line in live.stdout.splitlines() if line.startswith("COMPETENT_JUDGE_TEST_JSON=")
    ]
    assert marker_lines, live.stdout
    payload = json.loads(marker_lines[-1].split("=", 1)[1])
    result = payload["result"]
    meta = payload["meta"]

    assert meta["model_used"] in judge.COMPETENT_PREFER_ORDER
    assert Path(str(meta["gguf_path"])).is_file()
    assert len(result["scores"]) == len(judge.build_separable_fixture())
    assert result["verdicts_parse_rate"] > 0.9
    assert result["parser_constant_prediction"] is False
    assert result["fixture_auroc"] > 0.65
