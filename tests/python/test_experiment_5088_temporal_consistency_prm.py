"""Tests for Exp 5088 temporal-consistency PRM diagnostic.

Spec refs: REQ-VERIFY-5088, SCENARIO-VERIFY-5088.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5088_temporal_consistency_prm as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"
ARTIFACT_PATH = REPO / mod.RESULT_RELATIVE_PATH


class Clock:
    """Deterministic clock for duration fields."""

    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.index = 0

    def __call__(self) -> float:
        value = self.values[min(self.index, len(self.values) - 1)]
        self.index += 1
        return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _exp5085_artifact(*, ready: bool = False) -> dict[str, Any]:
    return {
        "honest_verdict": "success_llamacpp_logprob_endpoint_ready" if ready else "blocked_endpoint",
        "logprob_endpoint_ready": ready,
        "endpoint_url": "http://127.0.0.1:46097",
        "sample_completion": {
            "route": "http://127.0.0.1:46097/completion",
            "model_hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "model_path": "/models/gemma-26b.gguf",
        },
        "model_specs": {
            "resolved_models": {
                "flagship_moe": {
                    "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "resolved_path": "/models/qwen.gguf",
                },
                "flagship_dense": {
                    "hf_id": "unsloth/gemma-4-31B-it-GGUF",
                    "resolved_path": "/models/gemma-31b.gguf",
                },
                "middle_moe": {
                    "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "resolved_path": "/models/gemma-26b.gguf",
                },
            }
        },
        "flagged_adversarial": False,
    }


def _fallback_row(
    question_index: int,
    candidate_index: int,
    answer: str,
    *,
    gold: str,
    positive: bool,
) -> dict[str, Any]:
    top_logprobs = (
        {" consistent": math.log(0.80), " supported": math.log(0.70), " error": math.log(0.01)}
        if positive
        else {" consistent": math.log(0.05), " supported": math.log(0.04), " error": math.log(0.90)}
    )
    choices = ["A", "B"] if question_index == 0 else ["C", "D"]
    return {
        "schema": "carnot.shared_logprob_candidate_cache_v2.candidate_row.v1",
        "corpus": mod.MUSR_CORPUS,
        "question_id": f"MuSR/murder_mysteries:{question_index}",
        "question_index": question_index,
        "candidate_id": f"MuSR/murder_mysteries:{question_index}/cached-{candidate_index}",
        "candidate_index": candidate_index,
        "question": "Who is responsible?",
        "context": f"The case evidence supports {gold}; an unsupported answer is a process error.",
        "choices": choices,
        "gold": gold,
        "answer": answer,
        "mean_logprob": math.log(0.80 if positive else 0.75),
        "top_logprobs": [top_logprobs],
        "uprm_marker_logprobs": [{" +": math.log(0.70), " -": math.log(0.30)}],
        "model_id": "hidden-source-model",
    }


def _primary_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for question_index, answers in enumerate((("A", "B"), ("C", "D"))):
        choices = list(answers)
        for candidate_index, answer in enumerate(answers):
            rows.append(
                {
                    "schema": "carnot.experiment_5058_sota_candidate_refresh_inwriting.row.v1",
                    "row_id": f"MuSR/murder_mysteries:{question_index}/sota5058-{candidate_index:04d}",
                    "corpus": mod.MUSR_CORPUS,
                    "question_id": f"MuSR/murder_mysteries:{question_index}",
                    "question_index": question_index,
                    "candidate_index": candidate_index,
                    "question": "Who is responsible?",
                    "choices": choices,
                    "answer_text": answer,
                    "parsed_answer": answer,
                    "parse_status": "parsed",
                    "model_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "structured_constraints": {
                        "answer_in_allowed_choices": True,
                        "evidence_span_count": 1,
                        "constraint_checks": {
                            "allowed_choice": True,
                            "delayed_after_draft": True,
                            "nonempty_draft": True,
                        },
                    },
                }
            )
    return rows


def _fallback_rows() -> list[dict[str, Any]]:
    return [
        _fallback_row(0, 0, "A", gold="A", positive=True),
        _fallback_row(0, 1, "B", gold="A", positive=False),
        _fallback_row(1, 0, "C", gold="D", positive=False),
        _fallback_row(1, 1, "D", gold="D", positive=True),
    ]


def test_req_verify_5088_spec_declares_temporal_consistency_contract() -> None:
    """REQ-VERIFY-5088: OpenSpec anchors the diagnostic and artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-5088",
        "SCENARIO-VERIFY-5088",
        "experiment_5088_temporal_consistency_prm.py",
        "results/experiment_5088_temporal_consistency_prm_v467.json",
        "success_temporal_consistency_prm_improves_plus_",
        "complete_temporal_consistency_prm_no_win",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_verify_5088_temporal_refinement_improves_without_live_llm(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5088: proxy temporal states improve the one-pass baseline."""

    _write_json(tmp_path / mod.EXP5085_RELATIVE_PATH, _exp5085_artifact(ready=False))
    _write_jsonl(tmp_path / mod.EXP5058_CACHE_RELATIVE_PATH, _primary_rows())
    _write_jsonl(tmp_path / mod.EXP5029_CACHE_RELATIVE_PATH, _fallback_rows())

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        min_questions=2,
        limit_questions=2,
        now=Clock([10.0, 14.0]),
        write=True,
    )

    assert artifact["honest_verdict"] == "success_temporal_consistency_prm_improves_plus_0p500"
    assert artifact["inference_substrate"] == "deterministic_proxy_over_cached_candidate_traces"
    assert artifact["live_llm_invoked"] is False
    assert artifact["n_examples"] == 4
    assert artifact["one_pass_accuracy"] == pytest.approx(0.5)
    assert artifact["temporal_consistency_accuracy"] == pytest.approx(1.0)
    assert artifact["delta_vs_one_pass"] == pytest.approx(0.5)
    assert artifact["beats_one_pass"] is True
    assert artifact["stability_score"] > 0.6
    assert artifact["leakage_audit"]["passed"] is True
    assert artifact["preconditions_checked"]["label_proxy_availability"]["gold_labels_available"] is True
    assert artifact["preconditions_checked"]["exp5085_live_endpoint_fields"]["usable"] is False
    assert artifact["comparator_metrics"]["tuned_self_consistency_accuracy"] == pytest.approx(0.5)
    assert artifact["candidate_selection_value"]["temporal_consistency_accuracy"] == pytest.approx(1.0)
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []


def test_req_verify_5088_leakage_audit_catches_forbidden_scorers() -> None:
    """REQ-VERIFY-5088: answer-key and model-identity reads are blocked."""

    assert mod.leakage_audit()["passed"] is True

    def leaky_scorer(candidate: Any) -> float:
        return 0.0 if candidate.get("gold") else 1.0

    failed = mod.leakage_audit(extra_scorers=[leaky_scorer])

    assert failed["passed"] is False
    assert failed["answer_key_oracle_leakage"] is True
    assert any("gold" in item for item in failed["violations"])


def test_req_verify_5088_schema_validator_rejects_bad_artifacts() -> None:
    """REQ-VERIFY-5088: schema validation rejects missing and bad-typed fields."""

    assert "honest_verdict" in mod.artifact_schema_errors({})

    bad = {field: None for field in mod.REQUIRED_ARTIFACT_FIELDS}
    bad.update(
        {
            "honest_verdict": "maybe",
            "duration_s": -1,
            "inference_substrate": "unknown",
            "preconditions_checked": {},
            "model_specs": {"mandatory_models": [{"hf_id": "missing"}]},
            "live_llm_invoked": "no",
            "n_examples": 0,
            "one_pass_accuracy": 2.0,
            "temporal_consistency_accuracy": -0.1,
            "delta_vs_one_pass": "bad",
            "stability_score": 2.0,
            "leakage_audit": {"passed": False},
            "beats_one_pass": "yes",
            "flagged_adversarial": "no",
            "field_principles": {},
            "schema": "bad",
            "experiment_id": "5088",
            "spec_refs": [],
        }
    )

    errors = mod.artifact_schema_errors(bad)
    for field in (
        "honest_verdict",
        "duration_s",
        "inference_substrate",
        "model_specs",
        "live_llm_invoked",
        "n_examples",
        "one_pass_accuracy",
        "temporal_consistency_accuracy",
        "delta_vs_one_pass",
        "stability_score",
        "leakage_audit",
        "beats_one_pass",
        "flagged_adversarial",
        "schema",
        "experiment_id",
        "spec_refs",
    ):
        assert field in errors


def test_deliverable_file_validates_for_req_verify_5088() -> None:
    """REQ-VERIFY-5088: committed Exp5088 artifact satisfies the schema."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"].startswith(
        (
            "success_temporal_consistency_prm_improves_plus_",
            "complete_temporal_consistency_prm_no_win",
        )
    )
    assert artifact["live_llm_invoked"] is False
    assert artifact["model_specs"]["mandatory_model_ids"] == list(mod.MANDATED_MODEL_IDS)
