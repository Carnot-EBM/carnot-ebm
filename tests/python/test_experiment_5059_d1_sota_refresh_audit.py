"""Tests for Exp 5059 D1 SOTA refresh audit.

Spec refs: REQ-VERIFY-5059, SCENARIO-VERIFY-5059.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5059_d1_sota_refresh_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _question_fixture() -> list[tuple[str, str, list[str], list[str]]]:
    return [
        ("fixture:0", "A", ["A", "B"], ["B", "A", "A"]),
        ("fixture:1", "C", ["C", "D"], ["D", "C", "D"]),
        ("fixture:2", "E", ["E", "F"], ["E", "F", "F"]),
        ("fixture:3", "G", ["G", "H"], ["H", "G", "H"]),
    ]


def _frozen_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for question_index, (question_id, gold, choices, answers) in enumerate(_question_fixture()):
        for candidate_index, answer in enumerate(answers):
            rows.append(
                {
                    "schema": "carnot.shared_logprob_candidate_cache_v2.candidate_row.v1",
                    "candidate_id": f"{question_id}/cached-{candidate_index}",
                    "question_id": question_id,
                    "question_index": question_index,
                    "candidate_index": candidate_index,
                    "corpus": "MuSR/murder_mysteries",
                    "question": "Who is the most likely murderer?",
                    "context": "Small fixture case.",
                    "choices": choices,
                    "answer": answer,
                    "gold": gold,
                    "source": "fixture_frozen_464_cache",
                }
            )
    return rows


def _refreshed_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frozen in _frozen_rows():
        answer = str(frozen["answer"])
        question_id = str(frozen["question_id"])
        candidate_index = int(frozen["candidate_index"])
        rows.append(
            {
                "schema": "carnot.experiment_5058_sota_candidate_refresh_inwriting.row.v1",
                "row_id": f"{question_id}/sota5058-{candidate_index:04d}",
                "question_id": question_id,
                "question_index": int(frozen["question_index"]),
                "candidate_index": candidate_index,
                "corpus": "MuSR/murder_mysteries",
                "question": frozen["question"],
                "choices": list(frozen["choices"]),
                "answer_text": answer,
                "parsed_answer": answer,
                "parse_status": "parsed",
                "model_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "model_role": "flagship_moe",
                "model_path": "/models/qwen.gguf",
                "legacy_model_used": False,
                "source_provenance": {
                    "source": "frozen_464_musr_candidate_cache",
                    "source_candidate_id": frozen["candidate_id"],
                    "source_answer_text": answer,
                },
            }
        )
    return rows


def _refresh_artifact(root: Path, *, ready: bool = True) -> dict[str, Any]:
    return {
        "schema": "carnot.experiment_5058_sota_candidate_refresh_inwriting.v1",
        "honest_verdict": "complete_sota_candidate_refresh_ready_d1_d6"
        if ready
        else "complete_sota_candidate_refresh_not_ready_d1_d6",
        "candidate_refresh_ready": ready,
        "candidate_cache_path": (root / mod.EXP5058_CACHE_RELATIVE_PATH).as_posix(),
        "n_questions": 4 if ready else 0,
        "n_candidates": 12 if ready else 0,
        "parse_rate": 1.0 if ready else 0.0,
        "duplicate_rate": 1.0 if ready else 0.0,
        "answer_diversity": {
            "unique_answers": 8 if ready else 0,
            "unique_answer_rate": 0.666667 if ready else 0.0,
            "mean_unique_answers_per_question": 2.0 if ready else 0.0,
        },
        "legacy_models_smoke_only": True,
        "model_specs": {
            "flagship_moe": {"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"},
            "flagship_dense": {"hf_id": "unsloth/gemma-4-31B-it-GGUF"},
            "middle_moe": {"hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF"},
        },
    }


def _powered_scorer_artifact(*, skeleton: bool = False) -> dict[str, Any]:
    return {
        "schema": "carnot.experiment_5045_powered_lora_ebm_eorm_musr.v1",
        "honest_verdict": "blocked_sota_candidate_refresh_unavailable",
        "powered_scorer_available": not skeleton,
        "scorer_trained": not skeleton,
        "train_loss": None if skeleton else 0.123,
        "n_pairs": 0 if skeleton else 64,
        "duration_evidence_s": 0.0 if skeleton else 90.0,
        "checkpoint_path": "" if skeleton else "/checkpoints/d1/epoch_1",
        "verifier_is_oracle": False,
        "headroom_present": True,
        "evaluation": {
            "verifier": {"predictions": ["A", "C", "E", "H"]},
            "paired_correct": {"verifier": [1, 1, 1, 0]},
        },
    }


def _setup_ready_root(tmp_path: Path, *, skeleton: bool = False) -> Path:
    root = tmp_path / "root"
    _write_json(root / mod.EXP5058_RESULT_RELATIVE_PATH, _refresh_artifact(root))
    _write_jsonl(root / mod.EXP5058_CACHE_RELATIVE_PATH, _refreshed_rows())
    _write_jsonl(root / mod.FROZEN_CANDIDATE_CACHE_RELATIVE_PATH, _frozen_rows())
    _write_json(
        root / mod.EXP5045_RESULT_RELATIVE_PATH, _powered_scorer_artifact(skeleton=skeleton)
    )
    return root


def test_req_verify_5059_spec_declares_refresh_audit_contract() -> None:
    """REQ-VERIFY-5059: OpenSpec anchors the Exp5059 audit artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5059",
        "SCENARIO-VERIFY-5059",
        "experiment_5059_d1_sota_refresh_audit.py",
        "results/experiment_5059_d1_sota_refresh_audit.json",
        "blocked_candidate_refresh_unavailable",
        "frozen_candidate_delta",
        "candidate_diversity_sensitivity",
        "proper_musr_win",
        "legacy_models_smoke_only",
    ):
        assert marker in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_verify_5059_blocks_before_scoring_when_refresh_unavailable(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5059: refresh gate failure blocks frozen promotion."""

    root = tmp_path / "root"
    _write_json(root / mod.EXP5058_RESULT_RELATIVE_PATH, _refresh_artifact(root, ready=False))

    artifact = mod.run(root=root, artifact_path=tmp_path / "out.json", write=True)

    assert artifact["honest_verdict"] == "blocked_candidate_refresh_unavailable"
    assert artifact["candidate_refresh_used"] is False
    assert artifact["best_arm_available"] is False
    assert artifact["proper_musr_win"] is False
    assert artifact["accuracy"] is None
    assert artifact["frozen_candidate_delta"] is None
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads((tmp_path / "out.json").read_text(encoding="utf-8")) == artifact


def test_scenario_verify_5059_reports_refresh_metrics_and_frozen_comparison(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5059: refreshed audit isolates scorer and refresh value."""

    root = _setup_ready_root(tmp_path)

    artifact = mod.run(
        root=root,
        artifact_path=tmp_path / "out.json",
        bootstrap_samples=100,
        write=True,
    )

    assert artifact["honest_verdict"].startswith("complete_d1_sota_refresh_audit_no_proper_win")
    assert artifact["candidate_refresh_used"] is True
    assert artifact["best_arm_available"] is True
    assert artifact["legacy_models_smoke_only"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["n_questions"] == 4
    assert artifact["accuracy"] == pytest.approx(0.75)
    assert artifact["tuned_sc_accuracy"] == pytest.approx(0.25)
    assert artifact["delta_vs_tuned_sc"] == pytest.approx(0.5)
    assert artifact["oracle_at_k"] == pytest.approx(1.0)
    assert artifact["headroom_present"] is True
    assert artifact["mcnemar_p"] == pytest.approx(0.5)
    assert artifact["proper_musr_win"] is False
    assert artifact["frozen_candidate_delta"] == pytest.approx(0.5)
    assert artifact["candidate_refresh_value_delta"] == pytest.approx(0.0)
    assert artifact["cached_scorer_fallback_used"] is True
    assert artifact["scorer_source"]["source_artifact"].endswith(
        "experiment_5045_powered_lora_ebm_eorm_musr.json"
    )
    assert artifact["candidate_diversity_sensitivity"]["full_pool"][
        "unique_answer_rate"
    ] == pytest.approx(0.666667)
    assert artifact["candidate_diversity_sensitivity"]["deduplicated_answers"]["n_candidates"] == 8
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads((tmp_path / "out.json").read_text(encoding="utf-8")) == artifact


def test_skeleton_powered_scorer_blocks_instead_of_headlining(tmp_path: Path) -> None:
    """REQ-VERIFY-5059: skeleton scorer evidence cannot become a headline."""

    root = _setup_ready_root(tmp_path, skeleton=True)

    artifact = mod.run(root=root, artifact_path=tmp_path / "out.json", write=True)

    assert artifact["honest_verdict"] == "blocked_powered_scorer_unavailable"
    assert artifact["candidate_refresh_used"] is True
    assert artifact["best_arm_available"] is False
    assert artifact["proper_musr_win"] is False
    assert artifact["scorer_source"]["blocked_reason"] == "scorer_trained_false"
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads((tmp_path / "out.json").read_text(encoding="utf-8")) == artifact


def test_scorer_gate_and_helper_edge_cases(tmp_path: Path) -> None:
    """REQ-VERIFY-5059: helper gates fail closed on malformed inputs."""

    valid = _powered_scorer_artifact()
    cases = [
        (None, "missing_or_malformed_exp5045_artifact"),
        (dict(valid, train_loss=None), "train_loss_missing"),
        (dict(valid, n_pairs=0), "n_pairs_zero"),
        (dict(valid, checkpoint_path=""), "checkpoint_path_missing"),
        (dict(valid, verifier_is_oracle=True), "verifier_is_oracle"),
        (dict(valid, evaluation={"verifier": {"predictions": []}}), "cached_predictions_missing"),
    ]
    for payload, expected_reason in cases:
        ok, reason = mod._scorer_gate(payload)
        assert ok is False
        assert reason == expected_reason

    blank_jsonl = tmp_path / "blank.jsonl"
    blank_jsonl.write_text("\n{}\n", encoding="utf-8")
    assert mod._read_jsonl(blank_jsonl) == [{}]
    assert mod._number(True) is None
    assert mod._number("bad") is None
    assert mod._question_index({"question_index": "bad"}) == 0
    assert mod._candidate_index({"candidate_index": "bad"}) == 0
    assert mod._refresh_cache_path(tmp_path, {}) == tmp_path / mod.EXP5058_CACHE_RELATIVE_PATH
    assert mod._delta_label(None) == "unknown"
    assert (
        mod._build_eval_rows(
            [{"question_id": "missing-gold", "parsed_answer": "A"}],
            frozen_rows=_frozen_rows(),
            refreshed=True,
        )
        == []
    )
    rows = mod._build_eval_rows(_refreshed_rows(), frozen_rows=_frozen_rows(), refreshed=True)
    with pytest.raises(ValueError, match="cached scorer predictions length"):
        mod._projection_metrics(rows, [], seed=1, bootstrap_samples=2)


def test_artifact_schema_errors_report_malformed_required_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5059: schema validation catches malformed audit artifacts."""

    root = _setup_ready_root(tmp_path)
    artifact = mod.run(root=root, artifact_path=tmp_path / "out.json", write=False)

    cases = [
        (lambda a: (a.pop("honest_verdict"), a)[1], "honest_verdict"),
        (lambda a: dict(a, schema="wrong"), "schema"),
        (lambda a: dict(a, spec_refs=["WRONG"]), "spec_refs"),
        (lambda a: dict(a, model_specs=[]), "model_specs"),
        (lambda a: dict(a, candidate_refresh_used="yes"), "candidate_refresh_used"),
        (lambda a: dict(a, best_arm_available="yes"), "best_arm_available"),
        (lambda a: dict(a, verifier_is_oracle=True), "verifier_is_oracle"),
        (lambda a: dict(a, legacy_models_smoke_only=False), "legacy_models_smoke_only"),
        (lambda a: dict(a, n_questions=-1), "n_questions"),
        (lambda a: dict(a, paired_ci95=[0.0]), "paired_ci95"),
        (lambda a: dict(a, accuracy="bad"), "accuracy"),
        (lambda a: dict(a, accuracy=2.0), "accuracy"),
        (lambda a: dict(a, delta_vs_tuned_sc="bad"), "delta_vs_tuned_sc"),
    ]
    for mutate, expected in cases:
        assert expected in mod.artifact_schema_errors(mutate(dict(artifact)))
