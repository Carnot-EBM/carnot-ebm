"""Tests for Exp 5076 D6 efficiency replication.

Spec refs: REQ-VERIFY-5076, SCENARIO-VERIFY-5076.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5076_d6_efficiency_replication as exp


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


def _model_specs() -> dict[str, dict[str, str | None]]:
    return {
        role: {"hf_id": hf_id, "resolved_path": f"/models/{role}.gguf"}
        for role, hf_id in exp.MANDATED_MODEL_SPECS.items()
    }


def _candidate_rows(n_questions: int = 6, *, legacy: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for question_index in range(n_questions):
        for candidate_index, answer in enumerate(("A", "B")):
            rows.append(
                {
                    "schema": "carnot.experiment_5058_sota_candidate_refresh_inwriting.row.v1",
                    "row_id": f"fixture:{question_index}/sota5058-{candidate_index:04d}",
                    "question_id": f"fixture:{question_index}",
                    "question_index": question_index,
                    "candidate_index": candidate_index,
                    "question": "Who is the most likely murderer?",
                    "choices": ["A", "B"],
                    "answer_text": answer,
                    "parsed_answer": answer,
                    "parse_status": "parsed",
                    "prompt_hash": f"prompt-{question_index:03d}",
                    "model_id": exp.MANDATED_MODEL_SPECS["flagship_moe"],
                    "model_role": "flagship_moe",
                    "model_path": "/models/qwen.gguf",
                    "legacy_model_used": legacy,
                    "structured_constraints": {
                        "allowed_answers": ["A", "B"],
                        "answer_in_allowed_choices": True,
                    },
                }
            )
    return rows


def _exp5058(root: Path, *, ready: bool = True, flagged: bool = True) -> dict[str, Any]:
    return {
        "schema": "carnot.experiment_5058_sota_candidate_refresh_inwriting.v1",
        "honest_verdict": "complete_sota_candidate_refresh_ready_d1_d6"
        if ready
        else "blocked_candidate_refresh_unavailable",
        "candidate_refresh_ready": ready,
        "candidate_cache_path": (root / exp.EXP5058_CACHE_RELATIVE_PATH).as_posix(),
        "n_questions": 6 if ready else 0,
        "n_candidates": 12 if ready else 0,
        "legacy_models_smoke_only": True,
        "flagged_adversarial": flagged,
        "model_specs": _model_specs(),
    }


def _exp5059(
    *,
    verifier: list[int] | None = None,
    tuned_sc: list[int] | None = None,
    cached_judge: list[int] | None = None,
    predictions: list[str | None] | None = None,
    best_arm_available: bool = True,
) -> dict[str, Any]:
    verifier_correct = verifier if verifier is not None else [1, 1, 0, 1, 0, 1]
    tuned_correct = tuned_sc if tuned_sc is not None else [1, 1, 0, 1, 0, 1]
    paired_correct: dict[str, list[int]] = {
        "verifier": verifier_correct,
        "tuned_self_consistency": tuned_correct,
    }
    if cached_judge is not None:
        paired_correct["cached_sota_judge"] = cached_judge
    return {
        "schema": "carnot.experiment_5059_d1_sota_refresh_audit.v1",
        "honest_verdict": "complete_d1_sota_refresh_audit_no_proper_win_plus_0p000",
        "best_arm_available": best_arm_available,
        "verifier_is_oracle": False,
        "legacy_models_smoke_only": True,
        "flagged_adversarial": True,
        "model_specs": {"mandated_sota": dict(exp.MANDATED_MODEL_SPECS)},
        "refreshed_candidate_metrics": {
            "predictions": predictions or ["A", "B", "A", "B", "A", "B"],
            "paired_correct": paired_correct,
        },
    }


def _exp5061() -> dict[str, Any]:
    return {
        "schema": "carnot.experiment_5061_tool_first_cascade.v1",
        "honest_verdict": "success_tool_first_cascade_parity_at_0pct_judge_calls",
        "cascade_executed": True,
        "judge_call_fraction": 0.0,
        "efficiency_win": True,
        "verifier_is_oracle": False,
    }


def _exp5071(*, live: bool = False) -> dict[str, Any]:
    return {
        "schema": "carnot.experiment_5071_gguf_logprob_preflight.v466",
        "honest_verdict": "complete_gguf_logprob_preflight_partial_ready",
        "sota_models_ready": True,
        "completion_endpoint_ready": live,
        "logprob_endpoint_ready": live,
        "top_logprob_or_confidence_ready": live,
        "live_completion_invoked": live,
        "flagged_adversarial": False,
        "model_specs": {
            "headline_models": list(exp.MANDATED_MODEL_SPECS.values()),
            "resolved_models": _model_specs(),
        },
    }


def _setup_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    _write_json(root / exp.EXP5058_RESULT_RELATIVE_PATH, _exp5058(root))
    _write_json(root / exp.EXP5059_RESULT_RELATIVE_PATH, _exp5059())
    _write_json(root / exp.EXP5061_RESULT_RELATIVE_PATH, _exp5061())
    _write_json(root / exp.EXP5071_RESULT_RELATIVE_PATH, _exp5071())
    _write_jsonl(root / exp.EXP5058_CACHE_RELATIVE_PATH, _candidate_rows())
    return root


def test_req_verify_5076_spec_declares_replication_contract() -> None:
    """REQ-VERIFY-5076: OpenSpec anchors the D6 replication artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    module_text = (REPO / exp.MODULE_RELATIVE_PATH).read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5076",
        "SCENARIO-VERIFY-5076",
        "experiment_5076_d6_efficiency_replication.py",
        "results/experiment_5076_d6_efficiency_replication_v466.json",
        "judge-only",
        "tool-first",
        "uncertainty-routed",
        "success_d6_efficiency_pareto_win_no_accuracy_headline",
        "complete_d6_replication_no_pareto_win",
        "accuracy_headline_allowed",
    ):
        assert marker in spec
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for hf_id in exp.MANDATED_MODEL_SPECS.values():
        assert hf_id in spec
        assert hf_id in module_text


def test_scenario_verify_5076_reports_pareto_without_accuracy_headline(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5076: equal accuracy at lower cost is efficiency-only."""

    root = _setup_root(tmp_path)
    artifact_path = tmp_path / "out.json"

    artifact = exp.run(
        root=root,
        artifact_path=artifact_path,
        bootstrap_samples=128,
        now=lambda: 100.0,
        write=True,
    )

    assert artifact["honest_verdict"] == "success_d6_efficiency_pareto_win_no_accuracy_headline"
    assert artifact["n_questions"] == 6
    assert artifact["verifier_is_oracle"] is False
    assert artifact["judge_only_accuracy"] == pytest.approx(4 / 6)
    assert artifact["cascade_accuracy"] == pytest.approx(4 / 6)
    assert artifact["delta_vs_judge_only"] == pytest.approx(0.0)
    assert artifact["ci95_delta"] == [0.0, 0.0]
    assert artifact["mcnemar_p"] == pytest.approx(1.0)
    assert artifact["judge_call_fraction"] == pytest.approx(0.0)
    assert artifact["tool_call_count"] == 12
    assert artifact["cheap_verifier_call_count"] == 6
    assert artifact["judge_call_count"] == 0
    assert artifact["efficiency_win"] is True
    assert artifact["accuracy_headline_allowed"] is False
    assert artifact["flagged_adversarial"] is False
    assert artifact["same_question_candidate_set"] is True
    assert artifact["sample_cleanliness"]["row_clean"] is True
    assert artifact["upstream_flagged_adversarial_sources"] == [
        exp.EXP5058_RESULT_RELATIVE_PATH,
        exp.EXP5059_RESULT_RELATIVE_PATH,
    ]
    assert artifact["arms"]["judge_only"]["question_set_hash"] == artifact["question_set_hash"]
    assert artifact["arms"]["tool_first"]["question_set_hash"] == artifact["question_set_hash"]
    assert artifact["arms"]["uncertainty_routed"]["status"] == "not_executed_no_live_or_cached_judge"
    assert artifact["cost_proxy"]["tool_first"]["total_cost_units"] < artifact["cost_proxy"]["judge_only"]["total_cost_units"]
    assert artifact["latency"]["tool_first_s"] >= 0.0
    assert artifact["oracle_distinctness"]["selector_answer_key_visible"] is False
    assert exp.artifact_schema_errors(artifact) == []
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact


def test_req_verify_5076_blocks_when_sample_or_pairs_unavailable(tmp_path: Path) -> None:
    """REQ-VERIFY-5076: missing clean rows or paired vectors fails closed."""

    root = tmp_path / "root"
    _write_json(root / exp.EXP5058_RESULT_RELATIVE_PATH, _exp5058(root, ready=False))
    _write_json(root / exp.EXP5059_RESULT_RELATIVE_PATH, _exp5059())
    _write_json(root / exp.EXP5061_RESULT_RELATIVE_PATH, _exp5061())
    _write_json(root / exp.EXP5071_RESULT_RELATIVE_PATH, _exp5071())

    refresh_blocked = exp.run(root=root, artifact_path=tmp_path / "refresh.json", write=True)

    assert refresh_blocked["honest_verdict"] == "blocked_d6_replication_clean_sample_unavailable"
    assert refresh_blocked["n_questions"] == 0
    assert refresh_blocked["efficiency_win"] is False
    assert exp.artifact_schema_errors(refresh_blocked) == []

    _write_json(root / exp.EXP5058_RESULT_RELATIVE_PATH, _exp5058(root, ready=True))
    _write_jsonl(root / exp.EXP5058_CACHE_RELATIVE_PATH, _candidate_rows(6, legacy=True))
    dirty_blocked = exp.run(root=root, artifact_path=tmp_path / "dirty.json", write=False)
    assert dirty_blocked["honest_verdict"] == "blocked_d6_replication_clean_sample_unavailable"
    assert dirty_blocked["sample_cleanliness"]["row_clean"] is False

    _write_jsonl(root / exp.EXP5058_CACHE_RELATIVE_PATH, _candidate_rows())
    _write_json(
        root / exp.EXP5059_RESULT_RELATIVE_PATH,
        _exp5059(verifier=[], tuned_sc=[]),
    )
    pair_blocked = exp.run(root=root, artifact_path=tmp_path / "pairs.json", write=False)
    assert pair_blocked["honest_verdict"] == "blocked_d6_replication_paired_correctness_unavailable"
    assert pair_blocked["same_question_candidate_set"] is True
    assert exp.artifact_schema_errors(pair_blocked) == []

    _write_json(
        root / exp.EXP5059_RESULT_RELATIVE_PATH,
        _exp5059(best_arm_available=False),
    )
    best_arm_blocked = exp.run(root=root, artifact_path=tmp_path / "best_arm.json", write=False)
    assert best_arm_blocked["honest_verdict"] == "blocked_d6_replication_paired_correctness_unavailable"
    assert "Exp5059 best_arm_available" in best_arm_blocked["blocked_error"]


def test_req_verify_5076_no_pareto_when_cost_or_ci_gate_fails(tmp_path: Path) -> None:
    """REQ-VERIFY-5076: cost/Pareto claims require both cost and CI gates."""

    root = _setup_root(tmp_path)
    _write_json(
        root / exp.EXP5059_RESULT_RELATIVE_PATH,
        _exp5059(
            verifier=[1, 0, 0, 0, 0, 0],
            tuned_sc=[1, 1, 0, 1, 0, 1],
            predictions=["A", "B", "A", "B", "A", "B"],
        ),
    )

    artifact = exp.run(
        root=root,
        artifact_path=tmp_path / "no_pareto.json",
        bootstrap_samples=128,
        write=False,
    )

    assert artifact["honest_verdict"] == "complete_d6_replication_no_pareto_win"
    assert artifact["delta_vs_judge_only"] < 0.0
    assert artifact["ci95_delta"][0] < 0.0
    assert artifact["efficiency_win"] is False
    assert artifact["accuracy_headline_allowed"] is False
    assert exp.artifact_schema_errors(artifact) == []


def test_req_verify_5076_abstain_routes_charge_cached_and_replay_fallbacks(tmp_path: Path) -> None:
    """REQ-VERIFY-5076: abstentions charge cached judges separately from replay tools."""

    root = _setup_root(tmp_path)
    _write_json(
        root / exp.EXP5059_RESULT_RELATIVE_PATH,
        _exp5059(
            verifier=[0, 0, 0, 0, 0, 0],
            tuned_sc=[1, 1, 0, 1, 0, 1],
            predictions=[None, None, None, None, None, None],
        ),
    )
    replay_fallback = exp.run(root=root, artifact_path=tmp_path / "replay.json", write=False)

    assert replay_fallback["judge_call_count"] == 0
    assert replay_fallback["tool_call_count"] == 18
    assert replay_fallback["route_counts"]["comparator_replay_fallback"] == 6
    assert replay_fallback["arms"]["uncertainty_routed"]["status"] == "not_executed_no_live_or_cached_judge"

    _write_json(
        root / exp.EXP5059_RESULT_RELATIVE_PATH,
        _exp5059(
            verifier=[0, 0, 0, 0, 0, 0],
            tuned_sc=[0, 0, 0, 0, 0, 0],
            cached_judge=[1, 1, 0, 1, 0, 1],
            predictions=[None, None, None, None, None, None],
        ),
    )
    cached_fallback = exp.run(root=root, artifact_path=tmp_path / "cached.json", write=False)

    assert cached_fallback["judge_only_source"] == "cached_sota_judge"
    assert cached_fallback["judge_call_count"] == 6
    assert cached_fallback["judge_call_fraction"] == pytest.approx(1.0)
    assert cached_fallback["route_counts"]["judge_fallback"] == 6
    assert cached_fallback["honest_verdict"] == "complete_d6_replication_no_pareto_win"


def test_req_verify_5076_schema_and_helper_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-5076: helpers reject malformed metrics and oracle flags."""

    missing = tmp_path / "missing.json"
    assert exp.read_json_object(missing) is None
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    assert exp.read_json_object(bad_json) is None
    mixed_jsonl = tmp_path / "mixed.jsonl"
    mixed_jsonl.write_text('\n{bad}\n{"ok": true}\n[]\n', encoding="utf-8")
    assert exp.read_jsonl(mixed_jsonl) == [{"ok": True}]
    assert exp.read_jsonl(tmp_path / "missing.jsonl") == []

    assert exp._as_binary_list("bad") == []
    assert exp._as_binary_list([1, 0, "1"]) == [1, 0, 1]
    assert exp._as_binary_list([1, 2]) == []
    assert exp._number(True) is None
    assert exp._number("nan") is None
    assert exp._accuracy([]) is None
    assert exp._rate(1, 0) == 0.0
    assert exp._question_id({"question_index": 7}) == "question:7"
    assert exp._paired_correct(None) == {}
    assert exp._paired_correct({"paired_correct": {"verifier": [1]}}) == {"verifier": [1]}
    assert exp._prediction_list(None) == []
    assert exp._prediction_list({"predictions": "bad"}) == []
    assert exp._correct_vectors(
        {"refreshed_candidate_metrics": {"paired_correct": {"verifier": [1], "cached_sota_judge": [0]}}}
    ) == ([1], "cached_sota_judge", [0])

    dirty_rows = [
        {
            **_candidate_rows(1)[0],
            "row_id": "dup",
            "model_id": "legacy/model",
            "parse_status": "failed",
            "choices": [],
        },
        {**_candidate_rows(1)[1], "row_id": "dup"},
    ]
    cleanliness, question_ids = exp._sample_cleanliness(_exp5058(tmp_path), dirty_rows)
    assert question_ids == ["fixture:0"]
    assert cleanliness["row_clean"] is False
    assert set(cleanliness["errors"]) >= {
        "non_mandated_model_rows_present",
        "unparsed_rows_present",
        "missing_choices",
        "duplicate_row_ids",
    }

    malformed = {
        "schema": exp.SCHEMA,
        "spec_refs": list(exp.SPEC_REFS),
        "honest_verdict": "success_d6_efficiency_pareto_win_no_accuracy_headline",
        "duration_s": 0.0,
        "inference_substrate": exp.REPLAY_SUBSTRATE,
        "model_specs": {},
        "verifier_is_oracle": True,
        "n_questions": -1,
        "judge_only_accuracy": "bad",
        "cascade_accuracy": 0.5,
        "delta_vs_judge_only": "bad",
        "ci95_delta": [0.0],
        "judge_call_fraction": 0.0,
        "tool_call_count": 0,
        "latency": {},
        "cost_proxy": {},
        "efficiency_win": "yes",
        "accuracy_headline_allowed": False,
        "flagged_adversarial": False,
        "field_principles": {},
    }

    errors = exp.artifact_schema_errors(malformed)

    for field in (
        "model_specs",
        "verifier_is_oracle",
        "n_questions",
        "judge_only_accuracy",
        "delta_vs_judge_only",
        "ci95_delta",
        "efficiency_win",
        "field_principles",
    ):
        assert field in errors
