"""Tests for Exp 4245 held-out ARC Set-Encoder rerank gate.

Spec refs: REQ-VERIFY-4245, SCENARIO-VERIFY-4245,
SCENARIO-VERIFY-4245-NO-HEADROOM, SCENARIO-VERIFY-4245-DEFERRED.
"""

from __future__ import annotations

import gzip
import inspect
import json
from pathlib import Path

import pytest

from carnot.reporting import arc_set_encoder_beats_vote_4245 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _adversarial_clean(_path: Path) -> dict:
    return {
        "returncode": 0,
        "reports": [
            {
                "flag_count": 0,
                "flags": [],
                "max_severity": 0,
            }
        ],
    }


def _features(vote: float, confidence: float = 0.5) -> dict[str, float]:
    return {
        name: 0.0
        for name in mod.exp4244.FEATURE_NAMES
    } | {
        "vote_weight": float(vote),
        "self_consistency_margin": float(vote) - 0.5,
        "cell_confidence_mean": float(confidence),
        "cell_confidence_margin": float(confidence) - 0.5,
        "grid_height": 2.0,
        "grid_width": 2.0,
        "grid_cells": 4.0,
        "set_candidate_count": 3.0,
        "set_vote_max": float(vote),
        "set_vote_mean": 1.0 / 3.0,
    }


def _write_gate_fixture(
    root: Path,
    *,
    correct_indices: list[int],
    vote_weights: list[list[float]],
    set_encoder_scores: list[list[float]],
    aggregator_trained: bool = True,
    build_verifier_is_oracle: bool = False,
    model_verifier_is_oracle: bool = False,
) -> Path:
    task_ids = [f"mini:task-{index}" for index in range(len(correct_indices))]
    tasks = []
    oof_rows = []
    for task_index, task_id in enumerate(task_ids):
        candidates = []
        correct_index = correct_indices[task_index]
        for candidate_index, vote in enumerate(vote_weights[task_index]):
            candidate_id = f"{task_id}::candidate{candidate_index}"
            is_correct = candidate_index == correct_index
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_index": candidate_index,
                    "features": _features(vote, 0.9 if is_correct else 0.2),
                    "grid": [[candidate_index]],
                    "is_correct": is_correct,
                    "votes": vote,
                }
            )
            oof_rows.append(
                {
                    "candidate_id": candidate_id,
                    "correct": is_correct,
                    "fold": 0,
                    "score": float(set_encoder_scores[task_index][candidate_index]),
                    "task_id": task_id,
                    "train_task_ids": [other for other in task_ids if other != task_id],
                }
            )
        vote_top = max(candidates, key=lambda item: (item["features"]["vote_weight"], -item["candidate_index"]))
        tasks.append(
            {
                "candidate_count": len(candidates),
                "candidates": candidates,
                "oracle_present": True,
                "raw_task_id": f"task-{task_index}",
                "source_id": "mini",
                "task_id": task_id,
                "vote_top_candidate_id": vote_top["candidate_id"],
                "wrong_majority": vote_top["candidate_index"] != correct_index,
            }
        )

    pool_rel = Path("results/mini_4245_pool.json.gz")
    pool_path = root / pool_rel
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(pool_path, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                "candidate_n": sum(len(task["candidates"]) for task in tasks),
                "positive_candidate_n": len(tasks),
                "random_seed": 4243,
                "reproducibility_checksum": "sha256:" + "1" * 64,
                "schema": "carnot.arc_candidate_pool_grow.v1",
                "source_paths": [],
                "source_sha256": {},
                "spec_refs": ["REQ-CAPSTONE-4243"],
                "task_n": len(tasks),
                "tasks": tasks,
                "wrong_majority_n": sum(int(task["wrong_majority"]) for task in tasks),
            },
            handle,
        )
    _write_json(
        root / "results" / "experiment_4243_arc_candidate_pool_grow.json",
        {
            "arc_pool_grown": True,
            "held_out_task_n": len(tasks),
            "pool_artifact_path": str(pool_rel),
            "positive_candidate_n": len(tasks),
            "random_seed": 4243,
            "reproducibility_checksum": "sha256:" + "2" * 64,
            "verifier_is_oracle": False,
            "wrong_majority_n": sum(int(task["wrong_majority"]) for task in tasks),
        },
    )

    model_path = root / "results" / "experiment_4244_arc_set_encoder_aggregator_model.json"
    _write_json(
        model_path,
        {
            "feature_names": list(mod.exp4244.FEATURE_NAMES),
            "held_out_task_n": len(tasks),
            "model": {"model_type": "fixture_oof_only"},
            "model_specs": {"architecture": "fixture_set_encoder"},
            "pool_artifact_path": str(pool_path),
            "random_seed": 4244,
            "reproducibility_checksum": "sha256:" + "3" * 64,
            "set_encoder_oof": {
                "auroc": 1.0,
                "ci95": [1.0, 1.0],
                "fold_task_ids": [task_ids],
                "rows": oof_rows,
            },
            "spec_refs": ["REQ-VERIFY-4244"],
            "verifier_is_oracle": model_verifier_is_oracle,
        },
    )
    _write_json(
        root / "results" / "experiment_4244_arc_set_encoder_aggregator_build.json",
        {
            "aggregator_trained": aggregator_trained,
            "held_out_task_n": len(tasks),
            "honest_verdict": "complete: fixture",
            "learned_verifier_path": str(model_path),
            "oracle_distinct_auroc": 1.0,
            "random_seed": 4244,
            "reproducibility_checksum": "sha256:" + "4" * 64,
            "set_encoder_vs_logistic_auroc_delta": 0.1,
            "verifier_is_oracle": build_verifier_is_oracle,
            "wrong_majority_n": sum(int(task["wrong_majority"]) for task in tasks),
        },
    )
    return root


def test_req_4245_spec_declares_set_encoder_gate_contract() -> None:
    """REQ-VERIFY-4245: OpenSpec declares the decisive held-out rerank gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4245",
        "SCENARIO-VERIFY-4245",
        "SCENARIO-VERIFY-4245-NO-HEADROOM",
        "SCENARIO-VERIFY-4245-DEFERRED",
        "python/carnot/reporting/arc_set_encoder_beats_vote_4245.py",
        "results/experiment_4245_arc_set_encoder_beats_vote.py",
        "complete_arc_oracle_distinct_gate_deferred_no_built_aggregator",
        "set_encoder_minus_vote_delta",
        "set_encoder_minus_vote_ci95",
        "margin_override_minus_vote",
        "matched_control_delta",
        "held_out_task_n",
        "verifier_is_oracle=false",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_4245_set_encoder_beats_vote_with_controls(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4245: learned OOF Set-Encoder scores beat vote."""

    _write_gate_fixture(
        tmp_path,
        correct_indices=[1, 0, 2, 1],
        vote_weights=[[9, 1, 0], [9, 1, 0], [9, 0, 1], [9, 1, 0]],
        set_encoder_scores=[[0.1, 0.9, 0.2], [0.8, 0.2, 0.1], [0.1, 0.2, 0.9], [0.1, 0.9, 0.2]],
    )

    artifact = mod.run(tmp_path, adversarial_runner=_adversarial_clean)

    mod.validate_artifact(artifact)
    assert artifact["headline_outcome"] == "arc_oracle_distinct_set_encoder_beats_vote"
    assert artifact["honest_verdict"] == "complete: arc_oracle_distinct_set_encoder_beats_vote"
    assert artifact["oracle_distinct_beats_vote"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["held_out_task_n"] == 4
    assert artifact["pass_rates"]["vote_at_1"] == pytest.approx(0.25)
    assert artifact["pass_rates"]["set_encoder_at_1"] == pytest.approx(1.0)
    assert artifact["oracle_at_k"] == pytest.approx(1.0)
    assert artifact["pass_rates"]["matched_control_at_1"] == pytest.approx(0.25)
    assert artifact["set_encoder_minus_vote_delta"] == pytest.approx(0.75)
    assert artifact["set_encoder_minus_vote_ci95"][0] > 0.0
    assert artifact["matched_control_delta"] == pytest.approx(0.75)
    assert artifact["margin_override_minus_vote"] == pytest.approx(0.75)
    assert artifact["bootstrap_resamples"] >= 2000
    assert artifact["margin_trigger_threshold"] == pytest.approx(mod.MARGIN_TRIGGER_THRESHOLD)
    assert artifact["margin_threshold_policy"] == "pre_registered_a2_fold_fixed_threshold"
    assert artifact["headroom_exists"] is True
    assert artifact["clt_floor_caveat"] is True
    assert artifact["acceptance_gate"] is True
    assert artifact["adversarial_verify"]["status"] == "clean"
    assert artifact["task_rows"][0]["set_encoder_train_task_excluded"] is True


def test_scenario_4245_no_headroom_positive_control_blocks_false_null(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4245-NO-HEADROOM: oracle@K ~= vote is uninformative."""

    _write_gate_fixture(
        tmp_path,
        correct_indices=[0, 0, 0],
        vote_weights=[[9, 1], [8, 2], [7, 3]],
        set_encoder_scores=[[0.1, 0.9], [0.8, 0.2], [0.7, 0.1]],
    )

    artifact = mod.run(tmp_path, adversarial_runner=_adversarial_clean)

    mod.validate_artifact(artifact)
    assert artifact["headline_outcome"] == "arc_oracle_distinct_no_headroom_uninformative"
    assert artifact["honest_verdict"] == "complete_arc_oracle_distinct_no_headroom_uninformative"
    assert artifact["oracle_distinct_beats_vote"] is False
    assert artifact["oracle_at_k"] == artifact["pass_rates"]["vote_at_1"] == pytest.approx(1.0)
    assert artifact["headroom_exists"] is False
    assert "set_encoder_failed" not in artifact["honest_verdict"]

    tied_root = tmp_path / "tied"
    _write_gate_fixture(
        tied_root,
        correct_indices=[1, 0],
        vote_weights=[[9, 1], [9, 1]],
        set_encoder_scores=[[0.9, 0.1], [0.9, 0.1]],
    )
    tied = mod.run(tied_root, adversarial_runner=_adversarial_clean)
    assert tied["headline_outcome"] == "arc_oracle_distinct_ties_vote_at_power_on_grown_pool"
    assert tied["headroom_exists"] is True
    assert tied["oracle_distinct_beats_vote"] is False
    assert tied["set_encoder_minus_vote_ci95"] == [0.0, 0.0]


def test_scenario_4245_missing_or_unreadable_set_encoder_defers(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4245-DEFERRED: missing built Set-Encoder stops scoring."""

    artifact = mod.run(tmp_path, adversarial_runner=_adversarial_clean)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.DEFERRED_VERDICT
    assert artifact["headline_outcome"] == "arc_oracle_distinct_gate_deferred_no_built_aggregator"
    assert artifact["oracle_distinct_beats_vote"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["held_out_task_n"] == 0
    assert artifact["acceptance_gate"] is True

    _write_json(
        tmp_path / "results" / "experiment_4244_arc_set_encoder_aggregator_build.json",
        {
            "aggregator_trained": True,
            "learned_verifier_path": str(tmp_path / "results" / "missing.json"),
            "verifier_is_oracle": False,
        },
    )
    unreadable = mod.run(tmp_path, adversarial_runner=_adversarial_clean)
    assert unreadable["honest_verdict"] == mod.DEFERRED_VERDICT

    bad_cases = [
        {"aggregator_trained": False, "verifier_is_oracle": False, "learned_verifier_path": ""},
        {"aggregator_trained": True, "verifier_is_oracle": True, "learned_verifier_path": ""},
        {"aggregator_trained": True, "verifier_is_oracle": False, "learned_verifier_path": ""},
    ]
    for index, a2_payload in enumerate(bad_cases):
        case_root = tmp_path / f"bad-a2-{index}"
        _write_json(
            case_root / "results" / "experiment_4244_arc_set_encoder_aggregator_build.json",
            a2_payload,
        )
        assert mod.run(case_root, adversarial_runner=_adversarial_clean)["honest_verdict"] == (
            mod.DEFERRED_VERDICT
        )

    malformed_root = tmp_path / "malformed-a2"
    malformed_path = malformed_root / "results" / "experiment_4244_arc_set_encoder_aggregator_build.json"
    malformed_path.parent.mkdir(parents=True, exist_ok=True)
    malformed_path.write_text("[]", encoding="utf-8")
    assert mod.run(malformed_root, adversarial_runner=_adversarial_clean)["honest_verdict"] == (
        mod.DEFERRED_VERDICT
    )

    list_model_root = tmp_path / "list-model"
    list_model = list_model_root / "results" / "model.json"
    list_model.parent.mkdir(parents=True, exist_ok=True)
    list_model.write_text("[]", encoding="utf-8")
    _write_json(
        list_model_root / "results" / "experiment_4244_arc_set_encoder_aggregator_build.json",
        {
            "aggregator_trained": True,
            "learned_verifier_path": str(list_model),
            "verifier_is_oracle": False,
        },
    )
    assert mod.run(list_model_root, adversarial_runner=_adversarial_clean)["honest_verdict"] == (
        mod.DEFERRED_VERDICT
    )

    empty_rows_root = tmp_path / "empty-oof-rows"
    _write_gate_fixture(
        empty_rows_root,
        correct_indices=[0],
        vote_weights=[[1, 0]],
        set_encoder_scores=[[0.8, 0.1]],
    )
    empty_model = empty_rows_root / "results" / "experiment_4244_arc_set_encoder_aggregator_model.json"
    empty_payload = json.loads(empty_model.read_text(encoding="utf-8"))
    empty_payload["set_encoder_oof"]["rows"] = []
    _write_json(empty_model, empty_payload)
    assert mod.run(empty_rows_root, adversarial_runner=_adversarial_clean)["honest_verdict"] == (
        mod.DEFERRED_VERDICT
    )

    oracle_model_root = tmp_path / "oracle-model"
    _write_gate_fixture(
        oracle_model_root,
        correct_indices=[0],
        vote_weights=[[1, 0]],
        set_encoder_scores=[[0.8, 0.1]],
        model_verifier_is_oracle=True,
    )
    assert mod.run(oracle_model_root, adversarial_runner=_adversarial_clean)["honest_verdict"] == (
        mod.DEFERRED_VERDICT
    )


def test_validation_and_helper_edges_are_explicit(tmp_path: Path) -> None:
    """REQ-VERIFY-4245: schema, bootstrap, and score loading stay deterministic."""

    base = mod._deferred_artifact(
        mod.DEFERRED_VERDICT,
        random_seed=mod.RANDOM_SEED,
        checksum="sha256:abc",
        duration_s=0.1,
    )
    invalid_cases = [
        ({key: value for key, value in base.items() if key != "oracle_at_k"}, "missing required"),
        ({**base, "honest_verdict": "pending"}, "terminal-prefixed"),
        ({**base, "oracle_distinct_beats_vote": {"value": False}}, "bare bool"),
        ({**base, "set_encoder_minus_vote_delta": None}, "bare float"),
        ({**base, "set_encoder_minus_vote_ci95": [0.0]}, "ci95"),
        ({**base, "held_out_task_n": 4245.0}, "bare int"),
        ({**base, "verifier_is_oracle": True}, "verifier_is_oracle"),
        ({**base, "random_seed": 4245.0}, "bare int"),
        ({**base, "field_principles": {}}, "field_principles"),
        ({**base, "spec_refs": []}, "spec_refs"),
    ]
    for payload, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)

    assert mod._rate([]) == 0.0
    assert mod._bootstrap_ci95([], random_seed=1, resamples=10) == [0.0, 0.0]
    assert mod._bootstrap_ci95([1.0, -1.0], random_seed=1, resamples=0) == [0.0, 0.0]
    assert mod._ci_excludes_zero([0.1, 0.2]) is True
    assert mod._ci_excludes_zero([-0.2, -0.1]) is True
    assert mod._ci_excludes_zero([-0.1, 0.1]) is False
    assert mod._clean_adversarial_report({"reports": [{"flags": []}]})["status"] == "clean"
    flagged = mod._clean_adversarial_report(
        {"reports": [{"flags": [{"kind": "CIRCULAR_MOAT_OVERCLAIM", "severity": "critical"}]}]}
    )
    assert flagged["status"] == "flagged"
    assert flagged["circular_moat_overclaim_clean"] is False
    assert mod._oof_score_map(
        {"set_encoder_oof": {"rows": [None, {"candidate_id": 7}, {"candidate_id": "c", "task_id": "t"}]}}
    ) == {"c": (0.0, True, 0)}
    assert mod._oof_score_map({"set_encoder_oof": {"rows": {}}}) == {}
    assert mod._safe_float(True) == 0.0
    assert mod._safe_float("bad") == 0.0
    assert mod._safe_int(False) == 0
    assert mod._safe_int("bad") == 0

    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    with pytest.raises(mod.BlockedRun, match="blocked_malformed_json_artifact"):
        mod._read_json_object(list_json)

    with pytest.raises(mod.BlockedRun, match=mod.DEFERRED_VERDICT):
        mod.load_heldout_pool(
            tmp_path / "missing-grown-pool",
            {
                "set_encoder_oof": {
                    "rows": [
                        {
                            "candidate_id": "c",
                            "score": 0.1,
                            "task_id": "t",
                            "train_task_ids": [],
                        }
                    ]
                },
                "verifier_is_oracle": False,
            },
            tmp_path / "missing-model.json",
        )

    fallback_root = _write_gate_fixture(
        tmp_path / "fallback",
        correct_indices=[0],
        vote_weights=[[1, 0]],
        set_encoder_scores=[[0.8, 0.1]],
    )
    model_path = fallback_root / "results" / "experiment_4244_arc_set_encoder_aggregator_model.json"
    with pytest.raises(mod.BlockedRun, match="no_heldout_set_encoder_scores"):
        mod.load_heldout_pool(
            fallback_root,
            {"set_encoder_oof": {"rows": []}, "verifier_is_oracle": False},
            model_path,
        )

    nonheldout_root = _write_gate_fixture(
        tmp_path / "nonheldout-score",
        correct_indices=[0],
        vote_weights=[[1, 0]],
        set_encoder_scores=[[0.8, 0.1]],
    )
    nonheldout_model = nonheldout_root / "results" / "experiment_4244_arc_set_encoder_aggregator_model.json"
    nonheldout_payload = json.loads(nonheldout_model.read_text(encoding="utf-8"))
    for row in nonheldout_payload["set_encoder_oof"]["rows"]:
        row["train_task_ids"] = [row["task_id"]]
    with pytest.raises(mod.BlockedRun, match="no_heldout_set_encoder_scores"):
        mod.load_heldout_pool(nonheldout_root, nonheldout_payload, nonheldout_model)

    specs_root = _write_gate_fixture(
        tmp_path / "nondict-specs",
        correct_indices=[0],
        vote_weights=[[1, 0]],
        set_encoder_scores=[[0.8, 0.1]],
    )
    specs_model = specs_root / "results" / "experiment_4244_arc_set_encoder_aggregator_model.json"
    specs_payload = json.loads(specs_model.read_text(encoding="utf-8"))
    specs_payload["model_specs"] = []
    pool = mod.load_heldout_pool(specs_root, specs_payload, specs_model)
    assert pool.model_specs == {}


def test_module_does_not_rank_with_execution_or_correctness() -> None:
    """REQ-VERIFY-4245: learned inference remains oracle-distinct."""

    source = inspect.getsource(mod)
    assert "arc_gap4_execution_verifier" not in source
    assert "Gap4ExecutionVerifier" not in source
    assert "extract_dsl_rules" not in source
    assert "apply_rule" not in source
    assert "get_consistency_energy" not in source
    assert "key=lambda candidate: (candidate.correct" not in source
