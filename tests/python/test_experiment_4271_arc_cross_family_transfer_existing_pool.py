"""Tests for Exp 4271 ARC cross-family transfer on the existing pool.

Spec refs: REQ-VERIFY-4271, SCENARIO-VERIFY-4271.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import arc_cross_family_transfer_existing_pool_4271 as mod
from carnot.reporting import arc_set_encoder_aggregator_4244 as exp4244


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _adversarial_clean(_path: Path) -> dict[str, Any]:
    return {"returncode": 0, "reports": [{"flag_count": 0, "flags": [], "max_severity": 0}]}


def _features(vote: float, confidence: float) -> dict[str, float]:
    values = {name: 0.0 for name in exp4244.FEATURE_NAMES}
    values.update(
        {
            "vote_weight": float(vote),
            "self_consistency_margin": float(vote) - 0.5,
            "vote_weight_rank_fraction": float(vote),
            "cell_confidence_mean": float(confidence),
            "cell_confidence_margin": float(confidence) - 0.5,
            "cell_confidence_rank_fraction": float(confidence),
            "grid_height": 2.0,
            "grid_width": 2.0,
            "grid_cells": 4.0,
            "set_candidate_count": 2.0,
            "set_vote_mean": 0.5,
            "set_vote_max": float(vote),
        }
    )
    return values


def _write_cross_family_fixture(
    root: Path,
    *,
    feasible: bool = True,
    vote_all_correct: bool = False,
) -> dict[str, float]:
    task_specs = [
        ("family-a", 0, "task-0", 1, [0.9, 0.1], [0.1, 0.9]),
        ("family-a", 0, "task-1", 1, [0.9, 0.1], [0.1, 0.9]),
        ("family-b", 1, "task-2", 1, [0.9, 0.1], [0.1, 0.9]),
        ("family-b", 1, "task-3", 0, [0.9, 0.1], [0.9, 0.1]),
    ]
    if vote_all_correct:
        task_specs = [
            (family, fold, raw_task_id, 0, votes, [0.9, 0.1])
            for family, fold, raw_task_id, _correct_index, votes, _scores in task_specs
        ]

    score_by_candidate: dict[str, float] = {}
    tasks: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for family_id, fold, raw_task_id, correct_index, votes, scores in task_specs:
        task_id = f"fixture:{raw_task_id}"
        candidates = []
        for candidate_index, (vote, score) in enumerate(zip(votes, scores, strict=True)):
            candidate_id = f"{task_id}::candidate{candidate_index}"
            is_correct = candidate_index == correct_index
            score_by_candidate[candidate_id] = score
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_index": candidate_index,
                    "features": _features(vote, confidence=score),
                    "grid": [[candidate_index]],
                    "is_correct": is_correct,
                    "source_kinds": ["gold_flag"] if is_correct else ["pool_candidate"],
                    "votes": vote,
                }
            )
        tasks.append(
            {
                "candidate_count": len(candidates),
                "candidates": candidates,
                "oracle_present": True,
                "raw_task_id": raw_task_id,
                "source_id": "fixture",
                "task_id": task_id,
                "vote_top_candidate_id": f"{task_id}::candidate0",
                "wrong_majority": correct_index != 0,
            }
        )
        manifest_rows.append(
            {
                "family_id": family_id,
                "fold": fold,
                "game_id": None,
                "raw_task_id": raw_task_id,
                "recovered_by": "fixture_taxonomy",
                "source_id": "fixture",
                "source_join_found": True,
                "source_kind": "sampled",
                "target_hash": f"target-{raw_task_id}",
                "target_hash_recovered": True,
                "task_id": task_id,
            }
        )

    pool_rel = Path("results/experiment_4243_arc_candidate_pool_grow_pool.json.gz")
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
    _write_json(
        root / "results" / "experiment_4270_arc_family_manifest.json",
        {
            "schema": "carnot.arc_family_manifest.v1",
            "spec_refs": ["REQ-VERIFY-4270"],
            "random_seed": 4270,
            "rows": manifest_rows,
            "source_paths": [str(pool_path)],
            "source_sha256": {},
            "taxonomy_paths": [],
            "survey_path": None,
            "fold_task_counts": {"0": 2, "1": 2},
            "max_feasible_fold_count": 2,
            "min_held_out_task_n": 2,
            "fallback_rows": [],
            "target_hash_unavailable_rows": [],
        },
    )
    _write_json(
        root / "results" / "experiment_4270_arc_family_provenance_recovery.json",
        {
            "acceptance_gate": True,
            "distinct_family_n": 2,
            "family_split_feasible": feasible,
            "honest_verdict": (
                "complete: arc_family_manifest_recovered_existing_pool_feasible"
                if feasible
                else "complete: arc_family_manifest_recovered_existing_pool_infeasible"
            ),
            "provenance_manifest_path": "results/experiment_4270_arc_family_manifest.json",
            "random_seed": 4270,
            "reproducibility_checksum": "sha256:" + "3" * 64,
            "verifier_is_oracle": False,
        },
    )
    model_path = root / "results" / "experiment_4244_arc_set_encoder_aggregator_model.json"
    _write_json(
        model_path,
        {
            "feature_names": list(exp4244.FEATURE_NAMES),
            "model": {"model_type": "fixture"},
            "model_specs": {"architecture": "fixture_set_encoder", "training_epochs": 0},
            "random_seed": 4244,
            "reproducibility_checksum": "sha256:" + "4" * 64,
            "set_encoder_oof": {"fold_task_ids": [], "rows": []},
            "verifier_is_oracle": False,
        },
    )
    _write_json(
        root / "results" / "experiment_4244_arc_set_encoder_aggregator_build.json",
        {
            "aggregator_trained": True,
            "honest_verdict": "complete: fixture",
            "learned_verifier_path": str(model_path),
            "model_specs": {"architecture": "fixture_set_encoder", "training_epochs": 0},
            "random_seed": 4244,
            "reproducibility_checksum": "sha256:" + "5" * 64,
            "verifier_is_oracle": False,
        },
    )
    return score_by_candidate


def _fake_training_report(
    corpus: mod.FamilyAnnotatedCorpus,
    folds: list[mod.FamilyFold],
    score_by_candidate: dict[str, float],
) -> mod.CrossFamilyTrainingReport:
    rows = []
    for fold_index, fold in enumerate(folds):
        train_task_ids = tuple(sorted(fold.train_task_ids))
        for row in corpus.rows:
            if row.task_id in fold.held_out_task_ids:
                rows.append(
                    exp4244.OOFRow(
                        task_id=row.task_id,
                        candidate_id=row.candidate_id,
                        correct=row.correct,
                        score=score_by_candidate[row.candidate_id],
                        fold=fold_index,
                        train_task_ids=train_task_ids,
                    )
                )
    return mod.CrossFamilyTrainingReport(
        rows=rows,
        fold_summaries=[
            {
                "fold": index,
                "held_out_families": sorted(fold.held_out_families),
                "train_families": sorted(fold.train_families),
                "held_out_task_n": len(fold.held_out_task_ids),
            }
            for index, fold in enumerate(folds)
        ],
        training_config={"fixture": True},
    )


def test_req_4271_spec_declares_cross_family_contract() -> None:
    """REQ-VERIFY-4271: OpenSpec declares the held-out-family transfer gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4271",
        "SCENARIO-VERIFY-4271",
        "python/carnot/reporting/arc_cross_family_transfer_existing_pool_4271.py",
        "results/experiment_4271_arc_cross_family_transfer_existing_pool.py",
        "complete_arc_cross_family_deferred_pool_infeasible",
        "cross_family_win_holds",
        "cross_family_delta",
        "cross_family_ci95",
        "within_minus_cross_gap",
        "held_out_family_n",
        "online_adapt_cross_family_delta",
        "verifier_is_oracle=false",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_4271_builds_family_disjoint_folds(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4271: no family appears in train and test for a fold."""

    _write_cross_family_fixture(tmp_path)
    corpus = mod.load_family_annotated_corpus(tmp_path)
    folds = mod.build_family_disjoint_folds(corpus)

    assert corpus.held_out_family_n == 2
    assert corpus.held_out_task_n == 4
    assert {row.family_id for row in corpus.rows} == {"family-a", "family-b"}
    assert len(folds) == 2
    for fold in folds:
        assert fold.train_families.isdisjoint(fold.held_out_families)
        assert fold.train_task_ids.isdisjoint(fold.held_out_task_ids)


def test_scenario_4271_measures_static_and_online_family_gate(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4271: static and online selectors report separate deltas."""

    scores = _write_cross_family_fixture(tmp_path)
    corpus = mod.load_family_annotated_corpus(tmp_path)
    folds = mod.build_family_disjoint_folds(corpus)
    report = _fake_training_report(corpus, folds, scores)

    metrics = mod.measure_cross_family_gate(
        corpus,
        report.rows,
        random_seed=4271,
        bootstrap_resamples=200,
    )

    assert metrics["cross_family_win_holds"] is True
    assert metrics["cross_family_delta"] == pytest.approx(0.75)
    assert metrics["within_minus_cross_gap"] == pytest.approx(mod.WITHIN_POOL_DELTA_393 - 0.75)
    assert metrics["pass_rates"]["vote_at_1"] == pytest.approx(0.25)
    assert metrics["pass_rates"]["set_encoder_at_1"] == pytest.approx(1.0)
    assert metrics["pass_rates"]["online_adapt_at_1"] == pytest.approx(0.75)
    assert metrics["online_adapt_cross_family_delta"] == pytest.approx(0.5)
    assert metrics["oracle_at_k"] == pytest.approx(1.0)
    assert metrics["matched_control_delta"] == pytest.approx(0.75)
    assert metrics["held_out_family_n"] == 2
    assert metrics["held_out_task_n"] == 4


def test_scenario_4271_run_writes_cross_family_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-4271: run emits required fields and oracle-distinct metadata."""

    scores = _write_cross_family_fixture(tmp_path)

    def fake_train(
        corpus: mod.FamilyAnnotatedCorpus,
        folds: list[mod.FamilyFold],
        **_kwargs: object,
    ) -> mod.CrossFamilyTrainingReport:
        return _fake_training_report(corpus, folds, scores)

    monkeypatch.setattr(mod, "train_cross_family_oof", fake_train)
    artifact = mod.run(tmp_path, adversarial_runner=_adversarial_clean, bootstrap_resamples=200)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: cross_family_generalizes"
    assert artifact["cross_family_win_holds"] is True
    assert artifact["cross_family_delta"] == pytest.approx(0.75)
    assert artifact["cross_family_ci95"][0] > 0.0
    assert artifact["held_out_family_n"] == 2
    assert artifact["held_out_task_n"] == 4
    assert artifact["oracle_at_k"] == pytest.approx(1.0)
    assert artifact["online_adapt_cross_family_delta"] == pytest.approx(0.5)
    assert artifact["verifier_is_oracle"] is False
    assert artifact["adversarial_verify"]["circular_moat_overclaim_clean"] is True
    assert (tmp_path / mod.OUTPUT_REL).exists()


def test_scenario_4271_infeasible_and_no_headroom_are_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-4271: infeasible precondition and no-headroom stop honestly."""

    _write_cross_family_fixture(tmp_path / "infeasible", feasible=False)
    deferred = mod.run(tmp_path / "infeasible", adversarial_runner=_adversarial_clean)
    mod.validate_artifact(deferred)
    assert deferred["honest_verdict"] == mod.DEFERRED_INFEASIBLE_VERDICT
    assert deferred["cross_family_win_holds"] is False
    assert deferred["held_out_family_n"] == 0
    assert deferred["verifier_is_oracle"] is False

    scores = _write_cross_family_fixture(tmp_path / "no-headroom", vote_all_correct=True)

    def fake_train(
        corpus: mod.FamilyAnnotatedCorpus,
        folds: list[mod.FamilyFold],
        **_kwargs: object,
    ) -> mod.CrossFamilyTrainingReport:
        return _fake_training_report(corpus, folds, scores)

    monkeypatch.setattr(mod, "train_cross_family_oof", fake_train)
    no_headroom = mod.run(
        tmp_path / "no-headroom",
        adversarial_runner=_adversarial_clean,
        bootstrap_resamples=200,
    )
    mod.validate_artifact(no_headroom)
    assert no_headroom["honest_verdict"] == "complete: no_headroom"
    assert no_headroom["honest_read"] == "no_headroom"
    assert no_headroom["oracle_at_k"] == no_headroom["pass_rates"]["vote_at_1"]
    assert no_headroom["false_negative_risk"] is True


def test_req_4271_validation_and_defensive_helpers(tmp_path: Path) -> None:
    """REQ-VERIFY-4271: schema validation, checksums, and helper edges are deterministic."""

    _write_cross_family_fixture(tmp_path)
    corpus = mod.load_family_annotated_corpus(tmp_path)
    folds = mod.build_family_disjoint_folds(corpus)
    trained = mod.train_cross_family_oof(
        corpus,
        folds,
        bootstrap_n=0,
        training_epochs=0,
        hidden_dim=4,
        lr=0.01,
    )
    assert len(trained.rows) == len(corpus.rows)
    assert trained.training_config["training_epochs"] == 0

    assert mod._round_metric(1.23456789123) == 1.2345678912
    assert mod._safe_float(True) == 0.0
    assert mod._safe_float("bad") == 0.0
    assert mod._safe_float(math.inf) == 0.0
    assert mod._safe_int(False) == 0
    assert mod._safe_int("bad") == 0
    assert mod._bootstrap_ci95([], random_seed=1, resamples=10) == [0.0, 0.0]
    assert mod._bootstrap_ci95([0.25], random_seed=1, resamples=10) == [0.25, 0.25]
    assert mod._bootstrap_ci95([0.0, 1.0], random_seed=1, resamples=0) == [0.5, 0.5]
    assert mod._ci_excludes_zero([0.1, 0.2]) is True
    assert mod._ci_excludes_zero([-0.1, 0.1]) is False
    assert mod._minmax({"a": 1.0, "b": 1.0}, "a") == pytest.approx(0.5)
    assert mod._renormalize_weights(0.0, 0.0) == pytest.approx(
        (mod.ONLINE_INITIAL_SET_WEIGHT, mod.ONLINE_INITIAL_VOTE_WEIGHT)
    )
    assert mod._clean_adversarial_report({"reports": [None, {"flags": []}]})["status"] == "clean"
    assert mod._sha256_file(tmp_path / mod.POOL_REL)

    checksum = mod.reproducibility_checksum(
        corpus=corpus,
        folds=folds,
        metrics={"cross_family_delta": 0.5},
        random_seed=4271,
    )
    assert checksum.startswith("sha256:")

    blocked = mod._deferred_artifact(
        mod.DEFERRED_INFEASIBLE_VERDICT,
        random_seed=4271,
        checksum="sha256:" + "0" * 64,
        duration_s=0.01,
    )
    invalid_cases = [
        ({key: value for key, value in blocked.items() if key != "cross_family_win_holds"}, "missing required"),
        ({**blocked, "honest_verdict": "pending"}, "terminal-prefixed"),
        ({**blocked, "cross_family_win_holds": {"value": False}}, "cross_family_win_holds"),
        ({**blocked, "cross_family_delta": True}, "cross_family_delta"),
        ({**blocked, "cross_family_ci95": [0.0]}, "cross_family_ci95"),
        ({**blocked, "within_minus_cross_gap": True}, "within_minus_cross_gap"),
        ({**blocked, "held_out_family_n": 1.2}, "held_out_family_n"),
        ({**blocked, "held_out_task_n": 1.2}, "held_out_task_n"),
        ({**blocked, "oracle_at_k": True}, "oracle_at_k"),
        ({**blocked, "online_adapt_cross_family_delta": True}, "online_adapt_cross_family_delta"),
        ({**blocked, "verifier_is_oracle": True}, "verifier_is_oracle"),
        ({**blocked, "random_seed": "4271"}, "random_seed"),
        ({**blocked, "field_principles": {}}, "field_principles"),
        ({**blocked, "spec_refs": []}, "spec_refs"),
    ]
    for payload, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)

    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        mod._read_json_object(list_json)
    with pytest.raises(mod.BlockedRun, match=mod.BLOCKED_INPUTS_VERDICT):
        mod._resolve_required_path(tmp_path, None)
    with pytest.raises(mod.BlockedRun, match=mod.BLOCKED_INPUTS_VERDICT):
        mod._resolve_required_path(tmp_path, "missing.json")

    with pytest.raises(mod.BlockedRun, match=mod.BLOCKED_INPUTS_VERDICT):
        mod._load_required_artifacts(tmp_path / "missing")

    bad_root = tmp_path / "bad-required"
    _write_cross_family_fixture(bad_root)
    build_path = bad_root / mod.SET_ENCODER_BUILD_REL
    build = json.loads(build_path.read_text(encoding="utf-8"))
    build["verifier_is_oracle"] = True
    _write_json(build_path, build)
    with pytest.raises(mod.BlockedRun, match=mod.BLOCKED_INPUTS_VERDICT):
        mod._load_required_artifacts(bad_root)

    for index, edit in enumerate(
        (
            lambda payloads: payloads["provenance"].update({"verifier_is_oracle": True}),
            lambda payloads: payloads["build"].update({"aggregator_trained": False}),
            lambda payloads: payloads["model"].update({"verifier_is_oracle": True}),
        )
    ):
        case_root = tmp_path / f"bad-artifacts-{index}"
        _write_cross_family_fixture(case_root)
        paths = {
            "provenance": case_root / mod.PROVENANCE_REL,
            "build": case_root / mod.SET_ENCODER_BUILD_REL,
            "model": case_root / mod.SET_ENCODER_MODEL_REL,
        }
        payloads = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items()}
        edit(payloads)
        for key, path in paths.items():
            _write_json(path, payloads[key])
        with pytest.raises(mod.BlockedRun, match=mod.BLOCKED_INPUTS_VERDICT):
            mod._load_required_artifacts(case_root)


def test_req_4271_family_manifest_and_measurement_edge_cases(tmp_path: Path) -> None:
    """REQ-VERIFY-4271: malformed family manifests and non-winning reads are explicit."""

    conflict_root = tmp_path / "conflict"
    _write_cross_family_fixture(conflict_root)
    manifest_path = conflict_root / mod.MANIFEST_REL
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rows"][2]["family_id"] = "family-a"
    _write_json(manifest_path, manifest)
    with pytest.raises(mod.BlockedRun, match=mod.BLOCKED_INPUTS_VERDICT):
        mod.load_family_annotated_corpus(conflict_root)

    missing_root = tmp_path / "missing-task"
    _write_cross_family_fixture(missing_root)
    manifest_path = missing_root / mod.MANIFEST_REL
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rows"] = manifest["rows"][:-1]
    _write_json(manifest_path, manifest)
    with pytest.raises(mod.BlockedRun, match=mod.BLOCKED_INPUTS_VERDICT):
        mod.load_family_annotated_corpus(missing_root)

    extra_root = tmp_path / "extra-task"
    _write_cross_family_fixture(extra_root)
    manifest_path = extra_root / mod.MANIFEST_REL
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    extra = dict(manifest["rows"][0])
    extra["task_id"] = "fixture:extra-task"
    extra["raw_task_id"] = "extra-task"
    extra["family_id"] = "family-extra"
    manifest["rows"].append(extra)
    _write_json(manifest_path, manifest)
    with pytest.raises(mod.BlockedRun, match=mod.BLOCKED_INPUTS_VERDICT):
        mod.load_family_annotated_corpus(extra_root)

    one_fold_root = tmp_path / "one-fold"
    _write_cross_family_fixture(one_fold_root)
    manifest_path = one_fold_root / mod.MANIFEST_REL
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest["rows"]:
        row["fold"] = 0
    _write_json(manifest_path, manifest)
    one_fold_corpus = mod.load_family_annotated_corpus(one_fold_root)
    with pytest.raises(mod.BlockedRun, match=mod.BLOCKED_INPUTS_VERDICT):
        mod.build_family_disjoint_folds(one_fold_corpus)

    scores = _write_cross_family_fixture(tmp_path / "measure")
    corpus = mod.load_family_annotated_corpus(tmp_path / "measure")
    folds = mod.build_family_disjoint_folds(corpus)
    vote_scores = {row.candidate_id: row.vote_weight for row in corpus.rows}
    within_metrics = mod.measure_cross_family_gate(
        corpus,
        _fake_training_report(corpus, folds, vote_scores).rows,
        random_seed=4271,
        bootstrap_resamples=100,
    )
    assert within_metrics["honest_read"] == "within_pool_only"

    dropped = mod.measure_cross_family_gate(
        corpus,
        _fake_training_report(corpus, folds, scores).rows[:-1],
        random_seed=4271,
        bootstrap_resamples=10,
    )
    assert dropped["dropped_task_n"] == 1

    malformed_gate_root = tmp_path / "malformed-gate"
    _write_cross_family_fixture(malformed_gate_root)
    provenance = json.loads((malformed_gate_root / mod.PROVENANCE_REL).read_text(encoding="utf-8"))
    provenance["family_split_feasible"] = "yes"
    _write_json(malformed_gate_root / mod.PROVENANCE_REL, provenance)
    blocked = mod.run(malformed_gate_root, adversarial_runner=_adversarial_clean)
    assert blocked["honest_verdict"] == mod.BLOCKED_INPUTS_VERDICT
