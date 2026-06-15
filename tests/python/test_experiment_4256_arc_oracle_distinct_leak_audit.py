"""Tests for Exp 4256 ARC provenance leak audit.

Spec refs: REQ-VERIFY-4256, SCENARIO-VERIFY-4256.
"""

from __future__ import annotations

import gzip
import inspect
import json
from pathlib import Path

import pytest

from carnot.reporting import arc_oracle_distinct_leak_audit_4256 as mod
from carnot.reporting import arc_set_encoder_aggregator_4244 as exp4244


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _adversarial_clean(_path: Path) -> dict:
    return {"returncode": 0, "reports": [{"flag_count": 0, "flags": [], "max_severity": 0}]}


def _feature(*, vote: float, content: float, induced: bool) -> dict[str, float]:
    values = {name: 0.0 for name in mod.FULL_FEATURE_NAMES}
    values.update(
        {
            "vote_weight": vote,
            "self_consistency_margin": vote - 0.5,
            "vote_weight_rank_fraction": vote,
            "cell_confidence_mean": content,
            "cell_confidence_margin": content - 0.5,
            "cell_confidence_rank_fraction": content,
            "set_vote_mean": 0.5,
            "set_vote_max": vote,
            "set_vote_std": abs(vote - 0.5),
            "set_confidence_mean": content,
            "set_confidence_max": content,
            "set_confidence_std": abs(content - 0.5),
            "vote_weight_zscore": vote - 0.5,
            "cell_confidence_zscore": content - 0.5,
            "modal_cell_agreement_frac": content,
            "grid_height": 9.0 if induced else 2.0,
            "grid_width": 9.0 if induced else 2.0,
            "grid_cells": 81.0 if induced else 4.0,
            "grid_color_count": 8.0 if induced else 2.0,
            "program_demo_fit": 1.0 if induced else 0.0,
            "grid_duplicate_count": 1.0 if induced else 4.0,
            "shape_family_frac": 1.0 if induced else 0.0,
            "palette_family_frac": 1.0 if induced else 0.0,
        }
    )
    return values


def _write_audit_fixture(root: Path, *, include_source_kinds: bool = True) -> Path:
    task_specs = [
        ("mini:task-0", 1, [0.9, 0.1, 0.0], [0.1, 0.95, 0.2], [False, True, False]),
        ("mini:task-1", 0, [0.9, 0.1, 0.0], [0.95, 0.2, 0.1], [False, True, False]),
        ("mini:task-2", 1, [0.8, 0.2, 0.0], [0.2, 0.9, 0.1], [False, True, False]),
        ("mini:task-3", 1, [0.7, 0.2, 0.1], [0.1, 0.9, 0.2], [False, True, False]),
    ]
    tasks = []
    for task_id, correct_index, votes, contents, induced_flags in task_specs:
        candidates = []
        for candidate_index, vote in enumerate(votes):
            induced = induced_flags[candidate_index]
            is_correct = candidate_index == correct_index
            candidate = {
                "candidate_grid_hash": f"hash-{task_id}-{candidate_index}",
                "candidate_id": f"{task_id}::candidate{candidate_index}",
                "candidate_index": candidate_index,
                "features": _feature(vote=vote, content=contents[candidate_index], induced=induced),
                "grid": [[candidate_index]],
                "is_correct": is_correct,
                "q_mean": contents[candidate_index],
                "raw_candidate_indices": [candidate_index],
                "votes": vote,
            }
            if include_source_kinds:
                if induced:
                    candidate["source_kinds"] = ["induced_pred_grid"]
                elif is_correct:
                    candidate["source_kinds"] = ["gold_flag"]
                else:
                    candidate["source_kinds"] = ["pool_candidate"]
            candidates.append(candidate)
        vote_top = max(candidates, key=lambda item: (item["features"]["vote_weight"], -item["candidate_index"]))
        tasks.append(
            {
                "candidate_count": len(candidates),
                "candidates": candidates,
                "oracle_present": True,
                "raw_task_id": task_id.rsplit(":", 1)[-1],
                "source_id": "mini",
                "task_id": task_id,
                "vote_top_candidate_id": vote_top["candidate_id"],
                "wrong_majority": not vote_top["is_correct"],
            }
        )

    pool_rel = Path("results/experiment_4243_arc_candidate_pool_grow_pool.json.gz")
    pool_path = root / pool_rel
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(pool_path, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                "candidate_n": sum(len(task["candidates"]) for task in tasks),
                "positive_candidate_n": sum(
                    1 for task in tasks for candidate in task["candidates"] if candidate["is_correct"]
                ),
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
            "positive_candidate_n": 4,
            "random_seed": 4243,
            "reproducibility_checksum": "sha256:" + "2" * 64,
            "verifier_is_oracle": False,
            "wrong_majority_n": 2,
        },
    )
    model_path = root / "results" / "experiment_4244_arc_set_encoder_aggregator_model.json"
    fold_task_ids = [["mini:task-0", "mini:task-1"], ["mini:task-2", "mini:task-3"]]
    _write_json(
        model_path,
        {
            "feature_names": list(mod.FULL_FEATURE_NAMES),
            "model_specs": {"architecture": "fixture"},
            "random_seed": 4244,
            "set_encoder_oof": {"fold_task_ids": fold_task_ids, "rows": []},
            "verifier_is_oracle": False,
        },
    )
    _write_json(
        root / "results" / "experiment_4244_arc_set_encoder_aggregator_build.json",
        {
            "aggregator_trained": True,
            "learned_verifier_path": str(model_path),
            "set_encoder_minus_vote_delta": 0.4423076923,
            "set_encoder_minus_vote_ci95": [0.3076923077, 0.5961538462],
            "verifier_is_oracle": False,
        },
    )
    _write_json(
        root / "results" / "experiment_4245_arc_set_encoder_beats_vote.json",
        {
            "honest_verdict": "complete: arc_oracle_distinct_set_encoder_beats_vote",
            "set_encoder_minus_vote_delta": 0.4423076923,
            "set_encoder_minus_vote_ci95": [0.3076923077, 0.5961538462],
            "verifier_is_oracle": False,
        },
    )
    return root


def test_req_4256_spec_declares_leak_audit_contract() -> None:
    """REQ-VERIFY-4256: OpenSpec declares the provenance leak audit fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4256",
        "SCENARIO-VERIFY-4256",
        "python/carnot/reporting/arc_oracle_distinct_leak_audit_4256.py",
        "results/experiment_4256_arc_oracle_distinct_leak_audit.py",
        "blocked_arc_provenance_unrecoverable",
        "origin_probe_auroc",
        "origin_correctness_corr",
        "provenance_blind_delta",
        "provenance_blind_ci95",
        "win_survives_provenance_blind",
        "verifier_is_oracle=false",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_4256_origin_probe_reports_leak_signature(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4256: origin is predictable and correlated with correctness."""

    root = _write_audit_fixture(tmp_path)
    corpus = mod.load_audit_corpus(root)
    folds, source = mod.load_reference_folds(root, corpus)
    report = mod.origin_probe_report(corpus, folds, random_seed=4256)

    assert source == "exp4244_set_encoder_oof.fold_task_ids"
    assert report["origin_probe_auroc"] > 0.9
    assert report["origin_correctness_corr"] > 0.25
    assert report["induced_origin_positive_fraction"] == pytest.approx(0.75)
    assert "grid_height" in report["origin_probe_high_weight_features"]


def test_scenario_4256_blind_gate_uses_scores_not_provenance(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4256: blind scores re-run the beats-vote gate."""

    corpus = mod.load_audit_corpus(_write_audit_fixture(tmp_path))
    scores = {
        row.candidate_id: row.features["cell_confidence_mean"]
        for row in corpus.rows
    }

    metrics = mod.measure_provenance_blind_gate(
        corpus,
        scores,
        random_seed=4256,
        bootstrap_resamples=2000,
    )

    assert metrics["provenance_blind_delta"] == pytest.approx(0.75)
    assert metrics["provenance_blind_ci95"][0] >= 0.0
    assert metrics["win_survives_provenance_blind"] is True
    assert metrics["provenance_blind_pass_rates"]["vote_at_1"] == pytest.approx(0.25)
    assert metrics["provenance_blind_pass_rates"]["set_encoder_at_1"] == pytest.approx(1.0)


def test_scenario_4256_run_writes_complete_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-4256: run emits required bare fields and clean adversarial status."""

    root = _write_audit_fixture(tmp_path)

    def fake_train(corpus: mod.AuditCorpus, *_args: object, **_kwargs: object) -> mod.BlindTrainingReport:
        return mod.BlindTrainingReport(
            auroc=0.75,
            scores={row.candidate_id: row.features["cell_confidence_mean"] for row in corpus.rows},
        )

    monkeypatch.setattr(mod, "_train_blind_set_encoder_oof", fake_train)
    artifact = mod.run(root, adversarial_runner=_adversarial_clean)

    mod.validate_artifact(artifact)
    assert artifact["experiment"] == "experiment_4256_arc_oracle_distinct_leak_audit"
    assert artifact["honest_verdict"] == "complete: arc_set_encoder_win_survives_provenance_blind_audit"
    assert artifact["win_survives_provenance_blind"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["origin_probe_auroc"] > 0.9
    assert artifact["provenance_blind_delta"] == pytest.approx(0.75)
    assert artifact["bootstrap_resamples"] == 2000
    assert artifact["adversarial_verify"]["circular_moat_overclaim_clean"] is True
    specs = artifact["model_specs"]
    assert "vote_weight" in specs["provenance_blind_feature_partition"]["retained_features"]
    assert "grid_duplicate_count" in specs["provenance_blind_feature_partition"]["stripped_features"]
    assert "shape_family_frac" in specs["provenance_blind_feature_partition"]["stripped_features"]
    assert (root / mod.OUTPUT_REL).exists()


def test_scenario_4256_missing_provenance_blocks_honestly(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4256: unrecoverable provenance stops the headline."""

    artifact = mod.run(_write_audit_fixture(tmp_path, include_source_kinds=False), adversarial_runner=_adversarial_clean)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.BLOCKED_PROVENANCE_VERDICT
    assert artifact["win_survives_provenance_blind"] is False
    assert artifact["model_specs"]["status"] == "blocked"
    assert artifact["acceptance_gate"] is False


def test_training_wrapper_and_validation_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-4256: training, fallback folds, validation, and clean flags are deterministic."""

    corpus = mod.load_audit_corpus(_write_audit_fixture(tmp_path))
    original_features = exp4244.FEATURE_NAMES
    fallback_folds, fallback_source = mod.load_reference_folds(tmp_path / "missing", corpus)
    assert fallback_source == "reconstructed_exp4244_split"
    assert fallback_folds

    report = mod._train_blind_set_encoder_oof(
        corpus,
        fallback_folds,
        feature_names=("vote_weight", "cell_confidence_mean"),
        random_seed=4256,
        training_epochs=0,
        hidden_dim=4,
        lr=0.01,
    )
    assert set(report.scores) == {row.candidate_id for row in corpus.rows}
    assert exp4244.FEATURE_NAMES == original_features

    partition = mod.provenance_blind_feature_partition({"origin_probe_high_weight_features": ["set_vote_max"]})
    assert "set_vote_max" in partition["stripped_features"]
    assert partition["strip_reasons"]["set_vote_max"] == "origin_probe_high_weight"
    assert "program_demo_fit" in partition["stripped_features"]

    blocked = mod._blocked_artifact(
        mod.BLOCKED_PROVENANCE_VERDICT,
        random_seed=4256,
        checksum="sha256:" + "0" * 64,
        duration_s=0.01,
    )
    invalid_cases = [
        ({key: value for key, value in blocked.items() if key != "origin_probe_auroc"}, "missing required"),
        ({**blocked, "honest_verdict": "pending"}, "terminal-prefixed"),
        ({**blocked, "win_survives_provenance_blind": {"value": False}}, "bare bool"),
        ({**blocked, "origin_probe_auroc": True}, "bare float"),
        ({**blocked, "provenance_blind_ci95": [0.0]}, "ci95"),
        ({**blocked, "verifier_is_oracle": True}, "verifier_is_oracle"),
        ({**blocked, "random_seed": 4256.0}, "bare int"),
        ({**blocked, "field_principles": {}}, "field_principles"),
        ({**blocked, "spec_refs": []}, "spec_refs"),
    ]
    for payload, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)

    flagged = mod._clean_adversarial_report(
        {"returncode": 0, "reports": [{"flags": [{"kind": "CIRCULAR_MOAT_OVERCLAIM"}]}]}
    )
    assert flagged["status"] == "flagged"
    assert flagged["circular_moat_overclaim_clean"] is False
    assert mod._round_metric(1.23456789123) == 1.2345678912


def test_module_does_not_execute_oracle_or_modify_conductor() -> None:
    """REQ-VERIFY-4256: audit stays oracle-distinct and leaves conductor untouched."""

    source = inspect.getsource(mod)
    assert "scripts/research_conductor.py" not in source
    assert "Gap4ExecutionVerifier" not in source
    assert "extract_dsl_rules" not in source
    assert "apply_rule" not in source
    assert "key=lambda candidate: (candidate.correct" not in source
