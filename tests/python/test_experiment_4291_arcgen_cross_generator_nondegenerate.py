"""Tests for Exp 4291 ARC-GEN cross-generator non-degenerate replication.

Spec refs: REQ-VERIFY-4291, SCENARIO-VERIFY-4291.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import arc_cross_family_transfer_existing_pool_4271 as exp4271
from carnot.reporting import arc_set_encoder_aggregator_4244 as exp4244
from carnot.reporting import arcgen_cross_generator_nondegenerate_4291 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _adversarial_clean(_path: Path) -> dict[str, Any]:
    return {"returncode": 0, "reports": [{"flag_count": 0, "flags": [], "max_severity": 0}]}


def _fake_catalog(generator_n: int = 6) -> dict[str, tuple[Any, Any]]:
    catalog: dict[str, tuple[Any, Any]] = {}
    for index in range(generator_n):

        def generate(offset: int = index) -> dict[str, list[list[int]]]:
            base = random.randint(1, 7)
            return {
                "input": [[base, 0], [0, offset % 9]],
                "output": [[base, (offset + 1) % 9], [(base + offset) % 9, base]],
            }

        catalog[f"task{index:02d}"] = (generate, lambda: {"train": [], "test": []})
    return catalog


def _write_preconditions(root: Path) -> None:
    model_path = root / mod.SET_ENCODER_MODEL_REL
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
        root / mod.SET_ENCODER_BUILD_REL,
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
    _write_json(
        root / mod.EXISTING_CROSS_GENERATOR_REL,
        {
            "honest_verdict": "complete: cross_family_generalizes",
            "cross_family_win_holds": True,
            "cross_family_delta": 0.4038461538,
            "cross_family_ci95": [0.25, 0.55],
            "held_out_family_n": 52,
            "held_out_task_n": 52,
            "oracle_at_k": 0.8269230769,
            "verifier_is_oracle": False,
        },
    )
    _write_json(
        root / mod.ARC_TGI_REL,
        {
            "honest_verdict": "blocked_gate_check_failed",
            "status": "blocked",
            "gate_check_summary": "fixture blocked ARC-TGI read",
        },
    )
    (root / mod.ARCGEN_REL).mkdir(parents=True, exist_ok=True)


def _fake_training_report(
    corpus: exp4271.FamilyAnnotatedCorpus,
    folds: list[exp4271.FamilyFold],
) -> exp4271.CrossFamilyTrainingReport:
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
                        score=row.features["cell_confidence_mean"],
                        fold=fold_index,
                        train_task_ids=train_task_ids,
                    )
                )
    return exp4271.CrossFamilyTrainingReport(
        rows=rows,
        fold_summaries=[
            {
                "fold": index,
                "held_out_generators": sorted(fold.held_out_families),
                "train_generators": sorted(fold.train_families),
                "held_out_task_n": len(fold.held_out_task_ids),
            }
            for index, fold in enumerate(folds)
        ],
        training_config={"fixture": True},
    )


def test_req_4291_spec_declares_cross_generator_contract() -> None:
    """REQ-VERIFY-4291: OpenSpec declares the non-degenerate ARC-GEN gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4291",
        "SCENARIO-VERIFY-4291",
        "python/carnot/reporting/arcgen_cross_generator_nondegenerate_4291.py",
        "results/experiment_4291_arcgen_cross_generator_nondegenerate.py",
        "blocked_arcgen_unavailable",
        "cross_generator_holds",
        "cross_generator_delta",
        "vote_at_1 > 0.05",
        "oracle_at_k < 1.0",
        "cross_generator_delta < 0.95",
        "held_out_generator_n",
        "verifier_is_oracle=false",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_4291_builds_nondegenerate_arcgen_pool(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4291: pool rows carry generators and non-degenerate headroom."""

    _write_preconditions(tmp_path)
    built = mod.build_arcgen_pool(
        tmp_path,
        task_catalog=_fake_catalog(6),
        generator_limit=6,
        tasks_per_generator=2,
        candidates_per_task=8,
    )

    assert built.corpus.held_out_family_n == 6
    assert built.corpus.held_out_task_n == 12
    assert built.pool_path.exists()
    assert built.manifest_path.exists()
    assert built.pool_diagnostics["candidate_count_per_task"] == 8
    assert built.pool_diagnostics["vote_can_win_n"] > 0
    assert built.pool_diagnostics["wrong_majority_n"] > 0
    assert built.pool_diagnostics["oracle_missing_n"] > 0
    assert {row["source_kind"] for row in built.manifest_rows} == {"arcgen"}
    assert all(row["family_id"].startswith("arcgen_native_task:") for row in built.manifest_rows)
    assert all(row["generator_id"].startswith("arcgen_generator:") for row in built.manifest_rows)
    assert all(str(row["target_hash"]).startswith("sha256:") for row in built.manifest_rows)
    folds = exp4271.build_family_disjoint_folds(built.corpus)
    assert len(folds) >= 2
    assert all(fold.train_families.isdisjoint(fold.held_out_families) for fold in folds)


def test_scenario_4291_gate_passes_only_when_nondegenerate(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4291: cross-generator holds requires the guard fields."""

    _write_preconditions(tmp_path)
    built = mod.build_arcgen_pool(
        tmp_path,
        task_catalog=_fake_catalog(6),
        generator_limit=6,
        tasks_per_generator=2,
        candidates_per_task=8,
    )
    folds = exp4271.build_family_disjoint_folds(built.corpus)
    report = _fake_training_report(built.corpus, folds)
    metrics = mod.measure_cross_generator_gate(
        built.corpus,
        report.rows,
        random_seed=4291,
        bootstrap_resamples=200,
    )

    assert metrics["cross_generator_holds"] is True
    assert metrics["cross_generator_delta"] == pytest.approx(0.5)
    assert metrics["cross_generator_ci95"][0] > 0.0
    assert metrics["vote_at_1"] == pytest.approx(0.25)
    assert metrics["oracle_at_k"] == pytest.approx(0.75)
    assert metrics["non_degenerate_guards_pass"] is True
    assert metrics["held_out_generator_n"] == 6
    assert mod.cross_generator_holds_from_metrics(
        {
            "cross_generator_delta": 1.0,
            "cross_generator_ci95": [1.0, 1.0],
            "vote_at_1": 0.0,
            "oracle_at_k": 1.0,
        }
    ) is False


def test_scenario_4291_run_writes_artifact_and_blocked_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-4291: run emits required fields or an honest blocked verdict."""

    _write_preconditions(tmp_path)

    def fake_train(
        corpus: exp4271.FamilyAnnotatedCorpus,
        folds: list[exp4271.FamilyFold],
        **_kwargs: object,
    ) -> exp4271.CrossFamilyTrainingReport:
        return _fake_training_report(corpus, folds)

    monkeypatch.setattr(mod, "load_arcgen_task_catalog", lambda _path: _fake_catalog(6))
    monkeypatch.setattr(mod, "train_arcgen_generator_oof", fake_train)
    artifact = mod.run(
        tmp_path,
        adversarial_runner=_adversarial_clean,
        bootstrap_resamples=200,
        generator_limit=6,
        tasks_per_generator=2,
        candidates_per_task=8,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: arcgen_cross_generator_generalizes"
    assert artifact["cross_generator_holds"] is True
    assert artifact["vote_at_1"] > 0.05
    assert artifact["oracle_at_k"] < 1.0
    assert artifact["cross_generator_delta"] < 0.95
    assert artifact["per_substrate_delta"]["arcgen"]["held_out_task_n"] == 12
    assert artifact["per_substrate_delta"]["original_arc"]["cross_family_delta"] == pytest.approx(0.4038461538)
    assert artifact["model_specs"]["non_degenerate_pool_construction"]["candidates_per_task"] == 8
    assert artifact["adversarial_verify"]["degenerate_separation_clean"] is True
    assert artifact["verifier_is_oracle"] is False
    assert (tmp_path / mod.OUTPUT_REL).exists()

    blocked = mod.run(
        tmp_path / "missing",
        adversarial_runner=_adversarial_clean,
        bootstrap_resamples=10,
    )
    mod.validate_artifact(blocked)
    assert blocked["honest_verdict"] == mod.BLOCKED_ARCGEN_VERDICT
    assert blocked["cross_generator_holds"] is False
    assert blocked["verifier_is_oracle"] is False


def test_req_4291_validation_and_checksum_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-4291: bare-field validation and checksums are deterministic."""

    _write_preconditions(tmp_path)
    built = mod.build_arcgen_pool(
        tmp_path,
        task_catalog=_fake_catalog(5),
        generator_limit=5,
        tasks_per_generator=2,
        candidates_per_task=8,
    )
    folds = exp4271.build_family_disjoint_folds(built.corpus)
    checksum = mod.reproducibility_checksum(
        pool_sha256=built.pool_sha256,
        manifest_sha256=built.manifest_sha256,
        metrics={
            "cross_generator_delta": 0.5,
            "vote_at_1": 0.25,
            "oracle_at_k": 0.75,
        },
        random_seed=4291,
    )
    assert checksum.startswith("sha256:")
    assert mod._ci_excludes_zero([0.1, 0.2]) is True
    assert mod._ci_excludes_zero([-0.1, 0.2]) is False
    assert all(fold.train_families.isdisjoint(fold.held_out_families) for fold in folds)

    artifact = mod._blocked_artifact(
        mod.BLOCKED_ARCGEN_VERDICT,
        random_seed=4291,
        duration_s=0.01,
    )
    mod.validate_artifact(artifact)
    invalid_cases = [
        ({key: value for key, value in artifact.items() if key != "cross_generator_holds"}, "missing required"),
        ({**artifact, "honest_verdict": "pending"}, "terminal-prefixed"),
        ({**artifact, "cross_generator_holds": 0}, "cross_generator_holds"),
        ({**artifact, "cross_generator_delta": True}, "cross_generator_delta"),
        ({**artifact, "cross_generator_ci95": [0.0]}, "cross_generator_ci95"),
        ({**artifact, "vote_at_1": True}, "vote_at_1"),
        ({**artifact, "held_out_generator_n": 1.2}, "held_out_generator_n"),
        ({**artifact, "held_out_task_n": 1.2}, "held_out_task_n"),
        ({**artifact, "oracle_at_k": True}, "oracle_at_k"),
        ({**artifact, "matched_control_delta": True}, "matched_control_delta"),
        ({**artifact, "verifier_is_oracle": True}, "verifier_is_oracle"),
        ({**artifact, "random_seed": "4291"}, "random_seed"),
        ({**artifact, "field_principles": {}}, "field_principles"),
        ({**artifact, "spec_refs": []}, "spec_refs"),
    ]
    for payload, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)
