"""Tests for Exp 4273 ARC cross-family online adaptation.

Spec refs: REQ-VERIFY-4273, SCENARIO-VERIFY-4273.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import arc_cross_family_online_adaptation_4273 as mod
from carnot.reporting import arc_cross_family_transfer_existing_pool_4271 as exp4271


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _adversarial_clean(_path: Path) -> dict[str, Any]:
    return {"returncode": 0, "reports": [{"flag_count": 0, "flags": [], "max_severity": 0}]}


def _row(
    task_id: str,
    family_id: str,
    candidate_index: int,
    *,
    correct: bool,
    vote_weight: float,
    good_feature: float,
) -> exp4271.FamilyAnnotatedRow:
    return exp4271.FamilyAnnotatedRow(
        task_id=task_id,
        family_id=family_id,
        fold=0,
        candidate_id=f"{task_id}::candidate{candidate_index}",
        candidate_index=candidate_index,
        correct=correct,
        features={"good_feature": good_feature},
        vote_weight=vote_weight,
    )


def _online_fixture(tmp_path: Path, n_families: int = 8) -> tuple[
    exp4271.FamilyAnnotatedCorpus,
    list[dict[str, Any]],
]:
    rows: list[exp4271.FamilyAnnotatedRow] = []
    static_rows: list[dict[str, Any]] = []
    task_family_ids: dict[str, str] = {}
    task_folds: dict[str, int] = {}
    for index in range(n_families):
        family_id = f"family-{index:02d}"
        task_id = f"task-{index:02d}"
        wrong = _row(
            task_id,
            family_id,
            0,
            correct=False,
            vote_weight=0.9,
            good_feature=0.1,
        )
        correct = _row(
            task_id,
            family_id,
            1,
            correct=True,
            vote_weight=0.1,
            good_feature=0.9,
        )
        rows.extend([wrong, correct])
        task_family_ids[task_id] = family_id
        task_folds[task_id] = index % 2
        static_rows.append(
            {
                "task_id": task_id,
                "family_id": family_id,
                "fold": index % 2,
                "vote_candidate_id": wrong.candidate_id,
                "vote_correct": False,
                "set_encoder_candidate_id": wrong.candidate_id,
                "set_encoder_correct": False,
            }
        )
    corpus = exp4271.FamilyAnnotatedCorpus(
        rows=rows,
        task_family_ids=task_family_ids,
        task_folds=task_folds,
        manifest_path=tmp_path / "manifest.json",
        manifest_sha256="manifest-sha",
        pool_artifact_path=tmp_path / "pool.json.gz",
        pool_artifact_sha256="pool-sha",
        upstream_checksum="upstream-sha",
        held_out_family_n=n_families,
        held_out_task_n=n_families,
        candidate_n=len(rows),
    )
    return corpus, static_rows


def test_req_4273_spec_declares_online_adaptation_contract() -> None:
    """REQ-VERIFY-4273: OpenSpec declares the self-learning artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4273",
        "SCENARIO-VERIFY-4273",
        "python/carnot/reporting/arc_cross_family_online_adaptation_4273.py",
        "results/experiment_4273_arc_cross_family_online_adaptation.py",
        "complete_self_learning_deferred_to_fresh_pool",
        "online_adaptation_helps",
        "static_cross_family_delta",
        "online_cross_family_delta",
        "adaptation_curve",
        "verifier_is_oracle=false",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_4273_online_precision_reweighting_beats_static(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4273: prior-family feature precision can improve online selection."""

    corpus, static_rows = _online_fixture(tmp_path)
    metrics = mod.measure_online_adaptation(
        corpus,
        static_rows,
        feature_names=["good_feature"],
        random_seed=4273,
        bootstrap_resamples=400,
    )

    assert metrics["static_cross_family_delta"] == pytest.approx(0.0)
    assert metrics["online_cross_family_delta"] == pytest.approx(7 / 8)
    assert metrics["online_minus_static_delta"] == pytest.approx(7 / 8)
    assert metrics["online_minus_static_ci95"][0] > 0.0
    assert metrics["online_adaptation_helps"] is True
    assert metrics["adaptation_curve"][0]["online_minus_static_gain"] == pytest.approx(0.0)
    assert metrics["adaptation_curve"][-1]["cumulative_online_minus_static_gain"] > 0.0
    assert metrics["adaptation_curve"][1]["nearest_seen_family"] == "family-00"
    assert metrics["tier1_counter_update"] == mod.TIER1_COUNTER_UPDATE


def test_scenario_4273_run_writes_required_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-4273: run emits bare verdict fields and clean adversarial metadata."""

    corpus, static_rows = _online_fixture(tmp_path)
    inputs = mod.ExperimentInputs(
        corpus=corpus,
        prior_cross_family_artifact={
            "cross_family_delta": 0.0,
            "task_rows": static_rows,
            "reproducibility_checksum": "sha256:" + "1" * 64,
        },
        build_artifact={
            "model_specs": {"architecture": "fixture_set_encoder"},
            "reproducibility_checksum": "sha256:" + "2" * 64,
        },
        model_artifact={
            "feature_names": ["good_feature"],
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "3" * 64,
        },
        provenance_artifact={
            "family_split_feasible": True,
            "reproducibility_checksum": "sha256:" + "4" * 64,
        },
    )
    monkeypatch.setattr(mod, "load_inputs", lambda _root: inputs)

    artifact = mod.run(tmp_path, adversarial_runner=_adversarial_clean, bootstrap_resamples=400)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: online_adaptation_improves_generalization"
    assert artifact["online_adaptation_helps"] is True
    assert artifact["static_cross_family_delta"] == pytest.approx(0.0)
    assert artifact["online_cross_family_delta"] == pytest.approx(7 / 8)
    assert artifact["verifier_is_oracle"] is False
    assert artifact["adversarial_verify"]["circular_moat_overclaim_clean"] is True
    assert (tmp_path / mod.OUTPUT_REL).exists()


def test_scenario_4273_deferred_and_validation_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-4273: infeasible A1 and malformed artifacts fail closed."""

    assert mod._safe_float(True) == 0.0
    assert mod._safe_float("bad") == 0.0
    assert mod._bootstrap_ci95([], random_seed=1, resamples=10) == [0.0, 0.0]
    assert mod._bootstrap_ci95([0.25], random_seed=1, resamples=10) == [0.25, 0.25]
    assert mod._bootstrap_ci95([0.0, 1.0], random_seed=1, resamples=0) == [0.5, 0.5]

    _write_json(
        tmp_path / mod.PROVENANCE_REL,
        {
            "family_split_feasible": False,
            "honest_verdict": "complete: fixture infeasible",
            "random_seed": 4270,
            "reproducibility_checksum": "sha256:" + "0" * 64,
            "verifier_is_oracle": False,
        },
    )
    deferred = mod.run(tmp_path, adversarial_runner=_adversarial_clean)

    mod.validate_artifact(deferred)
    assert deferred["honest_verdict"] == mod.DEFERRED_TO_FRESH_POOL_VERDICT
    assert deferred["online_adaptation_helps"] is False
    assert deferred["static_cross_family_delta"] == pytest.approx(0.0)
    assert deferred["online_cross_family_delta"] == pytest.approx(0.0)
    assert deferred["adaptation_curve"] == []
    assert deferred["verifier_is_oracle"] is False

    def blocked_inputs(_root: Path) -> mod.ExperimentInputs:
        raise mod.BlockedRun(mod.BLOCKED_INPUTS_VERDICT)

    original_loader = mod.load_inputs
    mod.load_inputs = blocked_inputs
    try:
        blocked = mod.run(tmp_path / "blocked", adversarial_runner=_adversarial_clean)
    finally:
        mod.load_inputs = original_loader
    mod.validate_artifact(blocked)
    assert blocked["honest_verdict"] == mod.BLOCKED_INPUTS_VERDICT
    assert blocked["acceptance_gate"] is False

    invalid_cases = [
        ({key: value for key, value in deferred.items() if key != "online_adaptation_helps"}, "missing"),
        ({**deferred, "honest_verdict": "pending"}, "terminal-prefixed"),
        ({**deferred, "online_adaptation_helps": {"value": False}}, "online_adaptation_helps"),
        ({**deferred, "static_cross_family_delta": True}, "static_cross_family_delta"),
        ({**deferred, "online_cross_family_delta": True}, "online_cross_family_delta"),
        ({**deferred, "online_minus_static_ci95": [0.0]}, "online_minus_static_ci95"),
        ({**deferred, "adaptation_curve": {}}, "adaptation_curve"),
        ({**deferred, "verifier_is_oracle": True}, "verifier_is_oracle"),
        ({**deferred, "random_seed": "4273"}, "random_seed"),
        ({**deferred, "field_principles": {}}, "field_principles"),
        ({**deferred, "spec_refs": []}, "spec_refs"),
    ]
    for payload, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)
