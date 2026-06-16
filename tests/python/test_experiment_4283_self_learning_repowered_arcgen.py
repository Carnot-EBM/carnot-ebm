"""Tests for Exp 4283 repowered ARC self-learning adaptation.

Spec refs: REQ-VERIFY-4283, SCENARIO-VERIFY-4283.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import arc_cross_family_transfer_existing_pool_4271 as exp4271
from carnot.reporting import self_learning_repowered_arcgen_4283 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


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


def _repowered_fixture(
    tmp_path: Path,
    n_families: int = 60,
) -> tuple[exp4271.FamilyAnnotatedCorpus, list[dict[str, Any]]]:
    rows: list[exp4271.FamilyAnnotatedRow] = []
    static_rows: list[dict[str, Any]] = []
    task_family_ids: dict[str, str] = {}
    task_folds: dict[str, int] = {}
    for index in range(n_families):
        substrate = "arcgen_native_task" if index >= mod.V395_FAMILY_N else "original_arc_task"
        family_id = f"{substrate}:family-{index:02d}"
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
        task_folds[task_id] = index % 5
        static_rows.append(
            {
                "task_id": task_id,
                "family_id": family_id,
                "fold": index % 5,
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
        manifest_path=tmp_path / "combined_manifest.json",
        manifest_sha256="manifest-sha",
        pool_artifact_path=tmp_path / "combined_pool.json.gz",
        pool_artifact_sha256="pool-sha",
        upstream_checksum="upstream-sha",
        held_out_family_n=n_families,
        held_out_task_n=n_families,
        candidate_n=len(rows),
    )
    return corpus, static_rows


def test_req_4283_spec_declares_repowered_self_learning_contract() -> None:
    """REQ-VERIFY-4283: OpenSpec declares the Tier-1/Tier-2 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4283",
        "SCENARIO-VERIFY-4283",
        "python/carnot/reporting/self_learning_repowered_arcgen_4283.py",
        "results/experiment_4283_self_learning_repowered_arcgen.py",
        "results/experiment_4283_self_learning_repowered_arcgen.json",
        "online_adaptation_helps",
        "static_cross_family_delta",
        "online_cross_family_delta",
        "tier2_cross_family_delta",
        "family_count_vs_v395",
        "adaptation_curve",
        "verifier_is_oracle=false",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_4283_online_and_tier2_recover_static_headroom(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4283: adaptive arms use only prior-family updates."""

    corpus, static_rows = _repowered_fixture(tmp_path)
    metrics = mod.measure_self_learning(
        corpus,
        static_rows,
        feature_names=["good_feature"],
        arcgen_used=True,
        random_seed=4283,
        bootstrap_resamples=400,
    )

    assert metrics["static_cross_family_delta"] == pytest.approx(0.0)
    assert metrics["online_cross_family_delta"] == pytest.approx(59 / 60)
    assert metrics["tier2_cross_family_delta"] == pytest.approx(59 / 60)
    assert metrics["adaptive_minus_static_ci95"]["online"][0] > 0.0
    assert metrics["adaptive_minus_static_ci95"]["tier2"][0] > 0.0
    assert metrics["online_adaptation_helps"] is True
    assert metrics["family_count_vs_v395"]["held_out_family_n"] == 60
    assert metrics["family_count_vs_v395"]["power_gain_family_n"] == 8
    assert metrics["family_count_vs_v395"]["powered"] is True
    assert metrics["adaptation_curve"][0]["online_minus_static_gain"] == pytest.approx(0.0)
    assert metrics["adaptation_curve"][1]["tier2_minus_static_gain"] == pytest.approx(1.0)
    assert metrics["adaptation_curve"][-1]["cumulative_best_adaptive_minus_static_gain"] > 0.0
    assert metrics["tier2_memory_update"] == mod.TIER2_MEMORY_UPDATE


def test_scenario_4283_run_writes_required_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-4283: run emits bare fields and clean adversarial metadata."""

    corpus, static_rows = _repowered_fixture(tmp_path)
    inputs = mod.ExperimentInputs(
        corpus=corpus,
        static_task_rows=static_rows,
        build_artifact={
            "model_specs": {"architecture": "fixture_set_encoder"},
            "reproducibility_checksum": "sha256:" + "1" * 64,
        },
        model_artifact={
            "feature_names": ["good_feature"],
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "2" * 64,
        },
        original_artifact={"reproducibility_checksum": "sha256:" + "3" * 64},
        arcgen_artifact={
            "reproducibility_checksum": "sha256:" + "4" * 64,
            "held_out_family_n": 8,
        },
        arcgen_used=True,
        input_notes=["fixture_combined_arcgen"],
    )
    monkeypatch.setattr(mod, "load_inputs", lambda _root: inputs)

    artifact = mod.run(tmp_path, adversarial_runner=_adversarial_clean, bootstrap_resamples=400)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: adaptive_self_learning_improves_generalization"
    assert artifact["online_adaptation_helps"] is True
    assert artifact["static_cross_family_delta"] == pytest.approx(0.0)
    assert artifact["online_cross_family_delta"] == pytest.approx(59 / 60)
    assert artifact["tier2_cross_family_delta"] == pytest.approx(59 / 60)
    assert artifact["verifier_is_oracle"] is False
    assert artifact["adversarial_verify"]["circular_moat_overclaim_clean"] is True
    assert (tmp_path / mod.OUTPUT_REL).exists()


def test_req_4283_fallback_and_validation_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-4283: absent ARC-GEN remains honest and malformed artifacts fail closed."""

    corpus, static_rows = _repowered_fixture(tmp_path, n_families=mod.V395_FAMILY_N)
    metrics = mod.measure_self_learning(
        corpus,
        static_rows,
        feature_names=["good_feature"],
        arcgen_used=False,
        random_seed=4283,
        bootstrap_resamples=100,
    )
    assert metrics["family_count_vs_v395"]["held_out_family_n"] == mod.V395_FAMILY_N
    assert metrics["family_count_vs_v395"]["powered"] is False
    assert metrics["family_count_vs_v395"]["read"] == "fallback_still_under_powered"
    assert metrics["online_adaptation_helps"] is False

    blocked = mod._terminal_artifact(
        mod.BLOCKED_INPUTS_VERDICT,
        random_seed=4283,
        duration_s=0.01,
    )
    mod.validate_artifact(blocked)
    assert blocked["online_adaptation_helps"] is False
    assert blocked["verifier_is_oracle"] is False
    assert blocked["acceptance_gate"] is False

    invalid_cases = [
        ({key: value for key, value in blocked.items() if key != "online_adaptation_helps"}, "missing"),
        ({**blocked, "honest_verdict": "pending"}, "terminal-prefixed"),
        ({**blocked, "online_adaptation_helps": {"value": False}}, "online_adaptation_helps"),
        ({**blocked, "static_cross_family_delta": True}, "static_cross_family_delta"),
        ({**blocked, "online_cross_family_delta": True}, "online_cross_family_delta"),
        ({**blocked, "tier2_cross_family_delta": True}, "tier2_cross_family_delta"),
        ({**blocked, "adaptive_minus_static_ci95": {"online": [0.0]}}, "adaptive_minus_static_ci95"),
        ({**blocked, "family_count_vs_v395": []}, "family_count_vs_v395"),
        ({**blocked, "adaptation_curve": {}}, "adaptation_curve"),
        ({**blocked, "verifier_is_oracle": True}, "verifier_is_oracle"),
        ({**blocked, "random_seed": "4283"}, "random_seed"),
        ({**blocked, "field_principles": {}}, "field_principles"),
        ({**blocked, "spec_refs": []}, "spec_refs"),
    ]
    for payload, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)
