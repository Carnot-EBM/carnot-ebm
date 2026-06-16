"""Tests for Exp 4295 fixed Tier-2 self-learning retrieval run.

Spec refs: REQ-VERIFY-4295, SCENARIO-VERIFY-4295.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import arc_cross_family_transfer_existing_pool_4271 as exp4271
from carnot.reporting import self_learning_tier2_fixed_retrieval_4295 as mod


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


def _self_learning_fixture(
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


def test_req_4295_spec_declares_fixed_tier2_retrieval_contract() -> None:
    """REQ-VERIFY-4295: OpenSpec declares the fixed Tier-2/retrieval contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4295",
        "SCENARIO-VERIFY-4295",
        "python/carnot/reporting/self_learning_tier2_fixed_retrieval_4295.py",
        "results/experiment_4295_self_learning_tier2_fixed_retrieval.py",
        "results/experiment_4295_self_learning_tier2_fixed_retrieval.json",
        "tier2_memory_cross_family_delta",
        "tier2_retrieval_cross_family_delta",
        "tier2_not_noop",
        "verifier_is_oracle=false",
        "without mutating model weights",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_4295_online_memory_and_retrieval_adapt_prequentially(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4295: adaptive arms use only prior-family state."""

    corpus, static_rows = _self_learning_fixture(tmp_path)
    metrics = mod.measure_self_learning(
        corpus,
        static_rows,
        feature_names=["good_feature"],
        arcgen_used=True,
        random_seed=4295,
        bootstrap_resamples=300,
        retrieval_k=3,
    )

    assert metrics["static_cross_family_delta"] == pytest.approx(0.0)
    assert metrics["online_cross_family_delta"] == pytest.approx(59 / 60)
    assert metrics["tier2_memory_cross_family_delta"] == pytest.approx(59 / 60)
    assert metrics["tier2_retrieval_cross_family_delta"] == pytest.approx(59 / 60)
    assert metrics["adaptive_minus_static_ci95"]["best_adaptive"][0] > 0.0
    assert metrics["adaptive_minus_static_ci95"]["best_adaptive_arm"] in {
        "online",
        "tier2_memory",
        "tier2_retrieval",
    }
    assert metrics["online_adaptation_helps"] is True
    assert metrics["tier2_not_noop"] is True
    assert metrics["tier2_diagnostics"]["memory_differs_from_static_task_n"] > 0
    assert metrics["tier2_diagnostics"]["retrieval_differs_from_static_task_n"] > 0
    assert metrics["family_count_vs_v395"]["powered"] is True
    assert metrics["adaptation_curve"][0]["best_adaptive_minus_static_gain"] == pytest.approx(0.0)
    assert metrics["adaptation_curve"][1]["tier2_memory_minus_static_gain"] == pytest.approx(1.0)
    assert metrics["adaptation_curve"][1]["tier2_retrieval_minus_static_gain"] == pytest.approx(1.0)
    assert metrics["adaptation_curve"][-1]["cumulative_best_adaptive_minus_static_gain"] > 0.0


def test_scenario_4295_run_writes_required_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-4295: run emits bare fields and clean adversarial metadata."""

    corpus, static_rows = _self_learning_fixture(tmp_path)
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
            "held_out_generator_n": 8,
        },
        arcgen_used=True,
        input_notes=["fixture_exp4291_arcgen_appended"],
    )
    monkeypatch.setattr(mod, "load_inputs", lambda _root: inputs)

    artifact = mod.run(tmp_path, adversarial_runner=_adversarial_clean, bootstrap_resamples=300)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: adaptive_self_learning_improves_generalization"
    assert artifact["online_adaptation_helps"] is True
    assert artifact["tier2_not_noop"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["model_specs"]["tier1_online_reweighting_rule"]["model_training"] is False
    assert artifact["model_specs"]["tier2_memory_rule"]["model_training"] is False
    assert artifact["model_specs"]["tier2_retrieval_rule"]["weight_mutation"] is False
    assert artifact["adversarial_verify"]["circular_moat_overclaim_clean"] is True
    assert (tmp_path / mod.OUTPUT_REL).exists()

    monkeypatch.setattr(
        mod,
        "load_inputs",
        lambda _root: (_ for _ in ()).throw(mod.BlockedRun(mod.BLOCKED_INPUTS_VERDICT)),
    )
    blocked = mod.run(tmp_path, adversarial_runner=_adversarial_clean, bootstrap_resamples=10)
    assert blocked["status"] == "blocked"
    assert blocked["acceptance_gate"] is False


def test_req_4295_fallback_and_noop_validation_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-4295: fallback remains honest and no-op Tier-2 artifacts fail."""

    corpus, static_rows = _self_learning_fixture(tmp_path, n_families=mod.V395_FAMILY_N)
    metrics = mod.measure_self_learning(
        corpus,
        static_rows,
        feature_names=["good_feature"],
        arcgen_used=False,
        random_seed=4295,
        bootstrap_resamples=100,
    )
    assert metrics["family_count_vs_v395"]["held_out_family_n"] == mod.V395_FAMILY_N
    assert metrics["family_count_vs_v395"]["powered"] is False
    assert metrics["family_count_vs_v395"]["read"] == "fallback_to_v395_manifest_still_under_powered"
    assert metrics["online_adaptation_helps"] is False
    assert metrics["tier2_not_noop"] is True
    assert mod._family_count_vs_v395(["arcgen_native_task:thin"], arcgen_used=True)["read"] == (
        "exp4291_present_but_still_under_powered"
    )
    assert mod._static_row_for_task("task-x", "family-x", corpus.rows[:2])["vote_correct"] is False
    nominees = {mod.STATIC_ARM: corpus.rows[0]}
    assert mod._pick_from_pattern(nominees, ["missing"])[0] == mod.STATIC_ARM

    blocked = mod._terminal_artifact(
        mod.BLOCKED_INPUTS_VERDICT,
        random_seed=4295,
        duration_s=0.01,
    )
    mod.validate_artifact(blocked)
    assert blocked["tier2_not_noop"] is False
    assert blocked["acceptance_gate"] is False

    inputs = mod.ExperimentInputs(
        corpus=corpus,
        static_task_rows=static_rows,
        build_artifact={"model_specs": {}, "reproducibility_checksum": "sha256:" + "1" * 64},
        model_artifact={"feature_names": ["good_feature"], "reproducibility_checksum": "sha256:" + "2" * 64},
        original_artifact={"reproducibility_checksum": "sha256:" + "3" * 64},
        arcgen_artifact=None,
        arcgen_used=False,
        input_notes=["fixture_fallback"],
    )
    fallback_artifact = mod._complete_artifact(
        inputs=inputs,
        metrics=metrics,
        checksum="sha256:" + "4" * 64,
        random_seed=4295,
        duration_s=0.01,
    )
    assert fallback_artifact["honest_verdict"].endswith("fallback_still_under_powered_static_ceiling_unsettled")
    powered_null_metrics = dict(metrics)
    powered_null_metrics["family_count_vs_v395"] = dict(metrics["family_count_vs_v395"], powered=True)
    powered_null_metrics["online_adaptation_helps"] = False
    powered_null_artifact = mod._complete_artifact(
        inputs=inputs,
        metrics=powered_null_metrics,
        checksum="sha256:" + "5" * 64,
        random_seed=4295,
        duration_s=0.01,
    )
    assert powered_null_artifact["honest_verdict"].endswith(
        "powered_bug_free_static_is_the_ceiling_for_self_learning"
    )

    complete = dict(blocked)
    complete.update(
        {
            "honest_verdict": "complete: powered_bug_free_static_is_the_ceiling_for_self_learning",
            "status": "complete",
            "acceptance_gate": True,
            "tier2_not_noop": False,
            "family_count_vs_v395": {"powered": True},
        }
    )
    with pytest.raises(ValueError, match="tier2_not_noop"):
        mod.validate_artifact(complete)

    invalid_cases = [
        ({key: value for key, value in blocked.items() if key != "online_adaptation_helps"}, "missing"),
        ({**blocked, "honest_verdict": "pending"}, "terminal-prefixed"),
        ({**blocked, "online_adaptation_helps": {"value": False}}, "online_adaptation_helps"),
        ({**blocked, "static_cross_family_delta": True}, "static_cross_family_delta"),
        ({**blocked, "online_cross_family_delta": True}, "online_cross_family_delta"),
        ({**blocked, "tier2_memory_cross_family_delta": True}, "tier2_memory_cross_family_delta"),
        ({**blocked, "tier2_retrieval_cross_family_delta": True}, "tier2_retrieval_cross_family_delta"),
        ({**blocked, "tier2_not_noop": "false"}, "tier2_not_noop"),
        ({**blocked, "adaptive_minus_static_ci95": []}, "adaptive_minus_static_ci95"),
        ({**blocked, "adaptive_minus_static_ci95": {"best_adaptive": [0.0]}}, "adaptive_minus_static_ci95"),
        ({**blocked, "adaptation_curve": {}}, "adaptation_curve"),
        ({**blocked, "verifier_is_oracle": True}, "verifier_is_oracle"),
        ({**blocked, "random_seed": "4295"}, "random_seed"),
        ({**blocked, "field_principles": {}}, "field_principles"),
        ({**blocked, "spec_refs": []}, "spec_refs"),
    ]
    for payload, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)


def test_req_4295_result_entrypoint_imports_module() -> None:
    """REQ-VERIFY-4295: result entrypoint delegates to the reporting module."""

    entrypoint = REPO / "results" / "experiment_4295_self_learning_tier2_fixed_retrieval.py"
    if entrypoint.exists():
        text = entrypoint.read_text(encoding="utf-8")
    else:
        text = json.dumps({"missing": True})
    assert "self_learning_tier2_fixed_retrieval_4295" in text
