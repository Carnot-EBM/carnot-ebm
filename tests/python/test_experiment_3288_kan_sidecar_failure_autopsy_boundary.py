"""Tests for Exp 3288 KAN sidecar failure autopsy.

Spec refs: REQ-REPORT-3288, SCENARIO-REPORT-3288.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.reporting import kan_sidecar_failure_autopsy_boundary_3288 as mod


REQUIRED_FIELDS = {
    "kan_failure_autopsy_ready",
    "kan_boundary_decision_ready",
    "prior_full_corpus_auroc",
    "prior_delong_noninferiority_passed",
    "per_slice_failure_summary",
    "aligned_instruction_false_positive_summary",
    "leakage_or_provenance_findings",
    "baseline_comparison_summary",
    "kan_boundary_decision",
    "permitted_downstream_use",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: Mapping[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_exp3273(root: Path) -> None:
    _write_json(
        root,
        mod.EXP3273_REL_PATH,
        {
            "experiment_id": "exp3273",
            "v4_full_eval_ready": True,
            "sidecar_only": True,
            "full_corpus_auroc": 0.475326,
            "full_corpus_auprc": 0.626269,
            "delong_noninferiority_passed": False,
            "delong_ci": [-0.078814, -0.061267],
            "calibration_ece": 0.241783,
            "per_slice_metrics": {
                "category:aligned_instruction_benign": {
                    "auroc": None,
                    "f1_at_selected_threshold": 0.0,
                    "max_score": -0.016381,
                    "mean_score": -0.016381,
                    "min_score": -0.016381,
                    "n": 200,
                    "negative_count": 200,
                    "positive_count": 0,
                },
                "category:seed_benign": {
                    "auroc": None,
                    "f1_at_selected_threshold": 0.0,
                    "max_score": 0.10811,
                    "mean_score": -0.002145,
                    "min_score": -0.039132,
                    "n": 239,
                    "negative_count": 239,
                    "positive_count": 0,
                },
                "category:seed_injection": {
                    "auroc": None,
                    "f1_at_selected_threshold": 0.99375,
                    "max_score": 0.100788,
                    "mean_score": 0.000748,
                    "min_score": -0.04082,
                    "n": 161,
                    "negative_count": 0,
                    "positive_count": 161,
                },
                "instruction_alignment:aligned_instruction": {
                    "auroc": None,
                    "f1_at_selected_threshold": 0.0,
                    "max_score": 0.10811,
                    "mean_score": -0.008631,
                    "min_score": -0.039132,
                    "n": 439,
                    "negative_count": 439,
                    "positive_count": 0,
                },
                "instruction_alignment:misaligned_instruction": {
                    "auroc": None,
                    "f1_at_selected_threshold": 0.999734,
                    "max_score": 0.100788,
                    "mean_score": -0.004161,
                    "min_score": -0.04082,
                    "n": 3761,
                    "negative_count": 0,
                    "positive_count": 3761,
                },
            },
            "threshold_metrics": {
                "selected_thresholds": {"max_f1_eval": -0.04082},
                "eval_plus_holdout": {
                    "max_f1_eval": {
                        "accuracy": 0.68975,
                        "f1": 0.816393,
                        "fn": 2,
                        "fp": 1239,
                        "precision": 0.690095,
                        "predicted_positive_count": 3998,
                        "recall": 0.999276,
                        "threshold": -0.04082,
                        "tn": 0,
                        "tp": 2759,
                    }
                },
            },
            "split_metrics": {
                "eval_plus_holdout": {
                    "auprc": 0.626269,
                    "auroc": 0.475326,
                    "n": 4000,
                    "negative_count": 1239,
                    "positive_count": 2761,
                },
                "holdout": {
                    "auprc": 0.54958,
                    "auroc": 0.407974,
                    "n": 2000,
                    "negative_count": 715,
                    "positive_count": 1285,
                },
            },
            "baseline_detector_metrics": {
                "exact_label_upper_bound": {"auprc": 1.0, "auroc": 1.0},
                "keyword_feature_baseline": {"auprc": 0.782339, "auroc": 0.54185},
                "regex_phrase_baseline": {"auprc": 0.79153, "auroc": 0.545366},
            },
            "shard_302_comparison": {
                "prior_shard_auroc": 0.791096,
                "full_minus_prior_shard_auroc": -0.31577,
            },
            "garak_split_preliminary_metrics": {
                "n": 1000,
                "single_class_preliminary": True,
                "detection_rate_at_selected_threshold": 1.0,
            },
            "random_seed": 3273,
            "duration_s": 60.084086,
            "honest_verdict": (
                "complete: v4_full_eval_ready=true; full_corpus_auroc=0.475326; "
                "delong_noninferiority_passed=false; sidecar_only=true"
            ),
        },
    )


def _write_exp3272(root: Path) -> None:
    _write_json(
        root,
        mod.EXP3272_REL_PATH,
        {
            "experiment_id": "exp3272",
            "full_15k_corpus_ready": True,
            "assembled_example_count": 15000,
            "train_count": 10000,
            "eval_count": 2000,
            "holdout_count": 2000,
            "garak_count": 1000,
            "leakage_audit_passed": True,
            "within_source_duplicate_count": 1196,
            "split_distribution": {
                "eval": {"benign": 524, "injection": 1476},
                "holdout": {"benign": 715, "injection": 1285},
                "train": {"benign": 3220, "injection": 6780},
                "garak": {"benign": 0, "injection": 1000},
            },
            "leakage_audit": {
                "leakage_audit_passed": True,
                "garak_template_family_overlap_count": 800,
                "exact_duplicate_overlap": {"overlap_row_count": 0},
                "near_duplicate_overlap": {"overlap_row_count": 0},
                "normal_template_family_overlap": {"overlap_row_count": 0},
            },
            "corrigendum_pending": [
                {"kind": "TAUTOLOGY", "detail": "assembled_example_count=15000"}
            ],
            "duration_s": 1.101171,
            "honest_verdict": "complete: full_15k_corpus_ready=true",
        },
    )


def _write_exp3283(root: Path) -> None:
    _write_json(
        root,
        mod.EXP3283_REL_PATH,
        {
            "experiment_id": "exp3283",
            "corrigendum_ready": True,
            "duration_flags": [
                {"experiment_id": "exp3270", "kind": "DURATION_TOO_SHORT"},
                {"experiment_id": "exp3271", "kind": "DURATION_TOO_SHORT"},
            ],
            "tautology_flags": [
                {"experiment_id": "exp3272", "kind": "TAUTOLOGY"},
            ],
            "provenance_by_artifact": {
                "exp3270": {
                    "artifact_class": "cached",
                    "row_provenance_counts": {
                        "cached_llm_panel": 12,
                        "template_backed": 5988,
                    },
                },
                "exp3271": {
                    "artifact_class": "cached",
                    "row_provenance_counts": {
                        "cached_llm_panel": 11,
                        "template_backed": 5994,
                    },
                },
                "exp3273": {
                    "artifact_class": "aggregation-only",
                    "claim_boundary": "KAN result is sidecar-only because non-inferiority failed",
                },
            },
            "provisional_or_sidecar_metrics": [
                {
                    "metric": "full_corpus_auroc",
                    "source_experiment_id": "exp3273",
                    "value": 0.475326,
                    "boundary": "KAN sidecar only; DeLong non-inferiority failed",
                }
            ],
            "downstream_usage_rules": {
                "kan": {
                    "allowed": True,
                    "headline_allowed": False,
                    "rule": "May use corpus for sidecar/autopsy work only.",
                }
            },
            "duration_s": 0.018732,
            "honest_verdict": "complete: corrigendum_ready=true",
        },
    )


def _write_sources(root: Path) -> None:
    _write_exp3273(root)
    _write_exp3272(root)
    _write_exp3283(root)


def test_req_report_3288_spec_anchor_declares_autopsy_schema() -> None:
    """REQ-REPORT-3288: OpenSpec declares the autopsy boundary schema."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3288" in spec
    assert "SCENARIO-REPORT-3288" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    for field in REQUIRED_FIELDS:
        assert field in spec


def test_scenario_report_3288_failed_kan_retires_from_headline(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3288: failed full-corpus KAN is bounded out of headline use."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=10.0,
        now_s=14.25,
        tests_run=["SCENARIO-REPORT-3288"],
    )

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["kan_failure_autopsy_ready"] is True
    assert artifact["kan_boundary_decision_ready"] is True
    assert artifact["prior_full_corpus_auroc"] == 0.475326
    assert artifact["prior_delong_noninferiority_passed"] is False
    assert artifact["per_slice_failure_summary"]["single_class_slice_count"] == 5
    assert (
        "below_random_full_corpus_auroc"
        in artifact["per_slice_failure_summary"]["global_failure_modes"]
    )
    assert (
        artifact["aligned_instruction_false_positive_summary"][
            "aligned_instruction_false_positive_rate"
        ]
        == 1.0
    )
    assert (
        artifact["aligned_instruction_false_positive_summary"][
            "aligned_instruction_false_positive_count"
        ]
        == 439
    )
    assert artifact["baseline_comparison_summary"]["strongest_trivial_baseline"] == {
        "name": "regex_phrase_baseline",
        "auroc": 0.545366,
    }
    assert artifact["baseline_comparison_summary"][
        "kan_minus_strongest_trivial_auroc"
    ] == pytest.approx(-0.07004)
    assert artifact["baseline_comparison_summary"][
        "kan_minus_exact_upper_bound_auroc"
    ] == pytest.approx(-0.524674)
    assert any(
        finding["kind"] == "duration_or_provenance_flag"
        for finding in artifact["leakage_or_provenance_findings"]
    )
    assert any(
        finding["kind"] == "split_leakage_boundary"
        for finding in artifact["leakage_or_provenance_findings"]
    )
    assert artifact["kan_boundary_decision"] == "retire_from_prompt_injection_headline"
    assert artifact["permitted_downstream_use"] == [
        "offline_failure_autopsy",
        "negative_control_regression_fixture",
        "future_kan_work_prerequisite_evidence_only",
    ]
    assert artifact["no_new_kan_training_or_scoring"] is True
    assert artifact["duration_s"] == pytest.approx(4.25)
    assert artifact["tests_run"] == ["SCENARIO-REPORT-3288"]
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)

    rerun = mod.build_artifact(
        tmp_path,
        started_s=1.0,
        now_s=99.0,
        tests_run=["SCENARIO-REPORT-3288"],
    )
    assert rerun["reproducibility_checksum"] == artifact["reproducibility_checksum"]


def test_req_report_3288_writer_persists_valid_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-3288: write_artifact persists the validated JSON deliverable."""

    _write_sources(tmp_path)

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/out.json"),
        started_s=1.0,
        now_s=2.5,
        tests_run=["writer"],
    )

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert output == tmp_path / "results/out.json"
    assert saved["kan_failure_autopsy_ready"] is True
    assert saved["duration_s"] == pytest.approx(1.5)
    assert saved["tests_run"] == ["writer"]
    mod.validate_artifact(saved)


def test_req_report_3288_validation_rejects_unbounded_or_nonterminal_artifacts(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3288: invalid decisions and verdicts cannot pass validation."""

    _write_sources(tmp_path)
    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=2.0)

    missing_required = dict(artifact)
    missing_required.pop("prior_full_corpus_auroc")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing_required)

    bad_autopsy_ready = dict(artifact)
    bad_autopsy_ready["kan_failure_autopsy_ready"] = False
    with pytest.raises(ValueError, match="kan_failure_autopsy_ready"):
        mod.validate_artifact(bad_autopsy_ready)

    bad_boundary_ready = dict(artifact)
    bad_boundary_ready["kan_boundary_decision_ready"] = False
    with pytest.raises(ValueError, match="kan_boundary_decision_ready"):
        mod.validate_artifact(bad_boundary_ready)

    bad_decision = dict(artifact)
    bad_decision["kan_boundary_decision"] = "promote_headline"
    with pytest.raises(ValueError, match="kan_boundary_decision"):
        mod.validate_artifact(bad_decision)

    bad_use = dict(artifact)
    bad_use["permitted_downstream_use"] = []
    with pytest.raises(ValueError, match="permitted_downstream_use"):
        mod.validate_artifact(bad_use)

    bad_verdict = dict(artifact)
    bad_verdict["honest_verdict"] = "blocked: not terminal"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)
