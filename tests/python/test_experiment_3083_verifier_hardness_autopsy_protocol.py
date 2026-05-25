"""Tests for Exp 3083 verifier-hardness autopsy protocol.

Spec refs: REQ-REPORT-3083, SCENARIO-REPORT-3083.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import verifier_hardness_autopsy_protocol_3083 as mod


REQUIRED_FIELDS = {
    "verifier_hardness_protocol_ready",
    "prior_failure_modes",
    "perturbation_categories",
    "abstention_metrics_required",
    "formal_feedback_disqualifiers",
    "repair_disqualifiers",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path, text: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_sources(root: Path, *, omit: set[Path] | None = None) -> None:
    omit = omit or set()
    json_payloads = {
        mod.EXP3057_REL_PATH: {
            "artifact": "experiment_3057_local_sota_solution_verifier_gain_panel_v1",
            "verifier_gain_delta": -0.125,
            "false_negative_rate": 1.0,
            "false_positive_rate": 0.0,
            "one_shot_solver_accuracy": 0.125,
            "verifier_selected_accuracy": 0.0,
            "exact_solver_agreement": 1.0,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
            "honest_verdict": "complete: verifier_gain_delta=-0.125",
        },
        mod.EXP3070_REL_PATH: {
            "artifact": "experiment_3070_first_token_abstention_sota_panel_v1",
            "first_token_auc": 0.533333,
            "abstention_precision": 0.5,
            "abstention_coverage": 0.5,
            "rejection_recall": 0.25,
            "accepted_count": 2,
            "rejected_count": 2,
            "abstained_count": 4,
            "false_negative_rate": 0.25,
            "false_positive_rate": 0.25,
            "verifier_gain_delta_with_abstention": 0.5,
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {"kind": "TAUTOLOGY", "severity": "critical"},
                {"kind": "DURATION_TOO_SHORT", "severity": "critical"},
            ],
            "honest_verdict": "complete: abstention_precision=0.5",
        },
        mod.EXP3071_REL_PATH: {
            "artifact": "experiment_3071_verge_mcs_smt_correction_pilot_v1",
            "mcs_feedback_ready": True,
            "guided_success_count": 5,
            "solver_only_success_count": 5,
            "guidance_vs_solver_only": {
                "guidance_helped": False,
                "guided_minus_solver_only_success_count": 0,
            },
            "invalid_llm_proposal_count": 0,
            "formal_fallback_preserved": True,
            "mcs_count": 5,
            "fixture_count": 6,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
            "honest_verdict": "complete: mcs_feedback_ready=true",
        },
        mod.EXP3080_REL_PATH: {
            "artifact": "experiment_3080_capstone_v287",
            "capstone_ready": True,
            "paper_ready": False,
            "publication_blocker_count": 42,
            "verifier_gain_status": "flagged_or_gated_verifier_gain_recovery_incomplete",
            "repair_claim_status": "bounded_and_gated_skipped",
            "fr11_self_learning_status": "flagged_controller_only_budget_exceeded",
            "next_milestone_recommendation": (
                "2026.05.288: raise abstention_precision to the gate, rerun Exp3072, "
                "and run repair only after verifier-gain gates pass."
            ),
            "honest_verdict": "complete: capstone_ready=true; paper_ready=false",
        },
    }
    for rel_path, payload in json_payloads.items():
        if rel_path not in omit:
            _write_json(root, rel_path, payload)

    text_payloads = {
        mod.AGENTS_REL_PATH: "Read CODEX.md. Spec first, tests first, verify.\n",
        mod.CODEX_REL_PATH: "Spec First\nWrite Tests First\nVerify\n",
        mod.CLAUDE_REL_PATH: (
            "No adversarial-verify gaming. Verifier authenticity discipline. "
            "Self-verification is not independent authority.\n"
        ),
        mod.RESEARCH_REFERENCES_REL_PATH: (
            "Rethinking LLMs as Verifiers: verification can be harder than solving. "
            "I-CALM confidence-aware abstention uses humility and coverage accounting. "
            "Task Abstention for code generation uses execution consistency. "
            "Learning to Self-Verify shows generation-verification asymmetry. "
            "Dafny verifier feedback needs structural anchors and vacuity guards. "
            "VERGE uses MCS feedback for localized repair.\n"
        ),
    }
    for rel_path, text in text_payloads.items():
        if rel_path not in omit:
            _write_text(root, rel_path, text)


def test_req_report_3083_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3083: OpenSpec declares the protocol before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3083" in spec
    assert "SCENARIO-REPORT-3083" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "verifier_hardness_protocol_ready" in spec
    assert "Exp 3084 through Exp 3089" in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3083_builds_failure_aware_protocol(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3083: prior negative evidence becomes rerun gates."""

    _write_sources(tmp_path)
    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.25)

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["verifier_hardness_protocol_ready"] is True
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["prior_metrics"]["exp3057"]["verifier_gain_delta"] == pytest.approx(-0.125)
    assert artifact["prior_metrics"]["exp3070"]["abstention_precision"] == pytest.approx(0.5)
    assert artifact["prior_metrics"]["exp3071"]["guided_minus_solver_only_success_count"] == 0
    assert artifact["prior_metrics"]["exp3080"]["publication_blocker_count"] == 42

    failure_ids = {row["id"] for row in artifact["prior_failure_modes"]}
    assert {
        "negative_verifier_gain",
        "exact_good_false_negative",
        "low_abstention_precision",
        "tautological_abstention_metrics",
        "formal_feedback_solver_only_parity",
        "provenance_contamination",
        "capstone_publication_blockers",
    } <= failure_ids

    comparison_ids = {row["id"] for row in artifact["research_reference_comparisons"]}
    assert {
        "verifier_hardness",
        "i_calm_abstention",
        "task_abstention",
        "self_verification_asymmetry",
        "formal_feedback",
    } <= comparison_ids

    axes = {row["primary_axis"] for row in artifact["perturbation_categories"]}
    assert axes == {"solving", "verifying", "abstaining", "repairing"}
    assert all(row["separates_from"] for row in artifact["perturbation_categories"])

    metric_names = {row["name"] for row in artifact["abstention_metrics_required"]}
    assert {
        "acceptance_precision",
        "acceptance_coverage",
        "rejection_recall",
        "rejection_precision",
        "abstention_precision",
        "abstention_coverage",
        "false_accept_rate",
        "false_reject_rate",
    } <= metric_names

    mod.validate_artifact(artifact)


def test_req_report_3083_metric_contracts_and_disqualifiers(tmp_path: Path) -> None:
    """REQ-REPORT-3083: Exp 3084-3089 receive non-vacuous metric contracts."""

    _write_sources(tmp_path)
    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)
    contracts = artifact["experiment_metric_contracts"]

    assert set(contracts) == {"exp3084", "exp3085", "exp3086", "exp3087", "exp3088", "exp3089"}
    for contract in contracts.values():
        groups = set(contract["metric_groups"])
        assert {"acceptance", "rejection", "abstention", "formal_feedback"} <= groups
        assert contract["exact_label_provenance_required"] is True
        assert contract["solver_only_baseline_required"] is True
        assert contract["row_level_decision_counts_required"] is True

    global_disqualifiers = {row["id"] for row in artifact["global_disqualifiers"]}
    formal_disqualifiers = {row["id"] for row in artifact["formal_feedback_disqualifiers"]}
    repair_disqualifiers = {row["id"] for row in artifact["repair_disqualifiers"]}
    assert {
        "label_leakage",
        "solver_only_parity_without_lift",
        "tiny_model_headline_substitution",
    } <= global_disqualifiers
    assert "solver_only_parity_without_lift" in formal_disqualifiers
    assert "feedback_without_localized_counterexample" in formal_disqualifiers
    assert "tautological_repair" in repair_disqualifiers
    assert "syntax_only_success" in repair_disqualifiers
    assert "tiny_model_headline_substitution" in repair_disqualifiers

    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_upstream_artifacts",
        "source": "specified_protocol_sources",
        "executes_models": False,
        "executes_verifiers": False,
        "executes_solver": False,
        "executes_repair": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "no_live_llm_inference": True,
    }


def test_req_report_3083_source_traceability_and_fail_closed_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3083: missing evidence blocks readiness instead of inferring."""

    bad_json = tmp_path / "bad.json"
    list_json = tmp_path / "list.json"
    bad_json.write_text("{not-json", encoding="utf-8")
    list_json.write_text("[1, 2]\n", encoding="utf-8")
    assert mod.read_json_object(bad_json) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.read_text(tmp_path / "missing.md") == ""
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod._float("bad") == 0.0
    assert mod._int("bad") == 0
    assert mod._int(True) == 0
    assert mod._corrigendum_kinds({"corrigendum_pending": "bad"}) == []
    assert mod._duration(0.0, None) >= 0.0

    _write_sources(tmp_path, omit={mod.EXP3070_REL_PATH})
    blocked = mod.build_artifact(tmp_path, started_s=1.0, now_s=2.0)

    assert blocked["verifier_hardness_protocol_ready"] is False
    assert blocked["honest_verdict"].startswith("blocked_missing_source:")
    assert any(
        row["path"] == mod.EXP3070_REL_PATH.as_posix() and row["present"] is False
        for row in blocked["source_artifacts"]
    )
    mod.validate_artifact(blocked)

    _write_sources(tmp_path)
    ready = mod.build_artifact(tmp_path, started_s=1.0, now_s=2.0)
    written = mod.write_artifact(
        tmp_path,
        output_path=Path("results") / "exp3083-copy.json",
        started_s=1.0,
        now_s=2.0,
    )
    assert written == tmp_path / "results" / "exp3083-copy.json"
    assert (
        json.loads(written.read_text(encoding="utf-8"))["verifier_hardness_protocol_ready"] is True
    )

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="no_live_llm_inference"):
        mod.validate_artifact(
            ready
            | {
                "inference_substrate": ready["inference_substrate"]
                | {"no_live_llm_inference": False}
            }
        )
    with pytest.raises(ValueError, match="perturbation categories"):
        mod.validate_artifact(ready | {"perturbation_categories": []})
    with pytest.raises(ValueError, match="Exp 3084-3089"):
        mod.validate_artifact(ready | {"experiment_metric_contracts": {"exp3084": {}}})
    with pytest.raises(ValueError, match="disqualifiers"):
        mod.validate_artifact(ready | {"global_disqualifiers": []})
    with pytest.raises(ValueError, match="formal-feedback and repair disqualifiers"):
        mod.validate_artifact(ready | {"formal_feedback_disqualifiers": []})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(ready | {"honest_verdict": "ready"})
    with pytest.raises(ValueError, match="blocked_missing_source"):
        mod.validate_artifact(blocked | {"honest_verdict": "waiting"})
