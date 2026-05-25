"""Tests for Exp 3069 solver-verifier failure autopsy protocol.

Spec refs: REQ-REPORT-3069, SCENARIO-REPORT-3069.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import solver_verifier_failure_autopsy_protocol_3069 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "research-reporting" / "spec.md"
SCRIPT_PATH = REPO_ROOT / "scripts" / mod.SCRIPT_FILENAME
REQUIRED_FIELDS = {
    "verifier_failure_autopsy_ready",
    "root_cause_hypotheses",
    "recovery_protocol",
    "abstention_policy",
    "candidate_signals",
    "promotion_disqualifiers",
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
    payloads: dict[Path, dict[str, Any]] = {
        mod.EXP3057_REL_PATH: {
            "artifact": "experiment_3057_local_sota_solution_verifier_gain_panel_v1",
            "solution_verifier_calibration_ready": True,
            "false_negative_rate": 1.0,
            "false_positive_rate": 0.0,
            "one_shot_solver_accuracy": 0.125,
            "verifier_selected_accuracy": 0.0,
            "verifier_gain_delta": -0.125,
            "exact_solver_agreement": 1.0,
            "exact_solver_authority": "z3_solver",
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
            "honest_verdict": (
                "complete: solution_verifier_calibration_ready=true; verifier_gain_delta=-0.125"
            ),
        },
        mod.EXP3058_REL_PATH: {
            "artifact": "experiment_3058_aquaforte_style_llm_guided_smt_pilot_v1",
            "llm_guided_smt_pilot_ready": True,
            "guided_success_count": 6,
            "solver_only_success_count": 6,
            "invalid_llm_proposal_count": 0,
            "formal_fallback_preserved": True,
            "flagged_adversarial": True,
            "guidance_vs_solver_only": {
                "guided_minus_solver_only_success_count": 0,
                "guidance_helped": False,
            },
            "honest_verdict": (
                "complete: llm_guided_smt_pilot_ready=true; "
                "guided_success_count=6; solver_only_success_count=6"
            ),
        },
        mod.MATRIX_V20_REL_PATH: {
            "artifact": "experiment_3065_cross_corpus_matrix_v20",
            "matrix_v20_ready": True,
            "status_summaries": {
                "solver_grounded_verification": {
                    "status": "flagged_solver_grounded_no_gain",
                    "citations": [
                        {"source_field": "verifier_gain_delta", "value": -0.125},
                        {
                            "source_field": (
                                "guidance_vs_solver_only.guided_minus_solver_only_success_count"
                            ),
                            "value": 0,
                        },
                    ],
                }
            },
            "flagged_rows": [
                {
                    "row_id": "solver:local_sota_solution_verifier_gain_panel",
                    "status": "flagged",
                    "source_artifact": mod.EXP3057_REL_PATH.as_posix(),
                    "source_field": "verifier_gain_delta",
                },
                {
                    "row_id": "solver:aquaforte_smt_pilot",
                    "status": "flagged",
                    "source_artifact": mod.EXP3058_REL_PATH.as_posix(),
                    "source_field": (
                        "guidance_vs_solver_only.guided_minus_solver_only_success_count"
                    ),
                },
            ],
            "honest_verdict": "complete: matrix_v20_ready=true",
        },
        mod.CAPSTONE_V286_REL_PATH: {
            "artifact": "experiment_3066_capstone_v286",
            "capstone_ready": True,
            "paper_ready": False,
            "solver_grounding_status": "flagged_solver_grounded_no_gain",
            "blocked_claims": [
                {
                    "row_id": "solver:local_sota_solution_verifier_gain_panel",
                    "status": "flagged",
                    "source_artifact": mod.EXP3057_REL_PATH.as_posix(),
                    "source_field": "verifier_gain_delta",
                },
                {
                    "row_id": "solver:aquaforte_smt_pilot",
                    "status": "flagged",
                    "source_artifact": mod.EXP3058_REL_PATH.as_posix(),
                    "source_field": (
                        "guidance_vs_solver_only.guided_minus_solver_only_success_count"
                    ),
                },
            ],
            "honest_verdict": "complete: capstone_ready=true; paper_ready=false",
        },
    }
    for rel_path, payload in payloads.items():
        if rel_path not in omit:
            _write_json(root, rel_path, payload)
    text_payloads = {
        mod.CODEX_REL_PATH: "Spec First\nWrite Tests First\nVerify\nUpdate Ops\n",
        mod.CLAUDE_REL_PATH: (
            "No adversarial-verify gaming.\n"
            "Every verifier MUST match docstring claims.\n"
            "Never claim all tests pass when failures exist.\n"
        ),
        mod.RESEARCH_REFERENCES_REL_PATH: (
            "The First Token Knows: normalized entropy of the first token.\n"
            "Distributional Energy-Based Models use uncertainty to trigger abstention.\n"
            "VERGE and MCS feedback identify minimal correction sets.\n"
            "Lyapunov perturbation sensitivity measures stability under small changes.\n"
        ),
    }
    for rel_path, text in text_payloads.items():
        if rel_path not in omit:
            _write_text(root, rel_path, text)


def test_req_report_3069_spec_and_script_anchor_exists() -> None:
    """REQ-REPORT-3069: the autopsy protocol is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3069" in spec
    assert "SCENARIO-REPORT-3069" in spec
    assert mod.ARTIFACT_FILENAME in spec
    assert "verifier_failure_autopsy_ready" in spec
    assert "promotion_disqualifiers" in spec
    assert "Exp 3070, Exp 3071, Exp 3072, and Exp 3075" in spec
    assert SCRIPT_PATH.exists()


def test_scenario_report_3069_classifies_solver_verifier_failures(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3069: failed verifier gain becomes explicit failure modes."""

    _write_sources(tmp_path)
    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.5)

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["verifier_failure_autopsy_ready"] is True
    assert artifact["metrics_summary"]["exp3057"]["verifier_gain_delta"] == pytest.approx(-0.125)
    assert artifact["metrics_summary"]["exp3057"]["false_negative_rate"] == pytest.approx(1.0)
    assert artifact["metrics_summary"]["exp3058"]["guided_minus_solver_only_success_count"] == 0
    assert artifact["metrics_summary"]["exp3058"]["solver_only_success_count"] == 6
    assert artifact["matrix_capstone_context"]["solver_grounding_status"] == (
        "flagged_solver_grounded_no_gain"
    )
    assert artifact["duration_s"] == pytest.approx(2.5)

    failure_modes = {row["failure_mode"] for row in artifact["failure_mode_classification"]}
    assert {
        "false_negatives",
        "no_verifier_gain",
        "no_smt_lift",
        "self_verification_risk",
        "solver_only_equivalence",
    } <= failure_modes
    assert any(
        row["source_artifact"].endswith("3057_local_sota_solution_verifier_gain_panel_v1.json")
        for row in artifact["root_cause_hypotheses"]
    )
    assert artifact["honest_verdict"].startswith("complete:")

    mod.validate_artifact(artifact)


def test_req_report_3069_recovery_protocol_is_directly_consumable(tmp_path: Path) -> None:
    """REQ-REPORT-3069: next experiments receive fields, gates, and disqualifiers."""

    _write_sources(tmp_path)
    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)
    protocol = artifact["recovery_protocol"]
    signal_names = {row["name"] for row in artifact["candidate_signals"]}
    disqualifier_exps = {row["experiment_id"] for row in artifact["promotion_disqualifiers"]}

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(protocol["minimum_artifact_fields"])
    assert protocol["exact_solver_authority_requirements"]["llm_must_not_be_authority"] is True
    assert (
        protocol["exact_solver_authority_requirements"]["accepted_rows_require_exact_checked"]
        is True
    )
    assert protocol["acceptance_gates"]["verifier_gain_delta_min_exclusive"] == 0.0
    assert protocol["acceptance_gates"]["false_negative_rate_max"] < 1.0
    assert protocol["consumer_experiments"] == ["exp3070", "exp3071", "exp3072", "exp3075"]
    assert protocol["consumer_ready"] is True

    assert {
        "first_token_entropy",
        "abstention_precision",
        "rejection_recall",
        "confidence_coverage",
        "lyapunov_perturbation_sensitivity",
        "verge_mcs_feedback",
    } <= signal_names
    assert {
        "exp3070",
        "exp3071",
        "exp3072",
        "exp3075",
    } <= disqualifier_exps
    assert artifact["abstention_policy"]["forced_accept_reject_disallowed"] is True
    assert artifact["abstention_policy"]["minimum_reported_metrics"] == [
        "abstention_precision",
        "rejection_recall",
        "confidence_coverage",
    ]
    assert artifact["inference_substrate"]["live_llm_inference"] is False
    assert artifact["inference_substrate"]["protocol_only"] is True


def test_req_report_3069_source_traceability_and_fail_closed_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3069: missing source evidence blocks protocol readiness."""

    bad_json = tmp_path / "results" / "bad.json"
    bad_json.parent.mkdir(parents=True, exist_ok=True)
    bad_json.write_text("{not-json", encoding="utf-8")
    assert mod.read_json_object(bad_json) == {}
    assert mod.read_text(tmp_path / "missing.md") == ""
    assert mod._contains_all_failure_modes(None) is False
    assert mod._nested_status([], "missing") == ""
    assert mod._nested_status({"missing": []}, "missing") == ""
    assert mod._float("bad") == 0.0
    assert mod._duration(0.0, None) >= 0.0

    _write_sources(tmp_path, omit={mod.EXP3058_REL_PATH})
    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=2.0)

    assert artifact["verifier_failure_autopsy_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_missing_source:")
    assert any(
        row["path"] == mod.EXP3058_REL_PATH.as_posix() and row["present"] is False
        for row in artifact["source_artifacts"]
    )
    mod.validate_artifact(artifact)

    _write_sources(tmp_path)
    ready = mod.build_artifact(tmp_path, started_s=1.0, now_s=2.0)
    written = mod.write_artifact(
        tmp_path,
        output_path=Path("results") / "exp3069-copy.json",
        started_s=1.0,
        now_s=2.0,
    )
    assert written == tmp_path / "results" / "exp3069-copy.json"
    assert json.loads(written.read_text(encoding="utf-8"))["verifier_failure_autopsy_ready"] is True

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="live_llm_inference"):
        mod.validate_artifact(
            ready
            | {"inference_substrate": ready["inference_substrate"] | {"live_llm_inference": True}}
        )
    with pytest.raises(ValueError, match="promotion_disqualifiers"):
        mod.validate_artifact(ready | {"promotion_disqualifiers": []})
    with pytest.raises(ValueError, match="failure_mode_classification"):
        mod.validate_artifact(ready | {"failure_mode_classification": []})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(ready | {"honest_verdict": "ready"})
    with pytest.raises(ValueError, match="blocked_missing_source"):
        mod.validate_artifact(artifact | {"honest_verdict": "waiting"})
