"""Tests for Exp 1483 HalluGuard risk-bound fit audit.

Spec: REQ-REPORT-051, SCENARIO-REPORT-051.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import halluguard_risk_bound_fit_audit as exp1483


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_source_inputs(root: Path) -> None:
    (root / "research-references.md").write_text(
        "\n".join(
            [
                "# Research References",
                "",
                "### HalluGuard Formal Hallucination-Risk Bounds",
                "- What: Decomposes hallucination risk into data-driven and reasoning-driven risk-bound components.",
                "- Concrete experiment hook: Audit Carnot evidence into DHRB/RHRB-style fields.",
                "- Do not claim full HalluGuard reproduction unless NTK/certification assumptions are implemented and checked.",
                "",
                "### Next Entry",
                "unrelated",
            ]
        ),
        encoding="utf-8",
    )
    matrix = root / "docs" / "research-notes" / "paper_v6_anchored_claim_matrix.md"
    matrix.parent.mkdir(parents=True, exist_ok=True)
    matrix.write_text(
        "Does not claim universal verifier dominance or full external reproduction.\n",
        encoding="utf-8",
    )

    _write_json(
        root / "results" / "experiment_1468_live_sota_logprob_telemetry_preflight.json",
        {
            "status": "complete",
            "honest_verdict": "live_sota_topk_telemetry_ready",
            "live_sota_model_inference_used": True,
            "topk_logprobs_available": True,
            "logits_available": True,
            "telemetry_cases_completed": 12,
        },
    )
    _write_json(
        root / "results" / "experiment_1470_beaver_lite_deterministic_bound_smoke.json",
        {
            "status": "complete",
            "honest_verdict": "sound_bound_live_exp1468",
            "bound_is_sound": True,
            "mock_or_live_logprobs": "live_exp1468",
            "unsafe_mass_bounds": [0.0, 0.01],
            "empirical_violation_rates": [0.0, 0.0],
            "prefix_closed_constraint": [{"prefix_closed": True}],
        },
    )
    _write_json(
        root / "results" / "experiment_1473_live_telemetry_adversarial_validity_audit.json",
        {
            "status": "complete",
            "claim_allowed": False,
            "honest_verdict": "telemetry_claim_blocked_adversarial_audit",
            "telemetry_validity_verdict": "invalid_for_headline_claim_superficial_or_mechanical_gate",
            "superficial_baseline_results": {
                "telemetry": {
                    "label_key": "known_verifier_label",
                    "claim_blockers": ["source_diagnostic_lineage_retired"],
                }
            },
        },
    )
    _write_json(
        root / "results" / "experiment_1480_live_sota_balanced_telemetry_v2.json",
        {
            "status": "complete",
            "honest_verdict": "balanced_live_sota_telemetry_ready",
            "live_sota_model_inference_used": True,
            "topk_logprobs_available": True,
            "logits_available": True,
            "telemetry_cases_completed": 2,
            "telemetry_manifest_path": "results/live_sota_balanced_telemetry_manifest_1480.jsonl",
            "balanced_label_counts": {"correct": 1, "incorrect": 1},
        },
    )
    _write_json(
        root / "results" / "experiment_1482_beaver_lite_live_prefix_bound_calibration.json",
        {
            "status": "complete",
            "honest_verdict": "sound_bound_live_exp1480_plus_exp1468_calibrated",
            "bound_is_sound": True,
            "constraints_evaluated": 2,
            "mock_or_live_logprobs": "live_exp1480_plus_exp1468",
            "unsafe_mass_bounds": [0.1, 0.2],
            "empirical_violation_rates": [0.0, 0.0],
            "prefix_closed_constraints": [
                {"constraint_id": "c1", "prefix_closed": True, "source_family": "fover_claim"},
                {"constraint_id": "c2", "prefix_closed": True, "source_family": "arithmetic"},
            ],
        },
    )

    manifest = root / "results" / "live_sota_balanced_telemetry_manifest_1480.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "case_id": "case-correct",
                        "known_verifier_label": 1,
                        "correctness_label": "correct",
                        "format_valid": True,
                        "token_logprobs_available": True,
                        "topk_alternatives_available": True,
                    }
                ),
                json.dumps(
                    {
                        "case_id": "case-incorrect",
                        "known_verifier_label": 0,
                        "correctness_label": "incorrect",
                        "format_valid": False,
                        "token_logprobs_available": True,
                        "topk_alternatives_available": True,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_req_report_051_spec_anchor_exists() -> None:
    """REQ-REPORT-051, SCENARIO-REPORT-051: Exp 1483 is spec-anchored."""

    spec = (
        exp1483.REPO_ROOT / "openspec" / "capabilities" / "research-reporting" / "spec.md"
    ).read_text(encoding="utf-8")

    assert "REQ-REPORT-051" in spec
    assert "SCENARIO-REPORT-051" in spec
    assert "experiment_1483_halluguard_risk_bound_fit_audit.json" in spec


def test_req_report_051_writes_in_progress_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-051: seed the deliverable before source-artifact loading."""

    out_path = tmp_path / "results" / exp1483.OUTPUT_FILENAME

    artifact = exp1483.write_in_progress_artifact(out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert exp1483.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact == written
    assert artifact["status"] == "in_progress"
    assert artifact["source_artifacts"] == []
    assert artifact["risk_decomposition_complete"] is False
    assert artifact["claim_allowed"] is False
    assert artifact["honest_verdict"] == "in_progress"


def test_scenario_report_051_builds_decomposition_from_source_payloads(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-051: source fields map into separate risk components."""

    _write_source_inputs(tmp_path)
    inputs = exp1483.load_source_inputs(tmp_path)
    artifact = exp1483.build_artifact(
        inputs,
        audit_note_path="docs/research-notes/halluguard_carnot_risk_bound_fit.md",
    )

    assert exp1483.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["risk_decomposition_complete"] is True
    assert artifact["claim_allowed"] is False
    assert artifact["data_driven_fields_available"] == [
        "live_sota_model_inference_used",
        "topk_logprobs_available",
        "logits_available",
        "known_verifier_label",
        "balanced_label_counts",
        "telemetry_adversarial_validity_verdict",
        "missing_evidence_caveats",
    ]
    assert artifact["reasoning_driven_fields_available"] == [
        "bound_is_sound",
        "unsafe_mass_bounds",
        "empirical_violation_rates",
        "prefix_closed_constraints",
        "reasoning_step_validity_limitations",
    ]
    assert artifact["data_driven_hallucination_risk_proxy"]["telemetry_cases_completed"] == 2
    assert artifact["reasoning_driven_hallucination_risk_proxy"]["constraints_evaluated"] == 2
    assert artifact["reasoning_driven_hallucination_risk_proxy"]["max_unsafe_mass_bound"] == 0.2
    assert len(artifact["missing_assumptions"]) >= 5
    assert any("NTK" in item for item in artifact["missing_assumptions"])
    assert artifact["honest_verdict"] == "halluguard_style_fit_audit_only_no_full_reproduction"
    exp1483.validate_artifact(artifact)


def test_req_report_051_validator_blocks_claim_drift(tmp_path: Path) -> None:
    """REQ-REPORT-051: full HalluGuard claims stay blocked while assumptions are missing."""

    _write_source_inputs(tmp_path)
    artifact = exp1483.build_artifact(
        exp1483.load_source_inputs(tmp_path),
        audit_note_path="docs/research-notes/halluguard_carnot_risk_bound_fit.md",
    )

    with pytest.raises(ValueError, match="claim_allowed"):
        exp1483.validate_artifact(dict(artifact, claim_allowed=True))
    with pytest.raises(ValueError, match="missing_assumptions"):
        exp1483.validate_artifact(dict(artifact, missing_assumptions=[]))
    with pytest.raises(ValueError, match="risk_decomposition_complete"):
        exp1483.validate_artifact(dict(artifact, risk_decomposition_complete=False))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"status": "in_progress"}, "status"),
        ({"data_driven_fields_available": []}, "data_driven_fields_available"),
        ({"reasoning_driven_fields_available": []}, "reasoning_driven_fields_available"),
        ({"audit_note_path": ""}, "audit_note_path"),
        ({"honest_verdict": "complete"}, "honest_verdict"),
    ],
)
def test_req_report_051_validator_rejects_schema_drift(
    tmp_path: Path,
    override: dict,
    message: str,
) -> None:
    """REQ-REPORT-051: schema drift fails before the claim can be reused."""

    _write_source_inputs(tmp_path)
    artifact = exp1483.build_artifact(
        exp1483.load_source_inputs(tmp_path),
        audit_note_path="docs/research-notes/halluguard_carnot_risk_bound_fit.md",
    )

    with pytest.raises(ValueError, match=message):
        exp1483.validate_artifact(dict(artifact, **override))

    missing = dict(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required fields"):
        exp1483.validate_artifact(missing)


def test_scenario_report_051_run_writes_note_and_terminal_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-051: run writes the note and final blocked-claim JSON."""

    _write_source_inputs(tmp_path)
    out_path = tmp_path / "results" / exp1483.OUTPUT_FILENAME
    note_path = tmp_path / exp1483.AUDIT_NOTE_REL

    artifact = exp1483.run(root=tmp_path, out_path=out_path, audit_note_path=note_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))
    note = note_path.read_text(encoding="utf-8")

    assert artifact == written
    assert written["status"] == "complete"
    assert written["audit_note_path"] == exp1483.AUDIT_NOTE_REL
    assert "Full HalluGuard reproduction is not claimed" in note
    assert "Data-Driven Evidence-Availability Risk" in note
    assert "Reasoning-Step Risk" in note
