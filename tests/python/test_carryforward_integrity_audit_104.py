"""Tests for the Exp 1351 `.104` carry-forward artifact integrity audit.

Spec: REQ-REPORT-030, SCENARIO-REPORT-030.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.carryforward_integrity_audit_104 import (
    REQUIRED_ARTIFACT_FIELDS,
    SOURCE_FILES,
    build_artifact,
    run,
    write_in_progress_artifact,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _complete(verdict: str, **fields: object) -> dict[str, object]:
    payload: dict[str, object] = {"status": "complete", "honest_verdict": verdict}
    payload.update(fields)
    return payload


def _source_payloads() -> dict[int, dict[str, object]]:
    return {
        1337: _complete("environment_ready_stale_103_artifacts_classified", environment_ready=True),
        1338: _complete(
            "exp1325_stale_environment_failure_gates_closed_recovery_ready",
            exp1325_terminal_classification="stale_skeleton_environment_failure",
            stale_artifacts_not_modified=True,
        ),
        1339: _complete("dryrun_ready_pure_python_tagdispatch_xgrammar_absent", dynamic_grammar_ready=True),
        1341: _complete(
            "local_certificate_slice_diagnostic_exp1340_missing_no_universal_detector_claim",
            universal_detector_claim_allowed=False,
        ),
        1343: {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "blocked on missing exp1342 semantic validator artifact",
        },
        1344: _complete(
            "failure_type_memory_policy_dvi_ready_replay_non_headline",
            dvi_ready=True,
            headline_result_allowed=False,
        ),
        1347: _complete(
            "thrml_unavailable_mapping_notes_only_no_hardware_claim",
            hardware_claim_allowed=False,
        ),
        1348: _complete(
            "cpu_only_update_dynamics_dual_bram_packet_ready_hardware_not_run",
            hardware_claim_allowed=False,
        ),
        1349: _complete(
            "external_parity_gap_audit_complete_local_evidence_only_no_kona_or_external_dependency_claim",
            external_dependency_claim_allowed=False,
        ),
        1350: _complete(
            "milestone_104_9_of_12_criteria_met_carryforward_required",
            criteria_met=9,
            criteria_total=12,
        ),
    }


def _roadmap_text() -> str:
    return """
tasks:
  - id: exp1351-104-carryforward-artifact-integrity-audit
    prior_failures:
      - experiment_id: exp1350-milestone-104-retrospective-and-carryforward
        verdict: milestone_104_9_of_12_criteria_met_carryforward_required
        addressed_by: "Start .105 with a terminal handoff audit."
        retire_if_same_verdict: true
      - experiment_id: exp1340-triggered-certificate-extraction-v6-sota-gguf-semantic-guard
        verdict: missing_terminal_artifact_after_gate_order_failure
        addressed_by: "Keep descendants closed until a replacement certificate artifact exists."
        retire_if_same_verdict: true
  - id: exp1355-logitext-nsvif-partial-smt-validator
    prior_failures:
      - experiment_id: exp1342-triggered-certificate-semantic-validator-gated-on-exp1340-parse
        verdict: missing_or_gate_blocked_after_exp1340_absent
        addressed_by: "Run only after terminal parse evidence."
        retire_if_same_verdict: false
"""


def test_scenario_report_030_missing_exp1340_closes_downstream_gates() -> None:
    """SCENARIO-REPORT-030: missing Exp 1340 is carried forward explicitly."""

    artifact = build_artifact(
        sources=_source_payloads(),
        missing_source_ids={1340, 1342, 1345, 1346},
        artifact_path_records=[
            {"experiment_id": f"exp{exp_id}", "path": f"results/{path}", "exists": exp_id != 1340}
            for exp_id, path in SOURCE_FILES.items()
        ],
        prior_failure_requirements=[
            {
                "experiment_id": "exp1340-triggered-certificate-extraction-v6-sota-gguf-semantic-guard",
                "verdict": "missing_terminal_artifact_after_gate_order_failure",
                "addressed_by": "Keep descendants closed until a replacement certificate artifact exists.",
                "retire_if_same_verdict": True,
            }
        ],
        docs_reconciliation_needed=True,
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert "exp1340" in {entry["experiment_id"] for entry in artifact["missing_artifacts"]}
    assert artifact["terminal_certificate_required"] is True
    assert artifact["docs_reconciliation_needed"] is True
    assert "dynamic_grammar" in {gate["gate"] for gate in artifact["gates_open"]}
    assert {
        "terminal_certificate_parse_gate",
        "semantic_validator",
        "scheduler",
        "dvi_certificate_tail",
        "grpo_vprm",
    } <= {gate["gate"] for gate in artifact["gates_closed"]}
    assert artifact["stale_or_blocked_artifacts"] == [
        {
            "experiment_id": "exp1343",
            "path": "results/experiment_1343_margin_aware_beaver_cactus_scheduler.json",
            "status": "blocked",
            "reason": "blocked on missing exp1342 semantic validator artifact",
        }
    ]
    assert artifact["prior_failure_requirements"][0]["experiment_id"].startswith("exp1340")
    assert artifact["honest_verdict"] == (
        "handoff_state_missing_exp1340_terminal_certificate_semantic_scheduler_dvi_grpo_closed"
    )


def test_req_report_030_writes_in_progress_then_final_json(tmp_path: Path) -> None:
    """REQ-REPORT-030: Exp 1351 writes a bootstrap marker and final artifact."""

    out_path = tmp_path / "results" / "experiment_1351_104_carryforward_artifact_integrity_audit.json"
    bootstrap = write_in_progress_artifact(out_path)

    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    for exp_id, payload in _source_payloads().items():
        _write_json(tmp_path / "results" / SOURCE_FILES[exp_id], payload)
    (tmp_path / "ops").mkdir()
    (tmp_path / "openspec" / "change-proposals").mkdir(parents=True)
    (tmp_path / "_bmad").mkdir()
    (tmp_path / "research-roadmap.yaml").write_text(_roadmap_text(), encoding="utf-8")
    (tmp_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        "exp1340 missing; semantic/DVI gates remain closed", encoding="utf-8"
    )
    (tmp_path / "ops" / "conductor-log.md").write_text("exp1340 GATE_BLOCK", encoding="utf-8")
    (tmp_path / "ops" / "changelog.md").write_text("exp1350 only", encoding="utf-8")
    (tmp_path / "ops" / "status.md").write_text("Milestone .105 planned", encoding="utf-8")
    (tmp_path / "research-complete.yaml").write_text("2026.04.104", encoding="utf-8")
    (tmp_path / "research-references.md").write_text("exp1340 missing evidence", encoding="utf-8")
    (tmp_path / "_bmad" / "traceability.md").write_text("REQ-REPORT-029", encoding="utf-8")

    artifact = run(root=tmp_path, out_path=out_path)

    assert artifact["status"] == "complete"
    assert artifact["terminal_certificate_required"] is True
    assert len(artifact["artifact_paths_checked"]) == len(SOURCE_FILES) + 8
    assert [entry["experiment_id"] for entry in artifact["prior_failure_requirements"]] == [
        "exp1350-milestone-104-retrospective-and-carryforward",
        "exp1340-triggered-certificate-extraction-v6-sota-gguf-semantic-guard",
        "exp1342-triggered-certificate-semantic-validator-gated-on-exp1340-parse",
    ]
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "complete"


def test_req_report_030_terminal_certificate_and_semantic_evidence_open_gates() -> None:
    """REQ-REPORT-030: downstream gates open only when terminal evidence exists."""

    sources = _source_payloads()
    sources[1340] = _complete(
        "terminal_certificate_parse_gate_open",
        certificate_parse_rate=0.8,
        terminal_blocker=None,
    )
    sources[1342] = _complete(
        "semantic_validator_executed_unknown_preserved",
        validator_execution_pass_rate=0.7,
    )
    sources[1343] = _complete("scheduler_complete_after_semantic_gate")
    sources[1345] = _complete("dvi_certificate_tail_lossless", lossless_acceptance_claim_allowed=True)
    sources[1346] = _complete("grpo_vprm_micro_audit_terminal")

    artifact = build_artifact(
        sources=sources,
        missing_source_ids=set(),
        artifact_path_records=[
            {"experiment_id": f"exp{exp_id}", "path": f"results/{path}", "exists": True}
            for exp_id, path in SOURCE_FILES.items()
        ],
        prior_failure_requirements=[],
        docs_reconciliation_needed=False,
    )

    assert artifact["terminal_certificate_required"] is False
    assert {
        "terminal_certificate_parse_gate",
        "semantic_validator",
        "scheduler",
        "dvi_certificate_tail",
        "grpo_vprm",
    } <= {gate["gate"] for gate in artifact["gates_open"]}
    assert "semantic_validator" not in {gate["gate"] for gate in artifact["gates_closed"]}
