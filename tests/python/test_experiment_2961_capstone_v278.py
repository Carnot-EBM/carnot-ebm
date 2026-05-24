"""Tests for Exp 2961 milestone .278 capstone.

Spec refs: REQ-REPORT-2961, SCENARIO-REPORT-2961.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v278_2961 as mod


REQUIRED_FIELDS = {
    "honest_verdict",
    "milestone",
    "paper_ready",
    "headline_outcome",
    "clean_artifacts",
    "flagged_artifacts",
    "blocked_artifacts",
    "gated_skipped_artifacts",
    "missing_artifacts",
    "pilot_only_artifacts",
    "aggregation_only_artifacts",
    "gaps_closed",
    "gaps_remaining",
    "forbidden_claims_absent",
    "next_milestone_recommendations",
    "inference_substrate",
    "duration_s",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_roadmap(root: Path) -> None:
    (root / mod.ROADMAP_REL_PATH).write_text(
        """
milestone: "2026.05.278"
tasks:
  - id: exp2949
    deliverable: "results/experiment_2949_archive_v277_activate_v278.json"
    title: "Archive .277 + Activate .278"
    inference_substrate: aggregation_from_upstream_artifacts
  - id: exp2950
    deliverable: "results/experiment_2950_code_taxonomy_repair_prompt_manifest_v1.json"
    title: "SOTA Code Failure Taxonomy Repair Prompt Manifest v1"
    inference_substrate: aggregation_from_upstream_artifacts
  - id: exp2951
    deliverable: "results/experiment_2951_structured_candidate_manifest_adapter_v1.json"
    title: "Structured Candidate Manifest Adapter v1"
    inference_substrate: deterministic_wiring
  - id: exp2952
    deliverable: "results/experiment_2952_sota_taxonomy_guided_code_repair_eval_v1.json"
    title: "Gated SOTA Taxonomy-Guided Code Repair Evaluation v1"
    inference_substrate: live_llm_inference
    gated_on:
      - upstream: exp2950
        artifact_field: repair_prompt_manifest_ready
        op: "=="
        value: true
      - upstream: exp2951
        artifact_field: structured_decode_manifest_ready
        op: "=="
        value: true
  - id: exp2953
    deliverable: "results/experiment_2953_code_verifier_threshold_policy_v1.json"
    title: "Code Verifier Threshold Policy v1"
    inference_substrate: aggregation_from_upstream_artifacts
  - id: exp2954
    deliverable: "results/experiment_2954_fr11_utility_gated_replay_curriculum_v2.json"
    title: "FR-11 Utility-Gated Replay Curriculum v2"
    inference_substrate: aggregation_from_upstream_artifacts
  - id: exp2955
    deliverable: "results/experiment_2955_gatemate_constraints_materialization_v4.json"
    title: "GateMate Constraints Materialization v4"
    inference_substrate: deterministic_wiring
  - id: exp2956
    deliverable: "results/experiment_2956_gatemate_n16_bitstream_build_v4.json"
    title: "Gated GateMate n=16 Bitstream Build v4"
    inference_substrate: hardware_build
    gated_on:
      - upstream: exp2955
        artifact_field: gatemate_constraints_ready
        op: "=="
        value: true
  - id: exp2957
    deliverable: "results/experiment_2957_gatemate_flash_timing_smoke_v2.json"
    title: "Gated GateMate Flash + Timing Smoke v2"
    inference_substrate: hardware_smoke
    gated_on:
      - upstream: exp2956
        artifact_field: gatemate_bitstream_built
        op: "=="
        value: true
  - id: exp2958
    deliverable: "results/experiment_2958_polarfire_1000_clause_scorer_v2.json"
    title: "PolarFire 1000-Clause Scorer Hash v2"
    inference_substrate: hardware_smoke
  - id: exp2959
    deliverable: "results/experiment_2959_nl_to_z3_execution_repair_mini_v2.json"
    title: "NL-to-Z3 Execution Repair Mini v2"
    inference_substrate: live_llm_inference
  - id: exp2960
    deliverable: "results/experiment_2960_cross_corpus_matrix_v12.json"
    title: "Cross-Corpus Matrix v12"
    inference_substrate: aggregation_from_upstream_artifacts
  - id: exp2961
    deliverable: "results/experiment_2961_capstone_v278.json"
    title: "Capstone .278"
    inference_substrate: aggregation_from_upstream_artifacts
""".lstrip(),
        encoding="utf-8",
    )


def _write_ready_sources(root: Path) -> None:
    _write_roadmap(root)
    _write_json(
        root,
        mod.CAPSTONE_V277_REL_PATH,
        {
            "honest_verdict": "complete: milestone=2026.05.277; paper_ready=true",
            "milestone": "2026.05.277",
            "paper_ready": True,
            "deep_think_corrigenda_outcomes": {
                "headline_outcome": "narrow",
                "code_corpus_auprc": 0.8888888888888888,
                "same_schedule_speedup": 0.98225,
            },
            "gaps_for_278": [
                "Verifier signal has not improved generation.",
                "FR-11 is still a scheduler, not a proven self-learner.",
                "Hardware evidence is narrow and incomplete.",
            ],
            "paper_v6_forbidden_claims": ["No KV260 speedup claim."],
        },
    )
    _write_json(
        root,
        mod.EXP2950_REL_PATH,
        {
            "honest_verdict": "complete: repair prompt manifest ready",
            "repair_prompt_manifest_ready": True,
            "upstream_metrics": {"pass_at_1": 0.06, "pass_at_k": 0.16},
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
        },
    )
    _write_json(
        root,
        mod.EXP2951_REL_PATH,
        {
            "honest_verdict": "complete: structured candidate manifest adapter ready",
            "structured_decode_manifest_ready": True,
            "validation_fixture_passed": True,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
        },
    )
    _write_json(
        root,
        mod.EXP2952_REL_PATH,
        {
            "honest_verdict": "complete: taxonomy-guided repair delta passed",
            "n_tasks": 4,
            "baseline_pass_at_1": 0.0,
            "repair_pass_at_1": 0.25,
            "pass_at_1_delta": 0.25,
            "baseline_pass_at_k": 0.25,
            "repair_pass_at_k": 0.5,
            "pass_at_k_delta": 0.25,
            "syntax_failure_rate_delta": -0.0625,
            "false_accept_delta": -0.125,
            "taxonomy_repair_delta_pass": True,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
        },
    )
    _write_json(
        root,
        mod.EXP2953_REL_PATH,
        {
            "honest_verdict": "complete: threshold policy ready",
            "threshold_policy_ready": True,
            "selected_default_threshold": 1.0,
            "expected_false_accept_rate_at_default": 0.010135135135,
            "expected_recall_at_default": 1.0,
            "expected_ppv_at_default": 0.8888888888888888,
        },
    )
    _write_json(
        root,
        mod.EXP2954_REL_PATH,
        {
            "honest_verdict": "complete: utility_gated_replay_improved",
            "self_learning_utility_artifact_ready": True,
            "self_learning_utility_positive": True,
            "heldout_utility_baseline": 0.241027227723,
            "heldout_utility_after": 0.352886467009,
            "heldout_utility_delta": 0.111859239286,
            "forgetting_guard_passed": True,
            "model_weights_mutated": False,
            "rollback_triggered": False,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
        },
    )
    _write_json(
        root,
        mod.EXP2955_REL_PATH,
        {
            "honest_verdict": "complete: gatemate_constraints_materialized",
            "gatemate_constraints_ready": True,
            "dirtyjtag_detected": True,
            "constraints_sha256": "constraints-sha",
        },
    )
    _write_json(
        root,
        mod.EXP2956_REL_PATH,
        {
            "honest_verdict": "complete: gatemate_n16_bitstream_built",
            "gatemate_bitstream_built": True,
            "bitstream_sha256": "bitstream-sha",
            "timing_summary": {"timing_met": True, "max_frequency_mhz": 15.69},
        },
    )
    _write_json(
        root,
        mod.EXP2957_REL_PATH,
        {
            "honest_verdict": "blocked_board_not_detected",
            "board_detected": False,
            "flash_attempted": False,
            "flash_succeeded": False,
            "smoke_vector_passed": False,
            "no_speedup_claim": True,
            "no_boltzmann_claim": True,
        },
    )
    _write_json(
        root,
        mod.EXP2958_REL_PATH,
        {
            "honest_verdict": "complete: polarfire_1000_clause_hash_verified",
            "board_reachable": True,
            "clause_count": 1000,
            "polarfire_1000_clause_hash_verified": True,
            "elapsed_ms": 1029.3225100031123,
            "remote_arch": "riscv64",
        },
    )
    _write_json(
        root,
        mod.EXP2958_TRANSCRIPT_REL_PATH,
        {
            "schema": "carnot.polarfire_sat_scorer_transcript.v2",
            "total_wall_clock_s": 1.0293225100031123,
            "evaluation_cycles_per_clause": 18,
        },
    )
    _write_json(
        root,
        mod.EXP2959_REL_PATH,
        {
            "honest_verdict": "complete: local SOTA logic proposals accepted_or_rejected_by_z3",
            "z3_import_ok": True,
            "z3_execution_repaired": True,
            "z3_execution_rate": 0.083333,
            "solver_verified_accuracy": 0.0,
            "answer_accuracy": 0.083333,
            "parseability_rate": 0.083333,
            "n_items": 12,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
        },
    )
    _write_json(
        root,
        mod.EXP2960_REL_PATH,
        {
            "artifact": "experiment_2960_cross_corpus_matrix_v12",
            "honest_verdict": "complete: matrix_v12_ready=true",
            "matrix_v12_ready": True,
            "forbidden_claims_absent": True,
            "clean_rows": [
                "exp2953_threshold_policy",
                "exp2955_gatemate_constraints_materialized",
                "exp2956_gatemate_bitstream_built",
                "exp2958_polarfire_1000_clause_hash_verified",
            ],
            "flagged_rows": [
                "exp2950_repair_prompt_manifest",
                "exp2952_structured_repair_delta",
                "exp2954_self_learning_utility",
                "exp2959_nl_to_z3_execution_repair",
            ],
            "blocked_rows": ["exp2957_gatemate_flash_smoke"],
            "gated_skipped_rows": [],
            "pilot_only_rows": [],
            "aggregation_only_rows": [
                "exp2943_matrix_v11_carry_forward",
                "exp2948_capstone_v277_carry_forward",
            ],
            "code_repair_delta_summary": {
                "pass_at_1_delta": 0.25,
                "pass_at_k_delta": 0.25,
                "artifact_flagged": True,
            },
            "self_learning_delta_summary": {
                "artifact_ready": True,
                "heldout_utility_delta": 0.111859239286,
                "forgetting_guard_passed": True,
                "artifact_flagged": True,
            },
            "hardware_state_summary": {
                "gatemate": {
                    "constraints_ready": True,
                    "bitstream_built": True,
                    "flash_state": "blocked_board_not_detected",
                    "flash_succeeded": False,
                },
                "polarfire": {
                    "board_reachable": True,
                    "clause_count": 1000,
                    "hash_verified": True,
                },
            },
            "solver_state_summary": {
                "z3_execution_repaired": True,
                "solver_verified_accuracy": 0.0,
                "artifact_flagged": True,
            },
        },
    )


def test_req_report_2961_spec_anchor_exists() -> None:
    """REQ-REPORT-2961: OpenSpec declares the capstone contract first."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    assert "REQ-REPORT-2961" in spec
    assert "SCENARIO-REPORT-2961" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_report_2961_builds_capstone_from_available_278_artifacts(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-2961: .278 closes without fabricating missing branches."""

    _write_ready_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=13.25)

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete: milestone_278_capstone")
    assert artifact["milestone"] == "2026.05.278"
    assert artifact["paper_ready"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == pytest.approx(3.25)
    assert artifact["forbidden_claims_absent"] is True

    assert artifact["clean_artifacts"] == ["exp2953", "exp2955", "exp2956", "exp2958"]
    assert artifact["flagged_artifacts"] == [
        "exp2950",
        "exp2951",
        "exp2952",
        "exp2954",
        "exp2959",
    ]
    assert artifact["blocked_artifacts"] == ["exp2957"]
    assert artifact["gated_skipped_artifacts"] == []
    assert artifact["missing_artifacts"] == ["exp2949"]
    assert artifact["pilot_only_artifacts"] == []
    assert artifact["aggregation_only_artifacts"] == ["exp2960", "exp2961"]

    assert artifact["artifact_classification_counts"] == {
        "aggregation-only": 2,
        "blocked": 1,
        "clean": 4,
        "flagged": 5,
        "gated-skipped": 0,
        "missing": 1,
        "pilot-only": 0,
    }
    assert {row["task_id"] for row in artifact["classification_details"]} == {
        f"exp{number}" for number in range(2949, 2962)
    }
    assert {
        (row["task_id"], row["classification"]) for row in artifact["classification_details"]
    } >= {
        ("exp2949", "missing"),
        ("exp2957", "blocked"),
        ("exp2960", "aggregation-only"),
        ("exp2961", "aggregation-only"),
    }

    code = artifact["outcome_summaries"]["code_repair"]
    assert code["pass_at_1_delta"] == pytest.approx(0.25)
    assert code["safe_claim"] == "pilot_delta_flagged_not_paper_ready"
    assert code["threshold_policy_ready"] is True

    self_learning = artifact["outcome_summaries"]["self_learning"]
    assert self_learning["heldout_utility_delta"] == pytest.approx(0.111859239286)
    assert self_learning["safe_claim"] == "positive_utility_flagged_not_paper_ready"

    hardware = artifact["outcome_summaries"]["hardware"]
    assert hardware["gatemate"]["constraints_ready"] is True
    assert hardware["gatemate"]["bitstream_built"] is True
    assert hardware["gatemate"]["flash_state"] == "blocked_board_not_detected"
    assert hardware["polarfire"]["hash_verified"] is True

    solver = artifact["outcome_summaries"]["solver"]
    assert solver["z3_execution_repaired"] is True
    assert solver["safe_claim"] == "z3_execution_repaired_but_flagged_not_paper_ready"

    assert artifact["gaps_closed"] == [
        "Code verifier threshold policy is clean and deployment-bounded.",
        "GateMate n=16 constraints and bitstream materialization landed.",
        "PolarFire 1000-clause scorer hash verification landed.",
    ]
    assert any(
        "Taxonomy-guided repair delta remains flagged" in gap for gap in artifact["gaps_remaining"]
    )
    assert any(
        "GateMate flash/timing smoke remains blocked" in gap for gap in artifact["gaps_remaining"]
    )
    assert 2 <= len(artifact["next_milestone_recommendations"]) <= 4
    assert "GateMate" in artifact["next_milestone_recommendations"][0]

    safe_text = json.dumps(artifact["paper_v6_safe_claims"], sort_keys=True)
    assert "speedup" not in safe_text.lower()
    assert "thermalization" not in safe_text.lower()


def test_req_report_2961_gated_skip_is_not_fabricated_missing(tmp_path: Path) -> None:
    """REQ-REPORT-2961: unmet gates classify absent downstream tasks as gated-skipped."""

    _write_roadmap(tmp_path)
    _write_json(
        tmp_path,
        mod.EXP2955_REL_PATH,
        {
            "honest_verdict": "blocked_constraints_not_ready",
            "gatemate_constraints_ready": False,
        },
    )

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert "exp2956" in artifact["gated_skipped_artifacts"]
    assert "exp2956" not in artifact["missing_artifacts"]
    assert any(
        row["task_id"] == "exp2956"
        and row["gate_blocked_by"] == ["exp2955.gatemate_constraints_ready"]
        for row in artifact["classification_details"]
    )


def test_req_report_2961_blocks_paper_ready_when_matrix_forbidden_claims_fail(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-2961: paper readiness follows matrix and forbidden-claim gates."""

    _write_ready_sources(tmp_path)
    _write_json(
        tmp_path,
        mod.EXP2960_REL_PATH,
        {
            "honest_verdict": "complete: matrix_v12_ready=true",
            "matrix_v12_ready": True,
            "forbidden_claims_absent": False,
        },
    )

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.25)

    assert artifact["paper_ready"] is False
    assert artifact["forbidden_claims_absent"] is False
    assert "Forbidden claim scan failed in matrix v12." in artifact["gaps_remaining"]


def test_req_report_2961_write_artifact_persists_stable_json(tmp_path: Path) -> None:
    """REQ-REPORT-2961: write_artifact emits the required deliverable JSON."""

    _write_ready_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=3.0, now_s=4.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["milestone"] == "2026.05.278"
    assert saved["duration_s"] == pytest.approx(1.0)
    assert saved["source_checksums"][mod.EXP2952_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.EXP2952_REL_PATH
    )


def test_req_report_2961_helper_edges_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-2961: helper edges preserve malformed and nonnumeric inputs."""

    missing = tmp_path / "missing.json"
    malformed = tmp_path / "malformed.json"
    list_payload = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_payload.write_text("[1, 2, 3]\n", encoding="utf-8")

    assert mod.read_json_object(missing) == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_payload) == {}
    assert mod.sha256_file(missing) is None
    assert mod._coerce_float(True) is None
    assert mod._coerce_float("not-a-number") is None
    assert mod._coerce_int(False) is None
    assert mod._coerce_int("not-a-number") is None
    assert mod._get_path({"a": 1}, "a.b") is None
    assert mod._flag_kinds({"corrigendum_pending": "not-list"}) == []
    assert mod._blocked_verdict("gate_blocked_precondition") is True
    assert mod._blocked_verdict("complete: ok") is False
    (tmp_path / mod.ROADMAP_REL_PATH).write_text(
        """
tasks:
  - ignored
  - id: note123
    deliverable: "results/ignored.json"
  - id: exp2959
    deliverable: "results/pilot.json"
""".lstrip(),
        encoding="utf-8",
    )
    assert [task.task_id for task in mod._load_tasks(tmp_path)] == ["exp2959"]
    pilot_task = mod.TaskSpec(
        task_id="exp2959",
        title="Pilot",
        deliverable=Path("results/pilot.json"),
        inference_substrate="aggregation_from_upstream_artifacts",
        gated_on=(),
    )
    assert mod._classify_task(pilot_task, {"pilot_only": True}, True, []) == "pilot-only"
