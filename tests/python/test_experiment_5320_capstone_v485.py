"""Tests for REQ-CAPSTONE-5320 / SCENARIO-CAPSTONE-5320."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from shutil import copyfile
import sys

import pytest


REPO = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO / "python/carnot/experiment_5320_capstone_v485.py"
SPEC = importlib.util.spec_from_file_location("experiment_5320_capstone_v485_under_test", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
exp = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = exp
SPEC.loader.exec_module(exp)
SPEC_PATH = REPO / "openspec/capabilities/capstone/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_capstone_5320_spec_declares_capstone_contract() -> None:
    """REQ-CAPSTONE-5320: OpenSpec anchors artifact fields and no-laundering rules."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5320") :]
    normalized_section = " ".join(section.split())

    for marker in (
        "REQ-CAPSTONE-5320",
        "SCENARIO-CAPSTONE-5320",
        str(exp.RESULT_RELATIVE_PATH),
        exp.INFERENCE_SUBSTRATE,
        "Exp5309 as local GGUF runtime still blocked",
        "Exp5311 as gate-blocked quality",
        "Exp5318 as a deterministic SMT protocol that is flagged adversarial",
        "reachability-only hardware continuity",
        "no_false_speedup_claim=true",
        "no_false_sota_quality_claim=true",
    ):
        assert marker in section or marker in normalized_section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized_section


def test_scenario_capstone_5320_builds_honest_milestone_aggregate() -> None:
    """SCENARIO-CAPSTONE-5320: aggregation preserves blocked, flagged, and null evidence."""

    artifact = exp.build_result_artifact(root=REPO)

    assert artifact["experiment_id"]["value"] == exp.EXPERIMENT_ID
    assert artifact["milestone"]["value"] == exp.MILESTONE
    assert artifact["status"]["value"] == "complete"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE
    assert artifact["no_false_speedup_claim"] is True
    assert artifact["no_false_sota_quality_claim"] is True
    assert artifact["docs_updated"]["value"] == {
        "ops_status": False,
        "ops_changelog": False,
        "traceability": False,
    }

    artifacts_read = artifact["artifacts_read"]["value"]
    assert len(artifacts_read) == len(exp.EXPECTED_ARTIFACTS)
    assert artifact["missing_artifacts"]["value"] == []
    assert {row["experiment_number"] for row in artifacts_read} == set(range(5307, 5320))
    assert all(row["sha256"].startswith("sha256:") for row in artifacts_read)

    runtime = artifact["sota_runtime_status"]["value"]
    assert runtime["source_experiment"] == 5309
    assert runtime["sota_runtime_unblocked"] is False
    assert runtime["no_quality_claim"] is True
    assert runtime["offload_authenticated_model_count"] == 3
    assert runtime["completed_model_count"] == 0
    assert runtime["timeout_classes"] == {
        "flagship_dense": "generation_incomplete",
        "flagship_moe": "generation_incomplete",
        "middle_moe": "generation_incomplete",
    }

    quality = artifact["sota_quality_status"]["value"]
    assert quality["source_experiment"] == 5311
    assert quality["quality_measured"] is False
    assert quality["gate_blocked"] is True
    assert quality["failed_gate_upstream"] == "exp5309-sota-runtime-timeout-rootcause-matrix-v485"

    paraphrase = artifact["paraphrase_verification_status"]["value"]
    assert paraphrase == {
        "source_experiment": 5310,
        "paraphrase_fixture_ready": True,
        "paraphrase_group_count": 4,
        "label_preservation_pass_rate": 1.0,
        "contradiction_violation_caught_rate": 1.0,
        "invalid_premise_handled": True,
    }

    learning = artifact["continuous_self_learning_status"]["value"]
    assert learning["verifier_ready"] is True
    assert learning["rollout_complete"] is True
    assert learning["safe_transition_commits"] == [4, 4]
    assert learning["unsafe_transition_rejections"] == [4, 4]
    assert learning["quality_delta_vs_always_full"] == 0.0
    assert learning["transition_score_delta_vs_always_full"] == 0.0
    assert learning["full_verifier_calls_avoided"] == 3
    assert learning["unsafe_false_accepts"] == 0
    assert learning["rollback_events"] == 1
    assert learning["no_weight_mutation"] is True

    solver = artifact["solver_status"]["value"]
    assert solver["smooth_relaxation_ready"] is True
    assert solver["solver_guidance_ablation_complete"] is True
    assert solver["aggregate_conflict_delta"] == 9
    assert solver["cdcl_fallback_authoritative"] is True
    assert solver["misleading_class_blocked"] is True
    assert solver["no_hardware_speedup_claim"] is True

    kan = artifact["kan_certificate_status"]["value"]
    assert kan["kan_optimal_abstraction_ready"] is True
    assert kan["certificate_success_delta"] == 0.0
    assert kan["certificate_success_improved"] is False
    assert kan["envelope_gap_delta"] > 0.0
    assert kan["false_property_rejection_rate"] == 1.0
    assert kan["bounded_fixture_only"] is True

    ebt = artifact["ebt_telemetry_status"]["value"]
    assert ebt["ebt_telemetry_audited"] is True
    assert ebt["methodology_flag_cleared"] is True
    assert ebt["tiny_diagnostic_usable"] is True
    assert ebt["future_energy_descent_claims_eligible"] is False
    assert ebt["sota_quality_claims_eligible"] is False
    assert ebt["hardware_readiness_claims_eligible"] is False

    smt = artifact["smt_hint_protocol_status"]["value"]
    assert smt["smt_hint_protocol_ready"] is True
    assert smt["flagged_adversarial"] is True
    assert smt["clean_success_evidence"] is False
    assert smt["llm_invoked"] is False
    assert smt["future_llm_slot_gated_on_sota_runtime"] is True
    assert smt["unsound_hint_rejection_rate"] == 1.0

    hardware = artifact["hardware_status"]["value"]
    assert hardware["hardware_speedup_claimed"] is False
    assert hardware["no_speedup_claim"] is True
    assert hardware["authenticated_workload_run"] is False
    assert hardware["kv260_ssh_reachable"] is False
    assert hardware["polarfire_status_reachable"] is True
    assert hardware["gatemate_physical_jtag_changed"] is False

    recommendation = artifact["next_milestone_recommendation"]["value"]
    assert recommendation["recommended_branch"] == exp.NEXT_MILESTONE_BRANCH
    assert recommendation["primary_gate"] == "repair_sota_runtime_before_quality_claims"
    assert recommendation["do_not_reopen"] == [
        "hardware_speedup_without_authenticated_workload",
        "sota_quality_claim_until_exp5309_gate_passes",
        "clean_smt_success_from_flagged_exp5318",
    ]

    exp.validate_artifact(artifact)


def test_scenario_capstone_5320_run_writes_stable_artifact(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5320: run() writes the deterministic result artifact."""

    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(root=REPO, result_path=result_path)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact == exp.build_result_artifact(root=REPO)
    exp.validate_artifact(artifact)


def test_scenario_capstone_5320_missing_artifact_blocks_without_false_claims(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5320-BLOCKED-MISSING-INPUT: missing inputs fail closed."""

    for source in exp.EXPECTED_ARTIFACTS:
        destination = tmp_path / source.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(REPO / source.relative_path, destination)
    (tmp_path / exp.EXP5318.relative_path).unlink()

    artifact = exp.build_result_artifact(root=tmp_path)

    assert artifact["status"]["value"] == "blocked_missing_required"
    assert artifact["honest_verdict"]["value"].startswith("blocked_missing_required")
    assert artifact["no_false_speedup_claim"] is True
    assert artifact["no_false_sota_quality_claim"] is True
    assert artifact["docs_updated"]["value"] == {
        "ops_status": False,
        "ops_changelog": False,
        "traceability": False,
    }
    assert artifact["missing_artifacts"]["value"] == [
        {
            "experiment_number": 5318,
            "path": str(exp.EXP5318.relative_path),
            "reason": "missing",
        }
    ]
    assert artifact["smt_hint_protocol_status"]["value"]["status"] == "missing_or_unreadable"
    exp.validate_artifact(artifact)


def test_scenario_capstone_5320_malformed_and_missing_summaries_are_explicit(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5320-BLOCKED-MISSING-INPUT: malformed inputs are not hidden."""

    malformed = tmp_path / exp.EXP5307.relative_path
    malformed.parent.mkdir(parents=True, exist_ok=True)
    malformed.write_text("{", encoding="utf-8")

    payloads, artifacts_read, missing = exp.read_upstream_artifacts(tmp_path)

    assert payloads == {}
    assert artifacts_read == []
    assert missing[0] == {
        "experiment_number": 5307,
        "path": str(exp.EXP5307.relative_path),
        "reason": "malformed_json:Expecting property name enclosed in double quotes",
    }

    assert exp.summarize_sota_runtime(None) == {
        "source_experiment": 5309,
        "status": "missing_or_unreadable",
    }
    assert exp.summarize_sota_quality(None) == {
        "source_experiment": 5311,
        "status": "missing_or_unreadable",
    }
    assert exp.summarize_paraphrase(None) == {
        "source_experiment": 5310,
        "status": "missing_or_unreadable",
    }
    assert exp.summarize_continuous_self_learning(None, {}) == {
        "source_experiments": [5312, 5313],
        "status": "missing_or_unreadable",
        "missing_sources": [5312],
    }
    assert exp.summarize_solver({}, None) == {
        "source_experiments": [5314, 5315],
        "status": "missing_or_unreadable",
        "missing_sources": [5315],
    }
    assert exp.summarize_kan(None) == {
        "source_experiment": 5316,
        "status": "missing_or_unreadable",
    }
    assert exp.summarize_ebt(None) == {
        "source_experiment": 5317,
        "status": "missing_or_unreadable",
    }
    assert exp.summarize_smt(None) == {
        "source_experiment": 5318,
        "status": "missing_or_unreadable",
    }
    assert exp.summarize_hardware(None) == {
        "source_experiment": 5319,
        "status": "missing_or_unreadable",
    }


def test_req_capstone_5320_validation_rejects_schema_drift() -> None:
    """REQ-CAPSTONE-5320: validator rejects missing principles and false claims."""

    artifact = exp.build_result_artifact(root=REPO)

    missing_field = deepcopy(artifact)
    missing_field.pop("hardware_status")
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(missing_field)

    bad_wrapped = deepcopy(artifact)
    bad_wrapped["sota_runtime_status"] = bad_wrapped["sota_runtime_status"]["value"]
    with pytest.raises(ValueError, match="sota_runtime_status"):
        exp.validate_artifact(bad_wrapped)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"]["value"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(bad_verdict)

    bad_speedup = deepcopy(artifact)
    bad_speedup["no_false_speedup_claim"] = {"value": True}
    with pytest.raises(ValueError, match="no_false_speedup_claim"):
        exp.validate_artifact(bad_speedup)

    bad_quality = deepcopy(artifact)
    bad_quality["no_false_sota_quality_claim"] = False
    with pytest.raises(ValueError, match="no_false_sota_quality_claim"):
        exp.validate_artifact(bad_quality)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"]["value"] = "aggregation_from_upstream_artifacts"
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    bad_docs = deepcopy(artifact)
    bad_docs["docs_updated"]["value"]["ops_status"] = True
    with pytest.raises(ValueError, match="docs_updated"):
        exp.validate_artifact(bad_docs)
