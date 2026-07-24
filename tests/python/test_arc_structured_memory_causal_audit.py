"""Tests for Exp5901 structured memory causal audit.

Spec refs: REQ-ARC-WMTE-5901,
SCENARIO-ARC-WMTE-5901-IDENTICAL-BYTE-RETRIEVAL,
SCENARIO-ARC-WMTE-5901-CAUSAL-CONTROLS,
SCENARIO-ARC-WMTE-5901-ARTIFACT-BOUNDARY.
"""

from __future__ import annotations

import json
from pathlib import Path

import carnot.agentic.arc_structured_memory_causal_audit as audit_mod
from carnot.agentic.arc_structured_memory_causal_audit import (
    RESULT_RELATIVE_PATH,
    REQUIRED_RESULT_FIELDS,
    build_exp5901_artifact,
    run_causal_audit,
    write_exp5901_artifact,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"
RESULT_PATH = REPO / RESULT_RELATIVE_PATH


def test_req_arc_wmte_5901_spec_freezes_queries_interventions_and_schema() -> None:
    """REQ-ARC-WMTE-5901: the OpenSpec requirement names the audit contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5901") :]

    for marker in (
        "SCENARIO-ARC-WMTE-5901-IDENTICAL-BYTE-RETRIEVAL",
        "SCENARIO-ARC-WMTE-5901-CAUSAL-CONTROLS",
        "SCENARIO-ARC-WMTE-5901-ARTIFACT-BOUNDARY",
        RESULT_RELATIVE_PATH,
        "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "complete_positive:",
    ):
        assert marker in section

    for field in REQUIRED_RESULT_FIELDS:
        assert field in section


def test_scenario_arc_wmte_5901_identical_byte_retrieval_metrics() -> None:
    """SCENARIO-ARC-WMTE-5901-IDENTICAL-BYTE-RETRIEVAL: structure, not bytes, is the lever."""

    audit = run_causal_audit(REPO)
    metrics = audit["no_memory_raw_and_structured_metrics"]
    receipts = audit["identical_event_byte_receipts"]

    assert metrics["structured_index"]["exact_count"] == metrics["query_count"]
    assert metrics["no_memory"]["exact_count"] == 0
    assert metrics["structured_index"]["query_count"] == metrics["raw_tape"]["query_count"]
    assert metrics["structured_index"]["bytes_scanned"] <= metrics["raw_tape"]["bytes_scanned"]
    assert metrics["structured_over_raw_exact_delta"] > 0.0
    assert receipts["raw_and_structured_arms_deliberately_identical"] is True
    assert receipts["all_paired_cells_identical_event_bytes"] is True
    assert not receipts["violations"]

    fidelity = audit["exact_retrieval_fidelity_by_evidence_type"]
    assert set(fidelity) == {
        "action_effect",
        "object_spatial",
        "provenance",
        "temporal",
        "uncertainty",
    }
    assert all(row["structured_exact"] for row in fidelity.values())
    assert any(not row["raw_exact"] for row in fidelity.values())


def test_scenario_arc_wmte_5901_causal_interventions_and_controls() -> None:
    """SCENARIO-ARC-WMTE-5901-CAUSAL-CONTROLS: deletion and controls test causal use."""

    audit = run_causal_audit(REPO)

    deletion = audit["relevant_and_irrelevant_deletion_effects"]
    assert deletion["relevant_deletion_exact_count"] < deletion["baseline_structured_exact_count"]
    assert (
        deletion["irrelevant_deletion_exact_count"] == deletion["baseline_structured_exact_count"]
    )
    assert deletion["relevant_minus_irrelevant_utility_delta"] < 0.0
    assert deletion["promotion_requires_causal_evidence_use"] is True

    controls = audit["shuffled_stale_growth_restart_controls"]
    assert controls["shuffle_control"]["false_retrieval_count"] > 0
    assert controls["stale_evidence_control"]["stale_use_count"] == 0
    assert (
        controls["irrelevant_growth_control"]["structured_exact_count"]
        == (controls["irrelevant_growth_control"]["baseline_structured_exact_count"])
    )
    assert controls["restart_control"]["tape_hash_reproduced"] is True
    assert controls["restart_control"]["index_hash_reproduced"] is True

    accounting = audit["query_byte_latency_accounting"]
    assert accounting["budget_violations"] == []
    assert accounting["latency_budget_violations"] == []
    assert audit["false_retrieval_and_eviction_loss"]["eviction_loss"]["loss_receipt_count"] > 0


def test_scenario_arc_wmte_5901_budget_cutoff_and_unknown_query_guard() -> None:
    """SCENARIO-ARC-WMTE-5901-CAUSAL-CONTROLS: budget caps and query guards are enforced."""

    assert audit_mod._entry_matches({}, "unknown_query") is False

    memory = audit_mod._build_fixture_memory()
    answer_keys = audit_mod._derive_answer_keys(memory)
    spec = dict(audit_mod.FROZEN_QUERY_DEFINITIONS[0])
    original_budget = audit_mod.BUDGETS["max_query_bytes"]
    constrained_budget = 1

    try:
        audit_mod.BUDGETS["max_query_bytes"] = constrained_budget
        row = audit_mod._raw_query(memory, spec, answer_keys[str(spec["query_id"])])
    finally:
        audit_mod.BUDGETS["max_query_bytes"] = original_budget

    assert row["arm"] == "raw_tape"
    assert row["events_scanned"] == 1
    assert row["bytes_scanned"] > constrained_budget


def test_scenario_arc_wmte_5901_artifact_boundary_and_checksum(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-5901-ARTIFACT-BOUNDARY: artifact is complete and reproducible."""

    artifact = build_exp5901_artifact(REPO)
    for field in REQUIRED_RESULT_FIELDS:
        assert field in artifact

    assert artifact["status"] == "complete_positive"
    assert artifact["public_level_solve_claimed"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["structured_memory_causal_ready_score"] == 1.0
    assert artifact["inference_substrate"] == (
        "offline_arcade_live_agent_runtime_self_discovery_no_llm"
    )
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert (
        artifact["provenance_and_oracle_boundary"]["source_bfs_adapter_and_prior_game_access_count"]
        == 0
    )
    assert artifact["protected_files_unchanged"]["scripts/research_conductor.py"] is True

    output = tmp_path / "experiment_5901.json"
    written = write_exp5901_artifact(REPO, output_path=output)
    reread = json.loads(output.read_text(encoding="utf-8"))
    assert reread == written
    assert reread["reproducibility_checksum"] == artifact["reproducibility_checksum"]


def test_req_arc_wmte_5901_repository_artifact_is_current() -> None:
    """REQ-ARC-WMTE-5901: checked-in Exp5901 JSON matches the deterministic audit."""

    expected = build_exp5901_artifact(REPO)
    actual = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert actual == expected
