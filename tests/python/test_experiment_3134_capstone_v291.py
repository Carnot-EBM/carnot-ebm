"""Tests for the Exp 3134 milestone .291 capstone.

Spec refs: REQ-REPORT-3134, SCENARIO-REPORT-3134.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v291_3134 as mod


REQUIRED_FIELDS = {
    "capstone_ready",
    "paper_ready",
    "publication_blocker_count",
    "blocker_delta_from_v24",
    "next_top_gap",
    "sota_cache_status",
    "verifier_claim_status",
    "repair_claim_status",
    "fr11_self_learning_status",
    "ebt_arm_status",
    "kan_status",
    "sampler_hardware_status",
    "next_recommendation",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(row_id: str, status: str, claim_scope: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "status": status,
        "source_artifact": f"results/{row_id.replace(':', '_')}.json",
        "source_field": "status",
        "evidence_class": "test_evidence",
        "blocker_class": status if status != "clean" else "none",
        "claim_scope": claim_scope,
        "summary": summary,
        "row_origin": "milestone_291_test",
    }


def _source_artifacts() -> list[dict[str, Any]]:
    specs = [
        ("exp3122", mod.EXP3122_REL_PATH, "archive_v290_activate_v291"),
        ("exp3123", mod.EXP3123_REL_PATH, "sota_cache_coverage"),
        ("exp3124", mod.EXP3124_REL_PATH, "live_verifier_lift"),
        ("exp3125", mod.EXP3125_REL_PATH, "prefix_closed_bounds"),
        ("exp3126", mod.EXP3126_REL_PATH, "fragment_time_monitors"),
        ("exp3127", mod.EXP3127_REL_PATH, "repair_ladder"),
        ("exp3128", mod.EXP3128_REL_PATH, "fr11_evoenv"),
        ("exp3129", mod.EXP3129_REL_PATH, "fr11_constraint_memory"),
        ("exp3130", mod.EXP3130_REL_PATH, "arm_ebt_energy_budget"),
        ("exp3131", mod.EXP3131_REL_PATH, "kan_pwa_milp"),
        ("exp3132", mod.EXP3132_REL_PATH, "hardware_sampler_boundary"),
    ]
    return [
        {
            "experiment_id": exp_id,
            "path": path.as_posix(),
            "role": role,
            "required": False,
            "present": True,
            "readable_json_object": True,
            "ready_field": "ready",
        }
        for exp_id, path, role in specs
    ]


def _publication_blockers() -> list[dict[str, Any]]:
    named = [
        ("dot291:exp3123_sota_cache_coverage", "bounded", "local_sota_model_cache_policy"),
        ("dot291:exp3124_live_verifier_lift", "blocked", "live_sota_verifier_lift"),
        ("dot291:exp3125_prefix_bounds", "bounded", "bounded_prefix_correctness"),
        ("dot291:exp3126_fragment_time_monitors", "bounded", "fragment_time_monitor_boundary"),
        ("dot291:exp3127_repair_ladder", "blocked", "repair_live_rerun"),
        ("dot291:exp3128_fr11_evoenv", "bounded", "controller_only_environment_synthesis"),
        ("dot291:exp3129_fr11_memory", "bounded", "controller_only_constraint_memory"),
        ("dot291:exp3130_arm_ebt_energy_budget", "projection_only", "architecture_energy_budget_boundary"),
        ("dot291:exp3131_kan_pwa_milp", "bounded", "architecture_kan_verifier_boundary"),
        ("dot291:exp3132_hardware_sampler_boundary", "blocked", "architecture_hardware_sampler_boundary"),
    ]
    blockers = [
        {
            "row_id": row_id,
            "status": status,
            "blocker_class": status,
            "source_artifact": f"results/{row_id.replace(':', '_')}.json",
            "source_field": "status",
            "claim_scope": scope,
        }
        for row_id, status, scope in named
    ]
    blockers.extend(
        {
            "row_id": f"carry:blocker:{idx}",
            "status": "bounded",
            "blocker_class": "bounded",
            "source_artifact": "results/carry.json",
            "source_field": "status",
            "claim_scope": "prior_carry_forward",
        }
        for idx in range(36)
    )
    return blockers


def _matrix_v25(*, ready: bool = True, blockers: int = 46, delta: int = 10) -> dict[str, Any]:
    rows = [
        _row(
            "dot291:exp3123_sota_cache_coverage",
            "bounded" if blockers else "clean",
            "local_sota_model_cache_policy",
            {"cached_sota_pair_available": False, "missing_model_ids": ["qwen", "gemma31"]},
        ),
        _row(
            "dot291:exp3124_live_verifier_lift",
            "blocked" if blockers else "clean",
            "live_sota_verifier_lift",
            {"false_accept_rate": 0.5 if blockers else 0.0, "repair_gate_state": "blocked_false_accept"},
        ),
        _row(
            "dot291:exp3125_prefix_bounds",
            "bounded" if blockers else "clean",
            "bounded_prefix_correctness",
            {"explored_prefix_count": 453, "accepted_prefix_count": 2},
        ),
        _row(
            "dot291:exp3126_fragment_time_monitors",
            "bounded" if blockers else "clean",
            "fragment_time_monitor_boundary",
            {"monitor_violation_count": 2, "ledger_consistency_rate": 0.666667},
        ),
        _row(
            "dot291:exp3127_repair_ladder",
            "blocked" if blockers else "clean",
            "repair_live_rerun",
            {"blocked_at_layer": "conductor_pre_gate", "gate_check_summary": "failed repair gate"},
        ),
        _row(
            "dot291:exp3128_fr11_evoenv",
            "bounded" if blockers else "clean",
            "controller_only_environment_synthesis",
            {"admitted_environment_count": 3, "no_weight_update_claim": True},
        ),
        _row(
            "dot291:exp3129_fr11_memory",
            "bounded" if blockers else "clean",
            "controller_only_constraint_memory",
            {"ledger_consistency_rate": 0.666667, "no_weight_update_claim": True},
        ),
        _row(
            "dot291:exp3130_arm_ebt_energy_budget",
            "projection_only" if blockers else "clean",
            "architecture_energy_budget_boundary",
            {"live_integration": False, "integration_blocker_count": 6},
        ),
        _row(
            "dot291:exp3131_kan_pwa_milp",
            "bounded" if blockers else "clean",
            "architecture_kan_verifier_boundary",
            {"claim_boundary_does_not_prove": ["deployed verifier improvement"]},
        ),
        _row(
            "dot291:exp3132_hardware_sampler_boundary",
            "blocked" if blockers else "clean",
            "architecture_hardware_sampler_boundary",
            {
                "hardware_commands_run": [],
                "speedup_claim_allowed": False,
                "gatemate_evidence_complete": False,
                "ssqa_readback_ready": False,
                "sampler_boundary_decisions": {"clut": "CPU simulation"},
                "missing_operator_evidence_count": 7,
            },
        ),
        _row(
            "dot290:exp3118_clut_sampler_backend_integration",
            "bounded" if blockers else "clean",
            "cpu_backend_no_hardware_speedup",
            {"hardware_claim_made": False},
        ),
    ]
    return {
        "artifact": "experiment_3133_cross_corpus_matrix_v25",
        "matrix_v25_ready": ready,
        "rows_total": 113 if blockers else len(rows),
        "prior_publication_blocker_count": 36,
        "publication_blocker_count": blockers,
        "blocker_delta_from_v24": delta,
        "status_counts": {
            "blocked": 8 if blockers else 0,
            "bounded": 19 if blockers else 0,
            "clean": 32 if blockers else len(rows),
            "diagnostic_only": 4 if blockers else 0,
            "flagged": 7 if blockers else 0,
            "gated_skipped": 7 if blockers else 0,
            "missing": 2 if blockers else 0,
            "model_spec_gap": 0,
            "projection_only": 3 if blockers else 0,
            "retired": 31 if blockers else 0,
        },
        "publication_blockers": _publication_blockers() if blockers else [],
        "rows": rows,
        "missing_artifacts": [],
        "headline_claim_allowance_summary": {
            "cached_sota_pair_available": False,
            "comparative_sota_pair_allowed": False,
            "live_verifier_headline_allowed": False if blockers else True,
            "blocked_headline_claims": ["comparative_sota_pair", "live_verifier_lift"]
            if blockers
            else [],
            "false_accept_rate": 0.5 if blockers else 0.0,
            "present_model_ids": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "missing_model_ids": ["unsloth/Qwen3.6-35B-A3B-GGUF", "unsloth/gemma-4-31B-it-GGUF"],
        },
        "verifier_repair_summary": {
            "sota_cache_status": "bounded" if blockers else "clean",
            "live_verifier_status": "blocked" if blockers else "clean",
            "prefix_bounds_status": "bounded" if blockers else "clean",
            "fragment_time_monitor_status": "bounded" if blockers else "clean",
            "repair_ladder_status": "blocked" if blockers else "clean",
            "repair_gate_state": "blocked_false_accept" if blockers else "unblocked",
            "false_accept_rate": 0.5 if blockers else 0.0,
            "verifier_gain_delta": 0.0 if blockers else 0.2,
            "repair_ladder_blocked_at_layer": "conductor_pre_gate" if blockers else "",
            "repair_ladder_gate_check_summary": "failed repair gate" if blockers else "",
        },
        "fr11_summary": {
            "evoenv_status": "bounded" if blockers else "clean",
            "memory_status": "bounded" if blockers else "clean",
            "continuous_self_learning_targeted": True,
            "admitted_environment_count": 3,
            "no_weight_update_claim": True if blockers else False,
            "model_weight_learning_allowed": False if blockers else True,
            "ledger_consistency_rate": 0.666667 if blockers else 1.0,
            "soundness_errors": 0,
            "completeness_errors": 0,
            "forgetting_regression_count": 0,
            "promotion_recommendation": "promote_controller_environment_memory_only_block_model_weight_learning_until_ledger_consistency_is_1.0"
            if blockers
            else "promote",
        },
        "architecture_boundary_summary": {
            "arm_ebt_status": "projection_only" if blockers else "clean",
            "kan_pwa_milp_status": "bounded" if blockers else "clean",
            "hardware_sampler_status": "blocked" if blockers else "clean",
            "live_integration": False if blockers else True,
            "integration_blocker_count": 6 if blockers else 0,
            "speedup_claim_allowed": False if blockers else True,
            "hardware_commands_run": [],
            "gatemate_evidence_complete": False if blockers else True,
            "ssqa_readback_ready": False if blockers else True,
            "missing_operator_evidence_count": 7 if blockers else 0,
        },
        "source_artifacts": _source_artifacts(),
        "inference_substrate": {
            "kind": "aggregation_from_checked_in_dot291_artifacts",
            "executes_models": False,
            "executes_verifiers": False,
            "executes_repairs": False,
            "executes_solvers": False,
            "executes_hardware": False,
            "executes_conductor": False,
            "no_live_llm_inference": True,
        },
        "honest_verdict": "complete: matrix_v25_ready=true",
    }


def _write_matrix_and_sources(root: Path, matrix: dict[str, Any]) -> None:
    _write_json(root, mod.MATRIX_V25_REL_PATH, matrix)
    for source in matrix["source_artifacts"]:
        _write_json(
            root,
            source["path"],
            {"artifact": source["experiment_id"], "ready": True, "honest_verdict": "complete: ok"},
        )


def test_req_report_3134_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3134: OpenSpec declares the .291 capstone contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3134" in spec
    assert "SCENARIO-REPORT-3134" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3134_closes_from_matrix_v25_evidence(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3134: .291 capstone preserves blocked and bounded evidence."""

    matrix = _matrix_v25()
    _write_matrix_and_sources(tmp_path, matrix)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=14.25)
    sources = {row["path"]: row for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["capstone_ready"] is True
    assert artifact["paper_ready"] is False
    assert artifact["publication_blocker_count"] == 46
    assert artifact["blocker_delta_from_v24"] == 10
    assert artifact["next_top_gap"] == "live_verifier_false_accept_repair_gate"
    assert artifact["duration_s"] == pytest.approx(4.25)
    assert artifact["honest_verdict"].startswith("complete:")

    assert artifact["sota_cache_status"] == "bounded_missing_comparative_sota_pair"
    assert artifact["verifier_claim_status"] == "blocked_false_accept_rate_0.5_no_headline_lift"
    assert artifact["prefix_bounds_status"] == "bounded_finite_fixture_conditioned_prefix_frontier"
    assert artifact["monitor_status"] == "bounded_fragment_monitor_ledger_consistency_0.666667"
    assert artifact["repair_claim_status"] == "blocked_repair_ladder_gate_failed_by_live_verifier_gate"
    assert (
        artifact["fr11_self_learning_status"]
        == "bounded_controller_environment_memory_only_no_weight_update_ledger_0.666667"
    )
    assert artifact["ebt_arm_status"] == "projection_only_sidecar_diagnostic_no_live_integration"
    assert artifact["kan_status"] == "bounded_pwa_milp_abstraction_no_deployed_verifier_claim"
    assert artifact["sampler_hardware_status"] == "blocked_hardware_sampler_boundary_no_speedup_claim"
    assert artifact["clut_sampler_status"] == "bounded_cpu_simulation_no_authenticated_hardware_execution"
    assert artifact["gatemate_status"] == "blocked_operator_evidence_incomplete"
    assert artifact["ssqa_status"] == "blocked_host_visible_readback_missing"
    assert artifact["hardware_status"] == "blocked_no_commands_no_speedup_claim"

    assert artifact["paper_readiness_assessment"] == "not_closer_blockers_increased_by_10"
    assert artifact["paper_readiness_checks"][1]["reason"] == "publication_blocker_count=46"
    assert "matrix v25 is complete" in artifact["what_291_proved"][0]
    assert "live_verifier_headline_lift" in artifact["what_stayed_blocked"]
    assert "bounded_prefix_frontier_only" in artifact["bounded_claims"]
    assert "single_cached_gemma26_available" in artifact["allowed_claims"]
    assert "false-accept" in artifact["next_recommendation"]
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_checked_in_matrix_v25_and_dot291_artifacts",
        "source": "results/experiment_3133_cross_corpus_matrix_v25.json",
        "executes_models": False,
        "executes_verifiers": False,
        "executes_repairs": False,
        "executes_solvers": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "no_live_llm_inference": True,
        "local_repo_only": True,
        "live_model_calls_run_by_capstone": 0,
        "hardware_commands_run_by_capstone": [],
    }
    assert sources[mod.MATRIX_V25_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.MATRIX_V25_REL_PATH
    )


def test_req_report_3134_paper_ready_requires_zero_blockers(tmp_path: Path) -> None:
    """REQ-REPORT-3134: capstone completion alone never implies paper readiness."""

    matrix = _matrix_v25(blockers=0, delta=-46)
    _write_matrix_and_sources(tmp_path, matrix)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["capstone_ready"] is True
    assert artifact["paper_ready"] is True
    assert artifact["publication_blocker_count"] == 0
    assert artifact["blocker_delta_from_v24"] == -46
    assert artifact["paper_readiness_assessment"] == "closer_blockers_cleared"
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_report_3134_blocks_missing_or_unready_matrix(tmp_path: Path) -> None:
    """REQ-REPORT-3134: missing or unready matrix authority blocks capstone completion."""

    missing = mod.build_artifact(tmp_path)
    assert missing["capstone_ready"] is False
    assert missing["honest_verdict"].startswith("blocked:")
    assert "required source unreadable" in missing["invariant_violations"][0]

    matrix = _matrix_v25(ready=False)
    _write_matrix_and_sources(tmp_path, matrix)
    unready = mod.build_artifact(tmp_path)
    assert unready["capstone_ready"] is False
    assert "matrix_v25_ready is not true" in unready["invariant_violations"]


def test_req_report_3134_write_artifact_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3134: persistence and malformed helper edges stay deterministic."""

    _write_matrix_and_sources(tmp_path, _matrix_v25())
    malformed = tmp_path / "bad.json"
    list_json = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_json.write_text("[1]\n", encoding="utf-8")

    output = mod.write_artifact(tmp_path, started_s=2.0, now_s=3.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["artifact"] == mod.ARTIFACT
    assert saved["duration_s"] == pytest.approx(1.0)
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod._mapping({"x": 1}) == {"x": 1}
    assert mod._mapping(None) == {}
    assert mod._list([1]) == [1]
    assert mod._list("x") == []
    assert mod._int(True) == 0
    assert mod._int("bad") == 0
    assert mod._float(True) == 0.0
    assert mod._float("bad") == 0.0
    assert mod._status_by_row([], "missing") == "missing"
    assert mod._summary_by_row([], "missing") == {}
    assert mod._source_role(mod.MATRIX_V25_REL_PATH) == "matrix_v25_authority"
    assert mod._source_role(mod.EXP3124_REL_PATH) == "live_verifier_lift"
    assert mod._source_role(Path("results/unknown.json")) == "matrix_v25_source"
    assert mod._invariant_violations({"matrix_v25_ready": True}, [], []) == [
        "source_artifacts list is empty"
    ]
    assert (
        mod._sota_cache_status({"sota_cache_status": "bounded"}, {"comparative_sota_pair_allowed": True})
        == "bounded_sota_cache_not_promoted"
    )
    assert (
        mod._clut_sampler_status(
            [
                _row(
                    "dot290:exp3118_clut_sampler_backend_integration",
                    "clean",
                    "clut",
                    {},
                ),
                _row(
                    "dot291:exp3132_hardware_sampler_boundary",
                    "clean",
                    "hardware",
                    {"sampler_boundary_decisions": {"clut": "authenticated hardware"}},
                ),
            ]
        )
        == "clean_authenticated_clut_sampler"
    )
    assert (
        mod._hardware_status({"hardware_commands_run": ["flash"], "speedup_claim_allowed": True})
        == "clean_authenticated_hardware_speedup_claim_allowed"
    )
    assert (
        mod._paper_readiness_assessment(True, 5, -3)
        == "closer_blockers_reduced_by_3_but_not_ready"
    )
    assert mod._paper_readiness_assessment(True, 5, 0) == "not_closer_blockers_unchanged"
    clean_verifier = {"live_verifier_status": "clean", "repair_ladder_status": "clean"}
    assert (
        mod._next_top_gap(clean_verifier, {"comparative_sota_pair_allowed": False}, {})
        == "comparative_sota_cache_pair"
    )
    assert (
        mod._next_top_gap(
            clean_verifier,
            {"comparative_sota_pair_allowed": True},
            {"speedup_claim_allowed": False},
        )
        == "operator_authenticated_hardware_readback"
    )
    assert (
        mod._next_top_gap(
            clean_verifier,
            {"comparative_sota_pair_allowed": True},
            {"speedup_claim_allowed": True},
        )
        == "publication_scope_reconciliation"
    )

    edge_matrix = _matrix_v25()
    edge_matrix["status_counts"]["clean"] = 0
    edge_matrix["publication_blocker_count"] = 45
    edge_matrix["missing_artifacts"] = [{"path": "missing"}]
    edge_matrix["inference_substrate"]["executes_models"] = True
    _write_matrix_and_sources(tmp_path, edge_matrix)
    edge = mod.build_artifact(tmp_path)

    assert "status_counts do not reconcile with rows_total" in edge["invariant_violations"]
    assert "publication_blocker_count does not match matrix publication_blockers" in edge[
        "invariant_violations"
    ]
    assert "matrix reports missing .291 artifacts" in edge["invariant_violations"]
    assert "matrix inference_substrate is not aggregation-only" in edge["invariant_violations"]
