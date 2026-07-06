"""Tests for REQ-CAPSTONE-5294 / SCENARIO-CAPSTONE-5294."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5294_capstone_v483 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/capstone/spec.md"


def _wrap(value: Any, principle: str = "fixture principle") -> dict[str, Any]:
    return {"principle": principle, "value": value}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _base(verdict: str, substrate: str = mod.INFERENCE_SUBSTRATE) -> dict[str, Any]:
    return {
        "honest_verdict": _wrap(verdict),
        "inference_substrate": _wrap(substrate),
        "duration_s": 1.0,
        "reproducibility_checksum": "sha256:fixture",
    }


def _payloads() -> dict[int, dict[str, Any]]:
    return {
        5282: {
            **_base("complete: .482 archived and .483 activation-ready"),
            "activation_ready": True,
            "milestone_archived": True,
            "ops_docs_updated": _wrap(False),
            "research_complete_updated": _wrap(False),
        },
        5283: {
            **_base(
                "complete: 3 new actionable findings appended; executable .483 plan unchanged.",
                "literature_ingestion_network_sources",
            ),
            "new_references_added": _wrap(3),
            "references_md_updated": _wrap(True),
            "actionable_deltas": _wrap(
                [
                    {"title": "ConsFormer-LNS"},
                    {"title": "AS2"},
                    {"title": "EBT spectral-control companion artifact"},
                ]
            ),
        },
        5284: {
            **_base(
                "blocked_preconditions: sota_offload_ready=false "
                "flagship_moe:blocked_no_gpu_offload_evidence:offload=False",
                "blocked_preconditions_with_no_quality_claim",
            ),
            "sota_offload_ready": False,
            "sota_offload_ready_principle": "offload evidence absent",
            "MODEL_SPECS": _wrap(
                {"flagship_moe": {"runtime_status": "blocked_no_gpu_offload_evidence"}}
            ),
            "gpu_offload_receipts": _wrap(
                {"llama_cpp": {"gpu_offload_supported": False}, "gpu_visible": True}
            ),
        },
        5285: {
            **_base(
                "complete: knowledge-thought coherence fixture usable for exp5286/exp5290",
                "offline_deterministic_fixture_no_llm",
            ),
            "coherence_fixture_ready": True,
            "baseline_metrics": {
                "accuracy": 0.285714,
                "unsafe_false_accepts": 1,
                "metric": "claim_token_overlap",
            },
        },
        5286: {
            "schema": "blocked_gate_check_v1",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
        },
        5287: {
            **_base(
                "complete: trace DSL fixture usable for exp5288 solver-checked extraction",
                "offline_deterministic_fixture_no_llm",
            ),
            "trace_dsl_ready": True,
            "dsl_schema_summary": _wrap({"compiler_target": "solver_constraint_ir_v1"}),
        },
        5288: {
            "schema": "blocked_gate_check_v1",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
        },
        5289: {
            **_base(
                "complete: operation attribution is usable for Exp5290; "
                "all bounded operation-stage controls were attributed and unsafe_propagations=0"
            ),
            "memory_attribution_ready": True,
            "attribution_coverage": _wrap(
                {"attributed_cases": 7, "coverage_rate": 1.0, "total_cases": 7}
            ),
            "unsafe_propagations": _wrap({"count": 0}),
        },
        5290: {
            **_base(
                "complete: memory-assisted coherence dosing helped; governed memory "
                "preserved always-full quality, avoided 4/7 full claim/coherence checks, "
                "and kept unsafe_false_accepts=0"
            ),
            "coherence_dose_positive": True,
            "full_verifier_calls_avoided": _wrap(
                {"vs_always_full": 4, "rate_vs_always_full": 0.571429}
            ),
            "decision_quality_delta": _wrap({"governed_minus_always_full": 0.0}),
            "unsafe_false_accepts": _wrap({"count": 0}),
            "attribution_stage_contributions": _wrap(
                {"reductions_by_stage": {"use": 3}, "escalations_by_stage": {"routing": 1}}
            ),
        },
        5291: {
            **_base(
                "complete: low-order curriculum did not improve certificate success over "
                "the shuffled ordering; all bounded stages certified, so the value is "
                "measurement and factor-order telemetry",
                "offline_deterministic_certificate_no_llm",
            ),
            "low_order_curriculum_ready": True,
            "certificate_success_by_order": _wrap(
                {
                    "helped_certificate_success": False,
                    "success_advantage_over_shuffled": 0.0,
                    "all_curriculum_stages_certified": True,
                    "all_shuffled_stages_certified": True,
                }
            ),
            "claim_limits": ["bounded deterministic fixture only"],
        },
        5292: {
            **_base(
                "complete: p-bit/CDCL simulated CPU guidance helped aggregate conflicts "
                "on the bounded factor fixture while harming the misleading-assumption "
                "class; distribution sensitivity is expected",
                "offline_deterministic_certificate_no_llm",
            ),
            "pbit_cdcl_guidance_positive": True,
            "conflicts_saved": _wrap(
                {
                    "aggregate": 2,
                    "by_class": {
                        "aligned_factor_sat": 3,
                        "misleading_factor_sat": -1,
                        "neutral_factor_sat": 0,
                    },
                }
            ),
            "fallback_overwrite_count": _wrap(2),
            "correctness_preserved": _wrap(True),
            "instance_class_gate": _wrap(
                {
                    "helps": ["aligned_factor_sat"],
                    "harms": ["misleading_factor_sat"],
                    "neutral": ["neutral_factor_sat"],
                    "distribution_sensitivity_expected": True,
                }
            ),
            "hardware_speedup_claimed": _wrap(False),
        },
        5293: {
            **_base(
                "blocked_board_reachability: kv260=blocked_kv260_ssh_unreachable "
                "polfire=reachable_ssh_status_only "
                "gatemate=blocked_gatemate_physical_jtag_setup_unchanged no_speedup_claim",
                "hardware_probe_no_speedup_claim",
            ),
            "kv260_reachability": _wrap({"status": "blocked_kv260_ssh_unreachable"}),
            "polarfire_reachability": _wrap({"status": "reachable_ssh_status_only"}),
            "gatemate_reachability": _wrap(
                {"status": "blocked_gatemate_physical_jtag_setup_unchanged"}
            ),
            "hardware_speedup_claimed": _wrap(False),
            "hardware_evidence_level": _wrap("reachability_status_receipt_only"),
            "blocked_reason": _wrap({"KV260": {"reason": "blocked_kv260_ssh_unreachable"}}),
        },
    }


def _make_repo(root: Path, *, omit: set[int] | None = None, manifest: bool = True) -> None:
    omit = omit or set()
    by_number = _payloads()
    for source in mod.UPSTREAM_SOURCES:
        if source.experiment_number not in omit:
            _write_json(root / source.relative_path, by_number[source.experiment_number])
    conductor = root / mod.CONDUCTOR_LOG_PATH
    conductor.parent.mkdir(parents=True, exist_ok=True)
    conductor.write_text(
        "\n".join(
            [
                "| 2026-07-05 23:37 UTC | PHASE 0 transition -- archive .482 truth and prepa | OK | 87 passed |",
                "| 2026-07-06 01:05 UTC | PHASE 1 gated on exp5284 and exp5285 -- SOTA claim | GATE_BLOCK | exp5284 failed |",
                "| 2026-07-06 01:32 UTC | PHASE 1 gated on exp5284 and exp5287 -- SOTA trace | GATE_BLOCK | exp5284 failed |",
                "| 2026-07-06 02:54 UTC | PHASE 3 hardware continuity -- KV260, PolarFire, a | OK | 87 passed |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = root / mod.EXCLUSION_MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        (
            f"- id: {mod.SAME_VERDICT_RETIREMENT_ID}\n"
            "  retire_if_same_verdict: true\n"
            "  experiment_ids:\n"
            "  - exp5274\n"
            "  - exp5284\n"
        )
        if manifest
        else "retired: []\n",
        encoding="utf-8",
    )


def _commands() -> list[dict[str, str]]:
    return [{"command": "focused fixture", "outcome": "PASS"}]


def _value(artifact: dict[str, Any], field: str) -> Any:
    return mod.value_of(artifact[field])


def _source_ids(rows: list[dict[str, Any]]) -> set[int]:
    return {int(row["experiment_number"]) for row in rows}


def _finding_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row["id"]) for row in rows}


def test_req_capstone_5294_spec_declares_required_artifact_fields() -> None:
    """REQ-CAPSTONE-5294: OpenSpec declares the V483 capstone contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5294") :]
    normalized_section = " ".join(section.split())

    for marker in mod.SPEC_REFS:
        assert marker in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized_section


def test_scenario_capstone_5294_synthesizes_v483_without_laundering(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5294: V483 positives, nulls, gates, and blocks stay separate."""

    _make_repo(tmp_path)
    artifact = mod.build_artifact(root=tmp_path, duration_s=0.01, commands_run=_commands())

    mod.validate_artifact(artifact)
    assert _value(artifact, "honest_verdict").startswith("complete:")
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
    summary = _value(artifact, "tasks_summarized")
    assert summary["expected_count"] == 12
    assert summary["loadable_count"] == 12
    assert summary["milestone_synthesized"] is True
    assert summary["missing_artifacts"] == []
    assert summary["by_classification"] == {
        "blocked_precondition": 2,
        "clean_null": 1,
        "clean_positive": 6,
        "gated_skip": 2,
        "mixed_positive_with_harmful_class": 1,
    }
    assert _source_ids(summary["per_task"]) == set(range(5282, 5294))
    assert len(summary["conductor_log_entries"]) == 4

    assert _finding_ids(_value(artifact, "clean_positive_findings")) == {
        "transition_and_source_refresh_ready",
        "claim_level_coherence_fixture_ready",
        "compilable_trace_dsl_fixture_ready",
        "memory_operation_attribution_ready",
        "memory_assisted_coherence_dosing_positive",
        "pbit_cdcl_aggregate_guidance_positive",
    }
    assert _finding_ids(_value(artifact, "null_or_harmful_findings")) == {
        "low_order_curriculum_clean_null",
        "pbit_cdcl_misleading_assumption_harm",
    }
    assert _finding_ids(_value(artifact, "gated_or_blocked_findings")) == {
        "sota_runtime_offload_blocked",
        "claim_level_sota_pilot_gated_skip",
        "trace_dsl_sota_extraction_gated_skip",
        "hardware_reachability_blocked_no_speedup",
    }
    retirements = _value(artifact, "retirements_or_exclusions")
    assert retirements["manifest_updated"] is True
    assert retirements["same_verdict_retirements"][0]["id"] == mod.SAME_VERDICT_RETIREMENT_ID
    assert _value(artifact, "ops_docs_updated") == {
        "ops_status": False,
        "ops_changelog": False,
        "traceability": False,
        "reason": "stop_when_done_reconciler_deferred_ops_docs",
    }
    assert _value(artifact, "research_complete_updated") is False
    assert len(_value(artifact, "next_milestone_recommendations")) == 6


def test_scenario_capstone_5294_missing_required_artifact_blocks(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5294-BLOCKED-MISSING-INPUT: missing inputs fail closed."""

    _make_repo(tmp_path, omit={5290}, manifest=False)
    artifact = mod.build_artifact(root=tmp_path, duration_s=0.01, commands_run=_commands())

    mod.validate_artifact(artifact)
    assert _value(artifact, "honest_verdict").startswith("blocked_missing_required")
    summary = _value(artifact, "tasks_summarized")
    assert summary["milestone_synthesized"] is False
    assert summary["missing_artifacts"][0]["experiment_number"] == 5290
    blocked = _value(artifact, "gated_or_blocked_findings")
    assert "missing_required_artifacts" in _finding_ids(blocked)
    assert _value(artifact, "research_complete_updated") is False
    assert _value(artifact, "retirements_or_exclusions")["manifest_updated"] is False


def test_req_capstone_5294_defensive_helpers_and_validation(tmp_path: Path) -> None:
    """REQ-CAPSTONE-5294: helper and validation paths preserve evidence discipline."""

    missing_payload, missing_info = mod.read_json_mapping(tmp_path / "missing.json")
    assert missing_payload == {}
    assert missing_info["error"] == "missing"

    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")
    payload, info = mod.read_json_mapping(malformed)
    assert payload == {}
    assert str(info["error"]).startswith("malformed_json")

    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    payload, info = mod.read_json_mapping(scalar)
    assert payload == {}
    assert info["error"] == "not_json_object"

    assert mod.classify_payload(1, {"flagged_adversarial": True}) == "quarantined"
    assert mod.classify_payload(1, {"blocked_at_layer": "conductor_pre_gate"}) == "gated_skip"
    assert mod.classify_payload(5292, _payloads()[5292]) == "mixed_positive_with_harmful_class"
    assert mod.classify_payload(5291, _payloads()[5291]) == "clean_null"
    assert mod.classify_payload(5284, _payloads()[5284]) == "blocked_precondition"
    assert mod.classify_payload(1, {"honest_verdict": "harmful_regression: bad"}) == "harmful"
    assert mod.classify_payload(1, {"honest_verdict": "complete: useful"}) == "clean_positive"
    assert mod._summary(9999, {"honest_verdict": "complete: generic"}, "clean_positive") == (
        "complete: generic"
    )
    assert mod.read_conductor_log_entries(tmp_path) == []
    assert mod.load_commands(None) == []

    bad_commands = tmp_path / "bad_commands.json"
    bad_commands.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="commands JSON"):
        mod.load_commands(bad_commands)

    _make_repo(tmp_path)
    quarantined_payload = dict(_payloads()[5283])
    quarantined_payload["flagged_adversarial"] = True
    _write_json(tmp_path / mod.UPSTREAM_SOURCES[1].relative_path, quarantined_payload)
    quarantined_artifact = mod.build_artifact(root=tmp_path, duration_s=0.01, commands_run=[])
    assert "quarantined_artifacts" in _finding_ids(
        _value(quarantined_artifact, "gated_or_blocked_findings")
    )

    _write_json(tmp_path / mod.UPSTREAM_SOURCES[1].relative_path, _payloads()[5283])
    artifact = mod.build_artifact(root=tmp_path, duration_s=0.01, commands_run=[])
    missing_field = dict(artifact)
    del missing_field["schema"]
    with pytest.raises(ValueError, match="missing required field"):
        mod.validate_artifact(missing_field)

    artifact["clean_positive_findings"] = []
    with pytest.raises(ValueError, match="clean_positive_findings"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(root=tmp_path, duration_s=0.01, commands_run=[])
    artifact["clean_positive_findings"] = {"principle": "x", "value": "bad"}
    with pytest.raises(ValueError, match="clean_positive_findings"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(root=tmp_path, duration_s=0.01, commands_run=[])
    artifact["inference_substrate"] = {"principle": "x", "value": "live_llm_inference"}
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(root=tmp_path, duration_s=0.01, commands_run=[])
    artifact["commands_run"] = {"command": "bad"}
    with pytest.raises(ValueError, match="commands_run"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(root=tmp_path, duration_s=0.01, commands_run=[])
    artifact["commands_run"] = [{"command": "bad"}]
    with pytest.raises(ValueError, match="commands_run entries"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(root=tmp_path, duration_s=0.01, commands_run=[])
    artifact["honest_verdict"] = {"principle": "x", "value": "not_terminal"}
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(root=tmp_path, duration_s=0.01, commands_run=[])
    artifact["reproducibility_checksum"] = "bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(artifact)


def test_req_capstone_5294_cli_writes_stable_result(tmp_path: Path) -> None:
    """REQ-CAPSTONE-5294: the CLI writes a validated capstone artifact."""

    _make_repo(tmp_path)
    commands_path = tmp_path / "commands.json"
    commands_path.write_text(json.dumps(_commands()), encoding="utf-8")
    output = tmp_path / mod.RESULT_RELATIVE_PATH

    assert (
        mod.main(
            [
                "--root",
                str(tmp_path),
                "--output",
                str(output),
                "--commands-json",
                str(commands_path),
            ]
        )
        == 0
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert artifact["commands_run"] == _commands()
