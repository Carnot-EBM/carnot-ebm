"""Tests for REQ-CAPSTONE-5306 / SCENARIO-CAPSTONE-5306."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5306_capstone_v484 as mod


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
        5295: {
            **_base("complete: .483 archived and .484 activation-ready"),
            "milestone_archived": True,
            "activation_ready": True,
        },
        5296: {
            **_base(
                "complete: 4 new actionable findings appended; executable .484 plan unchanged.",
                "literature_ingestion_network_sources",
            ),
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {"kind": "DURATION_TOO_SHORT", "severity": "critical"},
                {"kind": "METHODOLOGY_MISSING", "severity": "warn"},
            ],
            "new_references_added": _wrap(4),
            "references_md_updated": _wrap(True),
            "actionable_deltas": _wrap([{"title": "EBM workloads"}]),
        },
        5297: {
            **_base(
                "blocked_preconditions: changed_runtime_sota_ready=false "
                "flagship_moe:blocked_native_cli_timeout:offload=True",
                "blocked_preconditions_with_no_quality_claim",
            ),
            "changed_runtime_sota_ready": False,
            "runtime_substrate_changed": _wrap(
                {
                    "backend_kind": "native_llama_cpp_cli",
                    "changed_from_exp5284": True,
                    "cuda_backend_evidence": True,
                }
            ),
            "MODEL_SPECS": _wrap(
                {
                    "flagship_moe": {"runtime_status": "blocked_native_cli_timeout"},
                    "flagship_dense": {"runtime_status": "blocked_native_cli_timeout"},
                    "middle_moe": {"runtime_status": "blocked_native_cli_timeout"},
                }
            ),
            "no_quality_claim": _wrap(True),
        },
        5298: {
            "schema": "blocked_gate_check_v1",
            "honest_verdict": "blocked_gate_check_failed",
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": (
                "1 of 1 gate(s) failed; first failure: "
                "exp5297.changed_runtime_sota_ready (actual=False == expected=True)"
            ),
            "gates_evaluated": [
                {
                    "artifact_field": "changed_runtime_sota_ready",
                    "expected": True,
                    "actual": False,
                    "passed": False,
                }
            ],
        },
        5299: {
            **_base(
                "complete: constraint-LNS fixture usable for exp5300 solver-repair guidance",
                "offline_deterministic_certificate_no_llm",
            ),
            "constraint_lns_fixture_ready": True,
            "solver_correctness_preserved": _wrap(True),
            "unsafe_false_accepts": _wrap(0),
        },
        5300: {
            **_base(
                "complete: p-bit/CDCL gate helped by blocking misleading-assumption classes "
                "while preserving aggregate conflict savings on deterministic fixtures",
                "offline_deterministic_certificate_no_llm",
            ),
            "pbit_gate_ready": _wrap(True),
            "correctness_preserved": _wrap(True),
            "misleading_class_blocked": _wrap(
                {
                    "all_misleading_blocked": True,
                    "blocked_classes": [
                        "misleading_factor_sat",
                        "misleading_repair",
                        "semantic_wrong_control",
                    ],
                }
            ),
            "aggregate_metrics": {
                "solver_only_vs_gated_delta": {"conflicts_saved": 9},
                "ungated_vs_gated_delta": {"conflicts_saved_by_gate": 3},
            },
            "hardware_speedup_claimed": _wrap(False),
        },
        5301: {
            **_base(
                "complete: spectral step-control is usable as a tiny deterministic stability "
                "diagnostic before energy-guided inner-loop claims",
                "offline_deterministic_certificate_no_llm",
            ),
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
            "spectral_control_ready": _wrap(True),
            "divergence_recovery": _wrap(
                {
                    "adaptive_recovered": True,
                    "adaptive_total_recovery_shrinks": 8,
                    "aggressive_diverged": True,
                }
            ),
            "hardware_speedup_claimed": False,
            "llm_quality_claimed": False,
        },
        5302: {
            **_base("complete: adaptive memory policy helped"),
            "adaptive_memory_policy_positive": _wrap(True),
            "memory_policy_candidate_ready": True,
            "heldout_quality_delta_vs_always_full": _wrap(
                {
                    "delta": 0.0,
                    "adaptive_memory_policy_quality_rate": 1.0,
                    "always_full_quality_rate": 1.0,
                }
            ),
            "full_verifier_calls_avoided": _wrap(
                {"vs_always_full": 3, "rate_vs_always_full": 0.428571}
            ),
            "unsafe_false_accepts": _wrap({"count": 0}),
            "rollback_exercised": _wrap({"trigger_count": 2, "value": True}),
            "no_weight_mutation": _wrap(True),
        },
        5303: {
            **_base(
                "complete: memory stress passed; adaptive policy matched always-full quality",
                "offline_deterministic_fixture_no_llm",
            ),
            "memory_stress_passed": _wrap(True),
            "calls_avoided": _wrap({"vs_always_full": 5, "rate_vs_always_full": 0.625}),
            "unsafe_false_accepts": _wrap({"count": 0}),
            "competency_metrics": {
                "accurate_retrieval": {"adaptive_quality_rate": 1.0},
                "test_time_learning": {"adaptive_quality_rate": 1.0},
                "long_range_understanding": {"adaptive_quality_rate": 1.0},
                "conflict_resolution": {"adaptive_quality_rate": 1.0},
                "selective_forgetting": {"adaptive_quality_rate": 1.0},
            },
            "stale_conflict_handling": _wrap({"rate": 1.0}),
            "selective_forgetting_correctness": {"rate": 1.0},
            "rollback_success_rate": _wrap({"rate": 1.0}),
        },
        5304: {
            **_base(
                "complete: dynamic abstraction helped diagnostic tightness and spot-check hit "
                "rate, while certificate success stayed unchanged on the bounded fixture",
                "offline_deterministic_certificate_no_llm",
            ),
            "dynamic_abstraction_helped": _wrap(
                {
                    "helped": True,
                    "help_kind": "diagnostic_tightness_not_certificate_success",
                    "spotcheck_hit_rate_delta": 0.8808888889,
                    "envelope_gap_reduction": 0.01125,
                    "success_improvement": 0.0,
                }
            ),
            "spotcheck_metrics": _wrap({"dynamic_hit_rate_delta": 0.8808888889}),
            "slack_metrics": _wrap({"dynamic_envelope_gap_reduction": 0.01125}),
            "certificate_success_by_method": _wrap(
                {
                    "dynamic_spotcheck_refinement": True,
                    "static_abstraction": True,
                    "low_order_exp5291": True,
                }
            ),
            "false_property_rejected": _wrap(True),
        },
        5305: {
            **_base(
                "blocked_board_reachability: kv260=blocked_kv260_ssh_unreachable "
                "polfire=reachable_ssh_status_only "
                "gatemate=blocked_gatemate_physical_jtag_setup_unchanged no_speedup_claim",
                "hardware_probe_no_speedup_claim",
            ),
            "kv260_status": _wrap({"status": "blocked_kv260_ssh_unreachable"}),
            "polarfire_status": _wrap({"status": "reachable_ssh_status_only"}),
            "gatemate_status": _wrap({"status": "blocked_gatemate_physical_jtag_setup_unchanged"}),
            "hardware_evidence_level": _wrap("reachability_status_receipt_only"),
            "hardware_speedup_claimed": _wrap(False),
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
                "| 2026-07-06 06:25 UTC | PHASE 0 transition -- archive .483 truth | OK |",
                "| 2026-07-06 09:12 UTC | PHASE 0 gated on exp5297 -- SOTA coherence and tra | GATE_BLOCK |",
                "| 2026-07-06 10:04 UTC | PHASE 1 gated on exp5299 -- p-bit/CDCL instance-cl | OK |",
                "| 2026-07-06 10:54 UTC | PHASE 2 gated on exp5302 -- memory conflict, forge | OK |",
                "| 2026-07-06 11:29 UTC | PHASE 3 hardware continuity -- KV260, PolarFire, a | OK |",
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


def _ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row["id"]) for row in rows}


def test_req_capstone_5306_spec_declares_required_artifact_fields() -> None:
    """REQ-CAPSTONE-5306: OpenSpec declares the V484 capstone contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5306") :]
    normalized_section = " ".join(section.split())

    for marker in mod.SPEC_REFS:
        assert marker in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized_section


def test_scenario_capstone_5306_synthesizes_v484_without_laundering(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5306: V484 evidence lanes stay separate."""

    _make_repo(tmp_path)
    artifact = mod.build_artifact(root=tmp_path, duration_s=0.01, commands_run=_commands())

    mod.validate_artifact(artifact)
    assert _value(artifact, "honest_verdict").startswith("complete:")
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE

    summary = _value(artifact, "tasks_summarized")
    assert summary["expected_count"] == 11
    assert summary["loadable_count"] == 11
    assert summary["milestone_synthesized"] is True
    assert summary["missing_artifact"] == []
    assert summary["by_classification"] == {
        "blocked_precondition": 2,
        "clean_positive": 4,
        "gated_skip": 1,
        "mixed_positive_with_harmful_class": 1,
        "quarantined": 2,
        "clean_null": 1,
    }
    assert [row["experiment_number"] for row in summary["quarantined"]] == [5296, 5301]
    assert [row["experiment_number"] for row in summary["gated_skip"]] == [5298]
    assert [row["experiment_number"] for row in summary["blocked_precondition"]] == [5297, 5305]
    assert summary["harmful_or_regression"] == []

    changed = _value(artifact, "changed_runtime_outcome")
    assert changed["changed_runtime_sota_ready"] is False
    assert changed["backend_kind"] == "native_llama_cpp_cli"
    assert changed["cuda_backend_evidence"] is True
    assert changed["no_quality_claim"] is True
    assert changed["sota_quality_measured"] is False

    sota = _value(artifact, "sota_quality_outcome")
    assert sota["measured"] is False
    assert sota["blocked_at_layer"] == "conductor_pre_gate"
    assert "SOTA smoke was not measured" in sota["summary"]

    learning = _value(artifact, "continuous_self_learning_outcome")
    assert learning["heldout_quality_delta_vs_always_full"]["delta"] == 0.0
    assert learning["full_verifier_calls_avoided"]["vs_always_full"] == 3
    assert learning["stress_calls_avoided"]["vs_always_full"] == 5
    assert learning["unsafe_false_accepts"] == 0
    assert learning["rollback_exercised"]["trigger_count"] == 2
    assert learning["stress_competency_quality_rates"]["conflict_resolution"] == 1.0

    solver = _value(artifact, "solver_energy_certificate_outcome")
    assert solver["constraint_lns_fixture_ready"] is True
    assert solver["pbit_gate_ready"] is True
    assert solver["pbit_blocked_classes"] == [
        "misleading_factor_sat",
        "misleading_repair",
        "semantic_wrong_control",
    ]
    assert solver["spectral_control"]["headline_eligible"] is False
    assert solver["kan_dynamic_abstraction"]["certificate_success_improvement"] == 0.0
    assert solver["kan_dynamic_abstraction"]["diagnostic_tightness_helped"] is True

    assert _value(artifact, "hardware_speedup_claimed") is False
    docs = _value(artifact, "docs_updated")
    assert docs["openspec_capstone_spec"] is True
    assert docs["ops_status"] is False
    assert docs["ops_changelog"] is False
    assert docs["traceability"] is False
    assert docs["docs_index"] is False

    retirements = _value(artifact, "retirements_or_exclusions_recommended")
    assert retirements["manifest_has_exp5284_cpu_path_retirement"] is True
    assert "repeat_exp5284_cpu_only_path" in _ids(retirements["recommendations"])
    assert len(_value(artifact, "next_milestone_recommendations")) >= 5


def test_scenario_capstone_5306_missing_required_artifact_blocks(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5306-BLOCKED-MISSING-INPUT: missing inputs fail closed."""

    _make_repo(tmp_path, omit={5303}, manifest=False)
    artifact = mod.build_artifact(root=tmp_path, duration_s=0.01, commands_run=_commands())

    mod.validate_artifact(artifact)
    assert _value(artifact, "honest_verdict").startswith("blocked_missing_required")
    summary = _value(artifact, "tasks_summarized")
    assert summary["milestone_synthesized"] is False
    assert summary["missing_artifact"][0]["experiment_number"] == 5303
    assert _value(artifact, "hardware_speedup_claimed") is False
    assert _value(artifact, "docs_updated")["docs_index"] is False
    assert (
        _value(artifact, "retirements_or_exclusions_recommended")[
            "manifest_has_exp5284_cpu_path_retirement"
        ]
        is False
    )


def test_req_capstone_5306_defensive_helpers_and_validation(tmp_path: Path) -> None:
    """REQ-CAPSTONE-5306: helper and validation paths preserve evidence discipline."""

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
    assert mod.classify_payload(5300, _payloads()[5300]) == "mixed_positive_with_harmful_class"
    assert mod.classify_payload(5304, _payloads()[5304]) == "clean_null"
    assert mod.classify_payload(5297, _payloads()[5297]) == "blocked_precondition"
    assert mod.classify_payload(1, {"honest_verdict": "harmful_regression: bad"}) == (
        "harmful_or_regression"
    )
    assert (
        mod.classify_payload(1, {"honest_verdict": "complete: rolled back harmful memory safely"})
        == "clean_positive"
    )
    assert mod.classify_payload(1, {"honest_verdict": "complete: honest null"}) == "clean_null"
    assert mod.classify_payload(1, {"honest_verdict": "complete: useful"}) == "clean_positive"
    assert mod._summary(9999, {"honest_verdict": "complete: generic"}, "clean_positive") == (
        "complete: generic"
    )
    assert mod.read_conductor_log_entries(tmp_path) == []
    assert mod.load_commands(None) == []
    assert mod._number(None) is None
    assert mod._number(True) is None
    assert mod._number("not-a-number") is None
    assert mod._count({"count": "not-an-int"}) == 0
    assert mod._model_statuses({}) == {}
    assert mod._blocked_classes({}) == []
    assert mod._competency_rates(
        {"competency_metrics": {"principle": "skip me", "edge": {"adaptive_quality_rate": "bad"}}}
    ) == {"edge": None}

    bad_commands = tmp_path / "bad_commands.json"
    bad_commands.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="commands JSON"):
        mod.load_commands(bad_commands)

    _make_repo(tmp_path)
    artifact = mod.build_artifact(root=tmp_path, duration_s=0.01, commands_run=[])
    missing_field = dict(artifact)
    del missing_field["schema"]
    with pytest.raises(ValueError, match="missing required field"):
        mod.validate_artifact(missing_field)

    artifact["tasks_summarized"] = []
    with pytest.raises(ValueError, match="tasks_summarized"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(root=tmp_path, duration_s=0.01, commands_run=[])
    artifact["tasks_summarized"] = {"principle": "x", "value": []}
    with pytest.raises(ValueError, match="tasks_summarized"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(root=tmp_path, duration_s=0.01, commands_run=[])
    artifact["tasks_summarized"] = {"principle": "x", "value": {}}
    with pytest.raises(ValueError, match="tasks_summarized missing"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(root=tmp_path, duration_s=0.01, commands_run=[])
    artifact["inference_substrate"] = {"principle": "x", "value": "live_llm_inference"}
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(root=tmp_path, duration_s=0.01, commands_run=[])
    artifact["hardware_speedup_claimed"] = {"principle": "x", "value": True}
    with pytest.raises(ValueError, match="hardware_speedup_claimed"):
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


def test_req_capstone_5306_cli_writes_stable_result(tmp_path: Path) -> None:
    """REQ-CAPSTONE-5306: the CLI writes a validated capstone artifact."""

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
