"""Tests for REQ-CAPSTONE-5281 / SCENARIO-CAPSTONE-5281."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5281_capstone_v482 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


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
        5269: {
            **_base("complete: .481 archived and .482 activation-ready"),
            "activation_ready": True,
            "milestone_archived": True,
        },
        5270: {
            **_base(
                "complete: 5 new actionable findings appended",
                "literature_ingestion_network_sources",
            ),
            "new_references_added": _wrap(5),
            "plan_change_required": False,
        },
        5271: {
            **_base(
                "complete: telemetry_receipts_ready=true via flagship_moe, flagship_dense, middle_moe",
                "live_llm_internal_telemetry_local_gguf_sota",
            ),
            "telemetry_harness_ready": True,
            "no_quality_claim": _wrap(True),
        },
        5272: {
            **_base(
                "complete: harmful internal/logit signal delta_over_lexical=-0.345679 "
                "auroc=0.654321 sample_count=27",
                "live_llm_internal_telemetry_local_gguf_sota",
            ),
            "internal_signal_available": _wrap(True),
            "auroc": _wrap(0.654320987654321),
            "delta_over_lexical_baseline": _wrap(-0.345679012345679),
        },
        5273: {
            **_base(
                "complete: solver_fixture_ready true for Exp 5274 deterministic gated retry",
                "offline_deterministic_certificate_no_llm",
            ),
            "solver_fixture_ready": True,
            "baseline_validity": _wrap(1.0),
            "counterexample_coverage": _wrap(1.0),
            "schema_checks_passed": _wrap(True),
        },
        5274: {
            **_base(
                "blocked_preconditions: llama_cpp_gpu_offload_unavailable; retry was unmeasured",
                "live_llm_inference_local_gguf_sota",
            ),
            "flagged_adversarial": False,
            "corrigendum_pending": [
                {
                    "kind": "DURATION_TOO_SHORT",
                    "severity": "critical",
                    "detail": "blocked precondition branch kept live substrate label",
                }
            ],
            "linter_flag_corrigendum": {
                "fresh_recheck_result": "0 flags after blocked-precondition exemption",
                "underlying_finding_preserved": "retry blocked and unmeasured",
            },
            "blockers": ["llama_cpp_gpu_offload_unavailable"],
            "solver_extraction_improved": _wrap(False),
            "validity_rate": _wrap(0.0),
            "unsafe_false_accepts": _wrap(0),
        },
        5275: {
            **_base("complete: governed decision-history memory is ready for Exp5276"),
            "memory_decision_history_ready": True,
            "provenance_fields_present": _wrap(True),
            "scope_enforcement_passed": _wrap(True),
            "stale_conflict_eviction_passed": _wrap(True),
            "harmful_memory_rollback_passed": _wrap(True),
            "unsafe_false_accepts": _wrap(0),
        },
        5276: {
            **_base(
                "complete: positive memory-assisted verifier dosing preserved always-full quality",
                "live_llm_inference_local_gguf_sota",
            ),
            "memory_verifier_dose_ready": _wrap(True),
            "calls_avoided_rate": _wrap(0.857143),
            "decision_quality_delta": _wrap(0.0),
            "unsafe_false_accepts": _wrap(0),
            "memory_scope_violations_blocked": _wrap(4),
        },
        5277: {
            **_base(
                "complete: scaled certificate positive for a bounded three-component PWA/MILP fixture",
                "offline_deterministic_certificate_no_llm",
            ),
            "certificate_scaled": _wrap(True),
            "false_property_rejected": _wrap(True),
            "approximation_slack": _wrap(0.0166),
        },
        5278: {
            **_base(
                "complete: factor-graph boundary is usable for the tiny solver fixture",
                "offline_deterministic_certificate_no_llm",
            ),
            "factor_graph_boundary_ready": _wrap(True),
            "mapping_roundtrip_passed": _wrap(True),
            "false_assignment_rejected": _wrap(True),
            "hardware_speedup_claimed": _wrap(False),
        },
        5279: {
            **_base(
                "blocked_board_reachability: kv260=blocked_kv260_ssh_unreachable "
                "polfire=blocked_polarfire_ssh_unreachable no_speedup_claim",
                "hardware_probe_no_speedup_claim",
            ),
            "hardware_evidence_level": _wrap("reachability_status_receipt_only"),
            "hardware_speedup_claimed": _wrap(False),
            "blocked_reason": _wrap({"KV260": {"reason": "blocked_kv260_ssh_unreachable"}}),
        },
        5280: {
            **_base("complete: producer evidence discipline is ready at the template boundary"),
            "normalizer_evidence_ready": _wrap(True),
            "missing_evidence_rejected": _wrap(True),
            "bare_gate_preservation_passed": _wrap(True),
            "duration_substrate_regression_passed": _wrap(True),
            "producer_coverage": _wrap(1.0),
        },
    }


def _make_repo(root: Path, *, omit: set[int] | None = None) -> None:
    omit = omit or set()
    by_number = _payloads()
    for source in mod.UPSTREAM_SOURCES:
        if source.experiment_number not in omit:
            _write_json(root / source.relative_path, by_number[source.experiment_number])
    for context in mod.SOURCE_CONTEXT_PATHS:
        path = root / context
        path.parent.mkdir(parents=True, exist_ok=True)
        if context != Path("research-roadmap-next.yaml"):
            path.write_text(f"context for {context}\n", encoding="utf-8")


def _commands() -> list[dict[str, str]]:
    return [{"command": "focused fixture", "outcome": "PASS"}]


def _ids(rows: list[dict[str, Any]]) -> set[int]:
    return {int(row["experiment_number"]) for row in rows}


def test_req_capstone_5281_spec_declares_required_artifact_fields() -> None:
    """REQ-CAPSTONE-5281: OpenSpec declares the V482 capstone contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5281") :]
    normalized_section = " ".join(section.split())

    for marker in mod.SPEC_REFS:
        assert marker in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized_section


def test_scenario_capstone_5281_classifies_v482_without_laundering(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5281: positives, harmfuls, flags, and blocks stay separate."""

    _make_repo(tmp_path)
    artifact = mod.build_artifact(root=tmp_path, duration_s=0.01, commands_run=_commands())

    mod.validate_artifact(artifact)
    assert mod.value_of(artifact["honest_verdict"]).startswith("complete:")
    assert mod.value_of(artifact["inference_substrate"]) == mod.INFERENCE_SUBSTRATE
    assert mod.value_of(artifact["milestone_synthesized"]) is True
    assert _ids(mod.value_of(artifact["clean_positives"])) == {
        5269,
        5270,
        5271,
        5273,
        5275,
        5276,
        5277,
        5278,
        5280,
    }
    assert mod.value_of(artifact["clean_nulls"]) == []
    assert _ids(mod.value_of(artifact["harmful_or_regressions"])) == {5272}
    assert _ids(mod.value_of(artifact["flagged_or_quarantined"])) == {5274}
    assert _ids(mod.value_of(artifact["honest_blocks"])) == {5279}
    assert mod.value_of(artifact["gated_skips"]) == []
    flagged_row = mod.value_of(artifact["flagged_or_quarantined"])[0]
    assert flagged_row["flagged_adversarial"] is False
    assert flagged_row["quarantined"] is True
    assert flagged_row["critical_corrigendum_kinds"] == ["DURATION_TOO_SHORT"]

    gaps = artifact["prd_gap_advancement"]
    assert gaps["receipt_clean_verifier_signals"]["advanced"] == "partial_negative_control"
    assert gaps["governed_continuous_self_learning"]["advanced"] is True
    assert gaps["hardware_evidence"]["advanced"] == "blocked_reachability_only"
    assert mod.value_of(artifact["continuous_self_learning_advanced"]) is True
    assert mod.value_of(artifact["hardware_speedup_claimed"]) is False
    assert mod.value_of(artifact["docs_updated"]) == {
        "openspec_capstone_spec": True,
        "research_complete": False,
        "ops_status": False,
        "ops_changelog": False,
        "traceability": False,
        "reason": "stop_when_done_reconciler_deferred_ops_docs",
    }
    assert artifact["missing_artifacts"] == []
    assert any(
        row["path"] == "research-roadmap-next.yaml" and not row["exists"]
        for row in artifact["source_context_read"]
    )


def test_scenario_capstone_5281_missing_required_artifact_blocks(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5281-BLOCKED-MISSING-INPUT: missing inputs fail closed."""

    _make_repo(tmp_path, omit={5276})
    artifact = mod.build_artifact(root=tmp_path, duration_s=0.01, commands_run=_commands())

    mod.validate_artifact(artifact)
    assert mod.value_of(artifact["honest_verdict"]).startswith("blocked_missing_required")
    assert mod.value_of(artifact["milestone_synthesized"]) is False
    assert any(row["experiment_number"] == 5276 for row in mod.value_of(artifact["honest_blocks"]))
    assert mod.value_of(artifact["hardware_speedup_claimed"]) is False


def test_req_capstone_5281_defensive_helpers_and_validation(tmp_path: Path) -> None:
    """REQ-CAPSTONE-5281: helper and validation paths do not invent evidence."""

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

    bad_source_path = tmp_path / mod.UPSTREAM_SOURCES[0].relative_path
    bad_source_path.parent.mkdir(parents=True, exist_ok=True)
    bad_source_path.write_text("{", encoding="utf-8")
    row, loaded = mod._row_for_source(mod.UPSTREAM_SOURCES[0], tmp_path)
    assert loaded is None
    assert row["classification"] == "malformed"

    assert mod.classify_payload(1, {"flagged_adversarial": True}) == "flagged_or_quarantined"
    assert (
        mod.classify_payload(
            1,
            {
                "flagged_adversarial": False,
                "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
            },
        )
        == "flagged_or_quarantined"
    )
    assert (
        mod.classify_payload(1, {"honest_verdict": "blocked_gated_skip: upstream"}) == "gated_skip"
    )
    assert (
        mod.classify_payload(1, {"honest_verdict": "complete: clean null no improvement"})
        == "clean_null"
    )
    assert (
        mod.classify_payload(1, {"honest_verdict": "complete: harmful regression"})
        == "harmful_or_regression"
    )
    assert (
        mod.classify_payload(1, {"honest_verdict": "blocked_preconditions: no board"})
        == "honest_block"
    )
    assert (
        mod.classify_payload(1, {"honest_verdict": "complete: useful result"}) == "clean_positive"
    )
    assert mod._critical_corrigendum_kinds({"corrigendum_pending": ["not-a-record"]}) == []
    assert mod._quarantine_reasons({"flagged_adversarial": True}) == ["flagged_adversarial_true"]
    assert mod._summary(9999, {"honest_verdict": "complete: generic"}, "clean_positive") == (
        "complete: generic"
    )
    assert mod.load_commands(None) == []

    bad_commands = tmp_path / "bad_commands.json"
    bad_commands.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="commands JSON"):
        mod.load_commands(bad_commands)

    artifact = mod.build_artifact(root=tmp_path, duration_s=0.01, commands_run=[])
    missing_field = dict(artifact)
    del missing_field["schema"]
    with pytest.raises(ValueError, match="missing required field"):
        mod.validate_artifact(missing_field)

    artifact["clean_positives"] = []
    with pytest.raises(ValueError, match="clean_positives"):
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
    artifact["reproducibility_checksum"] = "bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(root=tmp_path, duration_s=0.01, commands_run=[])
    artifact["honest_verdict"] = {"principle": "x", "value": "not_terminal"}
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact)


def test_req_capstone_5281_cli_writes_stable_result(tmp_path: Path) -> None:
    """REQ-CAPSTONE-5281: the CLI writes a validated capstone artifact."""

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
