"""Tests for Exp6549 production Safety-Net adapter artifact.

Spec refs: REQ-PIPELINE-6549, SCENARIO-PIPELINE-6549-DEFAULT-OFF,
SCENARIO-PIPELINE-6549-ENABLED-FALLBACK, SCENARIO-PIPELINE-6549-ATTACKS.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_6549_production_safety_net_adapter as mod


TESTS_RUN = [{"command": "focused-exp6549", "exit_code": 0}]


def test_req_pipeline_6549_spec_declares_production_adapter_contract() -> None:
    """REQ-PIPELINE-6549: OpenSpec owns the production adapter contract."""

    text = Path("openspec/capabilities/pipeline/spec.md").read_text(encoding="utf-8")
    section = text[text.index("REQ-PIPELINE-6549") :]
    for anchor in (
        "SCENARIO-PIPELINE-6549-DEFAULT-OFF",
        "SCENARIO-PIPELINE-6549-ENABLED-FALLBACK",
        "SCENARIO-PIPELINE-6549-ATTACKS",
    ):
        assert anchor in section


def test_scenario_pipeline_6549_artifact_schema_and_positive_reducer(tmp_path: Path) -> None:
    """SCENARIO-PIPELINE-6549-ENABLED-FALLBACK: artifact rows recompute readiness."""

    artifact = mod.build_artifact(
        result_path=tmp_path / "experiment_6549.json",
        write=False,
        duration_s=0.0,
        tests_run=TESTS_RUN,
    )

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_production_safety_net_adapter_positive"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["verdict_class"] == "positive"
    assert artifact["production_safety_net_adapter_ready_score"] == 1.0
    assert artifact["inference_substrate"] == (
        "production_verify_repair_compact_router_and_exact_fallback_no_llm"
    )
    assert artifact["verifier_is_oracle"] is False
    assert artifact["upstream_gate_receipt"]["gate_passed"] is True
    assert artifact["adapter_configuration_contract"]["enabled_default"] is False
    assert artifact["exact_output_equality_receipt"]["all_exact_outputs_equal"] is True
    assert artifact["candidate_preservation_receipt"]["all_candidates_preserved"] is True
    assert artifact["exception_table_immutability_receipt"]["held_write_attempt_count"] == 0
    assert artifact["fallback_and_rollback_receipts"]["fallback_reachable"] is True
    assert artifact["fallback_and_rollback_receipts"]["rollback_restores_disabled"] is True
    assert artifact["shortcut_attack_matrix"]["all_attacks_fail_closed"] is True
    assert artifact["aggregate_row_recomputation"]["ready_score_from_rows"] == 1.0
    assert artifact["gate_check_summary"]["all_gates_passed"] is True
    assert artifact["protected_files_unchanged"]["all_protected_files_unchanged"] is True
    assert mod.validate_artifact(artifact) == []


def test_scenario_pipeline_6549_identity_and_enabled_rows_are_recomputable() -> None:
    """SCENARIO-PIPELINE-6549-DEFAULT-OFF: identity and enabled rows carry receipts."""

    artifact = mod.build_artifact(write=False, duration_s=0.0, tests_run=TESTS_RUN)
    identity_rows = artifact["disabled_identity_rows"]
    enabled_rows = artifact["enabled_decision_rows"]
    per_unit = artifact["per_unit_rows"]

    assert identity_rows
    assert enabled_rows
    assert len(per_unit) == len(identity_rows) * len(mod.CONDITIONS)
    assert all(row["serialized_request_bytes_equal"] for row in identity_rows)
    assert all(row["candidate_order_equal"] for row in identity_rows)
    assert all(row["checker_calls_equal"] for row in identity_rows)
    assert all(row["outputs_equal"] for row in identity_rows)
    assert all(row["error_types_equal"] for row in identity_rows)
    assert all(row["side_effects_equal"] for row in identity_rows)
    assert all(row["persistence_equal"] for row in identity_rows)
    assert {row["condition"] for row in per_unit} == set(mod.CONDITIONS)
    assert any(row["fallback_reason"] == "abstention" for row in enabled_rows)
    assert any(row["fallback_reason"] == "forced_fallback" for row in per_unit)
    assert any(
        row["condition"] == "malformed_input" and row["attack_failed_closed"] for row in per_unit
    )


def test_scenario_pipeline_6549_validation_fail_closed_edges(tmp_path: Path) -> None:
    """SCENARIO-PIPELINE-6549-ATTACKS: tampering disables positive claims."""

    artifact = mod.build_artifact(write=False, duration_s=0.0, tests_run=TESTS_RUN)
    bad_identity = json.loads(json.dumps(artifact))
    bad_identity["disabled_identity_rows"][0]["serialized_request_bytes_equal"] = False
    bad_identity["aggregate_row_recomputation"] = mod.aggregate_row_recomputation(bad_identity)
    bad_identity["gate_check_summary"] = mod.gate_check_summary(
        bad_identity["aggregate_row_recomputation"]
    )
    bad_identity["production_safety_net_adapter_ready_score"] = 0.0
    bad_identity["verdict_class"] = "disqualified"
    bad_identity["status"] = "disqualified_production_safety_net_adapter"
    bad_identity["honest_verdict"] = (
        "disqualified_production_safety_net_adapter: disabled identity or exact equality changed"
    )
    bad_identity["reproducibility_checksum"] = mod.reproducibility_checksum(bad_identity)
    assert "disabled identity failed" in mod.validate_artifact(bad_identity)

    bad_score = json.loads(json.dumps(artifact))
    bad_score["production_safety_net_adapter_ready_score"] = 1.0
    bad_score["aggregate_row_recomputation"]["ready_score_from_rows"] = 0.0
    bad_score["reproducibility_checksum"] = mod.reproducibility_checksum(bad_score)
    assert "ready score mismatch" in mod.validate_artifact(bad_score)

    result_path = tmp_path / "cli-exp6549.json"
    assert mod.main(["--date", "20260823", "--result-path", str(result_path)]) == 0
    written = json.loads(result_path.read_text(encoding="utf-8"))
    assert written["status"] == "complete_production_safety_net_adapter_positive"
    assert mod.main(["--validate", "--result-path", str(result_path)]) == 0


def test_scenario_pipeline_6549_reducer_and_validation_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PIPELINE-6549-ATTACKS: reducers expose blocked and invalid states."""

    artifact = mod.build_artifact(write=False, duration_s=0.0, tests_run=TESTS_RUN)
    assert mod.sha256_file(tmp_path / "missing") == "missing"
    assert mod._read_json(tmp_path / "missing.json") == {}  # noqa: SLF001
    assert mod._source_key(tmp_path, Path("/outside")) == "/outside"  # noqa: SLF001

    blocked = json.loads(json.dumps(artifact))
    blocked["upstream_gate_receipt"]["gate_passed"] = False
    blocked_aggregate = mod.aggregate_row_recomputation(blocked)
    assert blocked_aggregate["verdict_class_from_rows"] == "blocked"
    assert mod._status_and_verdict(blocked_aggregate)[2] == "blocked"  # noqa: SLF001

    null = json.loads(json.dumps(artifact))
    for row in null["per_unit_rows"]:
        if row["condition"] == "enabled_router":
            row["enabled_benefit_units"] = 0.0
    null_aggregate = mod.aggregate_row_recomputation(null)
    assert null_aggregate["verdict_class_from_rows"] == "null"
    assert mod._status_and_verdict(null_aggregate)[2] == "null"  # noqa: SLF001

    disqualified = json.loads(json.dumps(artifact))
    disqualified["candidate_preservation_receipt"]["all_candidates_preserved"] = False
    disqualified_aggregate = mod.aggregate_row_recomputation(disqualified)
    assert disqualified_aggregate["verdict_class_from_rows"] == "disqualified"
    assert mod._status_and_verdict(disqualified_aggregate)[2] == "disqualified"  # noqa: SLF001

    def _with_checksum(payload: dict[str, object]) -> dict[str, object]:
        payload["reproducibility_checksum"] = mod.reproducibility_checksum(payload)
        return payload

    malformed_cases = [
        (
            "field-set",
            lambda data: data.pop("status"),
            "required field set mismatch",
        ),
        (
            "substrate",
            lambda data: data.__setitem__("inference_substrate", "wrong"),
            "inference_substrate mismatch",
        ),
        (
            "oracle",
            lambda data: data.__setitem__("verifier_is_oracle", True),
            "verifier_is_oracle must be false",
        ),
        (
            "prefix",
            lambda data: data.__setitem__("honest_verdict", "not-terminal"),
            "honest_verdict terminal prefix mismatch",
        ),
        (
            "verdict-class",
            lambda data: data.__setitem__("verdict_class", "surprise"),
            "verdict_class outside Exp6549 enum",
        ),
        (
            "provenance",
            lambda data: data.__setitem__("field_provenance", {}),
            "field_provenance must cover required fields",
        ),
        (
            "score-domain",
            lambda data: data.__setitem__("production_safety_net_adapter_ready_score", 0.5),
            "production_safety_net_adapter_ready_score must be 0.0 or 1.0",
        ),
        (
            "exact",
            lambda data: data["aggregate_row_recomputation"].__setitem__(
                "exact_outputs_equal", False
            ),
            "exact output equality failed",
        ),
        (
            "positive-score",
            lambda data: (
                data.__setitem__("production_safety_net_adapter_ready_score", 0.0),
                data.__setitem__("verdict_class", "positive"),
                data["aggregate_row_recomputation"].__setitem__("ready_score_from_rows", 0.0),
            ),
            "positive verdict requires ready score 1.0",
        ),
        (
            "positive-benefit",
            lambda data: (
                data.__setitem__("verdict_class", "positive"),
                data["aggregate_row_recomputation"].__setitem__(
                    "charged_enabled_path_benefit_units", 0.0
                ),
            ),
            "positive verdict requires charged enabled-path benefit",
        ),
        (
            "candidate",
            lambda data: data["candidate_preservation_receipt"].__setitem__(
                "all_candidates_preserved", False
            ),
            "candidate preservation failed",
        ),
        (
            "held-write",
            lambda data: data["exception_table_immutability_receipt"].__setitem__(
                "held_write_attempt_count", 1
            ),
            "held exception-table write detected",
        ),
        (
            "fallback",
            lambda data: data["fallback_and_rollback_receipts"].__setitem__(
                "fallback_reachable", False
            ),
            "fallback unreachable",
        ),
        (
            "rollback",
            lambda data: data["fallback_and_rollback_receipts"].__setitem__(
                "rollback_restores_disabled", False
            ),
            "rollback failed",
        ),
        (
            "attacks",
            lambda data: data["shortcut_attack_matrix"].__setitem__(
                "all_attacks_fail_closed", False
            ),
            "shortcut attack false accept",
        ),
        (
            "protected",
            lambda data: data["protected_files_unchanged"].__setitem__(
                "all_protected_files_unchanged", False
            ),
            "protected files changed",
        ),
    ]
    for _name, mutate, expected in malformed_cases:
        candidate = json.loads(json.dumps(artifact))
        mutate(candidate)
        if expected != "reproducibility_checksum mismatch":
            _with_checksum(candidate)
        assert expected in mod.validate_artifact(candidate)

    bad_checksum = json.loads(json.dumps(artifact))
    bad_checksum["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad_checksum)

    bad_validate_path = tmp_path / "bad.json"
    bad_validate_path.write_text("{}\n", encoding="utf-8")
    assert mod.main(["--validate", "--result-path", str(bad_validate_path)]) == 1

    monkeypatch.setattr(mod, "build_artifact", lambda **_kwargs: {"bad": "artifact"})
    assert mod.main(["--date", "20260823", "--result-path", str(tmp_path / "bad-build.json")]) == 1


def test_scenario_pipeline_6549_disabled_route_branch_is_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PIPELINE-6549-DEFAULT-OFF: per-unit helper handles a disabled adapter."""

    artifact = mod.build_artifact(write=False, duration_s=0.0, tests_run=TESTS_RUN)
    source = artifact["per_unit_rows"][0]
    monkeypatch.setattr(
        mod,
        "_adapter_for_condition",
        lambda condition, exception_table: mod.SafetyNetProductionAdapter(
            mod.SafetyNetRouterConfig(enabled=False)
        ),
    )
    row = mod._per_unit_row(  # noqa: SLF001
        source,
        condition="enabled_router",
        exception_table={},
    )
    assert row["route"] == "disabled"
    assert row["candidate_preserved"] is True
