"""Tests for Exp6501 V560 independent capstone.

Spec refs: REQ-CAPSTONE-6501,
SCENARIO-CAPSTONE-6501-INVENTORY,
SCENARIO-CAPSTONE-6501-GATES,
SCENARIO-CAPSTONE-6501-ROWS-AND-CLAIMS,
SCENARIO-CAPSTONE-6501-RETIREMENT-HANDOFF-ATTACKS,
SCENARIO-CAPSTONE-6501-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6501_v560_capstone as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
_ARTIFACT_CACHE: dict[str, Any] | None = None


def _artifact() -> dict[str, Any]:
    global _ARTIFACT_CACHE
    if _ARTIFACT_CACHE is None:
        _ARTIFACT_CACHE = mod.build_artifact(
            repo_root=REPO,
            date="20260821",
            result_path=Path("/tmp/experiment_6501_test_result.json"),
            write=False,
            duration_s=1.0,
            tests_run=[{"command": "focused", "exit_code": 0}],
        )
    return copy.deepcopy(_ARTIFACT_CACHE)


def test_req_capstone_6501_spec_declares_required_contract() -> None:
    """REQ-CAPSTONE-6501: OpenSpec owns the Exp6501 contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-CAPSTONE-6501") :]

    for marker in (
        "SCENARIO-CAPSTONE-6501-INVENTORY",
        "SCENARIO-CAPSTONE-6501-GATES",
        "SCENARIO-CAPSTONE-6501-ROWS-AND-CLAIMS",
        "SCENARIO-CAPSTONE-6501-RETIREMENT-HANDOFF-ATTACKS",
        "SCENARIO-CAPSTONE-6501-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES
        assert field in mod.FIELD_PROVENANCE


def test_scenario_capstone_6501_inventory_classifies_all_upstream_outcomes() -> None:
    """SCENARIO-CAPSTONE-6501-INVENTORY: closed gates stay evidence."""

    artifact = _artifact()
    manifest = {row["experiment_id"]: row for row in artifact["milestone_manifest_rows"]}

    assert mod.validate_artifact(artifact) == []
    assert len(manifest) == 13
    assert artifact["v560_capstone_ready_score"] == 1.0
    assert manifest["exp6488"]["classification"] == "complete"
    assert manifest["exp6489"]["classification"] == "complete"
    assert manifest["exp6490"]["classification"] == "disqualified"
    assert manifest["exp6491"]["classification"] == "invalid"
    assert manifest["exp6492"]["classification"] == "valid_null"
    assert manifest["exp6493"]["classification"] == "blocked_by_scientific_gate"
    assert manifest["exp6494"]["classification"] == "blocked_by_scientific_gate"
    assert manifest["exp6496"]["classification"] == "valid_null"
    assert manifest["exp6499"]["classification"] == "valid_null"
    assert manifest["exp6500"]["classification"] == "blocked_by_scientific_gate"
    assert manifest["exp6493"]["actual_path"].endswith(
        "results/experiment_6493_gated_decomposed_trajectory_energy_ab.json"
    )
    assert manifest["exp6494"]["sha256"] is None
    assert manifest["exp6494"]["gate_closed_without_artifact"] is True


def test_scenario_capstone_6501_gate_chain_and_blocked_diagnostics() -> None:
    """SCENARIO-CAPSTONE-6501-GATES: gate contracts use exact fields."""

    artifact = _artifact()
    gates = {
        (row["downstream_experiment_id"], row["upstream_field"]): row
        for row in artifact["gate_contract_rows"]
    }

    assert gates[("exp6493", "trajectory_signal_ready_score")]["observed"] == 0.0
    assert gates[("exp6493", "trajectory_signal_ready_score")]["result"] == "failed"
    assert gates[("exp6493", "causal_factor_signal_ready_score")]["observed"] == 0.0
    assert (
        gates[("exp6494", "decomposed_energy_ready_score")]["result"]
        == "blocked_by_scientific_gate"
    )
    assert gates[("exp6500", "arc_energy_alignment_ready_score")]["observed"] == 0.0
    assert gates[("exp6500", "arc_energy_alignment_ready_score")]["observed_type"] == "float"

    diagnostics = {row["experiment_id"]: row for row in artifact["blocked_diagnostic_rows"]}
    assert diagnostics["exp6493"]["diagnostic_complete"] is True
    assert diagnostics["exp6493"]["failed_field"] == "trajectory_signal_ready_score"
    assert diagnostics["exp6493"]["failed_expected"] == 1.0
    assert diagnostics["exp6493"]["failed_observed"] == 0.0
    assert diagnostics["exp6493"]["failed_evidence_sha256"].startswith("sha256:")
    assert diagnostics["exp6500"]["failed_field"] == "arc_energy_alignment_ready_score"
    assert diagnostics["exp6500"]["diagnostic_complete"] is True


def test_scenario_capstone_6501_recomputes_headlines_from_rows() -> None:
    """SCENARIO-CAPSTONE-6501-ROWS-AND-CLAIMS: rows drive claim decisions."""

    artifact = _artifact()
    headlines = {row["experiment_id"]: row for row in artifact["headline_recomputation_rows"]}

    trajectory = headlines["exp6490"]
    assert trajectory["recomputed"]["best_learned_balanced_accuracy"] == 0.880952
    assert trajectory["recomputed"]["best_shortcut_control_id"] == "checkpoint"
    assert trajectory["recomputed"]["harmful_flip_count"] == 6
    assert trajectory["matches_reported"] is True

    factor_replay = headlines["exp6492"]
    assert factor_replay["recomputed"]["accepted_model_factor_count"] == 0
    assert factor_replay["recomputed"]["factor_causal_audit_complete_score_from_rows"] == 1.0
    assert factor_replay["recomputed"]["causal_factor_signal_ready_score_from_rows"] == 0.0

    csl = artifact["csl_integrity_audit"]
    assert csl["chronology_valid"] is True
    assert csl["dose_valid"] is True
    assert csl["future_support_valid"] is True
    assert csl["held_future_benefit"] is False
    assert csl["continuous_learning_claim_eligible_from_rows"] is False

    arc = artifact["arc_integrity_audit"]
    assert arc["registry_precheck_passed"] is True
    assert arc["roster_game_count"] == 25
    assert arc["source_access_count"] == 0
    assert arc["per_game_adapter_count"] == 0
    assert arc["no_new_solve_claim"] is True
    assert arc["arc_energy_alignment_ready_score_from_rows"] == 0.0


def test_scenario_capstone_6501_claims_gaps_handoff_and_attacks() -> None:
    """SCENARIO-CAPSTONE-6501-RETIREMENT-HANDOFF-ATTACKS: claims fail closed."""

    artifact = _artifact()

    assert artifact["trajectory_energy_claim_eligible"]["eligible"] is False
    assert "checkpoint_shortcut" in artifact["trajectory_energy_claim_eligible"]["reasons"]
    assert artifact["continuous_learning_claim_eligible"]["eligible"] is False
    assert "held_future_benefit_failed" in artifact["continuous_learning_claim_eligible"]["reasons"]
    assert artifact["arc_policy_claim_eligible"]["eligible"] is False
    assert "arc_alignment_gate_closed" in artifact["arc_policy_claim_eligible"]["reasons"]
    assert artifact["hardware_claim_eligible"]["eligible"] is False

    gaps = {row["gap_id"]: row for row in artifact["gap_closure_rows"]}
    assert (
        gaps["leakage_resistant_authentic_energy"]["disposition"]
        == "open_retire_compact_learned_energy"
    )
    assert gaps["executed_continuous_self_learning"]["execution_complete"] is True
    assert gaps["executed_continuous_self_learning"]["claim_eligible"] is False
    assert gaps["arc_decision_alignment"]["disposition"] == "open_alignment_null_policy_deferred"

    handoff = artifact["v561_handoff"]
    assert (
        handoff["recommended_branch_id"]
        == "retire_learned_energy_defer_arc_policy_research_exact_structure"
    )
    assert len(handoff["recommended_branches"]) == 1
    assert "retire compact learned trajectory energy" in handoff["retire_actions"]
    assert "defer ARC policy A/B" in handoff["defer_actions"]
    assert (
        handoff["hardware_access_boundary"]
        == "no_special_hardware_claim_without_authenticated_local_device"
    )

    attacks = {row["attack_id"]: row for row in artifact["adversarial_attack_matrix"]}
    for attack_id in mod.ATTACK_IDS:
        assert attacks[attack_id]["detected"] is True
        assert attacks[attack_id]["fail_closed"] is True
        assert attacks[attack_id]["promoted_claim"] is False
    assert artifact["exclusion_manifest_receipt"]["mechanically_required_additions"] == []
    assert artifact["protected_files_unchanged"]["research_roadmap_yaml_unchanged"] is True


def test_scenario_capstone_6501_validation_edges(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-6501-FIELD-PRINCIPLES: schema checks fail closed."""

    artifact = _artifact()
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["documentation_reconciliation_rows"][0]["status"] == "updated"
    default_commands = [row["command"] for row in mod.tests_run_receipts(None)]
    assert mod.GATE_AUDIT_COMMAND in default_commands
    assert mod.PRIOR_FAILURE_COMMAND in default_commands
    assert mod.EXCLUSION_LINT_COMMAND in default_commands
    assert not any("gate_contract_audit.py" in command for command in default_commands)
    assert not any(
        command.endswith("exclusion_manifest_lint.py --help") for command in default_commands
    )
    for command in default_commands:
        script_parts = [
            part for part in command.split() if part.startswith("scripts/") and part.endswith(".py")
        ]
        for script_part in script_parts:
            assert (REPO / script_part).is_file(), command
    assert any(
        row["status"] == "deferred_by_operator_stop_rule"
        for row in artifact["documentation_reconciliation_rows"]
    )

    bad = copy.deepcopy(artifact)
    del bad["status"]
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert any("missing required fields" in error for error in mod.validate_artifact(bad))

    bad = copy.deepcopy(artifact)
    bad["honest_verdict"] = "ok"
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "honest_verdict lacks terminal prefix" in mod.validate_artifact(bad)

    bad = copy.deepcopy(artifact)
    bad["v560_capstone_ready_score"] = 0.0
    bad["gate_check_summary"]["capstone_audit_complete"] = True
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "ready score and gate summary disagree" in mod.validate_artifact(bad)

    bad = copy.deepcopy(artifact)
    bad["unexpected"] = True
    bad["field_principles"] = {}
    bad["field_provenance"] = {}
    bad["inference_substrate"] = "wrong"
    bad["verifier_is_oracle"] = False
    bad["milestone_manifest_rows"] = []
    bad["v561_handoff"] = {"recommended_branches": []}
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    errors = mod.validate_artifact(bad)
    assert any("unexpected fields" in error for error in errors)
    assert "field_principles must cover exactly required fields" in errors
    assert "field_provenance must cover exactly required fields" in errors
    assert "inference_substrate mismatch" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "milestone_manifest_rows must contain 13 rows" in errors
    assert "v561_handoff must contain exactly one recommended branch" in errors

    bad = copy.deepcopy(artifact)
    bad["milestone_manifest_rows"][0]["classification"] = "bad_class"
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert any("unknown classifications" in error for error in mod.validate_artifact(bad))

    bad = copy.deepcopy(artifact)
    bad["reproducibility_checksum"] = "sha256:" + "1" * 64
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert "unloadable artifact" in mod.validate_artifact(malformed)[0]

    written = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "experiment_6501_tmp.json",
        write=True,
        duration_s=1.0,
        tests_run=[{"command": "focused", "exit_code": 0}],
    )
    assert written["status"] == "complete_v560_capstone_reconciled"
    assert (tmp_path / "experiment_6501_tmp.json").is_file()

    with pytest.raises(ValueError, match="forced validation error"):
        original = mod.validate_artifact
        try:
            mod.validate_artifact = lambda _value: ["forced validation error"]  # type: ignore[method-assign]
            mod.build_artifact(repo_root=REPO, write=False, duration_s=1.0)
        finally:
            mod.validate_artifact = original  # type: ignore[method-assign]

    rc = mod.main(["--date", "20260821", "--output", str(tmp_path / "main.json")])
    assert rc == 0
    assert (tmp_path / "main.json").is_file()


def test_scenario_capstone_6501_defensive_helper_edges(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-6501-GATES: malformed inputs fail closed."""

    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod._exp_number("no-experiment") is None
    assert mod._experiment_id(None) == "unknown"
    assert mod._type_name(True) == "bool"
    assert mod._type_name(1) == "int"
    assert mod._type_name(1.0) == "float"
    assert mod._type_name("x") == "str"
    assert mod._type_name([]) == "list"
    assert mod._type_name({}) == "mapping"
    assert mod._type_name(object()) == "object"
    assert mod._status_text(None) == ""
    assert mod._is_invalid_payload(None) is False
    assert mod._readiness_fields(None) == {}
    assert mod._rows_from({"per_unit_rows": {"rows": [{"a": 1}, "bad"]}}) == [{"a": 1}]
    assert mod.selected_metrics_match({"a": 1}, []) is True
    assert mod.selected_metrics_match({"a": 1}, {"a": 2}) is False
    assert mod.classification_reason("missing", None, False).startswith("no artifact")
    assert mod.blocked_diagnostic_complete(None) is False
    assert mod.blocked_diagnostic_complete({"blocked_diagnostic_contract": []}) is False

    fake_repo = tmp_path / "repo"
    (fake_repo / "results").mkdir(parents=True)
    (fake_repo / "ops").mkdir()
    (fake_repo / "research-roadmap.yaml").write_text(
        """
tasks:
  - not-a-mapping
  - id: exp6488-fixture
    deliverable: results/experiment_6488_fixture.json
  - id: exp6489-fixture
    deliverable: results/experiment_6489_fixture.json
""",
        encoding="utf-8",
    )
    (fake_repo / "results/experiment_6489_fixture.json").write_text("{", encoding="utf-8")
    tasks = mod.load_v560_tasks(fake_repo)
    assert [task["id"] for task in tasks] == ["exp6488-fixture", "exp6489-fixture"]
    payloads, seed_rows = mod.load_payloads(fake_repo, tasks)
    assert payloads == {}
    classified = mod.classify_manifest_rows(seed_rows, [])
    assert {row["experiment_id"]: row["classification"] for row in classified} == {
        "exp6488": "missing",
        "exp6489": "invalid",
    }

    seed_rows = [
        {
            "task_id": "exp6488-upstream",
            "experiment_id": "exp6488",
            "actual_path": "results/up.json",
            "sha256": "sha256:abc",
        }
    ]
    gate_tasks = [
        {
            "id": "exp6489-downstream",
            "gated_on": [
                "bad-gate",
                {
                    "upstream": "exp6488-upstream",
                    "artifact_field": "score",
                    "op": "==",
                    "value": 1.0,
                },
                {
                    "upstream": "exp6488-upstream",
                    "artifact_field": "missing",
                    "op": "==",
                    "value": 1.0,
                },
            ],
        }
    ]
    gates = mod.build_gate_contract_rows(
        fake_repo, gate_tasks, seed_rows, {"exp6488": {"score": 1}}
    )
    assert gates[0]["result"] == "wrong_type"
    assert gates[1]["result"] == "broken_gate_contract"

    incomplete = mod.claim_eligibility(
        [{"experiment_id": "exp6493", "classification": "complete"}],
        [
            {
                "experiment_id": "exp6490",
                "recomputed": {
                    "trajectory_signal_ready_score_from_rows": 1.0,
                    "surviving_shortcut_ids": [],
                    "harmful_flip_count": 0,
                },
            },
            {
                "experiment_id": "exp6492",
                "recomputed": {"causal_factor_signal_ready_score_from_rows": 1.0},
            },
            {
                "experiment_id": "exp6496",
                "recomputed": {
                    "csl_execution_complete_score_from_rows": 0.0,
                    "held_future_benefit": True,
                },
            },
            {
                "experiment_id": "exp6499",
                "recomputed": {"arc_energy_alignment_ready_score_from_rows": 1.0},
            },
        ],
        {"exp6498": {"continuous_learning_claim_eligible": True}},
    )
    assert "csl_execution_incomplete" in incomplete[1]["reasons"]

    prior_rows = mod.prior_failure_rows(
        [{"id": "exp6488-fixture", "prior_failures": ["bad"]}],
        [{"task_id": "exp6488-fixture", "experiment_id": "exp6488"}],
        {},
    )
    assert prior_rows == []

    (fake_repo / "ops/exclusion_manifest.yaml").write_text(":", encoding="utf-8")
    assert mod.exclusion_manifest_receipt(fake_repo)["load_error"] is not None
