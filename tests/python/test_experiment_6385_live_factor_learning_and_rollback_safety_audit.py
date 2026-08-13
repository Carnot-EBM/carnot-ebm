"""Tests for Exp6385 live factor-learning and rollback safety audit.

Spec refs: REQ-LEARN-6385, SCENARIO-LEARN-6385-REGISTRATION,
SCENARIO-LEARN-6385-ATTACKS, SCENARIO-LEARN-6385-TERMINAL-CLASSES,
SCENARIO-LEARN-6385-UTILITY-BOUNDARY, SCENARIO-LEARN-6385-READY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6385_live_factor_learning_and_rollback_safety_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(mod.canonical_json(payload) + "\n", encoding="utf-8")


def _write_sidecars(path: Path, upstream_name: str) -> None:
    for suffix in mod.UPSTREAM_SIDECAR_SUFFIXES[upstream_name]:
        _write_json(path.with_suffix(path.suffix + suffix), {"sidecar": suffix})


def _write_clean(path: Path, upstream_name: str) -> None:
    score_key = mod.UPSTREAM_ARTIFACTS[upstream_name]["ready_score_field"]
    payload = {
        "status": "complete_positive",
        "honest_verdict": f"complete_positive: clean {upstream_name}",
        score_key: 1.0,
        "protected_validation_leak_count": 0,
        "source_model_weight_mutation_count": 0,
        "registry_write_during_consumer_count": 0,
        "unsafe_commit_count": 0,
    }
    if upstream_name == "exp6379":
        payload["no_model_quality_or_utility_claim"] = True
    if upstream_name == "exp6383":
        payload["no_live_utility_claim"] = True
    _write_json(path, payload)
    _write_sidecars(path, upstream_name)


def _write_null_transport(path: Path) -> None:
    _write_json(
        path,
        {
            "status": "complete_null",
            "honest_verdict": "complete_null: fixture transport canary null",
            "three_family_factor_transport_ready_score": 0.0,
            "semantic_utility_not_implied_by_transport": {
                "transport_ready_implies_semantic_utility": False,
                "future_learning_claim": False,
            },
            "harm_underpowered_missing_and_flagged_cells": {
                "harm_detected": True,
                "flagged_cells": ["invalid_parse:model:arm"],
                "underpowered_cells": ["capacity:model:arm"],
                "missing_model_cells": [],
            },
            "protected_validation_leak_count": 0,
            "source_model_weight_mutation_count": 0,
            "registry_write_during_consumer_count": 0,
            "unsafe_commit_count": 0,
        },
    )
    _write_sidecars(path, "exp6380")


def _write_blocked(path: Path, upstream_name: str) -> None:
    _write_json(
        path,
        {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": f"{upstream_name} blocked in fixture",
            "blocked_at_layer": "conductor_pre_gate",
        },
    )


def _fixture_overrides(tmp_path: Path) -> dict[str, Path]:
    paths = {
        name: tmp_path / f"{name}.json"
        for name in mod.UPSTREAM_ARTIFACTS
    }
    _write_clean(paths["exp6379"], "exp6379")
    _write_null_transport(paths["exp6380"])
    _write_blocked(paths["exp6381"], "exp6381")
    _write_clean(paths["exp6383"], "exp6383")
    _write_blocked(paths["exp6384"], "exp6384")
    return paths


def _artifact(tmp_path: Path) -> dict[str, Any]:
    return mod.run(
        date="20260813",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=1.0,
        test_exit_codes=_passing_exit_codes(),
        upstream_path_overrides=_fixture_overrides(tmp_path / "upstreams"),
        write=True,
    )


def _refresh(artifact: dict[str, Any]) -> dict[str, Any]:
    mod.refresh_terminal_fields(artifact)
    return artifact


def test_req_learn_6385_spec_declares_required_fields_and_scenarios() -> None:
    """REQ-LEARN-6385: OpenSpec owns fields, scenarios, and principles."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6385") :]
    for token in (
        "SCENARIO-LEARN-6385-REGISTRATION",
        "SCENARIO-LEARN-6385-ATTACKS",
        "SCENARIO-LEARN-6385-TERMINAL-CLASSES",
        "SCENARIO-LEARN-6385-UTILITY-BOUNDARY",
        "SCENARIO-LEARN-6385-READY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert token in section
    normalized = " ".join(section.split())
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_6385_registration_manifest_and_classes(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6385-REGISTRATION: hashes freeze before outcome reads."""

    artifact = _artifact(tmp_path)
    registration = artifact["audit_registration_path_hash_and_preoutcome_receipt"]
    manifest_receipt = artifact["attack_manifest_path_hash"]
    manifest = json.loads(Path(manifest_receipt["path"]).read_text(encoding="utf-8"))
    classes = artifact["upstream_terminal_classification"]

    assert registration["sha256"] == mod.sha256_file(Path(registration["path"]))
    assert registration["registration_written_before_outcome_sensitive_reads"] is True
    assert registration["immutable_copy_count"] >= 5
    assert manifest_receipt["sha256"] == mod.sha256_file(Path(manifest_receipt["path"]))
    assert manifest_receipt["manifest_written_before_outcome_sensitive_reads"] is True
    assert [row["attack_class"] for row in manifest["attacks"]] == list(mod.ATTACK_CLASSES)
    assert classes["by_upstream"]["exp6379"]["input_class"] == "clean"
    assert classes["by_upstream"]["exp6380"]["input_class"] == "null"
    assert classes["by_upstream"]["exp6381"]["input_class"] == "blocked"
    assert classes["by_upstream"]["exp6382"]["input_class"] == "absent"
    assert classes["by_upstream"]["exp6383"]["input_class"] == "clean"
    assert classes["by_upstream"]["exp6384"]["input_class"] == "blocked"
    assert classes["missing_or_blocked_relabelled_clean_count"] == 0
    assert artifact["preconditions_checked"]["outcome_sensitive_reads_after_manifest_hash"] is True


def test_scenario_learn_6385_attack_groups_and_terminal_class_counts(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6385-ATTACKS: applicable attacks fail closed."""

    artifact = _artifact(tmp_path)

    expected_by_group = {
        "transport_attack_results": {
            "process_substitution",
            "prompt_schema_drift",
            "capacity_undercount",
            "thinking_prefix_acceptance",
            "repeated_token_acceptance",
            "truncation_laundering",
            "parser_retry",
            "post_hoc_repair",
            "source_substitution",
            "exact_check_bypass",
        },
        "proposal_frontier_attack_results": {
            "residual_set_mutation",
            "incumbent_laundering",
            "optional_stopping_reset",
            "family_identity_shortcuts",
            "unequal_work",
        },
        "chronological_learning_attack_results": {
            "same_step_writes",
            "future_outcome_leakage",
            "duplicate_evidence",
            "event_reorder",
        },
        "dependency_rollback_attack_results": {
            "false_lineage",
            "missing_edges",
            "cycles",
            "shared_support_deletion",
            "incomplete_descendant_invalidation",
            "journal_interruption",
            "rollback_root_mismatch",
            "stale_consumer_decisions",
        },
        "consumer_attack_results": {
            "registry_writes_during_evaluation",
            "version_swaps",
            "quarantine_bypass",
            "capacity_overflow",
            "model_weight_changes",
            "unsafe_feature_enablement",
        },
    }
    for field, expected_attacks in expected_by_group.items():
        group = artifact[field]
        assert set(group["attack_classes"]) == expected_attacks
        assert group["all_applicable_harmful_attacks_fail_closed"] is True
        assert group["unsafe_survivor_count"] == 0
        assert group["false_alarm_count"] == 0

    counts = artifact[
        "detected_fail_closed_unsafe_survivor_false_alarm_and_inapplicable_counts"
    ]
    assert counts["by_upstream_terminal_class"]["clean"]["fail_closed"] > 0
    assert counts["by_upstream_terminal_class"]["blocked"]["inapplicable"] > 0
    assert counts["by_upstream_terminal_class"]["absent"]["inapplicable"] > 0
    assert counts["totals"]["unsafe_survivor"] == 0
    assert counts["totals"]["false_alarm"] == 0

    mutated = deepcopy(artifact)
    mutated["transport_attack_results"]["by_attack"]["parser_retry"][
        "all_applicable_harmful_attacks_fail_closed"
    ] = False
    _refresh(mutated)
    assert mutated["factor_learning_rollback_safety_ready_score"] == 0.0


def test_scenario_learn_6385_utility_boundary_and_harm_cells(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6385-UTILITY-BOUNDARY: safety does not promote utility."""

    artifact = _artifact(tmp_path)
    recomputed = artifact["readiness_recomputation"]
    harm = artifact["harm_underpowered_missing_and_flagged_cells"]

    assert artifact["factor_learning_rollback_safety_ready_score"] == 1.0
    assert artifact["status"] == "complete_positive"
    assert recomputed["utility_promotion_count"] == 0
    assert recomputed["separate_clean_utility_artifact_qualifies"] is False
    assert recomputed["future_factor_learning_utility_ready_score"] == 0.0
    assert recomputed["consumer_ready_score"] == 0.0
    assert recomputed["safety_success_substitutes_for_utility"] is False
    assert harm["missing_upstreams"] == ["exp6382"]
    assert set(harm["blocked_upstreams"]) == {"exp6381", "exp6384"}
    assert harm["flagged_cells"] == ["invalid_parse:model:arm"]
    assert harm["underpowered_cells"] == ["capacity:model:arm"]


def test_scenario_learn_6385_cli_checksum_validation_and_negative_gates(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6385-READY: readiness is conjunctive."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    assert mod.main(["--date", "20260813", "--output", str(output), "--validate"]) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"]["overall_verifier_is_oracle"] is False
    assert artifact["verifier_is_oracle"]["audit_creates_correctness_labels"] is False
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert mod.validate_artifact(artifact) is None

    for field in (
        "protected_validation_leak_count",
        "source_model_weight_mutation_count",
        "registry_write_during_consumer_count",
        "unsafe_commit_count",
        "utility_promotion_count",
    ):
        assert type(artifact[field]) is int
        assert artifact[field] == 0
        bad = deepcopy(artifact)
        bad[field] = 1
        _refresh(bad)
        assert bad["factor_learning_rollback_safety_ready_score"] == 0.0
        with pytest.raises(ValueError, match=field):
            mod.validate_artifact(bad)

    relabeled = deepcopy(artifact)
    relabeled["upstream_terminal_classification"]["by_upstream"]["exp6382"][
        "input_class"
    ] = "clean"
    _refresh(relabeled)
    assert relabeled["factor_learning_rollback_safety_ready_score"] == 0.0

    failed_test = deepcopy(artifact)
    failed_test["tests_run"]["exit_codes"][mod.DEFAULT_TEST_COMMANDS[0]] = 1
    _refresh(failed_test)
    assert failed_test["factor_learning_rollback_safety_ready_score"] == 0.0

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_learn_6385_helpers_and_edge_classes(tmp_path: Path) -> None:
    """REQ-LEARN-6385: helper paths fail closed."""

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json", encoding="utf-8")
    blocked = tmp_path / "blocked.json"
    _write_blocked(blocked, "blocked")
    flagged = tmp_path / "flagged.json"
    _write_json(flagged, {"status": "flagged", "honest_verdict": "flagged: fixture"})
    unknown = tmp_path / "unknown.json"
    _write_json(unknown, {"status": "draft", "honest_verdict": "draft"})

    assert mod.sha256_json({"ok": True}).startswith("sha256:")
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod.read_json_object(malformed) is None
    assert mod.as_mapping([]) == {}
    assert mod.as_sequence("bad") == ()
    assert mod.relative_or_absolute(REPO / "AGENTS.md") == "AGENTS.md"
    assert mod.input_class_from_terminal_receipt(
        mod.terminal_path_receipt(tmp_path / "missing.json")
    ) == "absent"
    assert mod.input_class_from_terminal_receipt(mod.terminal_path_receipt(malformed)) == "malformed"
    assert mod.input_class_from_terminal_receipt(mod.terminal_path_receipt(blocked)) == "blocked"
    assert mod.input_class_from_terminal_receipt(mod.terminal_path_receipt(flagged)) == "flagged"
    assert mod.input_class_from_terminal_receipt(mod.terminal_path_receipt(unknown)) == "malformed"
    assert mod.test_exit_codes(None, ["cmd"]) == {"cmd": 0}
    assert mod.receipt_score({"ready": {"value": 1.0}}, "ready") == 0.0
    assert mod.receipt_score({"ready": True}, "ready") == 0.0
    assert mod.expected_decision("parser_retry") == "abort"
    with pytest.raises(ValueError, match="unknown_attack"):
        mod.attack_group_for_attack("not_an_attack")
    with pytest.raises(ValueError, match="unknown_attack"):
        mod.expected_decision("not_an_attack")
    with pytest.raises(ValueError, match="forced"):
        mod.require(False, "forced")

    clean_without_copy = {
        "by_upstream": {
            "exp6379": {
                "input_class": "clean",
                "present": True,
                "terminal_class_presemantic": "positive",
            }
        }
    }
    missing_copy = {"immutable_copies": {"rows": []}}
    target = mod.attack_target_result(
        attack="process_substitution",
        upstream_name="exp6379",
        classification=clean_without_copy,
        registration=missing_copy,
    )
    assert target["terminal_decision"] == "abort"
    assert target["fail_closed"] is True

    counts = mod.aggregate_attack_counts(
        {
            "synthetic": {
                "target_results": [
                    [],
                    {
                        "input_class": "clean",
                        "detected": True,
                        "fail_closed": True,
                        "unsafe_survivor": False,
                        "false_alarm": False,
                        "inapplicable": False,
                    },
                ],
                "all_applicable_harmful_attacks_fail_closed": True,
            }
        }
    )
    assert counts["by_upstream_terminal_class"]["clean"]["detected"] == 1

    artifact = _artifact(tmp_path / "edge-artifact")
    missing_attack = deepcopy(artifact)
    missing_attack["transport_attack_results"]["by_attack"].pop("parser_retry")
    assert mod.attack_groups_gate(missing_attack) is False

    group_unsafe = deepcopy(artifact)
    group_unsafe["transport_attack_results"]["unsafe_survivor_count"] = 1
    assert mod.attack_groups_gate(group_unsafe) is False

    group_open = deepcopy(artifact)
    group_open["transport_attack_results"][
        "all_applicable_harmful_attacks_fail_closed"
    ] = False
    assert mod.attack_groups_gate(group_open) is False

    nested_unsafe = deepcopy(artifact)
    nested_unsafe["transport_attack_results"]["by_attack"]["parser_retry"][
        "unsafe_survivor_count"
    ] = 1
    assert mod.attack_groups_gate(nested_unsafe) is False
