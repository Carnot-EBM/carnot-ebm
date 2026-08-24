"""Tests for the Exp6580 V572 source and joint-method protocol.

Spec refs: REQ-REPORT-6580, REQ-REPORT-6580-PRECONDITIONS,
REQ-REPORT-6580-SOURCES, REQ-REPORT-6580-FIXTURES,
REQ-REPORT-6580-SOURCE-UNITS, REQ-REPORT-6580-PROMPTS-CONTEXTS,
REQ-REPORT-6580-ARMS, REQ-REPORT-6580-GATES,
REQ-REPORT-6580-ATTACKS, REQ-REPORT-6580-ATOMIC,
SCENARIO-REPORT-6580-SOURCES, SCENARIO-REPORT-6580-FIXTURES,
SCENARIO-REPORT-6580-PROTOCOL, SCENARIO-REPORT-6580-ATTACKS,
SCENARIO-REPORT-6580-ATOMIC.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6580_v572_source_and_joint_method_protocol as mod
from scripts import adversarial_verify


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/research-reporting/spec.md"
TESTS_RUN = [{"command": "focused Exp6580 fixture", "exit_code": 0, "duration_s": 0.01}]


def _report(tmp_path: Path) -> dict[str, Any]:
    return mod.build_report(
        REPO,
        date="20260824",
        duration_s=1.0,
        tests_run=TESTS_RUN,
        output_path=tmp_path / "experiment_6580.json",
    )


def _rehash(payload: dict[str, Any]) -> dict[str, Any]:
    payload["reproducibility_checksum"] = mod.artifact_checksum(payload)
    return payload


def test_req_report_6580_spec_declares_protocol_fields_and_scenarios() -> None:
    """REQ-REPORT-6580: the spec anchors exist before implementation."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-6580") :]

    for anchor in (
        "REQ-REPORT-6580-PRECONDITIONS",
        "REQ-REPORT-6580-SOURCES",
        "REQ-REPORT-6580-FIXTURES",
        "REQ-REPORT-6580-SOURCE-UNITS",
        "REQ-REPORT-6580-PROMPTS-CONTEXTS",
        "REQ-REPORT-6580-ARMS",
        "REQ-REPORT-6580-GATES",
        "REQ-REPORT-6580-ATTACKS",
        "REQ-REPORT-6580-ATOMIC",
        "SCENARIO-REPORT-6580-SOURCES",
        "SCENARIO-REPORT-6580-FIXTURES",
        "SCENARIO-REPORT-6580-PROTOCOL",
        "SCENARIO-REPORT-6580-ATTACKS",
        "SCENARIO-REPORT-6580-ATOMIC",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert anchor in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_6580_sources_bind_methods_and_exclude_claims(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6580-SOURCES: receipts bind hooks, not claims."""

    report = _report(tmp_path)
    receipts = {row["arxiv_id"]: row for row in report["primary_source_receipts"]}
    non_imported = {row["arxiv_id"]: row for row in report["non_imported_claim_rows"]}

    assert set(receipts) == set(mod.REQUIRED_ARXIV_IDS)
    assert set(non_imported) == set(mod.REQUIRED_ARXIV_IDS)
    assert report["gate_check_summary"]["checks_closed"] is True
    for arxiv_id, receipt in receipts.items():
        assert receipt["stable_url"] == f"https://arxiv.org/abs/{arxiv_id}"
        assert receipt["primary_source_url_bound"] is True
        assert receipt["local_reference_sha256"].startswith("sha256:")
        assert receipt["local_reference_contains_arxiv_id"] is True
        assert receipt["local_cache_hash_status"] in {"not_cached", "cached"}
        assert receipt["method_hook"].startswith("Carnot hook:")
        assert receipt["non_imported_claim"]
        assert receipt["imported_as"] == "bounded_method_control"
        assert receipt["receipt_hash"].startswith("sha256:")
        assert non_imported[arxiv_id]["claim_imported_into_carnot_evidence"] is False
    assert all(
        row["replacement_authority"] == "local_exact_replay" for row in non_imported.values()
    )


def test_scenario_report_6580_fixtures_replay_and_source_units_are_frozen(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6580-FIXTURES: Exp6574 expectations are preserved."""

    report = _report(tmp_path)
    replay = {row["replay_case_id"]: row for row in report["joint_method_replay_rows"]}
    units = report["source_unit_manifest"]["units"]

    assert list(replay) == list(mod.REPLAY_CASE_FIXTURES)
    assert replay["single_hop"]["source_fixture_id"] == "valid_single_hop"
    assert replay["valid_multi_hop"]["source_fixture_id"] == "valid_two_hop"
    assert replay["cycle"]["source_fixture_id"] == "cyclic_dependency"
    assert replay["ownership"]["source_fixture_id"] == "duplicate_node"
    assert replay["domination"]["source_fixture_id"] == "contradictory_nodes"
    assert replay["ambiguity"]["source_fixture_id"] == "disconnected_graph"
    assert all(row["expectation_preserved"] is True for row in replay.values())
    assert all(row["unsafe_release"] is False for row in replay.values())
    assert replay["single_hop"]["observed_action"] == "release"
    assert replay["valid_multi_hop"]["observed_action"] == "release"
    assert replay["missing_hop"]["observed_action"] == "abstain"
    assert replay["wrong_span"]["observed_action"] == "abstain"

    assert report["source_unit_manifest"]["selected_without_model_outcomes"] is True
    assert report["source_unit_manifest"]["manifest_hash"].startswith("sha256:")
    assert {unit["case_kind"] for unit in units} >= {
        "single_hop",
        "multi_hop",
        "unsupported",
        "ambiguity",
    }
    assert {unit["split"] for unit in units} == {"train", "calibration", "held"}
    for unit in units:
        assert unit["source_bytes_sha256"] == mod.sha256_text(unit["exact_source_bytes"])
        assert unit["inclusion_rule"].startswith("pre_outcome_")
        assert unit["model_outcome_fields_accessed"] is False


def test_scenario_report_6580_protocol_freezes_prompt_contexts_and_arms(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6580-PROTOCOL: families and arms receive matched work."""

    report = _report(tmp_path)
    prompt = report["prompt_seed_budget_contract"]
    contexts = {row["context_id"]: row for row in report["context_control_contract"]["contexts"]}
    proof = report["proof_arm_contract"]
    learning = report["learning_arm_contract"]
    downstream = {
        (row["owner_task_id"], row["artifact_field"])
        for row in report["downstream_gate_field_rows"]
    }

    assert prompt["family_neutral_prompt"] == mod.FAMILY_NEUTRAL_PROMPT
    assert prompt["prompt_sha256"] == mod.sha256_text(mod.FAMILY_NEUTRAL_PROMPT)
    assert prompt["raw_before_derived_write_order"] is True
    assert prompt["failure_retention_required"] is True
    assert prompt["fresh_process_per_family"] is True
    assert set(prompt["one_family_task_mapping"]) == set(mod.MODEL_TASK_FAMILIES)
    assert all(row["prompt_sha256"] == prompt["prompt_sha256"] for row in prompt["family_rows"])
    assert len({row["token_budget_hash"] for row in prompt["family_rows"]}) == 1

    assert contexts["clean"]["byte_count"] == len(contexts["clean"]["context_bytes"].encode())
    assert (
        contexts["prior_repair"]["byte_count"] == contexts["neutral_length_matched"]["byte_count"]
    )
    assert contexts["neutral_length_matched"]["length_matched_to"] == "prior_repair"
    assert report["context_control_contract"]["fresh_context_required"] is True
    assert report["context_control_contract"]["context_threshold_shift_credit_allowed"] is False

    assert set(proof["arms"]) == {"no_filter", "atomic_support", "joint_graph"}
    assert proof["exact_registry"]["release_authority"] == "compiler_plus_exact_fixture_checker"
    assert proof["exact_registry"]["llm_judge_release_authority"] is False
    assert len({arm["matched_input_hash"] for arm in proof["arms"].values()}) == 1
    assert len({arm["charged_cost_units"] for arm in proof["arms"].values()}) == 1
    assert proof["semantic_block_conditions"]["ambiguity_stop_required"] is True

    assert set(learning["arms"]) == {
        "frozen_no_update",
        "uniform_verified_replay",
        "graph_potts",
        "protected_core",
        "conflict_routed_specialist",
    }
    assert learning["prospective_only"] is True
    assert learning["weights_frozen"] is True
    assert learning["arms"]["conflict_routed_specialist"]["source_arxiv_id"] == "2608.21044"
    assert learning["arms"]["protected_core"]["trusted_core_mutable"] is False

    assert (
        "exp6580-v572-source-and-joint-method-protocol",
        "v572_source_method_ready_score",
    ) in downstream
    assert (
        "exp6580-v572-source-and-joint-method-protocol",
        "v572_joint_method_ready_score",
    ) in downstream


def test_scenario_report_6580_attacks_and_readiness_reducer_fail_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6580-ATTACKS: leakage and authority drift block readiness."""

    report = _report(tmp_path)
    attacks = {row["attack_id"]: row for row in report["attack_rows"]}

    assert set(attacks) == set(mod.REQUIRED_ATTACK_IDS)
    assert all(row["passed"] is True for row in attacks.values())
    assert all(row["candidate_source_ready_score"] == 0.0 for row in attacks.values())
    assert all(row["candidate_joint_ready_score"] == 0.0 for row in attacks.values())
    assert report["v572_source_method_ready_score"] == 1.0
    assert report["v572_joint_method_ready_score"] == 1.0
    assert mod.readiness_reducer(report)["source_ready"] is True
    assert mod.readiness_reducer(report)["joint_ready"] is True

    missing_unsupported = deepcopy(report)
    missing_unsupported["source_unit_manifest"]["units"] = [
        row
        for row in missing_unsupported["source_unit_manifest"]["units"]
        if row["case_kind"] != "unsupported"
    ]
    assert mod.readiness_reducer(missing_unsupported)["source_ready_score"] == 0.0

    family_prompt = deepcopy(report)
    family_prompt["prompt_seed_budget_contract"]["family_rows"][0]["prompt_sha256"] = "sha256:drift"
    assert mod.readiness_reducer(family_prompt)["source_ready_score"] == 0.0

    llm_authority = deepcopy(report)
    llm_authority["proof_arm_contract"]["exact_registry"]["llm_judge_release_authority"] = True
    assert mod.readiness_reducer(llm_authority)["joint_ready_score"] == 0.0

    changed_fixture = deepcopy(report)
    changed_fixture["joint_method_replay_rows"][0]["expected_action"] = "abstain"
    assert mod.readiness_reducer(changed_fixture)["joint_ready_score"] == 0.0

    spelling_drift = deepcopy(report)
    spelling_drift["downstream_gate_field_rows"][0]["artifact_field"] = "v572_source_method_ready"
    assert mod.readiness_reducer(spelling_drift)["source_ready_score"] == 0.0


def test_scenario_report_6580_atomic_artifact_and_no_llm_substrate_validate(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6580-ATOMIC: one null-class protocol artifact recomputes."""

    report = _report(tmp_path)
    output = tmp_path / "result.json"
    receipt = mod.atomic_write_report(output, report)
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert receipt["atomic_replace"] is True
    assert receipt["output_sha256"] == mod.sha256_file(output)
    assert loaded == report
    assert report["status"] == "complete_v572_source_and_joint_method_protocol_ready"
    assert report["honest_verdict"].startswith(
        "complete_v572_source_and_joint_method_protocol_ready:"
    )
    assert report["verdict_class"] is None
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["verifier_is_oracle"] is True
    assert report["protected_files_unchanged"]["all_unchanged"] is True
    assert report["reproducibility_checksum"] == mod.artifact_checksum(report)
    assert set(report["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert mod.validate_report(report) == []

    classification = adversarial_verify._classify_inference_substrate(report)
    verification = adversarial_verify.verify_artifact(output)
    assert classification["kind"] == adversarial_verify.SUBSTRATE_KIND_NO_LLM
    assert classification["matched_value"] == mod.INFERENCE_SUBSTRATE
    assert verification["flag_count"] == 0

    bad_class = _rehash({**deepcopy(report), "verdict_class": "positive"})
    assert "verdict_class must be null when ready" in mod.validate_report(bad_class)
    bad_checksum = {**deepcopy(report), "reproducibility_checksum": "sha256:stale"}
    assert "reproducibility_checksum mismatch" in mod.validate_report(bad_checksum)


def test_req_report_6580_defensive_helpers_and_validator_errors(tmp_path: Path) -> None:
    """REQ-REPORT-6580-ATOMIC: defensive helper paths fail visibly."""

    report = _report(tmp_path)

    assert mod.sha256_file(tmp_path / "missing.json") == "missing"
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json", encoding="utf-8")
    assert mod._read_json(bad_json) == {}  # noqa: SLF001
    assert mod._extract_section("no anchors here", "v572") == ""  # noqa: SLF001
    with pytest.raises(ValueError, match="neutral context base"):
        mod._neutral_context_for("short")  # noqa: SLF001

    cache = tmp_path / "data" / "paper-2608.21044.txt"
    cache.parent.mkdir()
    cache.write_text("cached", encoding="utf-8")
    assert mod._local_cache_hits(tmp_path, "2608.21044") == [cache]  # noqa: SLF001

    missing = deepcopy(report)
    del missing["status"]
    assert "missing required fields: status" in mod.validate_report(missing)

    bad_substrate = _rehash({**deepcopy(report), "inference_substrate": "wrong"})
    assert "inference_substrate mismatch" in mod.validate_report(bad_substrate)

    bad_oracle = _rehash({**deepcopy(report), "verifier_is_oracle": False})
    assert "verifier_is_oracle must be true" in mod.validate_report(bad_oracle)

    bad_duration = _rehash({**deepcopy(report), "duration_s": 0.0})
    assert "duration_s must be positive" in mod.validate_report(bad_duration)

    bad_source_score = _rehash({**deepcopy(report), "v572_source_method_ready_score": 0.0})
    assert "v572_source_method_ready_score mismatch" in mod.validate_report(bad_source_score)

    bad_joint_score = _rehash({**deepcopy(report), "v572_joint_method_ready_score": 0.0})
    assert "v572_joint_method_ready_score mismatch" in mod.validate_report(bad_joint_score)

    bad_protected = deepcopy(report)
    bad_protected["protected_files_unchanged"]["all_unchanged"] = False
    _rehash(bad_protected)
    assert "protected_files_unchanged failed" in mod.validate_report(bad_protected)

    bad_provenance = deepcopy(report)
    bad_provenance["field_provenance"].pop("status")
    _rehash(bad_provenance)
    assert "field_provenance missing required fields" in mod.validate_report(bad_provenance)

    with pytest.raises(ValueError, match="duration_s must be positive"):
        mod.atomic_write_report(tmp_path / "bad-output.json", bad_duration)
