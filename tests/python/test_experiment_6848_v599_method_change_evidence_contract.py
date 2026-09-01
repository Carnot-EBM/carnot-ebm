"""Focused tests for the immutable V599 method-change evidence contract.

Spec refs: REQ-REPORT-6848,
SCENARIO-REPORT-6848-SOURCE-DRIFT,
SCENARIO-REPORT-6848-PRODUCER-AUDITOR-DISAGREEMENT,
SCENARIO-REPORT-6848-MISSING-ARTIFACT,
SCENARIO-REPORT-6848-STALE-HARDCODED-PATH, and
SCENARIO-REPORT-6848-TERMINAL-NULL-PRESERVATION.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Callable
import json
from pathlib import Path

import pytest

from carnot import experiment_6848_v599_method_change_evidence_contract as exp


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def current_records() -> dict[str, dict]:
    """REQ-REPORT-6848: load each source once because Exp6842 is large."""

    return exp.load_source_records(REPO)


@pytest.fixture(scope="module")
def current_artifact(current_records: dict[str, dict]) -> dict:
    """REQ-REPORT-6848: build the current contract without writing tracked state."""

    return exp.build_artifact(
        REPO,
        run_date="20260901",
        duration_s=0.25,
        source_records=current_records,
    )


def test_req_report_6848_spec_declares_the_complete_contract() -> None:
    """REQ-REPORT-6848: OpenSpec owns every required field and scenario."""

    text = (REPO / exp.REPORT_SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("REQ-REPORT-6848", 1)[1].split("REQ-REPORT-6755", 1)[0]
    anchors = set(exp.spec_anchors(section))

    assert {
        "REQ-REPORT-6848",
        "SCENARIO-REPORT-6848-SOURCE-DRIFT",
        "SCENARIO-REPORT-6848-PRODUCER-AUDITOR-DISAGREEMENT",
        "SCENARIO-REPORT-6848-MISSING-ARTIFACT",
        "SCENARIO-REPORT-6848-STALE-HARDCODED-PATH",
        "SCENARIO-REPORT-6848-TERMINAL-NULL-PRESERVATION",
    } <= anchors
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in section
    assert exp.INFERENCE_SUBSTRATE in section
    assert exp.RESULT_PATH.as_posix() in section


def test_req_report_6848_current_contract_is_complete_and_row_supported(
    current_artifact: dict,
) -> None:
    """REQ-REPORT-6848: the real frozen inputs produce the exact downstream gate."""

    assert exp.validate_artifact(current_artifact) == []
    assert current_artifact["v599_evidence_contract_ready_score"] == 1
    assert current_artifact["verdict_class"] == "null"
    assert current_artifact["honest_verdict"].startswith("complete_")
    assert current_artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert current_artifact["verifier_is_oracle"] is False
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(current_artifact)
    assert set(current_artifact["field_principles"]) == set(current_artifact)
    assert {row["row_kind"] for row in current_artifact["rows"]} == {
        "source",
        "disputed_field",
        "method_decision",
        "conductor_skip",
    }


def test_scenario_report_6848_producer_auditor_disagreement_controls_readiness(
    current_artifact: dict,
) -> None:
    """SCENARIO-REPORT-6848-PRODUCER-AUDITOR-DISAGREEMENT uses Exp6847."""

    disagreements = {
        row["field"]: row for row in current_artifact["producer_auditor_disagreements"]
    }
    assert set(disagreements) == {
        "compile_parity",
        "obligation_pair_fixture_ready_score",
        "typed_obligation_program_ready_score",
    }
    assert disagreements["compile_parity"]["producer_value"] is True
    assert disagreements["compile_parity"]["auditor_value"] is False
    for field in (
        "obligation_pair_fixture_ready_score",
        "typed_obligation_program_ready_score",
    ):
        assert disagreements[field]["producer_value"] == 1
        assert disagreements[field]["auditor_value"] == 0
    assert all(row["controlling_authority"] == "exp6847" for row in disagreements.values())

    identity = current_artifact["exp6836_independent_recomputation"]
    assert identity["candidate_occurrence_count"] == 16
    assert identity["unique_candidate_identity_count"] == 8
    assert identity["duplicate_candidate_identity_count"] == 8
    assert identity["compile_receipt_count"] == 16
    assert identity["semantic_compile_checks_passed"] is True
    assert identity["candidate_identities_unique"] is False
    assert identity["recomputed_compile_parity"] is False
    assert identity["recomputed_typed_obligation_program_ready_score"] == 0


def test_req_report_6848_freezes_terminal_negative_and_blocked_results(
    current_artifact: dict,
) -> None:
    """REQ-REPORT-6848: readiness, benefit, and causal eligibility stay separate."""

    frozen = current_artifact["frozen_v598_results"]
    assert frozen["exp6837_resource_block"] == {
        "exclusive_gpu_leases": False,
        "live_canary_per_model": False,
        "scientific_row_count": 0,
        "compatibility_ready_score": 0,
    }
    assert frozen["exp6842_harmful_learning"] == {
        "held_future_rows": 540,
        "wins": 5,
        "losses": 69,
        "mean_effect_vs_no_memory": -0.118519,
        "continuous_self_learning_ready_score": 0.0,
    }
    assert frozen["exp6844_zero_headroom"] == {
        "action_row_count": 60,
        "zero_headroom_action_row_count": 60,
        "nonzero_headroom_action_row_count": 0,
        "supervisor_effect_eligible_score": 0,
    }
    assert frozen["exp6845_zero_obligations"] == {
        "obligation_row_count": 0,
        "stratum_count": 20,
        "zero_obligation_stratum_count": 20,
        "tool_gap_effect_eligible_score": 0,
    }


def test_scenario_report_6848_source_drift_blocks_pinned_evidence(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6848-SOURCE-DRIFT records both hashes and fails closed."""

    source = tmp_path / "source.json"
    source.write_text('{"value":1}\n', encoding="utf-8")
    manifest = {
        "fixture": {
            "path": "source.json",
            "file_sha256": exp.sha256_file(source),
            "immutable": True,
        }
    }
    assert exp.validate_pinned_hashes(tmp_path, manifest) == []

    source.write_text('{"value":2}\n', encoding="utf-8")
    failures = exp.validate_pinned_hashes(tmp_path, manifest)

    assert failures == [
        {
            "check": "source_hash.fixture",
            "expected": manifest["fixture"]["file_sha256"],
            "observed": exp.sha256_file(source),
            "passed": False,
        }
    ]


def test_scenario_report_6848_expected_source_hash_drift_blocks_build(
    current_records: dict[str, dict],
) -> None:
    """SCENARIO-REPORT-6848-SOURCE-DRIFT also controls a fresh contract build."""

    artifact = exp.build_artifact(
        REPO,
        run_date="20260901",
        duration_s=0.1,
        source_records=current_records,
        expected_hashes={"exp6836": "sha256:stale"},
    )

    assert artifact["v599_evidence_contract_ready_score"] == 0
    assert artifact["gate_check_summary"]["failed_check"] == "source_hash.exp6836"


def test_scenario_report_6848_missing_required_artifact_writes_blocked_contract(
    current_records: dict[str, dict],
) -> None:
    """SCENARIO-REPORT-6848-MISSING-ARTIFACT never invents Exp6844 evidence."""

    changed = dict(current_records)
    changed["exp6844"] = exp.missing_source_record(exp.SOURCE_SPECS["exp6844"])
    artifact = exp.build_artifact(
        REPO,
        run_date="20260901",
        duration_s=0.1,
        source_records=changed,
    )

    assert artifact["v599_evidence_contract_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == ("complete_blocked_v599_method_change_evidence_contract")
    failed = {row["check"]: row for row in artifact["gate_check_summary"]["failed_checks"]}
    assert failed["source.exp6844.readable_terminal"]["observed"] == "missing"
    assert any(
        row["source_id"] == "exp6844" and row["status"] == "missing" for row in artifact["rows"]
    )


def test_req_report_6848_invalid_or_nonterminal_source_is_blocking(tmp_path: Path) -> None:
    """REQ-REPORT-6848: unreadable and nonterminal inputs do not pass preconditions."""

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{not-json", encoding="utf-8")
    invalid_spec = {"source_id": "invalid", "path": "invalid.json"}
    invalid_record = exp.load_source_record(tmp_path, invalid_spec)
    assert invalid_record["state"] == "invalid"
    assert invalid_record["terminal"] is False

    pending = tmp_path / "pending.json"
    pending.write_text(
        json.dumps({"honest_verdict": "pending", "verdict_class": "partial"}),
        encoding="utf-8",
    )
    pending_record = exp.load_source_record(
        tmp_path, {"source_id": "pending", "path": "pending.json"}
    )
    assert pending_record["state"] == "nonterminal"
    assert pending_record["terminal"] is False

    annotated = tmp_path / "annotated.json"
    annotated.write_text(
        json.dumps(
            {
                "honest_verdict": {"value": "complete_null_annotated"},
                "verdict_class": "null",
            }
        ),
        encoding="utf-8",
    )
    assert (
        exp.load_source_record(tmp_path, {"source_id": "annotated", "path": "annotated.json"})[
            "terminal"
        ]
        is True
    )


def test_req_report_6848_absent_or_malformed_skip_log_returns_no_skip(tmp_path: Path) -> None:
    """REQ-REPORT-6848: only a structured conductor line can explain Exp6838."""

    assert exp.scan_conductor_skips(tmp_path) == []
    log = tmp_path / exp.CONDUCTOR_LOG_PATH
    log.parent.mkdir(parents=True)
    log.write_text(
        "Independent obligation compatibility shortcut GATE_BLOCK\n"
        "| unrelated | task | OK | complete |\n",
        encoding="utf-8",
    )
    assert exp.scan_conductor_skips(tmp_path) == []


def test_scenario_report_6848_stale_hardcoded_path_is_ignored(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6848-STALE-HARDCODED-PATH uses dynamic discovery."""

    discovered = tmp_path / "results" / "new-terminal.json"
    discovered.parent.mkdir()
    discovered.write_text("{}\n", encoding="utf-8")
    entry = {
        "stored_absolute_path": "/old/checkout/results/experiment_6681.json",
        "discovered_path": "results/new-terminal.json",
        "artifact_family": "supervisor_receipt",
        "generator_identity": {"generator_id": "canonical_arc_runtime"},
        "model_identity": {"model_id": "unsloth/Qwen3.8-27B-GGUF"},
        "process_ownership": {"task_owned": True, "lease_id": "lease-1"},
        "exact_outcome_authority": {"authority_kind": "exact_later_transition"},
    }

    qualified = exp.qualify_dynamic_artifact(tmp_path, entry)

    assert qualified["eligible"] is True
    assert qualified["resolved_path"] == discovered.resolve().as_posix()
    assert qualified["stored_absolute_path_ignored"] is True
    assert qualified["file_sha256"] == exp.sha256_file(discovered)


def test_scenario_report_6848_only_stale_or_unsafe_paths_are_ineligible(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6848-STALE-HARDCODED-PATH rejects unqualified paths."""

    only_stale = exp.qualify_dynamic_artifact(
        tmp_path, {"stored_absolute_path": "/old/experiment_6681.json"}
    )
    escaping = exp.qualify_dynamic_artifact(tmp_path, {"discovered_path": "../outside.json"})
    absolute = exp.qualify_dynamic_artifact(
        tmp_path, {"discovered_path": (tmp_path / "absolute.json").as_posix()}
    )
    missing = exp.qualify_dynamic_artifact(tmp_path, {"discovered_path": "results/missing.json"})
    incomplete_path = tmp_path / "results" / "incomplete.json"
    incomplete_path.parent.mkdir()
    incomplete_path.write_text("{}\n", encoding="utf-8")
    incomplete = exp.qualify_dynamic_artifact(
        tmp_path, {"discovered_path": "results/incomplete.json"}
    )

    assert only_stale == {
        "eligible": False,
        "reason": "execution_time_discovered_path_missing",
        "stored_absolute_path_ignored": True,
    }
    assert escaping["eligible"] is False
    assert escaping["reason"] == "discovered_path_outside_repo"
    assert absolute["reason"] == "discovered_path_must_be_repo_relative"
    assert missing["reason"] == "discovered_artifact_missing"
    assert incomplete["reason"] == "required_provenance_missing"
    assert incomplete["missing_fields"] == [
        "artifact_family",
        "generator_identity",
        "model_identity",
        "process_ownership",
        "exact_outcome_authority",
    ]


def test_scenario_report_6848_terminal_null_is_not_promoted(current_artifact: dict) -> None:
    """SCENARIO-REPORT-6848-TERMINAL-NULL-PRESERVATION keeps harmful nulls."""

    branches = {row["source_id"]: row for row in current_artifact["terminal_branch_manifest"]}
    memory = branches["exp6842"]

    assert memory["source_verdict_class"] == "null"
    assert memory["scientific_disposition"] == "harmful_learning_rule_retired"
    assert memory["benefit_claim_allowed"] is False
    assert current_artifact["verdict_class"] == "null"
    assert all(row["verdict_class"] in exp.CLOSED_VERDICT_CLASSES for row in branches.values())


def test_req_report_6848_reference_identifiers_match_primary_records(
    current_artifact: dict,
) -> None:
    """REQ-REPORT-6848: all eight promoted identifiers have bounded primary receipts."""

    rows = current_artifact["reference_verification_rows"]
    by_id = {row["identifier"]: row for row in rows}

    assert set(by_id) == {
        "arXiv:2604.27283",
        "arXiv:2604.15149",
        "arXiv:2607.16999",
        "arXiv:2608.11994",
        "arXiv:2606.19808",
        "arXiv:2608.31046",
        "arXiv:2608.30461",
        "arXiv:2608.29596",
    }
    assert all(row["identifier_verified"] is True for row in rows)
    assert all(row["access_boundary"] == "context_only_no_dependency" for row in rows)
    assert all(row["dependency_added"] is False for row in rows)
    assert by_id["arXiv:2608.30461"]["planner_title_matches_primary"] is False
    assert by_id["arXiv:2608.29596"]["planner_title_matches_primary"] is False
    assert by_id["arXiv:2608.29596"]["planner_date_matches_primary"] is False


def test_req_report_6848_dynamic_schema_covers_all_provenance_layers(
    current_artifact: dict,
) -> None:
    """REQ-REPORT-6848: one schema governs hashes, discovery, identity, and skips."""

    schema = current_artifact["dynamic_evidence_schema"]
    assert set(schema["properties"]) == {
        "immutable_hashes",
        "dynamic_artifact_manifest",
        "generator_identity",
        "model_identity",
        "process_ownership",
        "exact_outcome_authority",
        "conductor_skip",
    }
    assert schema["frozen_source_rule"]["hash_drift_action"] == "block"
    assert schema["execution_time_discovery_rule"]["stored_absolute_paths_authoritative"] is False
    assert schema["execution_time_discovery_rule"]["hard_coded_exp6681_allowed"] is False
    consumers = {
        consumer
        for row in current_artifact["changed_mechanism_manifest"]
        for consumer in row["consumers"]
    }
    assert {"exp6849", "exp6850", "exp6853", "exp6857"} <= consumers


def test_req_report_6848_conductor_skip_is_explicit_and_hash_bound(
    current_artifact: dict,
) -> None:
    """REQ-REPORT-6848: the absent Exp6838 has a terminal conductor record."""

    skips = current_artifact["conductor_skip_manifest"]
    assert len(skips) == 1
    assert skips[0]["task_id"] == "exp6838"
    assert skips[0]["timestamp"] == "2026-09-01 08:39 UTC"
    assert skips[0]["outcome"] == "GATE_BLOCK"
    assert skips[0]["upstream_task_id"] == "exp6837"
    assert skips[0]["log_path"] == "ops/conductor-log.md"
    assert "upstream retired" in skips[0]["reason"]
    assert skips[0]["log_line_sha256"].startswith("sha256:")


def test_req_report_6848_checksum_ignores_only_measured_duration(
    current_artifact: dict,
) -> None:
    """REQ-REPORT-6848: deterministic evidence content has a stable checksum."""

    changed = deepcopy(current_artifact)
    changed["duration_s"] = 999.0
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)

    assert changed["reproducibility_checksum"] == current_artifact["reproducibility_checksum"]
    changed["rows"][0]["status"] = "changed"
    assert exp.reproducibility_checksum(changed) != current_artifact["reproducibility_checksum"]


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (lambda artifact: artifact.pop("rows"), "missing required fields"),
        (
            lambda artifact: artifact.__setitem__("verifier_is_oracle", True),
            "verifier_is_oracle must be false",
        ),
        (
            lambda artifact: artifact.__setitem__("inference_substrate", "wrong"),
            "inference_substrate must equal deterministic CPU evidence replay",
        ),
        (
            lambda artifact: artifact.__setitem__("honest_verdict", "not_terminal"),
            "honest_verdict must start with complete_",
        ),
        (
            lambda artifact: artifact.__setitem__("verdict_class", "unknown"),
            "verdict_class must use the closed enum",
        ),
        (
            lambda artifact: artifact.__setitem__("v599_evidence_contract_ready_score", 2),
            "v599_evidence_contract_ready_score must be zero or one",
        ),
        (
            lambda artifact: artifact.__setitem__("verdict_class", "blocked"),
            "ready contract must be null and have no failed contract gates",
        ),
        (
            lambda artifact: (
                artifact.__setitem__("v599_evidence_contract_ready_score", 0),
                artifact.__setitem__("verdict_class", "null"),
            ),
            "blocked contract must record at least one failed gate",
        ),
        (
            lambda artifact: artifact["rows"][0].__setitem__("verdict_class", "unknown"),
            "every evidence row must use the closed verdict enum",
        ),
        (
            lambda artifact: artifact["terminal_branch_manifest"][0].__setitem__(
                "source_verdict_class", "unknown"
            ),
            "terminal_branch_manifest must preserve closed source verdicts",
        ),
        (
            lambda artifact: artifact.__setitem__("reference_verification_rows", []),
            "all eight promoted reference identifiers must be verified",
        ),
        (
            lambda artifact: artifact.__setitem__("changed_mechanism_manifest", []),
            "the exact V599 readiness consumers must be declared",
        ),
    ],
)
def test_req_report_6848_validator_rejects_contract_corruption(
    current_artifact: dict,
    mutation: Callable[[dict], object],
    expected_error: str,
) -> None:
    """REQ-REPORT-6848: structural corruption cannot validate as ready."""

    changed = deepcopy(current_artifact)
    mutation(changed)
    assert any(expected_error in error for error in exp.validate_artifact(changed))


def test_req_report_6848_atomic_writer_and_cli_use_only_requested_output(
    tmp_path: Path,
    current_artifact: dict,
) -> None:
    """REQ-REPORT-6848: tests and CLI writes stay inside the temporary directory."""

    direct = tmp_path / "direct.json"
    exp.write_json_atomic(direct, current_artifact)
    assert json.loads(direct.read_text(encoding="utf-8")) == current_artifact

    output = tmp_path / "cli.json"
    assert exp.main(["--repo-root", str(REPO), "--output", str(output), "--date", "20260901"]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["v599_evidence_contract_ready_score"] == 1
    assert payload["duration_s"] > 0
    assert exp.main(["--repo-root", str(REPO), "--output", str(output), "--validate"]) == 0

    corrupted = deepcopy(payload)
    corrupted["verifier_is_oracle"] = True
    exp.write_json_atomic(output, corrupted)
    assert exp.main(["--repo-root", str(REPO), "--output", str(output), "--validate"]) == 1


def test_scenario_report_6848_live_validation_detects_manifest_drift(
    tmp_path: Path,
    current_artifact: dict,
) -> None:
    """SCENARIO-REPORT-6848-SOURCE-DRIFT checks pinned files during validation."""

    errors = exp.validate_artifact(current_artifact, tmp_path)
    assert any(error.startswith("source drift:") for error in errors)


def test_req_report_6848_cli_fails_if_internal_validation_rejects_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-6848: an internal contract error prevents publication."""

    monkeypatch.setattr(exp, "validate_artifact", lambda artifact: ["forced invalid"])
    output = tmp_path / "not-written.json"
    assert exp.main(["--repo-root", str(tmp_path), "--output", str(output)]) == 1
    assert output.exists() is False


def test_scenario_report_6848_missing_repo_cli_still_writes_blocked_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6848-MISSING-ARTIFACT leaves a terminal diagnostic."""

    output = tmp_path / "blocked.json"
    assert (
        exp.main(["--repo-root", str(tmp_path), "--output", str(output), "--date", "20260901"]) == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["verdict_class"] == "blocked"
    assert payload["v599_evidence_contract_ready_score"] == 0
    assert payload["gate_check_summary"]["failed_checks"]


def test_req_report_6848_validate_mode_reports_bad_json(tmp_path: Path) -> None:
    """REQ-REPORT-6848: validation fails visibly for an unreadable artifact."""

    output = tmp_path / "bad.json"
    output.write_text("not-json", encoding="utf-8")
    assert exp.main(["--repo-root", str(REPO), "--output", str(output), "--validate"]) == 1
