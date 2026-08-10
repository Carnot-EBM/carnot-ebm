"""Tests for Exp6283 V541 adversarial capstone.

Spec refs: REQ-INFRA-6283, SCENARIO-INFRA-6283-1,
SCENARIO-INFRA-6283-2, SCENARIO-INFRA-6283-3,
SCENARIO-INFRA-6283-4, SCENARIO-INFRA-6283-5,
SCENARIO-INFRA-6283-6.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_6283_v541_adversarial_capstone as mod
from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/research-harnesses/spec.md"


def _write_artifact(path: Path, status: str, verdict: str | None = None, **extra: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "honest_verdict": verdict if verdict is not None else f"{status}: fixture",
        "duration_s": 1.0,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "verifier_is_oracle": False,
        "reproducibility_checksum": "sha256:fixture",
        **extra,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fake_reviews(task_ids: list[str], critical: str | None = None) -> dict[str, dict[str, object]]:
    reviews: dict[str, dict[str, object]] = {}
    for task_id in task_ids:
        flags = (
            [{"kind": "FIXTURE_CRITICAL", "severity": "critical", "detail": "fixture"}]
            if task_id == critical
            else []
        )
        reviews[task_id] = {
            "path": task_id,
            "present": True,
            "stamped_flagged_adversarial": False,
            "stamped_corrigendum_pending": False,
            "current_rule_flag_count": len(flags),
            "current_rule_critical_flag_count": len(flags),
            "current_rule_warn_flag_count": 0,
            "current_rule_flags": flags,
        }
    return reviews


def test_spec_declares_req_6283_fields_and_scenarios() -> None:
    """REQ-INFRA-6283: OpenSpec records the V541 capstone contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6283") :]

    for token in (
        "REQ-INFRA-6283",
        "SCENARIO-INFRA-6283-1",
        "SCENARIO-INFRA-6283-2",
        "SCENARIO-INFRA-6283-3",
        "SCENARIO-INFRA-6283-4",
        "SCENARIO-INFRA-6283-5",
        "SCENARIO-INFRA-6283-6",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert token in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_exact_path_precedence_ignores_receipts_and_aliases(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6283-1: exact artifact state outranks receipts."""

    declared = tmp_path / "results/experiment_6275_declared.json"
    alias = tmp_path / "results/experiment_6275_complete_alias.json"
    _write_artifact(declared, "in_progress", "in_progress")
    _write_artifact(alias, "complete", "complete: alias")
    tasks = [
        {
            "id": "exp6275-fixture",
            "title": "Exact fixture",
            "track": "live_verification",
            "deliverable": "results/experiment_6275_declared.json",
        },
        {
            "id": "exp6278-missing-fixture",
            "title": "Missing fixture",
            "track": "continuous_learning",
            "deliverable": "results/experiment_6278_missing.json",
        },
    ]
    receipts = {
        "exp6275-fixture": {"status": "OK", "receipt_found": True},
        "exp6278-missing-fixture": {"status": "OK", "receipt_found": True},
    }

    matrix = mod.build_exact_declared_deliverable_matrix(
        tmp_path, tasks, conductor_receipts=receipts
    )

    row = matrix["exp6275-fixture"]
    assert row["terminal_class"] == "running"
    assert row["terminal"] is False
    assert row["receipt_override_attempted"] is True
    assert row["receipt_overrode"] is False
    assert row["same_number_alias_used"] is False
    assert row["same_number_alias_candidates_ignored"] == [
        "results/experiment_6275_complete_alias.json"
    ]

    missing = matrix["exp6278-missing-fixture"]
    assert missing["terminal_class"] == "missing"
    assert missing["receipt_override_attempted"] is True
    assert missing["terminal"] is False


def test_gate_cascade_reads_only_terminal_exact_bare_fields(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6283-2: exact bare fields are the only gate inputs."""

    _write_artifact(tmp_path / "results/ready.json", "complete", "complete: ready", score=1)
    _write_artifact(
        tmp_path / "results/wrapped.json",
        "complete",
        "complete: wrapped",
        score={"value": 1, "principle": "wrapped fixture"},
    )
    _write_artifact(tmp_path / "results/nonterminal.json", "in_progress", "in_progress", score=1)
    tasks = [
        {"id": "ready", "deliverable": "results/ready.json"},
        {"id": "wrapped", "deliverable": "results/wrapped.json"},
        {"id": "nonterminal", "deliverable": "results/nonterminal.json"},
        {
            "id": "downstream",
            "deliverable": "results/downstream.json",
            "gated_on": [
                {"upstream": "ready", "artifact_field": "score", "op": "==", "value": 1},
                {"upstream": "wrapped", "artifact_field": "score", "op": "==", "value": 1},
                {
                    "upstream": "nonterminal",
                    "artifact_field": "score",
                    "op": "==",
                    "value": 1,
                },
                {"upstream": "ready", "artifact_field": "missing", "op": "exists", "value": True},
                {"upstream": "missing", "artifact_field": "score", "op": "==", "value": 1},
            ],
        },
    ]

    receipt = mod.evaluate_gate_cascade_receipts(tmp_path, tasks)

    assert receipt["passed_count"] == 1
    assert receipt["failed_count"] == 4
    assert receipt["gates"][0]["passed"] is True
    assert receipt["gates"][1]["eligibility"]["field_is_bare"] is False
    assert receipt["gates"][1]["reason"] == "ineligible_upstream_field"
    assert receipt["gates"][2]["eligibility"]["classification"]["terminal"] is False
    assert receipt["gates"][3]["eligibility"]["field_present"] is False
    assert receipt["gates"][4]["reason"] == "missing_upstream_task"


def test_current_flags_stay_separate_from_stamped_flags() -> None:
    """SCENARIO-INFRA-6283-3: stamped and current-rule flags stay distinct."""

    artifact_path = REPO / "results/experiment_6282_arc_mechanic_class_live_router.json"
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    current = {
        "path": artifact_path.as_posix(),
        "flag_count": 1,
        "flags": [{"kind": "CURRENT_FIXTURE", "severity": "warn", "detail": "fixture"}],
    }

    row = mod.adversarial_result_row(artifact_path, payload, current)

    assert row["stamped_flagged_adversarial"] is True
    assert row["stamped_corrigendum_pending"] is True
    assert row["current_rule_warn_flag_count"] == 1
    assert row["current_rule_critical_flag_count"] == 0
    assert row["current_rule_flags"][0]["kind"] == "CURRENT_FIXTURE"


def test_current_report_preserves_independent_branch_ledgers() -> None:
    """SCENARIO-INFRA-6283-4: branches cannot launder each other."""

    tasks = mod.roadmap_tasks(mod.load_roadmap(REPO))
    task_ids = [str(task["id"]) for task in tasks]
    report = mod.build_report(
        REPO,
        date="20260810",
        command_receipts=[{"command": "focused", "exit_code": 0}],
        artifact_reviews=_fake_reviews(task_ids),
        publication_result={
            "paper_ready": True,
            "gates": {f"G{i}": {"pass": True} for i in range(1, 5)},
            "unmet_gates": [],
        },
        before_hashes=mod.protected_hashes(REPO),
        started_at=0.0,
    )

    assert mod.validate_report(report) == []
    assert report["asp_semantic_compiler_state"]["promotion_allowed"] is True
    assert report["flagship_asp_benchmark_state"]["promotion_allowed"] is False
    assert report["certified_admission_state"]["certified_admission_ready_score"] == 0.0
    assert report["chronological_continuous_learning_state"]["terminal_class"] == "skipped"
    assert report["heldout_transfer_state"]["terminal_class"] == "missing"
    assert report["shadow_consumer_state"]["terminal_class"] == "skipped"
    assert report["variable_cardinality_backend_state"]["promotion_allowed"] is True
    assert report["mode_jump_safety_and_value_state"]["mode_jump_workload_value_ready_score"] == 0.0
    assert report["arc_mechanic_router_state"]["promotion_allowed"] is False

    ledger = report["branch_independent_promotion_ledger"]
    assert ledger["asp_verification"]["promotion_allowed"] is False
    assert "flagship_asp_benchmark_closed" in ledger["asp_verification"]["blocking_reasons"]
    assert ledger["continuous_self_learning"]["promotion_allowed"] is False
    assert "heldout_transfer_missing" in ledger["continuous_self_learning"]["blocking_reasons"]
    assert ledger["sampler"]["promotion_allowed"] is False
    assert "mode_jump_value_closed" in ledger["sampler"]["blocking_reasons"]
    assert ledger["arc"]["promotion_allowed"] is False
    assert "arc_mechanic_router_closed" in ledger["arc"]["blocking_reasons"]


def test_retirement_publication_and_arc_receipts_are_recorded() -> None:
    """SCENARIO-INFRA-6283-6: gate and retirement evidence is recorded."""

    tasks = mod.roadmap_tasks(mod.load_roadmap(REPO))
    task_ids = [str(task["id"]) for task in tasks]
    report = mod.build_report(
        REPO,
        date="20260810",
        command_receipts=[{"command": "focused", "exit_code": 0}],
        artifact_reviews=_fake_reviews(task_ids),
        publication_result={
            "paper_ready": False,
            "gates": {
                "G1": {"pass": True},
                "G2": {"pass": False},
                "G3": {"pass": True},
                "G4": {"pass": True},
            },
            "unmet_gates": ["G2"],
        },
        before_hashes=mod.protected_hashes(REPO),
        started_at=0.0,
    )

    actions = report["prior_failure_retirement_actions"]
    fired = {row["task_id"] for row in actions["actions"] if row["rule_fired"]}
    assert {
        "exp6277-chronological-certified-csl-ab",
        "exp6279-certified-memory-shadow-consumer",
    } <= fired
    assert actions["manifest_update_count"] == 0
    assert report["publication_gate_g1_g2_g3_g4_and_unmet_gates"]["unmet_gates"] == ["G2"]
    assert report["arc_provenance_and_registry_receipts"]["registry_update_count"] == 0
    assert report["arc_provenance_and_registry_receipts"]["game_level_solve_claimed"] is False
    assert report["arc_provenance_and_registry_receipts"]["hidden_game_source_access_count"] == 0
    assert (
        report["spec_traceability_status_changelog_reconciliation"][
            "ops_status_changelog_traceability_touched_by_this_task"
        ]
        is False
    )


def test_report_validation_requires_bare_zero_claims_and_checksum() -> None:
    """SCENARIO-INFRA-6283-5: forbidden claim counters are bare zero."""

    report = {field: f"fixture-{field}" for field in mod.REQUIRED_ARTIFACT_FIELDS}
    report["status"] = "complete"
    report["inference_substrate"] = mod.INFERENCE_SUBSTRATE
    report["verifier_is_oracle"] = False
    for field in mod.FORBIDDEN_ZERO_FIELDS:
        report[field] = 0
    report["field_principles"] = dict(mod.FIELD_PRINCIPLES)
    report["field_provenance"] = {
        field: {"sources": ["REQ-INFRA-6283"], "principle": mod.FIELD_PRINCIPLES[field]}
        for field in mod.REQUIRED_ARTIFACT_FIELDS
    }
    report["terminal_nonterminal_blocked_skipped_null_flagged_retired_and_ready_counts"] = {
        "count_principles": dict(mod.COUNT_PRINCIPLES)
    }
    report["gate_cascade_receipts"] = {"gates": [], "passed_count": 0, "failed_count": 0}
    report["publication_gate_g1_g2_g3_g4_and_unmet_gates"] = {
        "gates": {"G1": {}, "G2": {}, "G3": {}, "G4": {}},
        "unmet_gates": [],
    }
    report["honest_verdict"] = "complete: fixture"
    report["duration_s"] = 1.0
    report["reproducibility_checksum"] = ""
    report["reproducibility_checksum"] = mod.payload_checksum(report)

    assert mod.validate_report(report) == []

    broken = dict(report)
    broken["source_mutation_count"] = 0.0
    assert "source_mutation_count must be bare integer 0" in mod.validate_report(broken)

    broken = dict(report)
    broken["speed_power_or_energy_claim_count"] = 1
    assert "speed_power_or_energy_claim_count must be bare integer 0" in mod.validate_report(broken)


def test_write_report_uses_artifact_root_override(tmp_path: Path) -> None:
    """REQ-INFRA-6283: artifact writes are atomic and test-isolated."""

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    tasks = mod.roadmap_tasks(mod.load_roadmap(REPO))
    report = mod.build_report(
        REPO,
        date="20260810",
        command_receipts=[],
        artifact_reviews=_fake_reviews([str(task["id"]) for task in tasks]),
        publication_result={
            "paper_ready": True,
            "gates": {f"G{i}": {"pass": True} for i in range(1, 5)},
            "unmet_gates": [],
        },
        before_hashes=mod.protected_hashes(REPO),
        started_at=0.0,
    )

    path = mod.write_report(report, REPO, env={ARTIFACT_ROOT_ENV: str(artifact_root)})

    assert path == artifact_root / mod.RESULT_RELATIVE_PATH.name
    assert json.loads(path.read_text(encoding="utf-8")) == report


def test_helper_edges_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6283-5: malformed helper inputs stay explicit."""

    assert mod.read_yaml_mapping(tmp_path / "missing.yaml") == {}
    assert mod.roadmap_tasks({"tasks": "bad"}) == []
    assert mod.evaluate_operator("present", "exists", True) is True
    assert mod.evaluate_operator(None, "==", 1) is False
    assert mod.evaluate_operator(1, "!=", 2) is True
    assert mod.evaluate_operator("a", "in", ["a"]) is True
    assert mod.evaluate_operator(2, ">", 1) is True
    assert mod.evaluate_operator(2, ">=", 2) is True
    assert mod.evaluate_operator(1, "<", 2) is True
    assert mod.evaluate_operator(1, "<=", 1) is True
    assert mod.evaluate_operator("a", ">", 1) is False
    assert mod.evaluate_operator(1, "bad", 1) is False
    assert mod._score({"wrapped": {"value": 3, "principle": "fixture"}}, "wrapped") == 3
    assert mod._critical_count({"x": "bad"}, "x") == 0
    assert mod._blocking_reasons(
        [
            {
                "task_id": "expX",
                "terminal_class": "nonterminal",
                "terminal": False,
                "stamped_or_current_flagged": False,
                "promotion_allowed": False,
            }
        ],
        [("expX_missing", "expX_closed")],
    ) == ["expX_closed", "expX_nonterminal"]
    assert mod._status_from_commands([{"command": "fixture", "exit_code": 2}]) == (
        "blocked",
        "blocked: one or more recorded validation commands failed",
    )

    bad_gate = mod.evaluate_gate_cascade_receipts(
        tmp_path,
        [{"id": "downstream", "deliverable": "results/x.json", "gated_on": ["bad"]}],
    )
    assert bad_gate["gates"][0]["reason"] == "gate_not_mapping"

    actions = mod.prior_failure_retirement_actions(
        [
            {"id": "ignored", "prior_failures": ["bad"]},
            {
                "id": "same",
                "prior_failures": [
                    {
                        "experiment_id": "exp1",
                        "verdict": "blocked: same",
                        "retire_if_same_verdict": True,
                    }
                ],
            },
        ],
        {"same": {"terminal_class": "blocked", "honest_verdict_raw": "blocked: same"}},
    )
    assert actions["rule_fired_count"] == 1
    assert actions["actions"][0]["action"] == "retire_if_same_verdict_rule_fired_recorded_only"

    invalid = {"status": "complete"}
    errors = mod.validate_report(invalid)
    assert "missing required field: milestone_roadmap_path_and_hash" in errors
    assert "field_principles is not a mapping" in errors
    assert "field_provenance is not a mapping" in errors
    assert "counts field is not a mapping" in errors
    assert "gate_cascade_receipts.gates is not a list" in errors
    assert "source_mutation_count must be bare integer 0" in errors
    assert "wrong inference_substrate" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "publication gate missing G1" in errors
    assert "honest_verdict lacks terminal prefix" in errors
    assert "reproducibility_checksum missing" in errors

    broken = {field: None for field in mod.REQUIRED_ARTIFACT_FIELDS}
    broken.update(
        {
            "status": "complete",
            "field_principles": dict(mod.FIELD_PRINCIPLES),
            "field_provenance": {},
            "terminal_nonterminal_blocked_skipped_null_flagged_retired_and_ready_counts": {
                "count_principles": {}
            },
            "gate_cascade_receipts": {"gates": [{}]},
            "publication_gate_g1_g2_g3_g4_and_unmet_gates": {"gates": {}, "unmet_gates": []},
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
            "verifier_is_oracle": False,
            "honest_verdict": "complete: fixture",
            "reproducibility_checksum": "sha256:bad",
        }
    )
    for field in mod.FORBIDDEN_ZERO_FIELDS:
        broken[field] = 0
    broken_errors = mod.validate_report(broken)
    assert "missing field_provenance entry: status" in broken_errors
    assert "missing count principle: terminal" in broken_errors
    assert "gate missing principle" in broken_errors
    assert "reproducibility_checksum mismatch" in broken_errors

    try:
        mod.write_report({"status": "complete"}, tmp_path)
    except ValueError as exc:
        assert "invalid Exp6283 report" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid report unexpectedly wrote")
