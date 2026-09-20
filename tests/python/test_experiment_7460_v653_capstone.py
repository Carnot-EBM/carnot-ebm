"""Tests for the V653 fourteen-disposition capstone.

Spec refs: REQ-REPORT-7460 and SCENARIO-REPORT-7460-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7460_v653_capstone as capstone


@pytest.fixture(scope="module")
def contract() -> dict[str, object]:
    """Load the two current V653 authorities once."""

    return capstone.load_contract(capstone.REPO_ROOT)


@pytest.fixture(scope="module")
def evidence(contract: dict[str, object]) -> dict[str, dict[str, object]]:
    """Authenticate every predecessor once while keeping branches separate."""

    tasks = contract["tasks"]
    assert isinstance(tasks, list)
    return capstone.collect_evidence(capstone.REPO_ROOT, tasks)


def _passing_validation() -> dict[str, object]:
    names = (*capstone.REQUIRED_CHECK_NAMES, *capstone.TERMINAL_CHECK_NAMES)
    return {
        "required_checks_passed": True,
        "terminal_validation_passed": True,
        "validation_receipts": [
            {
                "name": name,
                "required": True,
                "passed": True,
                "exit_code": 0,
                "duration_s": 0.0,
                "log_path": f"/tmp/{name}.log",
                "log_sha256": "sha256:" + "0" * 64,
            }
            for name in names
        ],
        "repository_health": {"status": "outside_current_affected_validity"},
    }


@pytest.fixture(scope="module")
def artifact(
    contract: dict[str, object], evidence: dict[str, dict[str, object]]
) -> dict[str, object]:
    """Build deterministic terminal bytes for mutation tests."""

    return capstone.build_artifact(
        capstone.REPO_ROOT,
        contract,
        evidence,
        _passing_validation(),
        capstone.evaluate_publication_gates(),
        started_at_utc="2026-09-20T20:00:00+00:00",
        completed_at_utc="2026-09-20T20:00:01+00:00",
        duration_s=1.0,
        phase_spans=capstone.zero_test_phase_spans(),
    )


def test_scenario_7460_contract_has_exact_order_and_independent_authorities(
    contract: dict[str, object],
) -> None:
    # REQ-REPORT-7460; SCENARIO-REPORT-7460-CONTRACT.
    assert contract["comparison_passed"] is True
    assert contract["errors"] == []
    assert [row["id"] for row in contract["tasks"]] == list(capstone.EXPECTED_TASK_IDS)
    assert len(contract["contract_rows"]) == 14
    assert sum(len(row["gates"]) for row in contract["contract_rows"]) == 21
    assert all(row["passed"] for row in contract["contract_rows"])


def test_scenario_7460_contract_mutations_fail_closed() -> None:
    # REQ-REPORT-7460; SCENARIO-REPORT-7460-CONTRACT.
    markdown = (capstone.REPO_ROOT / capstone.DESIGN_PATH).read_text(encoding="utf-8")
    roadmap = capstone.load_yaml_mapping(capstone.REPO_ROOT / capstone.ROADMAP_PATH)

    wrong_markdown = markdown.replace("| 14 | exp7460-capstone |", "| 14 | exp7999-capstone |", 1)
    assert (
        "markdown_task_order"
        in capstone.compare_contract_authorities(wrong_markdown, roadmap)["errors"]
    )

    wrong_yaml = deepcopy(roadmap)
    wrong_yaml["milestone"] = "2026.09.652"
    wrong_yaml["tasks"] = wrong_yaml["tasks"][:-1]
    wrong_yaml["tasks"][0], wrong_yaml["tasks"][1] = (
        wrong_yaml["tasks"][1],
        wrong_yaml["tasks"][0],
    )
    errors = capstone.compare_contract_authorities(markdown, wrong_yaml)["errors"]
    assert {"yaml_milestone", "yaml_task_order", "task_count"} <= set(errors)

    undeclared = deepcopy(roadmap)
    undeclared["tasks"][4]["gated_on"][0]["artifact_field"] = "undeclared_score"
    errors = capstone.compare_contract_authorities(markdown, undeclared)["errors"]
    assert "producer_field_declaration" in errors

    wrong_markdown_milestone = markdown.replace(
        "**Milestone:** `2026.09.653`", "**Milestone:** `2026.09.652`", 1
    )
    errors = capstone.compare_contract_authorities(wrong_markdown_milestone, roadmap)["errors"]
    assert "markdown_milestone" in errors


def test_scenario_7460_predecessors_include_exact_pre_gate_and_disqualification(
    evidence: dict[str, dict[str, object]],
) -> None:
    # REQ-REPORT-7460; SCENARIO-REPORT-7460-BRANCHES.
    assert list(evidence) == list(capstone.EXPECTED_TASK_IDS[:-1])
    assert all(row["authenticated"] for row in evidence.values())
    assert evidence["exp7453-energy-calibration"]["source_kind"] == "structured_pre_gate"
    assert evidence["exp7453-energy-calibration"]["verdict_class"] == "blocked"
    assert evidence["exp7453-energy-calibration"]["gate_check_summary"] == {
        "upstream": "exp7452-source-embeddings",
        "path": "results/experiment_7453_energy_calibration.json",
        "check": "structured_pre_gate",
        "field": "embedding_capture_ready_score",
        "operator": "==",
        "expected": 1,
        "observed": 0,
        "passed": False,
    }
    assert evidence["exp7455-decision-audit"]["verdict_class"] == "disqualified"
    assert evidence["exp7455-decision-audit"]["required_validation_passed"] is False
    assert evidence["exp7455-decision-audit"]["flagged_adversarial"] is False


def test_scenario_7460_absent_input_is_terminal_blocked_without_hiding_peer(
    tmp_path: Path,
) -> None:
    # REQ-REPORT-7460; SCENARIO-REPORT-7460-BRANCHES.
    missing_task = {
        "id": "exp7453-energy-calibration",
        "deliverable": "results/experiment_7453_v653_energy_calibration.json",
    }
    missing = capstone.load_evidence_slot(tmp_path, missing_task)
    assert missing["verdict_class"] == "blocked"
    assert missing["honest_verdict"] == "blocked_missing_declared_evidence"
    assert missing["gate_check_summary"]["observed"] is False
    assert missing["available"] is False

    peer_task = {"id": "exp7454-continuous-learning", "deliverable": "peer.json"}
    peer = tmp_path / "peer.json"
    peer.write_text(
        json.dumps(
            {
                "schema": "carnot.test.v1",
                "experiment_id": "exp7454-v653-continuous-learning",
                "milestone": capstone.MILESTONE,
                "status": "complete_null",
                "honest_verdict": "complete_null_peer",
                "verdict_class": "null",
                "flagged_adversarial": False,
                "rows": [],
                "validation_receipts": [],
            }
        ),
        encoding="utf-8",
    )
    peer_row = capstone.load_evidence_slot(tmp_path, peer_task)
    assert peer_row["source_kind"] == "terminal_artifact"
    assert peer_row["task_id"] == "exp7454-continuous-learning"


def test_scenario_7460_recomputes_six_branch_metrics_from_rows(
    evidence: dict[str, dict[str, object]],
) -> None:
    # REQ-REPORT-7460; SCENARIO-REPORT-7460-BRANCHES/CLAIMS.
    metrics = capstone.reduce_branch_metrics(evidence)
    assert list(metrics) == list(capstone.CLAIM_BRANCHES[1:])

    assert metrics["static_representation"]["planned_cells"] == 748
    assert metrics["static_representation"]["unstarted_cells"] == 748
    assert metrics["static_representation"]["value"] is None

    online = metrics["delayed_learning"]
    assert online["prediction_rows"] == 5271
    assert online["group_count_per_arm"] == 753
    assert online["learned_minus_equal_log_loss"] == pytest.approx(0.0019923661017172)
    assert online["primary_benefit_passed"] is False
    assert online["independent_replay_valid"] is True

    extraction = metrics["extraction"]
    assert extraction["development_attempted"] == 8
    assert extraction["evaluation_unstarted"] == 96
    assert extraction["paired_evaluation_units"] == 0

    arc = metrics["arc_exposure"]
    assert arc["episode_count"] == 8
    assert arc["threshold_reached_count"] == 8
    assert arc["applied_redirect_count"] == 4
    assert arc["new_level_credit"] == 0

    durability = metrics["durability"]
    assert durability["paired_block_count"] == 30
    assert durability["total_service_ratio"] == pytest.approx(1.0006514391986738)
    assert durability["speed_gate_passed"] is False

    board = metrics["board_status"]
    assert board["board_count"] == 3
    assert board["blocked_boards"] == ["GateMate"]
    assert board["hardware_ready_score"] == 0

    changed = dict(evidence)
    extraction_source = dict(evidence["exp7456-extraction-audit"])
    extraction_payload = dict(extraction_source["payload"])
    extraction_rows = list(extraction_payload["rows"])
    opened = dict(next(row for row in extraction_rows if row["capture_phase"] == "evaluation"))
    opened.update({"attempted": True, "disposition": "complete"})
    extraction_rows.append(opened)
    extraction_payload["rows"] = extraction_rows
    extraction_source["payload"] = extraction_payload
    changed["exp7456-extraction-audit"] = extraction_source
    assert capstone.reduce_branch_metrics(changed)["extraction"]["paired_evaluation_units"] == 0


def test_scenario_7460_invalid_audit_cannot_supply_readiness(
    contract: dict[str, object], evidence: dict[str, dict[str, object]]
) -> None:
    # REQ-REPORT-7460; SCENARIO-REPORT-7460-CLAIMS.
    gates = capstone.audit_structured_gates(contract, evidence)
    failed = [row for row in gates if not row["passed"]]
    assert any(row["consumer"] == "exp7453-energy-calibration" for row in failed)

    claims = capstone.reduce_claim_matrix(evidence)
    assert claims["contract"]["completion_score"] == 1
    assert claims["static_representation"]["readiness_score"] == 0
    assert claims["delayed_learning"]["benefit_score"] == 0
    assert claims["delayed_learning"]["audit_artifact_admissible"] is False
    assert claims["extraction"]["completion_score"] == 1
    assert claims["arc_exposure"]["benefit_score"] == 0
    assert claims["durability"]["benefit_score"] == 0
    assert claims["board_status"]["readiness_score"] == 0


def test_scenario_7460_retirement_uses_exact_and_semantic_rules(
    contract: dict[str, object], evidence: dict[str, dict[str, object]]
) -> None:
    # REQ-REPORT-7460; SCENARIO-REPORT-7460-RETIREMENT.
    terminal = capstone.classify_terminal(contract, evidence, _passing_validation())
    assert terminal["honest_verdict"] == (
        "complete_disqualified_required_v653_science_with_fourteen_dispositions"
    )
    rows = capstone.retirement_rows(contract["tasks"], evidence, terminal)
    mixture = next(row for row in rows if row["mechanism"] == "four_expert_mixture")
    assert mixture["decision"] == "retire"
    assert mixture["same_measured_null"] is True
    assert mixture["permanent_retirement"] is True
    capstone_prior = next(row for row in rows if row["task_id"] == capstone.EXPERIMENT_ID)
    assert capstone_prior["same_exact_verdict"] is False
    assert capstone_prior["decision"] == "continue"
    assert not any(
        row["permanent_retirement"] for row in rows if row.get("environmental_failure") is True
    )

    continuations = capstone.continuation_rows(capstone.reduce_claim_matrix(evidence), rows)
    by_mechanism = {row["mechanism"]: row for row in continuations}
    assert by_mechanism["four_expert_mixture"]["decision"] == "retire"
    assert by_mechanism["source_conditioned_energy"]["decision"] == "defer"
    assert by_mechanism["compact_extraction"]["decision"] == "continue"


def test_scenario_7460_terminal_precedence_covers_blocked_and_clean_null(
    contract: dict[str, object], evidence: dict[str, dict[str, object]]
) -> None:
    # REQ-REPORT-7460; SCENARIO-REPORT-7460-BRANCHES/CLAIMS.
    unauthenticated = deepcopy(evidence)
    unauthenticated["exp7447-contract-methods"]["authenticated"] = False
    assert (
        capstone.classify_terminal(contract, unauthenticated, _passing_validation())[
            "verdict_class"
        ]
        == "disqualified"
    )

    broken_contract = deepcopy(contract)
    broken_contract["comparison_passed"] = False
    assert (
        capstone.classify_terminal(broken_contract, evidence, _passing_validation())[
            "verdict_class"
        ]
        == "disqualified"
    )

    valid = deepcopy(evidence)
    for row in valid.values():
        row["authenticated"] = True
        row["available"] = True
        row["valid"] = True
        row["verdict_class"] = "null"
        row["flagged_adversarial"] = False
    assert (
        capstone.classify_terminal(contract, valid, _passing_validation())["verdict_class"]
        == "null"
    )
    valid["exp7453-energy-calibration"]["available"] = False
    assert (
        capstone.classify_terminal(contract, valid, _passing_validation())["verdict_class"]
        == "blocked"
    )


def test_scenario_7460_artifact_has_fourteen_plain_dispositions(
    artifact: dict[str, object],
) -> None:
    # REQ-REPORT-7460; SCENARIO-REPORT-7460-ARTIFACT/CLAIMS.
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == capstone.ZERO_INVOCATION_COUNTS
    assert set(artifact["invocation_counts"]) == {
        "model_loads",
        "forward_calls",
        "generation_calls",
    }
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["flagged_adversarial"] is True
    assert artifact["promotion_score"] == 0
    assert artifact["capstone_complete_score"] == 1
    assert len(artifact["task_dispositions"]) == 14
    assert artifact["task_dispositions"][-1]["source_kind"] == "current_work"
    assert artifact["rows"] == artifact["task_dispositions"]
    assert artifact["publication_gates"]["headline_scope"] == "FoVer dual-condition AUROC"
    assert capstone.validate_artifact(artifact) == []


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("schema", "bad", "identity_invalid"),
        ("status", "running", "lifecycle_invalid"),
        ("MODEL_SPECS", ["unexpected"], "model_contract_invalid"),
        ("execution_venue", "device", "substrate_invalid"),
        ("promotion_score", 1, "promotion_invalid"),
        ("task_dispositions", [], "task_dispositions_invalid"),
        ("claim_matrix", {}, "claim_matrix_invalid"),
        ("branch_metrics", {}, "branch_metrics_invalid"),
        ("retirement_rows", [], "retirement_rows_invalid"),
        ("continuation_rows", [], "continuation_rows_invalid"),
        ("unresolved_obligations", [], "unresolved_obligations_invalid"),
        ("acceptance_gate_results", [], "acceptance_gates_invalid"),
        ("field_principles", {}, "field_principles_invalid"),
        ("reproducibility_checksum", "sha256:bad", "reproducibility_checksum_invalid"),
    ],
)
def test_scenario_7460_cold_validator_rejects_derived_mutations(
    artifact: dict[str, object], field: str, value: object, error: str
) -> None:
    # REQ-REPORT-7460; SCENARIO-REPORT-7460-ARTIFACT.
    changed = deepcopy(artifact)
    changed[field] = value
    assert error in capstone.validate_artifact(changed)


def test_scenario_7460_validator_rejects_missing_and_incomplete_terminal(
    artifact: dict[str, object],
) -> None:
    # REQ-REPORT-7460; SCENARIO-REPORT-7460-ARTIFACT.
    missing = deepcopy(artifact)
    del missing["schema"]
    assert capstone.validate_artifact(missing) == ["missing_required_field:schema"]
    assert capstone.validate_artifact([]) == ["artifact_mapping_required"]

    incomplete = deepcopy(artifact)
    incomplete["validation_receipts"] = []
    incomplete["capstone_complete_score"] = 0
    incomplete["task_dispositions"][-1]["authenticated"] = False
    incomplete["task_dispositions"][-1]["valid"] = False
    incomplete["rows"] = deepcopy(incomplete["task_dispositions"])
    errors = capstone.validate_artifact(incomplete)
    assert "terminal_validation_incomplete" in errors
    assert "capstone_score_invalid" not in errors
    assert capstone._receipt_passed(None, capstone.REQUIRED_CHECK_NAMES) is False
    assert capstone.independent_reduce(incomplete)


def test_scenario_7460_hashes_sidecars_and_reductions_fail_closed(
    artifact: dict[str, object], evidence: dict[str, dict[str, object]]
) -> None:
    # REQ-REPORT-7460; SCENARIO-REPORT-7460-ARTIFACT.
    changed = deepcopy(artifact)
    changed["source_artifact_hashes"] = []
    assert "source_hash_mismatch" in capstone.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["source_artifact_hashes"]["contract-comparison"]["sha256"] = "sha256:bad"
    assert "source_hash_mismatch" in capstone.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["source_artifact_hashes"]["current-module"]["path"] = 7
    assert "source_hash_mismatch" in capstone.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["source_artifact_hashes"]["current-module"]["sha256"] = "sha256:bad"
    assert "source_hash_mismatch" in capstone.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["publication_gates"] = {}
    assert "publication_gates_invalid" in capstone.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["structured_gate_audit"] = []
    assert "structured_gate_audit_invalid" in capstone.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["capstone_complete_score"] = 0
    assert "capstone_score_invalid" in capstone.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["flagged_adversarial"] = False
    assert "flagged_adversarial_invalid" in capstone.validate_artifact(changed)

    training = dict(evidence)
    source = dict(evidence["exp7452-source-embeddings"])
    payload = dict(source["payload"])
    payload["small_ebm_training"] = {"performed": True}
    source["payload"] = payload
    training["exp7452-source-embeddings"] = source
    assert any(
        row["kind"] == "historical_small_ebm_training"
        for row in capstone._historical_sidecars(training)
    )


def test_scenario_7460_validation_plan_is_scoped_and_date_is_frozen(
    tmp_path: Path,
) -> None:
    # REQ-REPORT-7460; SCENARIO-REPORT-7460-ARTIFACT.
    tmp_path.joinpath("basetemp").mkdir()
    tmp_path.joinpath("coverage").mkdir()
    commands = capstone.build_validation_plan(capstone.REPO_ROOT, tmp_path)
    assert [row.name for row in commands] == list(capstone.REQUIRED_CHECK_NAMES)
    assert capstone.validate_validation_plan(capstone.REPO_ROOT, commands) == []
    focused = next(row for row in commands if row.name == "focused_pytest")
    assert ("-n", "0") == focused.argv[1:3]
    assert "--no-cov" in focused.argv
    assert capstone.TEST_PATH.as_posix() in focused.argv
    assert capstone.date_argument(capstone.RUN_DATE) == capstone.RUN_DATE
    with pytest.raises(Exception, match="run date must be"):
        capstone.date_argument("20260919")


def test_scenario_7460_cli_supports_cold_modes_and_public_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # REQ-REPORT-7460; SCENARIO-REPORT-7460-ARTIFACT.
    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]", encoding="utf-8")
    assert capstone.main(["--validate", str(malformed)]) == 1

    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(capstone, "validate_artifact", lambda *_args, **_kwargs: [])
    assert capstone.main(["--validate", str(candidate)]) == 0
    monkeypatch.setattr(capstone, "independent_reduce", lambda *_args, **_kwargs: ["bad"])
    assert capstone.main(["--independent-reduce", str(candidate)]) == 1
    monkeypatch.setattr(
        capstone,
        "run_experiment",
        lambda *_args, **_kwargs: {
            "status": "complete",
            "verdict_class": "null",
            "capstone_complete_score": 1,
        },
    )
    assert capstone.main(["--date", capstone.RUN_DATE]) == 0
