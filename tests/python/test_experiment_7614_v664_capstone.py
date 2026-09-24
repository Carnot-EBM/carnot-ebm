"""Tests for REQ-REPORT-7614 and SCENARIO-REPORT-7614-*.

The tests read tracked evidence and use only private temporary paths for
candidate files. They never rewrite a producer or the terminal deliverable.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import carnot.experiment_7614_v664_capstone as capstone


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def artifact() -> dict[str, object]:
    """Build a deterministic candidate from the exact tracked V664 bytes."""

    return capstone.build_artifact_for_test()


def test_v664_authority_and_custody_have_exact_order(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-7614-CUSTODY keeps all fourteen identities literal."""

    contract = capstone.load_contract(ROOT)
    assert contract["comparison_passed"] is True
    assert contract["selected_roadmap_path"] == "research-roadmap.yaml"
    assert [task["id"] for task in contract["tasks"]] == list(capstone.EXPECTED_TASK_IDS)
    candidates = {row["path"]: row for row in contract["resolution_candidates"]}
    assert candidates["research-roadmap-next.yaml"]["exists"] is False
    assert candidates["research-roadmap.yaml"]["matches_milestone"] is True

    rows = artifact["task_dispositions"]
    assert isinstance(rows, list) and len(rows) == 14
    assert artifact["rows"] == rows
    by_id = {row["task_id"]: row for row in rows}
    assert by_id["exp7605-fit-evidence"]["evidence_state"] == "conductor_gate_blocked"
    assert by_id["exp7606-test-online-evidence"]["evidence_state"] == ("conductor_gate_blocked")
    for task_id in (
        "exp7607-evidence-energy",
        "exp7608-decision-evaluation",
        "exp7609-guarded-learning",
    ):
        assert by_id[task_id]["evidence_state"] == "missing"
    assert by_id["exp7614-capstone"]["evidence_state"] == "current_terminal"


def test_independent_sources_bound_branch_conclusions(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-7614-BRANCHES forbids producer-headline promotion."""

    branches = {row["branch"]: row for row in artifact["branch_conclusions"]}
    assert set(branches) == {
        "information_value",
        "probability",
        "action_cost",
        "causal_learning",
        "retention",
        "arc_support",
        "service_placement",
        "hardware",
    }
    for name in (
        "information_value",
        "probability",
        "action_cost",
        "causal_learning",
        "retention",
    ):
        assert branches[name]["source_task"] == "exp7610-evidence-audit"
        assert branches[name]["verdict_class"] == "blocked"
        assert branches[name]["benefit"] == "not_measured"
    arc = branches["arc_support"]
    assert arc["source_task"] == "exp7612-arc-history-measurement"
    assert arc["verdict_class"] == "blocked"
    assert arc["observed_independent_units"] == 1
    assert arc["support_sufficient"] is False
    service = branches["service_placement"]
    assert service["source_task"] == "exp7613-service-attribution"
    assert service["verdict_class"] == "null"
    assert service["cold_start_regression_preserved"] is True
    assert service["whole_service_spans_authenticated"] is True
    hardware = branches["hardware"]
    assert len(hardware["board_rows"]) == 3
    assert hardware["automatic_follow_up_authorized"] is False


def test_retirement_defers_blocked_science_and_stops_arc_collector(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-7614-RETIREMENT distinguishes blocks from nulls."""

    rows = {row["branch"]: row for row in artifact["retirement_rows"]}
    assert rows["eight_feature_evidence_construction"]["decision"] == "defer"
    assert rows["guarded_updates"]["decision"] == "defer"
    assert rows["matched_prefix_collector"]["decision"] == "stop_unchanged_mechanism"
    assert rows["matched_prefix_collector"]["scientific_hypothesis_retired"] is False
    assert rows["prior_capstone_scope"]["decision"] == "retire_narrow_repeated_scope"


def test_complete_artifact_is_blocked_not_partial(artifact: dict[str, object]) -> None:
    """REQ-REPORT-7614 separates completed accounting from scientific benefit."""

    assert artifact["experiment_id"] == "exp7614-capstone"
    assert artifact["milestone"] == "2026.09.664"
    assert artifact["run_date"] == "20260924"
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["planned_inference_substrate_class"] == "aggregation"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["honest_verdict"] == "complete_blocked_required_v664_external_evidence"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["capstone_complete_score"] == 1
    assert artifact["readiness"] is None
    assert artifact["submitted_externally"] is False
    assert artifact["flagged_adversarial"] is False
    assert artifact["publication_gates"]["unmet_gates"] == []
    assert artifact["operator_held_confirmations"] == {
        "E0": "operator_held",
        "Kaggle": "operator_held",
    }


def test_block_summary_names_every_operand(artifact: dict[str, object]) -> None:
    """REQ-REPORT-7614 records exact operands for every terminal block."""

    summary = artifact["gate_check_summary"]
    assert summary["passed"] is False
    assert summary["failed_count"] >= 5
    for row in summary["failed_checks"]:
        assert {"check", "upstream", "path", "field", "operator", "expected", "observed"} <= set(
            row
        )


@pytest.mark.parametrize(
    ("affected", "terminal", "states", "expected"),
    [
        (True, False, [], "partial"),
        (False, False, [], "disqualified"),
        (True, True, [{"evidence_state": "invalid"}], "disqualified"),
        (True, True, [{"evidence_state": "missing"}], "blocked"),
        (True, True, [{"verdict_class": "null"}], "null"),
    ],
)
def test_terminal_precedence_reserves_partial_for_owned_work(
    affected: bool, terminal: bool, states: list[dict[str, object]], expected: str
) -> None:
    """REQ-REPORT-7614 reserves partial for unfinished current validation."""

    reduced = capstone.classify_terminal(
        states, affected_complete=affected, terminal_complete=terminal
    )
    assert reduced["verdict_class"] == expected


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("identity", "identity_invalid"),
        ("terminal", "terminal_identity_invalid"),
        ("tasks", "task_dispositions_invalid"),
        ("branches", "branch_conclusions_invalid"),
        ("retirement", "retirement_rows_invalid"),
        ("model", "model_contract_invalid"),
        ("substrate", "substrate_invalid"),
        ("claim", "claim_contract_invalid"),
        ("source", "source_hash_mismatch"),
        ("acceptance", "acceptance_gates_invalid"),
        ("budget", "sample_size_budget_invalid"),
        ("publication", "publication_gates_invalid"),
        ("principles", "field_principles_invalid"),
        ("checksum", "reproducibility_checksum_invalid"),
    ],
)
def test_cold_validation_rejects_mutations(
    artifact: dict[str, object], mutation: str, error: str
) -> None:
    """SCENARIO-REPORT-7614-TERMINAL rejects altered terminal reductions."""

    changed = deepcopy(artifact)
    if mutation == "identity":
        changed["milestone"] = "2026.09.000"
    elif mutation == "terminal":
        changed["honest_verdict"] = "blocked_without_complete_prefix"
    elif mutation == "tasks":
        changed["task_dispositions"].pop()
    elif mutation == "branches":
        changed["branch_conclusions"][0]["benefit"] = True
    elif mutation == "retirement":
        changed["retirement_rows"][0]["decision"] = "retire"
    elif mutation == "model":
        changed["MODEL_SPECS"] = [{"hf_id": "invented/current"}]
    elif mutation == "substrate":
        changed["inference_substrate_class"] = "model_full_generation"
    elif mutation == "claim":
        changed["positive_claim"] = True
    elif mutation == "source":
        changed["source_artifact_hashes"][0]["sha256"] = "sha256:" + "0" * 64
    elif mutation == "acceptance":
        changed["acceptance_gate_results"] = []
    elif mutation == "budget":
        changed["sample_size_budget"]["planned"] = 13
    elif mutation == "publication":
        changed["publication_gates"]["paper_ready"] = False
    elif mutation == "principles":
        changed["field_principles"].pop("rows")
    else:
        changed["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert error in capstone.validate_artifact(changed)


def test_fresh_reduction_and_scoped_plan(artifact: dict[str, object], tmp_path: Path) -> None:
    """SCENARIO-REPORT-7614-TERMINAL keeps scratch output below private paths."""

    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")
    loaded = json.loads(candidate.read_text(encoding="utf-8"))
    assert capstone.validate_artifact(loaded) == []
    assert capstone.independent_reduce(loaded) == []
    commands = capstone.build_validation_plan(ROOT, tmp_path / "private")
    assert capstone.validate_validation_plan(ROOT, commands) == []
    focused = next(row for row in commands if row.name == "focused_pytest")
    assert capstone.TEST_PATH.as_posix() in focused.argv
    assert "tests/python" not in focused.argv
    assert {"-n", "0", "--no-cov"} <= set(focused.argv)


def test_arguments_sources_and_principles_fail_closed(
    artifact: dict[str, object], tmp_path: Path
) -> None:
    """REQ-REPORT-7614 rejects bad roots, dates, sources, and field coverage."""

    assert capstone.date_argument("20260924") == "20260924"
    with pytest.raises(ValueError, match="20260924"):
        capstone.date_argument("20260923")
    assert capstone.root_argument(ROOT) == ROOT
    with pytest.raises(ValueError, match="repository root"):
        capstone.root_argument(tmp_path)
    assert capstone.validate_artifact([]) == ["artifact_mapping_required"]
    assert capstone.validate_artifact({})[0].startswith("missing_required_field:")
    assert set(artifact["field_principles"]) == set(artifact) - {
        "field_principles",
        "reproducibility_checksum",
    }
    assert all(row["principle"] for row in artifact["acceptance_gate_results"])
    assert artifact["verifier_is_oracle"] is True
    assert artifact["sample_size_budget"]["seeds_and_windows_multiply_units"] is False


def test_terminal_receipt_requirement(artifact: dict[str, object]) -> None:
    """SCENARIO-REPORT-7614-TERMINAL requires every exact terminal reader."""

    changed = deepcopy(artifact)
    changed["validation_receipts"] = [
        row for row in changed["validation_receipts"] if row["name"] != "fresh_process_cold_replay"
    ]
    assert "terminal_validation_incomplete" in capstone.validate_artifact(
        changed, require_terminal=True
    )


def test_defensive_source_and_reduction_boundaries(
    artifact: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7614 fails closed on malformed sources and custody drift."""

    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        capstone.load_json(scalar)

    monkeypatch.setattr(
        capstone.contract_reader,
        "resolve_v664_roadmap",
        lambda _root: (tmp_path / "roadmap.yaml", {"tasks": None}, []),
    )
    monkeypatch.setattr(
        capstone.contract_reader,
        "compare_contract_authorities",
        lambda _text, _roadmap: {},
    )
    with pytest.raises(ValueError, match="task list"):
        capstone.load_contract(ROOT)
    monkeypatch.undo()

    assert (
        capstone._producer_failure("task", "path", {"flagged_adversarial": True})["field"]
        == "flagged_adversarial"
    )
    assert capstone._producer_failure("task", "path", {})["field"] == "verdict_class"
    assert (
        capstone._producer_failure(
            "task", "path", {"verdict_class": "null", "experiment_id": "wrong"}
        )["field"]
        == "identity_and_milestone"
    )

    invalid = tmp_path / "bad.json"
    invalid.write_text("{", encoding="utf-8")
    source = capstone.load_producer(tmp_path, {"id": "exp9999-private", "deliverable": "bad.json"})
    assert source["evidence_state"] == "invalid"
    assert (
        capstone._disposition(source, {"id": "exp9999-private"}, 1)["disposition"] == "disqualified"
    )

    invalid.write_text(
        json.dumps(
            {
                "experiment_id": "exp9999-private",
                "milestone": "wrong",
                "verdict_class": "null",
                "flagged_adversarial": False,
            }
        ),
        encoding="utf-8",
    )
    assert (
        capstone.load_producer(tmp_path, {"id": "exp9999-private", "deliverable": "bad.json"})[
            "evidence_state"
        ]
        == "invalid"
    )

    assert capstone._source_hashes_match({}, ROOT) is False
    assert capstone._source_hashes_match({"source_artifact_hashes": ["bad"]}, ROOT) is False
    failures = capstone.failure_rows(
        {"comparison_passed": False, "selected_roadmap_path": "roadmap"},
        {
            "bad": {
                "evidence_state": "invalid",
                "gate_check_summary": {
                    "check": "bad",
                    "upstream": "old",
                    "path": "bad.json",
                    "field": "x",
                    "op": "eq",
                    "expected": 1,
                    "observed": 0,
                },
            }
        },
        {"required_checks_passed": True, "terminal_validation_passed": True},
    )
    assert failures[0]["check"] == "contract_authorities_agree"
    assert failures[1]["operator"] == "eq"
    assert capstone._manifest_payload()["frozen_before_checks"] is True

    monkeypatch.setattr(
        capstone, "load_contract", lambda _root: (_ for _ in ()).throw(ValueError())
    )
    assert "independent_reduction_failed" in capstone.validate_artifact(artifact)
