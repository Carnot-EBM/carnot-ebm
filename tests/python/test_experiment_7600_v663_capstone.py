"""Tests for REQ-REPORT-7600 and SCENARIO-REPORT-7600-*.

The tests reduce tracked evidence without writing the terminal result. Any
writer boundary uses a private pytest directory.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import carnot.experiment_7600_v663_capstone as capstone


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def artifact() -> dict[str, object]:
    """Build one deterministic terminal candidate from tracked V663 bytes."""

    return capstone.build_artifact_for_test()


def test_v663_authority_has_exact_fourteen_task_contract() -> None:
    """REQ-REPORT-7600 accepts the active authority after staging is consumed."""

    contract = capstone.load_contract(ROOT)
    assert contract["comparison_passed"] is True
    assert contract["selected_roadmap_path"] == "research-roadmap.yaml"
    assert [task["id"] for task in contract["tasks"]] == list(capstone.EXPECTED_TASK_IDS)
    candidates = {row["path"]: row for row in contract["resolution_candidates"]}
    assert candidates["research-roadmap-next.yaml"]["exists"] is False
    assert candidates["research-roadmap.yaml"]["matches_milestone"] is True


def test_inventory_preserves_producer_pregate_and_missing_states(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-7600-INVENTORY keeps custody states distinct."""

    rows = artifact["task_dispositions"]
    assert isinstance(rows, list) and len(rows) == 14
    assert artifact["rows"] == rows
    by_id = {row["task_id"]: row for row in rows}
    assert by_id["exp7588-evidence-protocol"]["evidence_state"] == "terminal_blocked"
    assert by_id["exp7590-evidence-pilot"]["evidence_state"] == "conductor_gate_blocked"
    assert by_id["exp7590-evidence-pilot"]["evidence_path"] == (
        "results/experiment_7590_evidence_pilot.json"
    )
    for task_id in (
        "exp7591-fit-evidence",
        "exp7592-test-online-evidence",
        "exp7593-evidence-energy",
        "exp7594-decision-evaluation",
        "exp7595-guarded-learning",
    ):
        assert by_id[task_id]["evidence_state"] == "missing"
        assert by_id[task_id]["artifact_sha256"] is None
    assert by_id["exp7600-capstone"]["evidence_state"] == "current_terminal"
    assert by_id["exp7600-capstone"]["artifact_path"] is None


def test_branch_reduction_uses_only_qualified_terminal_sources(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-7600-BRANCHES prevents cross-branch promotion."""

    branches = {row["branch"]: row for row in artifact["branch_conclusions"]}
    assert set(branches) == {
        "evidence_gain_and_probability",
        "decision_quality",
        "delayed_learning",
        "retention",
        "arc_observability",
        "rust_consumer_readiness",
        "rust_consumer_speed",
        "board_continuity",
    }
    for name in (
        "evidence_gain_and_probability",
        "decision_quality",
        "delayed_learning",
        "retention",
    ):
        assert branches[name]["verdict_class"] == "blocked"
        assert branches[name]["benefit"] == "not_measured"
        assert branches[name]["source_task"] == "exp7596-evidence-audit"
    arc = branches["arc_observability"]
    assert arc["verdict_class"] == "null"
    assert arc["observed_independent_units"] == 6
    assert arc["history_support_ready_score"] == 0
    assert arc["new_solve_claimed"] is False
    assert arc["near_exact_gate_safety_claimed"] is False
    ready = branches["rust_consumer_readiness"]
    speed = branches["rust_consumer_speed"]
    assert ready["readiness"] is True and ready["benefit"] == "not_applicable"
    assert speed["verdict_class"] == "null"
    assert speed["benefit"] is False
    assert speed["comparison_summary"]["warm:8"]["paired_ratio"]["lower95"] > 1.0
    assert speed["comparison_summary"]["cold:1"]["paired_ratio"]["lower95"] < 1.0
    boards = branches["board_continuity"]
    assert len(boards["board_rows"]) == 3
    assert boards["placement_scope"] == "placement_unmeasured"


def test_missing_science_defers_construction_retirement(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-7600-RETIREMENT does not turn absence into a null."""

    decisions = {row["branch"]: row for row in artifact["retirement_rows"]}
    evidence = decisions["evidence_link_feature_construction"]
    guarded = decisions["guarded_update_construction"]
    assert evidence["decision"] == guarded["decision"] == "defer"
    assert evidence["scientific_hypothesis_retired"] is False
    assert guarded["scientific_hypothesis_retired"] is False
    assert evidence["resource_or_transport_blocker"] is True
    assert "unexposed inventory" in evidence["fresh_corpus_reopening_condition"]
    prior = decisions["prior_capstone_verdict"]
    assert prior["retire_if_same_verdict"] is True
    assert prior["literal_exact_match"] is False
    assert prior["normalized_scope_match"] is True
    assert prior["scientific_hypothesis_retired"] is False


@pytest.mark.parametrize(
    ("affected", "terminal", "states", "expected"),
    [
        (True, False, [], "partial"),
        (False, False, [], "disqualified"),
        (True, True, [{"evidence_state": "invalid"}], "disqualified"),
        (True, True, [{"evidence_state": "missing"}], "blocked"),
        (True, True, [{"evidence_state": "terminal", "verdict_class": "null"}], "null"),
    ],
)
def test_terminal_precedence_reserves_partial_for_owned_work(
    affected: bool, terminal: bool, states: list[dict[str, object]], expected: str
) -> None:
    """REQ-REPORT-7600 distinguishes external blocks from unfinished work."""

    reduced = capstone.classify_terminal(
        states, affected_complete=affected, terminal_complete=terminal
    )
    assert reduced["verdict_class"] == expected


def test_complete_artifact_has_required_reporting_contract(
    artifact: dict[str, object],
) -> None:
    """REQ-REPORT-7600 separates completed accounting from benefit."""

    assert artifact["experiment_id"] == "exp7600-capstone"
    assert artifact["milestone"] == "2026.09.663"
    assert artifact["run_date"] == "20260924"
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["planned_inference_substrate_class"] == "aggregation"
    assert artifact["capstone_complete_score"] == 1
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "complete_blocked_required_v663_external_evidence"
    assert artifact["positive_claim"] is False
    assert artifact["submitted_externally"] is False
    publication = artifact["publication_gates"]
    assert publication["paper_ready"] is all(
        publication["gates"][name]["pass"] is True for name in ("G1", "G2", "G3", "G4")
    )
    assert publication["unmet_gates"] == [
        name for name in ("G1", "G2", "G3", "G4") if publication["gates"][name]["pass"] is not True
    ]


def test_gate_summary_names_every_external_failure(artifact: dict[str, object]) -> None:
    """REQ-REPORT-7600 makes each blocked operand independently auditable."""

    summary = artifact["gate_check_summary"]
    assert summary["passed"] is False
    failures = summary["failed_checks"]
    upstreams = {row["upstream"] for row in failures}
    assert "exp7588-evidence-protocol" in upstreams
    assert "exp7590-evidence-pilot" in upstreams
    assert "exp7594-decision-evaluation" in upstreams
    assert "exp7595-guarded-learning" in upstreams
    for row in failures:
        assert {"check", "upstream", "path", "field", "op", "expected", "observed"} <= set(row)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("identity", "identity_invalid"),
        ("terminal", "terminal_identity_invalid"),
        ("task", "task_dispositions_invalid"),
        ("branch", "branch_conclusions_invalid"),
        ("retirement", "retirement_rows_invalid"),
        ("model", "model_contract_invalid"),
        ("substrate", "substrate_invalid"),
        ("claim", "current_claim_contract_invalid"),
        ("source", "source_hash_mismatch"),
        ("prior", "prior_failure_rows_invalid"),
        ("terminal_reduction", "terminal_reduction_invalid"),
        ("acceptance", "acceptance_gates_invalid"),
        ("budget", "sample_size_budget_invalid"),
        ("score", "capstone_score_invalid"),
        ("publication", "publication_gates_invalid"),
        ("principles", "field_principles_invalid"),
        ("checksum", "reproducibility_checksum_invalid"),
    ],
)
def test_cold_validation_rejects_terminal_mutations(
    artifact: dict[str, object], mutation: str, error: str
) -> None:
    """SCENARIO-REPORT-7600-VALIDATION rejects altered terminal reductions."""

    changed = deepcopy(artifact)
    if mutation == "identity":
        changed["milestone"] = "2026.09.000"
    elif mutation == "terminal":
        changed["honest_verdict"] = "blocked_without_complete_prefix"
    elif mutation == "task":
        changed["task_dispositions"].pop()
    elif mutation == "branch":
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
    elif mutation == "prior":
        changed["prior_failure_rows"] = []
    elif mutation == "terminal_reduction":
        changed["status"] = "complete_null_wrong"
    elif mutation == "acceptance":
        changed["acceptance_gate_results"] = []
    elif mutation == "budget":
        changed["sample_size_budget"]["planned"] = 13
    elif mutation == "score":
        changed["capstone_complete_score"] = 0
    elif mutation == "publication":
        changed["publication_gates"]["paper_ready"] = False
    elif mutation == "principles":
        changed["field_principles"].pop("rows")
    else:
        changed["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert error in capstone.validate_artifact(changed)


def test_fresh_reduction_and_frozen_scoped_plan(
    artifact: dict[str, object], tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7600-VALIDATION keeps all temporary paths private."""

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


def test_source_and_argument_boundaries_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-7600 rejects malformed evidence, roots, and dates."""

    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        capstone.load_json(scalar)
    assert capstone.date_argument("20260924") == "20260924"
    with pytest.raises(ValueError, match="20260924"):
        capstone.date_argument("20260923")
    assert capstone.root_argument(ROOT) == ROOT
    with pytest.raises(ValueError, match="repository root"):
        capstone.root_argument(tmp_path)
    assert capstone.validate_artifact([]) == ["artifact_mapping_required"]
    assert capstone.validate_artifact({})[0].startswith("missing_required_field:")

    task = {"id": "exp9999-private", "deliverable": "scalar.json"}
    source = capstone.load_producer(tmp_path, task)
    assert source["evidence_state"] == "invalid"
    assert capstone._disposition(source, task, 1)["disposition"] == "disqualified"

    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        json.dumps(
            {
                "experiment": 9999,
                "milestone": "wrong",
                "verdict_class": "null",
                "flagged_adversarial": False,
            }
        ),
        encoding="utf-8",
    )
    invalid_task = {"id": "exp9999-private", "deliverable": "invalid.json"}
    assert capstone.load_producer(tmp_path, invalid_task)["evidence_state"] == "invalid"
    assert (
        capstone._producer_failure("task", "path", {"flagged_adversarial": True})["field"]
        == "flagged_adversarial"
    )
    assert capstone._producer_failure("task", "path", {})["field"] == "verdict_class"


def test_defensive_hash_contract_and_failure_boundaries(
    artifact: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7600 fails closed on broken custody and terminal receipts."""

    assert capstone._source_hashes_match({}, ROOT) is False
    assert capstone._source_hashes_match({"source_artifact_hashes": ["bad"]}, ROOT) is False
    absent = {"source_artifact_hashes": [{"path": "absent", "sha256": "sha256:0"}]}
    assert capstone._source_hashes_match(absent, ROOT) is False

    failures = capstone.failure_rows(
        {"comparison_passed": False, "selected_roadmap_path": "roadmap"},
        {
            "blocked": {
                "evidence_state": "terminal_blocked",
                "verdict_class": "blocked",
                "evidence_path": "blocked.json",
                "gate_check_summary": {},
            }
        },
        {"required_checks_passed": False, "terminal_validation_passed": False},
    )
    assert [row["check"] for row in failures] == [
        "contract_authorities_agree",
        "current_required_checks_passed",
        "current_terminal_validation_passed",
        "producer_terminal_not_blocked",
    ]

    incomplete = deepcopy(artifact)
    incomplete["validation_receipts"] = [
        row
        for row in incomplete["validation_receipts"]
        if row["name"] != "fresh_process_cold_replay"
    ]
    assert "terminal_validation_incomplete" in capstone.validate_artifact(
        incomplete, require_terminal=True
    )

    monkeypatch.setattr(
        capstone, "load_contract", lambda _root: (_ for _ in ()).throw(ValueError())
    )
    assert "independent_reduction_failed" in capstone.validate_artifact(artifact)


def test_contract_reader_rejects_non_list_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-7600 rejects an authority without a task list."""

    monkeypatch.setattr(
        capstone,
        "resolve_v663_roadmap",
        lambda root: (root / "research-roadmap.yaml", {"tasks": None}, []),
    )
    monkeypatch.setattr(capstone, "compare_contract_authorities", lambda _text, _roadmap: {})
    with pytest.raises(ValueError, match="task list"):
        capstone.load_contract(ROOT)


def test_fields_gates_counts_and_claim_limits_are_explicit(
    artifact: dict[str, object],
) -> None:
    """REQ-REPORT-7600 binds principles, counts, censoring, and claim limits."""

    assert set(artifact["field_principles"]) == set(artifact) - {
        "field_principles",
        "reproducibility_checksum",
    }
    assert all(row["principle"] for row in artifact["acceptance_gate_results"])
    assert artifact["verifier_is_oracle"] is True
    budget = artifact["sample_size_budget"]
    assert budget["planned"] == 14
    assert budget["unstarted"] == 6
    assert budget["seeds_and_windows_multiply_source_groups"] is False
    signs = artifact["comparative_row_reduction"]["sign_counts"]
    assert signs == {"positive": 0, "negative": 0, "zero": 0, "missing": 14}
