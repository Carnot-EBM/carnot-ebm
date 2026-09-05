"""Tests for REQ-REPORT-7008 and the V613 evidence capstone."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7008_v613_capstone as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLICATION = {
    "paper_ready": True,
    "gates": {
        "G1": {"pass": True, "detail": "headline measured"},
        "G2": {"pass": True, "detail": "independently reproduced"},
        "G3": {"pass": True, "detail": "prose clean"},
        "G4": {"pass": True, "detail": "numbers trace"},
    },
    "unmet_gates": [],
}


def _payload(
    verdict_class: str = "positive",
    honest_verdict: str = "complete_positive_fixture",
    **updates: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "complete",
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
        "rows": [{"terminal": True}],
    }
    payload.update(updates)
    return payload


@pytest.fixture(scope="module")
def actual_artifact() -> dict[str, object]:
    """Build once because the three-family source artifact is large."""

    return mod.build_artifact(
        REPO_ROOT,
        "20260905",
        run_commands=False,
        publication_payload=PUBLICATION,
    )


def test_req_report_7008_spec_precedes_implementation() -> None:
    """REQ-REPORT-7008 names the complete schema and required scenarios."""

    text = (REPO_ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("### REQ-REPORT-7008", 1)[1]
    assert {
        "SCENARIO-REPORT-7008-CONTRACT",
        "SCENARIO-REPORT-7008-ARTIFACTS",
        "SCENARIO-REPORT-7008-BLOCKED",
        "SCENARIO-REPORT-7008-ROWS",
        "SCENARIO-REPORT-7008-CIRCULARITY",
        "SCENARIO-REPORT-7008-BOUNDARIES",
        "SCENARIO-REPORT-7008-PUBLICATION",
        "SCENARIO-REPORT-7008-RETIREMENT",
        "SCENARIO-REPORT-7008-HANDOFF",
        "SCENARIO-REPORT-7008-ARTIFACT",
    } <= set(mod.spec_anchors(section))
    assert mod.INFERENCE_SUBSTRATE in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section or field in section


def test_scenario_report_7008_contract_has_exact_parity_and_producers() -> None:
    """SCENARIO-REPORT-7008-CONTRACT checks every contract dimension."""

    document = mod.parse_document_contract(
        (REPO_ROOT / mod.DESIGN_PATH).read_text(encoding="utf-8")
    )
    roadmap = mod.load_yaml(REPO_ROOT / mod.ROADMAP_PATH)
    yaml_rows = mod.parse_yaml_contract(roadmap)
    tasks = mod.build_task_contract_rows(document, yaml_rows)
    gates = mod.build_gate_contract_rows(document, yaml_rows)
    producers = mod.build_gate_producer_rows(yaml_rows)
    priors = mod.build_prior_failure_rows(yaml_rows)

    assert [row["task_id"] for row in document] == mod.EXPECTED_TASK_IDS
    assert [row["task_id"] for row in yaml_rows] == mod.EXPECTED_TASK_IDS
    assert len(tasks) == 13
    assert len(gates) == 13
    assert len(producers) == 11
    assert len(priors) == 22
    assert all(row["passed"] for row in tasks + gates + producers + priors)
    assert mod.contract_conforms(document, yaml_rows, tasks, gates, producers, priors)

    changed = deepcopy(roadmap)
    changed["tasks"][0]["title"] = "Wrong title"
    changed["tasks"][1]["deliverable"] = "results/wrong.json"
    changed["tasks"][2]["gated_on"][0]["artifact_field"] = "wrong_field"
    changed["tasks"][3]["prior_failures"][0]["addressed_by"] = ""
    changed["tasks"].append(changed["tasks"].pop(4))
    bad_yaml = mod.parse_yaml_contract(changed)
    bad_tasks = mod.build_task_contract_rows(document, bad_yaml)
    bad_gates = mod.build_gate_contract_rows(document, bad_yaml)
    bad_producers = mod.build_gate_producer_rows(bad_yaml)
    bad_priors = mod.build_prior_failure_rows(bad_yaml)
    assert not mod.contract_conforms(
        document, bad_yaml, bad_tasks, bad_gates, bad_producers, bad_priors
    )
    assert {row["check"] for row in bad_tasks if not row["passed"]} >= {
        "title",
        "deliverable",
        "order",
    }
    assert any(not row["passed"] for row in bad_gates)
    assert any(not row["passed"] for row in bad_producers)
    assert any(not row["passed"] for row in bad_priors)


@pytest.mark.parametrize(
    ("payload", "verdict_class", "outcome", "eligible"),
    [
        (None, "blocked", "missing", False),
        (_payload(), "positive", "positive", True),
        (
            _payload(flagged_adversarial=True, adversarial_flags=[{"kind": "TAUTOLOGY"}]),
            "positive",
            "flagged",
            False,
        ),
        (
            _payload("circular_positive", "complete_circular_positive_fixture"),
            "circular_positive",
            "circular_positive",
            False,
        ),
        (_payload("null", "complete_null_fixture"), "null", "null", False),
        (_payload("blocked", "blocked_fixture"), "blocked", "blocked", False),
        (
            _payload("disqualified", "complete_disqualified_fixture"),
            "disqualified",
            "disqualified",
            False,
        ),
        (
            _payload("disqualified", "complete_disqualified_fixture", status="rejected"),
            "disqualified",
            "rejected",
            False,
        ),
        (_payload("partial", "partial_fixture"), "partial", "partial", False),
        (
            {"status": "complete", "honest_verdict": "complete_unclassified"},
            "partial",
            "unclassified",
            False,
        ),
    ],
)
def test_scenario_report_7008_verdict_classes_and_outcomes_propagate(
    payload: dict[str, object] | None,
    verdict_class: str,
    outcome: str,
    eligible: bool,
) -> None:
    """SCENARIO-REPORT-7008-ARTIFACTS trusts classes and rejects flags."""

    row = mod.classify_task(mod.EXPECTED_TASKS[0], payload)
    assert row["verdict_class"] == verdict_class
    assert row["outcome"] == outcome
    assert row["eligible_for_positive"] is eligible
    assert row["terminal"] is True
    if payload is None:
        assert row["verdict_class"] != "partial"


def test_scenario_report_7008_blocked_diagnostics_keep_exact_values() -> None:
    """SCENARIO-REPORT-7008-BLOCKED preserves source gate diagnostics."""

    payload = {
        "status": "blocked",
        "honest_verdict": "blocked_gate_check_failed",
        "gate_check_summary": "gate-unsat(final)",
        "failed_upstream": "exp6999-blinded-feature-cold-audit",
        "failed_field": "blinded_feature_bank_ready_score",
        "failed_expected": 1,
        "failed_observed": 0,
    }
    task = mod.EXPECTED_TASKS[4]
    row = mod.classify_task(task, payload)
    assert mod.blocked_diagnostic(task, payload, row) == {
        "number": 7000,
        "task_id": "exp7000-certified-blinded-pwa-kan",
        "failed_check": "exp6999-blinded-feature-cold-audit.blinded_feature_bank_ready_score",
        "expected_value": 1,
        "observed_value": 0,
        "gate_check_summary": "gate-unsat(final)",
        "terminal": True,
    }
    absent = mod.blocked_diagnostic(mod.EXPECTED_TASKS[5], None, {"verdict_class": "blocked"})
    assert absent["failed_check"] == "expected_deliverable_readable"
    assert absent["expected_value"] is True
    assert absent["observed_value"] is False


def test_scenario_report_7008_recomputes_isolation_and_commitment_rows(
    actual_artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-7008-ROWS recomputes the earliest failed gate."""

    isolation = actual_artifact["evidence_isolation_rows"]
    summary = next(row for row in isolation if row["evidence"] == "cold_audit_summary")
    assert summary["candidate_count"] == 138
    assert summary["direct_leakage_count"] == 0
    assert summary["invariance_condition_count"] == 5
    assert summary["invariance_passed"] is True
    assert summary["failed_prohibited_probes"] == [
        "mutation_metadata_only",
        "commitment_controls_only",
    ]
    assert summary["recomputed_ready_score"] == 0

    controls = actual_artifact["commitment_control_rows"]
    control = next(row for row in controls if row["evidence"] == "commitment_summary")
    assert control["row_count"] == 216
    assert control["pair_count"] == 12
    assert control["model_count"] == 3
    assert control["condition_count"] == 3
    assert control["recomputed_complete_score"] == 1


def test_scenario_report_7008_recomputes_arc_without_solve_credit(
    actual_artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-7008-BOUNDARIES keeps ARC quality separate."""

    envelopes = actual_artifact["arc_live_envelope_rows"]
    hashes = next(row for row in envelopes if row["evidence"] == "hash_replay")
    assert hashes["envelope_count"] == 4
    assert hashes["transition_count"] == 4
    assert hashes["all_hashes_passed"] is True
    quality = next(
        row
        for row in actual_artifact["arc_quality_rows"]
        if row["predictor"] == "engine" and row["stratum"] == "heldout"
    )
    assert quality["transition_count"] == 10
    assert quality["exact_next_frame_accuracy"] == pytest.approx(0.7)
    assert quality["calibrated_frame_error"] == pytest.approx(0.003857421875)
    assert actual_artifact["solve_claimed"] is False
    assert actual_artifact["level_claimed"] is False
    assert actual_artifact["registry_updated"] is False


def test_scenario_report_7008_recomputes_available_science_and_placement_rows() -> None:
    """SCENARIO-REPORT-7008-ROWS derives each optional branch from unit rows."""

    payloads = {
        7000: {
            "pwa_energy_rows": [
                {"exact_valid": True, "energy": 0.1, "terminal": True},
                {"exact_valid": False, "energy": 0.9, "terminal": True},
            ],
            "milp_certificate_rows": [{"query": "bounds", "passed": True, "terminal": True}],
        },
        7001: {"pwa_certificate_confirmed_score": 1},
        7002: {
            "selection_evidence_rows": [
                {"arm": "pwa_kan", "selected_correct": True, "terminal": True},
                {"arm": "pwa_kan", "selected_correct": False, "terminal": True},
                {"arm": "logistic", "selected_correct": False, "terminal": True},
                {"arm": "logistic", "selected_correct": False, "terminal": True},
            ]
        },
        7003: {
            "continuous_learning_rows": [
                {
                    "arm": "per_knot_anchor",
                    "future_correct_before": 0,
                    "future_correct_after": 1,
                    "prior_success_lost": False,
                    "best_at_k_before": 2,
                    "best_at_k_after": 2,
                    "terminal": True,
                }
            ],
            "per_knot_learning_positive_score": 1,
            "independent_outcome_authority": False,
        },
        7004: {"self_learning_audit_confirmed_score": 1},
        7006: {
            "z1t_typed_graph_rows": [
                {"node": "a", "placement": "z1_native", "terminal": True},
                {"node": "b", "placement": "unsupported", "terminal": True},
            ],
            "z1t_placement_rows": [{"edge": "a-b", "crossing": True, "terminal": True}],
            "hardware_executed": False,
        },
        7007: {"z1t_placement_confirmed_score": 1, "hardware_executed": False},
    }
    row_by_number = {
        number: {"number": number, "verdict_class": "positive", "terminal": True}
        for number in payloads
    }
    pwa, milp = mod.build_pwa_rows(payloads, row_by_number)
    assert pwa[0]["valid_mean_energy"] == pytest.approx(0.1)
    assert pwa[0]["invalid_mean_energy"] == pytest.approx(0.9)
    assert milp[0]["recomputed_passed"] is True
    selection = mod.build_selection_rows(payloads, row_by_number)
    summary = next(row for row in selection if row["evidence"] == "selection_summary")
    assert summary["pwa_kan_rate"] == pytest.approx(0.5)
    assert summary["best_control_rate"] == pytest.approx(0.0)
    assert summary["pwa_delta"] == pytest.approx(0.5)
    learning, support = mod.build_learning_rows(payloads, row_by_number)
    assert learning[0]["future_gain"] == 1
    assert support[0]["prior_successes_lost"] == 0
    assert support[0]["best_at_k_support_delta"] == 0
    assert learning[0]["verdict_class"] == "circular_positive"
    graph, placement = mod.build_z1t_rows(payloads, row_by_number)
    assert graph[0]["node_count"] == 2
    assert graph[0]["z1_native_count"] == 1
    assert graph[0]["unsupported_count"] == 1
    assert placement[0]["crossing_count"] == 1
    assert placement[0]["hardware_executed"] is False


@pytest.mark.parametrize(
    ("selection_positive", "certificate_confirmed", "expected"),
    [(1, 1, 1), (1, 0, 0), (0, 1, 0), (None, 1, 0)],
)
def test_scenario_report_7008_science_requires_confirmed_selection(
    selection_positive: int | None,
    certificate_confirmed: int,
    expected: int,
) -> None:
    """SCENARIO-REPORT-7008-CIRCULARITY excludes per-knot claims."""

    selection = {"oracle_distinct_selection_positive_score": selection_positive}
    certificate = {"pwa_certificate_confirmed_score": certificate_confirmed}
    assert mod.science_positive_score(selection, certificate) == expected
    learning = mod.classify_learning_positive(
        {"per_knot_learning_positive_score": 1, "independent_outcome_authority": False}
    )
    assert learning["verdict_class"] == "circular_positive"
    assert learning["science_positive"] is False
    independent = mod.classify_learning_positive(
        {"per_knot_learning_positive_score": 1, "independent_outcome_authority": True}
    )
    assert independent["verdict_class"] == "positive"
    assert independent["science_positive"] is False


def test_scenario_report_7008_publication_fields_are_copied() -> None:
    """SCENARIO-REPORT-7008-PUBLICATION copies stable G1-G4 state."""

    state = mod.publication_state(PUBLICATION)
    assert state["g1"] is True
    assert state["g2"] is True
    assert state["g3"] is True
    assert state["g4"] is True
    assert state["paper_ready"] is True
    assert state["unmet_gates"] == []
    assert [row["gate"] for row in state["publication_gate_rows"]] == [
        "G1",
        "G2",
        "G3",
        "G4",
    ]


def test_scenario_report_7008_retirement_requires_exact_repeat() -> None:
    """SCENARIO-REPORT-7008-RETIREMENT compares exact verdict strings."""

    yaml_rows = [
        {
            "number": 7000,
            "task_id": "exp7000-fixture",
            "prior_failures": [
                {
                    "experiment_id": "exp1",
                    "verdict": "blocked_gate_check_failed",
                    "addressed_by": "new method",
                    "retire_if_same_verdict": True,
                },
                {
                    "experiment_id": "exp2",
                    "verdict": "blocked_other",
                    "addressed_by": "new cause",
                    "retire_if_same_verdict": True,
                },
            ],
        }
    ]
    rows = mod.build_retirement_rows(
        yaml_rows, {7000: {"honest_verdict": "blocked_gate_check_failed"}}
    )
    assert rows[0]["exact_same_verdict"] is True
    assert rows[0]["retired"] is True
    assert rows[0]["recommendation"] == "no_rerun_without_new_cause_and_method"
    assert rows[1]["exact_same_verdict"] is False
    assert rows[1]["retired"] is False


def test_scenario_report_7008_actual_artifact_is_terminal_and_evidence_led(
    actual_artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-7008-HANDOFF starts at the first causal boundary."""

    assert actual_artifact["v613_capstone_complete_score"] == 1
    assert actual_artifact["v613_task_contract_conforms_score"] == 1
    assert actual_artifact["v613_science_positive_score"] == 0
    assert actual_artifact["verdict_class"] == "null"
    assert actual_artifact["honest_verdict"].startswith("complete_null_")
    assert len(actual_artifact["rows"]) == 13
    assert [row["number"] for row in actual_artifact["rows"]] == list(range(6996, 7009))
    assert [row["number"] for row in actual_artifact["missing_task_rows"]] == [
        7001,
        7002,
        7003,
        7004,
        7006,
        7007,
    ]
    assert all(row["verdict_class"] == "blocked" for row in actual_artifact["missing_task_rows"])
    assert len(actual_artifact["per_branch_results"]) == 5
    assert all(row["terminal"] for row in actual_artifact["per_branch_results"])
    assert actual_artifact["hardware_executed"] is False
    handoff = actual_artifact["v614_handoff_rows"][0]
    assert handoff["earliest_causal_boundary"] == "exp6999_prohibited_feature_shortcut_gate"
    assert "mutation_metadata_only" in handoff["recommendation"]
    forbidden = " ".join(handoff["forbidden_revivals"])
    assert "spilled_energy" in forbidden
    assert "public_arc_resolve" in forbidden
    assert set(actual_artifact["field_principles"]) == mod.REQUIRED_ARTIFACT_FIELDS


def test_scenario_report_7008_contract_block_is_schema_complete(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7008-BLOCKED returns the complete blocked shape."""

    blocked = mod.build_artifact(
        tmp_path,
        "20260905",
        run_commands=False,
        publication_payload=PUBLICATION,
    )
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"] == mod.BLOCKED_VERDICT
    assert blocked["gate_check_summary"] == {
        "failed_check": "contract_preconditions",
        "expected_value": "readable activated V613 contracts",
        "observed_value": "missing or unreadable",
        "passed": False,
    }
    assert blocked["v613_capstone_complete_score"] == 0
    assert set(blocked["field_principles"]) == mod.REQUIRED_ARTIFACT_FIELDS


def test_scenario_report_7008_validator_rejects_forgery(
    actual_artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-7008-ARTIFACT rejects all material forgeries."""

    assert mod.validate_artifact(actual_artifact) == []
    forged = deepcopy(actual_artifact)
    del forged["selection_evidence_rows"]
    forged["v613_capstone_complete_score"] = 0
    forged["v613_science_positive_score"] = 1
    forged["g1"] = False
    forged["paper_ready"] = False
    forged["unmet_gates"] = ["G1"]
    forged["solve_claimed"] = True
    forged["hardware_executed"] = True
    forged["honest_verdict"] = "partial_wrong"
    forged["reproducibility_checksum"] = "sha256:forged"
    errors = mod.validate_artifact(forged)
    assert any("missing required field" in error for error in errors)
    assert any("complete score" in error for error in errors)
    assert any("science score" in error for error in errors)
    assert any("publication gate" in error for error in errors)
    assert any("paper_ready" in error for error in errors)
    assert any("unmet_gates" in error for error in errors)
    assert any("claim boundary" in error for error in errors)
    assert any("prefix" in error for error in errors)
    assert any("checksum" in error for error in errors)


def test_command_receipt_and_cli_use_requested_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-7008 records commands and never writes the real result in tests."""

    class Result:
        returncode = 7
        stdout = "finding"
        stderr = "warning"

    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: Result())
    receipt = mod.run_command(REPO_ROOT, "fixture", ["tool", "--check"])
    assert receipt == {
        "name": "fixture",
        "command": "tool --check",
        "exit_code": 7,
        "finding": "finding\nwarning",
        "terminal": True,
    }

    output = tmp_path / "capstone.json"
    assert (
        mod.main(
            [
                "--date",
                "20260905",
                "--root",
                str(REPO_ROOT),
                "--output",
                str(output),
                "--no-commands",
            ]
        )
        == 0
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    assert mod.validate_artifact(written) == []
    assert mod.main(["--validate", str(output)]) == 0
    output.write_text("{}", encoding="utf-8")
    assert mod.main(["--validate", str(output)]) == 1
