"""Tests for counterfactual credit on bounded external-policy writes.

Spec refs: REQ-CL-6855 and SCENARIO-CL-6855-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import runpy
import sys

import pytest

from carnot import experiment_6855_counterfactual_memory_credit_audit as exp


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def controller_source() -> dict:
    """Load the completed controller artifact once for audit tests."""

    return exp.load_source(REPO / exp.CONTROLLER_RELATIVE_PATH)


@pytest.fixture(scope="session")
def fixture_source() -> dict:
    """Load the exact chronological opportunity fixture once."""

    return exp.load_source(REPO / exp.FIXTURE_RELATIVE_PATH)


@pytest.fixture(scope="session")
def current_artifact(controller_source: dict, fixture_source: dict) -> dict:
    """Build the complete audit without writing tracked state."""

    return exp.build_artifact(
        REPO,
        run_date="20260901",
        duration_s=0.1,
        controller=controller_source,
        fixture=fixture_source,
    )


def _write(
    write_id: str,
    *,
    sequence: int,
    loss: float = 0.1,
    action: str = "abstain",
    features: tuple[float, ...] | None = None,
) -> exp.WriteRecord:
    """Build one exact synthetic write for small reducer tests."""

    return exp.WriteRecord(
        write_id=write_id,
        decision_id=f"decision-{write_id}",
        update_sequence_index=sequence,
        action=action,
        features=features or (1.0,) + (0.0,) * 14,
        bounded_loss=loss,
        support_sha256=f"sha256:{sequence:064x}",
    )


def test_req_cl_6855_complete_artifact_has_required_row_supported_fields(
    current_artifact: dict,
) -> None:
    """REQ-CL-6855: a valid independent replay emits the full audit schema."""

    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(current_artifact)
    assert current_artifact["counterfactual_memory_audit_complete_score"] == 1
    assert current_artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert current_artifact["verifier_is_oracle"] is False
    assert current_artifact["honest_verdict"].startswith("complete_")
    assert current_artifact["rows"]
    assert all(
        {"decision_id", "write_id", "arm", "counterfactual_metric"} <= set(row)
        for row in current_artifact["rows"]
    )


@pytest.mark.parametrize(
    "failure",
    ["score", "decision_hash", "state_hash", "outcome", "contract", "duplicate"],
)
def test_scenario_cl_6855_preconditions_and_duplicates_fail_closed(
    controller_source: dict,
    fixture_source: dict,
    failure: str,
) -> None:
    """SCENARIO-CL-6855-PRECONDITIONS/DUPLICATE-WRITE: invalid evidence blocks."""

    controller = deepcopy(controller_source)
    fixture = deepcopy(fixture_source)
    contract = deepcopy(exp.VALID_COUNTERFACTUAL_CONTRACT)
    if failure == "score":
        controller["risk_sensitive_controller_complete_score"] = 0
    elif failure == "decision_hash":
        controller["rows"][0]["context_sha256"] = "sha256:" + "0" * 64
    elif failure == "state_hash":
        controller["rows"][0]["policy_state_sha256_before"] = "sha256:" + "0" * 64
    elif failure == "outcome":
        fixture["rows"][0]["exact_later_outcome"] = None
    elif failure == "contract":
        contract["declared_valid"] = False
    else:
        controller["update_rows"][1]["update_receipt_sha256"] = controller["update_rows"][0][
            "update_receipt_sha256"
        ]

    artifact = exp.build_artifact(
        REPO,
        run_date="20260901",
        duration_s=0.1,
        controller=controller,
        fixture=fixture,
        contract=contract,
    )
    assert artifact["counterfactual_memory_audit_complete_score"] == 0
    assert artifact["causal_memory_credit_eligible_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == exp.BLOCKED_VERDICT
    assert artifact["gate_check_summary"]["failed_checks"]
    assert artifact["gate_check_summary"]["observed"] is not None


def test_scenario_cl_6855_invalid_coalition_rejects_unknown_and_repeated_writes() -> None:
    """SCENARIO-CL-6855-INVALID-COALITION: each eligible write appears once."""

    with pytest.raises(exp.CounterfactualValidityError, match="duplicate"):
        exp.validate_coalition(["write-a", "write-a"], {"write-a"})
    with pytest.raises(exp.CounterfactualValidityError, match="ineligible"):
        exp.validate_coalition(["write-b"], {"write-a"})
    assert exp.validate_coalition(["write-a"], {"write-a"}) == ("write-a",)


def test_scenario_cl_6855_missing_support_rejects_incompatible_substitution() -> None:
    """SCENARIO-CL-6855-MISSING-SUPPORT: donors must be observed and compatible."""

    target = _write("target", sequence=1)
    wrong_action = _write("action", sequence=2, action="no_memory")
    wrong_features = _write(
        "features",
        sequence=3,
        features=(1.0, 1.0) + (0.0,) * 13,
    )
    assert exp.substitution_support(target, wrong_action) == (
        False,
        "incompatible_action",
    )
    assert exp.substitution_support(target, wrong_features) == (
        False,
        "incompatible_features",
    )
    assert exp.substitution_support(target, target) == (False, "duplicate_write")


def test_scenario_cl_6855_state_path_divergence_fails_closed() -> None:
    """SCENARIO-CL-6855-STATE-PATH-DIVERGENCE: invalid paths cannot reach a target."""

    reducer = exp.FreshReducer()
    with pytest.raises(exp.CounterfactualValidityError, match="target boundary"):
        reducer.replay([_write("late", sequence=4)], target_boundary=4)
    with pytest.raises(exp.CounterfactualValidityError, match="finite"):
        reducer.replay([_write("nan", sequence=1, loss=math.nan)], target_boundary=2)
    selection = reducer.replay([_write("ok", sequence=1)], target_boundary=2)
    assert selection.state_sha256.startswith("sha256:")
    assert selection.applied_write_ids == ("ok",)


def test_scenario_cl_6855_interaction_effect_has_exact_shapley_witness() -> None:
    """SCENARIO-CL-6855-INTERACTION: joint-only value stays visible."""

    def joint_value(coalition: frozenset[str]) -> float:
        return 1.0 if {"a", "b"} <= coalition else 0.0

    rows, receipt = exp.coalition_credit(
        ["a", "b"],
        joint_value,
        random_seed=7,
        exact_limit=8,
        permutation_count=16,
    )
    by_id = {row["write_id"]: row for row in rows}
    classified = exp.classify_credit(
        deletion_effect=0.0,
        coalition_credit=by_id["a"]["marginal_value"],
        zero_headroom=False,
        supported=True,
    )
    assert receipt is None
    assert by_id["a"]["method"] == "exact_enumeration"
    assert by_id["a"]["marginal_value"] == pytest.approx(0.5)
    assert classified["credit_class"] == "interaction_only"
    assert classified["interaction_witness_required"] is True


def test_scenario_cl_6855_methods_never_label_approximation_exact() -> None:
    """SCENARIO-CL-6855-METHODS: large windows carry seeded interval receipts."""

    identifiers = [f"w{index}" for index in range(9)]

    def additive_value(coalition: frozenset[str]) -> float:
        return float(len(coalition))

    rows, receipt = exp.coalition_credit(
        identifiers,
        additive_value,
        random_seed=11,
        exact_limit=8,
        permutation_count=24,
    )
    assert receipt is not None
    assert receipt["method"] == "seeded_permutation_approximation"
    assert receipt["random_seed"] == 11
    assert receipt["permutation_count"] == 24
    assert all(row["exact"] is False for row in rows)
    assert all(row["interval_low"] == row["interval_high"] == 1.0 for row in rows)


def test_scenario_cl_6855_placebo_changes_only_declared_feature(
    fixture_source: dict,
) -> None:
    """SCENARIO-CL-6855-PLACEBO: seeded context control preserves chronology."""

    source_rows = fixture_source["rows"][:12]
    placebo = exp.permute_placebo_contexts(source_rows, feature="age", seed=6855001)
    assert [row["decision_id"] for row in placebo] == [row["decision_id"] for row in source_rows]
    assert [row["available_actions"] for row in placebo] == [
        row["available_actions"] for row in source_rows
    ]
    for source, changed in zip(source_rows, placebo, strict=True):
        source_context = dict(source["decision_context"])
        changed_context = dict(changed["decision_context"])
        source_context.pop("age")
        changed_context.pop("age")
        assert changed_context == source_context
    assert placebo == exp.permute_placebo_contexts(source_rows, feature="age", seed=6855001)


def test_scenario_cl_6855_zero_headroom_separates_causal_and_benefit_fields() -> None:
    """SCENARIO-CL-6855-ZERO-HEADROOM: support does not manufacture benefit."""

    row = exp.classify_credit(
        deletion_effect=0.0,
        coalition_credit=0.0,
        zero_headroom=True,
        supported=True,
    )
    assert row["credit_class"] == "zero_headroom"
    assert row["causal_credit_eligible"] is True
    assert row["benefit_eligible"] is False
    assert row["benefit"] is None


def test_req_cl_6855_source_replay_is_independent_and_hash_stable(
    controller_source: dict,
    fixture_source: dict,
) -> None:
    """REQ-CL-6855: fresh state replay reproduces the source receipts exactly."""

    result = exp.replay_source_hashes(controller_source, fixture_source)
    assert result["passed"] is True
    assert result["decision_hash_mismatch_count"] == 0
    assert result["state_hash_mismatch_count"] == 0
    assert result["update_hash_mismatch_count"] == 0
    assert result["exact_outcome_mismatch_count"] == 0
    assert result["imported_exp6854_aggregate_calculator"] is False


def test_req_cl_6855_ledger_is_unique_and_exactly_supported(
    controller_source: dict,
    fixture_source: dict,
) -> None:
    """REQ-CL-6855: every source update becomes one observed write receipt."""

    ledger = exp.build_write_ledger(controller_source, fixture_source)
    assert len(ledger) == len(controller_source["update_rows"])
    assert len({row.write_id for row in ledger}) == len(ledger)
    assert all(len(row.features) == 15 for row in ledger)
    assert all(row.support_sha256.startswith("sha256:") for row in ledger)


def test_scenario_cl_6855_controls_are_matched_and_chronological(
    current_artifact: dict,
    fixture_source: dict,
) -> None:
    """SCENARIO-CL-6855-CONTROLS: all policies see the same ordered decisions."""

    expected_ids = [row["decision_id"] for row in fixture_source["rows"]]
    required_arms = {
        "learned_selection",
        "no_memory",
        "always_memory",
        "random_admission",
        "abstain",
        "placebo_context",
    }
    assert set(current_artifact["policy_control_summary"]) == required_arms
    for arm in required_arms:
        rows = [row for row in current_artifact["policy_control_rows"] if row["arm"] == arm]
        assert [row["decision_id"] for row in rows] == expected_ids
        assert all(row["action_available"] is True for row in rows)
    assert current_artifact["selection_skill_effect"]["comparison"] == (
        "learned_selection_minus_placebo_context"
    )
    assert current_artifact["stream_luck_effect"]["baseline_arm"] == "no_memory"


def test_req_cl_6855_per_write_summary_preserves_harmful_and_unsupported_evidence(
    current_artifact: dict,
) -> None:
    """REQ-CL-6855: aggregate credit cannot erase harmful or unsupported rows."""

    summary = current_artifact["per_write_credit_summary"]
    assert current_artifact["harmful_write_count"] == sum(
        bool(row.get("harmful_evidence")) for row in summary
    )
    assert current_artifact["redundant_write_count"] == sum(
        row["credit_class"] == "redundant" for row in summary
    )
    assert current_artifact["unsupported_counterfactual_rows"]
    assert current_artifact["deletion_rows"]
    assert current_artifact["substitution_rows"]
    assert current_artifact["order_rows"]
    assert current_artifact["coalition_rows"]
    assert {row["method"] for row in current_artifact["coalition_rows"]} == {
        "exact_enumeration",
        "seeded_permutation_approximation",
    }
    if current_artifact["harmful_write_count"]:
        assert current_artifact["verdict_class"] == "null"
        assert current_artifact["honest_verdict"] == (
            "complete_null_counterfactual_memory_credit_harmful_writes_present"
        )


def test_scenario_cl_6855_gate_completeness_and_credit_are_separate() -> None:
    """SCENARIO-CL-6855-GATES: complete replay can have no eligible benefit."""

    gates = exp.compute_terminal_gates(
        planned_decisions=2,
        completed_decisions=2,
        supported_replay_count=2,
        causal_benefit_count=0,
        validation_passed=True,
    )
    assert gates["counterfactual_memory_audit_complete_score"] == 1
    assert gates["causal_memory_credit_eligible_score"] == 0


def test_req_cl_6855_artifact_matches_deterministic_replay(
    current_artifact: dict,
) -> None:
    """REQ-CL-6855: the checked-in artifact matches stable replay content."""

    stored = exp.load_source(REPO / exp.RESULT_RELATIVE_PATH)
    assert stored["reproducibility_checksum"] == exp.reproducibility_checksum(stored)
    assert stored["rows"] == current_artifact["rows"]
    assert stored["per_write_credit_summary"] == current_artifact["per_write_credit_summary"]
    assert stored["source_artifact_hashes"] == exp.source_artifact_hashes(REPO)


def test_req_cl_6855_writer_cli_and_validation_use_requested_paths(
    current_artifact: dict,
    tmp_path: Path,
) -> None:
    """REQ-CL-6855: tests and callers can isolate every artifact write."""

    invalid = deepcopy(current_artifact)
    invalid["reproducibility_checksum"] = "stale"
    with pytest.raises(ValueError, match="reproducibility_checksum mismatch"):
        exp.write_artifact(tmp_path / "invalid.json", invalid)

    output = tmp_path / "nested" / "artifact.json"
    exp.write_artifact(output, current_artifact)
    assert json.loads(output.read_text(encoding="utf-8")) == current_artifact

    cli_output = tmp_path / "cli.json"
    assert exp.main(["--date", "20260901", "--output", str(cli_output)]) == 0
    cli_artifact = json.loads(cli_output.read_text(encoding="utf-8"))
    assert cli_artifact["counterfactual_memory_audit_complete_score"] == 1

    malformed = deepcopy(current_artifact)
    malformed.pop("rows")
    malformed["inference_substrate"] = "wrong"
    malformed["verifier_is_oracle"] = True
    malformed["honest_verdict"] = "pending"
    errors = exp.validate_artifact(malformed)
    assert any("missing required fields" in error for error in errors)
    assert "invalid inference_substrate" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "honest_verdict must start with complete_" in errors
    assert "reproducibility_checksum mismatch" in errors


def test_req_cl_6855_source_loader_reports_unreadable_inputs(tmp_path: Path) -> None:
    """REQ-CL-6855: unreadable evidence becomes explicit blocked input."""

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{invalid", encoding="utf-8")
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    assert exp.load_source(tmp_path / "missing.json") == {"_load_error": "FileNotFoundError"}
    assert exp.load_source(invalid) == {"_load_error": "JSONDecodeError"}
    assert exp.load_source(array) == {"_load_error": "not_object"}


def test_scenario_cl_6855_invalid_transition_inputs_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-CL-6855-STATE-PATH-DIVERGENCE: reject malformed state inputs."""

    assert exp.sha256_file(tmp_path / "missing.json") is None
    with pytest.raises(exp.CounterfactualValidityError, match="capacity"):
        exp.context_features({"capacity": []})
    with pytest.raises(exp.CounterfactualValidityError, match="category"):
        exp.context_features({"capacity": {}, "family": "unknown"})
    invalid_context = {
        "capacity": {"budget": 2},
        "family": exp.FAMILIES[0],
        "correction_status": exp.CORRECTION_STATUSES[0],
        "relevance": True,
        "uncertainty": 0.2,
        "age": 1,
        "false_positive_risk": 0.3,
    }
    with pytest.raises(exp.CounterfactualValidityError, match="number"):
        exp.context_features(invalid_context)

    reducer = exp.FreshReducer()
    with pytest.raises(exp.CounterfactualValidityError, match="action"):
        reducer.apply(_write("bad-action", sequence=1, action="unknown"))
    with pytest.raises(exp.CounterfactualValidityError, match="feature length"):
        reducer.apply(_write("bad-features", sequence=1, features=(1.0,)))
    with pytest.raises(exp.CounterfactualValidityError, match="loss exceeds"):
        reducer.apply(_write("bad-loss", sequence=1, loss=4.0))
    with pytest.raises(exp.CounterfactualValidityError, match="target feature"):
        reducer.scores((1.0,))
    duplicate = _write("duplicate", sequence=1)
    with pytest.raises(exp.CounterfactualValidityError, match="duplicate write"):
        reducer.replay([duplicate, duplicate], target_boundary=2)


def test_req_cl_6855_support_validation_covers_rejected_evidence(
    controller_source: dict,
    fixture_source: dict,
) -> None:
    """REQ-CL-6855: source and donor validation retain each rejected reason."""

    incomplete = deepcopy(controller_source)
    incomplete["rows"].pop()
    replay = exp.replay_source_hashes(incomplete, fixture_source)
    assert replay["decision_hash_mismatch_count"] > 0
    malformed_replay = exp.replay_source_hashes({}, {})
    assert malformed_replay["error"] == "KeyError"

    duplicate = deepcopy(controller_source)
    duplicate["update_rows"][1]["update_receipt_sha256"] = duplicate["update_rows"][0][
        "update_receipt_sha256"
    ]
    with pytest.raises(exp.CounterfactualValidityError, match="duplicate write"):
        exp.build_write_ledger(duplicate, fixture_source)

    target = _write("target", sequence=1)
    unsupported_donor = exp.WriteRecord(
        write_id="donor",
        decision_id="decision-donor",
        update_sequence_index=2,
        action=target.action,
        features=target.features,
        bounded_loss=target.bounded_loss,
        support_sha256="missing",
    )
    assert exp.substitution_support(target, unsupported_donor) == (
        False,
        "missing_exact_support",
    )
    with pytest.raises(ValueError, match="at least two"):
        exp.coalition_credit(
            [f"w{index}" for index in range(9)],
            lambda coalition: float(len(coalition)),
            random_seed=1,
            permutation_count=1,
        )
    unsupported = exp.classify_credit(
        deletion_effect=0.0,
        coalition_credit=0.0,
        zero_headroom=False,
        supported=False,
    )
    helpful = exp.classify_credit(
        deletion_effect=1.0,
        coalition_credit=1.0,
        zero_headroom=False,
        supported=True,
    )
    assert unsupported["credit_class"] == "unsupported"
    assert helpful["credit_class"] == "helpful"
    with pytest.raises(exp.CounterfactualValidityError, match="unavailable"):
        exp.permute_placebo_contexts([], feature="age", seed=1)


def test_scenario_cl_6855_per_write_reducer_preserves_helpful_class() -> None:
    """SCENARIO-CL-6855-INTERACTION: reduction keeps a helpful-only write helpful."""

    write = _write("helpful", sequence=1)
    rows = [
        {
            "write_id": write.write_id,
            "benefit_eligible": True,
            "deletion_effect": 1.0,
            "marginal_value": 1.0,
            "credit_class": "helpful",
            "harmful_evidence": False,
            "helpful_evidence": True,
        }
    ]
    summary, witnesses = exp._per_write_summary([write], rows)
    assert summary[0]["credit_class"] == "helpful"
    assert summary[0]["helpful_evidence"] is True
    assert witnesses == []


@pytest.mark.parametrize(
    ("causal_score", "expected_class", "expected_verdict"),
    [
        (
            1,
            "positive",
            "complete_positive_counterfactual_memory_credit_supported",
        ),
        (
            0,
            "null",
            "complete_null_counterfactual_memory_credit_no_eligible_effect",
        ),
    ],
)
def test_req_cl_6855_terminal_verdict_branches_are_row_supported(
    monkeypatch: pytest.MonkeyPatch,
    causal_score: int,
    expected_class: str,
    expected_verdict: str,
) -> None:
    """REQ-CL-6855: positive and null verdicts follow the computed causal gate."""

    write = _write("supported", sequence=1)
    checks = [exp._check("synthetic_source", True, True, True)]
    counterfactuals = {
        "deletion_rows": [
            {
                "decision_id": "decision-supported",
                "write_id": write.write_id,
                "deletion_effect": float(causal_score),
                "method": "exact_transition_replay",
            }
        ],
        "substitution_rows": [],
        "order_rows": [],
        "coalition_rows": [],
        "approximation_receipts": [],
        "unsupported_counterfactual_rows": [],
    }
    summary = [
        {
            "write_id": write.write_id,
            "credit_class": "helpful" if causal_score else "redundant",
            "harmful_evidence": False,
        }
    ]
    policy_rows = [
        {
            "decision_id": "decision-supported",
            "arm": "learned_selection",
            "reward": 0.0,
        }
    ]
    monkeypatch.setattr(
        exp,
        "validate_preconditions",
        lambda controller, fixture, contract: (checks, {"passed": True}),
    )
    monkeypatch.setattr(exp, "build_write_ledger", lambda controller, fixture: [write])
    monkeypatch.setattr(exp, "_counterfactual_rows", lambda fixture, ledger: counterfactuals)
    monkeypatch.setattr(exp, "_per_write_summary", lambda ledger, rows: (summary, []))
    monkeypatch.setattr(
        exp,
        "_policy_controls",
        lambda controller, fixture, ledger: (policy_rows, {}, [], {}, {}),
    )
    monkeypatch.setattr(
        exp,
        "compute_terminal_gates",
        lambda **kwargs: {
            "counterfactual_memory_audit_complete_score": 1,
            "causal_memory_credit_eligible_score": causal_score,
        },
    )

    artifact = exp.build_artifact(
        REPO,
        run_date="20260901",
        duration_s=0.1,
        controller={},
        fixture={"rows": [{"decision_id": "decision-supported"}]},
    )
    assert artifact["verdict_class"] == expected_class
    assert artifact["honest_verdict"] == expected_verdict


def test_req_cl_6855_artifact_validation_rejects_terminal_contradictions(
    current_artifact: dict,
) -> None:
    """REQ-CL-6855: complete artifacts cannot carry invalid classes or failed gates."""

    malformed = dict(current_artifact)
    malformed["verdict_class"] = "unknown"
    malformed["gate_check_summary"] = {"passed": False}
    malformed["reproducibility_checksum"] = exp.reproducibility_checksum(malformed)
    errors = exp.validate_artifact(malformed)
    assert "invalid verdict_class" in errors
    assert "complete score contradicts failed preconditions" in errors


def test_req_cl_6855_package_entrypoint_writes_isolated_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-6855: the package entry point exits after an isolated full replay."""

    output = tmp_path / "module-entrypoint.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "experiment_6855_counterfactual_memory_credit_audit.py",
            "--date",
            "20260901",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module(exp.__name__, run_name="__main__")
    assert exit_info.value.code == 0
    assert (
        json.loads(output.read_text(encoding="utf-8"))["counterfactual_memory_audit_complete_score"]
        == 1
    )
