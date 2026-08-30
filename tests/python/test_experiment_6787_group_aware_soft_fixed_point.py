"""Focused tests for the bounded group-aware soft fixed-point proposer.

Spec refs: REQ-VERIFY-6787 and SCENARIO-VERIFY-6787-*.
"""

from __future__ import annotations

from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from carnot import experiment_6787_group_aware_soft_fixed_point as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/verifiable-reasoning/spec.md"
SOURCE_PATH = REPO_ROOT / exp.SOURCE_ARTIFACT_RELATIVE_PATH


@pytest.fixture(scope="module")
def source() -> dict:
    """Load the frozen input once because all tests treat it as immutable."""

    return exp.load_json_object(SOURCE_PATH)


@pytest.fixture(scope="module")
def units(source: dict) -> list[dict]:
    """Project exact fixture units to the proposal-only schema once."""

    return exp.project_units(source)


def test_req_verify_6787_spec_declares_the_bounded_oracle_distinct_contract() -> None:
    """REQ-VERIFY-6787 anchors recurrence, isolation, replay, and blocking."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-VERIFY-6787")
    section = spec[start : spec.index("### SCENARIO-VERIFY-6745", start)]
    for marker in (
        "REQ-VERIFY-6787",
        "SCENARIO-VERIFY-6787-GROUP-RECURRENCE",
        "SCENARIO-VERIFY-6787-BOUND-AND-DECODE",
        "SCENARIO-VERIFY-6787-SPLIT-AND-ORACLE-ISOLATION",
        "SCENARIO-VERIFY-6787-DETERMINISTIC-REPLAY",
        "SCENARIO-VERIFY-6787-BLOCKED-PRECONDITION",
        "complete_blocked_soft_fixed_point",
        "soft_fixed_point_proposer_ready",
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section or field in exp.STANDARD_ARTIFACT_FIELDS


def test_scenario_verify_6787_group_serialization_preserves_every_identity(
    units: list[dict],
) -> None:
    """SCENARIO-VERIFY-6787-GROUP-RECURRENCE keeps graph and group IDs stable."""

    assert len(units) == 96
    assert exp.audit_feature_contract(units) == []
    assert {unit["split"] for unit in units} == set(exp.REQUIRED_SPLITS)
    for unit in units:
        features = unit["proposal_features"]
        assert set(features) == set(exp.FEATURE_ALLOWLIST)
        assert unit["graph_serialization"] == exp.canonical_json(features)
        assert unit["graph_id"] == features["graph_id"]
        assert unit["group_ids"] == [group["group_id"] for group in features["local_groups"]]
        assert unit["variable_ids"] == features["variables"]
        assert unit["dependency_ids"] == [
            edge["dependency_id"] for edge in features["dependency_edges"]
        ]


def test_req_verify_6787_parameter_count_and_initialization_are_deterministic() -> None:
    """REQ-VERIFY-6787 fixes parameter count and seed-derived initialization."""

    first = exp.GroupAwareSoftFixedPoint(hidden_width=8, seed=101)
    second = exp.GroupAwareSoftFixedPoint(hidden_width=8, seed=101)
    changed = exp.GroupAwareSoftFixedPoint(hidden_width=8, seed=102)

    assert exp.trainable_parameter_count(first) == exp.expected_parameter_count(8) == 91
    assert all(
        torch.equal(left, right) for left, right in zip(first.parameters(), second.parameters())
    )
    assert any(
        not torch.equal(left, right)
        for left, right in zip(first.parameters(), changed.parameters())
    )
    with pytest.raises(ValueError, match="hidden_width"):
        exp.GroupAwareSoftFixedPoint(hidden_width=0, seed=1)
    with pytest.raises(ValueError, match="hidden_width"):
        exp.expected_parameter_count(0)


def test_scenario_verify_6787_recurrent_update_has_separate_finite_messages(
    units: list[dict],
) -> None:
    """SCENARIO-VERIFY-6787-GROUP-RECURRENCE emits one group receipt per iteration."""

    unit = next(row for row in units if row["split"] == "development")
    model = exp.GroupAwareSoftFixedPoint(hidden_width=8, seed=6787001)
    initial = exp.initial_variable_state(unit["proposal_features"], seed=6787001)
    stepped = model.recurrent_step(initial, unit["proposal_features"])

    assert stepped.variable_state.shape == initial.shape
    assert stepped.group_messages.shape == initial.shape
    assert stepped.dependency_messages.shape[0] == len(unit["dependency_ids"])
    assert stepped.dependency_messages.shape[1] == 2
    assert not torch.equal(initial, stepped.variable_state)
    assert all(
        torch.isfinite(value).all()
        for value in (
            stepped.variable_state,
            stepped.group_messages,
            stepped.dependency_messages,
        )
    )

    result = exp.run_fixed_point(
        model,
        unit,
        seed=6787001,
        iteration_cap=3,
        convergence_tolerance=0.0,
    )
    assert result["iterations"] == 3
    assert result["stop_reason"] == "iteration_cap"
    assert result["finite_values"] is True
    assert len(result["group_message_receipts"]) == 3 * len(unit["group_ids"])
    assert {
        (receipt["iteration"], receipt["group_id"]) for receipt in result["group_message_receipts"]
    } == {(iteration, group_id) for iteration in range(1, 4) for group_id in unit["group_ids"]}

    with pytest.raises(ValueError, match="variable state shape"):
        model.recurrent_step(torch.ones((1, 2), dtype=torch.float64), unit["proposal_features"])
    with pytest.raises(ValueError, match="iteration_cap"):
        exp.run_fixed_point(
            model,
            unit,
            seed=1,
            iteration_cap=0,
            convergence_tolerance=0.0,
        )
    with pytest.raises(ValueError, match="convergence_tolerance"):
        exp.run_fixed_point(
            model,
            unit,
            seed=1,
            iteration_cap=1,
            convergence_tolerance=-1.0,
        )


def test_scenario_verify_6787_convergence_cap_and_candidate_decode_are_bounded(
    units: list[dict],
) -> None:
    """SCENARIO-VERIFY-6787-BOUND-AND-DECODE caps recurrence and fills all variables."""

    unit = next(row for row in units if row["split"] == "held_topology_test")
    model = exp.GroupAwareSoftFixedPoint(hidden_width=8, seed=6787002)
    result = exp.run_fixed_point(
        model,
        unit,
        seed=6787002,
        iteration_cap=7,
        convergence_tolerance=1.0,
    )
    candidates = exp.decode_candidates(
        result["variable_state_tensor"],
        unit,
        seed=6787002,
        threshold=0.5,
        candidate_count=3,
    )

    assert result["iterations"] == 1
    assert result["stop_reason"] == "converged"
    assert len(candidates) == 3
    assert len({candidate["candidate_hash"] for candidate in candidates}) == 3
    for index, candidate in enumerate(candidates):
        assert candidate["candidate_index"] == index
        assert set(candidate["assignment"]) == set(unit["variable_ids"])
        for group in unit["proposal_features"]["local_groups"]:
            assert sum(candidate["assignment"][name] for name in group["variables"]) == 1
        assert candidate["candidate_hash"] == exp.sha256_json(candidate["assignment"])
    with pytest.raises(ValueError, match="candidate_count"):
        exp.decode_candidates(
            result["variable_state_tensor"],
            unit,
            seed=1,
            threshold=0.5,
            candidate_count=0,
        )
    with pytest.raises(ValueError, match="decoding threshold"):
        exp.decode_candidates(
            result["variable_state_tensor"],
            unit,
            seed=1,
            threshold=1.1,
            candidate_count=1,
        )


def test_scenario_verify_6787_split_isolation_and_oracle_refusal(
    units: list[dict],
) -> None:
    """SCENARIO-VERIFY-6787-SPLIT-AND-ORACLE-ISOLATION keeps held rows out of fit."""

    train_units = [unit for unit in units if unit["split"] == "train"]
    held_units = [unit for unit in units if unit["split"] != "train"]
    model, receipt = exp.fit_seed(
        train_units,
        seed=6787003,
        hyperparameters={**exp.FROZEN_HYPERPARAMETERS, "training_steps": 2},
    )
    row = exp.propose_unit(
        model,
        held_units[0],
        seed=6787003,
        hyperparameters={**exp.FROZEN_HYPERPARAMETERS, "iteration_cap": 2},
    )

    assert receipt["training_steps"] == 2
    assert receipt["train_unit_ids"] == [unit["unit_id"] for unit in train_units]
    assert set(receipt["train_splits_seen"]) == {"train"}
    assert row["split"] in {"development", "held_topology_test"}
    assert row["unit_id"] not in receipt["train_unit_ids"]

    contaminated = deepcopy(held_units[0])
    contaminated["proposal_features"]["exact_valid"] = True
    assert exp.audit_feature_contract([contaminated]) == [
        f"{contaminated['unit_id']}.proposal_features.exact_valid"
    ]
    with pytest.raises(ValueError, match="oracle feature refusal"):
        exp.propose_unit(
            model,
            contaminated,
            seed=6787003,
            hyperparameters=exp.FROZEN_HYPERPARAMETERS,
        )
    with pytest.raises(ValueError, match="train split"):
        exp.fit_seed(held_units[:1], seed=1, hyperparameters=exp.FROZEN_HYPERPARAMETERS)

    contaminated_train = deepcopy(train_units[0])
    contaminated_train["proposal_features"]["local_groups"][0]["exact_label"] = 1
    with pytest.raises(ValueError, match="oracle feature refusal"):
        exp.fit_seed(
            [contaminated_train],
            seed=1,
            hyperparameters=exp.FROZEN_HYPERPARAMETERS,
        )


def test_req_verify_6787_preconditions_fail_closed_on_every_gate(
    source: dict, tmp_path: Path
) -> None:
    """REQ-VERIFY-6787 checks readiness, exact bytes, splits, and feature denial."""

    clean = exp.evaluate_preconditions(SOURCE_PATH)
    assert clean["all_passed"] is True
    assert [check["check"] for check in clean["checks"]] == [
        "exp6786_artifact_exists",
        "constraint_group_fixture_ready",
        "exact_artifact_hash_agreement",
        "train_split_nonempty",
        "development_split_nonempty",
        "held_topology_test_split_nonempty",
        "feature_denylist_clean",
    ]

    missing = exp.evaluate_preconditions(tmp_path / "missing.json")
    assert missing["first_failure"]["check"] == "exp6786_artifact_exists"

    drift = tmp_path / "drift.json"
    drift.write_text(SOURCE_PATH.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert exp.evaluate_preconditions(drift)["first_failure"]["check"] == (
        "exact_artifact_hash_agreement"
    )

    changes = (
        (
            lambda value: value.__setitem__("constraint_group_fixture_ready", False),
            "constraint_group_fixture_ready",
        ),
        (
            lambda value: value["split_by_topology"]["train"].__setitem__("unit_count", 0),
            "train_split_nonempty",
        ),
        (
            lambda value: value["split_by_topology"]["development"].__setitem__("unit_count", 0),
            "development_split_nonempty",
        ),
        (
            lambda value: value["split_by_topology"]["held_topology_test"].__setitem__(
                "unit_count", 0
            ),
            "held_topology_test_split_nonempty",
        ),
        (
            lambda value: value["rows"][0]["proposal_features"].__setitem__("exact_valid", False),
            "feature_denylist_clean",
        ),
    )
    for index, (mutate, expected) in enumerate(changes):
        changed = deepcopy(source)
        mutate(changed)
        path = tmp_path / f"changed-{index}.json"
        path.write_text(json.dumps(changed), encoding="utf-8")
        expected_hash = exp.sha256_file(path)
        assert (
            exp.evaluate_preconditions(path, expected_hash=expected_hash)["first_failure"]["check"]
            == expected
        )

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON root"):
        exp.load_json_object(malformed)

    assert exp.audit_feature_contract([{"unit_id": "bad-shape", "proposal_features": []}]) == [
        "bad-shape.proposal_features"
    ]
    nested = deepcopy(exp.project_units(source)[0])
    nested["proposal_features"]["local_groups"][0]["exact_valid"] = False
    assert exp.audit_feature_contract([nested]) == [
        f"{nested['unit_id']}.proposal_features.local_groups[0].exact_valid"
    ]


def test_req_verify_6787_unknown_relations_and_nonfinite_updates_fail_closed(
    units: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-6787 rejects unknown semantics and records non-finite recurrence."""

    unit = deepcopy(units[0])
    state = exp.initial_variable_state(unit["proposal_features"], seed=1)
    unit["proposal_features"]["dependency_edges"][0]["relation_type"] = "unknown"
    with pytest.raises(ValueError, match="unknown dependency relation"):
        exp.GroupAwareSoftFixedPoint._dependency_messages(state, unit["proposal_features"])
    with pytest.raises(ValueError, match="unknown dependency relation"):
        exp._dependency_violation(state, unit["proposal_features"])

    no_edges = deepcopy(units[0]["proposal_features"])
    no_edges["dependency_edges"] = []
    edge_messages, aggregate, degrees = exp.GroupAwareSoftFixedPoint._dependency_messages(
        state, no_edges
    )
    assert edge_messages.shape == (0, 2)
    assert torch.equal(aggregate, torch.full_like(aggregate, 0.5))
    assert torch.equal(degrees, torch.zeros_like(degrees))
    assert exp._dependency_violation(state, no_edges).item() == 0.0

    clean_unit = units[0]
    model = exp.GroupAwareSoftFixedPoint(hidden_width=8, seed=1)
    group_count = len(clean_unit["group_ids"])
    edge_count = len(clean_unit["dependency_ids"])

    def nonfinite_step(_state: torch.Tensor, _features: dict) -> exp.RecurrentStep:
        return exp.RecurrentStep(
            variable_state=torch.full((group_count, 2), float("nan"), dtype=torch.float64),
            group_messages=torch.full((group_count, 2), 0.5, dtype=torch.float64),
            dependency_messages=torch.full((edge_count, 2), 0.5, dtype=torch.float64),
            aggregated_dependency_messages=torch.full((group_count, 2), 0.5, dtype=torch.float64),
        )

    monkeypatch.setattr(model, "recurrent_step", nonfinite_step)
    result = exp.run_fixed_point(
        model,
        clean_unit,
        seed=1,
        iteration_cap=2,
        convergence_tolerance=0.0,
    )
    assert result["iterations"] == 1
    assert result["stop_reason"] == "non_finite"
    assert result["finite_values"] is False

    with pytest.raises(ValueError, match="replay precondition failed"):
        exp.replay_seed(
            Path("/definitely/missing/experiment-6786.json"),
            seed=1,
            expected_hash="sha256:missing",
        )


def test_scenario_verify_6787_artifact_and_fresh_replay_are_complete(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6787-DETERMINISTIC-REPLAY emits a ready row-derived artifact."""

    output = tmp_path / "experiment_6787.json"
    artifact = exp.write_outputs(
        run_date="20260830",
        source_artifact_path=SOURCE_PATH,
        artifact_path=output,
        repo_root=REPO_ROOT,
        duration_s=2.5,
    )
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded == artifact
    assert exp.validate_artifact(artifact) == []
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["source_artifact_hash"] == exp.EXPECTED_SOURCE_ARTIFACT_HASH
    assert artifact["frozen_hyperparameters"] == exp.FROZEN_HYPERPARAMETERS
    assert artifact["trainable_parameter_count"] == exp.expected_parameter_count(
        exp.FROZEN_HYPERPARAMETERS["hidden_width"]
    )
    assert len(artifact["training_receipts"]) == 5
    assert len(artifact["rows"]) == 64 * 5
    assert {row["split"] for row in artifact["rows"]} == {
        "development",
        "held_topology_test",
    }
    assert artifact["finite_value_failures"] == []
    assert artifact["deterministic_replay_agreement"]["agreement"] is True
    assert artifact["deterministic_replay_agreement"]["fresh_process"] is True
    assert artifact["soft_fixed_point_proposer_ready"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["reproducibility_checksum"] == exp.reproducibility_checksum(artifact)
    for split, summary in artifact["convergence_by_split"].items():
        assert split in {"development", "held_topology_test"}
        assert summary["row_count"] == 32 * 5

    ready_mutations = (
        (lambda value: value.__setitem__("status", "bad"), "ready artifact status mismatch"),
        (
            lambda value: value.__setitem__("source_artifact_hash", "bad"),
            "ready artifact source hash mismatch",
        ),
        (
            lambda value: value["frozen_hyperparameters"].__setitem__("training_steps", 99),
            "ready artifact hyperparameters drifted",
        ),
        (
            lambda value: value.__setitem__("trainable_parameter_count", 0),
            "ready artifact parameter count mismatch",
        ),
        (
            lambda value: value.__setitem__("oracle_feature_violations", ["bad"]),
            "ready artifact contains oracle feature violations",
        ),
        (
            lambda value: value.__setitem__("finite_value_failures", ["bad"]),
            "ready artifact contains finite-value failures",
        ),
        (
            lambda value: value["gate_check_summary"].__setitem__("all_passed", False),
            "ready artifact has failed preconditions",
        ),
        (
            lambda value: value["deterministic_replay_agreement"].__setitem__("agreement", False),
            "ready artifact lacks deterministic replay",
        ),
        (lambda value: value.__setitem__("rows", []), "ready artifact row count mismatch"),
        (
            lambda value: value.__setitem__("training_receipts", []),
            "ready artifact training receipt count mismatch",
        ),
        (
            lambda value: value.__setitem__("candidate_hashes", []),
            "ready artifact candidate hash index mismatch",
        ),
        (
            lambda value: value["rows"][0].__setitem__("group_message_receipts", []),
            "row group receipt count mismatch",
        ),
    )
    for mutate, expected_error in ready_mutations:
        changed = deepcopy(artifact)
        mutate(changed)
        assert any(expected_error in error for error in exp.validate_artifact(changed))


def test_scenario_verify_6787_blocked_artifact_and_defensive_validation(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6787-BLOCKED-PRECONDITION writes a full terminal block."""

    artifact = exp.build_artifact(
        run_date="20260830",
        source_artifact_path=tmp_path / "missing.json",
        repo_root=REPO_ROOT,
        duration_s=0.1,
    )
    assert exp.validate_artifact(artifact) == []
    assert artifact["status"] == "complete_blocked_soft_fixed_point"
    assert artifact["rows"] == []
    assert artifact["candidate_hashes"] == []
    assert artifact["soft_fixed_point_proposer_ready"] is False
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("complete_blocked_soft_fixed_point")
    assert artifact["gate_check_summary"]["first_failure"]["check"] == ("exp6786_artifact_exists")

    with pytest.raises(ValueError, match="YYYYMMDD"):
        exp.build_artifact(run_date="2026-08-30", source_artifact_path=SOURCE_PATH)

    mutations = (
        (lambda value: value.pop("schema"), "required field set mismatch"),
        (lambda value: value["field_principles"].pop("rows"), "field principle coverage mismatch"),
        (
            lambda value: value.__setitem__("inference_substrate", "bad"),
            "inference substrate mismatch",
        ),
        (lambda value: value.__setitem__("duration_s", -1), "duration_s must be non-negative"),
        (lambda value: value.__setitem__("random_seed", -1), "random seed mismatch"),
        (
            lambda value: value.__setitem__("verdict_class", "bad"),
            "verdict class is outside the closed enum",
        ),
        (
            lambda value: value.__setitem__("honest_verdict", "bad"),
            "honest verdict lacks a terminal prefix",
        ),
        (
            lambda value: value.__setitem__("verifier_is_oracle", True),
            "verifier_is_oracle must remain false",
        ),
        (
            lambda value: value.__setitem__("reproducibility_checksum", "bad"),
            "reproducibility checksum mismatch",
        ),
        (lambda value: value.__setitem__("rows", [{}]), "blocked artifact must not contain rows"),
        (lambda value: value.__setitem__("status", "bad"), "blocked artifact status mismatch"),
        (
            lambda value: value.__setitem__("candidate_hashes", ["bad"]),
            "blocked artifact must not contain candidate hashes",
        ),
        (
            lambda value: value.__setitem__("soft_fixed_point_proposer_ready", True),
            "blocked artifact cannot be ready",
        ),
    )
    for mutate, expected_error in mutations:
        changed = deepcopy(artifact)
        mutate(changed)
        assert expected_error in exp.validate_artifact(changed)


def test_req_verify_6787_worker_errors_and_cli_are_explicit(
    source: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-VERIFY-6787 exposes replay failures and keeps CLI writes configurable."""

    payload = {
        "source_artifact_path": str(SOURCE_PATH),
        "expected_hash": exp.EXPECTED_SOURCE_ARTIFACT_HASH,
        "seed": exp.FROZEN_HYPERPARAMETERS["seeds"][0],
    }
    monkeypatch.setattr(exp.sys, "stdin", StringIO(json.dumps(payload)))
    assert exp._replay_worker() == 0
    worker = json.loads(capsys.readouterr().out)
    assert worker["candidate_hashes"]

    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=3, stderr="forced replay failure"),
    )
    with pytest.raises(RuntimeError, match="forced replay failure"):
        exp.run_fresh_replay(
            SOURCE_PATH,
            seed=exp.FROZEN_HYPERPARAMETERS["seeds"][0],
            expected_hash=exp.EXPECTED_SOURCE_ARTIFACT_HASH,
            repo_root=REPO_ROOT,
        )

    monkeypatch.setattr(exp, "_replay_worker", lambda: 7)
    assert exp.main(["--replay-worker"]) == 7

    output = tmp_path / "cli-blocked.json"
    assert (
        exp.main(
            [
                "--date",
                "20260830",
                "--source-artifact",
                str(tmp_path / "missing.json"),
                "--artifact-path",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text())["status"] == "complete_blocked_soft_fixed_point"
    assert "complete_blocked_soft_fixed_point" in capsys.readouterr().out
