"""Tests for residual-pressure verified-memory learning.

Spec refs: REQ-CL-6828, SCENARIO-CL-6828-PRECONDITIONS,
SCENARIO-CL-6828-UPDATES, SCENARIO-CL-6828-EPISODES,
SCENARIO-CL-6828-ADMISSION, SCENARIO-CL-6828-CAPACITY,
SCENARIO-CL-6828-TRANSACTIONS, SCENARIO-CL-6828-FUTURE-SEAL,
SCENARIO-CL-6828-CAUSAL-CREDIT, and SCENARIO-CL-6828-VERDICT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from carnot import experiment_6828_residual_pressure_verified_memory_ab as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / exp.SOURCE_RELATIVE_PATH


@pytest.fixture(scope="module")
def source() -> dict:
    """Load the frozen stream once because every test uses the same bytes."""

    return exp.load_source(SOURCE_PATH)


@pytest.fixture(scope="module")
def artifact(source: dict, tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Build one full comparison for all row-derived assertions."""

    return exp.build_artifact(
        source,
        source_path=SOURCE_PATH,
        state_root=tmp_path_factory.mktemp("exp6828-state"),
        run_date="20260831",
        duration_s=0.25,
    )


def _proposal(
    proposal_id: str,
    key: str,
    value: dict,
    parent_sha256: str,
) -> exp.MemoryProposal:
    """Build one exact add proposal for the transaction tests."""

    return exp.MemoryProposal(
        proposal_id=proposal_id,
        kind="upsert",
        key=key,
        value=value,
        expected_parent_sha256=parent_sha256,
    )


def test_req_cl_6828_spec_precedes_implementation() -> None:
    """REQ-CL-6828 owns every path, field, and public scenario."""

    spec = (REPO_ROOT / exp.SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    section = spec.split("## REQ-CL-6828", 1)[1]
    for requirement_id in exp.OPEN_SPEC_IDS[1:]:
        assert requirement_id in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for path in (exp.MODULE_RELATIVE_PATH, exp.SCRIPT_RELATIVE_PATH, exp.RESULT_RELATIVE_PATH):
        assert path.as_posix() in section


def test_scenario_cl_6828_preconditions_accept_complete_stream(source: dict) -> None:
    """SCENARIO-CL-6828-PRECONDITIONS accepts every owned source gate."""

    summary = exp.check_preconditions(
        source,
        SOURCE_PATH,
        resource_overrides={"disk_free_bytes": 10**9, "ram_available_bytes": 10**9},
    )
    assert summary["passed"] is True
    assert summary["failed_checks"] == []
    assert all(check["passed"] for check in summary["checks"])


@pytest.mark.parametrize(
    ("fault", "failed_check"),
    [
        ("ready", "verified_memory_stream_ready"),
        ("orders", "complete_five_orders"),
        ("splits", "complete_splits"),
        ("headroom", "measurable_headroom"),
        ("admitted", "admissible_operations"),
        ("rejected", "rejected_operations"),
        ("reads", "later_reads"),
        ("serializer", "canonical_serializer"),
        ("disk", "sufficient_disk"),
        ("ram", "sufficient_ram"),
    ],
)
def test_scenario_cl_6828_preconditions_fail_closed(
    source: dict,
    tmp_path: Path,
    fault: str,
    failed_check: str,
) -> None:
    """SCENARIO-CL-6828-PRECONDITIONS keeps exact failed-gate evidence."""

    changed = deepcopy(source)
    overrides = {"disk_free_bytes": 10**9, "ram_available_bytes": 10**9}
    if fault == "ready":
        changed["verified_memory_stream_ready"] = False
    elif fault == "orders":
        changed["order_hashes"].pop("order_5")
    elif fault == "splits":
        changed["split_manifest"]["held_future_count_per_family"] = 0
    elif fault == "headroom":
        changed["headroom_metrics"]["capacity_pressure_case_count"] = 0
    elif fault == "admitted":
        changed["admissible_operation_count"] = 0
    elif fault == "rejected":
        changed["rejected_operation_count"] = 0
    elif fault == "reads":
        changed["later_read_opportunity_count"] = 0
    elif fault == "serializer":
        changed["operation_schema"]["canonical_encoding"] = "platform JSON"
    elif fault == "disk":
        overrides["disk_free_bytes"] = 0
    else:
        overrides["ram_available_bytes"] = 0

    summary = exp.check_preconditions(changed, SOURCE_PATH, resource_overrides=overrides)
    failed = next(check for check in summary["checks"] if check["check"] == failed_check)
    assert summary["passed"] is False
    assert failed["passed"] is False
    assert "expected" in failed and "observed" in failed

    blocked = exp.build_artifact(
        changed,
        source_path=SOURCE_PATH,
        state_root=tmp_path,
        run_date="20260831",
        duration_s=0.25,
        resource_overrides=overrides,
    )
    assert blocked["status"] == exp.BLOCKED_STATUS
    assert blocked["rows"] == []
    assert blocked["transaction_receipts"] == []
    assert blocked["route_learning_ab_complete"] is False
    assert blocked["verdict_class"] == "blocked"
    assert failed_check in blocked["gate_check_summary"]["failed_checks"]


def test_scenario_cl_6828_pressure_updates_have_finite_gain_and_release() -> None:
    """SCENARIO-CL-6828-UPDATES separates finite gain from raw accumulation."""

    residual = exp.residual_pressure_update(0.8, -0.25, gain=0.75, decay=0.25, ceiling=2.0)
    raw = exp.raw_pressure_update(0.8, -0.25, ceiling=2.0)
    assert residual == pytest.approx(0.0125)
    assert raw == pytest.approx(0.55)
    assert exp.residual_pressure_update(1.9, 1.0, gain=0.75, decay=1.0, ceiling=2.0) == 2.0
    assert exp.raw_pressure_update(1.9, 1.0, ceiling=2.0) == 2.0
    assert exp.residual_pressure_update(0.1, -1.0, gain=0.75, decay=0.25, ceiling=2.0) == 0.0
    assert exp.raw_pressure_update(0.1, -1.0, ceiling=2.0) == 0.0


def test_scenario_cl_6828_episode_commit_restart_and_rollback(tmp_path: Path) -> None:
    """SCENARIO-CL-6828-EPISODES and TRANSACTIONS preserve exact bytes."""

    store = exp.VerifiedMemoryStore(tmp_path / "store", arm="residual_pressure", order_id="o1")
    snapshot = store.begin_episode("event-1")
    parent = store.state_bytes()
    proposal = _proposal(
        "proposal-1",
        "revise",
        {"pressure": 0.75, "residual": 1.0},
        snapshot.state_sha256,
    )
    with pytest.raises(exp.ReadOnlyEpisodeError):
        store.commit(
            proposal,
            event_id="event-1",
            exact_receipt={"accepted": False, "reason": "stale_revision"},
            sealed_accept=True,
        )
    assert store.state_bytes() == parent

    store.end_episode()
    receipt = store.commit(
        proposal,
        event_id="event-1",
        exact_receipt={"accepted": False, "reason": "stale_revision"},
        sealed_accept=True,
    )
    assert receipt["accepted"] is True
    assert exp.decode_bytes(receipt["parent_state_bytes"]) == parent
    assert exp.sha256_bytes(exp.decode_bytes(receipt["new_state_bytes"])) == receipt[
        "new_state_sha256"
    ]
    assert receipt["receipt_sha256"] == exp.transaction_sha256(receipt)

    restarted = exp.VerifiedMemoryStore(
        tmp_path / "store", arm="residual_pressure", order_id="o1"
    )
    restart = restarted.restart_receipt("boundary-1", receipt["new_state_sha256"])
    assert restart["bytes_identity"] is True
    assert restarted.state_bytes() == exp.decode_bytes(receipt["new_state_bytes"])

    rollback = restarted.rollback(receipt)
    assert rollback["restored_parent_bytes"] is True
    assert restarted.state_bytes() == parent


def test_scenario_cl_6828_admission_rejects_stale_capacity_and_support(tmp_path: Path) -> None:
    """SCENARIO-CL-6828-ADMISSION rejects unsafe writes without byte changes."""

    store = exp.VerifiedMemoryStore(
        tmp_path / "store", arm="raw_violation_accumulation", order_id="o1", capacity=1
    )
    first = store.commit(
        _proposal("p1", "a", {"pressure": 1.0}, store.state_hash()),
        event_id="e1",
        exact_receipt={"accepted": False},
        sealed_accept=True,
    )
    parent = store.state_bytes()
    rejected = [
        store.commit(
            _proposal("p2", "b", {"pressure": 1.0}, store.state_hash()),
            event_id="e2",
            exact_receipt={"accepted": False},
            sealed_accept=True,
        ),
        store.commit(
            _proposal("p3", "a", {"pressure": 0.0}, "sha256:stale"),
            event_id="e3",
            exact_receipt={"accepted": True},
            sealed_accept=True,
        ),
        store.commit(
            _proposal("p4", "a", {"pressure": 0.0}, store.state_hash()),
            event_id="e4",
            exact_receipt={"accepted": True},
            sealed_accept=False,
        ),
    ]
    assert first["accepted"] is True
    assert {receipt["reason"] for receipt in rejected} == {
        "capacity_exceeded",
        "stale_parent",
        "sealed_support_rejected",
    }
    assert all(receipt["accepted"] is False for receipt in rejected)
    assert all(receipt["parent_state_bytes"] == receipt["new_state_bytes"] for receipt in rejected)
    assert store.state_bytes() == parent


def test_scenario_cl_6828_future_fields_are_denied(source: dict) -> None:
    """SCENARIO-CL-6828-FUTURE-SEAL exposes only the public event allowlist."""

    event = exp.stream_events(source)[0]
    public = exp.public_event(event)
    assert set(public) == set(exp.EVENT_FEATURE_ALLOWLIST)
    assert set(public).isdisjoint(exp.FUTURE_FIELD_DENYLIST)
    with pytest.raises(ValueError, match="denied future field"):
        exp.public_event({**event, "final_acceptance": True})


def test_scenario_cl_6828_equal_capacity_and_complete_rows(artifact: dict) -> None:
    """SCENARIO-CL-6828-CAPACITY gives all four arms the same frozen budget."""

    arms = artifact["arm_definitions"]
    assert set(arms) == set(exp.ARM_NAMES)
    assert {arm["capacity"] for arm in arms.values()} == {exp.MEMORY_CAPACITY}
    assert {arm["proposal_cadence"] for arm in arms.values()} == {exp.PROPOSAL_CADENCE}
    assert {arm["eviction"] for arm in arms.values()} == {exp.EVICTION_RULE}
    assert len(artifact["rows"]) == exp.ORDER_COUNT * exp.EVENT_COUNT * len(exp.ARM_NAMES)
    assert {
        (row["order_id"], row["arm"], row["event_id"]) for row in artifact["rows"]
    } == {
        (row["order_id"], row["arm"], row["event_id"])
        for row in artifact["rows"]
    }
    assert all(row["active_episode_write_count"] == 0 for row in artifact["rows"])


def test_scenario_cl_6828_transactions_and_causal_credit_are_exact(artifact: dict) -> None:
    """SCENARIO-CL-6828-CAUSAL-CREDIT requires a later changed action."""

    assert artifact["transaction_receipts"]
    for receipt in artifact["transaction_receipts"]:
        parent = exp.decode_bytes(receipt["parent_state_bytes"])
        new = exp.decode_bytes(receipt["new_state_bytes"])
        assert exp.sha256_bytes(parent) == receipt["parent_state_sha256"]
        assert exp.sha256_bytes(new) == receipt["new_state_sha256"]
        assert receipt["receipt_sha256"] == exp.transaction_sha256(receipt)
    assert {row["kind"] for row in artifact["causal_edge_counterfactuals"]} == {
        "remove",
        "substitute",
        "reorder",
    }
    assert artifact["causal_factor_witnesses"]
    assert all(
        witness["write"]
        and witness["later_read"]
        and witness["action_changed"]
        and witness["outcome_delta"] != 0
        for witness in artifact["causal_factor_witnesses"]
    )


def test_scenario_cl_6828_verdict_and_required_fields_are_row_derived(artifact: dict) -> None:
    """SCENARIO-CL-6828-VERDICT separates completion from positive effect."""

    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["route_learning_ab_complete"] is True
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["MODEL_SPECS"] == list(exp.MODEL_SPECS)
    assert artifact["model_weight_immutability_receipt"]["all_source_hashes_unchanged"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] in exp.VERDICT_CLASSES
    assert artifact["reproducibility_checksum"] == exp.reproducibility_checksum(artifact)
    assert exp.validate_artifact(artifact) == []

    null_copy = deepcopy(artifact)
    null_copy["acceptance_gate_positive"]["passed"] = False
    null_copy["verdict_class"] = "null"
    null_copy["honest_verdict"] = "complete_null: comparison completed without a positive gate"
    null_copy["reproducibility_checksum"] = exp.reproducibility_checksum(null_copy)
    assert null_copy["route_learning_ab_complete"] is True
    assert exp.validate_artifact(null_copy) == []


def test_validation_write_and_entry_points(
    artifact: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-6828 validates stable output and both command entry points."""

    changed = deepcopy(artifact)
    changed.pop("rows")
    assert "required field set mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["field_principles"].pop("rows")
    assert "field_principles coverage mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["inference_substrate"] = "LLM"
    assert "inference_substrate mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["verdict_class"] = "invented"
    assert "verdict_class outside closed enum" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:wrong"
    assert "reproducibility_checksum mismatch" in exp.validate_artifact(changed)

    result_path = tmp_path / "artifact.json"
    exp.write_artifact(result_path, artifact)
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert exp.main(["--validate", "--result-path", str(result_path)]) == 0

    script = REPO_ROOT / exp.SCRIPT_RELATIVE_PATH
    monkeypatch.setattr(
        "sys.argv",
        [script.name, "--validate", "--result-path", str(result_path)],
    )
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(script), run_name="__main__")
    assert stopped.value.code == 0
