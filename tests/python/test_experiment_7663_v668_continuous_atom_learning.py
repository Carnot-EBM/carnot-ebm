"""Focused contract checks for REQ-REPORT-7663."""

from pathlib import Path

import pytest

from carnot.reporting.experiment_7663_continuous_learning import (
    CostGuardedService,
    paired_block_interval,
    replay_arm,
)


def _prepared_service(tmp_path: Path, admission_labels: list[int]) -> CostGuardedService:
    service = CostGuardedService(tmp_path / "state.json", arm="source")
    labels = [1] * 5 + admission_labels
    for origin, label in enumerate(labels):
        service.predict(str(origin), origin, 0.1, 0, "update" if origin < 5 else "admission")
        service.release(str(origin), label, origin + 8)
    service.propose([str(i) for i in range(5)])
    return service


def test_req_report_7663_admission_accepts_and_uses_labels_once(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7663-ADMISSION: held-out Brier and cost govern acceptance."""
    service = _prepared_service(tmp_path, [1] * 5)
    outcome = service.admit([str(i) for i in range(5, 10)])
    assert outcome["accepted"]
    assert outcome["candidate_admission_brier"] < outcome["prior_admission_brier"]
    assert outcome["candidate_admission_cost"] <= outcome["prior_admission_cost"]
    assert service.state["used_updates"] == [str(i) for i in range(5)]
    assert service.state["used_admissions"] == [str(i) for i in range(5, 10)]
    with pytest.raises(ValueError, match="admission_used"):
        service.admit([str(i) for i in range(5, 10)])


def test_req_report_7663_admission_rolls_back(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7663-ADMISSION: a worse held-out candidate is rolled back."""
    service = _prepared_service(tmp_path, [0] * 5)
    before = service.numerical_hash
    outcome = service.admit([str(i) for i in range(5, 10)])
    assert not outcome["accepted"]
    assert outcome["rollback_reason"] == "brier_not_improved"
    assert service.numerical_hash == before


def test_req_report_7663_block_interval_is_paired() -> None:
    """SCENARIO-REPORT-7663-TERMINAL: block resampling keeps paired group identity."""
    paired = [(0.2, 0.4)] * 80
    interval = paired_block_interval(paired, 11, draws=100)
    assert interval["effective_blocks"] == 10
    assert interval["lower_ci95"] == pytest.approx(0.2)
    assert interval["upper_ci95"] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="block_roster"):
        paired_block_interval(paired[:-1], 11, draws=100)


def test_req_report_7663_replay_restart_and_missing(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7663-CAUSAL and RESTART: durable replay and omissions."""
    inputs = [
        {
            "unit_id": str(i),
            "raw_probability": 0.2,
            "label": i % 2,
            "partition": "update" if (i // 5) % 2 == 0 else "admission",
            "feature": {
                "checked_structural_propositions": i % 3,
                "unknown_claims": 1,
                "scoped_contradictions": 0,
                "denominator": 2,
                "ambiguity": 0,
                "source_sha256": f"source-{i}",
                "excluded": False,
                "censored": i % 3 == 0,
            },
        }
        for i in range(80)
    ]
    from carnot.reporting.experiment_7660_atom_energy import _head

    heads = {"selected": "atom", "heads": {"atom": _head([0.0] * 4)}}
    continuous = replay_arm(tmp_path, inputs, heads, "source", restart=False)
    resumed = replay_arm(tmp_path, inputs, heads, "source_restart", restart=True)
    assert [r["probability"] for r in continuous["rows"]] == [
        r["probability"] for r in resumed["rows"]
    ]
    assert continuous["numerical_hash"] == resumed["numerical_hash"]
    omitted = replay_arm(tmp_path, inputs, heads, "omission", restart=True)
    assert len(omitted["rows"]) == 80
    assert sum(r["feedback_status"] == "missing" for r in omitted["rows"]) == 20
    permuted = replay_arm(tmp_path, inputs, heads, "permuted", restart=False)
    assert permuted["rows"][0]["released_label"] is None
    assert all(row["released_label"] in (None, 0, 1) for row in permuted["rows"])


def test_req_report_7663_rejects_invalid_admission_and_roster(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7663-ADMISSION: corrupt or reused custody fails closed."""
    empty = CostGuardedService(tmp_path / "empty.json", arm="source")
    with pytest.raises(ValueError, match="proposal_missing"):
        empty.admit(["x"] * 5)
    service = _prepared_service(tmp_path / "prepared", [0] * 5)
    with pytest.raises(ValueError, match="five_distinct_admissions_required"):
        service.admit(["5"] * 5)
    service.state["proposal"]["prior_hash"] = "stale"
    with pytest.raises(ValueError, match="stale_proposal"):
        service.admit([str(i) for i in range(5, 10)])
    service.state["proposal"]["prior_hash"] = service.numerical_hash
    with pytest.raises(ValueError, match="admission_not_released"):
        service.admit([str(i) for i in range(4, 9)])
    with pytest.raises(ValueError, match="online_roster_invalid"):
        replay_arm(tmp_path, [], {}, "source", restart=False)
