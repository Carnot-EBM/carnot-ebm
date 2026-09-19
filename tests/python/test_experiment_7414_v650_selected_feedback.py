"""Tests for REQ-AUTO-7414 fixed selected-feedback source adaptation."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7414_v650_selected_feedback as exp


ROOT = Path(__file__).resolve().parents[2]


def _feature(index: int) -> dict[str, float]:
    return {
        "numeric_novelty_with_context": float(index % 2),
        "falsifiability_score": float((index // 2) % 2),
        "normalized_number_token_overlap": (index % 3) / 2,
        "normalized_content_token_overlap": (index % 4) / 3,
        "max_answer_source_sentence_overlap": (index % 5) / 4,
        "missing_or_empty_source": 0.0,
    }


def _row(partition: str, index: int, label: int | None = None) -> dict[str, Any]:
    return {
        "row_key": f"row-{partition}-{index:03d}",
        "group_id": f"group-{partition}-{index:03d}",
        "partition": partition,
        "source_features": _feature(index),
        "response_only_ablation": {
            "entity_uptake": float(index % 2),
            "falsifiability_score": float((index // 2) % 2),
        },
        "label": label,
        "label_authority": "machine_annotation",
    }


def _fixture_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for partition in ("train", "probability_calibration", "policy_calibration"):
        rows.extend(_row(partition, index, index % 2) for index in range(12))
    rows.extend(_row("online_stream", index, index % 2) for index in range(20))
    return rows


def _passing_receipts() -> list[dict[str, Any]]:
    names = (*exp.AFFECTED_CHECK_NAMES, *exp.TERMINAL_CHECK_NAMES)
    return [
        {
            "name": name,
            "required": True,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
        }
        for name in names
    ]


def test_req_auto_7414_spec_precedes_implementation() -> None:
    """REQ-AUTO-7414: the implementation has a complete driving contract."""

    text = (ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "### REQ-AUTO-7414:" in text
    for number in range(1, 8):
        assert f"SCENARIO-AUTO-7414-{number:02d}" in text


def test_scenario_7414_01_preconditions_and_registered_partitions() -> None:
    """SCENARIO-AUTO-7414-01: exact inputs pass and final-test access stays absent."""

    checks, hashes, loaded = exp.collect_preconditions(ROOT)
    assert checks
    assert all(row["passed"] for row in checks)
    assert exp.UPSTREAM_PATH.as_posix() in hashes
    assert loaded["upstream"]["source_feature_protocol_ready_score"] == 1

    rows = _fixture_rows()
    states = exp.initialize_seed_states(rows, seeds=(65001,), steps=4)
    assert len(states) == 1
    receipt = states[0]["training_receipt"]
    assert receipt["fit_partition"] == "train"
    assert receipt["calibration_partition"] == "probability_calibration"
    assert receipt["policy_partition"] == "policy_calibration"
    assert receipt["online_labels_used_for_initialization"] == 0
    with pytest.raises(ValueError, match="registered development partitions"):
        exp.initialize_seed_states([*rows, _row("final_test", 99, 1)], seeds=(65001,), steps=1)


def test_scenario_7414_02_adapter_prediction_update_and_authority() -> None:
    """SCENARIO-AUTO-7414-02: prediction precedes one bounded update."""

    adapter = exp.SourceAffineAdapter(exp.fixture_weights(), a=1.0, b=0.0)
    prediction = adapter.record_prediction("event-1", 0.4, prediction_index=0, available_at=1)
    before = adapter.state_hash
    assert prediction["prediction_before_feedback"] is True
    with pytest.raises(exp.FutureLabelAccessError):
        adapter.commit_feedback("event-1", 1, visible_at=0)
    committed = adapter.commit_feedback("event-1", 1, visible_at=1)
    assert committed["update_admitted"] is True
    assert committed["state_hash_before"] == before
    assert 0.25 <= adapter.a <= 4.0
    assert -8.0 <= adapter.b <= 8.0
    assert committed["gradient_norm_after_clip"] <= 1.0
    changed = adapter.state_hash
    duplicate = adapter.commit_feedback("event-1", 1, visible_at=2)
    assert duplicate["status"] == "duplicate"
    assert adapter.state_hash == changed
    assert adapter.commit_feedback("missing", 1, visible_at=2)["status"] == "unknown_event"


def test_scenario_7414_03_fixed_masks_and_constructed_orders() -> None:
    """SCENARIO-AUTO-7414-03: registered masks are shared and label blind."""

    rows = [_row("online_stream", index, index % 2) for index in range(40)]
    streams = exp.build_streams(rows, block_length=8)
    assert len(streams["hash_order"]) == 40
    assert streams["hash_order"] != streams["reversed_block_order"]
    assert {row["group_id"] for row in streams["hash_order"]} == {
        row["group_id"] for row in streams["reversed_block_order"]
    }
    ids = [row["observation_id"] for row in streams["hash_order"]]
    mask = exp.primary_availability_mask(ids)
    changed_labels = [{**row, "label": 1 - int(row["label"])} for row in rows]
    changed_ids = [row["observation_id"] for row in exp.build_streams(changed_labels)["hash_order"]]
    assert exp.primary_availability_mask(changed_ids) == mask
    assert 20 <= sum(mask.values()) <= 36

    initial = exp.initialize_seed_states(_fixture_rows(), seeds=(65001,), steps=3)[0]
    replay = exp.replay_condition(
        initial,
        streams["hash_order"][:8],
        ordering="hash_order",
        feedback_regime="primary_label_blind_75",
        delay=1,
    )
    assert len(replay["feedback_event_rows"]) == 8 * len(exp.ARMS)
    for observation_id in {row["observation_id"] for row in replay["feedback_event_rows"]}:
        selected = [
            row for row in replay["feedback_event_rows"] if row["observation_id"] == observation_id
        ]
        assert len({row["registered_feedback_mask"] for row in selected}) == 1
    frozen = [row for row in replay["feedback_event_rows"] if row["arm"] == exp.FROZEN_ARM]
    no_feedback = [
        row for row in replay["feedback_event_rows"] if row["arm"] == exp.NO_FEEDBACK_ARM
    ]
    assert [row["probability"] for row in frozen] == [row["probability"] for row in no_feedback]
    assert replay["pending_feedback_at_end"] >= 0


def test_scenario_7414_04_support_and_value_fail_closed() -> None:
    """SCENARIO-AUTO-7414-04: low label support forces the registered null."""

    event_rows = exp.synthetic_metric_rows(groups=96, seeds=5)
    reports = exp.condition_reports(event_rows)
    intervals = exp.paired_moving_block_intervals(event_rows, draws=200)
    support = {"independent_online_groups": 96, "label_counts": {"0": 9, "1": 87}}
    reduced = exp.reduce_online_value(reports, intervals, support)
    assert reduced["support_passed"] is False
    assert reduced["passed"] is False
    assert reduced["terminal_verdict"] == "complete_null_insufficient_online_support"
    assert {row["block_length"] for row in intervals} == {16, 32}
    assert all(row["seed"] == exp.BOOTSTRAP_SEED for row in intervals)


def test_scenario_7414_05_revocation_restart_and_erasure() -> None:
    """SCENARIO-AUTO-7414-05: trusted-journal reconstruction is deterministic."""

    controls, revocations = exp.run_analytic_controls()
    assert controls
    assert all(row["passed"] for row in controls.values())
    assert {row["operation"] for row in revocations} == {"replace_label", "erase_update"}
    assert all(row["reconstruction_duration_s"] >= 0 for row in revocations)
    assert all(row["trusted_journal_replayed"] for row in revocations)

    adapter = exp.SourceAffineAdapter(exp.fixture_weights())
    adapter.record_prediction("a", 0.2, prediction_index=0, available_at=1)
    adapter.commit_feedback("a", 1, visible_at=1)
    restored = exp.SourceAffineAdapter.from_dict(adapter.to_dict())
    assert restored.to_dict() == adapter.to_dict()
    replacement = restored.revoke_feedback("a", replacement_label=0)
    first_hash = replacement["state_hash_after"]
    replayed = exp.SourceAffineAdapter.from_dict(restored.to_dict())
    assert replayed.state_hash == first_hash
    erased = replayed.revoke_feedback("a", replacement_label=None)
    assert erased["status"] == "erased"
    assert replayed.update_count == 0


def test_scenario_7414_06_no_feedback_copy_and_defensive_inputs() -> None:
    """SCENARIO-AUTO-7414-06: withheld feedback keeps the copied state frozen."""

    adapter = exp.SourceAffineAdapter(exp.fixture_weights())
    before = adapter.state_hash
    adapter.record_prediction("withheld", -0.3, prediction_index=2, available_at=3)
    assert adapter.state_hash == before
    assert adapter.commit_feedback("withheld", None, visible_at=3)["status"] == "missing"
    assert adapter.state_hash == before
    with pytest.raises(ValueError):
        adapter.record_prediction("nan", math.nan, prediction_index=0, available_at=1)
    with pytest.raises(ValueError):
        adapter.record_prediction("late", 0.0, prediction_index=1, available_at=1)
    with pytest.raises(ValueError):
        exp.SourceAffineAdapter(exp.fixture_weights(), a=0.1)
    with pytest.raises(ValueError):
        exp.SourceAffineAdapter.from_dict({**adapter.to_dict(), "state_hash": "changed"})


def test_scenario_7414_07_fixture_artifact_cold_reduction_and_mutations(tmp_path: Path) -> None:
    """SCENARIO-AUTO-7414-07: raw evidence drift fails cold validation."""

    artifact = exp.build_fixture_artifact(validation_receipts=_passing_receipts())
    assert exp.validate_artifact(artifact) == []
    reduced = exp.independent_reduce(artifact)
    assert reduced["online_capture_complete_score"] == 1
    assert reduced["online_value_score"] == 0
    assert artifact["honest_verdict"].startswith("complete_")

    changed = deepcopy(artifact)
    changed["feedback_event_rows"][0]["probability"] = 2.0
    assert "event_probability_invalid:0" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:changed"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(changed)

    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp.cold_replay(path) == []
    path.write_text("[]", encoding="utf-8")
    assert exp.cold_replay(path) == ["artifact_unreadable_or_not_object"]


def test_blocked_artifact_and_cli_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-AUTO-7414: blocked inputs and reader modes retain terminal semantics."""

    failed = {
        "check": "upstream_protocol_ready",
        "upstream": "exp7412-source-features",
        "path": exp.UPSTREAM_PATH.as_posix(),
        "field": "source_feature_protocol_ready_score",
        "operator": "==",
        "expected": 1,
        "observed": None,
        "passed": False,
    }
    blocked = exp.build_blocked_artifact(failed)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked_")
    assert blocked["online_capture_complete_score"] == 0
    assert blocked["online_value_score"] == 0
    assert exp.validate_artifact(blocked) == []

    args = exp.parse_args(["--date", exp.RUN_DATE, "--cold-replay", "candidate.json"])
    assert args.cold_replay == Path("candidate.json")
    monkeypatch.setattr(exp, "cold_replay", lambda _path: [])
    assert exp.main(["--date", exp.RUN_DATE, "--cold-replay", "candidate.json"]) == 0
    monkeypatch.setattr(exp, "cold_replay", lambda _path: ["bad"])
    assert exp.main(["--date", exp.RUN_DATE, "--cold-replay", "candidate.json"]) == 1


def test_numeric_and_reducer_defenses() -> None:
    """REQ-AUTO-7414: malformed rows and unsupported conditions fail closed."""

    rows = _fixture_rows()
    with pytest.raises(ValueError, match="both labels"):
        exp.initialize_seed_states(
            [{**row, "label": 1} for row in rows], seeds=(65001,), steps=1
        )
    with pytest.raises(ValueError, match="registered replay condition"):
        exp.replay_condition(
            exp.initialize_seed_states(rows, seeds=(65001,), steps=1)[0],
            exp.build_streams(rows)["hash_order"],
            ordering="bad",
            feedback_regime="primary_label_blind_75",
            delay=1,
        )
    with pytest.raises(ValueError, match="paired rows"):
        exp.paired_moving_block_intervals([], draws=10)
    with pytest.raises(ValueError, match="positive"):
        exp.paired_moving_block_intervals(exp.synthetic_metric_rows(20, 1), draws=0)

    report = exp.condition_reports(exp.synthetic_metric_rows(20, 2))
    assert report
    assert all(row["real_world_chronology"] is False for row in report)
    assert all(row["iid_guarantee_asserted"] is False for row in report)
    assert all(row["conformal_guarantee_asserted"] is False for row in report)
