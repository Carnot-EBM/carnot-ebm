"""Tests for REQ-CL-7509 and its causal online scenarios."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7509_v657_causal_online as exp


def _public_rows(count: int = 24) -> list[dict[str, object]]:
    """Build label-free rows with three drift strata for protocol tests."""

    return [
        {
            "group_id": f"group-{index:03d}",
            "source_family": ("QA", "Summary", "Data2txt")[index % 3],
            "features": [
                float(index % 5) / 4.0,
                float((index + 1) % 5) / 4.0,
                float((index + 2) % 5) / 4.0,
                float((index + 3) % 5) / 4.0,
            ],
            "base_probability": 0.2 + 0.02 * (index % 20),
            "role": "online",
        }
        for index in range(count)
    ]


def _labels(rows: list[dict[str, object]]) -> dict[str, int]:
    """Return mixed labels without adding them to public predictor rows."""

    return {str(row["group_id"]): index % 2 for index, row in enumerate(rows)}


# REQ-CL-7509; SCENARIO-CL-7509-PRIMARY.
def test_protocol_freezes_registered_primary_and_seven_arms() -> None:
    value = exp.protocol()
    assert value["schedule_seeds"] == [656201, 656202, 656203, 656204, 656205]
    assert value["delays"] == [8, 0]
    assert value["block_size"] == 8
    assert value["audit_probability"] == 0.25
    assert value["bootstrap"] == {
        "replicates": 2000,
        "block_length": 16,
        "seed": 657009,
        "sensitivity_block_lengths": [8, 32],
    }
    assert value["arms"] == list(exp.ARMS)
    assert value["primary_contrasts"] == list(exp.PRIMARY_COMPARATORS)
    assert value["static_benefit_gate"] is False


# REQ-CL-7509; SCENARIO-CL-7509-CAUSAL.
def test_schedule_and_audit_use_only_public_identity_fields() -> None:
    rows = _public_rows()
    labels = _labels(rows)
    ordered = exp.build_arrival_order(rows, 656201)
    mask = exp.build_audit_mask(ordered, 656201)
    changed = deepcopy(rows)
    for row in changed:
        row["secret_label"] = 1 - labels[str(row["group_id"])]
    assert [row["group_id"] for row in ordered] == [
        row["group_id"] for row in exp.build_arrival_order(changed, 656201)
    ]
    assert mask == exp.build_audit_mask(changed, 656201)
    assert len(ordered) == len({str(row["group_id"]) for row in ordered})
    family_runs = [str(row["source_family"]) for row in ordered]
    assert sum(left != right for left, right in zip(family_runs, family_runs[1:])) <= 2


# REQ-CL-7509; SCENARIO-CL-7509-SHUFFLE.
def test_release_batch_shuffle_never_borrows_another_batch_label() -> None:
    mixed = exp.batch_permutation([0, 1, 1], ["a", "b", "c"], seed=7)
    assert mixed["mode"] in {"derangement", "best_effort_permutation"}
    assert set(mixed["label_origins"]) == {"a", "b", "c"}
    assert mixed["permutation_id"].startswith("sha256:")
    assert exp.batch_permutation([1], ["a"], seed=7)["mode"] == "singleton_noop"
    assert exp.batch_permutation([0, 0], ["a", "b"], seed=7)["mode"] == ("identical_labels_noop")
    with pytest.raises(ValueError, match="released_batch_invalid"):
        exp.batch_permutation([0, 1], ["same", "same"], seed=7)


# REQ-CL-7509; SCENARIO-CL-7509-CAUSAL/SHUFFLE.
@pytest.mark.parametrize("delay", [8, 0])
def test_replay_records_before_release_and_censors_final_feedback(
    tmp_path: Path, delay: int
) -> None:
    rows = _public_rows(19)
    labels = _labels(rows)
    training = [dict(row, role="training") for row in _public_rows(16)]
    result = exp.run_schedule(
        training,
        rows,
        labels,
        retention_rows=[],
        retention_labels={},
        schedule_seed=656201,
        delay=delay,
        checkpoint_dir=tmp_path,
    )
    assert len(result["per_source_results"]) == len(rows) * len(exp.ARMS)
    assert all(row["prediction_time"] <= row["update_time"] for row in result["per_update_rows"])
    assert all(row["availability_time"] == row["update_time"] for row in result["per_update_rows"])
    assert all(row["label_origin"] in row["release_event_ids"] for row in result["per_update_rows"])
    assert all("label" not in row["prediction_payload"] for row in result["per_source_results"])
    assert result["chronology_violations"] == 0
    assert result["censored_feedback_count"] >= 0
    assert result["reveal_counts_by_batch"][-1]["disposition"] == "censored"
    assert result["checkpoint_hashes"]


# REQ-CL-7509; SCENARIO-CL-7509-CAUSAL.
def test_future_label_change_cannot_change_earlier_predictions_or_updates(tmp_path: Path) -> None:
    rows = _public_rows(24)
    training = [dict(row, role="training") for row in _public_rows(16)]
    labels = _labels(rows)
    changed = dict(labels)
    ordered = exp.build_arrival_order(rows, 656202)
    for row in ordered[16:]:
        identity = str(row["group_id"])
        changed[identity] = 1 - changed[identity]
    first = exp.run_schedule(
        training,
        rows,
        labels,
        retention_rows=[],
        retention_labels={},
        schedule_seed=656202,
        delay=8,
        checkpoint_dir=tmp_path / "first",
        stop_after_arrival=15,
    )
    second = exp.run_schedule(
        training,
        rows,
        changed,
        retention_rows=[],
        retention_labels={},
        schedule_seed=656202,
        delay=8,
        checkpoint_dir=tmp_path / "second",
        stop_after_arrival=15,
    )
    assert first["per_source_results"] == second["per_source_results"]
    assert first["per_update_rows"] == second["per_update_rows"]
    assert first["final_state_hashes"] == second["final_state_hashes"]


# REQ-CL-7509; SCENARIO-CL-7509-RETENTION.
def test_retention_scores_final_heads_without_updates_or_rollback(tmp_path: Path) -> None:
    online = _public_rows(24)
    retention = [dict(row, role="test") for row in _public_rows(7)]
    training = [dict(row, role="training") for row in _public_rows(16)]
    result = exp.run_schedule(
        training,
        online,
        _labels(online),
        retention_rows=retention,
        retention_labels=_labels(retention),
        schedule_seed=656203,
        delay=8,
        checkpoint_dir=tmp_path,
    )
    assert len(result["retention_rows"]) == len(retention) * len(exp.ARMS)
    assert all(row["used_for_update"] is False for row in result["retention_rows"])
    assert all(row["used_for_selection_or_rollback"] is False for row in result["retention_rows"])
    assert result["retention_state_hash_before"] == result["retention_state_hash_after"]


# REQ-CL-7509; SCENARIO-CL-7509-RESTART.
def test_restart_matches_every_prediction_update_pending_queue_and_state(tmp_path: Path) -> None:
    rows = _public_rows(32)
    training = [dict(row, role="training") for row in _public_rows(16)]
    labels = _labels(rows)
    uninterrupted = exp.run_schedule(
        training,
        rows,
        labels,
        retention_rows=[],
        retention_labels={},
        schedule_seed=656204,
        delay=8,
        checkpoint_dir=tmp_path / "uninterrupted",
    )
    restarted = exp.run_schedule(
        training,
        rows,
        labels,
        retention_rows=[],
        retention_labels={},
        schedule_seed=656204,
        delay=8,
        checkpoint_dir=tmp_path / "restarted",
        restart_after_release_block=0,
    )
    assert exp.restart_parity(uninterrupted, restarted)["passed"] is True
    assert restarted["restart_performed"] is True


# REQ-CL-7509; SCENARIO-CL-7509-SUPPORT.
def test_weak_support_closes_benefit_but_not_complete_measurement() -> None:
    value = exp.build_test_artifact(support_override=False)
    reduced = exp.independent_reduce(value, require_validation=False)
    assert reduced["causal_evaluation_complete_score"] == 1
    assert reduced["causal_information_value_score"] == 0
    assert reduced["online_benefit_score"] == 0
    assert reduced["verdict_class"] == "null"


# REQ-CL-7509; SCENARIO-CL-7509-PRIMARY.
def test_seed_averaging_precedes_moving_block_bootstrap() -> None:
    rows: list[dict[str, object]] = []
    for source in range(20):
        for seed in exp.SCHEDULE_SEEDS:
            for arm, loss in (("local_brier", 0.10), ("frozen_base", 0.14)):
                rows.append(
                    {
                        "group_id": f"g-{source:02d}",
                        "source_family": "QA" if source < 10 else "Summary",
                        "schedule_seed": seed,
                        "delay": 8,
                        "arm": arm,
                        "brier_loss": loss + seed % 2 * 0.001,
                    }
                )
    contrast = exp.reduce_contrast(
        rows,
        comparator="frozen_base",
        delay=8,
        block_length=16,
        replicates=200,
        seed=657009,
    )
    assert contrast["source_count"] == 20
    assert contrast["replicate_unit"] == "schedule_seed_mean_within_source"
    assert contrast["mean_delta"] == pytest.approx(-0.04)
    assert contrast["upper95_delta"] < 0.0


# REQ-CL-7509; SCENARIO-CL-7509-CAUSAL/RETENTION.
def test_real_role_inputs_are_exact_and_temperature_is_frozen() -> None:
    public = exp.load_public_role_rows(exp.REPO_ROOT)
    assert {key: len(value) for key, value in public.items()} == {
        "training": 176,
        "calibration_tuning": 60,
        "online": 159,
        "test": 116,
    }
    assert all("label" not in row for rows in public.values() for row in rows)
    assert all(len(row["features"]) == 4 for rows in public.values() for row in rows)
    labels = exp.load_private_labels(exp.REPO_ROOT, roles=("online", "test"))
    assert set(labels) == {"online", "test"}
    assert len(labels["online"]) == 159
    assert len(labels["test"]) == 116
    assert exp.load_frozen_settings(exp.REPO_ROOT)["temperature"] == 1.5


# REQ-CL-7509; SCENARIO-CL-7509-ARTIFACT.
def test_preconditions_and_blocked_artifact_name_exact_failure(tmp_path: Path) -> None:
    checks, hashes = exp.collect_preconditions(exp.REPO_ROOT)
    assert checks and all(row["passed"] is True for row in checks)
    assert hashes
    missing_checks, _ = exp.collect_preconditions(tmp_path)
    failed = next(row for row in missing_checks if row["passed"] is False)
    blocked = exp.build_blocked_artifact(failed, missing_checks)
    assert blocked["honest_verdict"].startswith("complete_blocked_")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["rows"] == []
    assert blocked["gate_check_summary"]["first_failure"] == failed


# REQ-CL-7509; SCENARIO-CL-7509-ARTIFACT.
def test_artifact_readers_fail_closed_on_score_and_row_mutations(tmp_path: Path) -> None:
    artifact = exp.build_test_artifact(support_override=False)
    assert exp.validate_artifact(artifact, verify_sources=False, require_validation=False) == []
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["causal_evaluation_complete_score"] == 1
    assert set(artifact) == set(artifact["field_principles"])

    for field, replacement in (
        ("schema", "wrong"),
        ("run_date", "19000101"),
        ("model_invoked", True),
        ("invocation_counts", {}),
        ("causal_evaluation_complete_score", 0),
        ("causal_information_value_score", 1),
        ("online_benefit_score", 1),
        ("restart_parity_score", 0),
        ("verdict_class", "positive"),
        ("reproducibility_checksum", "sha256:wrong"),
    ):
        changed = deepcopy(artifact)
        changed[field] = replacement
        assert exp.validate_artifact(changed, verify_sources=False, require_validation=False)

    changed = deepcopy(artifact)
    changed["per_source_results"][0]["brier_loss"] += 0.1
    assert exp.validate_artifact(changed, verify_sources=False, require_validation=False)

    candidate = tmp_path / "candidate.json"
    exp.atomic_json(candidate, artifact)
    assert exp.cold_replay(candidate, verify_sources=False, require_validation=False) == []
    assert exp.cold_replay(tmp_path / "missing.json") == ["artifact_unreadable_or_not_object"]


# REQ-CL-7509; SCENARIO-CL-7509-ARTIFACT.
def test_cli_reader_modes_and_date_guard(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = exp.build_test_artifact(support_override=False)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact))
    assert (
        exp.main(["--date", exp.RUN_DATE, "--cold-replay", str(candidate), "--no-source-check"])
        == 0
    )
    assert '"errors": []' in capsys.readouterr().out
    assert (
        exp.main(
            ["--date", exp.RUN_DATE, "--independent-reduce", str(candidate), "--no-source-check"]
        )
        == 0
    )
    assert '"causal_evaluation_complete_score": 1' in capsys.readouterr().out
    with pytest.raises(SystemExit, match=f"--date must be {exp.RUN_DATE}"):
        exp.main(["--date", "19000101"])
    monkeypatch.setattr(exp, "run_experiment", lambda *_args, **_kwargs: artifact)
    assert exp.main(["--date", exp.RUN_DATE]) == 0
    assert '"result"' in capsys.readouterr().out


# REQ-CL-7509; SCENARIO-CL-7509-ARTIFACT.
def test_defensive_inputs_and_reducer_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _public_rows(8)
    duplicate = deepcopy(rows)
    duplicate[1]["group_id"] = duplicate[0]["group_id"]
    with pytest.raises(ValueError, match="public_identity_invalid"):
        exp.build_arrival_order(duplicate, 1)
    with pytest.raises(ValueError, match="private_label_roles_invalid"):
        exp.load_private_labels(exp.REPO_ROOT, roles=("training",))
    with pytest.raises(ValueError, match="delay_invalid"):
        exp.run_schedule(
            rows,
            rows,
            _labels(rows),
            retention_rows=[],
            retention_labels={},
            schedule_seed=1,
            delay=1,
            checkpoint_dir=tmp_path,
        )
    with pytest.raises(ValueError, match="online_label_identity_mismatch"):
        exp.run_schedule(
            rows,
            rows,
            {},
            retention_rows=[],
            retention_labels={},
            schedule_seed=1,
            delay=8,
            checkpoint_dir=tmp_path,
        )
    with pytest.raises(ValueError, match="retention_label_identity_mismatch"):
        exp.run_schedule(
            rows,
            rows,
            _labels(rows),
            retention_rows=[dict(rows[0], role="test")],
            retention_labels={},
            schedule_seed=1,
            delay=8,
            checkpoint_dir=tmp_path,
        )
    with pytest.raises(ValueError, match="checkpoint_schema_invalid"):
        exp._restore_checkpoint({})
    assert (
        exp.reduce_contrast(
            [], comparator="frozen_base", delay=8, block_length=16, replicates=2, seed=1
        )["source_count"]
        == 0
    )
    assert exp.independent_reduce({}, require_validation=False)["errors"] == ["raw_rows_missing"]
    names = ["one"]
    assert exp._receipts_pass([{"name": "one", "exit_code": 0, "timed_out": False}], names)
    assert not exp._receipts_pass([{"name": "one", "exit_code": 1}], names)

    monkeypatch.setattr(exp, "_load_object", lambda _path: {})
    with pytest.raises(ValueError, match="frozen_temperature_invalid"):
        exp.load_frozen_settings(tmp_path)


# REQ-CL-7509; SCENARIO-CL-7509-ARTIFACT.
def test_validator_names_each_malformed_terminal_surface() -> None:
    artifact = exp.build_test_artifact()
    mutations = (
        lambda value: value.update({"verdict_class": "wrong"}),
        lambda value: value.update({"honest_verdict": "unfinished"}),
        lambda value: value.update({"preconditions_checked": [{"passed": False}]}),
        lambda value: value.update({"acceptance_gate_results": [{}]}),
        lambda value: value.update({"field_principles": {}}),
        lambda value: value.update({"source_artifact_hashes": "bad"}),
        lambda value: value.update({"source_artifact_hashes": ["bad"]}),
        lambda value: value.update(
            {"source_artifact_hashes": [{"path": "missing", "sha256": "sha256:bad"}]}
        ),
    )
    for mutate in mutations:
        changed = deepcopy(artifact)
        mutate(changed)
        assert exp.validate_artifact(changed, verify_sources=True, require_validation=False)


# REQ-CL-7509; SCENARIO-CL-7509-ARTIFACT.
def test_role_reader_and_release_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exp, "load_jsonl", lambda _path: [])
    monkeypatch.setattr(exp, "_load_object", lambda _path: {"mean": [], "safe_scale": []})
    with pytest.raises(ValueError, match="normalization_invalid"):
        exp.load_public_role_rows(tmp_path)

    monkeypatch.setattr(
        exp,
        "_load_object",
        lambda _path: {"mean": [0.0] * 10, "safe_scale": [1.0] * 10},
    )
    monkeypatch.setattr(
        exp,
        "load_frozen_settings",
        lambda _root: {"temperature": 1.5, "learning_rate": 0.01, "residual_bound": 1.0},
    )
    with pytest.raises(ValueError, match="role_counts_invalid"):
        exp.load_public_role_rows(tmp_path)

    calls = iter(
        [
            [{"group_id": "bad", "role": "unknown", "features": [0.0] * 10}],
            [{"group_id": "bad", "source_family": "QA"}],
        ]
    )
    monkeypatch.setattr(exp, "load_jsonl", lambda _path: next(calls))
    with pytest.raises(ValueError, match="public_role_row_invalid"):
        exp.load_public_role_rows(tmp_path)

    public = {
        "training": [],
        "calibration_tuning": [],
        "online": [{"group_id": "a"}],
        "test": [],
    }
    monkeypatch.setattr(exp, "load_public_role_rows", lambda _root: public)
    monkeypatch.setattr(
        exp, "load_jsonl", lambda _path: [{"group_id": "a", "role": "online", "label": 2}]
    )
    with pytest.raises(ValueError, match="private_label_invalid"):
        exp.load_private_labels(tmp_path, roles=("online",))
    monkeypatch.setattr(exp, "load_jsonl", lambda _path: [])
    with pytest.raises(ValueError, match="private_label_identity_mismatch"):
        exp.load_private_labels(tmp_path, roles=("online",))

    training = [dict(row, role="training") for row in _public_rows(8)]
    heads = exp._create_heads(training, {"learning_rate": 0.01, "residual_bound": 1.0})
    with pytest.raises(ValueError, match="release_before_availability"):
        exp._release_batch(
            block_id=0,
            event_ids=["a"],
            labels={"a": 0},
            prediction_by_group={},
            heads=heads,
            schedule_seed=1,
            delay=8,
            update_time=0,
        )


# REQ-CL-7509; SCENARIO-CL-7509-SUPPORT/RETENTION/RESTART.
def test_independent_reducer_rejects_each_validity_failure() -> None:
    base = exp.build_test_artifact()
    mutations = (
        lambda value: value["rows"][0].update({"source_count": 1}),
        lambda value: value["per_source_results"].pop(),
        lambda value: value["retention_rows"].pop(),
        lambda value: value["rows"][0].update({"chronology_violations": 1}),
        lambda value: value["per_update_rows"][0].update({"update_time": -1}),
        lambda value: value["restart_parity_rows"][0].update({"passed": False}),
        lambda value: value["retention_rows"][0].update({"used_for_update": True}),
        lambda value: value["retention_rows"][0].update({"used_for_selection_or_rollback": True}),
        lambda value: value.update({"model_invoked": True}),
    )
    for mutate in mutations:
        changed = deepcopy(base)
        mutate(changed)
        reduced = exp.independent_reduce(changed, require_validation=False)
        assert reduced["causal_evaluation_complete_score"] == 0
        assert reduced["errors"]
    assert (
        exp.independent_reduce(base, require_validation=True)["causal_evaluation_complete_score"]
        == 0
    )
