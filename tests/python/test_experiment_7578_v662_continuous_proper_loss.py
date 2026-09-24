"""Tests for REQ-CL-7578 and SCENARIO-CL-7578-* causal measurement."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7578_v662_continuous_proper_loss as mod


def _rows(role: str, count: int, *, offset: int = 0) -> list[dict[str, Any]]:
    rows = []
    for index in range(count):
        value = index + offset
        rows.append(
            {
                "source_id": f"{role}-{value:03d}",
                "group_id": f"group-{role}-{value:03d}",
                "role": role,
                "official_split": "train" if role != "test" else "validation",
                "probability": 0.04 + 0.92 * ((value % 17) / 16),
                "label": int((value * 7 + value // 3) % 5 >= 2),
                "context_sha256": f"sha256:context-{role}-{value}",
                "response_sha256": f"sha256:response-{role}-{value}",
            }
        )
    return rows


def _roles() -> dict[str, list[dict[str, Any]]]:
    return {
        "fit": _rows("fit", 160),
        "tune": _rows("tune", 40, offset=200),
        "policy": _rows("policy", 40, offset=300),
        "online": _rows("online", 160, offset=400),
        "test": _rows("test", 80, offset=600),
    }


def _receipts() -> list[dict[str, Any]]:
    names = [*mod.validation_scope.REQUIRED_CHECK_NAMES, *mod.TERMINAL_CHECK_NAMES]
    return [
        {
            "name": name,
            "command": ["python", name],
            "cwd": "/tmp/worktree",
            "exit_code": 0,
            "timed_out": False,
            "log_sha256": f"sha256:{name}",
        }
        for name in names
    ]


@pytest.fixture(scope="module")
def built_artifact(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    root = tmp_path_factory.mktemp("terminal-artifact")
    return root, mod.build_test_artifact(root)


def test_metric_rows_preserve_raw_arithmetic_and_typed_action() -> None:
    """REQ-CL-7578: rows retain exact proper loss, direction, and action."""

    accept = mod.metric_row("unit-a", "bounded", 0.01, 0, seed=7, phase="online")
    reject = mod.metric_row("unit-b", "bounded", 0.99, 1, seed=7, phase="online")
    escalate = mod.metric_row("unit-c", "bounded", 0.50, 1, seed=7, phase="online")
    assert accept["raw_squared_error_numerator"] == pytest.approx(0.0001)
    assert accept["raw_squared_error_denominator"] == 1
    assert accept["typed_action"] == "accept"
    assert reject["typed_action"] == "reject"
    assert escalate["typed_action"] == "escalate"
    assert escalate["realized_action_cost"] == pytest.approx(0.2)
    assert accept["metric_direction"] == "lower_is_better"


def test_full_order_is_causal_restart_exact_and_retention_isolated(tmp_path: Path) -> None:
    """SCENARIO-CL-7578-CAUSAL/RETENTION/RESTART: run the fixed lifecycle."""

    roles = _roles()
    protocol = mod.freeze_protocol(roles)
    config = mod.count_config_from_rows(roles["fit"])
    result = mod.measure_order(
        roles["online"],
        roles["test"],
        protocol["orders"][str(mod.ORDER_SEEDS[0])],
        config,
        seed=mod.ORDER_SEEDS[0],
        state_dir=tmp_path,
    )
    assert len(result["event_rows"]) == 160
    assert len(result["comparison_rows"]) == 160 * len(mod.ARMS)
    assert len(result["retention_rows"]) == 4 * 80 * len(mod.ARMS)
    assert [row["event_count"] for row in result["retention_receipts"]] == [40, 80, 120, 160]
    assert len(result["release_rows"]) == 19
    assert len(result["ack_rows"]) == 19
    assert result["released_event_count"] == 152
    assert result["unreleased_tail_count"] == 8
    assert result["restart_mismatch_count"] == 0
    assert result["future_prediction_mismatch_count"] == 0
    assert result["retention_state_mutation_count"] == 0
    assert result["acknowledged_exactly_once"] is True
    assert result["initial_theta"] == pytest.approx(mod.IDENTITY_THETA)
    assert result["final_state_hash"] == result["uninterrupted_final_state_hash"]
    assert all(row["label_available_at_prediction"] is False for row in result["event_rows"])
    assert all("label" not in row for row in result["event_rows"])


def test_lifecycle_guards_and_crash_before_ack_recovery(tmp_path: Path) -> None:
    """SCENARIO-CL-7578-CAUSAL/RESTART: invalid releases fail without mutation."""

    roles = _roles()
    controls = mod.run_lifecycle_controls(
        roles["online"], mod.count_config_from_rows(roles["fit"]), tmp_path
    )
    assert controls == {
        "duplicate_release_rejected": True,
        "out_of_order_release_rejected": True,
        "future_label_sentinel_rejected": True,
        "crash_before_ack_recovered": True,
        "duplicate_ack_noop": True,
        "state_unchanged_after_rejections": True,
    }


def test_shuffle_changes_only_informative_released_blocks(tmp_path: Path) -> None:
    """SCENARIO-CL-7578-SHUFFLE: constant blocks cannot invent a contrast."""

    roles = _roles()
    for index in range(8):
        roles["online"][index]["label"] = 0
    protocol = mod.freeze_protocol(roles)
    order = protocol["orders"][str(mod.ORDER_SEEDS[0])]
    by_id = {row["source_id"]: row for row in roles["online"]}
    for source_id in order[:8]:
        by_id[source_id]["label"] = 0
    result = mod.measure_order(
        roles["online"],
        roles["test"],
        order,
        mod.count_config_from_rows(roles["fit"]),
        seed=mod.ORDER_SEEDS[0],
        state_dir=tmp_path,
    )
    first = result["release_rows"][0]
    assert first["shufflable"] is False
    assert first["shuffled_assignment_changed"] is False
    informative = [row for row in result["release_rows"] if row["shufflable"]]
    assert informative
    assert all(row["shuffled_assignment_changed"] for row in informative)
    assert all(row["marginal_labels_preserved"] for row in result["release_rows"])
    assert all(row["release_time_preserved"] for row in result["release_rows"])
    assert result["unshufflable_block_count"] >= 1


def test_small_bootstrap_retrains_each_draw_and_checkpoints(tmp_path: Path) -> None:
    """SCENARIO-CL-7578-UNCERTAINTY: each source draw starts from identity."""

    roles = _roles()
    protocol = mod.freeze_protocol(roles)
    checkpoint = tmp_path / "bootstrap.jsonl"
    result = mod.causal_bootstrap(
        roles["online"],
        roles["test"],
        protocol["orders"],
        mod.count_config_from_rows(roles["fit"]),
        replays_per_order=2,
        order_seeds=mod.ORDER_SEEDS[:2],
        checkpoint_path=checkpoint,
    )
    assert result["completed_replays"] == 4
    assert result["expected_replays"] == 4
    assert result["each_resample_retrained"] is True
    assert result["source_component_grouping_preserved"] is True
    assert result["resampled_adapted_loss_rows"] is False
    assert len(result["rows"]) == 4 * len(mod.ARMS)
    assert len(checkpoint.read_text(encoding="utf-8").splitlines()) == 4
    assert all(row["raw_squared_error_denominator"] == 160 for row in result["rows"])
    assert all(row["retention_denominator"] == 80 for row in result["rows"])
    assert {row["initial_state_hash"] for row in result["rows"] if row["arm"] == "bounded"} == {
        mod.identity_state_hash()
    }


def test_bootstrap_reducer_separates_measurement_benefit_and_retention(tmp_path: Path) -> None:
    """REQ-CL-7578: valid uncertainty can reduce to a null without becoming partial."""

    roles = _roles()
    protocol = mod.freeze_protocol(roles)
    bootstrap = mod.causal_bootstrap(
        roles["online"],
        roles["test"],
        protocol["orders"],
        mod.count_config_from_rows(roles["fit"]),
        replays_per_order=2,
        order_seeds=mod.ORDER_SEEDS[:2],
        checkpoint_path=tmp_path / "bootstrap.jsonl",
    )
    reduced = mod.reduce_bootstrap(bootstrap["rows"], expected_replays=4, expected_order_count=2)
    assert reduced["measurement_complete"] is True
    assert reduced["benefit_passed"] in (True, False)
    assert reduced["retention_passed"] in (True, False)
    assert set(reduced["contrasts"]) == {
        "brier_vs_raw",
        "brier_vs_global_count",
        "brier_vs_local_count",
        "brier_vs_shuffled_feedback",
        "action_cost_vs_raw",
    }
    mutated = deepcopy(bootstrap["rows"])
    mutated[0]["raw_squared_error_numerator"] += 1.0
    with pytest.raises(ValueError, match="mean_brier_mismatch"):
        mod.reduce_bootstrap(mutated, expected_replays=4, expected_order_count=2)


def test_retention_evaluator_returns_no_labels_to_learner() -> None:
    """SCENARIO-CL-7578-RETENTION: scoring cannot change sufficient statistics."""

    roles = _roles()
    machine = mod.RecalibrationEventMachine.create(mod.count_config_from_rows(roles["fit"]))
    before = machine.state_hash()
    result = mod.evaluate_retention(machine, roles["test"], seed=1, event_count=40)
    assert result["learner_labels_returned"] == 0
    assert result["learner_state_hash_before"] == before
    assert result["learner_state_hash_after"] == before
    assert len(result["rows"]) == 80 * len(mod.ARMS)


def test_sidecars_artifact_and_fresh_readers_are_hash_bound(
    built_artifact: tuple[Path, dict[str, Any]],
) -> None:
    """SCENARIO-CL-7578-TERMINAL: the artifact reduces from exact sidecar bytes."""

    tmp_path, artifact = built_artifact
    reduction = mod.validate_artifact(
        artifact, root=tmp_path, expected_replays=4, require_terminal=True
    )
    assert reduction["measurement_complete"] is True
    assert artifact["learning_measurement_complete_score"] == 1
    assert artifact["fresh_confirmatory_claim_allowed"] is False
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_specs"] == []
    assert artifact["no_model_load"] is True
    assert artifact["invocation_counts"] == mod.ZERO_INVOCATION_COUNTS
    assert artifact["verdict_class"] in {"positive", "null"}
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["causal_event_rows_path"] == artifact["raw_sidecars"]["causal_rows"]["path"]
    assert artifact["state_hashes"]["restart_equality"] is True
    assert set(mod.REQUIRED_PRINCIPLE_FIELDS) <= set(artifact["field_principles"])

    path = tmp_path / "artifact.json"
    mod.atomic_json(path, artifact)
    assert mod.cold_replay(path, root=tmp_path, expected_replays=4)["measurement_complete"]
    assert mod.independent_reduce(path, root=tmp_path, expected_replays=4)["measurement_complete"]

    changed = deepcopy(artifact)
    changed["learning_measurement_complete_score"] = 0
    with pytest.raises(ValueError, match="measurement_score_mismatch"):
        mod.validate_artifact(changed, root=tmp_path, expected_replays=4)
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:changed"
    with pytest.raises(ValueError, match="reproducibility_checksum_mismatch"):
        mod.validate_artifact(changed, root=tmp_path, expected_replays=4)


def test_sidecar_and_row_mutations_fail_closed(
    built_artifact: tuple[Path, dict[str, Any]],
) -> None:
    """SCENARIO-CL-7578-TERMINAL: missing or changed raw evidence is rejected."""

    tmp_path, artifact = built_artifact
    receipt = artifact["raw_sidecars"]["bootstrap_rows"]
    path = tmp_path / receipt["path"]
    original = path.read_bytes()
    path.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="sidecar_hash_mismatch:bootstrap_rows"):
        mod.validate_artifact(artifact, root=tmp_path, expected_replays=4)
    path.write_bytes(original)

    changed = deepcopy(artifact)
    changed["rows"][0]["raw_squared_error_denominator"] = 0
    changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
    with pytest.raises(ValueError, match="row_denominator_invalid"):
        mod.validate_artifact(changed, root=tmp_path, expected_replays=4)

    for key in ("bytes", "rows"):
        changed_receipt = deepcopy(receipt)
        changed_receipt[key] += 1
        with pytest.raises(
            ValueError, match=f"sidecar_{'size' if key == 'bytes' else 'row_count'}"
        ):
            mod._receipt_rows(tmp_path, "bootstrap_rows", changed_receipt)


def test_terminal_artifact_mutations_cover_every_public_guard(
    built_artifact: tuple[Path, dict[str, Any]],
) -> None:
    """REQ-CL-7578: each required terminal claim fails closed independently."""

    tmp_path, artifact = built_artifact
    mutations = (
        ("schema", "wrong", "artifact_identity_mismatch"),
        ("honest_verdict", "unfinished", "terminal_prefix_missing"),
        ("MODEL_SPECS", ["forbidden"], "current_model_roster_nonempty"),
        ("no_model_load", False, "no_model_load_contract_invalid"),
        ("deployment_promotion_available", True, "deployment_promotion_forbidden"),
        ("field_principles", {}, "field_principles_incomplete"),
        ("source_artifact_hashes", [], "source_artifact_hashes_missing"),
        ("order_seeds", [], "order_seeds_missing"),
        ("rows", {}, "rows_missing"),
        ("independent_reduction", {}, "independent_reduction_mismatch"),
        ("validation_receipts", {}, "validation_receipts_missing"),
        (
            "exploratory_learning_benefit_score",
            1 - artifact["exploratory_learning_benefit_score"],
            "benefit_score_mismatch",
        ),
        ("retention_pass_score", 1 - artifact["retention_pass_score"], "retention_score_mismatch"),
        ("raw_sidecars", None, "raw_sidecars_missing"),
        ("causal_event_rows_path", "wrong.jsonl", "causal_event_rows_path_mismatch"),
    )
    for field, value, error in mutations:
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
        with pytest.raises(ValueError, match=error):
            mod.validate_artifact(changed, root=tmp_path, expected_replays=4)

    changed = deepcopy(artifact)
    changed["raw_sidecars"].pop("retention_rows")
    changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
    with pytest.raises(ValueError, match="sidecar_roster_mismatch"):
        mod.validate_artifact(changed, root=tmp_path, expected_replays=4)

    changed = deepcopy(artifact)
    changed["rows"][0]["provenance"] = "changed"
    changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
    with pytest.raises(ValueError, match="bootstrap_rows_terminal_mismatch"):
        mod.validate_artifact(changed, root=tmp_path, expected_replays=4)

    changed = deepcopy(artifact)
    changed["raw_sidecars"]["learned_state"]["sha256"] = "sha256:changed"
    changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
    with pytest.raises(ValueError, match="sidecar_hash_mismatch:learned_state"):
        mod.validate_artifact(changed, root=tmp_path, expected_replays=4)

    changed = deepcopy(artifact)
    changed["validation_receipts"] = [
        row for row in changed["validation_receipts"] if row["name"] not in mod.TERMINAL_CHECK_NAMES
    ]
    changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
    with pytest.raises(ValueError, match="terminal_validation_incomplete"):
        mod.validate_artifact(changed, root=tmp_path, expected_replays=4, require_terminal=True)


def test_blocked_artifact_names_exact_external_operand() -> None:
    """REQ-CL-7578: absent upstream evidence is terminal blocked, never partial."""

    failed = mod.precondition_row(
        "exp7574_independent_qualification",
        "Experiment 7574",
        "results/missing.json",
        "recalibration_ready_score",
        1,
        None,
        False,
    )
    artifact = mod.build_blocked_artifact(failed, [failed], duration_s=0.1)
    assert artifact["honest_verdict"] == "complete_blocked_exp7574_independent_qualification"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["learning_measurement_complete_score"] == 0
    assert artifact["exploratory_learning_benefit_score"] == 0
    assert artifact["retention_pass_score"] == 0
    assert artifact["gate_check_summary"] == {
        "check": "exp7574_independent_qualification",
        "upstream": "Experiment 7574",
        "path": "results/missing.json",
        "field": "recalibration_ready_score",
        "op": "eq",
        "expected": 1,
        "observed": None,
        "passed": False,
    }
    assert mod.validate_artifact(artifact)["measurement_complete"] is False
    changed = deepcopy(artifact)
    changed["gate_check_summary"] = {}
    changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
    with pytest.raises(ValueError, match="blocked_gate_summary_invalid"):
        mod.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:changed"
    with pytest.raises(ValueError, match="reproducibility_checksum_mismatch"):
        mod.validate_artifact(changed)


def test_build_artifact_closes_positive_and_disqualified_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7578: validity and benefit select distinct terminal classes."""

    replay = {
        "seed": 1,
        "released_event_count": 152,
        "unreleased_tail_count": 8,
        "unshufflable_block_count": 0,
        "restart_mismatch_count": 0,
        "future_prediction_mismatch_count": 0,
        "retention_state_mutation_count": 0,
        "acknowledged_exactly_once": True,
        "comparison_rows": [],
        "retention_rows": [],
        "release_rows": [],
        "ack_rows": [],
        "initial_state_hash": "initial",
        "final_state_hash": "final",
        "final_numerical_state_hash": "numerical",
    }
    reduction = {
        "measurement_complete": True,
        "benefit_passed": True,
        "retention_passed": True,
    }
    monkeypatch.setattr(mod, "reduce_bootstrap", lambda *args, **kwargs: reduction)
    common = {
        "root": tmp_path,
        "sidecars": {"causal_rows": {"path": "causal.jsonl"}},
        "bootstrap_rows": [],
        "replays": [replay],
        "lifecycle_controls": {"lifecycle": True},
        "preconditions_checked": [{"required": True, "passed": True}],
        "source_artifact_hashes": [{"path": "input", "sha256": "hash"}],
        "duration_s": 0.1,
        "phase_spans": [],
        "expected_replays": 1,
        "expected_order_count": 1,
    }
    positive = mod.build_artifact(validation_receipts=_receipts(), **common)
    assert positive["verdict_class"] == "positive"
    disqualified = mod.build_artifact(validation_receipts=[], **common)
    assert disqualified["verdict_class"] == "disqualified"


def test_real_prerequisites_authenticate_exp7574_and_exp7575() -> None:
    """REQ-CL-7578: the changed prerequisite and protocol bytes are present."""

    checks, hashes = mod.collect_preconditions(mod.REPO_ROOT)
    assert checks
    assert all(row["passed"] for row in checks if row["required"])
    assert any(row["check"] == "exp7574_independent_qualification" for row in checks)
    assert any(row["check"] == "exp7575_protocol_ready" for row in checks)
    assert any(row["path"].endswith("cached_roles.jsonl") for row in hashes)
    roles, protocol = mod.load_inputs(mod.REPO_ROOT)
    assert {name: len(rows) for name, rows in roles.items()} == mod.ROLE_COUNTS
    assert protocol["order_seeds"] == list(mod.ORDER_SEEDS)
    assert protocol["bootstrap_replays_per_order"] == 1000


def test_cached_input_reader_rejects_each_custody_failure(tmp_path: Path) -> None:
    """REQ-CL-7578: cached roles and frozen protocol remain byte and field bound."""

    roles = _roles()
    flat = [row for name in mod.ROLE_COUNTS for row in roles[name]]
    protocol = mod.freeze_protocol(roles)
    role_path = tmp_path / "raw" / "roles.jsonl"
    protocol_path = tmp_path / "raw" / "protocol.json"
    artifact_path = tmp_path / mod.EXP7575_PATH

    def install(
        *,
        role_rows: list[dict[str, Any]] | None = None,
        protocol_value: Any = None,
        alter_artifact: Any = None,
    ) -> None:
        chosen_rows = flat if role_rows is None else role_rows
        role_path.parent.mkdir(parents=True, exist_ok=True)
        role_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in chosen_rows),
            encoding="utf-8",
        )
        chosen_protocol = protocol if protocol_value is None else protocol_value
        mod.atomic_json(protocol_path, chosen_protocol)
        artifact = {
            "protocol_sha256": protocol["protocol_sha256"],
            "raw_sidecars": {
                "cached_roles": {
                    "path": role_path.relative_to(tmp_path).as_posix(),
                    "sha256": mod.sha256_file(role_path),
                    "rows": len(chosen_rows),
                },
                "frozen_protocol": {
                    "path": protocol_path.relative_to(tmp_path).as_posix(),
                    "sha256": mod.sha256_file(protocol_path),
                },
            },
        }
        if alter_artifact is not None:
            alter_artifact(artifact)
        mod.atomic_json(artifact_path, artifact)

    with pytest.raises(ValueError, match="exp7575_artifact_missing"):
        mod.load_inputs(tmp_path / "absent")
    install()
    loaded, loaded_protocol = mod.load_inputs(tmp_path)
    assert sum(map(len, loaded.values())) == len(flat)
    assert loaded_protocol["protocol_sha256"] == protocol["protocol_sha256"]

    cases: list[tuple[str, Any]] = [
        (
            "cached_roles_hash_mismatch",
            lambda artifact: artifact["raw_sidecars"]["cached_roles"].__setitem__(
                "sha256", "sha256:changed"
            ),
        ),
        (
            "cached_roles_count_mismatch",
            lambda artifact: artifact["raw_sidecars"]["cached_roles"].__setitem__("rows", 1),
        ),
        (
            "frozen_protocol_hash_mismatch",
            lambda artifact: artifact["raw_sidecars"]["frozen_protocol"].__setitem__(
                "sha256", "sha256:changed"
            ),
        ),
        (
            "protocol_artifact_hash_mismatch",
            lambda artifact: artifact.__setitem__("protocol_sha256", "sha256:changed"),
        ),
    ]
    for error, mutation in cases:
        install(alter_artifact=mutation)
        with pytest.raises(ValueError, match=error):
            mod.load_inputs(tmp_path)

    unknown = deepcopy(flat)
    unknown[0]["role"] = "unknown"
    install(role_rows=unknown)
    with pytest.raises(ValueError, match="unknown_role:unknown"):
        mod.load_inputs(tmp_path)

    install(protocol_value=[])
    with pytest.raises(ValueError, match="frozen_protocol_invalid"):
        mod.load_inputs(tmp_path)

    changed_protocol = deepcopy(protocol)
    changed_protocol["protocol_sha256"] = "sha256:changed"
    install(
        protocol_value=changed_protocol,
        alter_artifact=lambda artifact: artifact.__setitem__("protocol_sha256", "sha256:changed"),
    )
    with pytest.raises(ValueError, match="protocol_role_hash_mismatch"):
        mod.load_inputs(tmp_path)

    changed_protocol = deepcopy(protocol)
    changed_protocol["order_seeds"] = [1]
    install(protocol_value=changed_protocol)
    with pytest.raises(ValueError, match="protocol_order_seed_mismatch"):
        mod.load_inputs(tmp_path)


def test_role_row_field_guards_and_external_path_label(tmp_path: Path) -> None:
    """REQ-CL-7578: cached row identities, labels, and probabilities are typed."""

    for field, value, error in (
        ("role", "wrong", "role_identity_invalid:fit"),
        ("label", 2, "binary_label_invalid:fit"),
        ("probability", float("inf"), "probability_invalid:fit"),
    ):
        roles = _roles()
        roles["fit"][0][field] = value
        with pytest.raises(ValueError, match=error):
            mod.validate_roles(roles)
    assert mod._path_label(Path("/outside-exp7578"), tmp_path) == "/outside-exp7578"


def test_validation_manifest_commands_and_cli_readers(tmp_path: Path) -> None:
    """SCENARIO-CL-7578-TERMINAL: validation stays scoped to affected files."""

    commands = mod.build_validation_commands(mod.REPO_ROOT, tmp_path / "private")
    assert [command.name for command in commands] == list(mod.validation_scope.REQUIRED_CHECK_NAMES)
    assert (tmp_path / "private" / "pytest").is_dir()
    assert all(
        str(mod.TEST_PATH) in " ".join(command.argv) or command.name != "pytest_scoped"
        for command in commands
    )
    terminal = mod.terminal_commands(tmp_path / "candidate.json", mod.REPO_ROOT)
    assert [command.name for command in terminal] == list(mod.TERMINAL_CHECK_NAMES)
    assert "--strict" in terminal[-1].argv
    preterminal = mod.terminal_commands(
        tmp_path / "candidate.json", mod.REPO_ROOT, allow_preterminal=True
    )
    assert "--allow-preterminal" in preterminal[0].argv

    args = mod.parse_args(["--root", str(tmp_path), "--date", mod.RUN_DATE])
    assert args.root == tmp_path
    assert args.date == mod.RUN_DATE


def test_cli_reader_modes_and_wrong_date(
    built_artifact: tuple[Path, dict[str, Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-CL-7578-TERMINAL: fresh CLI modes read serialized evidence."""

    tmp_path, artifact = built_artifact
    path = tmp_path / "artifact.json"
    mod.atomic_json(path, artifact)
    assert (
        mod.main(
            [
                "--root",
                str(tmp_path),
                "--date",
                mod.RUN_DATE,
                "--cold-replay",
                str(path),
                "--expected-replays",
                "4",
            ]
        )
        == 0
    )
    assert "cold_replay_passed" in capsys.readouterr().out
    assert (
        mod.main(
            [
                "--root",
                str(tmp_path),
                "--date",
                mod.RUN_DATE,
                "--independent-reduce",
                str(path),
                "--expected-replays",
                "4",
            ]
        )
        == 0
    )
    assert "independent_reduction_passed" in capsys.readouterr().out
    with pytest.raises(ValueError, match="run_date_must_equal"):
        mod.main(["--root", str(tmp_path), "--date", "19000101"])
    with pytest.raises(ValueError, match="artifact_unreadable_or_not_object"):
        mod.cold_replay(tmp_path / "missing.json", root=tmp_path, expected_replays=4)
    with pytest.raises(ValueError, match="artifact_unreadable_or_not_object"):
        mod.independent_reduce(tmp_path / "missing.json", root=tmp_path, expected_replays=4)


def test_thin_wrapper_and_affected_manifest_are_exact(tmp_path: Path) -> None:
    """REQ-CL-7578: the public entrypoint delegates and the file scope is frozen."""

    wrapper = mod.REPO_ROOT / mod.WRAPPER_PATH
    text = wrapper.read_text(encoding="utf-8") if wrapper.exists() else ""
    if text:
        assert len(text.splitlines()) <= 20
        assert "experiment_7578_v662_continuous_proper_loss" in text
    manifest = tmp_path / "manifest.json"
    mod.write_affected_manifest(manifest)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    assert value == {
        "experiment_id": mod.EXPERIMENT_ID,
        "test_paths": [mod.TEST_PATH.as_posix()],
        "changed_modules": [mod.MODULE_PATH.as_posix()],
        "static_paths": [mod.WRAPPER_PATH.as_posix()],
    }


def test_input_and_artifact_contract_mutations_fail_closed(
    built_artifact: tuple[Path, dict[str, Any]],
) -> None:
    """REQ-CL-7578: protocol drift and terminal claim drift are rejected."""

    roles = _roles()
    duplicate = deepcopy(roles)
    duplicate["online"][1]["source_id"] = duplicate["online"][0]["source_id"]
    with pytest.raises(ValueError, match="source_id_duplicate"):
        mod.validate_roles(duplicate)
    missing = deepcopy(roles)
    missing["test"].pop()
    with pytest.raises(ValueError, match="role_count_invalid:test"):
        mod.validate_roles(missing)

    tmp_path, artifact = built_artifact
    changed = deepcopy(artifact)
    changed["fresh_confirmatory_claim_allowed"] = True
    changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
    with pytest.raises(ValueError, match="fresh_claim_forbidden"):
        mod.validate_artifact(changed, root=tmp_path, expected_replays=4)
    changed = deepcopy(artifact)
    changed["invocation_counts"] = deepcopy(mod.ZERO_INVOCATION_COUNTS)
    changed["invocation_counts"]["tokens"]["completed"] = 1
    changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
    with pytest.raises(ValueError, match="current_invocations_nonzero"):
        mod.validate_artifact(changed, root=tmp_path, expected_replays=4)


def test_low_level_invalid_inputs_and_unreachable_release_paths(tmp_path: Path) -> None:
    """REQ-CL-7578: malformed rows and rosters fail before measurement."""

    with pytest.raises(ValueError, match="fit_probabilities_empty"):
        mod.count_config_from_rows([])
    with pytest.raises(ValueError, match="probability_or_label_invalid"):
        mod.metric_row("bad", "raw", 2.0, 0, seed=1, phase="online")
    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="jsonl_object_required"):
        mod._read_jsonl(bad_jsonl)
    roles = _roles()
    config = mod.count_config_from_rows(roles["fit"])
    order = mod.freeze_protocol(roles)["orders"][str(mod.ORDER_SEEDS[0])]
    with pytest.raises(ValueError, match="online_order_roster_invalid"):
        mod.measure_order(
            roles["online"][:-1], roles["test"], order, config, seed=1, state_dir=tmp_path
        )
    with pytest.raises(ValueError, match="retention_roster_invalid"):
        mod.measure_order(
            roles["online"], roles["test"][:-1], order, config, seed=1, state_dir=tmp_path
        )
    machine = mod.RecalibrationEventMachine.create(config)
    assert mod._reject_without_mutation(machine, lambda: None, "missing") is False
    assert mod._interval([]) == {"count": 0, "mean": None, "lower95": None, "upper95": None}


def test_bootstrap_and_reducer_defensive_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7578-UNCERTAINTY: malformed draws cannot satisfy completion."""

    roles = _roles()
    protocol = mod.freeze_protocol(roles)
    config = mod.count_config_from_rows(roles["fit"])
    with pytest.raises(ValueError, match="replays_per_order_invalid"):
        mod.causal_bootstrap(
            roles["online"],
            roles["test"],
            protocol["orders"],
            config,
            replays_per_order=0,
            checkpoint_path=tmp_path / "zero.jsonl",
        )
    with pytest.raises(ValueError, match="bootstrap_order_invalid"):
        mod.causal_bootstrap(
            roles["online"],
            roles["test"],
            {str(mod.ORDER_SEEDS[0]): []},
            config,
            replays_per_order=1,
            order_seeds=mod.ORDER_SEEDS[:1],
            checkpoint_path=tmp_path / "bad-order.jsonl",
        )
    callbacks: list[tuple[int, int]] = []
    small = mod.causal_bootstrap(
        roles["online"],
        roles["test"],
        protocol["orders"],
        config,
        replays_per_order=1,
        order_seeds=mod.ORDER_SEEDS[:1],
        checkpoint_path=tmp_path / "callback.jsonl",
        progress_callback=lambda done, total: callbacks.append((done, total)),
    )
    assert callbacks == [(1, 1)]
    rows = small["rows"]
    for field, error in (
        ("mean_action_cost", "mean_action_cost_mismatch"),
        ("retention_mean_brier", "retention_brier_mismatch"),
    ):
        changed = deepcopy(rows)
        changed[0][field] += 1.0
        with pytest.raises(ValueError, match=error):
            mod.reduce_bootstrap(changed, expected_replays=1, expected_order_count=1)
    duplicate = [deepcopy(row) for row in rows]
    duplicate[1]["arm"] = duplicate[0]["arm"]
    with pytest.raises(ValueError, match="duplicate_unit_arm"):
        mod.reduce_bootstrap(duplicate, expected_replays=1, expected_order_count=1)
    missing = [deepcopy(row) for row in rows[:-1]]
    with pytest.raises(ValueError, match="unit_arm_roster_invalid"):
        mod.reduce_bootstrap(missing, expected_replays=1, expected_order_count=1)

    monkeypatch.setattr(mod, "_bootstrap_one", lambda *args, **kwargs: [{"checkpoint": True}])
    durable = mod.causal_bootstrap(
        roles["online"],
        roles["test"],
        protocol["orders"],
        config,
        replays_per_order=50,
        order_seeds=mod.ORDER_SEEDS[:1],
        checkpoint_path=tmp_path / "fsync-50.jsonl",
    )
    assert durable["completed_replays"] == 50
