"""Tests for REQ-CL-7418 and SCENARIO-CL-7418-*.

The exact formula is the authority in these tests. Proof memory may avoid an
exact query only after the current source accepts every edge in its path.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time

import pytest

from carnot import experiment_7418_v650_revision_memory as experiment
from carnot.learning.implication_memory import ProofMemory, ProofPath, execute_query


@pytest.fixture(scope="module")
def schedule() -> dict:
    """SCENARIO-CL-7418-SCHEDULE: Freeze all sources before an arm runs."""

    return experiment.freeze_schedule()


@pytest.fixture(scope="module")
def evidence(schedule: dict, tmp_path_factory: pytest.TempPathFactory) -> experiment.Evidence:
    """SCENARIO-CL-7418-TRUTH: Measure every arm against enumerated truth."""

    return experiment.run_schedule(schedule, tmp_path_factory.mktemp("revision-memory"))


def test_schedule_is_complete_deterministic_and_mixed(schedule: dict) -> None:
    """REQ-CL-7418: The sealed schedule has 32 four-epoch streams."""

    assert schedule == experiment.freeze_schedule()
    assert schedule["seed"] == 6_501_801
    assert len(schedule["streams"]) == 32
    assert experiment.schedule_errors(schedule) == []
    for stream in schedule["streams"]:
        assert stream["n_vars"] <= 12
        assert [epoch["condition"] for epoch in stream["epochs"]] == [
            "initial",
            "deletion",
            "addition",
            "recurrence",
        ]
        assert all(len(epoch["requests"]) == 12 for epoch in stream["epochs"])
        for epoch in stream["epochs"]:
            formula = experiment.formula_for_epoch(stream, epoch)
            truths = [
                experiment.enumerate_formula(formula, request["assumptions"])[0]
                for request in epoch["requests"]
            ]
            assert set(truths) == {False, True}


def test_schedule_mutations_fail_closed(schedule: dict) -> None:
    """SCENARIO-CL-7418-SCHEDULE: Missing epochs and future labels are rejected."""

    changed = deepcopy(schedule)
    changed["streams"][0]["epochs"].pop()
    assert "epoch_count" in experiment.schedule_errors(changed)
    changed = deepcopy(schedule)
    changed["streams"][0]["epochs"][0]["requests"][0]["satisfiable"] = False
    assert "future_solution_exposed" in experiment.schedule_errors(changed)


def test_revision_rechecks_clause_content_not_identifier(schedule: dict) -> None:
    """SCENARIO-CL-7418-AUTHORITY: Source bytes, not reused IDs, grant authority."""

    stream = schedule["streams"][0]
    initial = experiment.formula_for_epoch(stream, stream["epochs"][0])
    deleted = experiment.formula_for_epoch(stream, stream["epochs"][1])
    recurrent = experiment.formula_for_epoch(stream, stream["epochs"][3])
    learned = execute_query(ProofMemory.empty(initial), [1, -3]).committed_memory
    assert learned.paths

    retained, rejected = experiment.recheck_paths(learned.paths, initial, recurrent)
    assert len(retained) == len(learned.paths)
    assert rejected == []
    assert all(path.source_hash == recurrent.source_hash for path in retained)
    assert all(path.formula_version == recurrent.version for path in retained)

    retained, rejected = experiment.recheck_paths(learned.paths, initial, deleted)
    assert retained == ()
    assert {row["reason"] for row in rejected} == {"supporting_clause_missing"}

    reused = replace(deleted, version=initial.version, source_hash=initial.source_hash)
    retained, rejected = experiment.recheck_paths(learned.paths, initial, reused)
    assert retained == ()
    assert {row["reason"] for row in rejected} == {"supporting_clause_missing"}

    corrupt = replace(learned.paths[0], source_hash="sha256:" + "0" * 64)
    retained, rejected = experiment.recheck_paths((corrupt,), initial, recurrent)
    assert retained == ()
    assert rejected[0]["reason"] == "original_proof_invalid"


def test_renamed_variables_do_not_inherit_authority(schedule: dict) -> None:
    """SCENARIO-CL-7418-AUTHORITY: Renamed literals need new source proofs."""

    stream = schedule["streams"][0]
    initial = experiment.formula_for_epoch(stream, stream["epochs"][0])
    learned = execute_query(ProofMemory.empty(initial), [1, -3]).committed_memory
    renamed = experiment.renamed_formula(initial)
    retained, rejected = experiment.recheck_paths(learned.paths, initial, renamed)
    assert retained == ()
    assert all(row["reason"] == "supporting_clause_missing" for row in rejected)


def test_all_five_arms_are_safe_and_restart_exactly(
    schedule: dict, evidence: experiment.Evidence
) -> None:
    """SCENARIO-CL-7418-TRUTH: Every decision matches fresh enumeration."""

    assert len(evidence.rows) == 32 * 48 * len(experiment.ARMS)
    assert {row["arm"] for row in evidence.rows} == set(experiment.ARMS)
    assert all(row["truth_match"] for row in evidence.rows)
    assert all(row["failed"] is False and row["censored"] is False for row in evidence.rows)
    assert all(row["complete_service_ns"] > 0 for row in evidence.rows)
    assert all(row["memory_bytes"] <= 65_536 for row in evidence.rows)
    assert all(row["retained_path_count"] <= 128 for row in evidence.rows)
    assert len(evidence.revision_rows) == 32 * 4
    assert all(row["cold_restart_equal"] for row in evidence.restart_rows)
    assert len(evidence.restart_rows) == 32
    assert all(row["passed"] for row in evidence.attack_rows)
    assert {
        "revoked_clause",
        "reused_clause_id",
        "renamed_variables",
        "corrupted_hash",
        "restart_during_commit",
        "eviction",
        "recurrence",
        "deliberately_stale_cache",
    } <= {row["control"] for row in evidence.attack_rows}


def test_retention_uses_one_snapshot_then_commits(evidence: experiment.Evidence) -> None:
    """SCENARIO-CL-7418-COMMIT: Later additions never affect their own request."""

    retained = [row for row in evidence.rows if row["arm"] == "version_checked_retention"]
    assert any(not row["paid_exact_query"] for row in retained)
    assert all(row["entry_snapshot_hash"] for row in retained)
    assert all(row["additions_committed"] <= 8 for row in retained)
    assert all(
        row["used_path_id"] in row["entry_path_ids"]
        for row in retained
        if row["used_path_id"] is not None
    )
    assert all(
        path_id not in row["entry_path_ids"]
        for row in retained
        for path_id in row["added_path_ids"]
        if path_id not in row["entry_path_ids"]
    )


def test_erasure_witnesses_remove_the_registered_benefit(
    evidence: experiment.Evidence,
) -> None:
    """SCENARIO-CL-7418-VALUE: Removing a used path restores exact work."""

    assert len(evidence.erasure_witness_rows) >= 8
    assert len({row["stream_id"] for row in evidence.erasure_witness_rows}) >= 4
    assert all(row["with_path_paid_exact"] is False for row in evidence.erasure_witness_rows)
    assert all(row["without_path_paid_exact"] is True for row in evidence.erasure_witness_rows)
    assert all(row["decision_unchanged"] for row in evidence.erasure_witness_rows)


def test_independent_reduction_and_value_gates_are_registered(
    schedule: dict, evidence: experiment.Evidence
) -> None:
    """SCENARIO-CL-7418-VALUE: Bootstrap, safety, and cost gates stay explicit."""

    metrics = experiment.reduce_evidence(evidence)
    assert metrics["bootstrap_draws"] == 10_000
    assert metrics["bootstrap_seed"] == 6_501_807
    assert metrics["unsafe_rejections"] == 0
    assert metrics["stale_proof_acceptances"] == 0
    assert metrics["cold_restart_mismatches"] == 0
    assert set(metrics["paid_query_ratio_ci95_upper"]) == set(experiment.COMPARATORS)
    assert set(metrics["total_cost_ratio_ci95_upper"]) == set(experiment.COMPARATORS)

    artifact = experiment.build_artifact_for_test(schedule, evidence)
    assert artifact["memory_revision_capture_complete_score"] == 1
    assert artifact["memory_revision_value_score"] in {0, 1}
    assert artifact["verdict_class"] in {"null", "circular_positive"}
    assert artifact["verifier_is_oracle"] is True
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["promotion_score"] == 0
    assert experiment.validate_artifact(artifact) == []


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda value: value.update(schema="bad"), "identity_mismatch"),
        (lambda value: value.update(MODEL_SPECS=["model"]), "model_declaration_mismatch"),
        (
            lambda value: value["invocation_counts"].update(generation_calls_attempted=1),
            "invocation_counts_nonzero",
        ),
        (lambda value: value.update(inference_substrate={}), "substrate_mismatch"),
        (lambda value: value.update(execution_venue="host_cpu"), "substrate_mismatch"),
        (lambda value: value.update(verifier_is_oracle=False), "oracle_mismatch"),
        (
            lambda value: value.update(continuous_self_learning_task=False),
            "continuous_learning_mismatch",
        ),
        (lambda value: value.update(promotion_score=1), "promotion_nonzero"),
        (lambda value: value.update(rows=[]), "reduction_mismatch"),
        (lambda value: value.update(revision_rows=[]), "reduction_mismatch"),
        (lambda value: value.update(erasure_witness_rows=[]), "reduction_mismatch"),
        (lambda value: value.update(memory_revision_value_score=-1), "value_score_mismatch"),
        (lambda value: value.update(verdict_class="positive"), "verdict_class_mismatch"),
        (lambda value: value.update(acceptance_gate_results=[]), "gates_mismatch"),
        (lambda value: value.update(field_principles={}), "field_principles_incomplete"),
        (
            lambda value: value.update(reproducibility_checksum="bad"),
            "reproducibility_checksum_mismatch",
        ),
        (lambda value: value.update(honest_verdict="wrong"), "honest_verdict_mismatch"),
    ],
)
def test_artifact_validation_rejects_drift(
    schedule: dict,
    evidence: experiment.Evidence,
    mutation,
    expected: str,
) -> None:
    """SCENARIO-CL-7418-ARTIFACT: Derived terminal fields cannot drift."""

    artifact = experiment.build_artifact_for_test(schedule, evidence)
    mutation(artifact)
    assert expected in experiment.validate_artifact(artifact)


def test_preconditions_authenticate_local_branches() -> None:
    """REQ-CL-7418: Available historical branches retain exact byte identities."""

    checks, hashes, sidecars = experiment.collect_preconditions(experiment.REPO_ROOT)
    assert all(row["passed"] for row in checks)
    assert experiment.UPSTREAM_7403_PATH.as_posix() in hashes
    assert experiment.UPSTREAM_7405_PATH.as_posix() in hashes
    assert all(row["counted_as_current"] is False for row in sidecars)


def test_raw_evidence_and_cold_replay_detect_changes(
    tmp_path: Path,
    schedule: dict,
    evidence: experiment.Evidence,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CL-7418-ARTIFACT: Raw rows and the schedule stay hash-bound."""

    checkpoint_dir = tmp_path / experiment.RAW_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "committed.json").write_text("{}\n", encoding="utf-8")
    hashes = experiment.write_raw_evidence(tmp_path, schedule, evidence, [])
    assert any(path.endswith("committed.json") for path in hashes)
    artifact = experiment.build_artifact_for_test(schedule, evidence, source_hashes=hashes)
    artifact["source_artifact_hashes"]["external/not-raw.json"] = "sha256:" + "2" * 64
    experiment.finalize_artifact(artifact)
    assert experiment.cold_reload_errors(artifact, tmp_path, rerun=False) == []

    path = tmp_path / experiment.RAW_DIR / "revision_evidence.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rows"].pop()
    experiment.atomic_json(path, payload)
    errors = experiment.cold_reload_errors(artifact, tmp_path, rerun=False)
    assert "raw_rows_mismatch" in errors
    assert any(error.startswith("raw_hash_mismatch:") for error in errors)

    experiment.atomic_json(
        path,
        {
            "rows": evidence.rows,
            "revision_rows": evidence.revision_rows,
            "erasure_witness_rows": evidence.erasure_witness_rows,
            "restart_rows": evidence.restart_rows,
            "authority_control_rows": evidence.attack_rows,
        },
    )
    schedule_path = tmp_path / experiment.RAW_DIR / "frozen_schedule.json"
    changed_schedule = deepcopy(schedule)
    changed_schedule["seed"] += 1
    experiment.atomic_json(schedule_path, changed_schedule)
    assert "raw_schedule_mismatch" in experiment.cold_reload_errors(artifact, tmp_path, rerun=False)
    experiment.atomic_json(schedule_path, schedule)

    changed_evidence = deepcopy(evidence)
    changed_evidence.rows[0]["decision"] = "changed"
    monkeypatch.setattr(experiment, "run_schedule", lambda *_args, **_kwargs: changed_evidence)
    assert "cold_semantic_replay_mismatch" in experiment.cold_reload_errors(
        artifact, tmp_path, rerun=True
    )


def test_scoped_plan_and_helpers_remain_bounded(tmp_path: Path, capsys) -> None:
    """SCENARIO-CL-7418-ARTIFACT: Only frozen affected checks can publish."""

    commands = experiment.scoped_command_plan(experiment.REPO_ROOT, tmp_path / "private")
    assert experiment.validate_scoped_command_plan(experiment.REPO_ROOT, commands) == []
    assert {command.name for command in commands} == set(experiment.REQUIRED_CHECK_NAMES)
    assert all(command.name != "full_python_suite" for command in commands)
    coverage = next(
        command for command in commands if command.name == "changed_module_coverage_report"
    )
    assert dict(getattr(coverage, "command_environment", ()))["COVERAGE_FILE"].startswith(
        str(tmp_path)
    )
    assert "required_command_names_changed" in experiment.validate_scoped_command_plan(
        experiment.REPO_ROOT, commands[:-1]
    )
    forbidden = [
        *commands,
        experiment.validation_scope.CommandSpec("full_python_suite", ("true",), "repository"),
    ]
    assert "full_python_suite_forbidden" in experiment.validate_scoped_command_plan(
        experiment.REPO_ROOT, forbidden
    )

    started = time.monotonic()
    experiment.progress(started, "test", "boundary", units=1)
    assert "phase=test event=boundary" in capsys.readouterr().out
    args = experiment.parse_args(["--date", "20260919", "--output", str(tmp_path / "out.json")])
    assert args.date == "20260919"
    assert experiment.validate_artifact([]) == ["artifact_not_object"]
    terminal = experiment.terminal_commands(tmp_path / "candidate.json")
    assert {row.spec.name for row in terminal} == set(experiment.TERMINAL_CHECK_NAMES[:-1])

    normalized = experiment._normalized_receipt(
        {
            "name": "worktree_imports",
            "command": "python check",
            "command_argv": ["python", "check"],
            "command_environment": {"A": "B"},
            "scope": "changed_modules",
            "exit_code": 0,
            "duration_s": 0.1,
            "log_path": "/tmp/log",
            "log_sha256": "sha256:" + "1" * 64,
            "passed": True,
            "timed_out": False,
            "resolved_imports": {"carnot.x": "/worktree/x.py"},
        }
    )
    assert normalized["resolved_imports"]
    span = experiment._span("test", started, started, experiment.utc_now(), checkpoints=1)
    assert span["checkpoints"] == 1


def test_recheck_rejects_malformed_and_eviction_is_bounded(schedule: dict) -> None:
    """SCENARIO-CL-7418-AUTHORITY: Invalid endpoints and oversized archives fail closed."""

    stream = schedule["streams"][0]
    formula = experiment.formula_for_epoch(stream, stream["epochs"][0])
    learned = execute_query(ProofMemory.empty(formula), [1, -3]).committed_memory
    malformed = replace(learned.paths[0], antecedent=0)
    retained, rejected = experiment.recheck_paths((malformed,), formula, formula)
    assert retained == ()
    assert rejected[0]["reason"] == "original_proof_invalid"

    duplicates: tuple[ProofPath, ...] = tuple(learned.paths[0] for _ in range(200))
    bounded, evicted = experiment.bound_archive(duplicates)
    assert len(bounded) <= 128
    assert experiment.archive_bytes(bounded) <= 65_536
    assert evicted > 0


def test_defensive_schedule_and_source_branches_fail_closed(
    tmp_path: Path, schedule: dict, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7418-SCHEDULE: Malformed private schedules cannot run."""

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[", encoding="utf-8")
    assert experiment.load_object(malformed) == {}
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    assert experiment.load_object(scalar) == {}

    changed = deepcopy(schedule)
    changed["schema"] = "bad"
    changed["seed"] = 0
    changed["streams"] = []
    assert set(experiment.schedule_errors(changed)) == {
        "schedule_schema",
        "schedule_seed",
        "stream_count",
    }
    changed = deepcopy(schedule)
    changed["streams"][0]["n_vars"] = 0
    assert "n_vars" in experiment.schedule_errors(changed)
    changed = deepcopy(schedule)
    changed["streams"][0]["epochs"][0]["condition"] = "wrong"
    assert "epoch_order" in experiment.schedule_errors(changed)
    changed = deepcopy(schedule)
    changed["streams"][0]["epochs"][0]["requests"].pop()
    assert "request_count" in experiment.schedule_errors(changed)
    changed = deepcopy(schedule)
    changed["streams"][0]["epochs"][0]["clauses"][0] = [0, 1]
    assert "formula_invalid" in experiment.schedule_errors(changed)
    changed = deepcopy(schedule)
    changed["streams"][0]["epochs"][3]["clauses"].append([1, 1])
    assert "recurrence_not_exact" in experiment.schedule_errors(changed)
    with pytest.raises(ValueError, match="invalid_schedule"):
        experiment.run_schedule(changed, tmp_path / "invalid")

    stream = schedule["streams"][0]
    formula = experiment.formula_for_epoch(stream, stream["epochs"][0])
    proof = execute_query(ProofMemory.empty(formula), [1, -3]).committed_memory.paths[0]
    retained, rejected, checked = experiment._paths_for_formula((proof,), {}, formula)
    assert retained == () and checked == 0
    assert rejected[0]["reason"] == "original_source_missing"

    mini = deepcopy(schedule)
    mini["streams"] = mini["streams"][:1]
    monkeypatch.setattr(experiment, "STREAM_COUNT", 1)
    experiment.run_schedule(mini, tmp_path / "mini", emit_progress=True)
    assert "stream_complete" in capsys.readouterr().out


def test_terminal_classification_branches_are_explicit(
    schedule: dict, evidence: experiment.Evidence, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7418-VALUE: Blocked, unsafe, positive, and null stay distinct."""

    blocked = experiment.build_artifact_for_test(
        schedule,
        evidence,
        preconditions=[experiment._precondition("missing", "input", "field", "present", None)],
    )
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]["status"] == "blocked_precondition"

    disqualified = experiment.build_artifact_for_test(schedule, evidence)
    disqualified["flagged_adversarial"] = True
    experiment.finalize_artifact(disqualified)
    assert disqualified["verdict_class"] == "disqualified"

    positive = experiment.build_artifact_for_test(schedule, evidence)
    reduced = deepcopy(positive["independent_reduction"])
    reduced["memory_revision_value_score"] = 1
    monkeypatch.setattr(experiment, "independent_reduce", lambda _value: reduced)
    experiment.finalize_artifact(positive)
    assert positive["verdict_class"] == "circular_positive"
