"""Tests for Exp6521 transactional refinement-gated conflict memory.

Spec refs: REQ-STORE-6521, SCENARIO-STORE-6521-VALID-REUSE,
SCENARIO-STORE-6521-INVALID-VETO, SCENARIO-STORE-6521-LIFECYCLE,
SCENARIO-STORE-6521-FIXED-WIDTH-MAPPING.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6521_transactional_refinement_conflict_memory as mod
from carnot.atomic_shard_transaction import CrashInjected, CrashPlan


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH

FOCUSED_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6521_transactional_refinement_conflict_memory.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6521_transactional_refinement_conflict_memory.py "
    "-m pytest tests/python/test_experiment_6521_transactional_refinement_conflict_memory.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6521_transactional_refinement_conflict_memory.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6521_transactional_refinement_conflict_memory.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6521_transactional_refinement_conflict_memory.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6521_transactional_refinement_conflict_memory.json"
)
EXACT_E2E_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py "
    "-q --no-cov -n 0"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m "
    "carnot.experiment_6521_transactional_refinement_conflict_memory --date 20260823"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6521_transactional_refinement_conflict_memory --validate"
)
GIT_STATUS_COMMAND = "git status --short"

TESTS_RUN = [
    {"command": FOCUSED_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": SPEC_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": EXACT_E2E_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": GIT_STATUS_COMMAND, "exit_code": 0},
]


def _memory(
    tmp_path: Path,
    *,
    capacity: int = 4,
    crash_plan: CrashPlan | None = None,
) -> mod.TransactionalConflictMemory:
    return mod.TransactionalConflictMemory(
        capacity=capacity,
        memory_path=tmp_path / "memory.json",
        transaction_work_dir=tmp_path / "tx",
        crash_plan=crash_plan,
    )


def _source_query() -> mod.ExactQuery:
    return mod.ExactQuery(variable_count=2, clauses=((1,),))


def _target_query() -> mod.ExactQuery:
    return mod.ExactQuery(variable_count=2, clauses=((1,), (2,)))


def _commit_record(
    memory: mod.TransactionalConflictMemory,
    *,
    source: mod.ExactQuery | None = None,
    target: mod.ExactQuery | None = None,
    clause: tuple[int, ...] = (1,),
    benefit_score: float = 1.0,
) -> mod.ConflictRecord:
    prepared = memory.prepare(
        source_query=source or _source_query(),
        target_query=target or _target_query(),
        clause=clause,
        benefit_score=benefit_score,
        benefit_observations=1,
    )
    memory.validate(prepared)
    return memory.commit(prepared)


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """REQ-STORE-6521: build a temp artifact without touching tracked results."""

    root = tmp_path_factory.mktemp("exp6521")
    return mod.build_artifact(
        repo_root=REPO,
        result_path=root / mod.RESULT_RELATIVE_PATH.name,
        work_root=root / "work",
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )


def test_req_store_6521_spec_declares_conflict_memory_contract() -> None:
    """REQ-STORE-6521: OpenSpec owns the conflict-memory contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-STORE-6521") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-STORE-6521-VALID-REUSE",
        "SCENARIO-STORE-6521-INVALID-VETO",
        "SCENARIO-STORE-6521-LIFECYCLE",
        "SCENARIO-STORE-6521-FIXED-WIDTH-MAPPING",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "conflict_memory_controller_ready_score",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_store_6521_valid_reuse_and_lifecycle(tmp_path: Path) -> None:
    """SCENARIO-STORE-6521-VALID-REUSE: exact refinement permits reuse."""

    memory = _memory(tmp_path)
    prepared = memory.prepare(
        source_query=_source_query(),
        target_query=_target_query(),
        clause=(1,),
        benefit_score=2.0,
        benefit_observations=3,
    )
    validation = memory.validate(prepared)
    committed = memory.commit(prepared)
    aborted = memory.abort(prepared)

    assert validation["accepted"] is True
    assert committed.lifecycle_state == "active"
    assert committed.use_count == 0
    assert aborted["durable_write_performed"] is False
    assert aborted["lifecycle_state"] == "aborted"
    assert committed.content_hash.startswith("sha256:")
    assert committed.replay_receipt["exact_replay_valid"] is True
    assert committed.refinement_witness["relation"] == mod.REFINEMENT_RELATION

    loaded = _memory(tmp_path)
    load_receipt = loaded.load()
    use_receipt = loaded.use(committed.content_hash, _target_query())

    assert load_receipt["active_record_count"] == 1
    assert use_receipt["memory_used"] is True
    assert use_receipt["exact_replay_valid"] is True
    assert loaded.records[committed.content_hash].use_count == 1
    assert loaded.state_hash().startswith("sha256:")


def test_scenario_store_6521_invalid_reuse_vetoes(tmp_path: Path) -> None:
    """SCENARIO-STORE-6521-INVALID-VETO: unsafe records never write or use."""

    memory = _memory(tmp_path)
    source = mod.ExactQuery(variable_count=2, clauses=((1,), (2,)))
    relaxed = mod.ExactQuery(variable_count=2, clauses=((1,),))
    unrelated = mod.ExactQuery(variable_count=2, clauses=((2,),))
    schema_mismatch = mod.ExactQuery(
        variable_count=2,
        clauses=((1,), (2,)),
        schema_version="bad.schema",
    )

    invalid_rows = [
        memory.prepare_veto_row(
            source_query=source,
            target_query=relaxed,
            clause=(1,),
            attack_id="relaxed_query",
        ),
        memory.prepare_veto_row(
            source_query=_source_query(),
            target_query=unrelated,
            clause=(1,),
            attack_id="unrelated_query",
        ),
        memory.prepare_veto_row(
            source_query=_source_query(),
            target_query=schema_mismatch,
            clause=(1,),
            attack_id="schema_mismatch",
        ),
        memory.prepare_veto_row(
            source_query=_source_query(),
            target_query=_target_query(),
            clause=(2,),
            attack_id="invalid_replay",
        ),
    ]
    assert {row["attack_id"] for row in invalid_rows} == {
        "relaxed_query",
        "unrelated_query",
        "schema_mismatch",
        "invalid_replay",
    }
    assert all(row["durable_write_performed"] is False for row in invalid_rows)
    assert all(row["unsafe_use_performed"] is False for row in invalid_rows)

    valid = memory.prepare(
        source_query=_source_query(),
        target_query=_target_query(),
        clause=(1,),
        benefit_score=1.0,
        benefit_observations=1,
    )
    stale = deepcopy(valid)
    stale.source_query_hash = "sha256:" + "0" * 64
    stale.content_hash = mod.conflict_record_content_hash(stale)
    with pytest.raises(mod.ConflictMemoryError, match="source_query_hash_mismatch"):
        memory.validate(stale)

    solver_mismatch = deepcopy(valid)
    solver_mismatch.solver_hash = "sha256:" + "1" * 64
    solver_mismatch.content_hash = mod.conflict_record_content_hash(solver_mismatch)
    with pytest.raises(mod.ConflictMemoryError, match="solver_hash_mismatch"):
        memory.commit(solver_mismatch)

    malformed = deepcopy(valid)
    malformed.clause_payload = (0,)
    malformed.content_hash = mod.conflict_record_content_hash(malformed)
    with pytest.raises(mod.ConflictMemoryError, match="malformed_clause"):
        memory.validate(malformed)

    assert memory.records == {}
    fallback = memory.native_fallback_solve(_target_query())
    assert fallback["memory_used"] is False
    assert fallback["native_status"] == "sat"


def test_scenario_store_6521_capacity_eviction_duplicates_and_crash(
    tmp_path: Path,
) -> None:
    """SCENARIO-STORE-6521-LIFECYCLE: bounded transactions stay deterministic."""

    memory = _memory(tmp_path, capacity=2)
    first = _commit_record(
        memory,
        source=mod.ExactQuery(3, ((1,),)),
        target=mod.ExactQuery(3, ((1,), (2,))),
        benefit_score=0.5,
    )
    duplicate = memory.commit(deepcopy(first))
    second = _commit_record(
        memory,
        source=mod.ExactQuery(3, ((2,),)),
        target=mod.ExactQuery(3, ((2,), (3,))),
        clause=(2,),
        benefit_score=1.0,
    )
    memory.use(second.content_hash, mod.ExactQuery(3, ((2,), (3,))))
    third = _commit_record(
        memory,
        source=mod.ExactQuery(3, ((3,),)),
        target=mod.ExactQuery(3, ((3,), (1,))),
        clause=(3,),
        benefit_score=1.0,
    )

    assert duplicate.content_hash == first.content_hash
    assert first.content_hash not in memory.records
    assert {second.content_hash, third.content_hash} == set(memory.records)
    assert memory.eviction_rows[-1]["evicted_content_hash"] == first.content_hash
    assert memory.eviction_rows[-1]["eviction_reason"] == "capacity_limit"

    conflicting = deepcopy(second)
    conflicting.clause_payload = (-2,)
    with pytest.raises(mod.ConflictMemoryError, match="content_hash_mismatch"):
        memory.commit(conflicting)

    stable = _memory(tmp_path / "stable", capacity=3)
    original = _commit_record(stable)
    before_hash = stable.state_hash()
    crashy = _memory(tmp_path / "stable", capacity=3, crash_plan=CrashPlan.once("before_replace"))
    crash_candidate = crashy.prepare(
        source_query=mod.ExactQuery(3, ((2,),)),
        target_query=mod.ExactQuery(3, ((2,), (3,))),
        clause=(2,),
        benefit_score=1.0,
        benefit_observations=1,
    )
    crashy.validate(crash_candidate)
    with pytest.raises(CrashInjected, match="before_replace"):
        crashy.commit(crash_candidate)

    restarted = _memory(tmp_path / "stable", capacity=3)
    restarted.load()
    assert restarted.state_hash() == before_hash
    assert set(restarted.records) == {original.content_hash}


def test_scenario_store_6521_restart_rollback_corruption_and_mapping(
    tmp_path: Path,
) -> None:
    """SCENARIO-STORE-6521-LIFECYCLE/FIXED-WIDTH-MAPPING: state replays exactly."""

    memory = _memory(tmp_path, capacity=4)
    first = _commit_record(memory, benefit_score=2.0)
    checkpoint = memory.checkpoint("baseline")
    baseline_hash = checkpoint["state_hash"]
    second = _commit_record(
        memory,
        source=mod.ExactQuery(3, ((2,),)),
        target=mod.ExactQuery(3, ((2,), (3,))),
        clause=(2,),
        benefit_score=1.5,
    )

    restarted = _memory(tmp_path, capacity=4)
    restart_receipt = restarted.load()
    assert restart_receipt["state_hash"] == memory.state_hash()
    assert {first.content_hash, second.content_hash} == set(restarted.records)

    rollback = restarted.rollback("baseline")
    assert rollback["rolled_back"] is True
    assert rollback["state_hash_after"] == baseline_hash
    assert set(restarted.records) == {first.content_hash}

    mapping_rows = restarted.fixed_width_cpu_mapping_rows()
    assert mapping_rows[0]["logical_bytes"] > 0
    assert mapping_rows[0]["mapped_bytes"] >= mapping_rows[0]["logical_bytes"]
    assert mapping_rows[0]["topology_expansion"] >= 1.0
    assert mapping_rows[0]["hardware_execution_claimed"] is False
    assert mapping_rows[0]["acceleration_claimed"] is False
    assert mapping_rows[0]["unsupported_fields"] == []

    restarted.memory_path.write_text("{not-json", encoding="utf-8")
    corrupted = _memory(tmp_path, capacity=4)
    load_receipt = corrupted.load()
    fallback = corrupted.use_or_native(first.content_hash, _target_query())

    assert load_receipt["corruption_quarantined"] is True
    assert Path(load_receipt["quarantine_path"]).exists()
    assert fallback["memory_used"] is False
    assert fallback["fallback_reason"] == "record_not_available"
    assert fallback["native_status"] == "sat"


def test_scenario_store_6521_artifact_rows_and_validation(
    artifact: dict[str, Any],
) -> None:
    """REQ-STORE-6521: the artifact is row-derived and validates."""

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_transactional_refinement_conflict_memory_ready"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["verdict_class"] != "positive"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["conflict_memory_controller_ready_score"] == 1.0
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["protected_files_unchanged"]["all_protected_files_unchanged"] is True
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert mod.validate_artifact(artifact) == []

    gate = artifact["upstream_gate_receipt"]
    assert gate["artifact_path"] == mod.EXP6517_RELATIVE_PATH.as_posix()
    assert gate["artifact_sha256"].startswith("sha256:")
    assert gate["all_gates_passed"] is True
    assert artifact["conflict_record_schema"]["schema_version"] == mod.RECORD_SCHEMA_VERSION
    assert artifact["refinement_relation_contract"]["relation"] == mod.REFINEMENT_RELATION
    assert artifact["preconditions_checked"]["run_date"] == "20260823"
    assert artifact["preconditions_checked"]["solver_capabilities"]["exact_replay"] is True

    assert all(row["passed"] is True for row in artifact["lifecycle_rows"])
    assert all(row["exact_replay_valid"] is True for row in artifact["valid_reuse_rows"])
    assert all(row["vetoed"] is True for row in artifact["invalid_reuse_veto_rows"])
    assert all(row["passed"] is True for row in artifact["capacity_and_eviction_rows"])
    assert all(row["passed"] is True for row in artifact["restart_rollback_rows"])
    assert all(row["passed"] is True for row in artifact["corruption_quarantine_rows"])
    assert all(row["memory_used"] is False for row in artifact["native_fallback_rows"])
    assert all(
        row["hardware_execution_claimed"] is False and row["acceleration_claimed"] is False
        for row in artifact["fixed_width_mapping_rows"]
    )
    assert artifact["aggregate_row_recomputation"]["unsafe_admission_count"] == 0
    assert artifact["aggregate_row_recomputation"]["unsafe_use_count"] == 0
    assert artifact["gate_check_summary"]["all_gates_passed"] is True


def test_scenario_store_6521_validation_and_cli_roundtrip(
    tmp_path: Path,
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-STORE-6521-LIFECYCLE: CLI writes and validates the artifact."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    work_root = tmp_path / "work"

    assert (
        mod.main(
            [
                "--date",
                "20260823",
                "--result-path",
                str(result_path),
                "--work-root",
                str(work_root),
            ]
        )
        == 0
    )
    assert mod.main(["--validate", "--result-path", str(result_path)]) == 0
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["conflict_memory_controller_ready_score"] == 1.0

    mutations = [
        ("required field set mismatch", lambda item: item.pop("status")),
        (
            "verdict_class cannot be positive",
            lambda item: item.__setitem__("verdict_class", "positive"),
        ),
        (
            "inference_substrate mismatch",
            lambda item: item.__setitem__("inference_substrate", "live_llm_inference"),
        ),
        (
            "verifier_is_oracle must be true",
            lambda item: item.__setitem__("verifier_is_oracle", False),
        ),
        ("field_principles mismatch", lambda item: item.__setitem__("field_principles", {})),
        (
            "field_provenance must cover required fields",
            lambda item: item.__setitem__("field_provenance", {}),
        ),
        (
            "unsafe admission or use detected",
            lambda item: item["invalid_reuse_veto_rows"][0].__setitem__(
                "durable_write_performed", True
            ),
        ),
        (
            "mapping row makes hardware claim",
            lambda item: item["fixed_width_mapping_rows"][0].__setitem__(
                "acceleration_claimed", True
            ),
        ),
        (
            "ready score mismatch",
            lambda item: item.__setitem__("conflict_memory_controller_ready_score", 0.0),
        ),
        (
            "protected files changed",
            lambda item: item["protected_files_unchanged"].__setitem__(
                "all_protected_files_unchanged", False
            ),
        ),
        (
            "reproducibility_checksum mismatch",
            lambda item: item.__setitem__("reproducibility_checksum", "sha256:bad"),
        ),
        (
            "honest_verdict lacks terminal prefix",
            lambda item: item.__setitem__("honest_verdict", "running"),
        ),
    ]
    for expected, mutate in mutations:
        broken = deepcopy(artifact)
        mutate(broken)
        assert expected in mod.validate_artifact(broken)

    invalid = deepcopy(payload)
    invalid["status"] = "running_bootstrap"
    invalid["reproducibility_checksum"] = mod.reproducibility_checksum(invalid)
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="status lacks terminal prefix"):
        mod.main(["--validate", "--result-path", str(invalid_path)])


def test_scenario_store_6521_defensive_paths(
    tmp_path: Path,
    artifact: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-STORE-6521-INVALID-VETO: defensive branches fail closed."""

    assert mod.sha256_file(tmp_path / "missing.json") == "missing"
    assert mod.entails_clause(mod.ExactQuery(1, ((0,),)), (1,))["reason"] == "malformed_clause"
    assert mod.entails_clause(mod.ExactQuery(1, ((),)), (1,))["reason"] == "empty_clause"
    assert mod.entails_clause(mod.ExactQuery(1, ((1,),)), ())["reason"] == "empty_conflict_clause"
    assert mod.entails_clause(mod.ExactQuery(0, ((),)), (0,))["valid"] is False
    with pytest.raises(mod.ConflictMemoryError, match="solver_hash_mismatch"):
        mod.native_exact_solve(mod.ExactQuery(1, ((1,),), solver_hash="sha256:bad"))
    assert mod.native_exact_solve(mod.ExactQuery(1, ((1,), (-1,))))["native_status"] == "unsat"
    with pytest.raises(ValueError, match="capacity must be positive"):
        mod.TransactionalConflictMemory(
            capacity=0,
            memory_path=tmp_path / "bad.json",
            transaction_work_dir=tmp_path / "bad-tx",
        )

    memory = _memory(tmp_path / "defensive")
    missing_load = memory.load()
    assert missing_load["missing_memory"] is True
    accepted_veto = memory.prepare_veto_row(
        source_query=_source_query(),
        target_query=_target_query(),
        clause=(1,),
        attack_id="valid_control",
    )
    assert accepted_veto["vetoed"] is False
    record = _commit_record(memory)
    fallback = memory.use_or_native(record.content_hash, mod.ExactQuery(2, ((2,),)))
    assert fallback["fallback_reason"] == "use_refinement_failed"
    with pytest.raises(mod.ConflictMemoryError, match="checkpoint_not_found"):
        memory.rollback("missing")
    assert memory._quarantine_memory_file("missing").exists()

    tampered_witness = deepcopy(record)
    tampered_witness.refinement_witness["witness_hash"] = "sha256:" + "2" * 64
    tampered_witness.content_hash = mod.conflict_record_content_hash(tampered_witness)
    with pytest.raises(mod.ConflictMemoryError, match="refinement_witness_hash_mismatch"):
        memory.validate(tampered_witness)

    tampered_replay = deepcopy(record)
    tampered_replay.replay_receipt["replay_receipt_hash"] = "sha256:" + "3" * 64
    tampered_replay.content_hash = mod.conflict_record_content_hash(tampered_replay)
    with pytest.raises(mod.ConflictMemoryError, match="replay_receipt_hash_mismatch"):
        memory.validate(tampered_replay)

    false_source = mod.ExactQuery(2, ((1,), (2,)))
    false_target = mod.ExactQuery(2, ((1,),))
    false_witness = mod.prove_refinement(false_source, false_target)
    false_replay = mod.build_replay_receipt(false_source, false_target, (1,), false_witness)
    false_record = mod.ConflictRecord(
        source_query_hash=false_source.query_hash(),
        source_query_payload=false_source.to_dict(),
        target_query_payload=false_target.to_dict(),
        clause_payload=(1,),
        solver_hash=mod.DEFAULT_SOLVER_HASH,
        solver_version_hash=mod.DEFAULT_SOLVER_HASH,
        refinement_witness=false_witness,
        replay_receipt=false_replay,
        lifecycle_state="prepared",
        use_count=0,
        benefit_score=0.0,
        benefit_observations=0,
        created_version=1,
    )
    false_record.content_hash = mod.conflict_record_content_hash(false_record)
    with pytest.raises(mod.ConflictMemoryError, match="refinement_witness_failed"):
        memory.validate(false_record)

    invalid_replay_witness = mod.prove_refinement(_source_query(), _target_query())
    invalid_replay = mod.build_replay_receipt(
        _source_query(),
        _target_query(),
        (2,),
        invalid_replay_witness,
    )
    invalid_replay_record = mod.ConflictRecord(
        source_query_hash=_source_query().query_hash(),
        source_query_payload=_source_query().to_dict(),
        target_query_payload=_target_query().to_dict(),
        clause_payload=(2,),
        solver_hash=mod.DEFAULT_SOLVER_HASH,
        solver_version_hash=mod.DEFAULT_SOLVER_HASH,
        refinement_witness=invalid_replay_witness,
        replay_receipt=invalid_replay,
        lifecycle_state="prepared",
        use_count=0,
        benefit_score=0.0,
        benefit_observations=0,
        created_version=1,
    )
    invalid_replay_record.content_hash = mod.conflict_record_content_hash(invalid_replay_record)
    with pytest.raises(mod.ConflictMemoryError, match="exact_replay_failed"):
        memory.validate(invalid_replay_record)

    missing_quarantine = _memory(tmp_path / "missing-quarantine")
    assert missing_quarantine._quarantine_memory_file("missing").read_text(encoding="utf-8") == (
        "missing\n"
    )

    assert mod._status_and_verdict(0.0, {"checks": {"upstream_gate_passed": False}})[2] == (
        "blocked"
    )
    assert mod._status_and_verdict(0.0, {"checks": {"upstream_gate_passed": True}})[2] == (
        "partial"
    )

    broken = deepcopy(artifact)
    broken["upstream_gate_receipt"]["all_gates_passed"] = False
    assert "upstream gate failed" in mod.validate_artifact(broken)
    broken = deepcopy(artifact)
    broken["conflict_memory_controller_ready_score"] = 0.5
    assert "conflict_memory_controller_ready_score must be 0.0 or 1.0" in (
        mod.validate_artifact(broken)
    )

    rows = {
        key: deepcopy(artifact[key])
        for key in (
            "lifecycle_rows",
            "valid_reuse_rows",
            "invalid_reuse_veto_rows",
            "capacity_and_eviction_rows",
            "restart_rollback_rows",
            "corruption_quarantine_rows",
            "native_fallback_rows",
            "fixed_width_mapping_rows",
        )
    }
    monkeypatch.setattr(mod, "_scenario_rows", lambda work_root: rows)
    monkeypatch.setattr(
        mod,
        "upstream_gate_receipt",
        lambda repo_root: deepcopy(artifact["upstream_gate_receipt"]),
    )
    monkeypatch.setattr(mod, "protected_file_hashes", lambda repo_root: {"p": "sha256:a"})
    relative = mod.build_artifact(
        repo_root=tmp_path,
        result_path=Path("relative.json"),
        work_root=Path("relative-work"),
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )
    assert relative["preconditions_checked"]["result_path"].endswith("relative.json")

    monkeypatch.setattr(mod, "validate_artifact", lambda payload: ["forced validation error"])
    with pytest.raises(ValueError, match="forced validation error"):
        mod.build_artifact(
            repo_root=REPO,
            result_path=tmp_path / "bad-artifact.json",
            work_root=tmp_path / "bad-work",
            write=False,
            duration_s=1.0,
            tests_run=TESTS_RUN,
            run_date="20260823",
        )
