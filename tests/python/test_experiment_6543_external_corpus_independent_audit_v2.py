"""Tests for Exp6543 independent external corpus audit v2.

Spec refs: REQ-BENCH-6543, SCENARIO-BENCH-6543-MISSING,
SCENARIO-BENCH-6543-SOURCE, SCENARIO-BENCH-6543-CHRONOLOGY,
SCENARIO-BENCH-6543-SPLIT, SCENARIO-BENCH-6543-EXACT,
SCENARIO-BENCH-6543-TRANSACTION, SCENARIO-BENCH-6543-ATTACKS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6543_external_corpus_independent_audit_v2 as mod
from carnot.atomic_shard_transaction import AtomicShardTransaction, canonical_json_bytes


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6543_external_corpus_independent_audit_v2.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6543_external_corpus_independent_audit_v2.py "
    "-m pytest tests/python/test_experiment_6543_external_corpus_independent_audit_v2.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6543_external_corpus_independent_audit_v2.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6543_external_corpus_independent_audit_v2.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6543_external_corpus_independent_audit_v2.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6543_external_corpus_independent_audit_v2.json"
)
EXACT_E2E_COMMAND = ".venv/bin/pytest tests/python/test_z3_live_benchmark.py -q --no-cov -n 0"
CHECKSUM_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6543_external_corpus_independent_audit_v2 --validate"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6543_external_corpus_independent_audit_v2 "
    "--date 20260823"
)
GIT_STATUS_COMMAND = "git status --short"

TESTS_RUN = [
    {"command": FOCUSED_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": SPEC_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": EXACT_E2E_COMMAND, "exit_code": 0},
    {"command": CHECKSUM_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": GIT_STATUS_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
]

MINI_Z3 = """
from __future__ import annotations

from z3 import Int, Solver, sat, unsat


def build_domain_solver(domain, entities, constraints, context=None, extra_assignment=None):
    del domain, entities, context, extra_assignment
    solver = Solver()
    marker = Int("marker")
    solver.add(marker >= 0)
    for index, constraint in enumerate(constraints):
        solver.add(Int(f"c_{index}") == index)
        if constraint.get("type") == "force_unsat":
            solver.add(marker < 0)
    return solver, {}


def check_satisfiability(constraints, domain, entities, context=None):
    solver, _aux = build_domain_solver(domain, entities, constraints, context)
    result = solver.check()
    return {"is_sat": result == sat, "result": str(result)}


def verify_with_z3(answer, cumulative_constraints, domain, entities, context=None):
    del domain, context
    if not isinstance(answer, dict) or any(row.get("type") == "force_unsat" for row in cumulative_constraints):
        return 0
    return 1 if all(name in answer for name in entities) else 0


def compute_mus(constraints, domain, entities, context=None):
    del domain, entities, context
    return [row for row in constraints if row.get("type") == "force_unsat"]
"""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))


def _problem(problem_id: str, domain: str, turn_count: int) -> dict[str, Any]:
    entities = [f"{problem_id}_entity_{index}" for index in range(3)]
    turns = []
    cumulative: list[dict[str, Any]] = []
    for turn_number in range(1, turn_count + 1):
        constraint = {
            "type": "assign",
            "args": [entities[(turn_number - 1) % len(entities)], "slot", turn_number],
            "nl": f"{problem_id} turn {turn_number}",
        }
        cumulative.append(constraint)
        turns.append(
            {
                "turn_number": turn_number,
                "user_message": f"Turn {turn_number} for {problem_id}.",
                "new_constraints": [constraint],
                "cumulative_constraints": list(cumulative),
                "gold_solution": {name: {"slot": index + 1} for index, name in enumerate(entities)},
                "is_satisfiable": True,
            }
        )
    return {
        "problem_id": problem_id,
        "domain": domain,
        "split": "test",
        "num_entities": len(entities),
        "entities": entities,
        "turns": turns,
    }


def _write_source(tmp_path: Path) -> Path:
    root = tmp_path / "drift-bench"
    (root / "data" / "problems" / "test").mkdir(parents=True)
    (root / "src").mkdir(parents=True)
    (root / "README.md").write_text(
        "DRIFT-Bench. The original run's SQLite databases suffered filesystem corruption.\n",
        encoding="utf-8",
    )
    (root / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    (root / "data" / "problems" / "README.md").write_text(
        "Schema fields: problem_id, domain, split, entities, turns, cumulative_constraints.\n",
        encoding="utf-8",
    )
    (root / "src" / "z3_checker.py").write_text(MINI_Z3, encoding="utf-8")
    for problem in (
        _problem("logic_grid_001", "logic_grid", 2),
        _problem("scheduling_001", "scheduling", 2),
        _problem("seating_001", "seating", 2),
    ):
        _write_json(root / "data" / "problems" / "test" / f"{problem['problem_id']}.json", problem)
    return root


def _row(problem: dict[str, Any], relpath: str, source_root: Path, split_name: str) -> list[dict[str, Any]]:
    path = source_root / relpath
    rows = []
    for turn_index, turn in enumerate(problem["turns"]):
        source_turn_id = f"{problem['problem_id']}:turn:{turn_index + 1}"
        payload = {
            "source_problem_id": problem["problem_id"],
            "source_file_relpath": relpath,
            "turn_index": turn_index,
            "turn": turn,
        }
        exact_receipt = {
            "local_unit_id": f"audit_{problem['problem_id']}_{turn_index}",
            "source_turn_id": source_turn_id,
            "domain": problem["domain"],
            "split_name": split_name,
            "exact_label": "satisfiable",
            "satisfiable": True,
            "assignment_validity": True,
            "solver": "exp6543_independent_source_z3_construction",
            "solver_version": mod._z3_version(),
            "z3_checker_sha256": mod.sha256_file(source_root / "src" / "z3_checker.py"),
            "constraint_count": len(turn["cumulative_constraints"]),
            "solver_assertion_count": len(turn["cumulative_constraints"]) + 1,
            "timeout": False,
            "censored": False,
            "error": None,
            "terminal_status": "terminal",
        }
        rows.append(
            {
                "local_unit_id": exact_receipt["local_unit_id"],
                "source_problem_id": problem["problem_id"],
                "base_problem_id": problem["problem_id"],
                "source_turn_id": source_turn_id,
                "source_split": "test",
                "split_name": split_name,
                "domain": problem["domain"],
                "family": problem["domain"],
                "num_entities": problem["num_entities"],
                "turn_index": turn_index,
                "turn_number": turn_index + 1,
                "turn_position": "early" if turn_index == 0 else "late",
                "chronology_index": turn_index,
                "source_file_relpath": relpath,
                "source_file_sha256": mod.sha256_file(path),
                "source_problem_hash": mod.sha256_json(problem),
                "source_row_hash": mod.sha256_json(payload),
                "source_turn_sha256": mod.sha256_json(turn),
                "constraints_sha256": mod.sha256_json(turn["cumulative_constraints"]),
                "cumulative_constraint_count": len(turn["cumulative_constraints"]),
                "pre_replay_effort_stratum": "low" if turn_index == 0 else "medium",
                "user_message_sha256": mod.sha256_json(turn["user_message"]),
                "gold_solution_sha256": mod.sha256_json(turn["gold_solution"]),
                "exact_label": "satisfiable",
                "satisfiable": True,
                "assignment_validity": True,
                "solver_effort": {
                    "constraint_count": len(turn["cumulative_constraints"]),
                    "solver_assertion_count": len(turn["cumulative_constraints"]) + 1,
                    "z3_check_calls": 2,
                    "wall_time_s": 0.0,
                },
                "timeout": False,
                "censored": False,
                "terminal_status": "terminal",
                "exact_receipt_hash": mod.sha256_json(exact_receipt),
                "row_order_key_components": [
                    "split_name",
                    "domain",
                    "source_problem_id",
                    "turn_index",
                ],
                "upstream_sqlite_result_inherited": False,
                "paper_aggregate_inherited": False,
            }
        )
    return rows


def _fixture_rows(source_root: Path) -> list[dict[str, Any]]:
    specs = [
        ("logic_grid_001", "train"),
        ("scheduling_001", "development"),
        ("seating_001", "held"),
    ]
    rows: list[dict[str, Any]] = []
    for problem_id, split_name in specs:
        relpath = f"data/problems/test/{problem_id}.json"
        problem = json.loads((source_root / relpath).read_text(encoding="utf-8"))
        rows.extend(_row(problem, relpath, source_root, split_name))
    return rows


def _write_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    for relpath in [
        "CODEX.md",
        "CLAUDE.md",
        "research-roadmap.yaml",
        "openspec/change-proposals/research-roadmap-vNEXT.md",
        "ops/e2e-test-plan.md",
        "ops/exclusion_manifest.yaml",
        "scripts/research_conductor.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
        "results/experiment_6530_external_constraint_corpus_audit.json",
        "results/experiment_6541_v566_direct_source_contract.json",
        "results/experiment_6514_atomic_shard_artifact_transaction.json",
    ]:
        path = repo / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"protected {relpath}\n", encoding="utf-8")
    return repo


def _write_transaction(repo: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    fixture_path = repo / mod.FIXTURE_RELATIVE_PATH
    work_dir = repo / mod.WORK_RELATIVE_PATH
    with AtomicShardTransaction(
        work_dir=work_dir,
        final_path=fixture_path,
        transaction_id="test-exp6542-fixture",
        stale_lock_s=0.01,
    ) as tx:
        unit_ids = [row["local_unit_id"] for row in rows]
        tx.plan_units(unit_ids)
        shard_receipts = [tx.write_terminal_unit(row["local_unit_id"], row) for row in rows]
        state = tx.resume_state()
        final = tx._atomic_replace_final(b"".join(canonical_json_bytes(row) for row in rows))
        final.update(
            {
                "final_path": str(fixture_path),
                "final_sha256": mod.sha256_file(fixture_path),
                "row_count": len(rows),
            }
        )
        journal_rows = tx.read_journal()
    return {
        "transaction_schema": "carnot.atomic_shard_transaction.v1",
        "transaction_id": "test-exp6542-fixture",
        "work_dir": str(work_dir),
        "journal_path": str(work_dir / "journal.jsonl"),
        "journal_sha256": mod.sha256_file(work_dir / "journal.jsonl"),
        "journal_record_count": len(journal_rows),
        "planned_unit_ids": sorted(unit_ids),
        "terminal_unit_ids": sorted(row["unit_id"] for row in shard_receipts),
        "shards": [
            {
                "unit_id": receipt["unit_id"],
                "shard_hash": receipt["shard_hash"],
                "shard_path": receipt["shard_path"],
                "shard_path_is_content_addressed": True,
            }
            for receipt in shard_receipts
        ],
        "all_shards_verified": True,
        "resume_receipts": [
            {
                "verified": state["all_planned_terminal"],
                "missing_unit_ids": state["missing_unit_ids"],
                "terminal_unit_count": len(state["terminal_unit_ids"]),
            }
        ],
        "corrupt_resume_receipt": {"corrupt_resume_rejected": True, "corrupt_shard_rows": []},
        "corrupt_resume_rejected": True,
        "final_atomic_write_receipt": final,
        "fixture_roundtrip_row_count": len(rows),
        "fixture_roundtrip_hash": mod.sha256_json(rows),
    }


def _write_intake(repo: Path, source_root: Path, rows: list[dict[str, Any]], shard: dict[str, Any]) -> None:
    aggregate = {
        "fixture_row_count": len(rows),
        "base_problem_count": 3,
        "domain_counts": {"logic_grid": 2, "scheduling": 2, "seating": 2},
        "split_counts": {"development": 2, "held": 2, "train": 2},
        "exact_label_counts": {"satisfiable": 6},
        "ready_score_from_rows": 1.0,
    }
    _write_json(
        repo / mod.INTAKE_RELATIVE_PATH,
        {
            "status": "complete_drift_bench_external_intake_v2",
            "honest_verdict": "complete_drift_bench_external_intake_v2",
            "verdict_class": None,
            "external_constraint_corpus_ready_score": 1.0,
            "source_revision_and_license_receipt": {
                "source_root": str(source_root),
                "immutable_revision": mod.DRIFT_EXPECTED_COMMIT,
                "expected_revision": mod.DRIFT_EXPECTED_COMMIT,
                "revision_matches_expected": True,
                "revision_is_immutable": True,
                "commit_date": mod.DRIFT_EXPECTED_COMMIT_DATE,
                "commit_date_matches_expected": True,
                "license": "MIT",
                "license_verified": True,
                "data_schema_path": "data/problems/README.md",
                "data_schema_verified": True,
                "z3_replay_path": "src/z3_checker.py",
                "z3_replay_code_present": True,
                "problem_file_count": 3,
                "expected_problem_file_count": 3,
                "problem_file_count_matches_expected": True,
                "upstream_corruption_warning_present": True,
            },
            "source_tree_and_file_hashes": {
                "checkout_path": str(source_root),
                "repo_url": mod.DRIFT_REPO_URL,
                "problem_file_count": 3,
                "required_file_sha256": {
                    "README.md": mod.sha256_file(source_root / "README.md"),
                    "LICENSE": mod.sha256_file(source_root / "LICENSE"),
                    "data/problems/README.md": mod.sha256_file(
                        source_root / "data" / "problems" / "README.md"
                    ),
                    "src/z3_checker.py": mod.sha256_file(source_root / "src" / "z3_checker.py"),
                },
            },
            "fixture_path_and_hash": {
                "path": str(repo / mod.FIXTURE_RELATIVE_PATH),
                "exists": True,
                "sha256": mod.sha256_file(repo / mod.FIXTURE_RELATIVE_PATH),
                "row_count": len(rows),
                "expected_row_count": len(rows),
                "roundtrip_matches_expected": True,
                "roundtrip_sha256": mod.sha256_json(rows),
            },
            "split_commitment": {
                "split_counts": {"development": 2, "held": 2, "train": 2},
                "passed": True,
            },
            "shard_manifest": shard,
            "planned_and_terminal_unit_counts": {
                "planned_count": len(rows),
                "terminal_count": len(rows),
                "missing_count": 0,
                "all_planned_terminal": True,
            },
            "aggregate_row_recomputation": aggregate,
            "leakage_attack_matrix": [{"attack": "placeholder", "passed": True}],
            "tests_run": TESTS_RUN,
        },
    )


def _audit_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Any], list[dict[str, Any]]]:
    source_root = _write_source(tmp_path / "source")
    repo = _write_repo(tmp_path)
    rows = _fixture_rows(source_root)
    shard = _write_transaction(repo, rows)
    _write_intake(repo, source_root, rows, shard)
    artifact = mod.build_artifact(
        repo_root=repo,
        result_path=tmp_path / "artifact.json",
        source_root=source_root,
        expected_problem_file_count=3,
        run_date="20260823",
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        now_utc="2026-08-23T12:00:00Z",
    )
    return repo, source_root, artifact, rows


def test_req_bench_6543_spec_declares_independent_audit_contract() -> None:
    """REQ-BENCH-6543: OpenSpec owns the independent audit contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-BENCH-6543") :]

    for token in (
        "SCENARIO-BENCH-6543-MISSING",
        "SCENARIO-BENCH-6543-SOURCE",
        "SCENARIO-BENCH-6543-CHRONOLOGY",
        "SCENARIO-BENCH-6543-SPLIT",
        "SCENARIO-BENCH-6543-EXACT",
        "SCENARIO-BENCH-6543-TRANSACTION",
        "SCENARIO-BENCH-6543-ATTACKS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "external_constraint_corpus_audited_ready_score",
        mod.INFERENCE_SUBSTRATE,
    ):
        assert token in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenarios_bench_6543_complete_independent_audit_closes(
    tmp_path: Path,
) -> None:
    """SCENARIO-BENCH-6543-SOURCE/EXACT/SPLIT/TRANSACTION: clean audit closes."""

    repo, source_root, artifact, rows = _audit_fixture(tmp_path)
    written = json.loads((tmp_path / "artifact.json").read_text(encoding="utf-8"))

    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_external_corpus_independent_audit_v2"
    assert artifact["honest_verdict"].startswith("complete_external_corpus_independent_audit_v2")
    assert artifact["verdict_class"] is None
    assert artifact["external_constraint_corpus_audited_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True

    source_receipt = artifact["independent_revision_license_and_schema_receipt"]
    assert source_receipt["source_root"] == str(source_root)
    assert source_receipt["problem_file_count"] == 3
    assert source_receipt["license_verified"] is True
    assert source_receipt["schema_verified"] is True
    assert source_receipt["corruption_warning_verified"] is True

    assert len(artifact["source_identity_audit_rows"]) == len(rows)
    assert all(row["passed"] for row in artifact["source_identity_audit_rows"])
    assert all(row["chronology_valid"] for row in artifact["chronology_replay_rows"])
    assert all(row["replayed_label_matches"] for row in artifact["independent_exact_replay_rows"])
    assert artifact["split_and_lineage_audit"]["base_problem_overlap_count"] == 0
    assert artifact["shard_and_transaction_audit"]["passed"] is True
    assert artifact["missing_input_disposition"]["blocked"] is False
    assert artifact["gate_check_summary"]["all_gates_passed"] is True
    assert all(row["passed"] for row in artifact["leakage_attack_matrix"])
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert (repo / mod.FIXTURE_RELATIVE_PATH).is_file()


def test_scenario_bench_6543_missing_inputs_write_blocked_artifact(tmp_path: Path) -> None:
    """SCENARIO-BENCH-6543-MISSING: absent intake, fixture, and source fail closed."""

    repo = _write_repo(tmp_path)
    artifact = mod.build_artifact(
        repo_root=repo,
        result_path=tmp_path / "missing.json",
        source_root=tmp_path / "missing-source",
        expected_problem_file_count=3,
        run_date="20260823",
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        now_utc="2026-08-23T12:00:00Z",
    )

    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "blocked_external_corpus_independent_audit_v2"
    assert artifact["honest_verdict"].startswith("blocked_external_corpus_independent_audit_v2")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["external_constraint_corpus_audited_ready_score"] == 0.0
    missing = artifact["missing_input_disposition"]
    assert missing["blocked"] is True
    assert {"intake_artifact", "fixture", "source_root"} <= {
        row["input"] for row in missing["missing_inputs"]
    }
    assert {row["check"] for row in artifact["gate_check_summary"]["failed_checks"]} >= {
        "intake_artifact_exists",
        "fixture_exists",
        "source_root_exists",
    }


def test_scenario_bench_6543_attack_helpers_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-BENCH-6543-ATTACKS: tampering, gaps, aliases, and aggregates fail."""

    repo, source_root, artifact, rows = _audit_fixture(tmp_path)
    bad_rows = deepcopy(rows)
    bad_rows[1]["turn_index"] = 3
    bad_rows[1]["chronology_index"] = 3
    bad_rows[1]["source_turn_id"] = bad_rows[0]["source_turn_id"]
    bad_rows[1]["local_unit_id"] = bad_rows[0]["local_unit_id"]
    bad_rows[1]["split_name"] = "held"
    bad_rows[1]["exact_label"] = "contradiction"
    bad_rows[1]["source_row_hash"] = "sha256:bad"
    bad_rows[2]["family"] = "Logic Grid"
    bad_rows[3]["family"] = "logic-grid"
    bad_rows[4]["post_held_repair"] = True
    bad_rows[5]["sampled_after_outcome"] = True
    _write_jsonl(repo / mod.FIXTURE_RELATIVE_PATH, bad_rows)

    source_identity = mod.source_identity_audit_rows(
        fixture_rows=bad_rows,
        source_root=source_root,
    )
    chronology = mod.chronology_replay_rows(bad_rows)
    exact = mod.independent_exact_replay_rows(
        fixture_rows=bad_rows,
        source_root=source_root,
        sample_seed=mod.RANDOM_SEED,
    )
    split = mod.split_and_lineage_audit(bad_rows, chronology)
    shard = mod.shard_and_transaction_audit(
        repo_root=repo,
        intake=artifact,
        fixture_path=repo / mod.FIXTURE_RELATIVE_PATH,
        fixture_rows=bad_rows,
    )
    aggregates = mod.independent_aggregate_rows(
        fixture_rows=bad_rows,
        exact_rows=exact,
        intake={"aggregate_row_recomputation": {"fixture_row_count": 999}},
    )
    attacks = mod.leakage_attack_matrix(
        fixture_rows=bad_rows,
        source_identity_rows=source_identity,
        chronology_rows=chronology,
        exact_rows=exact,
        split_audit=split,
        shard_audit=shard,
        aggregate_rows=aggregates,
        revision_receipt={"solver_path_identity_ok": False},
    )
    aggregate = mod.aggregate_row_recomputation(
        fixture_rows=bad_rows,
        source_identity_rows=source_identity,
        chronology_rows=chronology,
        split_audit=split,
        exact_rows=exact,
        shard_audit=shard,
        aggregate_rows=aggregates,
        attack_rows=attacks,
        revision_receipt={
            "revision_matches_expected": True,
            "license_verified": True,
            "schema_verified": True,
            "corruption_warning_verified": True,
            "problem_file_count_matches_expected": True,
            "z3_replay_code_present": True,
        },
        missing_input_disposition={"blocked": False},
        protected={"all_protected_files_unchanged": True},
    )
    gate = mod.gate_check_summary(
        missing_input_disposition={"blocked": False, "missing_inputs": []},
        revision_receipt={
            "revision_matches_expected": True,
            "license_verified": True,
            "schema_verified": True,
            "corruption_warning_verified": True,
            "problem_file_count_matches_expected": True,
            "z3_replay_code_present": True,
        },
        source_identity_rows=source_identity,
        chronology_rows=chronology,
        split_audit=split,
        exact_rows=exact,
        shard_audit=shard,
        aggregate_rows=aggregates,
        attack_rows=attacks,
        aggregate=aggregate,
        protected={"all_protected_files_unchanged": True},
    )

    assert any(row["passed"] is False for row in source_identity)
    assert any(row["chronology_valid"] is False for row in chronology)
    assert split["base_problem_overlap_count"] >= 1
    assert split["family_alias_collision_count"] >= 1
    assert split["post_held_repair_count"] == 1
    assert split["outcome_based_sampling_count"] == 1
    assert any(row["replayed_label_matches"] is False for row in exact)
    assert any(row["intake_matches"] is False for row in aggregates)
    assert any(row["passed"] is False for row in attacks)
    assert aggregate["ready_score_from_rows"] == 0.0
    assert gate["all_gates_passed"] is False
    assert {row["check"] for row in gate["failed_checks"]} >= {
        "source_identity",
        "chronology",
        "split_lineage",
        "exact_replay",
        "shard_transaction",
        "aggregate_tampering",
        "leakage_attacks",
    }


def test_scenario_bench_6543_validation_cli_and_edges(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-BENCH-6543-ATTACKS: validators, helpers, and CLI fail closed."""

    repo, source_root, artifact, rows = _audit_fixture(tmp_path)

    assert mod._utc_now().endswith("Z")
    assert mod.sha256_file(tmp_path / "missing") == "missing"
    assert mod._load_json(tmp_path / "missing.json") == {}
    assert mod._load_jsonl(tmp_path / "missing.jsonl") == []
    assert mod._tests_run_receipts(TESTS_RUN) == TESTS_RUN
    assert all(row["exit_code"] == 0 for row in mod._tests_run_receipts(None))
    assert mod._safe_int("bad", default=9) == 9
    assert mod._normalize_alias("Logic Grid!") == "logicgrid"
    assert mod._source_turn_id("p", 0) == "p:turn:1"
    assert mod._label_from_solver_result(True, False, None) == "satisfiable"
    assert mod._label_from_solver_result(False, False, None) == "contradiction"
    assert mod._label_from_solver_result(False, True, None) == "timeout"
    assert mod._label_from_solver_result(False, False, "error") == "error"
    assert mod._status_for_class("partial") == "partial_external_corpus_independent_audit_v2"
    assert mod._status_for_class("disqualified") == "disqualified_external_corpus_independent_audit_v2"
    assert mod._status_for_class("blocked") == "blocked_external_corpus_independent_audit_v2"
    assert mod._verdict_class({"all_gates_passed": True, "failed_checks": []}) is None
    assert mod._verdict_class(
        {"all_gates_passed": False, "failed_checks": [{"severity": "partial"}]}
    ) == "partial"
    assert mod._verdict_class(
        {"all_gates_passed": False, "failed_checks": [{"severity": "disqualified"}]}
    ) == "disqualified"

    missing = mod.missing_input_disposition(
        intake_path=tmp_path / "no.json",
        fixture_path=tmp_path / "no.jsonl",
        source_root=tmp_path / "no-source",
        fixture_rows=[],
    )
    assert missing["blocked"] is True
    assert mod.source_identity_audit_rows(fixture_rows=[], source_root=source_root) == []
    assert mod.chronology_replay_rows([]) == []
    assert (
        mod.independent_exact_replay_rows(
            fixture_rows=[],
            source_root=source_root,
            sample_seed=mod.RANDOM_SEED,
        )
        == []
    )
    assert mod.split_and_lineage_audit([], [])["passed"] is False
    assert mod._problem_file_count(source_root) == 3
    assert mod._source_root_from_intake({"source_tree_and_file_hashes": {"checkout_path": str(source_root)}}) == source_root
    assert mod._source_root_from_intake({}) is None
    assert mod._context_from_problem({"domain": "other"}) == {}

    blank_jsonl = tmp_path / "blank.jsonl"
    blank_jsonl.write_text("\n{}\n", encoding="utf-8")
    assert mod._load_jsonl(blank_jsonl) == [{}]

    git_root = tmp_path / "git-source"
    (git_root / ".git").mkdir(parents=True)

    class Completed:
        stdout = "abc123\n"

    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: Completed())
    assert mod._run_git(git_root, ["rev-parse", "HEAD"]) == "abc123"

    def _raise_run(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise mod.subprocess.SubprocessError("boom")

    monkeypatch.setattr(mod.subprocess, "run", _raise_run)
    assert mod._run_git(git_root, ["rev-parse", "HEAD"]) is None
    assert mod._git_state(repo)["status_short"] == "unavailable"
    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: Completed())

    monkeypatch.setenv("CARNOT_EXP6543_DRIFT_SOURCE_ROOT", str(source_root))
    assert mod._resolve_source_root(intake={}, source_root=None) == source_root
    monkeypatch.delenv("CARNOT_EXP6543_DRIFT_SOURCE_ROOT")
    assert mod._resolve_source_root(intake={}, source_root=None) == mod.DEFAULT_SOURCE_CACHE_ROOT

    monkeypatch.setattr(mod.importlib.util, "spec_from_file_location", lambda *args, **kwargs: None)
    assert mod._load_z3_checker(source_root) is None
    monkeypatch.undo()

    empty_fixture = tmp_path / "empty.jsonl"
    empty_fixture.write_text("", encoding="utf-8")
    empty_missing = mod.missing_input_disposition(
        intake_path=repo / mod.INTAKE_RELATIVE_PATH,
        fixture_path=empty_fixture,
        source_root=source_root,
        fixture_rows=[],
    )
    assert empty_missing["blocked"] is True
    assert any(row["input"] == "empty_success_labeled_fixture" for row in empty_missing["missing_inputs"])

    existence_with_blank = mod.source_existence_and_hash_receipts(
        repo_root=repo,
        intake_path=repo / mod.INTAKE_RELATIVE_PATH,
        fixture_path=repo / mod.FIXTURE_RELATIVE_PATH,
        source_root=source_root,
        fixture_rows=[{"source_file_relpath": ""}],
        before_protected={},
    )
    assert existence_with_blank["source_files"] == []

    missing_terminal_rows = deepcopy(rows)
    missing_terminal_rows[0]["terminal_status"] = "terminal_timeout"
    assert mod.split_and_lineage_audit(missing_terminal_rows, mod.chronology_replay_rows(missing_terminal_rows))[
        "missing_terminal_count"
    ] == 1
    assert mod._solver_assertion_count(None, domain="x", entities=[], constraints=[{}], context={}) == 1

    no_checker_source = tmp_path / "no-checker"
    (no_checker_source / "data" / "problems" / "test").mkdir(parents=True)
    row0 = deepcopy(rows[0])
    problem0 = json.loads((source_root / row0["source_file_relpath"]).read_text(encoding="utf-8"))
    _write_json(no_checker_source / row0["source_file_relpath"], problem0)
    unavailable = mod.independent_exact_replay_rows(
        fixture_rows=[row0],
        source_root=no_checker_source,
        sample_seed=mod.RANDOM_SEED,
    )[0]
    assert unavailable["recomputed_exact_label"] == "error"

    unsat_source = _write_source(tmp_path / "unsat-source")
    unsat_problem = json.loads(
        (unsat_source / "data/problems/test/logic_grid_001.json").read_text(encoding="utf-8")
    )
    unsat_problem["turns"][0]["cumulative_constraints"] = [{"type": "force_unsat"}]
    unsat_problem["turns"][0]["gold_solution"] = {}
    _write_json(unsat_source / "data/problems/test/logic_grid_001.json", unsat_problem)
    unsat_row = _row(
        unsat_problem,
        "data/problems/test/logic_grid_001.json",
        unsat_source,
        "train",
    )[0]
    unsat_row["exact_label"] = "contradiction"
    unsat_row["satisfiable"] = False
    unsat_row["assignment_validity"] = False
    unsat_exact = mod.independent_exact_replay_rows(
        fixture_rows=[unsat_row],
        source_root=unsat_source,
        sample_seed=mod.RANDOM_SEED,
    )[0]
    assert unsat_exact["conflict_or_mus_evidence"]["available"] is True

    timeout_source = _write_source(tmp_path / "timeout-source")
    timeout_checker = (timeout_source / "src" / "z3_checker.py")
    timeout_checker.write_text(
        MINI_Z3
        + "\n\ndef check_satisfiability(constraints, domain, entities, context=None):\n"
        + "    raise TimeoutError('timeout')\n",
        encoding="utf-8",
    )
    timeout_row = deepcopy(rows[0])
    timeout_row["source_file_relpath"] = "data/problems/test/logic_grid_001.json"
    timeout_row["exact_label"] = "timeout"
    timeout_row["satisfiable"] = False
    timeout_row["assignment_validity"] = False
    timeout_row["terminal_status"] = "terminal_timeout"
    timeout_exact = mod.independent_exact_replay_rows(
        fixture_rows=[timeout_row],
        source_root=timeout_source,
        sample_seed=mod.RANDOM_SEED,
    )[0]
    assert timeout_exact["recomputed_exact_label"] == "timeout"

    journal = tmp_path / "journal.jsonl"
    good_record = {
        "schema": "carnot.atomic_shard_transaction.v1",
        "record_type": "planned_unit",
        "unit_id": "u",
    }
    good_record["record_hash"] = mod._atomic_sha256_json(good_record)
    journal.write_text("\n" + json.dumps(good_record, sort_keys=True) + "\n", encoding="utf-8")
    assert len(mod._read_journal_rows(journal)) == 1
    assert mod._journal_hashes_valid([{}]) is False
    assert mod._journal_hashes_valid([{"record_hash": "bad"}]) is False

    relative_shard = deepcopy(artifact)
    first_shard = relative_shard["shard_and_transaction_audit"]["shard_rows"][0]
    rel_manifest = deepcopy(json.loads((repo / mod.INTAKE_RELATIVE_PATH).read_text(encoding="utf-8")))
    rel_manifest["shard_manifest"]["shards"] = [None, deepcopy(rel_manifest["shard_manifest"]["shards"][0])]
    abs_path = Path(rel_manifest["shard_manifest"]["shards"][1]["shard_path"])
    rel_manifest["shard_manifest"]["shards"][1]["shard_path"] = str(abs_path.relative_to(repo))
    rel_audit = mod.shard_and_transaction_audit(
        repo_root=repo,
        intake=rel_manifest,
        fixture_path=repo / mod.FIXTURE_RELATIVE_PATH,
        fixture_rows=rows,
    )
    assert rel_audit["shard_rows"][0]["path"].endswith(first_shard["path"].split("/")[-1])

    bad = deepcopy(artifact)
    bad.pop("status")
    bad["field_principles"] = {}
    bad["field_provenance"] = {}
    bad["verdict_class"] = "positive"
    bad["honest_verdict"] = "ready"
    bad["inference_substrate"] = "llm"
    bad["verifier_is_oracle"] = False
    bad["external_constraint_corpus_audited_ready_score"] = 0.5
    bad["reproducibility_checksum"] = "bad"
    errors = mod.validate_artifact(bad)
    assert "required field set mismatch" in errors
    assert "field_principles mismatch" in errors
    assert "field_provenance must cover required fields" in errors
    assert "verdict_class outside Exp6543 enum" in errors
    assert "honest_verdict terminal prefix mismatch" in errors
    assert "inference_substrate mismatch" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "ready score mismatch" in errors
    assert "reproducibility_checksum mismatch" in errors

    bad_ready = deepcopy(artifact)
    bad_ready["gate_check_summary"] = {"all_gates_passed": False, "failed_checks": []}
    bad_ready["reproducibility_checksum"] = mod.reproducibility_checksum(bad_ready)
    assert "ready artifact must have all gates passed" in mod.validate_artifact(bad_ready)

    bad_blocked = deepcopy(artifact)
    bad_blocked["external_constraint_corpus_audited_ready_score"] = 0.0
    bad_blocked["aggregate_row_recomputation"]["ready_score_from_rows"] = 0.0
    bad_blocked["gate_check_summary"] = {"all_gates_passed": True, "failed_checks": []}
    bad_blocked["reproducibility_checksum"] = mod.reproducibility_checksum(bad_blocked)
    assert "blocked artifact cannot have all gates passed" in mod.validate_artifact(bad_blocked)

    scalar_gate = mod.gate_check_summary(
        missing_input_disposition={"blocked": False, "missing_inputs": []},
        revision_receipt={
            "revision_matches_expected": True,
            "revision_is_immutable": True,
            "license_verified": True,
            "schema_verified": True,
            "z3_replay_code_present": True,
            "problem_file_count_matches_expected": True,
            "corruption_warning_verified": True,
            "solver_path_identity_ok": True,
        },
        source_identity_rows=artifact["source_identity_audit_rows"],
        chronology_rows=artifact["chronology_replay_rows"],
        split_audit=artifact["split_and_lineage_audit"],
        exact_rows=artifact["independent_exact_replay_rows"],
        shard_audit=artifact["shard_and_transaction_audit"],
        aggregate_rows=artifact["independent_aggregate_rows"],
        attack_rows=artifact["leakage_attack_matrix"],
        aggregate={"ready_score_from_rows": 0.5},
        protected={"all_protected_files_unchanged": False},
    )
    assert {row["check"] for row in scalar_gate["failed_checks"]} == {
        "protected_files_unchanged",
        "ready_score_scalar",
    }
    assert mod._verdict_class({"all_gates_passed": False, "failed_checks": [{"severity": "other"}]}) == "blocked"

    monkeypatch.setattr(mod, "REPO_ROOT", repo)
    cli_path = tmp_path / "cli.json"
    assert (
        mod.main(
            [
                "--date",
                "20260823",
                "--result-path",
                str(cli_path),
                "--source-root",
                str(source_root),
                "--expected-problem-file-count",
                "3",
            ]
        )
        == 0
    )
    assert f"wrote {mod.RESULT_RELATIVE_PATH.as_posix()}" in capsys.readouterr().out
    assert mod.main(["--result-path", str(cli_path), "--validate"]) == 0
    assert f"validated {mod.RESULT_RELATIVE_PATH.as_posix()}" in capsys.readouterr().out
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    assert mod.main(["--result-path", str(invalid), "--validate"]) == 1
    assert "required field set mismatch" in capsys.readouterr().out

    monkeypatch.setattr(mod, "build_artifact", lambda **kwargs: {"bad": True})
    monkeypatch.setattr(mod, "validate_artifact", lambda artifact: ["forced error"])
    assert mod.main(["--result-path", str(tmp_path / "forced.json")]) == 1
    assert "forced error" in capsys.readouterr().out

    assert rows
