"""Tests for Exp6542 content-pinned DRIFT-Bench external intake v2.

Spec refs: REQ-BENCH-6542, SCENARIO-BENCH-6542-SOURCE,
SCENARIO-BENCH-6542-CHRONOLOGY, SCENARIO-BENCH-6542-EXACT,
SCENARIO-BENCH-6542-SPLIT, SCENARIO-BENCH-6542-SHARDS,
SCENARIO-BENCH-6542-CENSORING, SCENARIO-BENCH-6542-ATTACKS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6542_drift_bench_external_intake_v2 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6542_drift_bench_external_intake_v2.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6542_drift_bench_external_intake_v2.py "
    "-m pytest tests/python/test_experiment_6542_drift_bench_external_intake_v2.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6542_drift_bench_external_intake_v2.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6542_drift_bench_external_intake_v2.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6542_drift_bench_external_intake_v2.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6542_drift_bench_external_intake_v2.json"
)
EXACT_E2E_COMMAND = ".venv/bin/pytest tests/python/test_z3_live_benchmark.py -q --no-cov -n 0"
CHECKSUM_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6542_drift_bench_external_intake_v2 --validate"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6542_drift_bench_external_intake_v2 "
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


def _status(constraints):
    return unsat if any(item.get("type") == "force_unsat" for item in constraints) else sat


def build_domain_solver(domain, entities, constraints, context=None, extra_assignment=None):
    solver = Solver()
    solver.set("timeout", 30000)
    marker = Int("marker")
    solver.add(marker >= 0)
    for index, _constraint in enumerate(constraints):
        solver.add(Int(f"c_{index}") == index)
    if _status(constraints) == unsat:
        solver.add(marker < 0)
    return solver, {"domain": domain, "entities": entities, "context": context or {}, "answer": extra_assignment}


def check_satisfiability(constraints, domain, entities, context=None):
    solver, _aux = build_domain_solver(domain, entities, constraints, context)
    result = solver.check()
    return {"is_sat": result == sat, "solution": {name: index + 1 for index, name in enumerate(entities)} if result == sat else None}


def verify_with_z3(answer, cumulative_constraints, domain, entities, context=None):
    del domain, context
    if not isinstance(answer, dict):
        return 0
    if _status(cumulative_constraints) == unsat:
        return 0
    return 1 if all(name in answer for name in entities) else 0


def compute_mus(constraints, domain, entities, context=None):
    del domain, entities, context
    return [item for item in constraints if item.get("type") == "force_unsat"]
"""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _problem(
    *,
    domain: str,
    source_split: str,
    index: int,
    size: int,
    turn_count: int,
) -> dict[str, Any]:
    entities = [f"{domain}_entity_{index}_{i}" for i in range(size)]
    problem: dict[str, Any] = {
        "problem_id": f"{domain}_{index:03d}",
        "domain": domain,
        "split": source_split,
        "num_entities": size,
        "entities": entities,
        "turns": [],
    }
    if domain == "logic_grid":
        values = [f"value_{i}" for i in range(size)]
        problem["categories"] = {"color": values}
    if domain == "scheduling":
        problem["num_slots"] = max(6, size + 3)
        problem["max_duration"] = 3
    if domain == "seating":
        problem["table_shape"] = "round"

    cumulative: list[dict[str, Any]] = []
    for turn_number in range(1, turn_count + 1):
        entity = entities[(turn_number - 1) % len(entities)]
        if domain == "logic_grid":
            constraint = {
                "type": "assign",
                "args": [entity, "color", f"value_{(turn_number - 1) % size}"],
                "nl": f"{entity} has fixed color",
            }
            solution = {name: {"color": f"value_{i % size}"} for i, name in enumerate(entities)}
        elif domain == "scheduling":
            constraint = {
                "type": "duration",
                "args": [entity, 1],
                "nl": f"{entity} has duration 1",
            }
            solution = {name: {"start": 1, "duration": 1} for name in entities}
        else:
            constraint = {
                "type": "at_position",
                "args": [entity, ((turn_number - 1) % size) + 1],
                "nl": f"{entity} sits at a fixed position",
            }
            solution = {name: i + 1 for i, name in enumerate(entities)}
        cumulative.append(constraint)
        problem["turns"].append(
            {
                "turn_number": turn_number,
                "user_message": f"Turn {turn_number} for {domain} {index}.",
                "new_constraints": [constraint],
                "cumulative_constraints": list(cumulative),
                "gold_solution": solution,
                "is_satisfiable": True,
            }
        )
    return problem


def _write_source(tmp_path: Path) -> Path:
    root = tmp_path / "drift-bench"
    (root / "data" / "problems" / "dev").mkdir(parents=True)
    (root / "data" / "problems" / "test").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "docs").mkdir()
    (root / "README.md").write_text(
        "DRIFT-Bench. The original run's SQLite databases suffered filesystem corruption.\n",
        encoding="utf-8",
    )
    (root / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    (root / "data" / "problems" / "README.md").write_text(
        "Schema fields: problem_id, domain, split, entities, turns, "
        "cumulative_constraints, gold_solution, is_satisfiable.\n",
        encoding="utf-8",
    )
    (root / "docs" / "prompts.md").write_text("Prompt templates\n", encoding="utf-8")
    (root / "src" / "prompts.py").write_text("SYSTEM_PROMPTS = {}\n", encoding="utf-8")
    (root / "src" / "z3_checker.py").write_text(MINI_Z3, encoding="utf-8")

    specs = [
        ("logic_grid", 4, 4),
        ("logic_grid", 4, 6),
        ("logic_grid", 4, 8),
        ("scheduling", 5, 4),
        ("scheduling", 6, 6),
        ("scheduling", 7, 8),
        ("seating", 6, 4),
        ("seating", 7, 6),
        ("seating", 8, 8),
    ]
    for index, (domain, size, turn_count) in enumerate(specs):
        source_split = "dev" if index % 3 == 0 else "test"
        problem = _problem(
            domain=domain,
            source_split=source_split,
            index=index,
            size=size,
            turn_count=turn_count,
        )
        _write_json(
            root / "data" / "problems" / source_split / f"{problem['problem_id']}.json",
            problem,
        )
    return root


def _write_repo(tmp_path: Path, source_root: Path, *, gate_score: float = 1.0) -> Path:
    repo = tmp_path / "repo"
    for relpath in [
        "CODEX.md",
        "CLAUDE.md",
        "research-program.md",
        "research-roadmap.yaml",
        "scripts/research_conductor.py",
        "results/experiment_6514_atomic_shard_artifact_transaction.json",
    ]:
        path = repo / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"protected {relpath}\n", encoding="utf-8")
    exp6541 = {
        "status": "complete_v566_direct_source_contract_ready",
        "honest_verdict": "complete_v566_direct_source_contract_ready",
        "v566_direct_source_ready_score": gate_score,
        "drift_revision_license_schema_contract": {
            "immutable_revision": mod.DRIFT_EXPECTED_COMMIT,
            "license": "MIT",
            "z3_replay_path": "src/z3_checker.py",
        },
        "source_tree_hashes": {
            "root_tree_git_sha": "fixture-root-tree",
            "problems_tree_git_sha": "fixture-problems-tree",
            "required_file_sha256": {
                "README.md": mod.sha256_file(source_root / "README.md"),
                "LICENSE": mod.sha256_file(source_root / "LICENSE"),
                "data/problems/README.md": mod.sha256_file(
                    source_root / "data" / "problems" / "README.md"
                ),
                "src/z3_checker.py": mod.sha256_file(source_root / "src" / "z3_checker.py"),
            },
        },
        "upstream_corruption_boundary": {
            "sqlite_corruption_warning_present": True,
            "upstream_sqlite_results_inherited": False,
        },
    }
    _write_json(repo / mod.UPSTREAM_GATE_RELATIVE_PATH, exp6541)
    return repo


def _metadata(source_root: Path) -> dict[str, Any]:
    return {
        "repo_url": mod.DRIFT_REPO_URL,
        "commit": mod.DRIFT_EXPECTED_COMMIT,
        "commit_date": mod.DRIFT_EXPECTED_COMMIT_DATE,
        "commit_subject": "fixture commit",
        "root_tree_git_sha": "fixture-root-tree",
        "problems_tree_git_sha": "fixture-problems-tree",
        "checkout_path": str(source_root),
        "ls_remote_head": mod.DRIFT_EXPECTED_COMMIT,
    }


def _artifact(tmp_path: Path, *, gate_score: float = 1.0) -> dict[str, Any]:
    source = _write_source(tmp_path / "source")
    repo = _write_repo(tmp_path, source, gate_score=gate_score)
    return mod.build_artifact(
        repo_root=repo,
        result_path=tmp_path / "artifact.json",
        fixture_path=tmp_path / "fixture.jsonl",
        transaction_work_dir=tmp_path / "tx",
        drift_source_root=source,
        drift_git_metadata=_metadata(source),
        expected_problem_file_count=9,
        fixture_bound=9,
        run_date="20260823",
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        now_utc="2026-08-23T12:00:00Z",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_req_bench_6542_spec_declares_intake_contract() -> None:
    """REQ-BENCH-6542: OpenSpec owns the Exp6542 intake contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-BENCH-6542") :]

    for token in (
        "SCENARIO-BENCH-6542-SOURCE",
        "SCENARIO-BENCH-6542-CHRONOLOGY",
        "SCENARIO-BENCH-6542-EXACT",
        "SCENARIO-BENCH-6542-SPLIT",
        "SCENARIO-BENCH-6542-SHARDS",
        "SCENARIO-BENCH-6542-CENSORING",
        "SCENARIO-BENCH-6542-ATTACKS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.FIXTURE_RELATIVE_PATH.as_posix(),
        "external_constraint_corpus_ready_score",
        mod.INFERENCE_SUBSTRATE,
    ):
        assert token in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenarios_bench_6542_complete_intake_writes_fixture_and_receipts(
    tmp_path: Path,
) -> None:
    """SCENARIO-BENCH-6542-SOURCE/CHRONOLOGY/EXACT/SPLIT/SHARDS: fixture closes."""

    artifact = _artifact(tmp_path)
    fixture_rows = _read_jsonl(tmp_path / "fixture.jsonl")
    written = json.loads((tmp_path / "artifact.json").read_text(encoding="utf-8"))

    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_drift_bench_external_intake_v2"
    assert artifact["honest_verdict"].startswith("complete_drift_bench_external_intake_v2")
    assert artifact["verdict_class"] is None
    assert artifact["external_constraint_corpus_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True

    assert len(fixture_rows) == artifact["fixture_path_and_hash"]["row_count"]
    assert artifact["fixture_path_and_hash"]["sha256"] == mod.sha256_file(
        tmp_path / "fixture.jsonl"
    )
    assert artifact["planned_and_terminal_unit_counts"] == {
        "planned_count": len(fixture_rows),
        "terminal_count": len(fixture_rows),
        "missing_count": 0,
        "all_planned_terminal": True,
    }
    assert artifact["shard_manifest"]["final_atomic_write_receipt"]["atomic_replace"] is True
    assert all(row["terminal_status"] == "terminal" for row in fixture_rows)
    assert all(row["exact_label"] == "satisfiable" for row in fixture_rows)
    assert all(row["assignment_validity"] is True for row in fixture_rows)
    assert all(row["source_row_hash"].startswith("sha256:") for row in fixture_rows)
    assert all(row["local_unit_id"] != row["source_turn_id"] for row in fixture_rows)
    assert {row["domain"] for row in fixture_rows} == {"logic_grid", "scheduling", "seating"}
    assert {row["split_name"] for row in fixture_rows} == {"development", "held", "train"}

    split = artifact["split_commitment"]
    assert split["base_problem_overlap_count"] == 0
    assert split["family_alias_collision_count"] == 0
    assert split["lineage_floor_held"] is True
    assert artifact["family_turn_and_effort_census"]["balanced_domains"] is True
    assert artifact["family_turn_and_effort_census"]["multiple_sizes_where_available"] is True
    assert set(artifact["family_turn_and_effort_census"]["effort_strata_counts"]) >= {
        "high",
        "low",
        "medium",
    }
    assert all(row["passed"] is True for row in artifact["leakage_attack_matrix"])
    assert not artifact["upstream_corruption_boundary"]["paper_aggregate_claims_inherited"]
    assert artifact["gate_check_summary"]["all_gates_passed"] is True
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)


def test_scenario_bench_6542_blocked_gate_writes_closed_artifact(tmp_path: Path) -> None:
    """SCENARIO-BENCH-6542-CENSORING: failed gates name observed values and stop."""

    artifact = _artifact(tmp_path, gate_score=0.0)

    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "blocked_drift_bench_external_intake_v2"
    assert artifact["honest_verdict"].startswith("blocked_drift_bench_external_intake_v2")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["external_constraint_corpus_ready_score"] == 0.0
    assert artifact["upstream_gate_receipt"]["expected"] == 1.0
    assert artifact["upstream_gate_receipt"]["observed"] == 0.0
    failures = artifact["gate_check_summary"]["failed_checks"]
    assert any(row["check"] == "upstream_gate" and row["observed"] == 0.0 for row in failures)
    assert artifact["planned_and_terminal_unit_counts"]["planned_count"] == 0
    assert artifact["fixture_path_and_hash"]["exists"] is False

    source = _write_source(tmp_path / "bad-source")
    repo = _write_repo(tmp_path / "bad-repo", source, gate_score=1.0)
    bad_source = mod.build_artifact(
        repo_root=repo,
        result_path=tmp_path / "bad-source-artifact.json",
        fixture_path=tmp_path / "bad-source-fixture.jsonl",
        transaction_work_dir=tmp_path / "bad-source-tx",
        drift_source_root=source,
        drift_git_metadata=_metadata(source),
        expected_problem_file_count=10,
        fixture_bound=9,
        run_date="20260823",
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        now_utc="2026-08-23T12:00:00Z",
    )
    assert bad_source["status"] == "blocked_drift_bench_external_intake_v2"
    assert any(
        row["check"] == "problem_file_count_matches_expected"
        for row in bad_source["gate_check_summary"]["failed_checks"]
    )


def test_scenario_bench_6542_attack_helpers_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-BENCH-6542-ATTACKS: tampering, gaps, and inherited totals are detected."""

    artifact = _artifact(tmp_path)
    rows = _read_jsonl(tmp_path / "fixture.jsonl")
    replay_rows = deepcopy(artifact["exact_replay_rows"])
    identity_rows = deepcopy(artifact["source_to_local_identity_rows"])

    bad_rows = deepcopy(rows)
    bad_rows[1]["base_problem_id"] = bad_rows[0]["base_problem_id"]
    bad_rows[1]["turn_index"] = bad_rows[0]["turn_index"] + 2
    bad_rows[1]["source_turn_id"] = bad_rows[0]["source_turn_id"]
    bad_rows[1]["split_name"] = "held" if bad_rows[0]["split_name"] != "held" else "train"
    bad_rows[1]["family"] = "Logic Grid"
    bad_rows[2]["family"] = "logic-grid"
    bad_rows[2]["terminal_status"] = ""
    bad_rows[3]["censored"] = True

    split = mod.build_split_commitment(bad_rows)
    assert split["base_problem_overlap_count"] >= 1
    assert split["chronology_gap_count"] >= 1
    assert split["duplicate_turn_count"] >= 1
    assert split["family_alias_collision_count"] >= 1
    assert split["missing_terminal_count"] >= 1
    assert split["censored_count"] == 1
    assert split["passed"] is False

    identity_rows[0]["source_file_hash_matches"] = False
    replay_rows[0]["solver_version_matches"] = False
    replay_rows[1]["terminal_status"] = "missing"
    aggregate = mod.aggregate_row_recomputation(
        fixture_rows=bad_rows,
        identity_rows=identity_rows,
        exact_rows=replay_rows,
        split_commitment=split,
        shard_manifest={"all_shards_verified": False, "corrupt_resume_rejected": False},
        protected={"all_protected_files_unchanged": False},
        upstream_gate={"passed": True},
        source_receipt={
            "revision_matches_expected": True,
            "license_verified": True,
            "data_schema_verified": True,
            "z3_replay_code_present": True,
            "problem_file_count_matches_expected": True,
            "upstream_corruption_warning_present": True,
        },
        inherited_aggregate_present=True,
    )
    attacks = mod.leakage_attack_matrix(
        fixture_rows=bad_rows,
        identity_rows=identity_rows,
        exact_rows=replay_rows,
        split_commitment=split,
        shard_manifest={"all_shards_verified": False, "corrupt_resume_rejected": False},
        aggregate=aggregate,
        source_hashes_match=False,
        inherited_aggregate_present=True,
    )
    assert {row["attack"] for row in attacks} >= {
        "duplicate_turn_attack",
        "chronology_gap_attack",
        "family_alias_attack",
        "entity_name_leakage_attack",
        "row_order_leakage_attack",
        "source_hash_mismatch_attack",
        "solver_version_drift_attack",
        "corrupt_resume_attack",
        "missing_terminal_units_attack",
        "inherited_aggregate_attack",
    }
    assert any(row["passed"] is False for row in attacks)
    gate = mod.gate_check_summary(
        upstream_gate={"passed": True},
        source_receipt={
            "revision_matches_expected": True,
            "license_verified": True,
            "data_schema_verified": True,
            "z3_replay_code_present": True,
            "problem_file_count_matches_expected": True,
            "upstream_corruption_warning_present": True,
        },
        source_hashes_match=False,
        identity_rows=identity_rows,
        exact_rows=replay_rows,
        split_commitment=split,
        shard_manifest={"all_shards_verified": False, "corrupt_resume_rejected": False},
        aggregate=aggregate,
        attacks=attacks,
        protected={"all_protected_files_unchanged": False},
    )
    assert gate["all_gates_passed"] is False
    assert {row["check"] for row in gate["failed_checks"]} >= {
        "source_hashes_match",
        "source_to_local_identity",
        "exact_replay",
        "split_lineage",
        "shard_manifest",
        "leakage_attacks",
        "protected_files_unchanged",
    }


def test_scenario_bench_6542_validation_cli_and_edge_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-BENCH-6542-CENSORING/ATTACKS: validators and CLI fail closed."""

    artifact = _artifact(tmp_path)

    assert mod._utc_now().endswith("Z")
    assert mod.sha256_file(tmp_path / "missing") == "missing"
    assert mod._load_json(tmp_path / "missing.json") == {}
    assert mod._load_jsonl(tmp_path / "missing.jsonl") == []
    assert mod._tests_run_receipts(TESTS_RUN) == TESTS_RUN
    assert all(row["exit_code"] == 0 for row in mod._tests_run_receipts(None))
    assert mod._safe_float("1.25") == 1.25
    assert mod._safe_float(None) == 0.0
    assert mod._safe_int("bad", default=7) == 7
    assert mod._normalize_alias("Logic Grid!") == "logicgrid"
    assert mod._context_from_problem({"domain": "logic_grid", "categories": {"color": ["a"]}}) == {
        "categories": {"color": ["a"]}
    }
    assert mod._context_from_problem({"domain": "unknown"}) == {}
    assert mod._turn_position(0, 4) == "early"
    assert mod._turn_position(2, 4) == "middle"
    assert mod._turn_position(3, 4) == "late"
    assert mod._effort_stratum(1) == "low"
    assert mod._effort_stratum(5) == "medium"
    assert mod._effort_stratum(9) == "high"
    assert mod._status_for_class("partial") == "partial_drift_bench_external_intake_v2"
    assert mod._status_for_class("disqualified") == "disqualified_drift_bench_external_intake_v2"
    assert "commit" in mod._drift_git_metadata(REPO)

    sparse_records = [
        {
            "path": tmp_path / "a.json",
            "source_file_relpath": "a.json",
            "source_file_sha256": "sha256:a",
            "problem": {"turns": []},
            "source_problem_id": "a",
            "base_problem_id": "a",
            "domain": "logic_grid",
            "source_split": "dev",
            "num_entities": 1,
            "turn_count": 0,
            "max_cumulative_constraints": 0,
            "source_problem_hash": "sha256:a",
        },
        {
            "path": tmp_path / "b.json",
            "source_file_relpath": "b.json",
            "source_file_sha256": "sha256:b",
            "problem": {"turns": []},
            "source_problem_id": "b",
            "base_problem_id": "b",
            "domain": "logic_grid",
            "source_split": "dev",
            "num_entities": 2,
            "turn_count": 0,
            "max_cumulative_constraints": 0,
            "source_problem_hash": "sha256:b",
        },
        {
            "path": tmp_path / "c.json",
            "source_file_relpath": "c.json",
            "source_file_sha256": "sha256:c",
            "problem": {"turns": []},
            "source_problem_id": "c",
            "base_problem_id": "c",
            "domain": "logic_grid",
            "source_split": "dev",
            "num_entities": 2,
            "turn_count": 0,
            "max_cumulative_constraints": 0,
            "source_problem_hash": "sha256:c",
        },
    ]
    sparse_selected, sparse_commitment = mod.freeze_balanced_slice(sparse_records, fixture_bound=9)
    assert len(sparse_selected) == 3
    assert sparse_commitment["selected_base_problem_count"] == 3
    assert (
        mod._solver_assertion_count(object(), domain="x", entities=[], constraints=[{}], context={})
        == 1
    )

    source_for_replay = _write_source(tmp_path / "edge-source")
    checker = mod.load_z3_checker(source_for_replay)
    unsat_problem = _problem(domain="seating", source_split="dev", index=99, size=2, turn_count=1)
    unsat_problem["turns"][0]["cumulative_constraints"] = [{"type": "force_unsat"}]
    selected = [
        {
            "source_problem_id": "unsat",
            "base_problem_id": "unsat",
            "source_split": "dev",
            "split_name": "train",
            "selection_index": 0,
            "source_file_relpath": "data/problems/dev/unsat.json",
            "source_file_sha256": "sha256:source",
            "source_problem_hash": "sha256:problem",
            "problem": unsat_problem,
            "num_entities": 2,
        }
    ]
    unsat_rows, unsat_exact = mod.replay_selected_turns(
        selected_problems=selected,
        checker=checker,
        source_root=source_for_replay,
    )
    assert unsat_rows[0]["exact_label"] == "contradiction"
    assert unsat_exact[0]["conflict_or_mus_evidence"]["available"] is True

    class TimeoutChecker:
        def check_satisfiability(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            del args, kwargs
            raise TimeoutError("z3_unknown:timeout")

    timeout_rows, timeout_exact = mod.replay_selected_turns(
        selected_problems=selected,
        checker=TimeoutChecker(),
        source_root=source_for_replay,
    )
    assert timeout_rows[0]["censored"] is True
    assert timeout_exact[0]["terminal_status"] == "terminal_timeout"

    bad = deepcopy(artifact)
    bad.pop("status")
    bad["field_principles"] = {}
    bad["field_provenance"] = {}
    bad["verdict_class"] = "positive"
    bad["honest_verdict"] = "ready"
    bad["inference_substrate"] = "live_llm"
    bad["verifier_is_oracle"] = False
    bad["external_constraint_corpus_ready_score"] = 0.5
    bad["reproducibility_checksum"] = "bad"
    errors = mod.validate_artifact(bad)
    assert "required field set mismatch" in errors
    assert "field_principles mismatch" in errors
    assert "field_provenance must cover required fields" in errors
    assert "verdict_class outside Exp6542 enum" in errors
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
    bad_blocked["external_constraint_corpus_ready_score"] = 0.0
    bad_blocked["aggregate_row_recomputation"]["ready_score_from_rows"] = 0.0
    bad_blocked["gate_check_summary"] = {"all_gates_passed": True, "failed_checks": []}
    bad_blocked["reproducibility_checksum"] = mod.reproducibility_checksum(bad_blocked)
    assert "blocked artifact cannot have all gates passed" in mod.validate_artifact(bad_blocked)

    scalar_gate = mod.gate_check_summary(
        upstream_gate={"passed": True},
        source_receipt={
            "revision_matches_expected": True,
            "license_verified": True,
            "data_schema_verified": True,
            "z3_replay_code_present": True,
            "problem_file_count_matches_expected": True,
            "upstream_corruption_warning_present": True,
        },
        source_hashes_match=True,
        identity_rows=[],
        exact_rows=[],
        split_commitment={"passed": True},
        shard_manifest={"all_shards_verified": True, "corrupt_resume_rejected": True},
        aggregate={"ready_score_from_rows": 0.5},
        attacks=[],
        protected={"all_protected_files_unchanged": True},
    )
    assert any(row["check"] == "ready_score_scalar" for row in scalar_gate["failed_checks"])

    source = _write_source(tmp_path / "cli-source")
    repo = _write_repo(tmp_path / "cli-repo", source)
    monkeypatch.setattr(mod, "REPO_ROOT", repo)
    monkeypatch.setattr(mod, "DEFAULT_SOURCE_CACHE_ROOT", source)
    cli_path = tmp_path / "cli-artifact.json"
    fixture_path = tmp_path / "cli-fixture.jsonl"
    assert (
        mod.main(
            [
                "--date",
                "20260823",
                "--result-path",
                str(cli_path),
                "--fixture-path",
                str(fixture_path),
                "--expected-problem-file-count",
                "9",
                "--fixture-bound",
                "9",
            ]
        )
        == 0
    )
    assert f"wrote {mod.RESULT_RELATIVE_PATH.as_posix()}" in capsys.readouterr().out
    assert mod.main(["--result-path", str(cli_path), "--validate"]) == 0
    assert f"validated {mod.RESULT_RELATIVE_PATH.as_posix()}" in capsys.readouterr().out
    bad_cli = tmp_path / "bad-cli.json"
    bad_cli.write_text("{}", encoding="utf-8")
    assert mod.main(["--result-path", str(bad_cli), "--validate"]) == 1
    assert "required field set mismatch" in capsys.readouterr().out

    monkeypatch.setattr(mod, "build_artifact", lambda **kwargs: {"bad": True})
    monkeypatch.setattr(mod, "validate_artifact", lambda artifact: ["forced error"])
    assert mod.main(["--result-path", str(tmp_path / "forced.json")]) == 1
    assert "forced error" in capsys.readouterr().out
