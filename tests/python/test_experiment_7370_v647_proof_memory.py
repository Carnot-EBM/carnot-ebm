"""Tests for versioned source-checked 2-CNF proof memory.

Spec refs: REQ-CL-7370 and SCENARIO-CL-7370-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7370_v647_proof_memory as exp
from carnot.learning import implication_memory as memory_module
from carnot.learning.implication_memory import (
    FormulaVersion,
    ImplicationEdge,
    ProofMemory,
    ProofPath,
    execute_query,
    proof_from_dict,
    validate_proof,
)


def _chain(version: str = "chain-v1", n_vars: int = 8) -> FormulaVersion:
    return FormulaVersion.from_clauses(
        version,
        n_vars,
        [(-variable, variable + 1) for variable in range(1, n_vars)],
    )


def test_source_checked_path_rejects_hostile_certificates() -> None:
    """SCENARIO-CL-7370-PROOF: only exact original edges authorize rejection."""

    formula = _chain()
    first = execute_query(ProofMemory.empty(formula), (1, -8))
    proof = first.discovered_paths[0]

    assert first.decision == "unsatisfiable"
    assert first.used_exact_solver is True
    assert validate_proof(formula, proof) == []
    assert all(edge in formula.implication_edges for edge in proof.edges)
    assert all(edge.source_clause_id >= 0 for edge in proof.edges)

    forged = deepcopy(proof.to_dict())
    forged["edges"][0]["to_literal"] = 7
    stale = {**proof.to_dict(), "formula_version": "chain-v0"}
    omitted = deepcopy(proof.to_dict())
    omitted["edges"].pop(1)
    negative_id = deepcopy(proof.to_dict())
    negative_id["edges"][0]["source_clause_id"] = -1
    cycle = deepcopy(proof.to_dict())
    cycle["edges"].append(deepcopy(cycle["edges"][0]))
    conflicting = {**proof.to_dict(), "source_hash": "sha256:" + "0" * 64}
    leaked = {**proof.to_dict(), "label": "unsat"}

    for payload in (forged, stale, omitted, negative_id, cycle, conflicting, leaked):
        with pytest.raises(ValueError):
            proof_from_dict(payload, formula)


def test_formula_and_proof_input_schemas_fail_closed() -> None:
    """SCENARIO-CL-7370-PROOF: malformed public inputs never become authority."""

    for args, expected in (
        (("", 2, [(1, 2)]), "formula_version_empty"),
        (("v1", 0, [(1, 2)]), "n_vars_invalid"),
        (("v1", 2, [(1, 2, -1)]), "clause_not_2cnf"),
        (("v1", 2, [(0, 2)]), "clause_literal_invalid"),
    ):
        with pytest.raises(ValueError, match=expected):
            FormulaVersion.from_clauses(*args)

    formula = _chain()
    assert formula.to_dict()["source_hash"] == formula.source_hash
    with pytest.raises(ValueError, match="assumption_literal_invalid"):
        formula.solve((0,))
    assert formula.verify_assignment({1: True}) is False

    proof = execute_query(ProofMemory.empty(formula), (1, -8)).discovered_paths[0]
    payload = proof.to_dict()
    mutations = []
    changed = deepcopy(payload)
    changed["edges"] = "not-a-list"
    mutations.append((changed, "proof_edges_not_list"))
    changed = deepcopy(payload)
    changed["edges"][0]["extra"] = 1
    mutations.append((changed, "proof_edge_schema_mismatch"))
    changed = deepcopy(payload)
    changed["edges"][0]["from_literal"] = "one"
    mutations.append((changed, "proof_edge_type_invalid"))
    changed = deepcopy(payload)
    changed["antecedent"] = "one"
    mutations.append((changed, "proof_endpoint_type_invalid"))
    for changed, expected in mutations:
        with pytest.raises(ValueError, match=expected):
            proof_from_dict(changed, formula)

    endpoint = ProofPath(formula.version, formula.source_hash, 99, 8, proof.edges)
    empty = ProofPath(formula.version, formula.source_hash, 1, 8, ())
    wrong_start = ProofPath(formula.version, formula.source_hash, 2, 8, proof.edges)
    assert "endpoint_invalid" in validate_proof(formula, endpoint)
    assert "path_empty" in validate_proof(formula, empty)
    assert "antecedent_endpoint_mismatch" in validate_proof(formula, wrong_start)


def test_prior_snapshot_changes_later_decision_and_single_path_erasure() -> None:
    """SCENARIO-CL-7370-LIFECYCLE/WITNESS: one old proof changes a later query."""

    formula = _chain()
    empty = ProofMemory.empty(formula)
    learning = execute_query(empty, (1, -8))

    assert learning.snapshot_sha256 == empty.sha256
    assert learning.used_exact_solver is True
    assert learning.committed_memory.sha256 != empty.sha256
    assert learning.proof_path_id is None

    later_assumptions = (1, -8, 3)
    later = execute_query(learning.committed_memory, later_assumptions)
    assert later.decision == "reject"
    assert later.used_exact_solver is False
    assert later.proof_path_id is not None

    erased = learning.committed_memory.without(later.proof_path_id)
    counterfactual = execute_query(erased, later_assumptions)
    assert counterfactual.decision == "unsatisfiable"
    assert counterfactual.used_exact_solver is True
    assert counterfactual.proof_path_id is None

    satisfiable = execute_query(learning.committed_memory, (1, 8))
    assert satisfiable.decision == "satisfiable"
    assert satisfiable.used_exact_solver is True
    assert satisfiable.assignment is not None
    assert formula.verify_assignment(satisfiable.assignment, (1, 8))


def test_bounded_memory_restart_and_version_edit_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7370-LIFECYCLE: caps, restart, and version changes are atomic."""

    formula = _chain(n_vars=12)
    memory = ProofMemory.empty(formula)
    query = execute_query(memory, (1, -12))
    assert len(query.discovered_paths) <= 8
    assert all(len(path.edges) <= 2 * formula.n_vars for path in query.discovered_paths)

    stored = tmp_path / "proof-memory.json"
    query.committed_memory.save(stored)
    restored = ProofMemory.load(stored, formula)
    assert restored.to_bytes() == query.committed_memory.to_bytes()
    assert len(restored.paths) <= 128
    assert len(restored.to_bytes()) <= 65_536

    edited = FormulaVersion.from_clauses(
        "chain-v2",
        12,
        [(-variable, variable + 1) for variable in range(1, 11)],
    )
    invalidated = ProofMemory.load(stored, edited)
    assert invalidated.paths == ()
    assert invalidated.formula.source_hash == edited.source_hash

    too_long = ProofPath(
        formula_version=formula.version,
        source_hash=formula.source_hash,
        antecedent=1,
        consequent=12,
        edges=tuple(ImplicationEdge(1, 2, 0) for _ in range(2 * formula.n_vars + 1)),
    )
    assert "path_edge_cap_exceeded" in validate_proof(formula, too_long)

    invalid = ProofPath(formula.version, formula.source_hash, 1, 12, ())
    assert query.committed_memory.commit((invalid,)).sha256 == query.committed_memory.sha256
    assert (
        query.committed_memory.commit(query.committed_memory.paths).sha256
        == query.committed_memory.sha256
    )

    stored.write_text("{}", encoding="utf-8")
    assert ProofMemory.load(stored, formula).paths == ()
    stored.write_text("not-json", encoding="utf-8")
    assert ProofMemory.load(stored, formula).paths == ()

    proof = query.committed_memory.paths[0]
    monkeypatch.setattr(memory_module, "MAX_STATE_BYTES", 1)
    assert ProofMemory.empty(formula).commit((proof,)).paths == ()
    monkeypatch.setattr(memory_module, "MAX_STATE_BYTES", 65_536)

    oversized = {
        "schema": "carnot.implication_memory.v1",
        "formula_version": formula.version,
        "source_hash": formula.source_hash,
        "paths": [proof.to_dict()] * 129,
    }
    stored.write_text(json.dumps(oversized), encoding="utf-8")
    assert ProofMemory.load(stored, formula).paths == ()

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("owned atomic replace failure")

    monkeypatch.setattr(memory_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="owned atomic replace failure"):
        query.committed_memory.save(tmp_path / "failed-save.json")
    assert not list(tmp_path.glob(".failed-save.json.*.tmp"))


def test_discovery_and_invalid_retained_paths_remain_bounded() -> None:
    """REQ-CL-7370: discovery stops at eight and bad retained paths are ignored."""

    clauses = [(-left, right) for left in range(1, 7) for right in range(7, 13)]
    formula = FormulaVersion.from_clauses("dense-v1", 12, clauses)
    result = execute_query(ProofMemory.empty(formula), (1, 2, 3, 4, 5, 6, -7, -8, -9, -10))
    assert result.decision == "unsatisfiable"
    assert len(result.discovered_paths) == 8

    invalid = ProofPath(formula.version, formula.source_hash, 1, 7, ())
    retained = ProofMemory(formula, (invalid,))
    fallback = execute_query(retained, (1, -7))
    assert fallback.used_exact_solver is True


def test_development_panel_has_fixed_shapes_and_independent_exact_checks() -> None:
    """REQ-CL-7370: all 16 formulas pass size-specific independent checks."""

    panel = exp.build_development_panel()
    assert len(panel) == 16
    assert {row["formula"].n_vars for row in panel} == {8, 12, 24, 32}
    assert {row["feature"] for row in panel} == {
        "long_chain",
        "independent_conflicts",
        "satisfiable_near_miss",
        "duplicate_and_version_edit",
    }

    rows, witnesses, controls = exp.run_development(panel)
    assert len({row["formula_id"] for row in rows}) == 16
    assert all(row["same_public_formula_and_assumptions"] for row in rows)
    assert all(row["independent_exact_match"] for row in rows)
    assert all(row["comparison_method"] == "enumeration" for row in rows if row["n_vars"] <= 12)
    assert all(row["comparison_method"] == "independent_dpll" for row in rows if row["n_vars"] > 12)
    assert witnesses
    assert all(
        witness["later_assumptions"] != witness["prior_assumptions"] for witness in witnesses
    )
    assert all(witness["early_decision_removed"] for witness in witnesses)
    assert {row["attack"] for row in controls} == {
        "stale_version",
        "forged_edge",
        "omitted_literal",
        "negative_clause_id",
        "cycle",
        "conflicting_metadata",
        "label_leakage",
    }
    assert all(row["passed"] for row in controls)


def test_cost_rows_charge_all_work_without_a_held_out_claim() -> None:
    """REQ-CL-7370: development comparators include complete measured costs."""

    rows, _witnesses, _controls = exp.run_development(exp.build_development_panel())
    costs = exp.reduce_development_costs(rows)

    assert {row["arm"] for row in costs} == {
        "proof_memory",
        "reset_exact_solver",
        "persistent_incremental_exact_solver",
        "persistent_source_graph_reachability_cache",
    }
    for row in costs:
        assert set(row["cost_ns"]) == {
            "proof_discovery",
            "proof_checking",
            "updates",
            "storage",
            "exact_solve",
            "total",
        }
        assert row["cost_ns"]["total"] == sum(
            value for key, value in row["cost_ns"].items() if key != "total"
        )
        assert row["development_only"] is True
        assert row["held_out_selected"] is False
        assert row["ten_x_gain_claimed"] is False


def test_preconditions_label_disqualified_history_without_blocking_fixture(tmp_path: Path) -> None:
    """REQ-CL-7370: V646 is historical evidence and never a readiness producer."""

    capstone = tmp_path / "capstone.json"
    capstone.write_text(
        json.dumps(
            {
                "experiment_id": "exp7368-capstone",
                "milestone": "2026.09.646",
                "verdict_class": "disqualified",
                "flagged_adversarial": False,
                "required_science_complete_score": 0,
            }
        ),
        encoding="utf-8",
    )
    checks, hashes, sidecars = exp.collect_preconditions(exp.REPO_ROOT, capstone_path=capstone)
    assert all(row["passed"] for row in checks)
    assert hashes[str(capstone)] == exp.sha256_file(capstone)
    assert sidecars[0]["historical_only"] is True
    assert sidecars[0]["authorizes_readiness"] is False
    assert sidecars[0]["historical_verdict_class"] == "disqualified"

    capstone.unlink()
    missing, _hashes, _sidecars = exp.collect_preconditions(exp.REPO_ROOT, capstone_path=capstone)
    failure = next(row for row in missing if row["check"] == "historical_capstone_path")
    assert failure == {
        "check": "historical_capstone_path",
        "upstream": str(capstone),
        "artifact_field": "path",
        "expected": "readable_nonempty_json_object",
        "observed": "missing",
        "passed": False,
        "terminal_blocking": False,
    }


def test_exp7358_scoped_plan_has_exact_affected_surface(tmp_path: Path) -> None:
    """SCENARIO-CL-7370-VALIDATION: the required plan stays explicit and local."""

    commands = exp.scoped_command_plan(exp.REPO_ROOT, tmp_path / "private")
    assert exp.validate_scoped_command_plan(exp.REPO_ROOT, commands) == []
    assert {command.name for command in commands} == set(exp.REQUIRED_CHECK_NAMES)
    assert not any("full_python_suite" in command.name for command in commands)
    for command in commands:
        joined = " ".join(command.argv)
        assert (
            "tests/python/test_experiment_7370_v647_proof_memory.py" in joined
            or command.name
            in {
                "worktree_imports",
                "changed_module_coverage_report",
                "changed_module_mypy",
            }
        )
    focused = next(command for command in commands if command.name == "focused_pytest")
    assert ("-n", "0") == focused.argv[1:3]
    assert "addopts=" in focused.argv
    assert "--no-cov" in focused.argv

    missing = commands[1:]
    assert "required_command_names_changed" in exp.validate_scoped_command_plan(
        exp.REPO_ROOT, missing
    )
    broad = [
        *commands,
        exp.validation_scope.CommandSpec("full_python_suite", ("pytest", "tests/python"), "bad"),
    ]
    assert "full_python_suite_forbidden" in exp.validate_scoped_command_plan(exp.REPO_ROOT, broad)


def test_helper_boundaries_and_sidecars_are_real_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-CL-7370: command receipts, sidecars, and CLI parsing keep exact bytes."""

    started = exp.time.monotonic()
    exp.progress(started, "unit", "boundary", count=1)
    assert "phase=unit event=boundary" in capsys.readouterr().out
    assert exp.utc_now().endswith("+00:00")

    target = tmp_path / "atomic.json"
    exp.atomic_json(target, {"value": 1})
    assert exp._load_object(target) == {"value": 1}
    target.write_text("[]", encoding="utf-8")
    assert exp._load_object(target) == {}
    target.write_text("bad", encoding="utf-8")
    assert exp._load_object(target) == {}

    normalized = exp._normalized_receipt(
        {
            "name": "worktree_imports",
            "command": "python check",
            "command_argv": ["python", "check"],
            "command_environment": {"A": "B"},
            "scope": "changed_modules",
            "exit_code": 0,
            "duration_s": 0.1,
            "log_path": "log",
            "log_sha256": "sha256:" + "1" * 64,
            "passed": True,
            "timed_out": False,
            "required": True,
            "command_category": "required_validation",
            "resolved_imports": {"carnot.x": "/tmp/x.py"},
        }
    )
    assert normalized["environment"] == {"A": "B"}
    assert normalized["resolved_imports"] == {"carnot.x": "/tmp/x.py"}

    panel = exp.build_development_panel()[:1]
    rows, witnesses, controls = exp.run_development(panel)
    hashes = exp._write_sidecars(tmp_path, panel, rows, witnesses, controls, [])
    assert len(hashes) == 3
    assert all((tmp_path / path).is_file() for path in hashes)
    span = exp._span("unit", started, started)
    assert span["phase"] == "unit" and span["duration_s"] >= 0
    commands = exp._terminal_commands(tmp_path / "candidate.json")
    assert {command.spec.name for command in commands} == {
        "cold_artifact_replay",
        "independent_reducer",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    }
    parsed = exp.parse_args(["--date", exp.RUN_DATE, "--output", str(target)])
    assert parsed.date == exp.RUN_DATE and parsed.output == target


def test_independent_dpll_branching_and_invalid_panel_type() -> None:
    """REQ-CL-7370: the larger-case solver branches independently and rejects bad panels."""

    branching = FormulaVersion.from_clauses("branch-v1", 3, [(1, 2), (-1, 3)])
    assert exp._independent_dpll(branching, ()) is True
    contradiction = FormulaVersion.from_clauses("contradiction-v1", 1, [(1, 1), (-1, -1)])
    assert exp._independent_dpll(contradiction, ()) is False
    branch_unsat = FormulaVersion.from_clauses(
        "branch-unsat-v1",
        2,
        [(1, 2), (1, -2), (-1, 2), (-1, -2)],
    )
    assert exp._independent_dpll(branch_unsat, ()) is False
    with pytest.raises(TypeError, match="panel_formula_invalid"):
        exp.run_development(
            [{"formula": {}, "formula_id": "bad", "feature": "bad", "assumption_queries": []}]
        )


def test_artifact_reduction_is_cold_and_scores_fail_closed() -> None:
    """SCENARIO-CL-7370-VALIDATION: raw rows determine all terminal scores."""

    rows, witnesses, controls = exp.run_development(exp.build_development_panel())
    receipts = exp.passing_test_receipts()
    artifact = exp.build_artifact_for_test(rows, witnesses, controls, receipts)

    assert exp.validate_artifact(artifact) == []
    assert exp.independent_reduce(artifact) == artifact["independent_reduction"]
    assert artifact["proof_fixture_ready_score"] == 1
    assert artifact["learning_value_score"] == 0
    assert artifact["promotion_score"] == 0
    assert artifact["development_witnesses"][0]["source_clauses"]
    assert artifact["memory_contract"]["satisfying_answer_authority"] is False

    broken = deepcopy(artifact)
    next(row for row in broken["validation_receipts"] if row["required"])["passed"] = False
    broken["independent_reduction"] = exp.independent_reduce(broken)
    broken.update(
        proof_fixture_ready_score=0,
        learning_value_score=0,
        promotion_score=0,
        verdict_class="disqualified",
        status="complete_disqualified_required_validation_failure",
        honest_verdict="complete_disqualified_required_validation_failure",
    )
    broken["acceptance_gate_results"] = exp.build_acceptance_gates(broken)
    broken["gate_check_summary"] = exp.gate_summary(broken["acceptance_gate_results"])
    broken["field_principles"] = exp.field_principles(broken)
    broken["reproducibility_checksum"] = exp.reproducibility_checksum(broken)
    assert exp.validate_artifact(broken) == []

    disqualified = exp.build_artifact_for_test(rows, witnesses, controls, receipts[:-1])
    assert disqualified["verdict_class"] == "disqualified"

    blocking = exp._precondition(
        "missing",
        "unit",
        "path",
        "present",
        "missing",
        terminal_blocking=True,
    )
    blocked = exp._base_artifact(
        rows,
        witnesses,
        controls,
        receipts,
        preconditions=[blocking],
        source_hashes={},
        historical_sidecars=[],
        phase_spans=[],
        started_at_utc="2026-09-17T00:00:00+00:00",
        completed_at_utc="2026-09-17T00:00:01+00:00",
        duration_s=1.0,
        repository_health={"affects_required_checks": False},
    )
    assert blocked["verdict_class"] == "blocked"


def test_artifact_validator_names_each_fail_closed_boundary() -> None:
    """SCENARIO-CL-7370-VALIDATION: corrupt terminal fields are diagnosed exactly."""

    rows, witnesses, controls = exp.run_development(exp.build_development_panel())
    artifact = exp.build_artifact_for_test(rows, witnesses, controls, exp.passing_test_receipts())
    mutations = (
        ("artifact_not_object", []),
        ("identity_mismatch", {**artifact, "milestone": "wrong"}),
        ("model_declaration_mismatch", {**artifact, "model_invoked": True}),
        (
            "current_invocation_counts_nonzero",
            {**artifact, "invocation_counts": {"current": {"model_loads_attempted": 1}}},
        ),
        ("substrate_class_mismatch", {**artifact, "inference_substrate_class": "wrong"}),
        ("truth_or_venue_mismatch", {**artifact, "execution_venue": "board"}),
        ("verdict_class_invalid", {**artifact, "verdict_class": "unknown"}),
        ("deferred_scores_nonzero", {**artifact, "learning_value_score": 1}),
        ("independent_reduction_mismatch", {**artifact, "independent_reduction": {}}),
        ("fixture_score_mismatch", {**artifact, "proof_fixture_ready_score": 0}),
        ("acceptance_gates_mismatch", {**artifact, "acceptance_gate_results": []}),
        ("gate_summary_mismatch", {**artifact, "gate_check_summary": {}}),
        (
            "failed_state_scores_nonzero",
            {**artifact, "verdict_class": "disqualified", "proof_fixture_ready_score": 1},
        ),
        (
            "adversarial_readiness_nonzero",
            {**artifact, "flagged_adversarial": True, "proof_fixture_ready_score": 1},
        ),
        ("field_principles_incomplete", {**artifact, "field_principles": {}}),
        ("reproducibility_checksum_mismatch", {**artifact, "reproducibility_checksum": "bad"}),
    )
    for expected, changed in mutations:
        assert expected in exp.validate_artifact(changed)


def test_thin_entrypoint_delegates_without_old_launcher_import() -> None:
    """SCENARIO-CL-7370-VALIDATION: the public script stays a thin entrypoint."""

    text = (exp.REPO_ROOT / exp.WRAPPER_PATH).read_text(encoding="utf-8")
    assert "experiment_7370_v647_proof_memory import main" in text
    assert "experiment_7362" not in text
    assert "experiment_7358" not in text
