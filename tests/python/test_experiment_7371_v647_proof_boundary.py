"""Boundary tests for REQ-CL-7371 and SCENARIO-CL-7371-*.

The oracle and path assertions in this file do not use the Exp7370 reducer.
They protect the boundary before any live proposal bytes are collected.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time

import pytest

from carnot import experiment_7371_v647_proof_boundary as experiment
from carnot.learning.implication_memory import FormulaVersion, ProofMemory, ProofPath
from carnot.reporting.experiment_7303_validation_scope import CommandSpec


@pytest.fixture(scope="module")
def completed_fixture():
    """Build the full deterministic panel once for terminal reducer tests."""

    protocol = experiment.build_protocol()
    rows, witnesses = experiment.evaluate_synthetic(protocol)
    attacks = experiment.run_authority_controls(protocol)
    artifact = experiment.build_artifact_for_test(protocol, rows, witnesses, attacks)
    return protocol, rows, witnesses, attacks, artifact


def _proof_fixture() -> tuple[dict[str, object], dict[str, object]]:
    formula = FormulaVersion.from_clauses("fixture-v1", 4, [(-1, 2), (-2, 3), (-3, 4)])
    edges = formula.find_path(1, 4)
    assert edges is not None
    proof = ProofPath(formula.version, formula.source_hash, 1, 4, edges)
    return formula.to_dict(), proof.to_dict()


def test_protocol_freezes_disjoint_cohorts_and_live_prompts() -> None:
    """REQ-CL-7371: Freeze all cohort sizes and hide formal labels from live prompts."""

    protocol = experiment.build_protocol()

    assert experiment.validate_protocol(protocol) == []
    assert len(protocol["development_formulas"]) == 16
    assert len(protocol["evaluation_streams"]) == 32
    assert len(protocol["live_proposal_streams"]) == 8
    assert {row["n_vars"] for row in protocol["evaluation_streams"]} == {8, 12, 24, 32}
    assert len({row["family"] for row in protocol["evaluation_streams"]}) == 8
    assert all(
        [request["split"] for request in stream["requests"]].count("warm_up") == 8
        and [request["split"] for request in stream["requests"]].count("later_distinct") == 8
        and [request["split"] for request in stream["requests"]].count("recurrence") == 4
        and [request["split"] for request in stream["requests"]].count("version_change") == 4
        for stream in protocol["evaluation_streams"]
    )
    all_ids = {
        row["formula_id"]
        for key in ("development_formulas", "evaluation_streams", "live_proposal_streams")
        for row in protocol[key]
    }
    assert len(all_ids) == 16 + 32 + 8
    live_requests = [
        request for stream in protocol["live_proposal_streams"] for request in stream["requests"]
    ]
    assert len(live_requests) == 32
    assert sum(request["proposal_count"] for request in live_requests) == 64
    assert all(
        "SAT" not in request["prompt"] and "label" not in request["prompt"]
        for request in live_requests
    )
    assert protocol["arms"] == list(experiment.ARMS)


def test_protocol_rejects_missing_queries_bad_hash_and_label_leakage() -> None:
    """SCENARIO-CL-7371-PROTOCOL: Sealed bytes fail closed after structural edits."""

    protocol = experiment.build_protocol()
    missing = deepcopy(protocol)
    missing["evaluation_streams"][0]["requests"].pop()
    assert "evaluation_request_count" in experiment.validate_protocol(missing)

    bad_hash = deepcopy(protocol)
    bad_hash["evaluation_streams"][0]["versions"][0]["source_hash"] = "sha256:" + "0" * 64
    assert "formula_source_hash" in experiment.validate_protocol(bad_hash)

    leaked = deepcopy(protocol)
    leaked["live_proposal_streams"][0]["requests"][0]["prompt"] += " SAT label"
    assert "live_prompt_label_leakage" in experiment.validate_protocol(leaked)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda p: p.update(schema="bad"), "protocol_identity"),
        (lambda p: p["development_formulas"].pop(), "development_formula_count"),
        (lambda p: p["evaluation_streams"].pop(), "evaluation_stream_count"),
        (lambda p: p["live_proposal_streams"].pop(), "live_stream_count"),
        (
            lambda p: p["live_proposal_streams"][0].update(
                formula_id=p["development_formulas"][0]["formula_id"]
            ),
            "cohort_formula_overlap",
        ),
        (lambda p: p.update(arms=[]), "arm_order"),
        (
            lambda p: p["development_formulas"][0].update(formula={"n_vars": 0}),
            "formula_schema",
        ),
        (lambda p: p["evaluation_streams"][0]["versions"].pop(), "evaluation_version_count"),
        (
            lambda p: p["evaluation_streams"][0]["requests"][0].update(split="later_distinct"),
            "evaluation_request_splits",
        ),
        (
            lambda p: p["evaluation_streams"][0]["requests"][0].update(request_index=9),
            "evaluation_request_order",
        ),
        (
            lambda p: p["evaluation_streams"][0]["requests"][20].update(assumptions=[1, -1]),
            "version_change_expected_sat_invalid",
        ),
        (lambda p: p["live_proposal_streams"][0]["requests"].pop(), "live_request_count"),
        (
            lambda p: p["live_proposal_streams"][0]["requests"][0].update(
                minimum_distinct_literals=1
            ),
            "live_proposal_shape",
        ),
    ],
)
def test_protocol_validator_covers_every_closed_branch(mutation, expected: str) -> None:
    """SCENARIO-CL-7371-PROTOCOL: Every declared protocol boundary fails closed."""

    protocol = experiment.build_protocol()
    mutation(protocol)
    assert expected in experiment.validate_protocol(protocol)


def test_truth_table_oracle_checks_partial_assignment_extendibility() -> None:
    """REQ-CL-7371: The small-n oracle judges whether assumptions can be extended."""

    formula = FormulaVersion.from_clauses("oracle-v1", 3, [(-1, 2), (-2, 3)]).to_dict()
    assert experiment.truth_table_extendible(formula, [1, 3]) is True
    assert experiment.truth_table_extendible(formula, [1, -3]) is False
    with pytest.raises(ValueError, match="truth_table_variable_cap"):
        experiment.truth_table_extendible({**formula, "n_vars": 13}, [])
    with pytest.raises(ValueError, match="assumption_literal_invalid"):
        experiment.truth_table_extendible(formula, [0])


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda _f, p: p.update(formula_version="stale"), "formula_version_mismatch"),
        (lambda _f, p: p.update(source_hash="sha256:" + "0" * 64), "source_hash_mismatch"),
        (lambda _f, p: p.update(antecedent=0), "endpoint_invalid"),
        (lambda _f, p: p.update(edges=[]), "path_empty"),
        (lambda _f, p: p["edges"][0].update(from_literal=-1), "antecedent_endpoint_mismatch"),
        (lambda _f, p: p["edges"][-1].update(to_literal=-4), "consequent_endpoint_mismatch"),
        (lambda _f, p: p["edges"][0].update(source_clause_id=-1), "negative_source_clause_id"),
        (lambda _f, p: p["edges"][0].update(to_literal=-2), "edge_not_in_source_formula"),
        (lambda _f, p: p["edges"][1].update(from_literal=-2), "path_literal_omitted"),
        (lambda _f, p: p["edges"].append(deepcopy(p["edges"][0])), "path_cycle"),
        (lambda _f, p: p.update(path_id="sha256:" + "f" * 64), "path_id_mismatch"),
        (lambda f, _p: f["clauses"][0].update(clause_id=-1), "formula_clause_id_invalid"),
    ],
)
def test_independent_checker_rejects_every_path_authority_branch(mutation, expected: str) -> None:
    """SCENARIO-CL-7371-INDEPENDENT: Each malformed authority branch is observable."""

    formula, proof = _proof_fixture()
    mutation(formula, proof)
    assert expected in experiment.independent_validate_path(formula, proof)


def test_independent_checker_accepts_only_canonical_source_path() -> None:
    """SCENARIO-CL-7371-INDEPENDENT: Original source clauses reconstruct the path."""

    formula, proof = _proof_fixture()
    assert experiment.independent_validate_path(formula, proof) == []
    over_cap = deepcopy(proof)
    over_cap["edges"] = [deepcopy(proof["edges"][0])] * 9
    assert "path_edge_cap_exceeded" in experiment.independent_validate_path(formula, over_cap)


def test_crash_restart_and_version_change_isolate_committed_bytes(tmp_path: Path) -> None:
    """SCENARIO-CL-7371-ARMS: Restart preserves bytes and version edits invalidate them."""

    formula = FormulaVersion.from_clauses("restart-v1", 4, [(-1, 2), (-2, 3), (-3, 4)])
    edges = formula.find_path(1, 4)
    assert edges is not None
    proof = ProofPath(formula.version, formula.source_hash, 1, 4, edges)
    memory = ProofMemory.empty(formula).commit([proof])
    state_path = tmp_path / "state.json"
    memory.save(state_path)
    assert ProofMemory.load(state_path, formula).to_bytes() == memory.to_bytes()

    changed = FormulaVersion.from_clauses("restart-v2", 4, [(-1, 2), (-2, 3), (3, 4)])
    assert ProofMemory.load(state_path, changed).paths == ()
    state_path.write_text("{crash", encoding="utf-8")
    assert ProofMemory.load(state_path, formula).paths == ()


def test_five_arms_share_inputs_and_preserve_incremental_state() -> None:
    """SCENARIO-CL-7371-ARMS: Five arms see equal ordered requests and exact outcomes."""

    stream = experiment.build_protocol()["evaluation_streams"][0]
    rows, witnesses = experiment.evaluate_stream(stream)

    assert len(rows) == 24 * 5
    assert {row["arm"] for row in rows} == set(experiment.ARMS)
    assert all(row["independent_exact_match"] and row["final_exact_validation"] for row in rows)
    for request_index in range(24):
        request_rows = [row for row in rows if row["request_index"] == request_index]
        assert [row["arm"] for row in request_rows] == list(experiment.ARMS)
        assert len({row["assumption_bytes_sha256"] for row in request_rows}) == 1
        assert len({row["formula_source_hash"] for row in request_rows}) == 1
    incremental = [row for row in rows if row["arm"] == "persistent_incremental_exact_solver"]
    assert len({row["state_instance_id"] for row in incremental[:20]}) == 1
    assert len({row["state_instance_id"] for row in incremental[20:]}) == 1
    assert incremental[0]["state_instance_id"] != incremental[-1]["state_instance_id"]
    assert all(
        row["matched_state_size"]
        for row in rows
        if row["arm"] == "proof_memory_matched_non_applicable"
    )
    assert witnesses


def test_authority_controls_cover_all_named_attacks() -> None:
    """REQ-CL-7371: Negative controls reject every unauthorized shortcut."""

    rows = experiment.run_authority_controls(experiment.build_protocol())

    assert {row["attack"] for row in rows} == set(experiment.AUTHORITY_ATTACKS)
    assert all(
        row["expected"] == "reject" and row["observed"] == "reject" and row["passed"]
        for row in rows
    )


def test_progress_time_and_stream_heartbeat(capsys) -> None:
    """REQ-CL-7371: Long-loop progress uses real time and flushes completed units."""

    started = time.monotonic()
    experiment.progress(started, "test", "boundary", detail="one")
    assert "phase=test event=boundary" in capsys.readouterr().out
    assert "+00:00" in experiment.utc_now()
    protocol = experiment.build_protocol()
    one_stream = {"evaluation_streams": protocol["evaluation_streams"][:1]}
    rows, _witnesses = experiment.evaluate_synthetic(
        one_stream, emit_progress=True, started=started
    )
    assert len(rows) == 120
    assert "stream_complete" in capsys.readouterr().out


def test_acceptance_manifest_is_fixed_before_outcomes() -> None:
    """SCENARIO-CL-7371-PROTOCOL: Outcomes cannot change fixed gate thresholds."""

    manifest = experiment.frozen_acceptance_manifest("sha256:" + "1" * 64)

    assert manifest["bootstrap_draws"] == 10_000
    assert manifest["bootstrap_seed"] == 7_371_307
    assert manifest["minimum_erasure_witnesses"] == 8
    assert manifest["minimum_witness_streams"] == 4
    assert manifest["paid_exact_query_ratio_upper_bound_lt"] == 0.90
    assert manifest["complete_service_cost_ratio_lte"] == 1.0
    assert manifest["comparators"] == [
        "persistent_incremental_exact_solver",
        "persistent_source_graph_reachability_cache",
    ]


@pytest.mark.parametrize(
    ("change", "failed_check"),
    [
        (lambda _p: None, None),
        (lambda p: p.update(proof_fixture_ready_score=0), "producer_proof_fixture_ready_score"),
        (lambda p: p.update(verdict_class="disqualified"), "producer_verdict_class"),
        (lambda p: p.update(flagged_adversarial=True), "producer_flagged_adversarial"),
        (lambda p: p.update(status="blocked_missing_input"), "producer_terminal_status"),
    ],
)
def test_preconditions_gate_exact_producer_fields(
    tmp_path: Path, change, failed_check: str | None
) -> None:
    """REQ-CL-7371: Dependent work rejects each unavailable producer class."""

    root = tmp_path
    producer = {
        "experiment_id": "exp7370-v647-proof-memory",
        "milestone": "2026.09.647",
        "status": "complete_proof_fixture_ready_development_only",
        "proof_fixture_ready_score": 1,
        "verdict_class": "null",
        "flagged_adversarial": False,
    }
    change(producer)
    path = root / "producer.json"
    path.write_text(json.dumps(producer), encoding="utf-8")

    checks, hashes, _sidecars = experiment.collect_preconditions(
        experiment.REPO_ROOT, producer_path=path
    )
    failures = [row["check"] for row in checks if row["terminal_blocking"] and not row["passed"]]
    if failed_check is None:
        assert failures == []
        assert str(path) in hashes
    else:
        assert failed_check in failures


def test_missing_producer_is_a_terminal_block(tmp_path: Path) -> None:
    """REQ-CL-7371: Missing external input is blocked, not partial."""

    checks, _hashes, _sidecars = experiment.collect_preconditions(
        experiment.REPO_ROOT, producer_path=tmp_path / "missing.json"
    )
    failed = [row for row in checks if row["terminal_blocking"] and not row["passed"]]
    assert failed[0]["check"] == "producer_path"
    assert failed[0]["observed"] == "missing"


def test_manifest_raw_evidence_and_artifact_round_trip(tmp_path: Path, completed_fixture) -> None:
    """SCENARIO-CL-7371-READINESS: Sealed raw bytes rebuild one honest terminal record."""

    protocol, rows, witnesses, attacks, artifact = completed_fixture
    protocol_path = tmp_path / "data" / "manifest.json"
    raw_dir = tmp_path / "raw"
    hashes = experiment.seal_protocol_and_fixtures(protocol_path, raw_dir, protocol)
    loaded = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert loaded == protocol
    assert hashes[str(protocol_path)] == experiment.sha256_file(protocol_path)
    assert {path.name for path in raw_dir.iterdir()} == {
        "development_formulas.json",
        "evaluation_streams.json",
        "live_proposal_requests.json",
    }

    assert artifact["proof_boundary_ready_score"] == 1
    assert artifact["learning_value_score"] == 0
    assert artifact["promotion_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_")
    assert experiment.validate_artifact(artifact) == []
    assert experiment.independent_reduce(artifact)["proof_boundary_ready_score"] == 1

    tampered = deepcopy(artifact)
    tampered["formula_stream_rows"].pop()
    assert "independent_reduction_mismatch" in experiment.validate_artifact(tampered)
    tampered = deepcopy(artifact)
    tampered["reproducibility_checksum"] = "bad"
    assert "reproducibility_checksum_mismatch" in experiment.validate_artifact(tampered)


def test_terminal_classification_covers_blocked_and_disqualified(completed_fixture) -> None:
    """REQ-CL-7371: External absence is blocked and failed own checks are disqualified."""

    protocol, rows, witnesses, attacks, _artifact = completed_fixture
    blocked = experiment.build_artifact_for_test(
        protocol,
        rows,
        witnesses,
        attacks,
        preconditions=[
            experiment._check("producer_path", "missing", "path", "present", "missing", False)
        ],
    )
    assert blocked["verdict_class"] == "blocked"
    assert blocked["proof_boundary_ready_score"] == 0

    disqualified = experiment.build_artifact_for_test(
        protocol, rows, witnesses, attacks, receipts=[]
    )
    assert disqualified["verdict_class"] == "disqualified"


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda a: a.update(schema="bad"), "identity_mismatch"),
        (lambda a: a.update(MODEL_SPECS=["model"]), "model_declaration_mismatch"),
        (
            lambda a: a["invocation_counts"].update(current={}),
            "current_invocation_counts_nonzero",
        ),
        (lambda a: a.update(execution_venue="gpu"), "substrate_mismatch"),
        (lambda a: a.update(verifier_is_oracle=False), "oracle_declaration_mismatch"),
        (lambda a: a.update(learning_value_score=1), "deferred_score_nonzero"),
        (lambda a: a.update(flagged_adversarial=True), "adversarial_readiness_nonzero"),
        (lambda a: a.update(field_principles={}), "field_principles_incomplete"),
    ],
)
def test_artifact_validator_covers_fail_closed_declarations(
    completed_fixture, mutation, expected: str
) -> None:
    """SCENARIO-CL-7371-READINESS: Terminal declarations cannot drift silently."""

    artifact = deepcopy(completed_fixture[-1])
    mutation(artifact)
    assert expected in experiment.validate_artifact(artifact)


def test_artifact_validator_rejects_non_object() -> None:
    """SCENARIO-CL-7371-READINESS: A scalar cannot masquerade as an artifact."""

    assert experiment.validate_artifact([]) == ["artifact_not_object"]


def test_cold_reload_detects_raw_hash_and_row_drift(tmp_path: Path, completed_fixture) -> None:
    """SCENARIO-CL-7371-READINESS: Cold replay compares terminal rows with raw bytes."""

    artifact = deepcopy(completed_fixture[-1])
    raw = tmp_path / experiment.RAW_DIR
    raw.mkdir(parents=True)
    experiment.atomic_json(
        raw / "synthetic_evidence.json",
        {"rows": [], "erasure_witness_rows": []},
    )
    experiment.atomic_json(raw / "authority_attacks.json", {"authority_attack_rows": []})
    artifact["source_artifact_hashes"] = {experiment.MANIFEST_PATH.as_posix(): "sha256:" + "0" * 64}
    errors = experiment.cold_reload_errors(artifact, tmp_path)
    assert any(error.startswith("raw_hash_mismatch:") for error in errors)
    assert "raw_rows_mismatch" in errors
    assert "raw_witnesses_mismatch" in errors
    assert "raw_attacks_mismatch" in errors


def test_entrypoint_helpers_preserve_receipts_and_raw_bytes(tmp_path: Path) -> None:
    """REQ-CL-7371: Entrypoint helpers retain exact commands, timing, and raw hashes."""

    receipt = experiment._normalized_receipt(
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
    assert receipt["return_code"] == 0
    assert receipt["resolved_imports"]
    assert {
        row.spec.name for row in experiment._terminal_commands(tmp_path / "candidate.json")
    } == {
        "cold_artifact_replay",
        "independent_reducer",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    }
    started = time.monotonic()
    span = experiment._span("test", started, started)
    assert span["duration_s"] >= 0
    hashes = experiment._write_measured_raw(tmp_path, "sha256:" + "2" * 64, [], [], [], [])
    assert len(hashes) == 3
    assert all((tmp_path / path).is_file() for path in hashes)
    args = experiment.parse_args(["--date", "20260917", "--output", str(tmp_path / "x.json")])
    assert args.date == "20260917"


def test_scoped_plan_validator_rejects_drift_and_full_suite(tmp_path: Path) -> None:
    """REQ-CL-7371: Command drift and invented full-suite checks are explicit failures."""

    commands = experiment.scoped_command_plan(experiment.REPO_ROOT, tmp_path / "private")
    duplicate = [*commands, commands[0]]
    assert "required_command_names_changed" in experiment.validate_scoped_command_plan(
        experiment.REPO_ROOT, duplicate
    )
    full = [*commands, CommandSpec("full_python_suite", ("true",), "repository")]
    assert "full_python_suite_forbidden" in experiment.validate_scoped_command_plan(
        experiment.REPO_ROOT, full
    )


def test_scoped_plan_has_only_actual_exp7358_commands(tmp_path: Path) -> None:
    """REQ-CL-7371: Required receipts equal the bounded Exp7358 command plan."""

    commands = experiment.scoped_command_plan(experiment.REPO_ROOT, tmp_path / "private")

    assert experiment.validate_scoped_command_plan(experiment.REPO_ROOT, commands) == []
    assert {command.name for command in commands} == set(experiment.REQUIRED_CHECK_NAMES)
    assert all(command.name != "full_python_suite" for command in commands)
