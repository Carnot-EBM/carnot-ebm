"""Tests for REQ-CL-7344 and SCENARIO-CL-7344-*.

All result and raw paths use pytest temporary directories. The tests do not
rewrite the repository research record.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7344_v645_executor_fixture as mod
from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES


ROOT = Path(__file__).resolve().parents[2]


def _receipt(name: str, passed: bool = True) -> dict[str, Any]:
    """Build one complete bounded-command receipt for injected validation."""

    return {
        "name": name,
        "command": f"check {name}",
        "command_argv": ["check", name],
        "scope": "test_fixture",
        "exit_code": 0 if passed else 1,
        "duration_s": 0.01,
        "log_path": f"/tmp/{name}.log",
        "log_sha256": "sha256:" + "1" * 64,
        "passed": passed,
        "timed_out": False,
        "output_tail": "ok" if passed else "failed",
    }


def _validation(**kwargs: Any) -> dict[str, Any]:
    """Return passing scoped receipts without nesting pytest inside pytest."""

    assert kwargs["test_paths"] == [str(mod.TEST_PATH)]
    assert kwargs["changed_modules"] == [str(mod.MODULE_PATH)]
    assert kwargs["static_paths"] == [str(mod.WRAPPER_PATH)]
    history = list(kwargs["historical_failures"])
    return {
        "validation_receipts": [_receipt(name) for name in REQUIRED_CHECK_NAMES],
        "required_checks_passed": True,
        "failed_required_commands": [],
        "repository_health": {
            "status": "degraded_open",
            "incident_open": True,
            "historical_failures": history,
            "historical_failure_count": len(history),
            "unresolved_collection_error_observation_count": 0,
            "affects_required_checks": False,
        },
    }


def _terminal(_root: Path, candidate: Path, _raw_dir: Path) -> list[dict[str, Any]]:
    """Require a measured candidate before returning terminal check receipts."""

    assert candidate.is_file()
    return [_receipt(name) for name in mod.TERMINAL_CHECK_NAMES]


@pytest.fixture(scope="module")
def built_fixture(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    """Run the real learner and evaluator processes once for all E2E assertions."""

    root = tmp_path_factory.mktemp("exp7344-output")
    artifact = mod.build_artifact(
        ROOT,
        mod.RUN_DATE,
        output_path=root / "result.json",
        raw_dir=root / "raw",
        checkpoint_path=root / "checkpoint.json",
        validation_runner=_validation,
        terminal_runner=_terminal,
    )
    return root, artifact


def test_scenario_cl_7344_panel_seals_streams_twins_and_private_labels(tmp_path: Path) -> None:
    """SCENARIO-CL-7344-PANEL fixes cohort counts, warmups, twins, and seals."""

    public_path = tmp_path / "public.json"
    private_path = tmp_path / "private/private.json"
    receipt_path = tmp_path / "private/receipt.json"
    receipt = mod.build_manifests(public_path, private_path, receipt_path)
    public = json.loads(public_path.read_text(encoding="utf-8"))
    private = json.loads(private_path.read_text(encoding="utf-8"))

    assert receipt["development_stream_count"] == 32
    assert len(public["development_streams"]) == 32
    assert all(len(stream["requests"]) == 12 for stream in public["development_streams"])
    assert {
        cohort: sum(stream["cohort"] == cohort for stream in public["development_streams"])
        for cohort in mod.COHORTS
    } == {cohort: 8 for cohort in mod.COHORTS}
    development_requests = [
        request for stream in public["development_streams"] for request in stream["requests"]
    ]
    assert len({request["request_id"] for request in development_requests}) == 384
    assert all(
        request["warmup"] is (index < 4)
        for stream in public["development_streams"]
        for index, request in enumerate(stream["requests"])
    )

    assert len(public["public_model_streams"]) == 8
    assert all(len(stream["requests"]) == 4 for stream in public["public_model_streams"])
    assert sum(len(stream["requests"]) for stream in public["public_model_streams"]) == 32
    assert all(
        row["warmup"] is (index < 2)
        for stream in public["public_model_streams"]
        for index, row in enumerate(stream["requests"])
    )
    assert len(public["live_proposal_panel"]) == 32
    assert all(
        row["original"]["request_id"] != row["twin"]["request_id"]
        and set(row["renaming_map"]) == set(row["original"]["activities"])
        for row in public["live_proposal_panel"]
    )
    public_text = public_path.read_text(encoding="utf-8")
    assert "private_rules" not in public_text
    assert "acceptance_witness" not in public_text
    assert private["public_manifest_sha256"] == mod.sha256_file(public_path)
    assert json.loads(receipt_path.read_text()) == receipt


def test_scenario_cl_7344_controls_cover_hostile_and_lifecycle_cases() -> None:
    """SCENARIO-CL-7344-CONTROLS retains every required fail-closed case."""

    result = mod.build_control_rows()
    rows = result["rows"]
    assert result["passed"] is True
    assert {row["control"] for row in rows} == {
        "accepted_witness",
        "rejected_witness",
        "malformed_reply",
        "stale_version",
        "compound_only_conflict",
        "learner_restart",
        "mismatched_request_id",
    }
    assert all(row["passed"] for row in rows)
    assert result["zero_private_rule_reads"] is True
    assert result["unsound_learned_atom_count"] == 0
    compound = next(row for row in rows if row["control"] == "compound_only_conflict")
    assert compound["observed"] == {
        "full_accepted": False,
        "pair_projections_accepted": True,
        "learned_atom_count": 0,
    }


def test_scenario_cl_7344_scope_uses_only_explicit_files(tmp_path: Path) -> None:
    """SCENARIO-CL-7344-SCOPE forbids broad pytest and creates base-temp parents."""

    basetemp = tmp_path / "private" / "pytest"
    commands = mod.scoped_command_plan(ROOT, basetemp, tmp_path / ".coverage")
    assert basetemp.is_dir()
    assert [command.name for command in commands] == list(REQUIRED_CHECK_NAMES)
    pytest_commands = [
        command
        for command in commands
        if command.name in {"focused_pytest", "changed_module_coverage"}
    ]
    assert all(str(mod.TEST_PATH) in command.argv for command in pytest_commands)
    assert all("tests/python" not in command.argv for command in pytest_commands)
    assert all("run_full_python_suite" not in " ".join(command.argv) for command in commands)
    assert all("-n" in command.argv and "0" in command.argv for command in pytest_commands)
    assert all("--no-cov" in command.argv for command in pytest_commands)

    source = (ROOT / mod.WRAPPER_PATH).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert [(node.module, [alias.name for alias in node.names]) for node in imported] == [
        ("carnot.experiment_7344_v645_executor_fixture", ["main"])
    ]


def test_scenario_cl_7344_boundary_runs_real_boolean_processes(
    built_fixture: tuple[Path, dict[str, Any]],
) -> None:
    """SCENARIO-CL-7344-BOUNDARY proves the real separate-process E2E."""

    _root, artifact = built_fixture
    boundary = artifact["isolation_controls"]
    assert boundary["process_separation_passed"] is True
    assert (
        len(
            {
                boundary["orchestrator_pid"],
                boundary["learner_pid"],
                boundary["evaluator_pid"],
            }
        )
        == 3
    )
    assert boundary["learner_private_access_count"] == 0
    assert boundary["learner_private_rule_message_count"] == 0
    assert boundary["evaluator_response_keys"] == ["accepted", "query_id"]
    assert boundary["hostile_process_security_claim"] is False
    assert artifact["private_manifest_receipt"]["private_opened_by_learner"] is False
    assert artifact["sample_size_budget"]["development"]["completed"] == 384
    assert len(artifact["rows"]) >= 384


def test_scenario_cl_7344_terminal_replaces_old_score_zero_with_current_evidence(
    built_fixture: tuple[Path, dict[str, Any]],
) -> None:
    """SCENARIO-CL-7344-TERMINAL separates V644 health from current readiness."""

    _root, artifact = built_fixture
    assert artifact["status"] == "complete"
    assert artifact["executor_fixture_ready_score"] == 1
    assert artifact["executor_value_score"] == 0
    assert artifact["promotion_score"] == 0
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["flagged_adversarial"] is False
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["verifier_is_oracle"] is True
    assert artifact["repository_health"]["affects_required_checks"] is False
    historical = artifact["repository_health"]["historical_failures"]
    assert any(
        row["exit_code"] == -15 and row["source_experiment_id"] == 7330 for row in historical
    )
    assert artifact["public_manifest_path"] == artifact["public_manifest"]["path"]
    assert artifact["cohort_manifest"]["synthetic_streams_per_cohort"] == 8
    assert artifact["cohort_manifest"]["public_model_requests"] == 32
    assert artifact["gate_check_summary"]["passed"] is True
    assert all(row["passed"] for row in artifact["acceptance_gate_results"].values())
    assert mod.validate_artifact(artifact, root=ROOT) == []


def test_req_cl_7344_independent_reducer_detects_changed_raw_rows(
    built_fixture: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    """REQ-CL-7344 binds raw rows, process receipts, and terminal claims."""

    _root, artifact = built_fixture
    assert mod.independent_reduce(artifact) == []
    changed = deepcopy(artifact)
    changed["rows"][0]["executor_calls"] += 1
    assert "rows_hash_mismatch" in mod.independent_reduce(changed)
    changed = deepcopy(artifact)
    changed["source_artifact_hashes"][str(mod.MODULE_PATH)] = "sha256:" + "0" * 64
    assert "source_hash_mismatch" in mod.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "reproducibility_checksum_invalid" in mod.validate_artifact(changed, root=ROOT)
    assert mod.validate_artifact([], root=ROOT) == ["artifact_mapping_required"]

    target = tmp_path / "atomic.json"
    receipt = mod.write_artifact(target, artifact, root=ROOT)
    assert receipt["sha256"] == mod.sha256_file(target)
    assert json.loads(target.read_text()) == artifact


def test_req_cl_7344_affected_failure_disqualifies_but_history_does_not(
    built_fixture: tuple[Path, dict[str, Any]],
) -> None:
    """REQ-CL-7344 makes only required current failures disqualifying."""

    _root, artifact = built_fixture
    historical_only = deepcopy(artifact)
    historical_only["repository_health"]["status"] = "degraded_open"
    historical_only["repository_health"]["incident_open"] = True
    mod.apply_terminal_state(historical_only)
    assert historical_only["executor_fixture_ready_score"] == 1

    failed = deepcopy(artifact)
    receipt = next(row for row in failed["validation_receipts"] if row["name"] == "focused_pytest")
    receipt.update(passed=False, exit_code=1)
    mod.apply_terminal_state(failed)
    assert failed["executor_fixture_ready_score"] == 0
    assert failed["verdict_class"] == "disqualified"
    assert failed["honest_verdict"].startswith("complete_disqualified")


def test_req_cl_7344_missing_precondition_writes_terminal_block(tmp_path: Path) -> None:
    """REQ-CL-7344 writes a row-free blocked result before dependent work."""

    output = tmp_path / "blocked.json"
    artifact = mod.build_artifact(
        ROOT,
        mod.RUN_DATE,
        output_path=output,
        raw_dir=tmp_path / "raw",
        checkpoint_path=tmp_path / "checkpoint.json",
        validation_runner=_validation,
        terminal_runner=_terminal,
        precondition_overrides={"historical_artifact_available": False},
    )
    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["rows"] == []
    assert artifact["executor_fixture_ready_score"] == 0
    assert artifact["gate_check_summary"]["failed_check"] == "historical_artifact_available"
    assert json.loads(output.read_text()) == artifact
    assert mod.validate_artifact(artifact, root=ROOT) == []


def test_req_cl_7344_date_and_process_failure_paths(tmp_path: Path) -> None:
    """REQ-CL-7344 rejects a wrong date and retains a failed child receipt."""

    assert mod.date_argument(mod.RUN_DATE) == mod.RUN_DATE
    with pytest.raises(Exception, match=mod.RUN_DATE):
        mod.date_argument("20260915")
    receipt = mod.run_streamed_process(
        [str(ROOT / ".venv/bin/python"), "-u", "-c", "raise SystemExit(3)"],
        ROOT,
        tmp_path / "failed.log",
        name="expected_failure",
    )
    assert receipt["exit_code"] == 3
    assert receipt["passed"] is False


def test_req_cl_7344_manifest_and_scoped_runner_defenses(
    built_fixture: tuple[Path, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-7344 diagnoses every seal mutation and rejects broad commands."""

    _root, artifact = built_fixture
    manifest = json.loads(Path(artifact["public_manifest_path"]).read_text())
    mutations = {
        "development_stream_count": lambda row: row["development_streams"].pop(),
        "development_request_count": lambda row: row["development_streams"][0]["requests"].pop(),
        "development_cohorts": lambda row: row["development_streams"][0].update(cohort="changed"),
        "development_warmup": lambda row: row["development_streams"][0]["requests"][0].update(
            warmup=False
        ),
        "development_distinct_requests": lambda row: row["development_streams"][0][
            "requests"
        ].__setitem__(1, deepcopy(row["development_streams"][0]["requests"][0])),
        "public_model_streams": lambda row: row["public_model_streams"].pop(),
        "public_model_warmup": lambda row: row["public_model_streams"][0]["requests"][0].update(
            warmup=False
        ),
        "public_model_twins": lambda row: row["live_proposal_panel"][0]["twin"].update(
            request_id=row["live_proposal_panel"][0]["original"]["request_id"]
        ),
        "manifest_hash": lambda row: row.update(manifest_hash="sha256:bad"),
        "private_data_in_public_manifest": lambda row: row.update(private_rules={}),
    }
    for expected, mutate in mutations.items():
        changed = deepcopy(manifest)
        mutate(changed)
        assert expected in mod._manifest_errors(changed)

    not_object = tmp_path / "list.json"
    not_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected_object"):
        mod._load_object(not_object)

    original_loader = mod._load_object

    def missing_history(path: Path) -> dict[str, Any]:
        if path == ROOT / mod.HISTORY_PATH:
            raise FileNotFoundError(path)
        return original_loader(path)

    monkeypatch.setattr(mod, "_load_object", missing_history)
    preconditions, _hashes = mod.collect_preconditions(
        ROOT, tmp_path / "out.json", tmp_path / "raw", tmp_path / "checkpoint.json"
    )
    assert (
        next(row for row in preconditions if row["check"] == "historical_artifact_available")[
            "available"
        ]
        is False
    )
    monkeypatch.setattr(mod, "_load_object", original_loader)

    monkeypatch.setattr(
        mod,
        "build_scoped_commands",
        lambda *_args, **_kwargs: [
            mod.CommandSpec("broad", ("pytest", "tests/python"), "forbidden")
        ],
    )
    with pytest.raises(ValueError, match="broad_validation_command"):
        mod.scoped_command_plan(ROOT, tmp_path / "broad", tmp_path / ".coverage")
    monkeypatch.undo()

    captured: dict[str, Any] = {}

    def scoped_runner(_root: Path, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"validation_receipts": [], "required_checks_passed": False}

    result = mod.run_affected_validation(
        ROOT,
        tmp_path / "validation",
        historical_failures=[{"resolved": False}],
        scoped_runner=scoped_runner,
    )
    assert result["required_checks_passed"] is False
    assert captured["basetemp"].is_dir()
    assert captured["historical_failures"] == [{"resolved": False}]

    names: list[str] = []

    def command_runner(_root: Path, commands: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        names.extend(command.name for command in commands)
        return [_receipt(command.name) for command in commands]

    monkeypatch.setattr(mod, "run_commands", command_runner)
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}", encoding="utf-8")
    receipts = mod.run_terminal_validation(ROOT, candidate, tmp_path)
    assert names == list(mod.TERMINAL_CHECK_NAMES)
    assert [row["name"] for row in receipts] == list(mod.TERMINAL_CHECK_NAMES)


def test_req_cl_7344_raw_reducer_failure_modes(
    built_fixture: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    """REQ-CL-7344 independently detects missing, changed, short, and unsound raw evidence."""

    _root, artifact = built_fixture
    unavailable = deepcopy(artifact)
    unavailable["raw_evidence_paths"] = {}
    assert mod.independent_reduce(unavailable) == ["raw_evidence_unavailable"]

    learner = json.loads(Path(artifact["raw_evidence_paths"]["learner_evidence"]).read_text())
    controls = json.loads(Path(artifact["raw_evidence_paths"]["control_evidence"]).read_text())
    evaluator = json.loads(Path(artifact["raw_evidence_paths"]["evaluator_receipt"]).read_text())
    learner["rows"].pop()
    controls["unsound_learned_atom_count"] = 1
    evaluator["query_rows"].pop()
    learner_path = tmp_path / "learner.json"
    controls_path = tmp_path / "controls.json"
    evaluator_path = tmp_path / "evaluator.json"
    learner_path.write_text(json.dumps(learner), encoding="utf-8")
    controls_path.write_text(json.dumps(controls), encoding="utf-8")
    evaluator_path.write_text(json.dumps(evaluator), encoding="utf-8")
    changed = deepcopy(artifact)
    changed["raw_evidence_paths"].update(
        learner_evidence=str(learner_path),
        control_evidence=str(controls_path),
        evaluator_receipt=str(evaluator_path),
    )
    assert {
        "raw_rows_hash_mismatch",
        "rows_hash_mismatch",
        "evaluator_rows_hash_mismatch",
        "raw_row_count_mismatch",
        "unsound_learned_atoms",
    } <= set(mod.independent_reduce(changed))


def test_req_cl_7344_terminal_validator_rejects_named_mutations(
    built_fixture: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    """REQ-CL-7344 cold validation rejects identity, lifecycle, gate, and receipt drift."""

    _root, artifact = built_fixture
    missing = deepcopy(artifact)
    del missing["schema"]
    assert mod.validate_artifact(missing, root=ROOT) == ["missing_required_field:schema"]

    mutations = {
        "identity_invalid": ("milestone", "2026.09.000"),
        "lifecycle_invalid": ("run_date", "20260915"),
        "model_contract_invalid": ("model_invoked", True),
        "substrate_invalid": ("execution_venue", "remote"),
        "field_principles_invalid": ("field_principles", {}),
        "flagged_adversarial_invalid": ("flagged_adversarial", True),
        "ready_state_invalid": ("verdict_class", "positive"),
        "phase_spans_invalid": (
            "phase_spans",
            [{"start_elapsed_s": 2.0, "end_elapsed_s": 1.0}],
        ),
    }
    for expected, (field, value) in mutations.items():
        changed = deepcopy(artifact)
        changed[field] = value
        assert expected in mod.validate_artifact(changed, root=ROOT)

    failed_score = deepcopy(artifact)
    failed_score.update(verdict_class="disqualified", executor_fixture_ready_score=1)
    assert "failed_scores_invalid" in mod.validate_artifact(failed_score, root=ROOT)
    bad_receipt = deepcopy(artifact)
    bad_receipt["validation_receipts"][0]["command"] = ""
    assert "validation_receipt_shape_invalid" in mod.validate_artifact(bad_receipt, root=ROOT)
    missing_receipt = deepcopy(artifact)
    missing_receipt["validation_receipts"].pop()
    assert "validation_receipts_invalid" in mod.validate_artifact(missing_receipt, root=ROOT)

    blocked_shape = deepcopy(artifact)
    blocked_shape.update(
        status="blocked",
        verdict_class="blocked",
        honest_verdict="not_blocked_prefix",
        executor_fixture_ready_score=0,
    )
    assert {"blocked_shape_invalid", "blocked_verdict_invalid"} <= set(
        mod.validate_artifact(blocked_shape, root=ROOT)
    )
    with pytest.raises(ValueError, match="artifact_validation_failed"):
        mod.write_artifact(tmp_path / "invalid.json", blocked_shape, root=ROOT)
