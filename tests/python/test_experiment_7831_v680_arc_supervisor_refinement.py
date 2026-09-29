"""REQ-ARC-7831: current redirect ledger evidence and CLI contract."""

from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from carnot import experiment_7831_v680_arc_supervisor_refinement as exp


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _source(root: Path, name: str, rows: list[dict], *, flagged: bool = False) -> dict:
    raw = Path(f"raw/{name}.json")
    digest = _write(root / raw, {"rows": rows})
    producer = Path(f"results/{name}.json")
    _write(
        root / producer,
        {
            "honest_verdict": "complete_disqualified_fixture"
            if flagged
            else "complete_null_fixture",
            "verdict_class": "disqualified" if flagged else "null",
            "flagged_adversarial": flagged,
            "source_artifact_hashes": {raw.as_posix(): digest},
        },
    )
    return {"producer": producer.as_posix(), "raw": raw.as_posix()}


def _receipt(game: str, *, mode: str = "applied", helped: bool = False) -> dict:
    redirect = {
        "arm": "drop_goal_bias",
        "action_index": 20,
        "level": 0,
        "resolved_by_levelup": helped,
        "actions_to_levelup": 4 if helped else None,
    }
    receipt = {
        "mode": mode,
        "enabled": mode == "applied",
        "stagnations_unredirected": 0,
    }
    if mode == "applied":
        receipt["redirects"] = [redirect]
        receipt["arm_outcomes"] = {"drop_goal_bias": {"fired": 1, "helped": int(helped)}}
    else:
        receipt["would_have_redirects"] = [redirect]
        receipt["would_have_arm_outcomes"] = {"drop_goal_bias": {"fired": 1, "helped": int(helped)}}
    return {
        "game": game,
        "seed": 1,
        "termination": {"reason": "level_up" if helped else "action_limit"},
        "trajectory_supervisor": receipt,
    }


# SCENARIO-ARC-7831-INVENTORY: copied rows and flagged sources cannot add benefit.
def test_inventory_authenticates_deduplicates_and_quarantines(tmp_path: Path) -> None:
    applied = _receipt("g1", helped=True)
    qualified = _source(tmp_path, "qualified", [applied, applied, _receipt("g2", mode="shadow")])
    flagged = _source(tmp_path, "flagged", [_receipt("g3", helped=True)], flagged=True)
    goal_only = _source(tmp_path, "goal", [{"game": "g4", "goal_firing": True}])
    result = exp.inspect_candidates(tmp_path, [qualified, flagged, goal_only], set())
    assert len(result["eligible"]) == 2
    assert [r["disposition"] for r in result["inventory"]] == [
        "eligible",
        "disqualified",
        "unqualified_no_redirect_schema",
    ]
    assert result["inventory"][0]["duplicate_rows"] == 1
    assert result["schema_missing_rows"] == []
    assert exp.reduce_eligible(result["eligible"])["new_source_count"] == 2


# SCENARIO-ARC-7831-INVENTORY: exact producer hashes are an eligibility gate.
def test_source_mutation_and_missing_path_are_visible(tmp_path: Path) -> None:
    source = _source(tmp_path, "changed", [_receipt("g1")])
    (tmp_path / source["raw"]).write_text("{}", encoding="utf-8")
    result = exp.inspect_candidates(tmp_path, [source, {"producer": "missing.json"}], set())
    assert [r["disposition"] for r in result["inventory"]] == ["hash_mismatch", "missing"]
    assert result["eligible"] == []


# SCENARIO-ARC-7831-OUTCOMES: only applied complete receipts count as firings.
def test_applied_shadow_censoring_and_missing_outcomes(tmp_path: Path) -> None:
    helped = _receipt("g1", helped=True)
    censored = _receipt("g2")
    shadow = _receipt("g3", mode="shadow")
    malformed = _receipt("g4")
    del malformed["trajectory_supervisor"]["arm_outcomes"]
    source = _source(tmp_path, "mixed", [helped, censored, shadow, malformed])
    result = exp.inspect_candidates(tmp_path, [source], set())
    assert len(result["schema_missing_rows"]) == 1
    assert result["schema_missing_rows"][0]["field"] == "trajectory_supervisor.arm_outcomes"
    reduced = exp.reduce_eligible(result["eligible"])
    assert len(reduced["rows"]) == 2
    assert reduced["arm_statistics"]["drop_goal_bias"]["uncensored"] == 1
    assert reduced["arm_statistics"]["drop_goal_bias"]["censored"] == 1
    assert reduced["arm_statistics"]["drop_goal_bias"]["helped"] == 1


# SCENARIO-ARC-7831-NULL: no new receipt differs from a complete zero-firing receipt.
def test_distinct_terminal_nulls(tmp_path: Path) -> None:
    empty = exp.reduce_eligible([])
    assert empty["honest_verdict"] == "complete_null_no_new_eligible_receipts"
    source = _source(tmp_path, "zero", [_receipt("g1", mode="shadow")])
    inventory = exp.inspect_candidates(tmp_path, [source], set())
    assert exp.reduce_eligible(inventory["eligible"])["honest_verdict"] == (
        "complete_null_no_firings_nothing_to_refine"
    )


# SCENARIO-ARC-7831-OUTCOMES: the preregistered screen needs three games.
def test_screen_uses_uncensored_firings_and_wilson_bound() -> None:
    rows = []
    for index in range(8):
        item = _receipt(f"g{index % 3}", helped=True)
        item["seed"] = index
        rows.append({"row": item, "path": f"p{index}", "sha256": f"h{index}"})
    result = exp.reduce_eligible(rows)
    assert result["recommendation_rows"][0]["kind"] == "priority_consideration"
    for item in rows:
        item["row"]["trajectory_supervisor"]["redirects"][0]["resolved_by_levelup"] = False
        item["row"]["trajectory_supervisor"]["redirects"][0]["actions_to_levelup"] = None
        item["row"]["termination"]["reason"] = "stopped"
        item["row"]["trajectory_supervisor"]["arm_outcomes"]["drop_goal_bias"]["helped"] = 0
    assert exp.reduce_eligible(rows)["recommendation_rows"][0]["kind"] == "retire_candidate"


# SCENARIO-ARC-7831-VALIDATION: no undeclared or mutable child is trusted.
def test_real_cli_dispatcher_and_cold_log_replay(tmp_path: Path) -> None:
    manifest = json.loads((exp.ROOT / exp.MANIFEST_REL).read_text(encoding="utf-8"))
    assert [row["name"] for row in manifest["commands"]] == [
        "worktree_imports",
        "focused_pytest",
        "changed_module_coverage",
        "changed_module_coverage_report",
        "ruff_check",
        "ruff_format",
        "changed_module_mypy",
        "scoped_spec_coverage",
        "all_python_tests",
        "repository_health_180s",
    ]
    assert [row["class"] for row in manifest["commands"]] == ["required"] * 9 + ["diagnostic"]
    for row in manifest["commands"]:
        assert row["argv"] and isinstance(row["argv"], list)
    log = tmp_path / "first.log"
    log.write_bytes(b"completed\n")
    receipt = {
        "name": "focused_pytest",
        "command_argv": manifest["commands"][1]["argv"],
        "exit_code": 0,
        "log_path": str(log),
        "log_sha256": "sha256:" + hashlib.sha256(log.read_bytes()).hexdigest(),
    }
    assert exp.validate_child_receipts([receipt], manifest, require_all=False) == []
    mutated = dict(receipt, name="extra_child")
    assert "undeclared_child:extra_child" in exp.validate_child_receipts(
        [receipt, mutated], manifest, require_all=False
    )
    log.write_bytes(b"mutated\n")
    assert "log_hash_mismatch:focused_pytest" in exp.validate_child_receipts(
        [receipt], manifest, require_all=False
    )
    later = tmp_path / "later.log"
    later.write_bytes(b"completed\n")
    retry = dict(receipt, log_path=str(later))
    assert exp.validate_child_receipts([retry], manifest, require_all=False) == []
    cli = exp.ROOT / "scripts/experiments/experiment_7831_v680_arc_supervisor_refinement.py"
    outcome = subprocess.run(
        [str(exp.ROOT / ".venv/bin/python"), str(cli), "--check-manifest"],
        cwd=exp.ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert outcome.returncode == 0, outcome.stderr
    assert json.loads(outcome.stdout)["command_names"] == [
        row["name"] for row in manifest["commands"]
    ]


# SCENARIO-ARC-7831-VALIDATION: check the real CLI's replay error on byte mutation.
def test_cli_rejects_mutated_sealed_log(tmp_path: Path) -> None:
    manifest = json.loads((exp.ROOT / exp.MANIFEST_REL).read_text(encoding="utf-8"))
    log = tmp_path / "sealed.log"
    log.write_bytes(b"before")
    candidate = tmp_path / "candidate.json"
    _write(
        candidate,
        {
            "validation_receipts": [
                {
                    "name": "focused_pytest",
                    "command_argv": manifest["commands"][1]["argv"],
                    "exit_code": 0,
                    "log_path": str(log),
                    "log_sha256": "sha256:" + hashlib.sha256(log.read_bytes()).hexdigest(),
                }
            ]
        },
    )
    log.write_bytes(b"after")
    cli = exp.ROOT / "scripts/experiments/experiment_7831_v680_arc_supervisor_refinement.py"
    outcome = subprocess.run(
        [str(exp.ROOT / ".venv/bin/python"), str(cli), "--cold-replay", str(candidate)],
        cwd=exp.ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert outcome.returncode != 0
    assert "log_hash_mismatch:focused_pytest" in outcome.stdout


# SCENARIO-ARC-7831-VALIDATION: the runner waits for closed logs and seals retries.
def test_runner_seals_completed_logs_with_distinct_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(exp, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    manifest = {
        "commands": [
            {
                "name": "probe",
                "argv": [sys.executable, "-c", "print('done')"],
                "class": "required",
            }
        ]
    }
    first = exp._run_declared(manifest, 0.0)
    second = exp._run_declared(manifest, 0.0)
    assert first[0]["exit_code"] == second[0]["exit_code"] == 0
    assert first[0]["log_sha256"] == second[0]["log_sha256"]
    assert exp.validate_child_receipts(first, manifest, require_all=True) == []
    sealed = tmp_path / first[0]["log_path"]
    sealed.write_bytes(b"mutated")
    assert "log_hash_mismatch:probe" in exp.validate_child_receipts(
        first, manifest, require_all=True
    )


# SCENARIO-ARC-7831-OUTCOMES: exhaustion is text, never an automatic new arm.
def test_exhaustion_only_writes_general_requirement() -> None:
    row = _receipt("g1", mode="shadow")
    receipt = row["trajectory_supervisor"]
    receipt["stagnations_unredirected"] = 2
    receipt["arms_enabled"] = ["drop_goal_bias"]
    receipt["arms_used"] = ["drop_goal_bias"]
    result = exp.reduce_eligible([{"row": row, "path": "p", "sha256": "h"}])
    assert result["general_mechanism_requirement"]
    assert result["recommendation_rows"] == []
    assert result["stagnations_unredirected"] == 2


# SCENARIO-ARC-7831-OUTCOMES: exact arm outcome companions prevent false credits.
@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (lambda r: r.update(redirects="wrong"), "trajectory_supervisor.redirects_or_arm_outcomes"),
        (lambda r: r.update(redirects=[None]), "trajectory_supervisor.redirects[]"),
        (
            lambda r: r.update(redirects=[{"arm": "drop_goal_bias"}]),
            "trajectory_supervisor.redirects[].resolved_by_levelup",
        ),
        (
            lambda r: r.update(arm_outcomes={"drop_goal_bias": {"fired": 2, "helped": 0}}),
            "trajectory_supervisor.arm_outcomes",
        ),
    ],
)
def test_malformed_redirect_operand_is_named(mutate: object, field: str) -> None:
    receipt = _receipt("g1")["trajectory_supervisor"]
    mutate(receipt)
    assert exp._complete_receipt(receipt, "applied") == field


# SCENARIO-ARC-7831-INVENTORY: source formats and historical copies have exact dispositions.
def test_inventory_handles_nested_hash_and_historical_duplicate(tmp_path: Path) -> None:
    source = _source(tmp_path, "nested", [_receipt("g1")])
    producer = tmp_path / source["producer"]
    payload = json.loads(producer.read_text())
    raw_hash = payload["source_artifact_hashes"][source["raw"]]
    payload["source_artifact_hashes"][source["raw"]] = {"sha256": raw_hash.removeprefix("sha256:")}
    _write(producer, payload)
    first = exp.inspect_candidates(tmp_path, [source], set())
    assert first["inventory"][0]["disposition"] == "eligible"
    second = exp.inspect_candidates(tmp_path, [source], {raw_hash})
    assert second["inventory"][0]["disposition"] == "historical_duplicate"
    assert second["eligible"] == []


# SCENARIO-ARC-7831-NULL: missing external paths and schemas become blocked gates.
def test_artifact_blocked_gate_names_exact_operand(tmp_path: Path) -> None:
    source = _source(tmp_path, "bad", [_receipt("g1")])
    doc = json.loads((tmp_path / source["raw"]).read_text())
    del doc["rows"][0]["trajectory_supervisor"]["stagnations_unredirected"]
    raw_hash = _write(tmp_path / source["raw"], doc)
    producer = tmp_path / source["producer"]
    payload = json.loads(producer.read_text())
    payload["source_artifact_hashes"][source["raw"]] = raw_hash
    _write(producer, payload)
    inventory = exp.inspect_candidates(tmp_path, [source, {"producer": "absent.json"}], set())
    artifact = exp._artifact(inventory, exp.reduce_eligible(inventory["eligible"]), 0.0)
    assert artifact["verdict_class"] == "blocked"
    assert {row["field"] for row in artifact["gate_check_summary"]} == {
        "trajectory_supervisor.stagnations_unredirected",
        "path_or_authenticated_hash",
    }
    assert artifact["supervisor_inventory_ready_score"] == 0


# SCENARIO-ARC-7831-VALIDATION: current main publishes its own null with no gameplay.
def test_main_builds_terminal_null_from_current_inventory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        exp,
        "_run_declared",
        lambda manifest, started, reuse=None: [
            {"name": "probe", "command_argv": ["x"], "class": "required", "duration_s": 0.1}
        ],
    )
    monkeypatch.setattr(exp, "validate_child_receipts", lambda *args, **kwargs: [])
    monkeypatch.setattr(exp, "RESULT_REL", tmp_path / "result.json")
    assert exp.main(["--date", "20260928"]) == 0
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["honest_verdict"] == "complete_null_no_new_eligible_receipts"
    assert result["new_source_count"] == 0
    assert result["model_invocation_counts"]["calls"] == 0


# SCENARIO-ARC-7831-VALIDATION: command identity and failure gates remain strict.
def test_dispatch_rejects_duplicate_argv_failure_and_missing(tmp_path: Path) -> None:
    log = tmp_path / "sealed.log"
    log.write_bytes(b"ok")
    manifest = {"commands": [{"name": "one", "argv": ["true"], "class": "required"}]}
    receipt = {
        "name": "one",
        "command_argv": ["false"],
        "exit_code": 1,
        "log_path": str(log),
        "log_sha256": "sha256:" + hashlib.sha256(log.read_bytes()).hexdigest(),
    }
    assert exp.validate_child_receipts([receipt, receipt], manifest, require_all=True) == [
        "argv_mismatch:one",
        "failed_required:one",
        "duplicate_child:one",
        "argv_mismatch:one",
        "failed_required:one",
    ]
    assert exp.validate_child_receipts([], manifest, require_all=True) == ["missing_required:one"]


# SCENARIO-ARC-7831-INVENTORY: absent raw bytes and no producer hash cannot be eligible.
def test_inventory_missing_raw_and_unproven_hash(tmp_path: Path) -> None:
    source = _source(tmp_path, "rawmissing", [_receipt("g1")])
    (tmp_path / source["raw"]).unlink()
    unknown = _source(tmp_path, "unproven", [_receipt("g2")])
    producer = tmp_path / unknown["producer"]
    payload = json.loads(producer.read_text())
    payload["source_artifact_hashes"] = {}
    _write(producer, payload)
    result = exp.inspect_candidates(tmp_path, [source, unknown], set())
    assert [item["disposition"] for item in result["inventory"]] == [
        "missing",
        "hash_mismatch",
    ]
    assert exp._expected_hash({}, "anything") is None


# SCENARIO-ARC-7831-VALIDATION: the same CLI parser handles check and replay paths.
def test_main_check_replay_and_bad_date(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert exp.main(["--check-manifest"]) == 0
    assert "command_names" in capsys.readouterr().out
    candidate = tmp_path / "candidate.json"
    _write(candidate, {"validation_receipts": []})
    assert exp.main(["--cold-replay", str(candidate)]) == 0
    assert json.loads(capsys.readouterr().out)["errors"] == []
    with pytest.raises(SystemExit):
        exp.main([])
    monkeypatch.setattr(exp, "_run_declared", lambda manifest, started, reuse=None: [])
    monkeypatch.setattr(exp, "validate_child_receipts", lambda *args, **kwargs: ["failed"])
    monkeypatch.setattr(exp, "RESULT_REL", tmp_path / "failed.json")
    assert exp.main(["--date", "20260928"]) == 1
    failed = json.loads((tmp_path / "failed.json").read_text())
    assert failed["verdict_class"] == "disqualified"
    assert failed["supervisor_inventory_ready_score"] == 1
    assert failed["acceptance_gate_results"]["readiness"] == 0


# SCENARIO-ARC-7831-VALIDATION: module execution reaches the thin CLI path.
def test_module_execution_uses_main(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["module", "--check-manifest"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_module(exp.__name__, run_name="__main__")
    assert stopped.value.code == 0


# SCENARIO-ARC-7831-VALIDATION: only the owned late child is killed at deadline.
def test_runner_heartbeats_and_stops_owned_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class LateChild:
        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

        def wait(self, timeout: int | None = None) -> int:
            if timeout is not None:
                raise subprocess.TimeoutExpired("owned", timeout)
            return -9

    monkeypatch.setattr(exp, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(exp.subprocess, "Popen", lambda *args, **kwargs: LateChild())
    counter = [0]

    def clock() -> float:
        counter[0] += 50
        return float(counter[0])

    phases: list[str] = []
    monkeypatch.setattr(exp.time, "monotonic", clock)
    monkeypatch.setattr(exp.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(exp, "progress", lambda started, phase, completed: phases.append(phase))
    manifest = {
        "commands": [{"name": "repository_health_180s", "argv": ["sleep"], "class": "diagnostic"}]
    }
    receipt = exp._run_declared(manifest, 0.0)
    assert receipt[0]["exit_code"] == -9
    assert "child_outstanding:repository_health_180s" in phases


# SCENARIO-ARC-7831-VALIDATION: a preexisting sealed name with wrong bytes fails.
def test_runner_rejects_sealed_name_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(exp, "ROOT", tmp_path)
    sealed = tmp_path / "results/raw/experiment_7831_v680_arc_supervisor_refinement/sealed_logs"
    sealed.mkdir(parents=True)
    (sealed / "abc.log").write_bytes(b"other")
    real_sha = exp.sha256_file
    monkeypatch.setattr(
        exp,
        "sha256_file",
        lambda path: "sha256:abc" if path.name.endswith("probe.log") else "sha256:wrong",
    )
    manifest = {
        "commands": [
            {"name": "probe", "argv": [sys.executable, "-c", "print(1)"], "class": "required"}
        ]
    }
    with pytest.raises(ValueError, match="sealed log collision"):
        exp._run_declared(manifest, 0.0)
    monkeypatch.setattr(exp, "sha256_file", real_sha)


# SCENARIO-ARC-7831-VALIDATION: frozen basetemp argv resolves to a new attempt.
def test_runner_creates_pytest_parent_before_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(exp, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    alias = Path("/tmp/carnot-exp7831-attempt/fixed")
    previous_target = alias.resolve() if alias.is_symlink() else None
    if alias.is_symlink():
        alias.unlink()
    manifest = {
        "commands": [
            {
                "name": "probe",
                "argv": [
                    sys.executable,
                    "-c",
                    "import pathlib; assert pathlib.Path('/tmp/carnot-exp7831-attempt/fixed').is_dir()",
                    "--basetemp=/tmp/carnot-exp7831-attempt/fixed/focused",
                ],
                "class": "required",
            }
        ]
    }
    try:
        first = exp._run_declared(manifest, 0.0)
        first_target = alias.resolve()
        second = exp._run_declared(manifest, 0.0)
        assert first[0]["exit_code"] == second[0]["exit_code"] == 0
        assert alias.resolve() != first_target
        assert exp.validate_child_receipts(second, manifest, require_all=True) == []
    finally:
        if alias.is_symlink():
            alias.unlink()
        if previous_target is not None:
            alias.symlink_to(previous_target, target_is_directory=True)


# SCENARIO-ARC-7831-VALIDATION: the runner cannot replace someone else's directory.
def test_runner_refuses_unowned_pytest_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(exp, "ROOT", tmp_path)
    alias = Path("/tmp/carnot-exp7831-attempt/fixed")
    alias.parent.mkdir(parents=True, exist_ok=True)
    previous_target = alias.resolve() if alias.is_symlink() else None
    if alias.is_symlink():
        alias.unlink()
    alias.mkdir(exist_ok=True)
    manifest = {
        "commands": [
            {
                "name": "probe",
                "argv": ["--basetemp=/tmp/carnot-exp7831-attempt/fixed/focused"],
                "class": "required",
            }
        ]
    }
    try:
        with pytest.raises(ValueError, match="not owned"):
            exp._run_declared(manifest, 0.0)
    finally:
        alias.rmdir()
        if previous_target is not None:
            alias.symlink_to(previous_target, target_is_directory=True)


# SCENARIO-ARC-7831-VALIDATION: a completed failed full suite stays failed on retry.
def test_main_reuses_sealed_negative_full_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = json.loads((exp.ROOT / exp.MANIFEST_REL).read_text())
    full = next(row for row in manifest["commands"] if row["name"] == "all_python_tests")
    log = tmp_path / "full.log"
    log.write_bytes(b"F at 3 percent")
    prior_path = tmp_path / "prior.json"
    _write(
        prior_path,
        {
            "validation_receipts": [
                {
                    "name": "all_python_tests",
                    "command_argv": full["argv"],
                    "exit_code": 1,
                    "log_path": str(log),
                    "log_sha256": "sha256:" + hashlib.sha256(log.read_bytes()).hexdigest(),
                }
            ]
        },
    )
    seen: list[str] = []

    def fake_run(manifest: object, started: float, reuse: dict) -> list[dict]:
        seen.extend(reuse)
        return []

    monkeypatch.setattr(exp, "RESULT_REL", prior_path)
    monkeypatch.setattr(exp, "_run_declared", fake_run)
    assert exp.main(["--date", "20260928"]) == 1
    assert seen == ["all_python_tests"]


# SCENARIO-ARC-7831-VALIDATION: reusing a sealed unit does not spawn a child.
def test_runner_reuses_completed_receipt_without_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(exp, "ROOT", tmp_path)
    monkeypatch.setattr(
        exp.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("unexpected child")
    )
    manifest = {"commands": [{"name": "probe", "argv": ["x"], "class": "diagnostic"}]}
    receipt = {"name": "probe", "command_argv": ["x"], "exit_code": 1}
    result = exp._run_declared(manifest, 0.0, {"probe": receipt})
    assert result[0]["reused_from_prior_attempt"] is True
