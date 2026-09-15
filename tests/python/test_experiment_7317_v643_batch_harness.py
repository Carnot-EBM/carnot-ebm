"""Tests for the V643 batch harness and its scoped validation path.

Spec refs: REQ-VERIFY-7317 and SCENARIO-VERIFY-7317-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7317_v643_batch_harness as harness
from carnot.reporting import experiment_7303_validation_scope as scoped


ROOT = Path(__file__).resolve().parents[2]


def _fake_toolchain(root: Path) -> tuple[Path, dict[str, str]]:
    """Create real child executables that record argv and controlled failures."""

    bin_dir = root / ".venv/bin"
    bin_dir.mkdir(parents=True)
    ledger = root / "argv.jsonl"
    tool = bin_dir / "tool"
    tool.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

with Path(os.environ["EXP7317_ARGV_LEDGER"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({"program": Path(sys.argv[0]).name, "argv": sys.argv[1:]}) + "\\n")
if "--exp7303-resolve-imports" in sys.argv:
    marker = sys.argv.index("--exp7303-resolve-imports")
    root = Path(sys.argv[marker + 1])
    resolved = {
        name: str(root / "python" / Path(*name.split("."))).replace(".py", "") + ".py"
        for name in sys.argv[marker + 2:]
    }
    print(json.dumps({"resolved_imports": resolved}, sort_keys=True), flush=True)
needle = os.environ.get("EXP7317_FAIL_MATCH", "")
if needle and any(needle in value for value in sys.argv[1:]):
    print(f"controlled failure: {needle}", flush=True)
    raise SystemExit(int(os.environ.get("EXP7317_FAIL_CODE", "9")))
print("controlled pass", flush=True)
""",
        encoding="utf-8",
    )
    tool.chmod(0o755)
    for name in ("python", "pytest", "coverage", "ruff", "mypy"):
        (bin_dir / name).symlink_to(tool)
    return ledger, {"EXP7317_ARGV_LEDGER": str(ledger)}


def _materialize_scope(root: Path, target: dict[str, object]) -> None:
    """Create the explicit files required by one private scope fixture."""

    paths = [
        *target["test_paths"],
        *target["changed_modules"],
        *target["static_paths"],
    ]
    for relative in paths:
        path = root / str(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# private scope fixture\n", encoding="utf-8")


def _passing_validation(history: list[dict[str, object]] | None = None) -> dict[str, object]:
    """Build the complete named validation set used by reducer tests."""

    receipts = [
        {
            "name": name,
            "command": f"fixture:{name}",
            "command_argv": [name],
            "scope": "explicit_fixture",
            "exit_code": 0,
            "duration_s": 0.001,
            "log_path": f"/tmp/{name}.log",
            "log_sha256": "sha256:" + "a" * 64,
            "passed": True,
            "timed_out": False,
        }
        for name in scoped.REQUIRED_CHECK_NAMES
    ]
    return {
        **scoped.validation_outcome(receipts, history or []),
        "validation_entrypoint_receipt": {
            "runner": harness.SCOPED_RUNNER,
            "called": True,
            "test_paths": list(harness.validation_scope("harness")["test_paths"]),
            "changed_modules": list(harness.validation_scope("harness")["changed_modules"]),
            "static_paths": list(harness.validation_scope("harness")["static_paths"]),
            "legacy_launcher_called": False,
            "repository_wide_target_present": False,
        },
    }


def _required_files(root: Path) -> None:
    """Materialize current dependencies for one isolated orchestration run."""

    for relative in harness.REQUIRED_CURRENT_PATHS.values():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("current dependency\n", encoding="utf-8")
    (root / "ops/exclusion_manifest.yaml").write_text("retired: []\n", encoding="utf-8")


def test_scenario_verify_7317_scope_failure_uses_real_subprocess_argv(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7317-SCOPE keeps an affected child failure exact."""

    root = tmp_path / "failed-scope"
    target = harness.validation_scope("harness")
    _materialize_scope(root, target)
    ledger, environment = _fake_toolchain(root)
    environment["EXP7317_FAIL_MATCH"] = str(target["test_paths"][0])

    result = harness.run_scoped_checks(root, target, extra_env=environment)

    focused = next(row for row in result["validation_receipts"] if row["name"] == "focused_pytest")
    assert focused["exit_code"] == 9
    assert focused["passed"] is False
    assert focused["command_argv"][-2] == target["test_paths"][0]
    assert result["required_checks_passed"] is False
    assert harness.qualification_state(result, mechanics_passed=True)["ready_score"] == 0
    argv = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert len(argv) == len(scoped.REQUIRED_CHECK_NAMES)
    assert all("tests/python" not in row["argv"] for row in argv)


def test_scenario_verify_7317_scope_rejects_implicit_or_legacy_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-7317 rejects malformed scope before any child starts."""

    with pytest.raises(ValueError, match="unsupported validation consumer"):
        harness.validation_scope("old")
    target = harness.validation_scope("harness")
    invalid_targets = []
    for field, value in (
        ("runner", "old.runner"),
        ("test_paths", []),
        ("changed_modules", ["python/carnot/not_python.txt"]),
        ("static_paths", None),
        ("test_paths", ["tests/python"]),
    ):
        changed = deepcopy(target)
        changed[field] = value
        invalid_targets.append(changed)
    for changed in invalid_targets:
        assert harness._scope_errors(changed)
        with pytest.raises(ValueError, match="invalid scoped validation target"):
            harness.run_scoped_checks(tmp_path, changed)

    fake_result = _passing_validation()
    fake_result.pop("validation_entrypoint_receipt")
    fake_result["validation_receipts"][0]["command_argv"].append("tests/python")
    monkeypatch.setattr(scoped, "run_scoped_validation", lambda *_args, **_kwargs: fake_result)
    observed = harness.run_scoped_checks(tmp_path, target)
    assert observed["required_checks_passed"] is False
    assert observed["failed_required_commands"] == ["repository_wide_target"]


def test_scenario_verify_7317_missing_check_and_historical_health() -> None:
    """SCENARIO-VERIFY-7317-MISSING and -HEALTH keep the two states separate."""

    validation = _passing_validation(
        [
            {
                "experiment_id": "exp7306-batch-fixture",
                "milestone": "2026.09.642",
                "observed_at_utc": "2026-09-14T00:00:00Z",
                "terminal_class": "disqualified",
                "command": "pytest tests/python -q",
                "exit_code": 2,
                "collection_errors": [{"test_path": "tests/python/test_old.py"}],
                "resolved": False,
            }
        ]
    )
    validation["validation_receipts"] = validation["validation_receipts"][:-1]
    validation.update(scoped.reduce_required_checks(validation["validation_receipts"]))

    state = harness.qualification_state(validation, mechanics_passed=True)
    assert state == {
        "ready_score": 0,
        "status": "complete",
        "verdict_class": "disqualified",
        "honest_verdict": "complete_disqualified_batch_harness_validation_failed",
    }
    assert validation["missing_required_commands"] == ["scoped_spec_coverage"]
    healthy = _passing_validation(validation["repository_health"]["historical_failures"])
    assert healthy["required_checks_passed"] is True
    assert healthy["repository_health"]["status"] == "degraded_open"
    assert healthy["repository_health"]["affects_required_checks"] is False


def test_scenario_verify_7317_fixture_controls_and_pass_region() -> None:
    """SCENARIO-VERIFY-7317-FIXTURE, -CONTROLS, and -GATES preserve mechanics."""

    evidence = harness.build_harness_evidence()

    assert len(evidence["public"]["development_groups"]) == 8
    assert len(evidence["public"]["evaluation_groups"]) == 16
    assert len(evidence["rows"]) == 384
    assert {row["arm"] for row in evidence["rows"]} == set(harness.fixture.ARMS)
    assert all(row["passed"] for row in evidence["controls"])
    assert [row["control"] for row in evidence["controls"]][-3:] == [
        "mixed_source_versions",
        "absent_source_span",
        "unsupported_claim",
    ]
    assert all(row["passed"] for row in evidence["gates"])
    gate_map = {row["criterion"]: row for row in evidence["gates"]}
    assert gate_map["accuracy_difference_lower_vs_direct"]["observed"] >= -0.02
    assert gate_map["coverage_difference_lower_vs_direct"]["observed"] >= -0.02
    assert gate_map["full_cost_speedup_lower_vs_serial"]["observed"] >= 1.5
    assert gate_map["full_cost_speedup_lower_vs_direct"]["observed"] >= 1.5
    assert evidence["independent_reduction"]["semantic_mismatches"] == 0
    assert evidence["independent_reduction"]["stale_constraints_served"] == 0
    assert evidence["panel_manifest"]["development_group_count"] == 8
    assert evidence["panel_manifest"]["held_out_group_count"] == 16
    assert evidence["panel_manifest"]["labels_separate_from_public_input"] is True
    assert harness.validate_raw_reduction(evidence["rows"], evidence["independent_reduction"]) == []
    bad_reduction = deepcopy(evidence["independent_reduction"])
    bad_reduction["row_count"] = 0
    assert harness.validate_raw_reduction(evidence["rows"], bad_reduction) == ["raw_reduction"]


def test_scenario_verify_7317_panel_reuse_and_declared_fallbacks(tmp_path: Path) -> None:
    """REQ-VERIFY-7317 reuses valid bytes and rebuilds only invalid views."""

    public, scorer = harness.fixture.build_fixture()
    root = tmp_path / "panels"
    public_path = root / harness.V642_PUBLIC_PATH
    label_path = root / harness.V642_LABEL_PATH
    public_path.parent.mkdir(parents=True)
    public_path.write_text(json.dumps(public), encoding="utf-8")
    label_path.write_text(json.dumps(scorer), encoding="utf-8")
    reused_public, reused_scorer, receipt = harness.load_or_build_panel(root)
    assert reused_public == public and reused_scorer == scorer
    assert receipt["origin"] == "reused_v642_bytes"

    public_path.write_text("not json", encoding="utf-8")
    assert harness.load_or_build_panel(root)[2]["origin"].endswith("unreadable_history")
    public_path.write_text(json.dumps({**public, "development_groups": []}), encoding="utf-8")
    assert harness.load_or_build_panel(root)[2]["origin"].endswith("contract_mismatch")
    public_path.unlink()
    assert harness.load_or_build_panel(root)[2]["origin"].endswith("missing_history")

    malformed_public = deepcopy(public)
    malformed_public["development_groups"] = []
    malformed_public["evaluation_groups"][0]["source_versions"] = []
    malformed_public["evaluation_groups"][1]["claims"] = []
    malformed_public["authority"] = {"expected_decision": "bad"}
    malformed_scorer = {"labels": []}
    errors = harness.panel_contract_errors(malformed_public, malformed_scorer)
    assert {
        "development_groups",
        "source_versions",
        "claims_per_version",
        "public_authority_fields",
        "label_denominator",
    }.issubset(errors)
    duplicate_labels = deepcopy(scorer)
    duplicate_labels["labels"][-1]["unit_id"] = duplicate_labels["labels"][0]["unit_id"]
    assert "label_identity" in harness.panel_contract_errors(public, duplicate_labels)
    with pytest.raises(ValueError, match="panel contract"):
        harness.build_harness_evidence(malformed_public, scorer, draws=10)


def test_scenario_verify_7317_consumers_reject_failed_terminal_classes() -> None:
    """SCENARIO-VERIFY-7317-CONSUMERS returns explicit, fail-closed inputs."""

    evidence = harness.build_harness_evidence(draws=100)
    qualified = harness.assemble_artifact(
        evidence,
        _passing_validation(),
        preconditions=[],
        diagnostics=[],
        duration_s=1.0,
        phase_spans=[],
        timestamps={
            "started_at_utc": "2026-09-15T00:00:00Z",
            "completed_at_utc": "2026-09-15T00:00:01Z",
        },
    )
    canary = harness.build_canary_inputs(qualified)
    capture = harness.build_capture_inputs(qualified)
    assert canary["ok"] is True and len(canary["group_ids"]) == 2
    assert capture["ok"] is True and len(capture["group_ids"]) == 16
    for row in (canary, capture):
        scope_row = row["validation_scope"]
        assert scope_row["runner"] == harness.SCOPED_RUNNER
        assert all(path.endswith(".py") for path in scope_row["test_paths"])
        assert all(path.endswith(".py") for path in scope_row["changed_modules"])

    for terminal_class in ("blocked", "partial", "disqualified"):
        failed = deepcopy(qualified)
        failed["verdict_class"] = terminal_class
        result = harness.build_canary_inputs(failed)
        assert result["ok"] is False
        assert result["gate_check_summary"]["observed_value"] == terminal_class
    missing = harness.build_capture_inputs(None)
    assert missing["ok"] is False
    assert missing["gate_check_summary"]["observed_value"] == "missing_artifact"
    quarantined = deepcopy(qualified)
    quarantined["flagged_adversarial"] = True
    assert harness.build_canary_inputs(quarantined)["gate_check_summary"]["failed_check"] == (
        "dependency_quarantine"
    )
    not_ready = deepcopy(qualified)
    not_ready["batch_harness_ready_score"] = 0
    assert harness.build_capture_inputs(not_ready)["ok"] is False
    short = deepcopy(qualified)
    short["sealed_panel_manifest"]["groups"] = []
    assert harness.build_canary_inputs(short)["gate_check_summary"]["failed_check"] == (
        "sealed_group_denominator"
    )


def test_scenario_verify_7317_artifact_block_and_cold_validation(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7317-E2E checks terminal shape and exact failed gates."""

    evidence = harness.build_harness_evidence(draws=100)
    artifact = harness.assemble_artifact(
        evidence,
        _passing_validation(),
        preconditions=[],
        diagnostics=[],
        duration_s=1.25,
        phase_spans=[{"phase": "fixture", "duration_s": 1.25, "units": 384}],
        timestamps={
            "started_at_utc": "2026-09-15T00:00:00Z",
            "completed_at_utc": "2026-09-15T00:00:02Z",
        },
    )
    assert artifact["batch_harness_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert harness.validate_artifact(artifact, require_terminal_checks=False) == []
    corrupted = deepcopy(artifact)
    corrupted["MODEL_SPECS"] = ["model"]
    assert "MODEL_SPECS" in harness.validate_artifact(corrupted, require_terminal_checks=False)

    failed_evidence = deepcopy(evidence)
    failed_evidence["gates"][0]["passed"] = False
    failed_evidence["gates"][0]["observed"] = False
    null_artifact = harness.assemble_artifact(
        failed_evidence,
        _passing_validation(),
        preconditions=[],
        diagnostics=[],
        duration_s=1.0,
        phase_spans=[],
        timestamps=artifact["timestamps"],
    )
    assert null_artifact["verdict_class"] == "null"
    assert null_artifact["gate_check_summary"]["failed_check"] == "fixture_mechanics"

    failed_validation = _passing_validation()
    failed_validation["validation_receipts"] = failed_validation["validation_receipts"][:-1]
    failed_validation.update(
        scoped.reduce_required_checks(failed_validation["validation_receipts"])
    )
    disqualified = harness.assemble_artifact(
        evidence,
        failed_validation,
        preconditions=[],
        diagnostics=[],
        duration_s=1.0,
        phase_spans=[],
        timestamps=artifact["timestamps"],
    )
    assert disqualified["gate_check_summary"]["observed_value"] == "missing"

    failed_check = harness.gate_row(
        "dependency_terminal_class",
        "exp-current",
        "verdict_class",
        "not_failure_class",
        "disqualified",
        False,
    )
    blocked = harness.blocked_artifact(
        harness.RUN_DATE,
        [failed_check],
        duration_s=0.1,
        timestamps={
            "started_at_utc": "2026-09-15T00:00:00Z",
            "completed_at_utc": "2026-09-15T00:00:01Z",
        },
    )
    assert blocked["status"] == "blocked"
    assert blocked["batch_harness_ready_score"] == 0
    assert blocked["gate_check_summary"] == {
        "failed_check": "dependency_terminal_class",
        "upstream": "exp-current",
        "field": "verdict_class",
        "expected_value": "not_failure_class",
        "observed_value": "disqualified",
    }
    assert harness.validate_artifact(blocked, require_terminal_checks=False) == []
    output = tmp_path / "results/blocked.json"
    harness.write_json(tmp_path, output, blocked)
    assert json.loads(output.read_text(encoding="utf-8")) == blocked

    blocked_bad = deepcopy(blocked)
    blocked_bad["batch_harness_ready_score"] = 1
    blocked_bad["honest_verdict"] = "complete_wrong"
    blocked_bad["verdict_class"] = "wrong"
    blocked_bad["reproducibility_checksum"] = harness.artifact_checksum(blocked_bad)
    blocked_errors = harness.validate_artifact(blocked_bad, require_terminal_checks=False)
    assert {"verdict_class", "blocked_readiness", "honest_verdict"}.issubset(blocked_errors)

    mutations = {
        "rows": lambda value: value.__setitem__("rows", []),
        "row_arm_denominator": lambda value: value["rows"][0].__setitem__("arm", "wrong"),
        "independent_reduction": lambda value: value["independent_reduction"].__setitem__(
            "row_count", 0
        ),
        "batch_control_rows": lambda value: value.__setitem__("batch_control_rows", []),
        "acceptance_gate_results": lambda value: value["acceptance_gate_results"][0].__setitem__(
            "passed", False
        ),
        "validation_entrypoint_receipt": lambda value: value[
            "validation_entrypoint_receipt"
        ].__setitem__("legacy_launcher_called", True),
        "required_scoped_validation": lambda value: value.__setitem__("validation_receipts", []),
        "ready_verdict_class": lambda value: value.__setitem__("verdict_class", "null"),
        "honest_verdict": lambda value: value.__setitem__("honest_verdict", "wrong"),
        "status": lambda value: value.__setitem__("status", "partial"),
    }
    for expected_error, mutate in mutations.items():
        changed = deepcopy(artifact)
        mutate(changed)
        changed["reproducibility_checksum"] = harness.artifact_checksum(changed)
        assert expected_error in harness.validate_artifact(changed, require_terminal_checks=False)
    assert "terminal_checks" in harness.validate_artifact(artifact)


def test_scenario_verify_7317_authentication_diagnostics_and_health(tmp_path: Path) -> None:
    """REQ-VERIFY-7317 authenticates current gates and old failures separately."""

    diagnostics, hashes = harness.historical_diagnostics(ROOT)
    assert len(diagnostics) == 2
    assert all(row["readiness_gate"] is False for row in diagnostics)
    assert all(row["passed"] is True for row in diagnostics)
    assert any("full_python_suite" in key for key in hashes)
    assert harness.repository_health(ROOT)["status"] == "degraded_open"

    malformed = tmp_path / "malformed"
    _required_files(malformed)
    (malformed / "ops/exclusion_manifest.yaml").write_text("[", encoding="utf-8")
    checks, _hashes = harness.authenticate_inputs(malformed)
    assert next(row for row in checks if row["check"] == "exclusion_manifest")["passed"] is True

    for relative in harness.HISTORICAL_ARTIFACTS:
        path = malformed / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not json", encoding="utf-8")
    malformed_diagnostics, _ = harness.historical_diagnostics(malformed)
    assert all(row["check"] == "historical_artifact_parse" for row in malformed_diagnostics)

    health_path = malformed / harness.HEALTH_SOURCE
    health_path.parent.mkdir(parents=True, exist_ok=True)
    health_path.write_text("not json", encoding="utf-8")
    assert harness.repository_health(malformed)["status"] == "degraded_open"
    health_path.write_text("{}", encoding="utf-8")
    assert harness.repository_health(malformed)["status"] == "healthy"


def test_scenario_verify_7317_orchestration_fail_closed_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-7317 covers blocked, disqualified, and cold-invalid exits."""

    with pytest.raises(ValueError, match="--date must be"):
        harness.run_experiment(tmp_path, "20260914")

    blocked_root = tmp_path / "blocked-run"
    blocked_root.mkdir()
    blocked = harness.run_experiment(blocked_root, harness.RUN_DATE)
    assert blocked["status"] == "blocked"
    assert (blocked_root / harness.RESULT_PATH).is_file()

    invalid_blocked_root = tmp_path / "blocked-invalid"
    invalid_blocked_root.mkdir()
    with monkeypatch.context() as patch:
        patch.setattr(harness, "validate_artifact", lambda *_args, **_kwargs: ["forced"])
        with pytest.raises(ValueError, match="blocked artifact invalid"):
            harness.run_experiment(invalid_blocked_root, harness.RUN_DATE)

    failed_validation = _passing_validation()
    failed_validation["validation_receipts"][0]["passed"] = False
    failed_validation["validation_receipts"][0]["exit_code"] = 6
    failed_validation["failed_required_commands"] = [scoped.REQUIRED_CHECK_NAMES[0]]
    failed_validation["required_checks_passed"] = False
    checks = harness._validation_failure_checks(failed_validation)
    assert checks[0]["observed_value"] == 6

    public, scorer = harness.fixture.build_fixture()
    evidence = harness.build_harness_evidence(public, scorer, draws=10)

    def prepare(root: Path) -> dict[str, str]:
        root.mkdir()
        target = harness.validation_scope("harness")
        _materialize_scope(root, target)
        _required_files(root)
        return _fake_toolchain(root)[1]

    candidate_root = tmp_path / "candidate-invalid"
    candidate_env = prepare(candidate_root)
    real_validate = harness.validate_artifact
    with monkeypatch.context() as patch:
        patch.setattr(harness, "build_harness_evidence", lambda *_args, **_kwargs: evidence)
        patch.setattr(
            harness,
            "validate_artifact",
            lambda value, *, require_terminal_checks=True: (
                ["forced_candidate"]
                if value.get("status") == "complete" and not require_terminal_checks
                else real_validate(value, require_terminal_checks=require_terminal_checks)
            ),
        )
        with pytest.raises(ValueError, match="terminal candidate invalid"):
            harness.run_experiment(candidate_root, harness.RUN_DATE, extra_env=candidate_env)

    terminal_root = tmp_path / "terminal-failure"
    terminal_env = prepare(terminal_root)
    terminal_env["EXP7317_FAIL_MATCH"] = "adversarial_verify.py"
    with monkeypatch.context() as patch:
        patch.setattr(harness, "build_harness_evidence", lambda *_args, **_kwargs: evidence)
        disqualified = harness.run_experiment(
            terminal_root, harness.RUN_DATE, extra_env=terminal_env
        )
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["batch_harness_ready_score"] == 0

    final_root = tmp_path / "final-invalid"
    final_env = prepare(final_root)
    calls = 0

    def fail_final(value: dict[str, object], *, require_terminal_checks: bool = True) -> list[str]:
        nonlocal calls
        calls += 1
        if require_terminal_checks:
            return ["forced_final"]
        return real_validate(value, require_terminal_checks=False)

    with monkeypatch.context() as patch:
        patch.setattr(harness, "build_harness_evidence", lambda *_args, **_kwargs: evidence)
        patch.setattr(harness, "validate_artifact", fail_final)
        with pytest.raises(ValueError, match="terminal artifact invalid"):
            harness.run_experiment(final_root, harness.RUN_DATE, extra_env=final_env)
    assert calls >= 2


def test_scenario_verify_7317_full_orchestration_publishes_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7317-E2E runs scoped and terminal child calls once."""

    root = tmp_path / "orchestration"
    root.mkdir()
    target = harness.validation_scope("harness")
    _materialize_scope(root, target)
    _required_files(root)
    ledger, environment = _fake_toolchain(root)
    original_builder = harness.build_harness_evidence
    monkeypatch.setattr(
        harness,
        "build_harness_evidence",
        lambda *args, **kwargs: original_builder(*args, draws=100, **kwargs),
    )

    artifact = harness.run_experiment(root, harness.RUN_DATE, extra_env=environment)

    output = root / harness.RESULT_PATH
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["batch_harness_ready_score"] == 1
    assert artifact["validation_entrypoint_receipt"]["legacy_launcher_called"] is False
    assert harness.validate_artifact(artifact) == []
    calls = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert len(calls) == len(scoped.REQUIRED_CHECK_NAMES) + len(harness.TERMINAL_CHECK_NAMES)
    assert all("tests/python" not in row["argv"] for row in calls)

    assert harness.main(["--date", harness.RUN_DATE, "--check-artifact", str(output)]) == 0
    assert (
        harness.main(
            [
                "--date",
                harness.RUN_DATE,
                "--check-artifact",
                str(output),
                "--raw-rows",
                str(root / harness.ROW_PATH),
            ]
        )
        == 0
    )
    bad_output = root / "results/bad.json"
    bad = deepcopy(artifact)
    bad["schema"] = "bad"
    bad_output.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact check failed"):
        harness.main(["--date", harness.RUN_DATE, "--check-artifact", str(bad_output)])
    with monkeypatch.context() as patch:
        patch.setattr(harness, "find_repo_root", lambda **_kwargs: root)
        patch.setattr(
            harness,
            "run_experiment",
            lambda _root, _date: {"honest_verdict": "complete_fixture"},
        )
        assert harness.main(["--date", harness.RUN_DATE]) == 0
    with pytest.raises(SystemExit) as error:
        harness.main(["--date", "20260914"])
    assert error.value.code == 2
