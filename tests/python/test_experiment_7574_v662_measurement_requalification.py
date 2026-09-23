"""Tests for REQ-CL-7574 and REQ-ARC-WMTE-7574."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_7574_v662_measurement_requalification as exp


def _passing_receipts() -> list[dict]:
    return [
        {"name": name, "passed": True, "exit_code": 0, "timed_out": False}
        for name in exp.REQUIRED_RECEIPT_NAMES
    ]


def _tiny_evidence(tmp_path: Path) -> dict:
    contrasts = [
        exp.reduce_fixture_contrast(
            fixture="improving",
            baseline_predictions=[0.8, 0.2],
            candidate_predictions=[0.9, 0.1],
            labels=[1, 0],
            seed=7574001,
            provenance="private_test",
        )
    ]
    rows = [row for contrast in contrasts for row in contrast["rows"]]
    raw = tmp_path / "rows.json"
    exp.atomic_json(raw, {"prediction_label_rows": contrasts[0]["prediction_label_rows"]})
    return {
        "rows": rows,
        "fixture_contrasts": [exp.public_contrast(row) for row in contrasts],
        "raw_prediction_rows": exp.sidecar_reference(raw, tmp_path),
        "numerical_qualification": {
            "passed": True,
            "maximum_objective_delta": 0.0,
            "independent_solver_error": 0.0,
        },
        "lifecycle_qualification": {
            "passed": True,
            "duplicate_feedback_rejection_count": 3,
            "restart_mismatch_count": 0,
        },
        "benchmark_receipt": {
            "bootstrap_replicates_completed": 1000,
            "order_replays_completed": 5000,
            "events_completed": 800000,
            "fits_2400_seconds_with_reserve": True,
        },
        "arc_custody": {"passed": True, "checks": []},
        "private_directory_control": {"passed": True},
    }


def test_signed_brier_rows_recompute_both_directions() -> None:
    """SCENARIO-CL-7574-ROWS keeps positive improvement direction explicit."""

    improving = exp.reduce_fixture_contrast(
        fixture="improving",
        baseline_predictions=[0.8, 0.2],
        candidate_predictions=[0.9, 0.1],
        labels=[1, 0],
        seed=1,
        provenance="test",
    )
    worsening = exp.reduce_fixture_contrast(
        fixture="worsening",
        baseline_predictions=[0.9, 0.1],
        candidate_predictions=[0.8, 0.2],
        labels=[1, 0],
        seed=2,
        provenance="test",
    )
    assert improving["signed_improvement"] > 0
    assert worsening["signed_improvement"] < 0
    for contrast in (improving, worsening):
        reduced = exp.independent_reduce_comparison_rows(contrast["rows"])
        assert reduced[contrast["fixture"]]["baseline_loss"] == pytest.approx(
            contrast["baseline_loss"]
        )
        assert reduced[contrast["fixture"]]["candidate_loss"] == pytest.approx(
            contrast["candidate_loss"]
        )
        assert reduced[contrast["fixture"]]["signed_improvement"] == pytest.approx(
            contrast["signed_improvement"]
        )
        assert all(
            row["metric_direction"] == "higher_signed_improvement_is_better"
            for row in contrast["rows"]
        )


def test_exp7561_rows_are_independently_requalified(tmp_path: Path) -> None:
    """SCENARIO-CL-7574-LIFECYCLE recomputes every fixture from raw rows."""

    evidence = exp.run_fixture_requalification(tmp_path)
    assert {row["fixture"] for row in evidence["fixture_contrasts"]} == {
        "calibration_shift",
        "no_shift",
        "recurrence",
    }
    assert all(row["signed_improvement"] > 0 for row in evidence["fixture_contrasts"])
    assert evidence["lifecycle_qualification"]["passed"] is True
    assert evidence["raw_prediction_rows"]["sha256"].startswith("sha256:")
    reduced = exp.independent_reduce_comparison_rows(evidence["rows"])
    for contrast in evidence["fixture_contrasts"]:
        assert reduced[contrast["fixture"]]["signed_improvement"] == pytest.approx(
            contrast["signed_improvement"]
        )


def test_private_parent_failure_is_reproduced_then_repaired(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7574-PRIVATE preserves the diagnosed failure."""

    receipt = exp.reproduce_private_parent_failure(tmp_path)
    assert receipt["failure_reproduced"] is True
    assert receipt["observed_exception"] == "FileNotFoundError"
    assert receipt["repair_passed"] is True
    target = tmp_path / "commands" / "pytest" / "focused"
    command = exp.command_for_test_basetemp(target)
    prepared = exp.prepare_command_parent(command)
    assert prepared == target.parent
    assert target.parent.is_dir()


def test_arc_custody_binds_absolute_root_path_hash_and_identity(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7574-CUSTODY rejects absent and changed evidence."""

    missing = exp.check_arc_custody(tmp_path.resolve(), exp.EXP7562_PATH, exp.EXP7562_SHA256)
    assert missing["passed"] is False
    failure = next(row for row in missing["checks"] if row["passed"] is False)
    assert failure == {
        "check": "exp7562_artifact_exists",
        "upstream": exp.EXP7562_PATH.as_posix(),
        "path": str(tmp_path.resolve() / exp.EXP7562_PATH),
        "field": "path",
        "op": "eq",
        "expected": "readable_file",
        "observed": "missing",
        "passed": False,
    }

    custody = exp.check_arc_custody(exp.REPO_ROOT, exp.EXP7562_PATH, exp.EXP7562_SHA256)
    assert custody["passed"] is True
    assert custody["absolute_root"] == str(exp.REPO_ROOT)
    assert custody["artifact_path"] == str(exp.REPO_ROOT / exp.EXP7562_PATH)
    assert custody["artifact_sha256"] == exp.EXP7562_SHA256
    assert custody["experiment_id"] == "exp7562-arc-plan-lineage"


def test_blocked_no_run_artifact_is_complete_and_exact(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7574-CUSTODY forms an honest blocked artifact."""

    custody = exp.check_arc_custody(tmp_path.resolve(), exp.EXP7562_PATH, exp.EXP7562_SHA256)
    artifact = exp.build_blocked_artifact(custody["checks"], duration_s=0.002)
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["planned_inference_substrate_class"] == "no_model_load"
    assert artifact["MODEL_SPECS"] == []
    assert not any(artifact["invocation_counts"].values())
    summary = artifact["gate_check_summary"]
    assert summary["first_failure"] == custody["checks"][0]
    assert set(("upstream", "path", "field", "op", "expected", "observed")) <= set(
        summary["first_failure"]
    )
    assert exp.validate_artifact(artifact, root=tmp_path, require_terminal=False)["blocked"] is True


def test_artifact_keeps_readiness_and_benefit_independent(tmp_path: Path) -> None:
    """SCENARIO-CL-7574-TERMINAL makes circular fixture evidence non-empirical."""

    evidence = _tiny_evidence(tmp_path)
    artifact = exp.build_artifact(
        evidence=evidence,
        preconditions_checked=[],
        validation_receipts=_passing_receipts(),
        source_artifact_hashes={"fixture": "sha256:" + "a" * 64},
        duration_s=1.0,
        phase_spans=[],
        terminal=True,
    )
    assert artifact["honest_verdict"] == "complete_circular_positive_fixture_rows_requalified"
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["recalibration_ready_score"] == 1
    assert artifact["learning_compute_feasible_score"] == 1
    assert artifact["arc_runner_ready_score"] == 1
    assert artifact["positive_claim"] is False
    assert artifact["empirical_benefit_score"] == 0
    assert artifact["verifier_is_oracle"] is True
    assert (
        exp.validate_artifact(artifact, root=tmp_path, require_terminal=True)["fixture_claim_class"]
        == "circular_positive"
    )

    null_evidence = deepcopy(evidence)
    null_evidence["rows"][1]["raw_squared_error_numerator"] = 0.2
    null_evidence["rows"][1]["mean_brier"] = 0.1
    null_evidence["rows"][1]["signed_improvement_delta"] = -0.06
    null_evidence["fixture_contrasts"][0].update(
        candidate_loss=0.1,
        candidate_numerator=0.2,
        signed_improvement=-0.06,
    )
    null_artifact = exp.build_artifact(
        evidence=null_evidence,
        preconditions_checked=[],
        validation_receipts=_passing_receipts(),
        source_artifact_hashes={},
        duration_s=1.0,
        phase_spans=[],
        terminal=True,
    )
    assert null_artifact["verdict_class"] == "null"
    assert null_artifact["recalibration_ready_score"] == 1


def test_strict_lint_failure_is_reproduced_without_reader_change(tmp_path: Path) -> None:
    """SCENARIO-CL-7574-ROWS reproduces the prior sign failure exactly."""

    receipt = exp.reproduce_exp7561_strict_failure(exp.REPO_ROOT, tmp_path)
    assert receipt["failure_reproduced"] is True
    assert receipt["exit_code"] == 1
    assert "WINS_NOT_EXCEEDING_LOSSES" in receipt["output"]
    assert receipt["source_sha256"] == exp.sha256_file(exp.REPO_ROOT / exp.EXP7561_PATH)


def test_artifact_mutations_fail_cold_reduction(tmp_path: Path) -> None:
    """SCENARIO-CL-7574-TERMINAL binds raw arithmetic and current call custody."""

    evidence = _tiny_evidence(tmp_path)
    artifact = exp.build_artifact(
        evidence=evidence,
        preconditions_checked=[],
        validation_receipts=_passing_receipts(),
        source_artifact_hashes={},
        duration_s=1.0,
        phase_spans=[],
        terminal=True,
    )
    path = tmp_path / "artifact.json"
    exp.atomic_json(path, artifact)
    assert exp.cold_replay(path, root=tmp_path, require_terminal=True)["valid"] is True

    mutations = (
        ("MODEL_SPECS", ["forbidden"], "model_specs_not_empty"),
        ("inference_substrate_class", "model_full_generation", "substrate_invalid"),
        ("verdict_class", "positive", "verdict_mismatch"),
        ("recalibration_ready_score", 2, "score_invalid"),
    )
    for field, value, error in mutations:
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        with pytest.raises(ValueError, match=error):
            exp.validate_artifact(changed, root=tmp_path, require_terminal=True)

    changed = deepcopy(artifact)
    changed["rows"][1]["raw_squared_error_numerator"] = 99.0
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    with pytest.raises(ValueError, match="row_reduction_mismatch"):
        exp.validate_artifact(changed, root=tmp_path, require_terminal=True)


def test_validation_commands_are_scoped_and_prepare_each_parent(tmp_path: Path) -> None:
    """REQ-CL-7574 freezes affected files and private command paths."""

    commands = exp.build_validation_commands(exp.REPO_ROOT, tmp_path / "private")
    assert [row.name for row in commands] == list(exp.AFFECTED_CHECK_NAMES)
    for command in commands:
        parent = exp.prepare_command_parent(command)
        if parent is not None:
            assert parent.is_dir()
        joined = " ".join(command.argv)
        if command.name in {"focused_pytest", "changed_module_coverage"}:
            assert "-n 0" in joined
            assert "-o addopts=" in joined
            assert "--no-cov" in joined
    assert any("COVERAGE_FILE" in " ".join(row.argv) for row in commands)

    e2e = exp.build_arc_e2e_commands(exp.REPO_ROOT, tmp_path / "e2e")
    assert {row.name for row in e2e} == set(exp.ARC_E2E_NAMES)
    alternate = next(row for row in e2e if row.name == "alternate_cwd_e3_smoke")
    assert "CARNOT_ARC_DISABLE_INDUCTION=1" in alternate.argv
    assert "-C" in alternate.argv
    assert str(exp.REPO_ROOT / "scripts/arc_loop_solve.py") in alternate.argv


def test_cli_parsing_progress_and_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-CL-7574 keeps the public entrypoint thin and observable."""

    args = exp.parse_args(
        ["--root", str(exp.REPO_ROOT), "--date", "20260923", "--cold-replay", "x.json"]
    )
    assert args.root == exp.REPO_ROOT
    assert args.cold_replay == Path("x.json")
    exp.progress(0.0, "test", "boundary", completed_units=1)
    assert "phase=test event=boundary" in capsys.readouterr().out
    manifest = exp.write_affected_manifest(tmp_path / "manifest.json")
    loaded = json.loads(manifest.read_text(encoding="utf-8"))
    assert loaded["experiment_id"] == exp.EXPERIMENT_ID
    assert loaded["changed_modules"] == [exp.MODULE_PATH.as_posix()]


def test_precondition_and_numeric_error_branches(tmp_path: Path) -> None:
    """REQ-CL-7574 rejects malformed custody and invalid numeric rows."""

    checks, custody = exp.collect_preconditions(exp.REPO_ROOT)
    assert custody["passed"] is True
    assert all(row["passed"] is True for row in checks)

    malformed = tmp_path / exp.EXP7562_PATH
    malformed.parent.mkdir(parents=True)
    malformed.write_text("{", encoding="utf-8")
    broken = exp.check_arc_custody(tmp_path, exp.EXP7562_PATH, "sha256:wrong")
    assert broken["passed"] is False
    assert broken["experiment_id"] is None

    with pytest.raises(ValueError, match="binary_label_required"):
        exp._squared_error(0.5, 2)
    with pytest.raises(ValueError, match="finite_probability_required"):
        exp._squared_error(float("nan"), 1)
    with pytest.raises(ValueError, match="prediction_label_length_mismatch"):
        exp.reduce_fixture_contrast(
            fixture="bad",
            baseline_predictions=[],
            candidate_predictions=[],
            labels=[],
            seed=0,
            provenance="test",
        )


def test_comparison_reducer_rejects_ambiguous_rows() -> None:
    """SCENARIO-CL-7574-ROWS rejects duplicate, missing, and mismatched arms."""

    contrast = exp.reduce_fixture_contrast(
        fixture="unit",
        baseline_predictions=[0.8],
        candidate_predictions=[0.9],
        labels=[1],
        seed=1,
        provenance="test",
    )
    rows = contrast["rows"]
    with pytest.raises(ValueError, match="duplicate_fixture_arm"):
        exp.independent_reduce_comparison_rows([rows[0], rows[0]])
    with pytest.raises(ValueError, match="fixture_arm_set_invalid"):
        exp.independent_reduce_comparison_rows([rows[0]])
    wrong = deepcopy(rows)
    wrong[1]["raw_denominator"] = 2
    with pytest.raises(ValueError, match="fixture_denominator_invalid"):
        exp.independent_reduce_comparison_rows(wrong)


def test_parent_preparation_and_runner_cover_all_path_forms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-7574-PRIVATE prepares all private parents."""

    data = tmp_path / "data" / ".coverage"
    environment = tmp_path / "environment" / ".coverage"
    command = SimpleNamespace(
        argv=("coverage", f"--data-file={data}"),
        command_environment=(("COVERAGE_FILE", str(environment)),),
        name="probe",
    )
    assert exp.prepare_command_parent(command) is None
    assert data.parent.is_dir()
    assert environment.parent.is_dir()

    calls: list[str] = []

    def fake_run(_root: Path, commands: list, **_kwargs: object) -> list[dict]:
        calls.append(commands[0].name)
        return [{"name": commands[0].name, "passed": True, "exit_code": 0}]

    monkeypatch.setattr(exp.validation_scope, "run_commands", fake_run)
    commands = [
        exp.command_for_test_basetemp(tmp_path / "one" / "base"),
        exp.command_for_test_basetemp(tmp_path / "two" / "base"),
    ]
    receipts = exp.run_prepared_commands(exp.REPO_ROOT, commands, log_dir=tmp_path / "logs")
    assert calls == ["private_parent_probe", "private_parent_probe"]
    assert len(receipts) == 2
    assert all((tmp_path / name).is_dir() for name in ("one", "two"))

    terminal = exp.terminal_commands(exp.REPO_ROOT, tmp_path / "candidate.json")
    assert [row.name for row in terminal] == list(exp.TERMINAL_CHECK_NAMES)


def test_candidate_disqualified_and_blocked_guard_branches(tmp_path: Path) -> None:
    """REQ-CL-7574 classifies unfinished and invalid work without benefit claims."""

    evidence = _tiny_evidence(tmp_path)
    candidate = exp.build_artifact(
        evidence=evidence,
        preconditions_checked=[],
        validation_receipts=_passing_receipts(),
        source_artifact_hashes={},
        duration_s=0.1,
        phase_spans=[],
        terminal=False,
    )
    assert candidate["verdict_class"] == "partial"
    assert exp.validate_artifact(candidate, root=tmp_path, require_terminal=False)["valid"]

    receipts = _passing_receipts()[:-1]
    disqualified = exp.build_artifact(
        evidence=evidence,
        preconditions_checked=[],
        validation_receipts=receipts,
        source_artifact_hashes={},
        duration_s=0.1,
        phase_spans=[],
        terminal=True,
    )
    assert disqualified["verdict_class"] == "disqualified"
    assert (
        exp.validate_artifact(disqualified, root=tmp_path, require_terminal=True)["valid"] is False
    )

    failed_adversarial = deepcopy(_passing_receipts())
    row = next(item for item in failed_adversarial if item["name"] == "adversarial_verify")
    row.update(passed=False, exit_code=1)
    flagged = exp.build_artifact(
        evidence=evidence,
        preconditions_checked=[],
        validation_receipts=failed_adversarial,
        source_artifact_hashes={},
        duration_s=0.1,
        phase_spans=[],
        terminal=True,
    )
    assert flagged["flagged_adversarial"] is True

    with pytest.raises(ValueError, match="blocked_artifact_requires_failed_check"):
        exp.build_blocked_artifact([{"passed": True}], duration_s=0.0)


def test_sidecar_and_raw_reduction_mutations_fail(tmp_path: Path) -> None:
    """SCENARIO-CL-7574-TERMINAL catches raw evidence drift."""

    evidence = _tiny_evidence(tmp_path)
    artifact = exp.build_artifact(
        evidence=evidence,
        preconditions_checked=[],
        validation_receipts=_passing_receipts(),
        source_artifact_hashes={},
        duration_s=1.0,
        phase_spans=[],
        terminal=True,
    )
    bad_receipts = (
        ({"path": "missing", "sha256": "sha256:x", "bytes": 0}, "sidecar_missing"),
        (
            {**artifact["raw_prediction_rows"], "sha256": "sha256:wrong"},
            "sidecar_hash_mismatch",
        ),
        ({**artifact["raw_prediction_rows"], "bytes": 999}, "sidecar_size_mismatch"),
    )
    for receipt, error in bad_receipts:
        changed = deepcopy(artifact)
        changed["raw_prediction_rows"] = receipt
        with pytest.raises(ValueError, match=error):
            exp.independent_reduce(changed, root=tmp_path, require_terminal=True)

    nonobject = tmp_path / "nonobject.json"
    nonobject.write_text("[]", encoding="utf-8")
    receipt = exp.sidecar_reference(nonobject, tmp_path)
    changed = deepcopy(artifact)
    changed["raw_prediction_rows"] = receipt
    with pytest.raises(ValueError, match="sidecar_not_object"):
        exp.independent_reduce(changed, root=tmp_path, require_terminal=True)

    changed = deepcopy(artifact)
    changed["rows"] = "bad"
    with pytest.raises(ValueError, match="comparison_rows_invalid"):
        exp.independent_reduce(changed, root=tmp_path, require_terminal=True)

    changed = deepcopy(artifact)
    changed["fixture_contrasts"] = []
    with pytest.raises(ValueError, match="row_reduction_mismatch"):
        exp.independent_reduce(changed, root=tmp_path, require_terminal=True)

    changed = deepcopy(artifact)
    changed["rows"][0]["mean_brier"] = 99.0
    with pytest.raises(ValueError, match="row_mean_mismatch"):
        exp.independent_reduce(changed, root=tmp_path, require_terminal=True)

    raw_cases = (
        ({"prediction_label_rows": "bad"}, "raw_prediction_rows_invalid"),
        ({"prediction_label_rows": []}, "raw_fixture_set_mismatch"),
    )
    for payload, error in raw_cases:
        path = tmp_path / f"{error}.json"
        exp.atomic_json(path, payload)
        changed = deepcopy(artifact)
        changed["raw_prediction_rows"] = exp.sidecar_reference(path, tmp_path)
        with pytest.raises(ValueError, match=error):
            exp.independent_reduce(changed, root=tmp_path, require_terminal=True)

    raw_value = json.loads((tmp_path / "rows.json").read_text(encoding="utf-8"))
    raw_value["prediction_label_rows"][0]["candidate_prediction"] = 0.0
    drift = tmp_path / "drift.json"
    exp.atomic_json(drift, raw_value)
    changed = deepcopy(artifact)
    changed["raw_prediction_rows"] = exp.sidecar_reference(drift, tmp_path)
    with pytest.raises(ValueError, match="raw_prediction_reduction_mismatch"):
        exp.independent_reduce(changed, root=tmp_path, require_terminal=True)


def test_all_artifact_validation_guards_and_io(tmp_path: Path) -> None:
    """REQ-CL-7574 rejects every top-level claim mutation."""

    artifact = exp.build_artifact(
        evidence=_tiny_evidence(tmp_path),
        preconditions_checked=[],
        validation_receipts=_passing_receipts(),
        source_artifact_hashes={},
        duration_s=1.0,
        phase_spans=[],
        terminal=True,
    )
    mutations = (
        ("experiment_id", "wrong", "identity_mismatch"),
        ("model_invoked", True, "model_invoked_invalid"),
        ("invocation_counts", {}, "invocation_counts_invalid"),
        ("positive_claim", True, "empirical_claim_invalid"),
        ("verifier_is_oracle", False, "oracle_declaration_missing"),
        ("field_principles", {}, "field_principles_incomplete"),
        ("reproducibility_checksum", "wrong", "checksum_mismatch"),
        ("flagged_adversarial", True, "flagged_adversarial_mismatch"),
    )
    for field, value, error in mutations:
        changed = deepcopy(artifact)
        changed[field] = value
        if field != "reproducibility_checksum":
            changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        with pytest.raises(ValueError, match=error):
            exp.validate_artifact(changed, root=tmp_path, require_terminal=True)

    score = deepcopy(artifact)
    score["recalibration_ready_score"] = 0
    score["reproducibility_checksum"] = exp.reproducibility_checksum(score)
    with pytest.raises(ValueError, match="score_mismatch"):
        exp.validate_artifact(score, root=tmp_path, require_terminal=True)

    failed = exp._gate_row("missing", "upstream", tmp_path, "path", "file", None, False)
    blocked = exp.build_blocked_artifact([failed], duration_s=0.0)
    blocked["recalibration_ready_score"] = 1
    blocked["reproducibility_checksum"] = exp.reproducibility_checksum(blocked)
    with pytest.raises(ValueError, match="blocked_score_mismatch"):
        exp.validate_artifact(blocked, root=tmp_path, require_terminal=False)

    blocked = exp.build_blocked_artifact([failed], duration_s=0.0)
    del blocked["gate_check_summary"]["first_failure"]["upstream"]
    blocked["reproducibility_checksum"] = exp.reproducibility_checksum(blocked)
    with pytest.raises(ValueError, match="blocked_schema_invalid"):
        exp.validate_artifact(blocked, root=tmp_path, require_terminal=False)

    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="artifact_unreadable"):
        exp.cold_replay(missing)
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact_not_object"):
        exp.cold_replay(array)
    with pytest.raises(ValueError, match="artifact_not_object"):
        exp._load_artifact(array)
    object_path = tmp_path / "object.json"
    exp.atomic_json(object_path, {"value": 1})
    assert exp._load_artifact(object_path) == {"value": 1}

    manifest = exp.write_affected_manifest(tmp_path / "manifest.json")
    hashes = exp._source_hashes(exp.REPO_ROOT, exp.REPO_ROOT / exp.RAW_DIR / "placeholder")
    assert exp.MODULE_PATH.as_posix() in hashes
    assert manifest.is_file()
    tick = exp.time.monotonic()
    assert exp._span("test", tick, tick, 1)["completed_units"] == 1


def test_fixture_internal_mismatch_and_valid_null_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7574 covers internal disagreement and a valid worsening null."""

    original = exp.independent_reduce_comparison_rows

    def disagree(rows: list[dict]) -> dict[str, dict]:
        reduced = original(rows)
        reduced["calibration_shift"]["signed_improvement"] = 99.0
        return reduced

    monkeypatch.setattr(exp, "independent_reduce_comparison_rows", disagree)
    with pytest.raises(RuntimeError, match="fixture_independent_reduction_mismatch"):
        exp.run_fixture_requalification(tmp_path / "mismatch")
    monkeypatch.setattr(exp, "independent_reduce_comparison_rows", original)

    contrast = exp.reduce_fixture_contrast(
        fixture="worsening",
        baseline_predictions=[0.9, 0.1],
        candidate_predictions=[0.8, 0.2],
        labels=[1, 0],
        seed=9,
        provenance="private_test",
    )
    raw = tmp_path / "worsening.json"
    exp.atomic_json(raw, {"prediction_label_rows": contrast["prediction_label_rows"]})
    evidence = _tiny_evidence(tmp_path / "seed")
    evidence.update(
        rows=contrast["rows"],
        fixture_contrasts=[exp.public_contrast(contrast)],
        raw_prediction_rows=exp.sidecar_reference(raw, tmp_path),
    )
    artifact = exp.build_artifact(
        evidence=evidence,
        preconditions_checked=[],
        validation_receipts=_passing_receipts(),
        source_artifact_hashes={},
        duration_s=1.0,
        phase_spans=[],
        terminal=True,
    )
    assert artifact["verdict_class"] == "null"
    assert (
        exp.validate_artifact(artifact, root=tmp_path, require_terminal=True)["fixture_claim_class"]
        == "null"
    )
