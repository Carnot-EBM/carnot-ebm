"""Tests for REQ-REPORT-7627 native total-cost measurement."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7627_v665_native_cost as exp


ROOT = Path(__file__).resolve().parents[2]


def test_req_report_7627_reduces_fixed_three_arm_blocks() -> None:
    """REQ-REPORT-7627: fixed comparators and paired blocks control benefit."""

    rows = exp.synthetic_timing_rows()
    reduced = exp.reduce_timing_rows(rows)

    assert len(rows) == 360
    assert reduced["complete"] is True
    assert reduced["independent_blocks"] == 120
    assert set(reduced["strata"]) == {"cold:1", "cold:8", "warm:1", "warm:8"}
    assert reduced["primary_comparator"] == "python_inprocess"
    assert reduced["equal_stratum_geometric_mean"]["python_over_direct_native"]["estimate"] > 1.1
    assert reduced["native_speed_benefit_score"] == 1
    assert reduced["nfr_10x_met"] is False
    assert all(row["bootstrap_draws"] == 2_000 for row in reduced["strata"].values())


@pytest.mark.parametrize(
    ("mutator", "error"),
    [
        (lambda rows: rows.pop(), "paired_row_count"),
        (
            lambda rows: rows[0].__setitem__("decision_parity", False),
            "parity_or_error_failure",
        ),
        (
            lambda rows: rows[0].__setitem__("durability_policy", "memory_only"),
            "durability_policy_mismatch",
        ),
        (
            lambda rows: rows[0].__setitem__("arm", "rust_jsonl"),
            "paired_arms",
        ),
    ],
)
def test_scenario_report_7627_reducer_rejects_incomplete_evidence(
    mutator: object, error: str
) -> None:
    """SCENARIO-REPORT-7627-REDUCTION: corrupt paired evidence fails closed."""

    rows = exp.synthetic_timing_rows()
    mutator(rows)
    with pytest.raises(ValueError, match=error):
        exp.reduce_timing_rows(rows)


def test_scenario_report_7627_primary_comparator_cannot_change_after_timing() -> None:
    """SCENARIO-REPORT-7627-REDUCTION: old Rust never replaces Python."""

    rows = exp.synthetic_timing_rows()
    for row in rows:
        if row["arm"] == "rust_jsonl":
            row["total_ns"] = 1
            row["numerator"] = 1
    reduced = exp.reduce_timing_rows(rows)

    assert reduced["primary_comparator"] == "python_inprocess"
    assert "rust_over_direct_native" in reduced["equal_stratum_geometric_mean"]


def test_scenario_report_7627_instrumentation_controls_are_paired() -> None:
    """SCENARIO-REPORT-7627-OVERHEAD: 40 units bound telemetry overhead."""

    rows = exp.synthetic_instrumentation_rows()
    reduced = exp.reduce_instrumentation_rows(rows)

    assert len(rows) == 80
    assert reduced["independent_blocks"] == 40
    assert reduced["comparator_selection_authority"] is False
    assert set(reduced["strata"]) == {"cold:1", "cold:8", "warm:1", "warm:8"}
    with pytest.raises(ValueError, match="instrumentation_row_count"):
        exp.reduce_instrumentation_rows(rows[:-1])


def test_scenario_report_7627_preconditions_authenticate_real_native_import() -> None:
    """SCENARIO-REPORT-7627-PRECONDITIONS: readiness includes an actual import."""

    context = exp.collect_preconditions(ROOT)

    assert context["blocker"] is None
    assert context["native_extension"].is_file()
    assert context["native_binding"].RustPortableRecalibrationService
    assert context["build_manifest"]["module_sha256"] == exp.sha256_file(
        context["native_extension"]
    )
    assert len(context["hardware_dispositions"]) == 6
    assert context["hardware_operations_issued"] == []


def test_scenario_report_7627_native_absence_preserves_hardware_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-7627-PRECONDITIONS: native block keeps board accounting."""

    original = exp.load_native_extension

    def unavailable(_path: Path) -> object:
        raise ImportError("test native absence")

    monkeypatch.setattr(exp, "load_native_extension", unavailable)
    context = exp.collect_preconditions(ROOT)
    monkeypatch.setattr(exp, "load_native_extension", original)
    artifact = exp.build_artifact(
        ROOT,
        [],
        [],
        preconditions=context,
        duration_s=0.01,
    )

    assert context["blocker"]["check"] == "actual_native_import"
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["paired_timing_rows"] == []
    assert len(artifact["hardware_dispositions"]) == 6
    assert set(artifact["gate_check_summary"][0]) >= {
        "check",
        "upstream",
        "path",
        "field",
        "operator",
        "expected",
        "observed",
    }


def test_req_report_7627_test_artifact_has_complete_terminal_contract() -> None:
    """REQ-REPORT-7627: all governed fields remain independently checkable."""

    artifact = exp.build_test_artifact(ROOT)

    assert exp.validate_artifact(artifact, root=ROOT) == []
    assert artifact["native_cost_valid_score"] == 1
    assert artifact["native_speed_benefit_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["target_model"] == "none:no_model_load"
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["execution_venue"] == "host"
    assert artifact["invocation_counts"] == exp.ZERO_INVOCATIONS
    assert artifact["sample_size_budget"]["paired_blocks"]["observed"] == 120
    assert artifact["sample_size_budget"]["instrumentation_blocks"]["observed"] == 40
    assert artifact["repository_health"]["affects_required_checks"] is False
    assert {row["category"] for row in artifact["acceptance_gate_results"]} == {
        "validity",
        "readiness",
        "benefit",
        "retention",
        "freshness",
    }
    assert all(key in artifact["field_principles"] for key in artifact)


@pytest.mark.parametrize(
    ("mutator", "error"),
    [
        (lambda value: value.pop("rows"), "required_fields:rows"),
        (lambda value: value.__setitem__("MODEL_SPECS", [{}]), "model_contract"),
        (lambda value: value.__setitem__("target_model", "historical"), "model_contract"),
        (lambda value: value.__setitem__("model_invoked", True), "model_contract"),
        (lambda value: value.__setitem__("execution_venue", "free text"), "execution_venue"),
        (
            lambda value: value.__setitem__("native_speed_benefit_score", 0),
            "speed_score",
        ),
        (lambda value: value.__setitem__("nfr_10x_met", True), "nfr_10x_met"),
        (lambda value: value.__setitem__("hardware_operations_issued", ["probe"]), "hardware"),
        (
            lambda value: value["repository_health"].__setitem__("affects_required_checks", True),
            "repository_health_scope",
        ),
        (lambda value: value.__setitem__("flagged_adversarial", True), "flagged"),
        (lambda value: value.__setitem__("reproducibility_checksum", "bad"), "checksum"),
    ],
)
def test_req_report_7627_validator_rejects_claim_drift(mutator: object, error: str) -> None:
    """REQ-REPORT-7627: claim, custody, and no-model mutations fail closed."""

    artifact = exp.build_test_artifact(ROOT)
    mutator(artifact)

    assert any(error in item for item in exp.validate_artifact(artifact))


def test_req_report_7627_independent_replay_rejects_reduction_drift(tmp_path: Path) -> None:
    """REQ-REPORT-7627: a fresh reader recomputes ratios from rows."""

    artifact = exp.build_test_artifact(ROOT)
    candidate = tmp_path / "candidate.json"
    exp.atomic_json(candidate, artifact)
    assert exp.cold_replay(candidate)["valid"] is True
    assert exp.independent_replay(candidate)["valid"] is True

    artifact["timing_reduction"]["independent_blocks"] = 1
    exp.atomic_json(candidate, artifact)
    replay = exp.independent_replay(candidate)
    assert replay["valid"] is False
    assert replay["error"] == "timing_reduction_drift"

    candidate.write_text("not-json", encoding="utf-8")
    assert exp.cold_replay(candidate)["valid"] is False


def test_req_report_7627_hardware_dispositions_keep_scopes_separate() -> None:
    """SCENARIO-REPORT-7627-HARDWARE: historical hardware claims do not broaden."""

    rows = {row["hardware"]: row for row in exp.hardware_dispositions(ROOT)}

    assert rows["KV260"]["k_max"] == 5
    assert rows["KV260"]["current_execution"] is False
    assert rows["PolarFire"]["fpga_sampling_measured"] is False
    assert rows["GateMate"]["last_observed"] == "0xffffffff"
    assert rows["local_rtx3090_pair"]["current_model_invocation"] is False
    assert rows["Extropic_TSU"]["availability"] == "unavailable"
    assert rows["AMD_XDNA"]["availability"] == "unavailable"


def test_req_report_7627_commands_are_scoped_and_private(tmp_path: Path) -> None:
    """REQ-REPORT-7627: validation names explicit files and private state."""

    commands = exp.build_validation_commands(ROOT, tmp_path)
    names = {command.name for command in commands}
    rendered = "\n".join(" ".join(command.argv) for command in commands)

    assert set(exp.VALIDATION_NAMES).issubset(names)
    assert exp.TEST_PATH.as_posix() in rendered
    assert exp.MODULE_PATH.as_posix() in rendered
    assert str(tmp_path) in rendered
    assert "scripts/research_conductor.py" not in rendered

    terminal = exp.terminal_commands(tmp_path / "candidate.json", ROOT)
    assert {command.name for command in terminal} == set(exp.TERMINAL_NAMES)


def test_req_report_7627_cli_modes_and_checksum(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-7627: the thin entrypoint exposes read-only terminal modes."""

    artifact = exp.build_test_artifact(ROOT)
    candidate = tmp_path / "candidate.json"
    exp.atomic_json(candidate, artifact)

    assert exp.main(["--cold-replay", str(candidate)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True
    assert exp.main(["--independent-replay", str(candidate)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True

    parsed = exp.parse_args(["--date", exp.RUN_DATE, "--output", str(candidate)])
    assert parsed.date == exp.RUN_DATE
    assert parsed.output == candidate
    copied = deepcopy(artifact)
    copied["duration_s"] = 99.0
    assert exp.reproducibility_checksum(copied) == artifact["reproducibility_checksum"]


def test_req_report_7627_reducer_helper_failures_are_explicit() -> None:
    """REQ-REPORT-7627: invalid bootstrap and pairing operands fail clearly."""

    with pytest.raises(ValueError, match="percentile_requires_values"):
        exp._percentile([], 0.5)
    assert exp._percentile([3.0], 0.5) == 3.0
    with pytest.raises(ValueError, match="paired_ratio_requires_30_blocks"):
        exp._ratio_interval([1.0], exp.RANDOM_SEED)
    with pytest.raises(ValueError, match="geometric_strata"):
        exp._equal_stratum_interval({}, exp.RANDOM_SEED)

    rows = exp.synthetic_timing_rows()
    for row in rows:
        if row["pair_id"] == "cold:1:0":
            row["pair_id"] = "cold:99:0"
    with pytest.raises(ValueError, match="paired_stratum_count"):
        exp.reduce_timing_rows(rows)

    controls = exp.synthetic_instrumentation_rows()
    controls[0]["telemetry_enabled"] = True
    with pytest.raises(ValueError, match="instrumentation_pairs"):
        exp.reduce_instrumentation_rows(controls)


def test_req_report_7627_source_verification_rejects_shape_and_bytes(tmp_path: Path) -> None:
    """REQ-REPORT-7627: source custody rejects malformed and stale receipts."""

    source = tmp_path / "source.txt"
    source.write_text("evidence", encoding="utf-8")
    value = {
        "source_artifact_hashes": {
            "bad-shape": "not-a-receipt",
            "bad-hash": {"path": str(source), "sha256": "sha256:" + "0" * 64},
        }
    }
    assert exp._verify_sources(value, ROOT) == [
        "source_receipt:bad-shape",
        "source_hash:bad-hash",
    ]


def test_req_report_7627_accepts_repo_relative_build_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-7627: a producer may retain its manifest below the worktree."""

    original = exp._load_object
    upstream = original(ROOT / exp.EXP7626_PATH)
    absolute = Path(upstream["native_build_manifest_path"])
    upstream["native_build_manifest_path"] = absolute.relative_to(ROOT).as_posix()

    def relative_manifest(path: Path) -> dict[str, object]:
        if path == ROOT / exp.EXP7626_PATH:
            return deepcopy(upstream)
        return original(path)

    monkeypatch.setattr(exp, "_load_object", relative_manifest)
    assert exp.collect_preconditions(ROOT)["blocker"] is None


def test_req_report_7627_small_runner_helpers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-REPORT-7627: phase and producer dispatch helpers keep exact outcomes."""

    span = exp._span("unit", 1.0, 0.5, 3)
    assert span["phase"] == "unit" and span["completed_units"] == 3
    receipts = [{"name": "a", "passed": True, "exit_code": 0}]
    assert exp._all_passed(receipts, ("a",)) is True
    assert exp._all_passed(receipts, ("a", "b")) is False

    called: list[tuple[Path, str, Path]] = []

    def fake_run(root: Path, date: str, output: Path) -> dict[str, object]:
        called.append((root, date, output))
        return {}

    monkeypatch.setattr(exp, "run_experiment", fake_run)
    assert (
        exp.main(["--root", str(ROOT), "--date", exp.RUN_DATE, "--output", str(tmp_path / "x")])
        == 0
    )
    assert called == [(ROOT.resolve(), exp.RUN_DATE, tmp_path / "x")]


@pytest.fixture(scope="module")
def complete_artifact() -> dict[str, object]:
    """Create one exact fixture for claim-drift mutation tests."""

    return exp.build_test_artifact(ROOT)


def _make_positive_without_benefit(value: dict[str, object]) -> None:
    rows = value["paired_timing_rows"]
    assert isinstance(rows, list)
    for row in rows:
        if row["arm"] == "python_inprocess":
            row["total_ns"] = 50_000
            row["numerator"] = 50_000
    reduction = exp.reduce_timing_rows(rows)
    value["timing_reduction"] = reduction
    value["native_speed_benefit_score"] = 0
    value["nfr_10x_met"] = False
    value["verdict_class"] = "positive"


@pytest.mark.parametrize(
    ("mutator", "error"),
    [
        (lambda value: value.__setitem__("verdict_class", "unknown"), "verdict_class"),
        (lambda value: value.__setitem__("honest_verdict", "not-terminal"), "honest_verdict"),
        (lambda value: value["paired_timing_rows"].pop(), "reduction:paired_row_count"),
        (
            lambda value: value["timing_reduction"].__setitem__("independent_blocks", 1),
            "timing_reduction",
        ),
        (
            lambda value: value["instrumentation_reduction"].__setitem__("independent_blocks", 1),
            "instrumentation_reduction",
        ),
        (lambda value: value.__setitem__("native_cost_valid_score", 0), "valid_score"),
        (lambda value: value["hardware_dispositions"].pop(), "hardware_identity"),
        (
            lambda value: value["hardware_dispositions"][0].__setitem__("k_max", 6),
            "hardware_kv260",
        ),
        (
            lambda value: value["hardware_dispositions"][1].__setitem__(
                "fpga_sampling_measured", True
            ),
            "hardware_polarfire",
        ),
        (
            lambda value: value["hardware_dispositions"][2].__setitem__(
                "last_observed", "different"
            ),
            "hardware_gatemate",
        ),
        (
            lambda value: value["hardware_dispositions"][3].__setitem__(
                "current_model_invocation", True
            ),
            "hardware_gpu",
        ),
        (
            lambda value: value["hardware_dispositions"][4].__setitem__(
                "availability", "available"
            ),
            "hardware_prospective",
        ),
        (lambda value: value.__setitem__("acceptance_gate_results", []), "acceptance_gates"),
        (lambda value: value.__setitem__("upstream_e2e_results", {}), "upstream_e2e"),
        (lambda value: value.__setitem__("validation_receipts", []), "validation_receipts"),
        (lambda value: value.__setitem__("field_principles", {}), "field_principles"),
        (_make_positive_without_benefit, "positive_without_benefit"),
    ],
)
def test_req_report_7627_validator_covers_terminal_mutations(
    complete_artifact: dict[str, object], mutator: object, error: str
) -> None:
    """REQ-REPORT-7627: each terminal claim is derived from exact evidence."""

    value = deepcopy(complete_artifact)
    mutator(value)
    assert error in exp.validate_artifact(value)


def test_req_report_7627_blocked_validator_and_replay(
    complete_artifact: dict[str, object], tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7627-PRECONDITIONS: blocked artifacts keep exact operands."""

    context = exp.collect_preconditions(ROOT)
    context["blocker"] = {
        "check": "missing_native",
        "upstream": "fixture",
        "path": "/missing",
        "field": "exists",
        "operator": "eq",
        "expected": True,
        "observed": False,
    }
    blocked = exp.build_blocked_artifact(ROOT, context, duration_s=0.01)
    candidate = tmp_path / "blocked.json"
    exp.atomic_json(candidate, blocked)
    assert exp.independent_replay(candidate) == {"valid": True, "blocked": True}

    blocked["gate_check_summary"] = []
    blocked["paired_timing_rows"] = [{}]
    errors = exp.validate_artifact(blocked)
    assert "blocked_gate_summary" in errors
    assert "blocked_measurement" in errors

    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")
    assert exp.independent_replay(invalid)["error"] == "artifact_unreadable"

    broken = deepcopy(complete_artifact)
    broken["paired_timing_rows"] = []
    exp.atomic_json(candidate, broken)
    assert exp.independent_replay(candidate)["error"] == "paired_row_count"

    drift = deepcopy(complete_artifact)
    drift["instrumentation_reduction"]["independent_blocks"] = 1
    exp.atomic_json(candidate, drift)
    assert exp.independent_replay(candidate)["error"] == "instrumentation_reduction_drift"
