"""Requalify the V661 numerical and ARC measurement paths.

The task uses analytical labels and an induction-disabled ARC smoke. It loads
no language model and makes no empirical benefit or solve claim.

Spec refs: REQ-CL-7574, SCENARIO-CL-7574-*, REQ-ARC-WMTE-7574, and
SCENARIO-ARC-WMTE-7574-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
from typing import Any

from carnot import experiment_7561_v661_recalibration_prototype as recalibration
from carnot.experiment_7358_v646_validation_contract import AffectedManifest
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


Json = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260923"
MILESTONE = "2026.09.662"
EXPERIMENT_ID = "exp7574-v662-measurement-requalification"
SCHEMA = "carnot.exp7574.v662.measurement_requalification.v1"
RESULT_PATH = Path("results/experiment_7574_v662_measurement_requalification.json")
RAW_DIR = Path("results/raw/experiment_7574_v662_measurement_requalification")
MODULE_PATH = Path("python/carnot/experiment_7574_v662_measurement_requalification.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7574_v662_measurement_requalification.py")
TEST_PATH = Path("tests/python/test_experiment_7574_v662_measurement_requalification.py")
CL_SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
ARC_SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
EXP7561_PATH = Path("results/experiment_7561_v661_recalibration_prototype.json")
EXP7570_PATH = Path("results/experiment_7570_v661_arc_live_lineage.json")
EXP7562_PATH = Path("results/experiment_7562_v661_arc_plan_lineage.json")
EXP7561_SHA256 = "sha256:52ccfe42f6e8398f066dc13b7d72ac4254f8dd776e76f19abc262ffea0c6d23e"
EXP7570_SHA256 = "sha256:36608c8d7421db623345d0e5af29c6c466340f627d2843635ca9cd25d8ce921d"
EXP7562_SHA256 = "sha256:8dd3ab0acab63a63a64c54fd02ed2acfe31b40576b7d77f099498e8a11135b98"

MODEL_SPECS: list[str] = []
ZERO_INVOCATION_COUNTS = {
    **{
        f"{kind}_{state}": 0
        for kind in ("model_loads", "forward_calls", "generation_calls")
        for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
    },
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
}

AFFECTED_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)
AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
ARC_E2E_NAMES = (
    "e2e_009",
    "e2e_010",
    "e2e_011",
    "e2e_012",
    "e2e_013",
    "alternate_cwd_e3_smoke",
)
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint",
    "fresh_process_cold_replay",
    "independent_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
REQUIRED_RECEIPT_NAMES = (*AFFECTED_CHECK_NAMES, *ARC_E2E_NAMES, *TERMINAL_CHECK_NAMES)


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Emit a flushed boundary before and after each potentially slow phase."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    elapsed = time.monotonic() - started
    print(
        f"[exp7574] phase={phase} event={event} elapsed_s={elapsed:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def sidecar_reference(path: Path, root: Path) -> Json:
    """Bind one current raw file without embedding its potentially large rows."""

    resolved = path.resolve()
    try:
        label = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        label = str(resolved)
    return {
        "path": label,
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
        "scope": "current_work",
    }


def _gate_row(
    check: str,
    upstream: str,
    path: Path,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
    *,
    op: str = "eq",
) -> Json:
    return {
        "check": check,
        "upstream": upstream,
        "path": str(path),
        "field": field,
        "op": op,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
    }


def check_arc_custody(root: Path, artifact_path: Path, expected_hash: str) -> Json:
    """Authenticate the exact Exp7562 producer bytes before ARC checks start."""

    absolute_root = root.resolve()
    path = (absolute_root / artifact_path).resolve()
    exists = path.is_file() and path.stat().st_size > 0
    checks = [
        _gate_row(
            "exp7562_artifact_exists",
            artifact_path.as_posix(),
            path,
            "path",
            "readable_file",
            "readable_file" if exists else "missing",
            exists,
        )
    ]
    value: Json = {}
    observed_hash: str | None = None
    if exists:
        observed_hash = sha256_file(path)
        checks.append(
            _gate_row(
                "exp7562_artifact_hash",
                artifact_path.as_posix(),
                path,
                "sha256",
                expected_hash,
                observed_hash,
                observed_hash == expected_hash,
            )
        )
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            value = dict(loaded) if isinstance(loaded, Mapping) else {}
        except (OSError, json.JSONDecodeError):
            value = {}
        expected_fields = (
            ("experiment_id", "exp7562-arc-plan-lineage"),
            ("plan_lineage_ready_score", 1),
            ("flagged_adversarial", False),
        )
        for field, expected in expected_fields:
            observed = value.get(field)
            checks.append(
                _gate_row(
                    f"exp7562_{field}",
                    artifact_path.as_posix(),
                    path,
                    field,
                    expected,
                    observed,
                    observed == expected,
                )
            )
    return {
        "passed": bool(checks and all(row["passed"] for row in checks)),
        "absolute_root": str(absolute_root),
        "artifact_path": str(path),
        "artifact_sha256": observed_hash,
        "experiment_id": value.get("experiment_id"),
        "checks": checks,
    }


REQUIRED_INPUTS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7561_v661_recalibration_prototype.py"),
    Path("python/carnot/experiment_7570_v661_arc_live_lineage.py"),
    Path("python/carnot/experiment_7562_v661_arc_plan_lineage.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    CL_SPEC_PATH,
    ARC_SPEC_PATH,
    EXP7561_PATH,
    EXP7570_PATH,
    EXP7562_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)


def collect_preconditions(root: Path) -> tuple[list[Json], Json]:
    """Resolve the root and verify all owned and external inputs before work."""

    absolute_root = root.resolve()
    checks: list[Json] = []
    checks.append(
        _gate_row(
            "absolute_repository_root",
            "--root",
            absolute_root,
            "root",
            str(REPO_ROOT),
            str(absolute_root),
            absolute_root == REPO_ROOT,
        )
    )
    for relative in REQUIRED_INPUTS:
        path = absolute_root / relative
        exists = path.is_file()
        checks.append(
            _gate_row(
                f"required_input:{relative.as_posix()}",
                relative.as_posix(),
                path,
                "path",
                "readable_file",
                "readable_file" if exists else "missing",
                exists,
            )
        )
    for relative, expected in (
        (EXP7561_PATH, EXP7561_SHA256),
        (EXP7570_PATH, EXP7570_SHA256),
    ):
        path = absolute_root / relative
        observed = sha256_file(path) if path.is_file() else None
        checks.append(
            _gate_row(
                f"historical_bytes_unchanged:{relative.name}",
                relative.as_posix(),
                path,
                "sha256",
                expected,
                observed,
                observed == expected,
            )
        )
    custody = check_arc_custody(absolute_root, EXP7562_PATH, EXP7562_SHA256)
    checks.extend(custody["checks"])
    spec_checks = (
        (CL_SPEC_PATH, "REQ-CL-7574"),
        (ARC_SPEC_PATH, "REQ-ARC-WMTE-7574"),
    )
    for relative, requirement in spec_checks:
        text = (absolute_root / relative).read_text(encoding="utf-8")
        checks.append(
            _gate_row(
                f"requirement_present:{requirement}",
                relative.as_posix(),
                absolute_root / relative,
                requirement,
                True,
                requirement in text,
                requirement in text,
            )
        )
    return checks, custody


def _squared_error(probability: float, label: int) -> float:
    if label not in (0, 1):
        raise ValueError("binary_label_required")
    value = float(probability)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("finite_probability_required")
    return (value - label) ** 2


def reduce_fixture_contrast(
    *,
    fixture: str,
    baseline_predictions: Sequence[float],
    candidate_predictions: Sequence[float],
    labels: Sequence[int],
    seed: int,
    provenance: str,
) -> Json:
    """Build two arm rows from exact predictions and labels.

    Positive signed improvement means the candidate has lower Brier loss. The
    raw numerator remains present so a reader need not trust either mean.
    """

    count = len(labels)
    if not count or len(baseline_predictions) != count or len(candidate_predictions) != count:
        raise ValueError("prediction_label_length_mismatch")
    prediction_rows: list[Json] = []
    baseline_numerator = 0.0
    candidate_numerator = 0.0
    for index, (baseline, candidate, label) in enumerate(
        zip(baseline_predictions, candidate_predictions, labels, strict=True)
    ):
        baseline_error = _squared_error(float(baseline), int(label))
        candidate_error = _squared_error(float(candidate), int(label))
        baseline_numerator += baseline_error
        candidate_numerator += candidate_error
        prediction_rows.append(
            {
                "fixture": fixture,
                "unit_index": index,
                "label": int(label),
                "baseline_prediction": float(baseline),
                "candidate_prediction": float(candidate),
                "baseline_squared_error": baseline_error,
                "candidate_squared_error": candidate_error,
                "seed": int(seed),
                "censored": False,
                "provenance": provenance,
            }
        )
    baseline_loss = baseline_numerator / count
    candidate_loss = candidate_numerator / count
    improvement = baseline_loss - candidate_loss
    common = {
        "fixture": fixture,
        "comparison_unit": fixture,
        "raw_denominator": count,
        "metric": "brier_loss",
        "metric_direction": "higher_signed_improvement_is_better",
        "seed": int(seed),
        "censored": False,
        "provenance": provenance,
        "comparator_arm": "raw_baseline",
    }
    rows = [
        {
            **common,
            "arm": "raw_baseline",
            "raw_squared_error_numerator": baseline_numerator,
            "mean_brier": baseline_loss,
        },
        {
            **common,
            "arm": "constrained_candidate",
            "raw_squared_error_numerator": candidate_numerator,
            "mean_brier": candidate_loss,
            "signed_improvement_delta": improvement,
        },
    ]
    return {
        "fixture": fixture,
        "baseline_loss": baseline_loss,
        "candidate_loss": candidate_loss,
        "baseline_numerator": baseline_numerator,
        "candidate_numerator": candidate_numerator,
        "denominator": count,
        "signed_improvement": improvement,
        "metric_direction": "higher_signed_improvement_is_better",
        "seed": int(seed),
        "censored": False,
        "provenance": provenance,
        "rows": rows,
        "prediction_label_rows": prediction_rows,
    }


def public_contrast(value: Mapping[str, Any]) -> Json:
    """Drop unit rows after their hash-bound sidecar has been written."""

    return {
        key: deepcopy(item)
        for key, item in value.items()
        if key not in {"rows", "prediction_label_rows"}
    }


def independent_reduce_comparison_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Json]:
    """Recompute both arm means and the signed contrast from raw numerators."""

    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        fixture = str(row.get("fixture"))
        arm = str(row.get("arm"))
        if arm in grouped.setdefault(fixture, {}):
            raise ValueError(f"duplicate_fixture_arm:{fixture}:{arm}")
        grouped[fixture][arm] = row
    reduced: dict[str, Json] = {}
    for fixture, arms in grouped.items():
        if set(arms) != {"raw_baseline", "constrained_candidate"}:
            raise ValueError(f"fixture_arm_set_invalid:{fixture}")
        baseline = arms["raw_baseline"]
        candidate = arms["constrained_candidate"]
        denominator = int(baseline.get("raw_denominator", 0))
        if denominator <= 0 or int(candidate.get("raw_denominator", 0)) != denominator:
            raise ValueError(f"fixture_denominator_invalid:{fixture}")
        baseline_loss = float(baseline["raw_squared_error_numerator"]) / denominator
        candidate_loss = float(candidate["raw_squared_error_numerator"]) / denominator
        reduced[fixture] = {
            "baseline_loss": baseline_loss,
            "candidate_loss": candidate_loss,
            "signed_improvement": baseline_loss - candidate_loss,
            "denominator": denominator,
        }
    return reduced


def run_fixture_requalification(raw_root: Path) -> Json:
    """Replay lifecycle checks and derive each Brier contrast from unit rows."""

    checkpoint_root = raw_root / "fixture_checkpoints"
    panel = recalibration.run_fixture_panel(checkpoint_root)
    contrasts: list[Json] = []
    prediction_rows: list[Json] = []
    fixture_rows = {str(row["fixture"]): row for row in panel["rows"]}
    for index, (fixture, events) in enumerate(recalibration.analytical_fixtures().items()):
        probabilities = [float(row["probability"]) for row in events]
        labels = [int(row["label"]) for row in events]
        theta = fixture_rows[fixture]["final_theta"]
        candidate = [recalibration.map_probability(value, theta) for value in probabilities]
        contrast = reduce_fixture_contrast(
            fixture=fixture,
            baseline_predictions=probabilities,
            candidate_predictions=candidate,
            labels=labels,
            seed=7_574_001 + index,
            provenance="exp7561_fixture_recomputed_from_prediction_label_rows",
        )
        contrasts.append(contrast)
        prediction_rows.extend(contrast["prediction_label_rows"])

    raw_path = raw_root / "fixture_prediction_label_rows.json"
    atomic_json(
        raw_path,
        {
            "schema": "carnot.exp7574.fixture_prediction_rows.v1",
            "prediction_label_rows": prediction_rows,
        },
    )
    rows = [deepcopy(row) for contrast in contrasts for row in contrast["rows"]]
    independently_reduced = independent_reduce_comparison_rows(rows)
    for contrast in contrasts:
        reduced = independently_reduced[contrast["fixture"]]
        if not math.isclose(
            float(reduced["signed_improvement"]),
            float(contrast["signed_improvement"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RuntimeError("fixture_independent_reduction_mismatch")
    lifecycle = {
        "passed": bool(panel["fixture_gates_passed"]),
        "parameter_change_count": panel["parameter_change_count"],
        "monotonicity_violation_count": panel["monotonicity_violation_count"],
        "movement_violation_count": panel["movement_violation_count"],
        "normalization_failure_count": panel["normalization_failure_count"],
        "duplicate_feedback_rejection_count": panel["duplicate_feedback_rejection_count"],
        "future_label_rejection_count": panel["future_label_rejection_count"],
        "restart_mismatch_count": panel["restart_mismatch_count"],
    }
    return {
        "rows": rows,
        "fixture_contrasts": [public_contrast(row) for row in contrasts],
        "raw_prediction_rows": sidecar_reference(raw_path, REPO_ROOT),
        "lifecycle_qualification": lifecycle,
    }


def reproduce_private_parent_failure(root: Path) -> Json:
    """Reproduce pytest's missing direct parent and then apply the repair."""

    target = root / "missing" / "pytest" / "focused"
    observed: str | None = None
    try:
        target.mkdir()
    except FileNotFoundError as error:
        observed = type(error).__name__
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    return {
        "failure_reproduced": observed == "FileNotFoundError",
        "observed_exception": observed,
        "missing_target": str(target),
        "repair": "create_direct_parent_before_subprocess",
        "repair_passed": target.is_dir(),
        "passed": observed == "FileNotFoundError" and target.is_dir(),
    }


def command_for_test_basetemp(target: Path) -> validation_scope.CommandSpec:
    """Return a small command used to prove private parent preparation."""

    return validation_scope.CommandSpec(
        "private_parent_probe",
        (str(REPO_ROOT / ".venv/bin/pytest"), f"--basetemp={target}"),
        "private_control",
        60.0,
    )


def prepare_command_parent(command: validation_scope.CommandSpec) -> Path | None:
    """Create every private output parent immediately before one child starts."""

    prepared: Path | None = None
    for argument in command.argv:
        if argument.startswith("--basetemp="):
            prepared = Path(argument.split("=", 1)[1]).parent
            prepared.mkdir(parents=True, exist_ok=True)
        elif argument.startswith("--data-file="):
            Path(argument.split("=", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
    environment = dict(getattr(command, "command_environment", ()))
    if coverage_file := environment.get("COVERAGE_FILE"):
        Path(coverage_file).parent.mkdir(parents=True, exist_ok=True)
    return prepared


def reproduce_exp7561_strict_failure(root: Path, private_root: Path) -> Json:
    """Run the unchanged strict reader on the exact bad-sign historical rows."""

    source = root / EXP7561_PATH
    value = json.loads(source.read_text(encoding="utf-8"))
    diagnostic = {
        "honest_verdict": "complete_circular_positive_fixture_qualification",
        "rows": deepcopy(value["rows"]),
    }
    path = private_root / "exp7561_positive_verdict_bad_sign_rows.json"
    atomic_json(path, diagnostic)
    command = (
        str(root / ".venv/bin/python"),
        "-u",
        str(root / "scripts/verdict_row_consistency_lint.py"),
        "--strict",
        str(path),
    )
    completed = subprocess.run(  # noqa: S603 - fixed repository command and private input.
        command,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60.0,
        check=False,
    )
    output = completed.stdout + completed.stderr
    return {
        "command": shlex.join(command),
        "exit_code": completed.returncode,
        "output": output,
        "failure_reproduced": completed.returncode == 1 and "WINS_NOT_EXCEEDING_LOSSES" in output,
        "source_path": EXP7561_PATH.as_posix(),
        "source_sha256": sha256_file(source),
        "diagnostic_sha256": sha256_file(path),
    }


def build_validation_commands(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Freeze serial tests, isolated coverage, lint, type, and spec checks."""

    commands = validation_scope.build_scoped_commands(
        root,
        AFFECTED_MANIFEST.test_paths,
        AFFECTED_MANIFEST.changed_modules,
        static_paths=AFFECTED_MANIFEST.static_paths,
        basetemp=private_root / "pytest",
        coverage_file=private_root / "coverage" / ".coverage.exp7574",
    )
    coverage_file = str(private_root / "coverage" / ".coverage.exp7574")
    transformed: list[validation_scope.CommandSpec] = []
    for command in commands:
        argv = [item for item in command.argv if not item.startswith("--data-file=")]
        if command.name in {"changed_module_coverage", "changed_module_coverage_report"}:
            argv = ["/usr/bin/env", f"COVERAGE_FILE={coverage_file}", *argv]
        transformed.append(
            validation_scope.CommandSpec(
                command.name,
                tuple(argv),
                command.scope,
                command.timeout_s,
            )
        )
    return transformed


def build_arc_e2e_commands(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Declare applicable CPU ARC checks and one alternate-directory E3 run."""

    pytest = str(root / ".venv/bin/pytest")
    python = str(root / ".venv/bin/python")
    common = ("-n", "0", "-o", "addopts=", "--no-cov")
    targets = {
        "e2e_009": ("tests/python/test_arc_induction_state_persistence.py",),
        "e2e_010": ("tests/python/test_arc_tool_grammar_transport.py",),
        "e2e_011": ("tests/python/test_arc_decision_telemetry.py",),
        "e2e_012": (
            "tests/python/test_experiment_7491_e6_timed_live_profile.py",
            "tests/python/test_experiment_7492_e6_timed_cost_profile.py",
            "tests/python/test_arc_decision_telemetry.py",
        ),
        "e2e_013": (
            "tests/python/test_arc_decision_telemetry.py",
            "tests/python/test_experiment_7491_e6_timed_live_profile.py",
            "tests/python/test_experiment_7531_b2_induction_gate_measurement.py",
            "tests/python/test_semif_arc_readout_eval.py",
        ),
    }
    commands = [
        validation_scope.CommandSpec(
            name,
            (
                pytest,
                *common,
                f"--basetemp={private_root / name}",
                *test_paths,
                "-q",
            ),
            f"{name.replace('_', '-').upper()} CPU contract",
            900.0,
        )
        for name, test_paths in targets.items()
    ]
    alternate = private_root / "alternate-cwd"
    alternate.mkdir(parents=True, exist_ok=True)
    output = private_root / "alternate-cwd-e3-smoke.json"
    commands.append(
        validation_scope.CommandSpec(
            "alternate_cwd_e3_smoke",
            (
                "/usr/bin/env",
                "-C",
                str(alternate),
                "CARNOT_ARC_DISABLE_INDUCTION=1",
                python,
                "-u",
                str(root / "scripts/arc_loop_solve.py"),
                "--mechanism",
                "e3",
                "--game",
                "r11l",
                "--max-actions",
                "12",
                "--output",
                str(output),
            ),
            "private LLM-off real E3AgentPolicy episode from alternate cwd",
            300.0,
        )
    )
    return commands


def terminal_commands(root: Path, candidate: Path) -> list[validation_scope.CommandSpec]:
    """Build fresh-process and exact-byte readers for one terminal candidate."""

    python = str(root / ".venv/bin/python")
    wrapper = str(root / WRAPPER_PATH)
    common = ("--root", str(root), "--date", RUN_DATE)
    return [
        validation_scope.CommandSpec(
            "declared_entrypoint",
            (python, "-u", wrapper, *common, "--validate", str(candidate)),
            "declared read-only entrypoint",
            300.0,
        ),
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (python, "-u", wrapper, *common, "--cold-replay", str(candidate)),
            "fresh-process cold replay",
            300.0,
        ),
        validation_scope.CommandSpec(
            "independent_reduction",
            (python, "-u", wrapper, *common, "--independent-reduce", str(candidate)),
            "independent raw-row reduction",
            300.0,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", str(root / "scripts/adversarial_verify.py"), str(candidate)),
            "exact terminal candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                str(root / "scripts/verdict_row_consistency_lint.py"),
                "--strict",
                str(candidate),
            ),
            "exact terminal candidate",
            300.0,
        ),
    ]


def run_prepared_commands(
    root: Path,
    commands: Sequence[validation_scope.CommandSpec],
    *,
    log_dir: Path,
) -> list[Json]:
    """Create one command's private parents immediately before that child runs."""

    receipts: list[Json] = []
    for command in commands:
        prepare_command_parent(command)
        receipts.extend(
            validation_scope.run_commands(
                root,
                [command],
                log_dir=log_dir / command.name,
                heartbeat_s=60.0,
            )
        )
    return receipts


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    counts = Counter(str(row.get("name")) for row in receipts)
    return all(
        counts[name] == 1
        and next(row for row in receipts if row.get("name") == name).get("passed") is True
        and next(row for row in receipts if row.get("name") == name).get("exit_code") == 0
        and next(row for row in receipts if row.get("name") == name).get("timed_out") is not True
        for name in names
    )


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    principles = {
        "honest_verdict": "Use a complete terminal prefix; completion does not establish benefit.",
        "verdict_class": "Use exactly one closed scientific disposition class.",
        "flagged_adversarial": "Persist the exact terminal verification outcome.",
        "gate_check_summary": "Name exact failed operands for every blocked verdict.",
        "acceptance_gate_results": "Separate validity, readiness, and benefit checks.",
        "rows": "Keep one arm row per unit with raw numerator and denominator.",
        "inference_substrate_class": "Record the actual substrate separately from the plan.",
        "MODEL_SPECS": "No current LLM task means an empty model list.",
        "invocation_counts": "Count current calls and tokens independently from history.",
        "duration_s": "Measure current monotonic work without inherited time or sleeps.",
        "random_seed": "Bind every fixture, order, and benchmark draw to frozen seeds.",
        "source_artifact_hashes": "Bind each conclusion to exact source bytes.",
        "validation_receipts": "Bind each check to command, worktree, exit, and log hash.",
        "field_principles": "Carry one-line interpretation rules with emitted fields.",
        "verifier_is_oracle": "Label-accessing controls cannot support an oracle-distinct claim.",
        "recalibration_ready_score": "Require numerical, lifecycle, and exact row readers.",
        "learning_compute_feasible_score": "Require all 5,000 replays within 2,400 seconds.",
        "arc_runner_ready_score": "Require custody, alternate CWD, and private directories.",
        "fixture_contrasts": "Keep baseline loss, candidate loss, and signed improvement.",
        "independent_solver_error": "Measure solver agreement without claiming efficacy.",
    }
    return {
        key: principles.get(key, f"Bind {key} so an independent reader can detect drift.")
        for key in keys
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    copied = deepcopy(dict(value))
    copied.pop("reproducibility_checksum", None)
    return canonical_hash(copied)


def _acceptance_gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    passed: bool,
) -> Json:
    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": "eq",
        "passed": bool(passed),
        "principle": "A failed gate cannot be hidden by another branch.",
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> Json:
    failures = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    return {
        "passed": not failures,
        "failed_count": len(failures),
        "failed_checks": [str(row["check"]) for row in failures],
        "first_failure": failures[0] if failures else None,
    }


def build_blocked_artifact(checks: Sequence[Mapping[str, Any]], *, duration_s: float) -> Json:
    """Publish a complete no-run record without substitute evidence."""

    failed = next((deepcopy(dict(row)) for row in checks if row.get("passed") is not True), None)
    if failed is None:
        raise ValueError("blocked_artifact_requires_failed_check")
    reason = "".join(character if character.isalnum() else "_" for character in failed["check"])
    artifact: Json = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": f"complete_blocked_{reason}",
        "honest_verdict": f"complete_blocked_{reason}",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "positive_claim": False,
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_model_identity": {},
        "planned_inference_substrate_class": "no_model_load",
        "inference_substrate_class": "blocked_no_run",
        "inference_substrate": "blocked_no_run",
        "execution_venue": "host",
        "duration_s": max(0.0, float(duration_s)),
        "phase_spans": [],
        "source_artifact_hashes": {},
        "validation_receipts": [],
        "rows": [],
        "fixture_contrasts": [],
        "acceptance_gate_results": [deepcopy(failed)],
        "gate_check_summary": {
            "passed": False,
            "failed_count": 1,
            "failed_checks": [str(failed["check"])],
            "first_failure": failed,
        },
        "verifier_is_oracle": True,
        "recalibration_ready_score": 0,
        "learning_compute_feasible_score": 0,
        "arc_runner_ready_score": 0,
        "empirical_benefit_score": 0,
        "independent_solver_error": None,
        "sample_size_budget": {
            "fixture_units": {"planned": 3, "attempted": 0, "completed": 0, "unstarted": 3},
            "benchmark_replays": {
                "planned": 5000,
                "attempted": 0,
                "completed": 0,
                "unstarted": 5000,
            },
        },
    }
    artifact["field_principles"] = _field_principles(
        (*artifact.keys(), "field_principles", "reproducibility_checksum")
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_artifact(
    *,
    evidence: Mapping[str, Any],
    preconditions_checked: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, str],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    terminal: bool,
) -> Json:
    """Classify validity, readiness, and fixture benefit independently."""

    rows = [deepcopy(dict(row)) for row in evidence.get("rows") or []]
    contrasts = [deepcopy(dict(row)) for row in evidence.get("fixture_contrasts") or []]
    row_reduction = independent_reduce_comparison_rows(rows)
    rows_valid = len(contrasts) > 0 and all(
        name in row_reduction
        and math.isclose(
            float(row_reduction[name]["signed_improvement"]),
            float(contrast["signed_improvement"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for contrast in contrasts
        for name in (str(contrast["fixture"]),)
    )
    numerical = evidence.get("numerical_qualification") or {}
    lifecycle = evidence.get("lifecycle_qualification") or {}
    benchmark = evidence.get("benchmark_receipt") or {}
    custody = evidence.get("arc_custody") or {}
    private_control = evidence.get("private_directory_control") or {}
    preconditions_passed = all(row.get("passed") is True for row in preconditions_checked)
    affected_passed = _receipts_pass(validation_receipts, AFFECTED_CHECK_NAMES)
    e2e_passed = _receipts_pass(validation_receipts, ARC_E2E_NAMES)
    terminal_passed = _receipts_pass(validation_receipts, TERMINAL_CHECK_NAMES)
    validation_passed = affected_passed and e2e_passed and (terminal_passed if terminal else True)
    numerical_ready = bool(
        numerical.get("passed") is True
        and lifecycle.get("passed") is True
        and rows_valid
        and evidence.get("strict_failure_control", {"failure_reproduced": True}).get(
            "failure_reproduced"
        )
        is True
    )
    compute_feasible = bool(
        benchmark.get("bootstrap_replicates_completed") == 1000
        and benchmark.get("order_replays_completed") == 5000
        and benchmark.get("events_completed") == 800000
        and benchmark.get("fits_2400_seconds_with_reserve") is True
    )
    arc_ready = bool(
        custody.get("passed") is True
        and private_control.get("passed") is True
        and affected_passed
        and e2e_passed
        and (terminal_passed if terminal else True)
    )
    recalibration_ready = bool(
        preconditions_passed
        and numerical_ready
        and affected_passed
        and (terminal_passed if terminal else True)
    )
    own_rows_positive = bool(contrasts) and all(
        float(row.get("signed_improvement", 0.0)) > 0.0 for row in contrasts
    )
    if not terminal:
        honest_verdict = "partial_terminal_readers_pending"
        verdict_class = "partial"
    elif not validation_passed or not preconditions_passed:
        honest_verdict = "complete_disqualified_required_validation"
        verdict_class = "disqualified"
    elif own_rows_positive:
        honest_verdict = "complete_circular_positive_fixture_rows_requalified"
        verdict_class = "circular_positive"
    else:
        honest_verdict = "complete_null_fixture_rows_do_not_support_positive"
        verdict_class = "null"
    adversarial = next(
        (row for row in validation_receipts if row.get("name") == "adversarial_verify"), None
    )
    flagged = bool(adversarial is not None and adversarial.get("passed") is not True)
    gates = [
        _acceptance_gate(
            "preconditions", "validity", True, preconditions_passed, preconditions_passed
        ),
        _acceptance_gate("affected_validation", "validity", True, affected_passed, affected_passed),
        _acceptance_gate("terminal_readers", "validity", True, terminal_passed, terminal_passed),
        _acceptance_gate(
            "numerical_and_lifecycle", "readiness", True, numerical_ready, numerical_ready
        ),
        _acceptance_gate(
            "registered_compute", "readiness", True, compute_feasible, compute_feasible
        ),
        _acceptance_gate("arc_runner", "readiness", True, arc_ready, arc_ready),
        _acceptance_gate(
            "empirical_predictive_benefit",
            "benefit",
            "measured_untouched_labels",
            "analytical_fixtures_only",
            False,
        ),
    ]
    artifact: Json = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": honest_verdict,
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": flagged,
        "positive_claim": False,
        "empirical_benefit_score": 0,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions_checked],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_model_identity": {
            "exp7561": "no_model_load",
            "exp7570": "historical_planned_Qwen3.8-27B-GGUF_but_no_current_calls",
            "counted_as_current": False,
        },
        "planned_inference_substrate_class": "no_model_load",
        "inference_substrate_class": "no_model_load",
        "inference_substrate": "independent_structural_control_row_and_solver_receipt_replay_no_llm",
        "execution_venue": "host",
        "random_seed": {
            "fixture_seeds": [7_574_001, 7_574_002, 7_574_003],
            "ordering_seeds": list(recalibration.ORDER_SEEDS),
            "benchmark_seed": recalibration.BOOTSTRAP_SEED,
        },
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "source_artifact_hashes": dict(source_artifact_hashes),
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "rows": rows,
        "fixture_contrasts": contrasts,
        "raw_prediction_rows": deepcopy(evidence.get("raw_prediction_rows")),
        "numerical_qualification": deepcopy(dict(numerical)),
        "lifecycle_qualification": deepcopy(dict(lifecycle)),
        "benchmark_receipt": deepcopy(dict(benchmark)),
        "arc_custody": deepcopy(dict(custody)),
        "private_directory_control": deepcopy(dict(private_control)),
        "strict_failure_control": deepcopy(dict(evidence.get("strict_failure_control") or {})),
        "alternate_cwd_e3_smoke": {
            "real_policy_path": "scripts/arc_loop_solve.py --mechanism e3",
            "induction_disabled": True,
            "model_calls": 0,
            "solve_credit": 0,
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "verifier_is_oracle": True,
        "recalibration_ready_score": int(recalibration_ready),
        "learning_compute_feasible_score": int(compute_feasible),
        "arc_runner_ready_score": int(arc_ready),
        "independent_solver_error": numerical.get("independent_solver_error"),
        "sample_size_budget": {
            "fixture_units": {
                "planned": 3,
                "attempted": len(contrasts),
                "completed": len(contrasts),
                "unstarted": max(0, 3 - len(contrasts)),
            },
            "benchmark_replays": {
                "planned": 5000,
                "attempted": int(benchmark.get("order_replays_completed", 0)),
                "completed": int(benchmark.get("order_replays_completed", 0)),
                "unstarted": max(0, 5000 - int(benchmark.get("order_replays_completed", 0))),
            },
        },
        "retired_literal_prior_verdict": {
            "literal": "complete_circular_positive_fixture_qualification",
            "scope": "Exp7561 published signed-row representation only",
            "scientific_hypothesis_retired": False,
            "replacement": honest_verdict,
        },
    }
    artifact["field_principles"] = _field_principles(
        (*artifact.keys(), "field_principles", "reproducibility_checksum")
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _read_sidecar(root: Path, receipt: Mapping[str, Any]) -> Json:
    path = Path(str(receipt.get("path") or ""))
    resolved = path if path.is_absolute() else root / path
    if not resolved.is_file():
        raise ValueError("raw_prediction_sidecar_missing")
    if sha256_file(resolved) != receipt.get("sha256"):
        raise ValueError("raw_prediction_sidecar_hash_mismatch")
    if int(receipt.get("bytes", -1)) != resolved.stat().st_size:
        raise ValueError("raw_prediction_sidecar_size_mismatch")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("raw_prediction_sidecar_not_object")
    return dict(value)


def _reduce_raw_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Json]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("fixture")), []).append(row)
    reduced: dict[str, Json] = {}
    for fixture, units in grouped.items():
        units = sorted(units, key=lambda row: int(row.get("unit_index", -1)))
        contrast = reduce_fixture_contrast(
            fixture=fixture,
            baseline_predictions=[float(row["baseline_prediction"]) for row in units],
            candidate_predictions=[float(row["candidate_prediction"]) for row in units],
            labels=[int(row["label"]) for row in units],
            seed=int(units[0]["seed"]),
            provenance=str(units[0]["provenance"]),
        )
        reduced[fixture] = public_contrast(contrast)
    return reduced


def independent_reduce(
    value: Mapping[str, Any], *, root: Path = REPO_ROOT, require_terminal: bool = True
) -> Json:
    """Recompute claims from raw rows, sidecar bytes, and command receipts."""

    if value.get("verdict_class") == "blocked":
        failure = (value.get("gate_check_summary") or {}).get("first_failure")
        return {
            "valid": bool(
                str(value.get("honest_verdict") or "").startswith("complete_blocked_")
                and isinstance(failure, Mapping)
                and set(("upstream", "path", "field", "op", "expected", "observed")) <= set(failure)
                and value.get("inference_substrate_class") == "blocked_no_run"
                and value.get("MODEL_SPECS") == []
                and not any((value.get("invocation_counts") or {}).values())
            ),
            "blocked": True,
            "fixture_claim_class": "blocked",
        }

    rows = value.get("rows")
    contrasts = value.get("fixture_contrasts")
    if not isinstance(rows, list) or not isinstance(contrasts, list):
        raise ValueError("comparison_rows_invalid")
    row_reduction = independent_reduce_comparison_rows(rows)
    contrast_by_fixture = {str(row.get("fixture")): row for row in contrasts}
    if set(row_reduction) != set(contrast_by_fixture):
        raise ValueError("row_reduction_mismatch")
    for fixture, reduced in row_reduction.items():
        expected = contrast_by_fixture[fixture]
        comparisons = (
            (reduced["baseline_loss"], expected.get("baseline_loss")),
            (reduced["candidate_loss"], expected.get("candidate_loss")),
            (reduced["signed_improvement"], expected.get("signed_improvement")),
            (reduced["denominator"], expected.get("denominator")),
        )
        if any(
            not math.isclose(float(observed), float(claimed), rel_tol=0.0, abs_tol=1e-12)
            for observed, claimed in comparisons
        ):
            raise ValueError("row_reduction_mismatch")
        arm_rows = [row for row in rows if row.get("fixture") == fixture]
        for row in arm_rows:
            expected_mean = float(row["raw_squared_error_numerator"]) / int(row["raw_denominator"])
            if not math.isclose(
                expected_mean, float(row["mean_brier"]), rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError("row_mean_mismatch")

    raw = _read_sidecar(root.resolve(), value.get("raw_prediction_rows") or {})
    prediction_rows = raw.get("prediction_label_rows")
    if not isinstance(prediction_rows, list):
        raise ValueError("raw_prediction_rows_invalid")
    raw_reduction = _reduce_raw_prediction_rows(prediction_rows)
    if set(raw_reduction) != set(contrast_by_fixture):
        raise ValueError("raw_fixture_set_mismatch")
    for fixture, reduced in raw_reduction.items():
        expected = contrast_by_fixture[fixture]
        for field in ("baseline_loss", "candidate_loss", "signed_improvement"):
            if not math.isclose(
                float(reduced[field]), float(expected[field]), rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError("raw_prediction_reduction_mismatch")

    receipts = value.get("validation_receipts") or []
    affected = _receipts_pass(receipts, AFFECTED_CHECK_NAMES)
    e2e = _receipts_pass(receipts, ARC_E2E_NAMES)
    terminal = _receipts_pass(receipts, TERMINAL_CHECK_NAMES)
    validation = affected and e2e and (terminal if require_terminal else True)
    numerical = value.get("numerical_qualification") or {}
    lifecycle = value.get("lifecycle_qualification") or {}
    benchmark = value.get("benchmark_receipt") or {}
    numerical_ready = numerical.get("passed") is True and lifecycle.get("passed") is True
    compute = bool(
        benchmark.get("bootstrap_replicates_completed") == 1000
        and benchmark.get("order_replays_completed") == 5000
        and benchmark.get("events_completed") == 800000
        and benchmark.get("fits_2400_seconds_with_reserve") is True
    )
    custody = value.get("arc_custody") or {}
    private_control = value.get("private_directory_control") or {}
    arc_ready = bool(
        custody.get("passed") is True
        and private_control.get("passed") is True
        and affected
        and e2e
        and (terminal if require_terminal else True)
    )
    preconditions = all(
        row.get("passed") is True for row in value.get("preconditions_checked") or []
    )
    own_rows_positive = bool(contrasts) and all(
        float(row.get("signed_improvement", 0.0)) > 0.0 for row in contrasts
    )
    return {
        "valid": bool(validation and preconditions),
        "blocked": False,
        "affected_validation_passed": affected,
        "arc_e2e_passed": e2e,
        "terminal_validation_passed": terminal,
        "numerical_ready": numerical_ready,
        "learning_compute_feasible": compute,
        "arc_runner_ready": arc_ready,
        "recalibration_ready": bool(validation and preconditions and numerical_ready),
        "fixture_claim_class": "circular_positive" if own_rows_positive else "null",
        "row_reduction": row_reduction,
    }


def validate_artifact(
    value: Mapping[str, Any], *, root: Path = REPO_ROOT, require_terminal: bool = True
) -> Json:
    """Reject identity, custody, raw-row, claim, score, or checksum drift."""

    if (
        value.get("schema") != SCHEMA
        or value.get("experiment_id") != EXPERIMENT_ID
        or value.get("milestone") != MILESTONE
        or value.get("run_date") != RUN_DATE
    ):
        raise ValueError("identity_mismatch")
    if value.get("MODEL_SPECS") != [] or value.get("model_specs") != []:
        raise ValueError("model_specs_not_empty")
    if value.get("model_invoked") is not False:
        raise ValueError("model_invoked_invalid")
    if value.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        raise ValueError("invocation_counts_invalid")
    expected_substrate = (
        "blocked_no_run" if value.get("verdict_class") == "blocked" else "no_model_load"
    )
    if value.get("inference_substrate_class") != expected_substrate:
        raise ValueError("substrate_invalid")
    for field in (
        "recalibration_ready_score",
        "learning_compute_feasible_score",
        "arc_runner_ready_score",
    ):
        if type(value.get(field)) is not int or value.get(field) not in (0, 1):
            raise ValueError(f"score_invalid:{field}")
    if value.get("positive_claim") is not False or value.get("empirical_benefit_score") != 0:
        raise ValueError("empirical_claim_invalid")
    if value.get("verifier_is_oracle") is not True:
        raise ValueError("oracle_declaration_missing")
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or not set(value).issubset(principles):
        raise ValueError("field_principles_incomplete")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        raise ValueError("checksum_mismatch")

    reduction = independent_reduce(value, root=root, require_terminal=require_terminal)
    if reduction["blocked"]:
        if any(
            value.get(field) != 0
            for field in (
                "recalibration_ready_score",
                "learning_compute_feasible_score",
                "arc_runner_ready_score",
            )
        ):
            raise ValueError("blocked_score_mismatch")
        if reduction["valid"] is not True:
            raise ValueError("blocked_schema_invalid")
        return reduction

    expected_scores = {
        "recalibration_ready_score": int(reduction["recalibration_ready"]),
        "learning_compute_feasible_score": int(reduction["learning_compute_feasible"]),
        "arc_runner_ready_score": int(reduction["arc_runner_ready"]),
    }
    for field, expected in expected_scores.items():
        if value.get(field) != expected:
            raise ValueError(f"score_mismatch:{field}")
    if not require_terminal:
        expected_verdict = ("partial_terminal_readers_pending", "partial")
    elif not reduction["valid"]:
        expected_verdict = ("complete_disqualified_required_validation", "disqualified")
    elif reduction["fixture_claim_class"] == "circular_positive":
        expected_verdict = (
            "complete_circular_positive_fixture_rows_requalified",
            "circular_positive",
        )
    else:
        expected_verdict = ("complete_null_fixture_rows_do_not_support_positive", "null")
    if (value.get("honest_verdict"), value.get("verdict_class")) != expected_verdict:
        raise ValueError("verdict_mismatch")
    if value.get("flagged_adversarial") is not False:
        raise ValueError("flagged_adversarial_mismatch")
    return reduction


def cold_replay(path: Path, *, root: Path = REPO_ROOT, require_terminal: bool = True) -> Json:
    """Reload serialized evidence through a fresh-reader boundary."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("artifact_unreadable") from error
    if not isinstance(value, Mapping):
        raise ValueError("artifact_not_object")
    return validate_artifact(value, root=root, require_terminal=require_terminal)


def write_affected_manifest(path: Path) -> Path:
    """Freeze every file that scoped validation may inspect."""

    atomic_json(
        path,
        {
            "experiment_id": AFFECTED_MANIFEST.experiment_id,
            "test_paths": list(AFFECTED_MANIFEST.test_paths),
            "changed_modules": list(AFFECTED_MANIFEST.changed_modules),
            "static_paths": list(AFFECTED_MANIFEST.static_paths),
        },
    )
    return path


def _source_hashes(root: Path, manifest: Path) -> dict[str, str]:
    paths = (*REQUIRED_INPUTS, manifest.relative_to(root))
    return {
        relative.as_posix(): sha256_file(root / relative)
        for relative in paths
        if (root / relative).is_file()
    }


def _span(phase: str, phase_started: float, run_started: float, units: int) -> Json:
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
    }


def run_experiment(
    root: Path, run_date: str, *, output_path: Path | None = None
) -> Json:  # pragma: no cover - exercised as the declared capability E2E.
    """Authenticate, requalify, validate, and atomically publish Exp7574."""

    if run_date != RUN_DATE:
        raise ValueError(f"run_date_must_equal_{RUN_DATE}")
    absolute_root = root.resolve()
    destination = output_path or absolute_root / RESULT_PATH
    raw_root = absolute_root / RAW_DIR
    raw_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    spans: list[Json] = []
    receipts: list[Json] = []

    progress(started, "preconditions", "before", root=absolute_root)
    phase_started = time.monotonic()
    checks, custody = collect_preconditions(absolute_root)
    spans.append(_span("preconditions", phase_started, started, len(checks)))
    failed = next((row for row in checks if row.get("passed") is not True), None)
    progress(started, "preconditions", "after", passed=failed is None, units=len(checks))
    if failed is not None:
        blocked = build_blocked_artifact(checks, duration_s=time.monotonic() - started)
        progress(started, "publish", "before_atomic_blocked", check=failed["check"])
        atomic_json(destination, blocked)
        progress(started, "publish", "after_atomic_blocked", path=destination)
        return blocked

    progress(started, "manifest", "before")
    phase_started = time.monotonic()
    manifest = write_affected_manifest(raw_root / "affected_validation_manifest.json")
    spans.append(_span("manifest", phase_started, started, 1))
    progress(started, "manifest", "after", path=manifest)

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7574-validation-", dir="/tmp"))
    progress(started, "private_parent_control", "before")
    phase_started = time.monotonic()
    private_control = reproduce_private_parent_failure(private_root / "control")
    spans.append(_span("private_parent_control", phase_started, started, 1))
    progress(started, "private_parent_control", "after", passed=private_control["passed"])

    progress(started, "strict_failure_control", "before_subprocess")
    phase_started = time.monotonic()
    strict_control = reproduce_exp7561_strict_failure(absolute_root, private_root / "strict")
    spans.append(_span("strict_failure_control", phase_started, started, 1))
    progress(
        started,
        "strict_failure_control",
        "after_subprocess",
        reproduced=strict_control["failure_reproduced"],
    )

    progress(started, "numerical_qualification", "before_solvers", cases=106)
    phase_started = time.monotonic()
    numerical = recalibration.run_numerical_qualification()
    numerical["independent_solver_error"] = numerical["maximum_objective_delta"]
    spans.append(_span("numerical_qualification", phase_started, started, 106))
    progress(started, "numerical_qualification", "after_solvers", passed=numerical["passed"])

    progress(started, "fixture_requalification", "before_replay", fixtures=3)
    phase_started = time.monotonic()
    evidence = run_fixture_requalification(raw_root)
    spans.append(_span("fixture_requalification", phase_started, started, 3))
    progress(
        started,
        "fixture_requalification",
        "after_replay",
        passed=evidence["lifecycle_qualification"]["passed"],
    )

    protocol = recalibration.freeze_learning_protocol()
    progress(started, "registered_benchmark", "before_benchmark", replays=5000, events=800000)
    phase_started = time.monotonic()
    benchmark = recalibration.run_replay_benchmark(
        protocol,
        progress_hook=lambda completed: progress(
            started,
            "registered_benchmark",
            "units_complete",
            completed_units=completed,
        ),
    )
    spans.append(
        _span(
            "registered_benchmark",
            phase_started,
            started,
            int(benchmark["order_replays_completed"]),
        )
    )
    progress(
        started,
        "registered_benchmark",
        "after_benchmark",
        completed=benchmark["order_replays_completed"],
        feasible=benchmark["fits_2400_seconds_with_reserve"],
    )
    evidence.update(
        numerical_qualification=numerical,
        benchmark_receipt=benchmark,
        arc_custody=custody,
        private_directory_control=private_control,
        strict_failure_control=strict_control,
    )

    progress(started, "affected_validation", "before_subprocesses", commands=8)
    phase_started = time.monotonic()
    affected_commands = build_validation_commands(absolute_root, private_root / "affected")
    affected = run_prepared_commands(
        absolute_root,
        affected_commands,
        log_dir=raw_root / "validation" / "affected",
    )
    receipts.extend(affected)
    spans.append(_span("affected_validation", phase_started, started, len(affected)))
    progress(
        started,
        "affected_validation",
        "after_subprocesses",
        passed=_receipts_pass(affected, AFFECTED_CHECK_NAMES),
    )

    progress(started, "arc_capability_e2e", "before_subprocesses", commands=6)
    phase_started = time.monotonic()
    arc_receipts = run_prepared_commands(
        absolute_root,
        build_arc_e2e_commands(absolute_root, private_root / "arc-e2e"),
        log_dir=raw_root / "validation" / "arc-e2e",
    )
    receipts.extend(arc_receipts)
    spans.append(_span("arc_capability_e2e", phase_started, started, len(arc_receipts)))
    progress(
        started,
        "arc_capability_e2e",
        "after_subprocesses",
        passed=_receipts_pass(arc_receipts, ARC_E2E_NAMES),
    )

    source_hashes = _source_hashes(absolute_root, manifest)
    candidate = build_artifact(
        evidence=evidence,
        preconditions_checked=checks,
        validation_receipts=receipts,
        source_artifact_hashes=source_hashes,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        terminal=False,
    )
    validate_artifact(candidate, root=absolute_root, require_terminal=False)
    candidate_path = raw_root / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    progress(started, "terminal_readers", "before_subprocesses", commands=5)
    phase_started = time.monotonic()
    terminal_receipts = run_prepared_commands(
        absolute_root,
        terminal_commands(absolute_root, candidate_path),
        log_dir=raw_root / "validation" / "terminal",
    )
    receipts.extend(terminal_receipts)
    spans.append(_span("terminal_readers", phase_started, started, len(terminal_receipts)))
    progress(
        started,
        "terminal_readers",
        "after_subprocesses",
        passed=_receipts_pass(terminal_receipts, TERMINAL_CHECK_NAMES),
    )

    final = build_artifact(
        evidence=evidence,
        preconditions_checked=checks,
        validation_receipts=receipts,
        source_artifact_hashes=source_hashes,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        terminal=True,
    )
    validate_artifact(final, root=absolute_root, require_terminal=True)
    exact_path = raw_root / "exact_terminal_candidate.json"
    atomic_json(exact_path, final)

    progress(started, "exact_terminal_readers", "before_subprocesses", commands=5)
    exact_receipts = run_prepared_commands(
        absolute_root,
        terminal_commands(absolute_root, exact_path),
        log_dir=raw_root / "validation" / "exact-terminal",
    )
    progress(
        started,
        "exact_terminal_readers",
        "after_subprocesses",
        passed=_receipts_pass(exact_receipts, TERMINAL_CHECK_NAMES),
    )
    if not _receipts_pass(exact_receipts, TERMINAL_CHECK_NAMES):
        raise RuntimeError("exact_terminal_candidate_validation_failed")
    atomic_json(raw_root / "exact_terminal_validation_receipts.json", {"rows": exact_receipts})
    progress(started, "publish", "before_atomic_terminal", path=destination)
    atomic_json(destination, final)
    progress(
        started,
        "publish",
        "after_atomic_terminal",
        verdict=final["honest_verdict"],
        recalibration_ready=final["recalibration_ready_score"],
        arc_runner_ready=final["arc_runner_ready_score"],
    )
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the producer and read-only fresh-process modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", choices=[RUN_DATE], required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def _load_artifact(path: Path) -> Json:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("artifact_not_object")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary.
    """Run measurement or one read-only exact-byte check."""

    args = parse_args(argv)
    root = args.root.resolve()
    read_path = args.validate or args.cold_replay or args.independent_reduce
    if read_path is not None:
        value = _load_artifact(read_path)
        require_terminal = value.get("verdict_class") != "partial"
        if args.cold_replay is not None:
            reduction = cold_replay(read_path, root=root, require_terminal=require_terminal)
            event = "cold_replay_passed"
        else:
            reduction = validate_artifact(value, root=root, require_terminal=require_terminal)
            event = (
                "independent_reduction_passed"
                if args.independent_reduce is not None
                else "declared_entrypoint_validation_passed"
            )
        print(json.dumps({"event": event, **reduction}, sort_keys=True), flush=True)
        return 0
    artifact = run_experiment(root, args.date, output_path=args.output)
    print(
        json.dumps(
            {
                "result": str(args.output or root / RESULT_PATH),
                "honest_verdict": artifact["honest_verdict"],
                "recalibration_ready_score": artifact["recalibration_ready_score"],
                "learning_compute_feasible_score": artifact["learning_compute_feasible_score"],
                "arc_runner_ready_score": artifact["arc_runner_ready_score"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return int(not str(artifact["honest_verdict"]).startswith("complete_"))


if __name__ == "__main__":  # pragma: no cover - wrapper is the public entrypoint.
    raise SystemExit(main())
