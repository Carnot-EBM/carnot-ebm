"""Publish response calibration on unchanged cached judgments.

REQ-REPORT-7972. Private fitting and sealed scalar coefficients preserve label
roles. Readiness certifies the measurement; benefit is a separate science gate.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import UTC, datetime
import importlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot import experiment_7969_v691_qwen_calibration_capture as capture
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify import qwen_energy_calibration_7972 as c
from carnot.verify import response_role_targets_7968 as roles

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7972_v691_qwen_energy_calibration"
TASK = "exp7972-qwen-energy-calibration"
MODEL_SPECS: list[str] = []
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/verify/qwen_energy_calibration_7972.py",
    f"scripts/experiments/{NAME}.py",
]
TESTS = [
    "tests/python/test_qwen_energy_calibration_7972.py",
    f"tests/python/test_{NAME}.py",
    "tests/python/test_source_boundary_7852.py",
    "tests/python/test_experiment_7942_v689_sentence_labels.py",
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_current_work_receipt.py",
    "tests/python/test_experiment_7303_v642_validation_scope.py",
    "tests/python/test_response_role_targets_7968.py",
    "tests/python/test_qwen_response_risk_7958.py",
    "tests/python/test_qwen_calibration_capture_7969.py",
]
INCLUDE = ",".join(str(ROOT / p) for p in OWNED)
PINS = {
    7969: (
        "qwen_calibration_capture",
        "sha256:3f25b2e4b43d50536525e64ace321db55e9d6889aab97cc2581e4665014f2f51",
    ),
    7955: (
        "response_targets",
        "sha256:dbfac2d991450601add8506a00452c9b6d39a594ad91d3c291f742cfe378c62a",
    ),
}
reference, operand = capture.reference, capture.operand
checked_reference = capture.prior.targets.prior.checked_reference


def progress(phase: str, started: float) -> None:
    """Expose phase boundaries so real CPU work cannot appear stalled."""
    print(f"[exp7972] phase={phase} elapsed_s={time.monotonic() - started:.3f}", flush=True)


def authenticate(root: Path) -> tuple[list[Json], Json]:
    """Bind upstream primaries and exact public requests before reading labels."""
    _, upstream = capture.authenticate(root)
    checks = upstream["checks"]
    for eid, (suffix, pin) in PINS.items():
        path = root / f"results/experiment_{eid}_v{'691' if eid == 7969 else '690'}_{suffix}.json"
        checks.append(
            operand(f"exp{eid}", path, "sha256", pin, sha256_file(path) if path.is_file() else None)
        )
        value = json.loads(path.read_text()) if checks[-1]["passed"] else {}
        expected = dict(
            experiment_id=eid,
            task_id=f"exp{eid}-{suffix.replace('_', '-')}",
            run_date="20261001",
            verdict_class="null",
            flagged_adversarial=False,
            milestone="2026.10.691" if eid == 7969 else "2026.09.690",
        )
        expected["qwen_capture_ready_score" if eid == 7969 else "response_targets_ready_score"] = 1
        checks.extend(operand(f"exp{eid}", path, k, v, value.get(k)) for k, v in expected.items())
        upstream[f"exp{eid}"] = value
    plan: Json = dict(
        upstream={k: v for k, v in upstream.items() if k.startswith("exp")},
        checks=checks,
        protocol_match_rows=[],
        source_artifact_hashes=[],
    )
    if all(r["passed"] for r in checks):
        try:
            current, history = upstream["exp7969"], upstream["exp7958"]
            capture.replay(current)
            protocol = upstream["protocol"]
            checks.append(
                operand(
                    "exp7969",
                    root / "results/experiment_7969_v691_qwen_calibration_capture.json",
                    "protocol_fingerprint",
                    canonical_hash(protocol),
                    current["protocol_fingerprint"],
                )
            )
            checks.append(
                operand(
                    "exp7969",
                    root / "results/experiment_7969_v691_qwen_calibration_capture.json",
                    "response_target_definition",
                    history["response_target_definition"],
                    current["response_target_definition"],
                )
            )
            frozen = capture.capture.freeze(capture.load_public(current["public_role_manifests"]))
            manifest = json.loads(checked_reference(current["request_manifest"]).read_text())
            checks.append(
                operand(
                    "exp7969",
                    Path(current["request_manifest"]["path"]),
                    "request_rows",
                    canonical_hash(frozen),
                    canonical_hash(manifest["rows"]),
                )
            )
            for row, slot in zip(current["rows"], frozen, strict=True):
                if row["request"] != slot["request"] or row[
                    "parsed"
                ] != capture.prior.risk.transport.parse_response(
                    row["raw_response"], row["visible_ids"]
                ):
                    raise ValueError("request_parse_drift")
            import yaml

            retired = yaml.safe_load((root / "ops/exclusion_manifest.yaml").read_text())
            for eid in (7955, 7969):
                checks.append(
                    operand(
                        "exclusion_manifest",
                        root / "ops/exclusion_manifest.yaml",
                        f"exp{eid}_retired",
                        False,
                        any(
                            r.get("experiment_id") == eid
                            for key in ("retired", "retired_experiments")
                            for r in retired.get(key, [])
                        ),
                    )
                )
            plan["protocol_match_rows"] = checks[-5:]
            for key, value in plan["upstream"].items():
                eid = int(key[3:])
                path = next((root / "results").glob(f"experiment_{eid}_*.json"))
                plan["source_artifact_hashes"].append(reference(path))
                for item in value.get("raw_response_shards", []) + value.get(
                    "code_config_hashes", []
                ):
                    checked_reference(item)
                    plan["source_artifact_hashes"].append(item)
                for field in ("public_role_manifests", "evaluator_role_manifests"):
                    for item in value.get(field, {}).values():
                        checked_reference(item)
                        plan["source_artifact_hashes"].append(item)
            plan["protocol"] = protocol
        except (OSError, ValueError, KeyError, TypeError) as error:
            checks.append(
                operand(
                    "protocol", root / capture.HISTORY, "authenticated_requests", True, str(error)
                )
            )
    return [r for r in checks if not r["passed"]], plan


def scalar_rows(predictions: list[Json], labels: list[Json]) -> list[Json]:
    """IDs join custody only; the numeric fitting library sees only q as input."""
    indexed = capture.prior.risk.custody.index_unique(labels, "family_id")
    result = []
    for row in predictions:
        label = indexed[row["family_id"]]
        parsed = capture.prior.risk.transport.parse_response(
            row["raw_response"], row["visible_ids"]
        )
        if parsed != row["parsed"]:
            raise ValueError("parse_drift")
        result.append(
            dict(
                family_id=row["family_id"],
                source_cluster_id=label["source_cluster_id"],
                q=parsed["probability"],
                y=label["y"],
                status=row["status"],
            )
        )
    return result


def role_loader(plan: Json) -> Callable[[str, Json], list[Json]]:
    """Evaluator files are opened only for the explicitly authorized role."""

    def load(role: str, seals: Json) -> list[Json]:
        purpose = (
            "fitting"
            if role in ("fit", "tune")
            else ("threshold_design" if role == "policy_design" else "evaluation")
        )
        item = plan["upstream"]["exp7968"]["evaluator_role_manifests"][role]
        labels = roles.read_view(Path(item["path"]), item["sha256"], role, purpose, seals)["rows"]
        if role == "evaluation":
            historical = [
                r
                for r in plan["upstream"]["exp7955"]["response_union_rows"]
                if r["role"] == "evaluation"
            ]
            if labels != historical:
                raise ValueError("historical_label_drift")
            predictions = [
                r for r in plan["upstream"]["exp7958"]["rows"] if r["arm"] == "full_source"
            ]
        else:
            predictions = [r for r in plan["upstream"]["exp7969"]["rows"] if r["role"] == role]
        return scalar_rows(predictions, labels)

    return load


def base(failures: list[Json]) -> Json:
    """Declare unavailable measurements explicitly rather than synthesizing data."""
    value: Json = dict(
        schema="carnot.exp7972.qwen_energy_calibration.v1",
        experiment_id=7972,
        task_id=TASK,
        milestone="2026.10.691",
        run_date="20261001",
        execution_date="20261001",
        started_at=None,
        finished_at=None,
        duration_s=0.0,
        phase_spans=[],
        random_seed=69100,
        honest_verdict="complete_blocked_external_prerequisite",
        verdict_class="blocked",
        flagged_adversarial=False,
        gate_check_summary=failures,
        preconditions_checked=[],
        qwen_calibration_ready_score=0,
        qwen_calibration_benefit_score=0,
        MODEL_SPECS=[],
        model_specs=[],
        target_model=None,
        trained_head_specs=[],
        model_invocation_counts={k: 0 for k in capture.prior.base([])["model_invocation_counts"]},
        inference_substrate="blocked_no_run",
        inference_substrate_class="blocked_no_run",
        planned_inference_substrate_class="no_model_load",
        execution_venue="host",
        verifier_is_oracle=False,
        claim_scope="exposed_development",
        field_principles={},
        oracle_distinct_corrigendum=capture.prior.base([])["oracle_distinct_corrigendum"],
        production_defaults_changed=False,
        generator_weights_changed=False,
        source_artifact_hashes=[],
        cited_upstream_artifacts=[],
        resolved_imports={},
        code_config_hashes=[],
        reproducibility_checksum=None,
        validation_receipts=[],
        validation_command_manifest_path=None,
        observed_child_commands=[],
        coverage_statement_counts={},
        historical_required_failures=[],
        repository_health={},
        primary_resolution_receipt=None,
        terminal_validation_sidecar_path=None,
        scratch_root_receipt={},
        calibrator_checkpoints={},
        protocol_match_rows=[],
        role_label_access_events=[],
        parameter_counts={},
        coefficient_touch_counts={},
        comparisons={},
        evaluation_support={},
        rows=[],
        probability_rows=[],
        decision_rows=[],
        confidence_intervals={},
        raw_p_values={},
        adjusted_p_values={},
        false_accept_counts={},
        risk_coverage_rows=[],
        positive_control_rows=[],
        fit_support={},
        tune_support={},
        heads_seal=None,
        policies_seal=None,
        sample_size_budget=dict(
            unit="complete_response_slot",
            intended=64,
            eligible=0,
            started=0,
            completed=0,
            failed=0,
            censored=0,
            excluded=0,
            independent=0,
            independent_unit="original_source_cluster",
        ),
        acceptance_gate_results=dict(
            validity=False,
            readiness=False,
            calibration=False,
            decision_benefit=False,
            retention="not_measured",
            efficiency="descriptive_only",
        ),
        methodology="Five scalar-only arms; fit and tune labels precede sealed heads. "
        "Loss and thresholds are frozen before evaluator access. Small CPU heads only.",
        methodology_note="Readiness and synthetic checks do not establish human-label benefit. "
        "Unknown labels and invalid predictions escalate. Seed means are clustered before inference.",
        research_references=[
            dict(
                url="https://arxiv.org/abs/2602.02056",
                scope="spline locality only; no FPGA reproduction",
            ),
            dict(
                url="https://arxiv.org/abs/2606.16667",
                scope="deployed scoring pipeline motivation; no visual acquisition or conformal guarantee",
            ),
        ],
    )
    return value


def main(argv: list[str] | None = None) -> int:
    """Use one parameterized CPU CLI for production, private fixtures and replay."""
    started, started_at = time.monotonic(), datetime.now(UTC).isoformat()
    progress("start", started)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261001"], default="20261001")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--validation-worker",
        action="store_true",
        help="Run one private scenario without launching its parent validation plan.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    route = parser.add_mutually_exclusive_group()
    route.add_argument("--fixture-input", type=Path)
    route.add_argument("--cold-replay", type=Path)
    route.add_argument("--terminal-recheck", type=Path)
    args = parser.parse_args(argv)
    raw = args.output.absolute().parent / "raw" / args.output.stem
    try:
        if args.cold_replay or args.terminal_recheck:
            path = args.cold_replay or args.terminal_recheck
            replay(json.loads(path.read_text()))
            if args.terminal_recheck and not terminal_check(path)["passed"]:
                raise ValueError("terminal_recheck_failed")
            progress("replay_passed", started)
            return 0
        if args.output.name != NAME + ".json":
            raise ValueError("primary_name")
        if args.validation_worker and args.output.absolute().parent == ROOT / "results":
            raise ValueError("private_worker_output_required")
        with TemporaryDirectory(prefix="carnot-7972-") as workspace:
            scratch = Path(workspace)
            progress("authenticate", started)
            failures, plan = ([], {}) if args.fixture_input else authenticate(args.root)
            manifest = (
                freeze_commands(raw, scratch)
                if not args.fixture_input and not args.validation_worker
                else None
            )
            progress("fit_and_seal", started)
            data = json.loads(args.fixture_input.read_text()) if args.fixture_input else None
            value = (
                base(failures)
                if failures
                else measure(data if data is not None else role_loader(plan), raw)
            )
            if args.fixture_input:
                value["source_artifact_hashes"] = [reference(args.fixture_input)]
                value["honest_verdict"] = "complete_circular_positive_calibration_fixture"
                value["verdict_class"] = "circular_positive"
                value["qwen_calibration_benefit_score"] = 0
                value["acceptance_gate_results"]["decision_benefit"] = False
            else:
                value.update(
                    source_artifact_hashes=plan["source_artifact_hashes"],
                    preconditions_checked=plan["checks"],
                    protocol_match_rows=plan["protocol_match_rows"],
                )
                value["cited_upstream_artifacts"] = [
                    dict(
                        producer_id=int(k[3:]),
                        task_id=a["task_id"],
                        run_date=a["run_date"],
                        milestone=a["milestone"],
                        honest_verdict=a["honest_verdict"],
                        role="historical_input",
                        imported_fields=["rows", "protocol", "role_manifests"],
                    )
                    for k, a in plan["upstream"].items()
                    if a
                ]
                value["historical_required_failures"] = [
                    r
                    for a in plan["upstream"].values()
                    for r in a.get("historical_required_failures", [])
                ]
                value["historical_required_failures"] += [
                    dict(
                        scope="earlier_owned_code_revision", receipt_source=reference(p), failure=r
                    )
                    for p in (ROOT / "results/raw" / NAME).glob("*/receipts.json")
                    for r in json.loads(p.read_text())
                    if not r["passed"]
                ]
                value["prior_verdicts_unchanged"] = [
                    dict(task_id=a["task_id"], honest_verdict=a["honest_verdict"])
                    for a in plan["upstream"].values()
                    if a
                ]
            value.update(
                code_config_hashes=[reference(ROOT / p) for p in OWNED],
                config=c.CONFIG,
                started_at=started_at,
                finished_at=datetime.now(UTC).isoformat(),
                duration_s=time.monotonic() - started,
                scratch_root_receipt=dict(
                    path=str(scratch),
                    private=True,
                    outside_checkout=True,
                    pytest_basetemp=str(scratch / "pytest"),
                    retained=False,
                ),
            )
            value["reproducibility_checksum"] = canonical_hash(
                dict(
                    config=c.CONFIG,
                    code=value["code_config_hashes"],
                    inputs=value["source_artifact_hashes"],
                )
            )
            value["resolved_imports"] = {
                name: str(Path(importlib.import_module(name).__file__).resolve())
                for name in (
                    "carnot.verify.qwen_energy_calibration_7972",
                    "carnot.verify.qwen_response_risk_7958",
                    "carnot.verify.response_role_targets_7968",
                    "carnot.reporting.primary_publication",
                )
            }
            progress("measured_candidate", started)
            atomic_json(raw / "measured_candidate.json", value)
            if manifest:
                progress("before_validation_benchmark", started)
                receipts = execute_commands(manifest, raw, scratch)
                apply_validation(value, receipts)
                value["validation_command_manifest_path"] = str(raw / "validation_commands.json")
                coverage = scratch / "coverage.json"
                if coverage.is_file():
                    report = json.loads(coverage.read_text())
                    value["coverage_statement_counts"] = {
                        name: row["summary"] for name, row in report["files"].items()
                    }
                    atomic_json(raw / "coverage.json", report)
                    files = set(value["coverage_statement_counts"])
                    if files != set(OWNED) or any(
                        v["num_statements"] <= 0 or v["missing_lines"]
                        for v in value["coverage_statement_counts"].values()
                    ):
                        apply_validation(
                            value,
                            receipts
                            + [dict(name="nonempty_exact_coverage", passed=False, exit_code=1)],
                        )
                archive = raw / "coverage_data"
                archive.mkdir(exist_ok=True)
                for path in scratch.glob(".coverage*"):
                    shutil.copy2(path, archive / path.name)
                progress("after_validation_benchmark", started)
            for item in value["source_artifact_hashes"]:
                checked_reference(item)
            value["original_reply_hashes_unchanged"] = True
            value["finished_at"] = datetime.now(UTC).isoformat()
            value["duration_s"] = time.monotonic() - started
            value["phase_spans"] = [
                dict(phase="owned_cpu_run", start_s=0.0, end_s=value["duration_s"])
            ]
            progress("terminal_publication", started)
            publish(args.output.absolute(), value)
            progress("published", started)
            return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[exp7972] rejected={type(error).__name__}:{error}", flush=True)
        return 1


def replay(value: Json) -> Json:
    """Cold predictions use sealed parameters and reject primitive or claim drift."""
    for item in value.get("source_artifact_hashes", []) + value.get("code_config_hashes", []):
        checked_reference(item)
    checkpoints = value["calibrator_checkpoints"]
    if not checkpoints:
        if value["qwen_calibration_ready_score"]:
            raise ValueError("unsafe_readiness")
        return dict(qwen_calibration_ready_score=0)
    data = json.loads(checked_reference(checkpoints["primitives"]).read_text())
    if "heads" not in checkpoints:
        if c.support(data["fit"], 128, 16)["passed"] and c.support(data["tune"], 32, 4)["passed"]:
            raise ValueError("support_drift")
        return dict(qwen_calibration_ready_score=0)
    head_value = json.loads(checked_reference(checkpoints["heads"]).read_text())
    if head_value["config"] != c.CONFIG or any(
        head_value[k + "_hash"] != canonical_hash(data[k]) for k in ("fit", "tune")
    ):
        raise ValueError("checkpoint_identity_drift")
    heads = head_value["heads"]
    policies = json.loads(checked_reference(checkpoints["policies"]).read_text())
    if policies != c.design(heads, data["policy_design"]):
        raise ValueError("policy_drift")
    c.validate_data(data)
    reduced = c.evaluate(heads, policies, data["evaluation"])
    for key, observed in reduced.items():
        if key in {
            "honest_verdict",
            "verdict_class",
            "qwen_calibration_ready_score",
            "qwen_calibration_benefit_score",
        } and value["verdict_class"] in {"disqualified", "circular_positive"}:
            continue
        if value[key] != observed:
            raise ValueError("reduction_drift:" + key)
    for arm in ("platt", "gibbs", "spline"):
        if any(h["parameters"] == h["initial_parameters"] for h in heads[arm]):
            raise ValueError("no_parameter_changes")
    for control in value["positive_control_rows"]:
        synthetic = c.evaluate(control["heads"], control["policies"], control["evaluation_rows"])
        if control["comparisons"] != synthetic["comparisons"] or control["detected"] != (
            synthetic["comparisons"]["raw_qwen_brier"]["gain"] > 0.1
        ):
            raise ValueError("positive_control_drift")
    return reduced


def terminal_check(candidate: Path) -> Json:
    """Both validators inspect the same bytes that cold reduction just checked."""
    replay(json.loads(candidate.read_text()))
    commands = [
        CommandSpec(
            name,
            (str(ROOT / ".venv/bin/python"), "-u", script, flag, str(candidate)),
            "terminal_candidate",
            60,
        )
        for name, script, flag in [
            ("adversarial", "scripts/adversarial_verify.py", "--json"),
            ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
        ]
    ]
    receipts = run_commands(
        ROOT, commands, log_dir=candidate.parent / "terminal_logs", heartbeat_s=10
    )
    return dict(
        passed=all(r["passed"] for r in receipts),
        receipts=receipts,
        flagged_adversarial=not receipts[0]["passed"],
    )


def publish(output: Path, value: Json) -> None:
    """Publish checked bytes once and bind both live consumers to their hash."""
    raw = output.parent / "raw" / output.stem
    value["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
    value["primary_resolution_receipt"] = dict(path=str(raw / "primary_resolution.json"))
    value["field_principles"] = {
        k: "Bind producer identity, measured work, primitive custody or independent validity; completion is not improvement."
        for k in value
    }
    value["field_principles"].update(
        qwen_calibration_ready_score="Support floors, sealed same-information protocol and passing owned checks; no positive benefit threshold.",
        qwen_calibration_benefit_score="Gibbs must beat all three primary scalar controls on cost and Brier with joint Holm correction and safe automation.",
        coefficient_touch_counts="Actual optimized coefficients times steps; spline locality does not remove full-batch or regularization work.",
        risk_coverage_rows="Thresholds use policy_design only at target .50; observed evaluation automation can differ; no conformal guarantee.",
        positive_control_rows="Synthetic distortion is circular_positive and cannot support human-label science.",
        model_invocation_counts="Current pretrained calls are zero; historical Qwen generated fixed input judgments.",
    )
    atomic_json(raw / "terminal_candidate.json", value)
    report = terminal_check(raw / "terminal_candidate.json")
    if not report["passed"]:
        if value["inference_substrate_class"] == "blocked_no_run":
            value.update(
                inference_substrate="aggregation_from_upstream_artifacts",
                inference_substrate_class="no_model_load",
            )
        value.update(
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_terminal_validation",
            flagged_adversarial=report.get("flagged_adversarial", False),
            qwen_calibration_ready_score=0,
            qwen_calibration_benefit_score=0,
        )
        value["acceptance_gate_results"].update(
            validity=False, readiness=False, decision_benefit=False
        )
        value["historical_required_failures"].append(
            dict(scope="owned_terminal_before_verdict_change", report=report)
        )
    published = publish_primary(output, value, terminal_check)
    atomic_json(
        raw / "terminal_validation.json",
        dict(
            primary_path=str(output),
            primary_sha256=published["primary_sha256"],
            validator=published["sidecar_path"],
        ),
    )
    atomic_json(
        raw / "newer_nested_sidecar.json", dict(task_id=TASK, qwen_calibration_ready_score=99)
    )
    receipt = reader_receipt(
        TASK,
        output.parent,
        field="qwen_calibration_ready_score",
        expected=value["qwen_calibration_ready_score"],
    )
    if not receipt["passed"]:
        raise ValueError("primary_resolution")
    atomic_json(raw / "primary_resolution.json", receipt)


def fixture_data() -> Json:
    """Separate synthetic sources exercise the same private fitting CLI."""
    return {
        role: [
            dict(
                family_id=f"{role}-{i}",
                source_cluster_id=f"{role}-{i}",
                q=0.1 if i % 2 == 0 else 0.9,
                y=i % 2,
                status="completed",
            )
            for i in range(n)
        ]
        for role, n in dict(fit=128, tune=32, policy_design=32, evaluation=64).items()
    }


def freeze_commands(raw: Path, scratch: Path) -> Json:
    """Freeze private scenario outputs, expected reasons and coverage before science."""
    cov = str(ROOT / ".venv/bin/coverage")
    py = str(ROOT / ".venv/bin/python")
    cli = str(ROOT / OWNED[2])
    covered = [
        cov,
        "run",
        "--parallel-mode",
        "--data-file=" + str(scratch / ".coverage"),
        "--include=" + INCLUDE,
    ]
    fixture = scratch / "input.json"
    bad = scratch / "bad.json"
    atomic_json(fixture, fixture_data())
    atomic_json(bad, dict(evaluation=[]))
    success = scratch / "success" / (NAME + ".json")
    commands: list[Json] = []

    def add(
        name: str,
        argv: list[str],
        expected: int = 0,
        reason: str | None = None,
        required: bool = True,
        deadline: int = 300,
    ) -> None:
        commands.append(
            dict(
                name=name,
                argv=argv,
                expected_exit=expected,
                reason=reason,
                required=required,
                deadline_s=deadline,
            )
        )

    add(
        "unit_consumers_e2e015_019",
        [
            *covered,
            "-m",
            "pytest",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            "--basetemp=" + str(scratch / "pytest"),
            *TESTS,
            "-q",
        ],
    )
    for name, args, exit_code, reason in [
        ("fitting_cli", ["--fixture-input", str(fixture), "--output", str(success)], 0, None),
        ("cold_fixture_cli", ["--cold-replay", str(success)], 0, None),
        (
            "live_private_cpu_cli",
            ["--root", str(ROOT), "--output", str(scratch / "live-private" / (NAME + ".json"))],
            0,
            None,
        ),
        (
            "blocked_cli",
            [
                "--root",
                str(scratch / "absent"),
                "--output",
                str(scratch / "blocked" / (NAME + ".json")),
            ],
            0,
            None,
        ),
        (
            "invalid_input_cli",
            ["--fixture-input", str(bad), "--output", str(scratch / "invalid" / (NAME + ".json"))],
            1,
            "role_roster",
        ),
        ("invalid_date_cli", ["--date", "20260930"], 2, "invalid choice"),
        ("cold_parameters_cli", ["--cold-replay", str(raw / "measured_candidate.json")], 0, None),
        (
            "terminal_recheck_cli",
            ["--terminal-recheck", str(raw / "measured_candidate.json")],
            0,
            None,
        ),
    ]:
        add(name, [*covered, cli, "--validation-worker", *args], exit_code, reason)
    add(
        "coverage_combine",
        [cov, "combine", "--keep", "--data-file=" + str(scratch / ".coverage"), str(scratch)],
    )
    add(
        "coverage_json",
        [
            cov,
            "json",
            "--data-file=" + str(scratch / ".coverage"),
            "--include=" + INCLUDE,
            "-o",
            str(scratch / "coverage.json"),
        ],
    )
    add(
        "coverage_report",
        [
            cov,
            "report",
            "--data-file=" + str(scratch / ".coverage"),
            "--include=" + INCLUDE,
            "--show-missing",
            "--fail-under=100",
        ],
    )
    for name, args in [("ruff_check", ["check"]), ("ruff_format", ["format", "--check"])]:
        add(name, [str(ROOT / ".venv/bin/ruff"), *args, *OWNED, *TESTS[:2]])
    add("strict_mypy", [str(ROOT / ".venv/bin/mypy"), "--strict", *OWNED[:2]])
    add("spec_coverage", [py, "scripts/check_spec_coverage.py", *TESTS])
    add(
        "repository_health_full_suite",
        [
            str(ROOT / ".venv/bin/pytest"),
            "tests/python",
            "-q",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            "--basetemp=" + str(scratch / "full-suite"),
        ],
        required=False,
        deadline=300,
    )
    health = raw / "repository_health_receipt.json"
    if health.is_file():
        commands[-1]["reuse_receipt"] = reference(health)
    value = dict(
        commands=commands,
        coverage_includes=INCLUDE,
        affected_files=OWNED,
        test_paths=TESTS,
        test_input_hashes=[reference(ROOT / path) for path in TESTS],
        config=c.CONFIG,
        code_config_hashes=[reference(ROOT / p) for p in OWNED],
        scratch_root=str(scratch),
        pytest_basetemp=str(scratch / "pytest"),
        full_suite_policy="One real run; repository health stays separate from owned required checks.",
    )
    atomic_json(raw / "validation_commands.json", value)
    return value


def execute_commands(manifest: Json, raw: Path, scratch: Path) -> list[Json]:
    """Run each frozen child with heartbeats and preserve every observed exit."""
    receipts = []
    for index, spec in enumerate(manifest["commands"]):
        if spec.get("reuse_receipt"):
            cached = json.loads(checked_reference(spec["reuse_receipt"]).read_text())
            if cached["name"] != "repository_health_full_suite" or cached["required"] is not False:
                raise ValueError("repository_health_receipt_identity")
            checked_reference(dict(path=cached["log_path"], sha256=cached["log_sha256"]))
            receipts.append(
                dict(
                    cached,
                    reused=True,
                    receipt_scope="prior_owned_repository_health",
                    reuse_receipt=spec["reuse_receipt"],
                )
            )
            continue
        observed = run_commands(
            ROOT,
            [
                CommandSpec(
                    spec["name"], tuple(spec["argv"]), "frozen_owned_scope", spec["deadline_s"]
                )
            ],
            log_dir=raw / "validation_logs" / str(index),
            heartbeat_s=10,
            extra_env={"COVERAGE_FILE": str(scratch / ".coverage")},
        )[0]
        observed.update(
            required=spec["required"],
            expected_exit=spec["expected_exit"],
            expected_reason=spec["reason"],
            deadline_s=spec["deadline_s"],
        )
        text = Path(observed["log_path"]).read_text()
        observed["passed"] = (
            observed["exit_code"] == spec["expected_exit"]
            and not observed["timed_out"]
            and (not spec["reason"] or spec["reason"] in text)
        )
        receipts.append(observed)
    return receipts


def apply_validation(value: Json, receipts: list[Json]) -> None:
    """Keep failed historical health separate and never conceal owned failures."""
    value["validation_receipts"] = receipts
    value["observed_child_commands"] = [
        r.get("command_argv", []) for r in receipts if not r.get("reused")
    ]
    health = [r for r in receipts if not r.get("required", True)]
    value["repository_health"] = dict(
        current=health,
        status="degraded_open" if any(not r["passed"] for r in health) else "healthy",
        historical_failures=value["historical_required_failures"],
    )
    if any(not r["passed"] for r in receipts if r.get("required", True)):
        if value["inference_substrate_class"] == "blocked_no_run":
            value.update(
                inference_substrate="aggregation_from_upstream_artifacts",
                inference_substrate_class="no_model_load",
            )
        value.update(
            honest_verdict="complete_disqualified_required_validation",
            verdict_class="disqualified",
            qwen_calibration_ready_score=0,
            qwen_calibration_benefit_score=0,
        )
        value["acceptance_gate_results"].update(
            validity=False, readiness=False, decision_benefit=False
        )


def measure(data: Json | Callable[[str, Json], list[Json]], raw: Path) -> Json:
    """Seal fitted parameters and policy thresholds before evaluation label access."""
    raw.mkdir(parents=True, exist_ok=True)
    value, collected, events = base([]), {}, []
    seals: Json = {}

    def load(role: str) -> list[Json]:
        rows = data(role, seals) if callable(data) else data[role]
        collected[role] = rows
        events.append(
            dict(
                role=role,
                purpose="fitting" if role in ("fit", "tune") else role,
                sequence=len(events),
                seals=dict(seals),
                rows=len(rows),
            )
        )
        return rows

    if not callable(data):
        c.validate_data(data)
    fitting, tuning = load("fit"), load("tune")
    value.update(
        fit_support=c.support(fitting, 128, 16),
        tune_support=c.support(tuning, 32, 4),
        role_label_access_events=events,
        inference_substrate_class="no_model_load",
        inference_substrate="verifier_ensemble_against_cached_candidates",
    )
    if not value["fit_support"]["passed"] or not value["tune_support"]["passed"]:
        value.update(
            honest_verdict="complete_null_insufficient_calibration_support", verdict_class="null"
        )
        atomic_json(raw / "primitives.json", collected)
        value["calibrator_checkpoints"] = dict(primitives=reference(raw / "primitives.json"))
        return value
    heads = c.fit(fitting, tuning)
    atomic_json(
        raw / "heads.json",
        dict(
            config=c.CONFIG,
            heads=heads,
            fit_hash=canonical_hash(fitting),
            tune_hash=canonical_hash(tuning),
        ),
    )
    seals["heads"] = reference(raw / "heads.json")
    policies = c.design(heads, load("policy_design"))
    atomic_json(raw / "policies.json", policies)
    seals["policies"] = reference(raw / "policies.json")
    evaluation = load("evaluation")
    c.validate_data(collected)
    atomic_json(raw / "primitives.json", collected)
    value.update(c.evaluate(heads, policies, evaluation))
    control = c.positive_control()
    value.update(
        calibrator_checkpoints=dict(seals, primitives=reference(raw / "primitives.json")),
        heads_seal=seals["heads"],
        policies_seal=seals["policies"],
        positive_control_rows=[control],
        parameter_counts={a: [h["parameter_count"] for h in heads[a]] for a in c.ARMS},
        coefficient_touch_counts={
            a: [h.get("coefficient_touches", 0) for h in heads[a]] for a in c.ARMS
        },
        trained_head_specs=[
            dict(
                arm=a,
                seeds=[h.get("seed") for h in heads[a]],
                parameter_counts=[h["parameter_count"] for h in heads[a]],
            )
            for a in c.ARMS[1:]
        ],
        calibrator_timing_rows=[
            dict(
                arm=a,
                seed=h.get("seed"),
                fit_tune_duration_s=h.get("duration_s", 0.0),
                optimizer_steps=h.get("optimizer_steps", 0),
                coefficient_touches=h.get("coefficient_touches", 0),
                fit_duration_s=h.get("fit_duration_s", h.get("duration_s", 0.0)),
                tune_duration_s=h.get("tune_duration_s", 0.0),
            )
            for a in c.ARMS
            for h in heads[a]
        ],
        optimizer_work=dict(
            fit_pairs=len(c.valid(fitting)),
            learned_heads=9,
            full_batch_steps=1800,
            synthetic_control_full_batch_steps=1800,
            total_current_fitting_steps=3600,
            coefficient_touches=sum(
                h["coefficient_touches"] for a in ("platt", "gibbs", "spline") for h in heads[a]
            ),
        ),
    )
    changed = all(
        h["changed_coefficients"] > 0 for a in ("platt", "gibbs", "spline") for h in heads[a]
    )
    if not control["detected"] or not changed:
        value.update(
            honest_verdict="complete_disqualified_positive_control",
            verdict_class="disqualified",
            qwen_calibration_ready_score=0,
            qwen_calibration_benefit_score=0,
        )
    ready = bool(value["qwen_calibration_ready_score"])
    value["acceptance_gate_results"].update(
        validity=ready,
        readiness=ready,
        calibration=changed,
        decision_benefit=bool(value["qwen_calibration_benefit_score"]),
    )
    return value
