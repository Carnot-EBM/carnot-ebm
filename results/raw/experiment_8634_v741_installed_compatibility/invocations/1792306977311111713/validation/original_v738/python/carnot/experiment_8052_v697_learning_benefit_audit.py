"""REQ-REPORT-8052: publish an independent finite-trajectory learning audit."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

from carnot import experiment_8051_v697_feedback_constrained_learning as upstream
from carnot.experiment_artifacts import artifact_output_root
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting import learning_recovery_8052 as recovery
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify import learning_benefit_8052 as a

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_8052_v697_learning_benefit_audit"
TASK = "exp8052-learning-benefit-audit"
SCRIPT = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_learning_benefit_8052.py"
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/verify/learning_benefit_8052.py",
    SCRIPT,
    "python/carnot/reporting/learning_recovery_8052.py",
]
MODEL_SPECS: list[str] = []


def load_inputs(root: Path, raw: Path) -> tuple[Json, list[Json]]:
    """Authenticate upstream bindings and close the learner's retention boundary."""
    data, failures = upstream.load_inputs(root, raw)
    for operand in (
        "python/carnot/verify/learning_benefit_8039.py",
        "python/carnot/verify/learning_retention_audit_8026.py",
        "python/carnot/reporting/v696_capstone_reduction.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
        "scripts/check_spec_coverage.py",
    ):
        resource = ROOT / operand
        present = resource.is_file()
        gate = dict(
            upstream_id="local_resource",
            path=str(resource),
            sha256=sha256_file(resource) if present else None,
            check_name="resource_exists",
            artifact_field="resource_exists",
            expected=True,
            observed=present,
            passed=present,
        )
        data["gate_checks"].append(gate)
        if not present:
            failures.append(gate)
    values = {}
    for identity, name, ready in [
        (8051, upstream.NAME, "learning_trajectory_ready_score"),
        (8039, "experiment_8039_v696_learning_benefit_audit", "learning_audit_ready_score"),
    ]:
        path = root / "results" / (name + ".json")
        try:
            value = json.loads(path.read_text())
            values[identity] = value
            terminal = Path(value["terminal_validation_sidecar_path"])
            binding = json.loads(terminal.read_text())["publication"]
            report_path = Path(binding["sidecar_path"])
            report = json.loads(report_path.read_text())
            for field, expected, observed in [
                ("experiment_id", identity, value["experiment_id"]),
                (ready, 1, value[ready]),
                ("flagged_adversarial", False, value["flagged_adversarial"]),
                ("primary_sha256", sha256_file(path), binding["primary_sha256"]),
                ("sidecar_primary_sha256", sha256_file(path), report["primary_sha256"]),
                ("primary_path", str(path), binding["primary_path"]),
                ("report.passed", True, report["report"]["passed"]),
            ]:
                gate = dict(
                    upstream_id=value["task_id"],
                    path=str(path),
                    sha256=sha256_file(path),
                    check_name=field,
                    artifact_field=field,
                    expected=expected,
                    observed=observed,
                    passed=expected == observed,
                )
                data["gate_checks"].append(gate)
                if not gate["passed"]:
                    failures.append(gate)
            data["references"].extend(
                upstream.prior.upstream.copy_bound(reference(p), raw)
                for p in (path, terminal, report_path)
            )
        except (KeyError, ValueError, OSError) as error:
            failures.append(
                dict(
                    upstream_id=f"exp{identity}",
                    path=str(path),
                    sha256=sha256_file(path) if path.is_file() else None,
                    check_name="input_contract",
                    artifact_field="input_contract",
                    expected="present byte-bound primary and terminal sidecar",
                    observed=str(error),
                    passed=False,
                )
            )
    if not failures:
        try:
            produced = values[8051]
            a.equal("learner_retention_opened", False, produced["retained_labels_opened"])
            trajectory = Path(produced["trajectory_directory"])
            for ref in produced["raw_shard_hashes"]:
                checked(ref)
            copied = raw / "trajectory"
            shutil.copytree(trajectory, copied)
            inputs = json.loads((copied / "inputs.json").read_text())
            for field in ("head", "sources", "seeds"):
                a.equal("producer_input." + field, data[field], inputs[field])
            protocol = json.loads(
                (root / "results" / (upstream.m.protocol.NAME + ".json")).read_text()
            )
            retention_public = json.loads(
                checked(protocol["role_manifests"]["retention"]).read_text()
            )["rows"]
            sealed = json.loads(
                (root / "results" / (upstream.prior.upstream.NAME + ".json")).read_text()
            )
            retention_target = sealed["role_manifests"]["evaluator"]["retention"]
            checked(retention_target)
            stream_ids = {r["family_id"] for r in data["sources"]}
            a.equal(
                "retention_disjoint",
                True,
                not stream_ids & {r["family_id"] for r in retention_public},
            )
            data.update(
                trajectory=str(copied),
                retention_public=retention_public,
                retention_target=retention_target,
                historical_exposure=protocol["historical_exposure"],
                prior_failure=dict(
                    honest_verdict=values[8039]["honest_verdict"],
                    primary_hypothesis_results=values[8039]["primary_hypothesis_results"],
                ),
                learner_access_boundary=dict(
                    retained_labels_opened=False,
                    learner_input_keys=sorted(inputs),
                    retention_path_in_learner_inputs=False,
                    scope="Authenticated inputs and shipped learner target reader contain stream targets only. Recorded declaration and code boundary; no operating-system file-access trace was recorded upstream.",
                ),
            )
        except (KeyError, ValueError, OSError) as error:
            failures.append(
                dict(
                    upstream_id="exp8051",
                    path=str(root / "results" / (upstream.NAME + ".json")),
                    sha256=sha256_file(root / "results" / (upstream.NAME + ".json")),
                    check_name="primitive_contract",
                    artifact_field="primitive_contract",
                    expected="authenticated disjoint primitive inputs",
                    observed=str(error),
                    passed=False,
                )
                | getattr(error, "operand", {})
            )
    return data, failures


def validation_plan(scratch: Path) -> list[CommandSpec]:
    """Freeze scoped checks while keeping full-suite health outside acceptance."""
    scratch.mkdir(parents=True, exist_ok=True)
    commands = upstream.validation_plan(scratch)
    (scratch / "coverage.ini").write_text(
        "[run]\nparallel = True\ndata_file = "
        + str(scratch / ".coverage")
        + "\ninclude =\n    "
        + "\n    ".join(str(ROOT / p) for p in OWNED)
        + "\n"
    )
    substitutions = list(zip(upstream.OWNED, OWNED[:3], strict=True)) + [(upstream.TEST, TEST)]
    result = []
    for command in commands:
        argv = list(command.argv)
        for before, after in substitutions:
            argv = [x.replace(before, after) for x in argv]
        if command.name in ("ruff_check", "ruff_format", "strict_mypy"):
            argv.append(OWNED[3])
        result.append(
            replace(
                command,
                argv=tuple(argv),
                timeout_s=420 if command.name == "unit_consumers_e2e015_019" else command.timeout_s,
            )
        )
    return result


def terminal(path: Path) -> Json:
    """Run the real cold reducer and both established artifact readers."""
    py = str(ROOT / ".venv/bin/python")
    commands = [
        CommandSpec(
            "cold_reduction",
            (py, "-u", str(ROOT / SCRIPT), "--cold-replay", str(path)),
            "terminal",
            180,
        ),
        CommandSpec(
            "adversarial",
            (py, "scripts/adversarial_verify.py", str(path), "--json"),
            "terminal",
            120,
        ),
        CommandSpec(
            "strict_rows",
            (py, "scripts/verdict_row_consistency_lint.py", "--strict", str(path)),
            "terminal",
            120,
        ),
    ]
    raw = path.parent / "raw" / NAME if path.parent.name == "results" else path.parent
    receipts = run_commands(
        ROOT,
        commands,
        log_dir=raw / "terminal_logs" / sha256_file(path).split(":")[1],
        heartbeat_s=30,
    )
    for r in receipts:
        r.update(expected_exit_code=0, actual_exit_code=r["exit_code"])
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def replay(path: Path) -> Json:
    """Fresh primitive reduction rejects aggregate or readiness tampering."""
    value = json.loads(path.read_text())
    for ref in value["raw_shard_hashes"] + value["code_config_hashes"]:
        checked(ref)
    for ref in value["code_config_hashes"]:
        a.equal("active_code_hash", ref["sha256"], sha256_file(Path(ref["original_path"])))
    if value["acceptance_gate_results"]["validity"]:
        bundle = json.loads(checked(value["audit_bundle"]).read_text())
        labels = json.loads(checked(bundle["target_reference"]).read_text())["rows"]
        reduced = a.replay(
            Path(bundle["trajectory"]), {r["family_id"]: r["eligible_y"] for r in labels}
        )
        retained = a.retention(
            bundle, reduced, Path(value["retention_prediction_seal"]["path"]), cold=True
        )
        a.equal("auditor_retention_access", True, value["retained_labels_opened"])
        compared = a.compare(reduced["rows"], retained)
        for fields in (reduced, dict(retention_rows=retained), compared):
            for key, observed in fields.items():
                a.equal("reduction_drift:" + key, observed, value[key])
        a.equal(
            "recovery_rows",
            json.loads(checked(value["recovery_reference"]).read_text())["rows"],
            value["recovery_rows"],
        )
    a.equal("generalization_claim", 0, value["generalized_learning_benefit_score"])
    a.equal(
        "readiness",
        True,
        not value["learning_audit_ready_score"]
        or (
            value["verdict_class"] in ("null", "positive")
            and all(value["acceptance_gate_results"].values())
        ),
    )
    return dict(passed=True, sha256=sha256_file(path))


def base(failures: list[Json]) -> Json:
    """Keep absence explicit while retaining the standard no-model artifact fields."""
    value = upstream.prior.base(failures)
    value.update(
        experiment_id=8052,
        task_id=TASK,
        milestone="2026.10.697",
        schema="carnot.v697.learning_benefit_audit.v1",
        run_date="20261003",
        honest_verdict="complete_blocked_upstream_inputs"
        if failures
        else "complete_null_learning_benefit",
        learning_audit_ready_score=0,
        benefit_ready_score=0,
        claim_scope="This invocation independently audits one historically exposed source trajectory. Guard checks, future utility, retention and private recovery are separate. Finite conditional uncertainty gives no deployment or generalized learning claim.",
        config=dict(a.CONFIG, bootstrap_seed=6968039),
        random_seed=6968039,
        acceptance_gate_results=dict(validity=False, recovery=False, owned_checks=False),
        preconditions_checked=[],
        substrate_declaration=dict(
            custody="aggregation_from_upstream_artifacts",
            numerical="verifier_scoring",
            pretrained="no_model_load",
        ),
        methodology_note="Separate cubic, calibrated BCE and alpha equations; original256-slot paired blocks preserve all masks. Seed effects average within source slots. No pretrained model loads or generation. Failed science gates supply family p=1.",
        inference_substrate="verifier_scoring",
        inference_substrate_class="no_model_load",
    )
    if failures:
        value["honest_verdict"] = "complete_blocked_" + Path(
            failures[0].get("path", "upstream_inputs")
        ).stem.replace(".", "_")
    for key in (
        "independent_reduction_rows",
        "primary_hypothesis_results",
        "later_source_rows",
        "per_seed_false_accept_rows",
        "retention_rows",
        "recovery_rows",
        "guard_role_secondary_results",
    ):
        value[key] = []
    value["model_invocation_counts"] = dict(ZERO_INVOCATION_COUNTS)
    return value


def main(argv: list[str] | None = None) -> int:
    """Freeze evidence and bounded methods, qualify owned work, then publish once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261003"], default="20261003")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--fixture-bundle", type=Path)
    parser.add_argument("--validation-worker", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--recovery-worker", type=Path)
    parser.add_argument("--store-dir", type=Path)
    parser.add_argument("--boundary", default="none")
    args = parser.parse_args(argv)
    began = time.monotonic()
    a.progress("start_preconditions_no_pretrained_calls")
    try:
        if args.recovery_worker:
            if args.store_dir is None:
                raise ValueError("store_directory_required")
            recovery.worker(args.recovery_worker, args.store_dir, args.boundary)
            return 0
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        output = artifact_output_root(root=args.root) / (NAME + ".json")
        raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True)
        with tempfile.TemporaryDirectory(prefix="carnot-8052-") as temporary:
            scratch = Path(temporary)
            commands = validation_plan(scratch)
            bundle, failures = (
                (json.loads(args.fixture_bundle.read_text()), [])
                if args.fixture_bundle
                else load_inputs(args.root, raw)
            )
            value = base(failures)
            value.update(
                gate_check_summary=bundle.get("gate_checks", []) + failures,
                preconditions_checked=bundle.get("gate_checks", []) + failures,
                cited_upstream_artifacts=bundle.get("references", []),
                historical_exposure=bundle.get("historical_exposure", {}),
                prior_failure=bundle.get("prior_failure", {}),
                learner_access_boundary=bundle.get("learner_access_boundary", {}),
            )
            dependencies = [
                a.math.__file__,
                upstream.m.__file__,
                upstream.m.old.__file__,
                str(ROOT / "python/carnot/reporting/current_work_receipt.py"),
                str(ROOT / "python/carnot/reporting/primary_publication.py"),
            ]
            dependencies.append(str(ROOT / "python/carnot/verify/learning_benefit_8039.py"))
            value["code_config_hashes"] = [
                upstream.prior.upstream.copy_bound(reference(p), raw, "code")
                for p in [*(ROOT / p for p in OWNED + [TEST]), *(Path(p) for p in dependencies)]
            ]
            atomic_json(
                raw / "configuration.json",
                dict(
                    identity=8052,
                    task_id=TASK,
                    frozen=True,
                    code=value["code_config_hashes"],
                    config=value["config"],
                    commands=[
                        dict(name=c.name, argv=list(c.argv), timeout_s=c.timeout_s, scope=c.scope)
                        for c in commands
                    ],
                    input_references=bundle.get("references", []),
                    artifact_guard_enabled=True,
                    retention_target_identity=bundle.get("retention_target"),
                    frozen_at_ns=time.time_ns(),
                ),
            )
            frozen = time.monotonic()
            a.progress("methods_inputs_code_budgets_frozen")
            if not failures:
                labels = json.loads(checked(bundle["target_reference"]).read_text())["rows"]
                reduced = a.replay(
                    Path(bundle["trajectory"]), {r["family_id"]: r["eligible_y"] for r in labels}
                )
                value.update(reduced)
                value["checkpoint_references"] = [
                    reference(p)
                    for p in sorted((Path(bundle["trajectory"]) / "heads").glob("*.json"))
                ]
                seal = raw / "retention_predictions.json"
                value["retention_rows"] = a.retention(bundle, reduced, seal)
                value["retained_labels_opened"] = True
                value.update(a.compare(reduced["rows"], value["retention_rows"]))
                bundle["retention_target"] = upstream.prior.upstream.copy_bound(
                    bundle["retention_target"], raw, "evaluator"
                )
                atomic_json(raw / "bundle.json", bundle)
                value["audit_bundle"] = reference(raw / "bundle.json")
                value["retention_prediction_seal"] = reference(seal)
                a.progress("private_recovery_before")
                value["recovery_rows"] = recovery.recover(
                    reduced["initial_head"], scratch / "recovery", raw / "recovery", ROOT, SCRIPT
                )
                value["recovery_reference"] = reference(raw / "recovery/rows.json")
                a.progress("private_recovery_after", len(value["recovery_rows"]), 0)
                value["positive_control_results"] = a.controls()
                value["acceptance_gate_results"].update(
                    validity=True, recovery=all(r["passed"] for r in value["recovery_rows"])
                )
                value["genuine_headroom"] = dict(
                    measured=True,
                    beneficial_changed_groups=value["primary_hypothesis_results"][0][
                        "beneficial_changed_groups"
                    ],
                    scope="natural later update-role decisions; empirical guard is not future evidence",
                )
                value["trained_head_specs"] = [
                    dict(
                        arm=r["arm"],
                        seed=r["seed"],
                        parameters=110,
                        imported_updates=r["attempt_count"],
                        substrate="CPU small head",
                    )
                    for r in reduced["count_rows"]
                ]
                atomic_json(
                    raw / "independent_measurement.json",
                    {
                        k: value[k]
                        for k in (
                            "rows",
                            "independent_reduction_rows",
                            "primary_hypothesis_results",
                            "later_source_rows",
                            "retention_rows",
                            "recovery_rows",
                        )
                    },
                )
                for field, expected, observed, passed in [
                    (
                        "later_support",
                        dict(minimum=120, per_class=15),
                        value["later_support"],
                        value["support_passed"],
                    ),
                    (
                        "retention_support",
                        dict(minimum=48, per_class=8),
                        value["retention_support"],
                        value["retention_support_passed"],
                    ),
                ]:
                    gate = dict(
                        upstream_id="exp8051",
                        path=bundle["trajectory"],
                        sha256=canonical_hash(reduced["rows"]),
                        check_name=field,
                        artifact_field=field,
                        expected=expected,
                        observed=observed,
                        passed=passed,
                    )
                    value["gate_check_summary"].append(gate)
                    if not passed:
                        failures.append(gate)
                        value.update(
                            verdict_class="blocked", honest_verdict="complete_blocked_" + field
                        )
            measured = time.monotonic()
            if not args.validation_worker:
                a.progress("owned_validation_before")
                receipts = run_commands(
                    ROOT,
                    commands,
                    log_dir=raw / "validation_logs",
                    heartbeat_s=30,
                    extra_env=dict(
                        CARNOT_8052_COVERAGE_CONFIG=str(scratch / "coverage.ini"),
                        COVERAGE_FILE=str(scratch / ".coverage-health"),
                        PYTHONUNBUFFERED="1",
                        JAX_PLATFORMS="cpu",
                        OPENBLAS_NUM_THREADS="1",
                    ),
                )
                for r in receipts:
                    r.update(expected_exit_code=0, actual_exit_code=r["exit_code"])
                value["validation_receipts"] = [r for r in receipts if r["scope"] == "owned"]
                value["repository_health"] = [
                    r for r in receipts if r["scope"] == "repository_health"
                ]
                report = (
                    json.loads((scratch / "coverage.json").read_text())
                    if (scratch / "coverage.json").exists()
                    else dict(files={})
                )
                counts = {k: v["summary"] for k, v in report["files"].items()}
                value["coverage_statement_counts"] = counts
                passed = (
                    len(counts) == len(OWNED)
                    and all(r["missing_lines"] == 0 for r in counts.values())
                    and all(r["passed"] for r in value["validation_receipts"])
                )
                value["acceptance_gate_results"]["owned_checks"] = passed
                if not passed and not failures:
                    value.update(
                        verdict_class="disqualified",
                        honest_verdict="complete_disqualified_owned_validation",
                    )
                a.progress("owned_validation_after")
            if args.fixture_bundle:
                value.update(
                    verifier_is_oracle=True,
                    verdict_class="circular_positive",
                    honest_verdict="complete_circular_positive_learning_audit_fixture",
                    benefit_ready_score=0,
                )
            elif value["verdict_class"] == "null":
                value["learning_audit_ready_score"] = int(
                    all(value["acceptance_gate_results"].values())
                )
                if value["benefit_ready_score"] and value["learning_audit_ready_score"]:
                    value.update(
                        verdict_class="positive",
                        honest_verdict="complete_positive_local_learning_benefit",
                    )
            for key in (
                "intended",
                "eligible",
                "completed",
                "excluded",
                "failed",
                "censored",
                "independent",
            ):
                value[key + "_count"] = value["sample_size_budget"][key]
            value.update(
                duration_s=time.monotonic() - began,
                phase_spans=[
                    dict(phase="freeze", duration_s=frozen - began),
                    dict(phase="numerical_and_recovery", duration_s=measured - frozen),
                    dict(phase="validation", duration_s=time.monotonic() - measured),
                ],
                terminal_validation_sidecar_path=str(
                    output.parent / "raw" / NAME / "terminal_validation.json"
                ),
            )
            value["raw_shard_hashes"] = [
                reference(p) for p in sorted(raw.rglob("*")) if p.is_file()
            ]
            value["reproducibility_checksum"] = canonical_hash(
                dict(
                    code=value["code_config_hashes"],
                    raw=value["raw_shard_hashes"],
                    config=value["config"],
                )
            )
            value["field_principles"] = {
                k: "Record "
                + k.replace("_", " ")
                + " for this invocation. Bind counts to original source groups, references to exact bytes, and readiness to owned checks. Imported work gives no current model or generalization credit."
                for k in value
            }
            a.progress("publication_before")
            value["field_principles"].update(
                learning_audit_ready_score="Qualified primitive measurement and all owned checks establish audit readiness. A scientific win is not required.",
                benefit_ready_score="H3 must meet the margin, support, changed decisions, every-seed false-accept safety and both adaptive arms' retention gates.",
                generalized_learning_benefit_score="Fixed at zero: finite exposed development evidence and artificial controls cannot certify deployment benefit.",
                gate_check_summary="Every failed gate retains its source path, byte identity, named operand, expected value and observed value. Missing fields are contract failures.",
                retained_labels_opened="The auditor opens retention targets after terminal heads and label-free predictions are sealed. The learner's separate access declaration remains closed.",
                random_seed="This is the actual moving-block generator seed; imported learner seeds are algorithm repetitions within the same source timeline.",
                raw_shard_hashes="Exact durable primitive and validation bytes qualify this invocation. Each file remains below the shard size limit.",
                duration_s="Actual freeze, numerical, recovery and owned validation elapsed time. Terminal consumer timings are recorded separately in their receipts.",
            )
            publication = publish_primary(output, value, terminal)
            published = terminal(output)
            atomic_json(
                Path(value["terminal_validation_sidecar_path"]),
                dict(
                    publication=publication,
                    published=published,
                    reader=reader_receipt(
                        TASK,
                        output.parent,
                        field="learning_audit_ready_score",
                        expected=value["learning_audit_ready_score"],
                    ),
                ),
            )
            a.equal("published_validation", True, published["passed"])
            a.progress("complete", value["completed_count"], 0)
            return 0
    except (ValueError, OSError, KeyError, TimeoutError) as error:
        print(json.dumps(dict(passed=False, error=str(error))), flush=True)
        return 1
