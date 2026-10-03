"""REQ-REPORT-7999: independent benefit, retention and recovery audit of cached replay."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import checked, operand, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify import learning_causal_audit_7999 as m

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7999_v693_learning_causal_audit"
TASK = "exp7999-learning-causal-audit"
MODEL_SPECS: list[str] = []
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/verify/learning_causal_audit_7999.py",
    "python/carnot/reporting/learning_audit_validation_7999.py",
    f"scripts/experiments/{NAME}.py",
]
TESTS = ["tests/python/test_learning_causal_audit_7999.py", f"tests/python/test_{NAME}.py"]
INPUTS = {
    7994: (
        "experiment_7994_v693_development_cohort",
        "cohort_ready_score",
        "sha256:22ffc346ca244d8f2623ec52e56a814207cbeec3d95627346cbd058edd586a2f",
    ),
    7995: (
        "experiment_7995_v693_qwen_development_capture",
        "retention_capture_ready_score",
        "sha256:df6eb8b559181455348b1d806f23c36d13c7f49202c2d49b2233aa80edebe02d",
    ),
    7998: (
        "experiment_7998_v693_selective_feedback_learning",
        "learning_measurement_ready_score",
        "sha256:3c4edde6a059e945e22d02334a8b6fb8d5983562376c181ae0a76926ab964ba6",
    ),
}


def progress(phase: str) -> None:
    """Expose each boundary without adding time to the experiment."""
    print(f"[exp7999] phase={phase}", flush=True)


def authenticate(root: Path) -> tuple[list[Json], Json]:
    """Check exact producer bytes and explicit readiness fields before reduction."""
    checks, refs, producers = [], [], {}
    for eid, (name, ready, pin) in INPUTS.items():
        path = root / "results" / (name + ".json")
        checks.append(
            operand(eid, path, "sha256", pin, sha256_file(path) if path.is_file() else None)
        )
        if not checks[-1]["passed"]:
            continue
        value = json.loads(path.read_text())
        fields = dict(
            experiment_id=eid,
            execution_date="20261001",
            milestone="2026.10.693",
            flagged_adversarial=False,
            **{ready: 1},
        )
        if eid == 7998:
            fields["retention_labels_opened"] = False
        for field, expected in fields.items():
            if field not in value:
                raise ValueError("upstream_contract:" + field)
            checks.append(operand(eid, path, field, expected, value[field]))
        producers[eid] = value
        refs.append(
            dict(
                reference(path),
                producer_id=eid,
                producer_invocation_date=value["execution_date"],
                imported_fields=list(fields)
                + {
                    7994: ["public_role_manifests", "evaluator_role_manifests"],
                    7995: ["rows"],
                    7998: ["checkpoints", "algorithm_seeds", "config"],
                }[eid],
            )
        )
    if len(producers) == len(INPUTS):
        assets = [(7998, r) for r in producers[7998]["raw_shard_hashes"]]
        assets += [
            (7994, producers[7994][key][role])
            for key in ("public_role_manifests", "evaluator_role_manifests")
            for role in ("stream", "retention")
        ]
        for eid, ref in assets:
            path = Path(ref["path"])
            checks.append(
                operand(
                    eid,
                    path,
                    "sha256",
                    ref["sha256"],
                    sha256_file(path) if path.is_file() else None,
                )
            )
    return [r for r in checks if not r["passed"]], dict(
        checks=checks, refs=refs, upstream=producers
    )


def load_bundle(plan: Json) -> Json:
    """Read replay primitives and public retention inputs, keeping retention targets sealed."""
    cohort, capture, replay = (plan["upstream"][eid] for eid in (7994, 7995, 7998))
    checkpoints = replay["checkpoints"]
    head = json.loads(checked(checkpoints["frozen_head"]).read_text())
    sources = json.loads(checked(checkpoints["public_stream"]).read_text())["rows"]
    acquisition = json.loads(checked(checkpoints["acquisition"]).read_text())["rows"]
    trajectories = {
        f"{a}-{s}": json.loads(checked(checkpoints[f"{a}-{s}"]).read_text())
        for s in replay["algorithm_seeds"]
        for a in m.ARMS
    }
    stream_targets = {
        r["family_id"]: r["y"]
        for r in json.loads(checked(cohort["evaluator_role_manifests"]["stream"]).read_text())[
            "rows"
        ]
    }
    public = json.loads(checked(cohort["public_role_manifests"]["retention"]).read_text())
    features = {r["family_id"]: r for r in public["features"]}
    captured = {r["family_id"]: r for r in capture["rows"] if r["role"] == "retention"}
    retention = []
    for r in public["request_rows"]:
        f, c = features[r["family_id"]], captured[r["family_id"]]
        retention.append(
            dict(
                family_id=r["family_id"],
                source_cluster_id=f["source_normalized_hash"],
                features=f["values"],
                q=c["parsed"]["probability"] if c.get("parsed") else None,
                status="completed"
                if c["status"] == "generated" and c["parsed"]["completed"]
                else c["status"],
            )
        )
    return dict(
        head=head,
        sources=sources,
        acquisition=acquisition,
        trajectories=trajectories,
        seeds=replay["algorithm_seeds"],
        stream_targets=stream_targets,
        retention_public=retention,
        retention_target_ref=cohort["evaluator_role_manifests"]["retention"],
    )


def base(failures: list[Json]) -> Json:
    """Give blocked and completed results the same explicit terminal contract."""
    return dict(
        experiment_id=7999,
        task_id=TASK,
        milestone="2026.10.693",
        execution_date="20261001",
        run_date="20261001",
        honest_verdict="complete_blocked_learning_causal_audit"
        if failures
        else "complete_null_learning_causal_audit",
        verdict_class="blocked" if failures else "null",
        gate_check_summary=failures,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        trained_head_specs=[],
        duration_s=0.0,
        phase_spans=[],
        random_seed=69399,
        reproducibility_checksum=None,
        cited_upstream_artifacts=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
        rows=[],
        sample_size_budget=dict(
            intended=256,
            eligible=0,
            started=0,
            completed=0,
            excluded=256,
            failed=0,
            censored=0,
            independent=0,
        ),
        verifier_is_oracle=False,
        claim_scope="One adaptive development replay with fallible human source-support annotations; public pretraining and wider historical exposure are unknown. No deployment generalization claim.",
        acceptance_gate_results={},
        positive_control_results={},
        preconditions_checked=[],
        validation_command_manifest_path=None,
        validation_receipts=[],
        coverage_statement_counts={},
        flagged_adversarial=False,
        terminal_validation_sidecar_path=None,
        learning_audit_ready_score=0,
        finite_replay_benefit_score=0,
        generalized_learning_benefit_score=0,
        independent_reduction_rows=[],
        paired_comparisons={},
        retention_rows=[],
        crash_recovery_rows=[],
        support_by_class={},
        effective_blocks=0,
        dependence_limits="Finite development replay cannot establish deployment generalization.",
        headroom_diagnostics={},
        retention_labels_opened=False,
        checkpoints={},
        label_access_events=[],
        config=m.CONFIG,
        repository_health=[],
        historical_required_failures=[],
    )


def recover(bundle: Json, raw: Path) -> list[Json]:
    """Kill real private workers on both sides of the atomic commit for every primary arm."""
    rows = []
    seed = bundle["seeds"][0]
    for arm in ("targeted_ipw", *m.CONTROLS):
        for where in ("before", "after"):
            directory = raw / "recovery" / f"{arm}-{seed}-{where}"
            argv = [
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / OWNED[-1]),
                "--recovery-input",
                str(raw / "bundle.json"),
                "--recovery-arm",
                arm,
                "--recovery-seed",
                str(seed),
                "--recovery-directory",
                str(directory),
            ]
            if os.environ.get("CARNOT_AUDIT_COVERAGE_FILE"):
                argv[:2] = [
                    str(ROOT / ".venv/bin/coverage"),
                    "run",
                    "--parallel-mode",
                    "--data-file=" + os.environ["CARNOT_AUDIT_COVERAGE_FILE"],
                    "--include=" + ",".join(str(ROOT / p) for p in OWNED),
                ]
            commands = [
                CommandSpec(
                    "crash_" + where, tuple(argv + ["--crash-phase", where]), "recovery", 60
                ),
                CommandSpec("resume", tuple(argv), "recovery", 60),
            ]
            receipts = run_commands(ROOT, commands, log_dir=directory / "logs", heartbeat_s=15)
            restored = json.loads((directory / "result.json").read_text())
            saved = bundle["trajectories"][f"{arm}-{seed}"]["trajectory"]
            same = (
                restored["final_state"] == saved["final_state"]
                and restored["issued_predictions"] == saved["issued_predictions"]
            )
            passed = same and receipts[0]["exit_code"] == 86 and receipts[1]["passed"]
            rows.append(
                dict(
                    arm=arm,
                    seed=seed,
                    boundary_slot=128,
                    crash_position=where,
                    weights_rng_pending_seen_and_later_predictions_match=same,
                    passed=passed,
                    numerator=int(passed),
                    denominator=1,
                    eligibility=True,
                    failure_status=not passed,
                    censor_status=False,
                    receipts=receipts,
                    restored_result=reference(directory / "result.json"),
                )
            )
    return rows


def measure(bundle: Json, raw: Path, fixture: bool = False) -> Json:
    """Freeze final states and retention predictions before the reserved label read."""
    value, rows, heads = base([]), [], {}
    sources = bundle["sources"]
    retention_public = bundle["retention_public"]
    if {r["source_cluster_id"] for r in sources} & {
        r["source_cluster_id"] for r in retention_public
    }:
        raise ValueError("role_overlap")
    atomic_json(raw / "bundle.json", bundle)
    value["checkpoints"]["bundle"] = reference(raw / "bundle.json")
    progress("independent_reduction_benchmark_begin")
    for seed in bundle["seeds"]:
        for arm in m.ARMS:
            key = f"{arm}-{seed}"
            schedule = [r for r in bundle["acquisition"] if r["arm"] == arm and r["seed"] == seed]
            got = m.reduce(
                bundle["head"],
                sources,
                schedule,
                bundle["trajectories"][key],
                bundle["stream_targets"],
            )
            rows.extend(got["rows"])
            atomic_json(raw / "gradients" / (key + ".json"), dict(rows=got["gradient_rows"]))
            heads[key] = got["final_state"]["head"]
            value["independent_reduction_rows"].append(
                dict(
                    arm=arm,
                    seed=seed,
                    updates=len(got["update_rows"]),
                    prediction_changed=got["prediction_changed"],
                    final_state_checksum=canonical_hash(got["final_state"]),
                    gradients=reference(raw / "gradients" / (key + ".json")),
                    numerator=len(got["rows"]),
                    denominator=len(sources),
                    eligibility=True,
                    failure_status=False,
                    censor_status=False,
                )
            )
            progress("reduction_checkpoint_" + key)
    atomic_json(raw / "final_heads.json", dict(heads=heads))
    value["checkpoints"]["final_heads"] = reference(raw / "final_heads.json")
    progress("final_states_frozen_retention_predictions_begin")
    matrix, offsets = m.geometry(bundle["head"], retention_public)
    predictions = []
    for key, head in dict(heads, initial=bundle["head"]).items():
        arm, seed = key.rsplit("-", 1) if key != "initial" else ("initial", "17")
        for i, r in enumerate(retention_public):
            usable = r["status"] == "completed" and r["q"] is not None and r["features"] is not None
            predictions.append(
                dict(
                    family_id=r["family_id"],
                    source_cluster_id=r["source_cluster_id"],
                    arm=arm,
                    seed=int(seed),
                    slot=i,
                    probability=m.probability(head, matrix[i], float(offsets[i]))
                    if usable
                    else None,
                )
            )
    atomic_json(raw / "retention_predictions.json", dict(rows=predictions))
    value["checkpoints"]["retention_predictions"] = reference(raw / "retention_predictions.json")
    progress("retention_predictions_sealed_target_open")
    targets = (
        bundle["retention_targets"]
        if fixture
        else {
            r["family_id"]: r["y"]
            for r in json.loads(checked(bundle["retention_target_ref"]).read_text())["rows"]
        }
    )
    atomic_json(raw / "retention_targets.json", targets)
    value["checkpoints"]["retention_targets"] = reference(raw / "retention_targets.json")
    value["label_access_events"] = [
        dict(
            role="retention",
            purpose="sealed_evaluation_only",
            preceded_by=value["checkpoints"]["retention_predictions"],
            final_heads=value["checkpoints"]["final_heads"],
        )
    ]
    summary, retention = m.compare(rows), m.retention(predictions, targets)
    value.update({k: v for k, v in summary.items() if k != "benefit"})
    value.update(
        rows=rows,
        retention_rows=retention["rows"],
        retention_results=retention,
        positive_control_results=m.controls(),
        verifier_is_oracle=fixture,
        retention_labels_opened=True,
        inference_substrate="verifier_ensemble_against_cached_candidates",
    )
    progress("reduction_benchmark_end_recovery_begin")
    value["crash_recovery_rows"] = recover(bundle, raw)
    value["recovery_scope"] = (
        "Every primary arm at registered acquisition seed 101; all twenty natural schedules independently reconstructed from saved state checksums."
    )
    mutations = []
    for name in ("future_label", "missing_update", "propensity", "group"):
        import copy

        saved = copy.deepcopy(bundle["trajectories"][f"targeted_ipw-{bundle['seeds'][0]}"])
        schedule = copy.deepcopy(
            [
                r
                for r in bundle["acquisition"]
                if r["arm"] == "targeted_ipw" and r["seed"] == bundle["seeds"][0]
            ]
        )
        if name == "future_label":
            saved["trajectory"]["reveal_rows"][0]["origin_slot"] += 1
        elif name == "missing_update":
            saved["trajectory"]["update_rows"].pop()
        elif name == "propensity":
            schedule[0]["pi"] = 0.9
        else:
            saved["trajectory"]["issued_predictions"][0]["source_cluster_id"] = "changed"
        try:
            m.reduce(bundle["head"], sources, schedule, saved, bundle["stream_targets"])
            detected = False
        except ValueError:
            detected = True
        mutations.append(
            dict(
                mutation=name,
                passed=detected,
                numerator=int(detected),
                denominator=1,
                eligibility=True,
                failure_status=not detected,
                censor_status=False,
            )
        )
    value["mutation_rows"] = mutations
    support = m.support([r for r in rows if r["arm"] == "targeted_ipw"], 0, 0)
    value["sample_size_budget"] = dict(
        intended=len(sources),
        eligible=support["independent"],
        started=sum(r["status"] != "excluded" for r in sources),
        completed=support["independent"],
        excluded=sum(r["status"] == "excluded" for r in sources),
        failed=sum(r["status"] == "failed" for r in sources),
        censored=sum(r["status"] == "censored" for r in sources),
        independent=support["independent"],
        seeds_are_independent=False,
    )
    value["trained_head_specs"] = [
        dict(
            arm=a,
            pretrained=False,
            parameter_count=109,
            optimizer="independent_reconstruction_of_sparse_gradient_descent",
            historical_initial_producer=7996,
            fitted_current_steps=sum(
                r["updates"] for r in value["independent_reduction_rows"] if r["arm"] == a
            ),
        )
        for a in m.ARMS
    ]
    valid = value["positive_control_results"]["passed"] and all(
        r["passed"] for r in value["crash_recovery_rows"] + mutations
    )
    durable = any(
        r["prediction_changed"] and r["updates"] > 0
        for r in value["independent_reduction_rows"]
        if r["arm"] == "targeted_ipw"
    )
    value["acceptance_gate_results"] = dict(
        validity=valid,
        durable_future_prediction_change=durable,
        benefit=summary["benefit"] and retention["passed"] and durable and valid and not fixture,
        retention=retention["passed"],
        readiness=False,
    )
    if not valid:
        value.update(
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_learning_causal_audit",
        )
    return value


def replay(value: Json) -> Json:
    """Recompute science from byte-bound primitive shards instead of trusting aggregates."""
    if value["generalized_learning_benefit_score"] or (
        value["verdict_class"] in ("blocked", "disqualified")
        and value["learning_audit_ready_score"]
    ):
        raise ValueError("unsafe_readiness")
    if not value["checkpoints"]:
        return dict(passed=True, blocked=True)
    for ref in value["raw_shard_hashes"] + value["code_config_hashes"]:
        checked(ref)
    bundle = json.loads(checked(value["checkpoints"]["bundle"]).read_text())
    rows, heads = [], {}
    for seed in bundle["seeds"]:
        for arm in m.ARMS:
            key = f"{arm}-{seed}"
            schedule = [r for r in bundle["acquisition"] if r["arm"] == arm and r["seed"] == seed]
            got = m.reduce(
                bundle["head"],
                bundle["sources"],
                schedule,
                bundle["trajectories"][key],
                bundle["stream_targets"],
            )
            rows.extend(got["rows"])
            heads[key] = got["final_state"]["head"]
    saved_heads = json.loads(checked(value["checkpoints"]["final_heads"]).read_text())["heads"]
    predictions = json.loads(checked(value["checkpoints"]["retention_predictions"]).read_text())[
        "rows"
    ]
    targets = json.loads(checked(value["checkpoints"]["retention_targets"]).read_text())
    matrix, offsets = m.geometry(bundle["head"], bundle["retention_public"])
    for r in predictions:
        head = bundle["head"] if r["arm"] == "initial" else heads[f"{r['arm']}-{r['seed']}"]
        public = bundle["retention_public"][r["slot"]]
        p = (
            m.probability(head, matrix[r["slot"]], float(offsets[r["slot"]]))
            if public["status"] == "completed"
            and public["q"] is not None
            and public["features"] is not None
            else None
        )
        if (
            r["family_id"] != public["family_id"]
            or r["source_cluster_id"] != public["source_cluster_id"]
            or p != r["probability"]
        ):
            raise ValueError("retention_prediction_drift")
    summary = m.compare(rows)
    if (
        rows != value["rows"]
        or heads != saved_heads
        or any(value[k] != summary[k] for k in summary if k != "benefit")
        or m.retention(predictions, targets) != value["retention_results"]
    ):
        raise ValueError("reduction_drift")
    for row in value["independent_reduction_rows"]:
        checked(row["gradients"])
    for row in value["crash_recovery_rows"]:
        checked(row["restored_result"])
    return dict(passed=True, rows=len(rows))


def apply_validation(value: Json, receipts: list[Json], counts: Json) -> None:
    """Owned failures disqualify; a complete scientific null can still be ready."""
    coverage_ok = set(counts) == set(OWNED) and all(
        c["num_statements"] > 0 and c["missing_lines"] == 0 for c in counts.values()
    )
    value.update(
        validation_receipts=receipts,
        coverage_statement_counts=counts,
        repository_health=[r for r in receipts if not r["required"]],
    )
    owned_ok = (
        bool(receipts) and coverage_ok and all(r["passed"] for r in receipts if r["required"])
    )
    if not owned_ok or value["acceptance_gate_results"].get("validity") is False:
        value.update(
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_learning_causal_audit",
            learning_audit_ready_score=0,
            finite_replay_benefit_score=0,
        )
    elif value["verdict_class"] != "blocked" and not value["verifier_is_oracle"]:
        benefit = value["acceptance_gate_results"].get("benefit", False)
        value.update(
            learning_audit_ready_score=1,
            finite_replay_benefit_score=int(benefit),
            verdict_class="positive" if benefit else "null",
            honest_verdict="complete_positive_learning_causal_audit"
            if benefit
            else "complete_null_learning_causal_audit",
        )
    value["acceptance_gate_results"]["readiness"] = bool(value["learning_audit_ready_score"])


def terminal_check(candidate: Path) -> Json:
    """Unmodified validators check the exact bytes that publication will expose."""
    replay(json.loads(candidate.read_text()))
    commands = [
        CommandSpec(
            name,
            (str(ROOT / ".venv/bin/python"), "-u", str(ROOT / script), flag, str(candidate)),
            "terminal",
            60,
        )
        for name, script, flag in (
            ("adversarial", "scripts/adversarial_verify.py", "--json"),
            ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
        )
    ]
    receipts = run_commands(
        ROOT, commands, log_dir=candidate.parent / "terminal_logs", heartbeat_s=15
    )
    return dict(
        passed=all(r["passed"] for r in receipts),
        receipts=receipts,
        flagged_adversarial=not receipts[0]["passed"],
    )


def publish(output: Path, value: Json, scratch: Path) -> None:
    """Expose only checked primary bytes and bind both readers to that identity."""
    raw = output.parent / "raw" / output.stem
    value["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
    value["field_principles"] = {
        k: "Bind independent development evidence to immutable primitives; finite replay never proves deployment generalization."
        for k in value
    }
    atomic_json(scratch / "candidate.json", value)
    receipt = publish_primary(output, value, terminal_check)
    atomic_json(raw / "terminal_validation.json", receipt)
    reader = reader_receipt(
        TASK,
        output.parent,
        field="learning_audit_ready_score",
        expected=value["learning_audit_ready_score"],
    )
    atomic_json(raw / "primary_resolution.json", reader)
    if (
        not reader["passed"]
        or reader["gate_path"] != str(output)
        or reader["gate_sha256"] != sha256_file(output)
    ):
        raise ValueError("primary_resolution")


def main(argv: list[str] | None = None) -> int:
    """Run frozen audit, private fixtures or independent cold replay with no model load."""
    from carnot.reporting import learning_audit_validation_7999 as validation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261001", choices=["20261001"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--validation-worker", action="store_true")
    parser.add_argument("--recovery-input", type=Path)
    parser.add_argument("--recovery-arm", choices=m.ARMS)
    parser.add_argument("--recovery-seed", type=int, default=101)
    parser.add_argument("--recovery-directory", type=Path)
    parser.add_argument("--crash-phase", choices=["before", "after"])
    args = parser.parse_args(argv)
    began = time.monotonic()
    progress("begin")
    try:
        if args.recovery_input:
            bundle = json.loads(args.recovery_input.read_text())
            schedule = [
                r
                for r in bundle["acquisition"]
                if r["arm"] == args.recovery_arm and r["seed"] == args.recovery_seed
            ]
            try:
                restored = m.execute(
                    bundle["head"],
                    bundle["sources"],
                    schedule,
                    bundle["stream_targets"],
                    args.recovery_directory,
                    args.crash_phase,
                )
            except m.Crash:
                progress("atomic_boundary_crash_" + str(args.crash_phase))
                return 86
            atomic_json(args.recovery_directory / "result.json", restored)
            progress("independent_recovery_complete")
            return 0
        if args.cold_replay:
            replay(json.loads(args.cold_replay.read_text()))
            progress("replay_passed")
            return 0
        scratch = Path(tempfile.mkdtemp(prefix="carnot-7999-"))
        raw = scratch / "evidence"
        frozen = [reference(ROOT / p) for p in OWNED + TESTS]
        manifest = (
            validation.freeze(raw, scratch)
            if not args.validation_worker
            else dict(commands=[], config=m.CONFIG)
        )
        atomic_json(
            raw / "task_configuration.json",
            dict(config=m.CONFIG, code=frozen, validation_manifest=manifest),
        )
        progress("configuration_and_commands_frozen")
        frozen_at = time.monotonic()
        if args.fixture_input:
            bundle = json.loads(args.fixture_input.read_text())
            failures, plan = [], dict(checks=[], refs=[])
        else:
            failures, plan = authenticate(args.root)
            if not failures:
                bundle = load_bundle(plan)
        value = base(failures) if failures else measure(bundle, raw, bool(args.fixture_input))
        measured_at = time.monotonic()
        value.update(
            preconditions_checked=plan["checks"],
            cited_upstream_artifacts=plan["refs"],
            code_config_hashes=frozen,
        )
        if not args.validation_worker:
            receipts = validation.execute(manifest, raw)
            value["validation_command_manifest_path"] = str(raw / "validation_manifest.json")
            apply_validation(value, receipts, validation.coverage_counts(scratch))
        value["raw_shard_hashes"] = list(value["checkpoints"].values()) + [
            reference(raw / "task_configuration.json")
        ]
        value["reproducibility_checksum"] = canonical_hash(
            dict(config=m.CONFIG, code=frozen, raw=value["raw_shard_hashes"], upstream=plan["refs"])
        )
        ended = time.monotonic()
        value["duration_s"] = ended - began
        value["phase_spans"] = [
            dict(phase="freeze", duration_s=frozen_at - began),
            dict(phase="independent_audit", duration_s=measured_at - frozen_at),
            dict(phase="owned_validation", duration_s=ended - measured_at),
        ]
        value["duration_scope"] = (
            "Actual invocation through owned checks; final-byte validator timings remain in the sidecar."
        )
        progress("publish_begin")
        publish(args.output.absolute(), value, scratch)
        progress("publish_end")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"[exp7999] failed={error}", flush=True)
        return 1
