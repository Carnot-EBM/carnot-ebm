"""REQ-REPORT-7997: static development decisions with sealed target custody.

This CPU reduction invokes no pretrained model and never refits coefficients.
It does not gate online learning or claim globally fresh public observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import evidence_features_custody_7980 as custody
from carnot.reporting import typed_evaluation_7997 as evaluator
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify import typed_development_7997 as m

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7997_v693_typed_development_decisions"
TASK = "exp7997-typed-development-decisions"
MODEL_SPECS: list[str] = []
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/verify/typed_development_7997.py",
    "python/carnot/reporting/typed_evaluation_7997.py",
    "python/carnot/reporting/typed_validation_7997.py",
    f"scripts/experiments/{NAME}.py",
]
TESTS = ["tests/python/test_typed_development_7997.py", f"tests/python/test_{NAME}.py"]
INPUTS = {
    7994: (
        "experiment_7994_v693_development_cohort",
        "cohort_ready_score",
        "sha256:22ffc346ca244d8f2623ec52e56a814207cbeec3d95627346cbd058edd586a2f",
    ),
    7995: (
        "experiment_7995_v693_qwen_development_capture",
        "capture_ready_score",
        "sha256:df6eb8b559181455348b1d806f23c36d13c7f49202c2d49b2233aa80edebe02d",
    ),
    7996: (
        "experiment_7996_v693_sparse_energy_training",
        "sparse_fit_ready_score",
        "sha256:ab4193d06c43b8abae95d2dc866a544246eff17179ec6fb90f79d7bab8b20bda",
    ),
}
reference, checked = custody.reference, custody.checked


def progress(phase: str) -> None:
    """Boundaries let supervisors detect stalls without padding measured time."""
    print(f"[exp7997] phase={phase}", flush=True)


def authenticate(root: Path) -> tuple[list[Json], Json]:
    """Authenticate producer bytes and explicit fields before importing inputs."""
    checks, upstream, refs = [], {}, []
    for eid, (name, ready, pin) in INPUTS.items():
        path = root / "results" / (name + ".json")
        check = custody.operand(
            eid, path, "sha256", pin, sha256_file(path) if path.is_file() else None
        )
        checks.append(check)
        if not check["passed"]:
            continue
        value = json.loads(path.read_text())
        expected = dict(
            experiment_id=eid,
            execution_date="20261001",
            milestone="2026.10.693",
            flagged_adversarial=False,
            **{ready: 1},
        )
        for field, want in expected.items():
            if field not in value:
                raise ValueError("upstream_contract:" + field)
            checks.append(custody.operand(eid, path, field, want, value[field]))
        upstream[eid] = value
        refs.append(
            dict(
                reference(path),
                producer_id=eid,
                producer_invocation_date=value["execution_date"],
                imported_fields=[ready]
                + {
                    7994: ["public_role_manifests", "evaluator_role_manifests"],
                    7995: ["rows"],
                    7996: ["checkpoints", "frozen_scalar_comparator"],
                }[eid],
            )
        )
    if len(upstream) == len(INPUTS):
        assets = [(7996, upstream[7996]["checkpoints"][k]) for k in ("heads", "inputs")]
        assets.append((7996, upstream[7996]["frozen_scalar_comparator"]["source_checkpoint"]))
        assets += [(7994, ref) for ref in upstream[7994]["public_role_manifests"].values()]
        assets += [
            (7994, upstream[7994]["evaluator_role_manifests"][r]) for r in ("calibration", "stream")
        ]
        for eid, ref in assets:
            path = Path(ref["path"])
            checks.append(
                custody.operand(
                    eid,
                    path,
                    "sha256",
                    ref["sha256"],
                    sha256_file(path) if path.is_file() else None,
                )
            )
    return [c for c in checks if not c["passed"]], dict(checks=checks, upstream=upstream, refs=refs)


def disjoint(roles: Json) -> Json:
    """Source hashes establish bounded role separation without a freshness claim."""
    seen, hashes = set(), {}
    for role, rows in roles.items():
        ids = {r["source_cluster_id"] for r in rows}
        if ids & seen:
            raise ValueError("role_overlap")
        seen.update(ids)
        hashes[role] = dict(count=len(ids), sha256=canonical_hash(sorted(ids)))
    return hashes


def load_public(plan: Json) -> tuple[Json, Json, Json, Json]:
    """Open only public development views and historical heads, never targets."""
    cohort, capture, training = (plan["upstream"][i] for i in (7994, 7995, 7996))
    trained = json.loads(checked(training["checkpoints"]["heads"]).read_text())
    heads = {a: trained["heads"][a] for a in ("spline", "logistic", "mlp")}
    scalar_ref = training["frozen_scalar_comparator"]["source_checkpoint"]
    heads["scalar"] = json.loads(checked(scalar_ref).read_text())["heads"]["gibbs"]
    captured = {
        r["family_id"]: r for r in capture["rows"] if r["role"] in ("calibration", "stream")
    }
    public, labels, roles = {}, {}, {}
    for role in ("calibration", "stream", "retention"):
        view = json.loads(checked(cohort["public_role_manifests"][role]).read_text())
        features = {r["family_id"]: r for r in view["features"]}
        roles[role] = [
            dict(source_cluster_id=features[r["family_id"]]["source_normalized_hash"])
            for r in view["request_rows"]
        ]
        if role == "retention":
            continue
        labels[role] = cohort["evaluator_role_manifests"][role]
        public[role] = [
            dict(
                family_id=r["family_id"],
                source_cluster_id=f["source_normalized_hash"],
                q=c["parsed"]["probability"] if c.get("parsed") else None,
                features=f["values"],
                status="completed"
                if c["status"] == "generated" and c["parsed"]["completed"]
                else ("failed" if c["status"] == "generated" else c["status"]),
            )
            for r in view["request_rows"]
            for f, c in [(features[r["family_id"]], captured[r["family_id"]])]
        ]
    historical = json.loads(checked(training["checkpoints"]["inputs"]).read_text())["data"]
    roles.update(historical)
    disjoint(roles)
    return heads, public, labels, roles


def base(failures: list[Json]) -> Json:
    """Terminal blocking names exact operands and makes no measured benefit claim."""
    return dict(
        experiment_id=7997,
        task_id=TASK,
        milestone="2026.10.693",
        run_date="20261001",
        execution_date="20261001",
        schema="carnot.typed_development_decisions.v1",
        honest_verdict="complete_blocked_typed_development_decisions"
        if failures
        else "complete_null_typed_development_decisions",
        verdict_class="blocked" if failures else "null",
        gate_check_summary=failures,
        inference_substrate="aggregation_from_upstream_artifacts"
        if failures
        else "verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        trained_head_specs=[],
        duration_s=0.0,
        phase_spans=[],
        random_seed=69397,
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
        claim_scope="Frozen source-disjoint development evaluation with fallible human source-support targets. Wider historical and pretraining exposure remain unknown. Static results do not gate online learning.",
        evaluation_exposure_scope="bounded_recorded_source_disjoint_development",
        retention_labels_opened=False,
        acceptance_gate_results=dict(readiness=False, benefit=False),
        positive_control_results={},
        preconditions_checked=[],
        validation_command_manifest_path=None,
        validation_receipts=[],
        coverage_statement_counts={},
        flagged_adversarial=False,
        terminal_validation_sidecar_path=None,
        decision_measurement_ready_score=0,
        decision_benefit_score=0,
        policies={},
        checkpoint_hashes={},
        checkpoints={},
        paired_comparisons={},
        adjusted_p_values={},
        confidence_intervals={},
        all_intended_bounds={},
        calibration_rows=[],
        config=m.CONFIG,
        label_access_events=[],
        repository_health={},
        historical_required_failures=[],
    )


def read_targets(ref: Json) -> list[Json]:
    """Only this evaluator boundary opens known development target files."""
    document = json.loads(checked(ref).read_text())
    return [dict(family_id=r["family_id"], y=r["y"]) for r in document["rows"]]


def measure(heads: Json, public: Json, labels: Json, roles: Json, raw: Path) -> Json:
    """Seal coefficients and predictions in order before each evaluator exposure."""
    value = base([])
    value["positive_control_results"] = evaluator.controls()
    progress("circular_controls_complete")
    value["role_hashes"] = disjoint(roles)
    role_sources = {
        role: [dict(source_cluster_id=r["source_cluster_id"]) for r in rows]
        for role, rows in roles.items()
    }
    for name, data in [("heads", heads), ("public", public), ("roles", role_sources)]:
        atomic_json(raw / (name + ".json"), data)
        value["checkpoints"][name] = reference(raw / (name + ".json"))
    before = value["checkpoints"]["heads"]["sha256"]
    progress("heads_sealed_calibration_prediction_begin")
    calibration = m.predictions(heads, public["calibration"])
    atomic_json(raw / "calibration_predictions.json", dict(rows=calibration))
    value["checkpoints"]["calibration_predictions"] = reference(
        raw / "calibration_predictions.json"
    )
    progress("calibration_predictions_sealed_target_open")
    policies, scores = evaluator.calibrate(calibration, read_targets(labels["calibration"]))
    value["label_access_events"].append(
        dict(
            role="calibration",
            purpose="temperature_brier_only",
            preceded_by=value["checkpoints"]["calibration_predictions"],
            **labels["calibration"],
        )
    )
    atomic_json(raw / "policies.json", policies)
    value["checkpoints"]["policies"] = reference(raw / "policies.json")
    stream = m.select(m.predictions(heads, public["stream"]), policies)
    atomic_json(raw / "stream_predictions.json", dict(rows=stream))
    value["checkpoints"]["stream_predictions"] = reference(raw / "stream_predictions.json")
    progress("stream_predictions_sealed_target_open")
    reduced = evaluator.evaluate(stream, read_targets(labels["stream"]))
    value["label_access_events"].append(
        dict(
            role="stream",
            purpose="static_evaluation_only",
            preceded_by=value["checkpoints"]["stream_predictions"],
            **labels["stream"],
        )
    )
    value.update(
        reduced,
        policies=policies,
        calibration_rows=scores,
        evaluator_targets=labels,
        checkpoint_hashes=dict(
            before=before, after=sha256_file(checked(value["checkpoints"]["heads"])), unchanged=True
        ),
    )
    value["decision_benefit_score"] = int(reduced["benefit"])
    if reduced["benefit"]:
        value.update(
            verdict_class="positive", honest_verdict="complete_positive_typed_development_decisions"
        )
    value["static_method_disposition"] = (
        "benefit_detected"
        if reduced["benefit"]
        else (
            "closed_in_this_development_scope"
            if reduced["evaluation_support"]["passed"] and reduced["genuine_headroom"]
            else (
                "underpowered_does_not_refute"
                if not reduced["evaluation_support"]["passed"]
                else "no_headroom_does_not_refute"
            )
        )
    )
    value["trained_head_specs"] = [
        dict(
            arm=a,
            seeds=[h.get("seed", i) for i, h in enumerate(hs)],
            pretrained=False,
            fitted_current_steps=0,
            parameter_count_per_seed=len(hs[0]["parameters"]),
        )
        for a, hs in heads.items()
    ]
    atomic_json(raw / "rows.json", dict(rows=value["rows"]))
    value["checkpoints"]["rows"] = reference(raw / "rows.json")
    return value


def replay(value: Json) -> Json:
    """Cold reconstruction rejects both row and summary drift from sealed inputs."""
    if value["retention_labels_opened"] or (
        value["verdict_class"] in ("blocked", "disqualified")
        and value["decision_measurement_ready_score"]
    ):
        raise ValueError("unsafe_readiness")
    if not value["checkpoints"]:
        return dict(passed=True, blocked=True)
    for ref in (
        list(value["checkpoints"].values())
        + value["code_config_hashes"]
        + value["raw_shard_hashes"]
    ):
        checked(ref)
    heads = json.loads(checked(value["checkpoints"]["heads"]).read_text())
    public = json.loads(checked(value["checkpoints"]["public"]).read_text())
    if (
        disjoint(json.loads(checked(value["checkpoints"]["roles"]).read_text()))
        != value["role_hashes"]
    ):
        raise ValueError("role_hash_drift")
    candidates = m.predictions(heads, public["calibration"])
    if (
        candidates
        != json.loads(checked(value["checkpoints"]["calibration_predictions"]).read_text())["rows"]
    ):
        raise ValueError("calibration_prediction_drift")
    policies, calibration = evaluator.calibrate(
        candidates, read_targets(value["evaluator_targets"]["calibration"])
    )
    if policies != value["policies"] or calibration != value["calibration_rows"]:
        raise ValueError("policy_drift")
    stream = m.select(m.predictions(heads, public["stream"]), policies)
    if (
        stream
        != json.loads(checked(value["checkpoints"]["stream_predictions"]).read_text())["rows"]
    ):
        raise ValueError("stream_prediction_drift")
    reduced = evaluator.evaluate(stream, read_targets(value["evaluator_targets"]["stream"]))
    if (
        any(value[k] != v for k, v in reduced.items())
        or value["checkpoint_hashes"]["before"] != value["checkpoint_hashes"]["after"]
    ):
        raise ValueError("reduction_drift")
    return dict(passed=True, rows=len(value["rows"]))


def apply_validation(value: Json, receipts: list[Json], counts: Json) -> None:
    """Owned failures invalidate readiness; repository health keeps its true exit."""
    value.update(
        validation_receipts=receipts,
        coverage_statement_counts=counts,
        repository_health=dict(current=[r for r in receipts if not r["required"]]),
    )
    coverage_ok = set(counts) == set(OWNED) and all(
        c["num_statements"] > 0 and c["missing_lines"] == 0 for c in counts.values()
    )
    failed = any(not r["passed"] for r in receipts if r["required"])
    science_ok = value.get("positive_control_results", {}).get("passed", False) and value.get(
        "equivalent_classifier_identity", {}
    ).get("passed", False)
    custody_failure = (
        value.get("prior_exposure_receipt", {}).get("predictions_sealed_before_stream_label_access")
        is False
    )
    if (
        failed
        or custody_failure
        or (receipts and value["checkpoints"] and (not coverage_ok or not science_ok))
    ):
        value.update(
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_typed_development_decisions",
            decision_measurement_ready_score=0,
            decision_benefit_score=0,
            static_method_disposition="disqualified_no_valid_method_conclusion",
        )
    elif receipts and coverage_ok and value["checkpoints"] and not value["verifier_is_oracle"]:
        value["decision_measurement_ready_score"] = 1
    value["acceptance_gate_results"] = dict(
        readiness=bool(value["decision_measurement_ready_score"]),
        benefit=bool(value["decision_benefit_score"]),
    )


def terminal_check(candidate: Path) -> Json:
    """Independent unmodified validators inspect actual serialized bytes."""
    replay(json.loads(candidate.read_text()))
    commands = [
        CommandSpec(
            name,
            (str(ROOT / ".venv/bin/python"), "-u", str(ROOT / script), flag, str(candidate)),
            "terminal",
            60,
        )
        for name, script, flag in [
            ("adversarial", "scripts/adversarial_verify.py", "--json"),
            ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
        ]
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
    """Only checked bytes become the primary selected by both conductor readers."""
    raw = output.parent / "raw" / output.stem
    value["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
    value["field_principles"] = {
        k: "Bind current bytes and owned checks; readiness is separate from benefit." for k in value
    }
    candidate = scratch / "candidate.json"
    atomic_json(candidate, value)
    first = terminal_check(candidate)
    if not first["passed"]:
        value["flagged_adversarial"] = first["flagged_adversarial"]
        value["historical_required_failures"].append(first)
        apply_validation(
            value,
            value["validation_receipts"]
            + [dict(name="terminal_failure", required=True, passed=False)],
            value["coverage_statement_counts"],
        )
    receipt = publish_primary(output, value, terminal_check)
    atomic_json(raw / "terminal_validation.json", receipt)
    reader = reader_receipt(
        TASK,
        output.parent,
        field="decision_measurement_ready_score",
        expected=value["decision_measurement_ready_score"],
    )
    atomic_json(raw / "primary_resolution.json", reader)
    if (
        not reader["passed"]
        or reader["gate_path"] != str(output)
        or reader["gate_sha256"] != sha256_file(output)
    ):
        raise ValueError("primary_resolution")


def main(argv: list[str] | None = None) -> int:
    """Real CLI retains private artifacts and exposes bounded cold reconstruction."""
    from carnot.reporting import typed_validation_7997 as validation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261001", choices=["20261001"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--validation-worker", action="store_true")
    parser.add_argument("--check-exposure", type=Path)
    parser.add_argument("--prior-exposure-receipt", type=Path)
    args = parser.parse_args(argv)
    began = time.monotonic()
    progress("begin")
    try:
        if args.check_exposure:
            if (
                json.loads(args.check_exposure.read_text())[
                    "predictions_sealed_before_stream_label_access"
                ]
                is not True
            ):
                raise ValueError("stream_label_custody")
            return 0
        if args.cold_replay:
            replay(json.loads(args.cold_replay.read_text()))
            progress("replay_passed")
            return 0
        scratch = Path(tempfile.mkdtemp(prefix="carnot-7997-"))
        raw = scratch / "evidence"
        frozen = [reference(ROOT / p) for p in OWNED + TESTS]
        atomic_json(raw / "task_configuration.json", dict(config=m.CONFIG, code=frozen))
        manifest = (
            validation.freeze(raw, scratch, args.prior_exposure_receipt)
            if not args.validation_worker
            else {}
        )
        progress("configuration_code_and_commands_frozen")
        if args.fixture_input:
            fixture = json.loads(args.fixture_input.read_text())
            heads, public = fixture["heads"], fixture["public"]
            roles, labels = public, {}
            for role, targets in fixture["targets"].items():
                atomic_json(raw / "evaluator" / (role + ".json"), dict(rows=targets))
                labels[role] = reference(raw / "evaluator" / (role + ".json"))
            failures, plan = [], dict(checks=[], refs=[], upstream={})
        else:
            failures, plan = authenticate(args.root)
            if not failures:
                heads, public, labels, roles = load_public(plan)
        value = base(failures) if failures else measure(heads, public, labels, roles, raw)
        if args.fixture_input:
            value.update(
                verdict_class="circular_positive",
                honest_verdict="complete_circular_positive_typed_development_wiring",
                verifier_is_oracle=True,
                decision_benefit_score=0,
            )
        value.update(
            preconditions_checked=plan["checks"],
            cited_upstream_artifacts=plan["refs"],
            code_config_hashes=frozen,
        )
        if args.prior_exposure_receipt:
            value["prior_exposure_receipt"] = json.loads(args.prior_exposure_receipt.read_text())
            atomic_json(raw / "prior_exposure_receipt.json", value["prior_exposure_receipt"])
            value["prior_exposure_reference"] = reference(raw / "prior_exposure_receipt.json")
        value["imported_checkpoint_hashes"] = [
            ref for p in plan["upstream"].values() for ref in p.get("checkpoints", {}).values()
        ]
        value["cited_upstream_artifacts"] += [
            dict(ref, imported_fields=["heads_seal"])
            for ref in plan["upstream"].get(7996, {}).get("cited_upstream_artifacts", [])
            if ref.get("producer_id") == 7972
        ]
        if manifest:
            receipts = validation.execute(manifest, raw)
            value["validation_command_manifest_path"] = str(raw / "validation_manifest.json")
            apply_validation(value, receipts, validation.coverage_counts(scratch))
        value["raw_shard_hashes"] = (
            list(value["checkpoints"].values())
            + list(value.get("evaluator_targets", {}).values())
            + [reference(raw / "task_configuration.json")]
            + ([value["prior_exposure_reference"]] if args.prior_exposure_receipt else [])
        )
        value["reproducibility_checksum"] = canonical_hash(
            dict(config=m.CONFIG, code=frozen, raw=value["raw_shard_hashes"], upstream=plan["refs"])
        )
        value["duration_s"] = time.monotonic() - began
        value["duration_scope"] = (
            "Configuration, measurement and owned checks; terminal check durations are retained in the sidecar."
        )
        value["phase_spans"] = [
            dict(
                phase="frozen_development_evaluation_and_validation", duration_s=value["duration_s"]
            )
        ]
        progress("publish_begin")
        publish(args.output.absolute(), value, scratch)
        progress("publish_end")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"[exp7997] failed={error}", flush=True)
        return 1
