"""REQ-REPORT-8000: checked confidence replay without pretrained calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting import evidence_features_custody_7980 as custody
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify import delayed_confidence_8000 as m
from carnot.verify.qwen_energy_calibration_7972 import predict

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_8000_v693_delayed_confidence"
TASK = "exp8000-delayed-confidence"
MODEL_SPECS: list[str] = []
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/verify/delayed_confidence_8000.py",
    "python/carnot/reporting/delayed_confidence_validation_8000.py",
    f"scripts/experiments/{NAME}.py",
]
TESTS = ["tests/python/test_delayed_confidence_8000.py", f"tests/python/test_{NAME}.py"]
INPUTS = {
    7972: (
        "experiment_7972_v691_qwen_energy_calibration",
        "qwen_calibration_ready_score",
        "sha256:a47d19ca7af500014117ff083249469597cd7e6bc0ff57778098fa39cfa41c63",
    ),
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
}


def authenticate(root: Path) -> tuple[list[Json], Json]:
    """Missing bytes block, while absent required fields are contract errors."""
    checks, upstream, refs = [], {}, []
    for eid, (name, ready, pin) in INPUTS.items():
        path = root / "results" / (name + ".json")
        checks.append(
            custody.operand(eid, path, "sha256", pin, sha256_file(path) if path.is_file() else None)
        )
        if not checks[-1]["passed"]:
            continue
        value = json.loads(path.read_text())
        for field, expected in dict(
            experiment_id=eid, execution_date="20261001", flagged_adversarial=False, **{ready: 1}
        ).items():
            if field not in value:
                raise ValueError("upstream_contract:" + field)
            checks.append(custody.operand(eid, path, field, expected, value[field]))
        upstream[eid] = value
        refs.append(
            dict(
                custody.reference(path),
                producer_id=eid,
                producer_invocation_date=value["execution_date"],
                imported_fields=[ready]
                + (
                    ["calibrator_checkpoints"]
                    if eid == 7972
                    else ["public_role_manifests", "evaluator_role_manifests"]
                    if eid == 7994
                    else ["rows", "raw_shard_hashes"]
                ),
            )
        )
    if {7972, 7994, 7995} <= set(upstream):
        assets = [(7972, r) for r in upstream[7972]["calibrator_checkpoints"].values()]
        assets += [(7995, r) for r in upstream[7995]["raw_shard_hashes"]]
        assets += [
            (7994, upstream[7994][kind][role])
            for kind in ("public_role_manifests", "evaluator_role_manifests")
            for role in ("calibration", "stream")
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


def load(plan: Json) -> Json:
    """Historical coefficients consume only new scalar capture probabilities."""
    upstream = plan["upstream"]
    head = json.loads(
        custody.checked(upstream[7972]["calibrator_checkpoints"]["heads"]).read_text()
    )["heads"]["gibbs"][0]
    captures = {r["family_id"]: r for r in upstream[7995]["rows"]}
    roles, targets, shards = {}, {}, []
    for ref in upstream[7995]["raw_shard_hashes"]:
        custody.checked(ref)
        shards.append(ref)
    for role in ("calibration", "stream"):
        public_ref = upstream[7994]["public_role_manifests"][role]
        target_ref = upstream[7994]["evaluator_role_manifests"][role]
        view = json.loads(custody.checked(public_ref).read_text())
        gold = (
            json.loads(custody.checked(target_ref).read_text())
            if role == "calibration"
            else {"rows": []}
        )
        features = {r["family_id"]: r for r in view["features"]}
        targets[role] = {r["family_id"]: r["y"] for r in gold["rows"]}
        roles[role] = []
        for r in view["request_rows"]:
            c = captures[r["family_id"]]
            q = c["parsed"]["probability"] if c.get("parsed") and c["eligibility"] else None
            p = (
                float(predict(dict(head, temperature=1.0), np.array([q]))[0])
                if q is not None
                else None
            )
            roles[role].append(
                dict(
                    family_id=r["family_id"],
                    source_cluster_id=features[r["family_id"]]["source_normalized_hash"],
                    p=p,
                    status="completed" if q is not None else c["status"],
                )
            )
        shards.extend([public_ref, target_ref])
    historical = json.loads(
        custody.checked(upstream[7972]["calibrator_checkpoints"]["primitives"]).read_text()
    )
    seen = {r["source_cluster_id"] for rows in historical.values() for r in rows}
    calibration_ids = {r["source_cluster_id"] for r in roles["calibration"]}
    stream_ids = {r["source_cluster_id"] for r in roles["stream"]}
    if calibration_ids & stream_ids or seen & (calibration_ids | stream_ids):
        raise ValueError("role_overlap")
    return dict(
        calibration=[
            dict(r, y=targets["calibration"][r["family_id"]])
            for r in roles["calibration"]
            if r["p"] is not None and targets["calibration"][r["family_id"]] in (0, 1)
        ],
        stream=roles["stream"],
        stream_target_ref=upstream[7994]["evaluator_role_manifests"]["stream"],
        head=head,
        raw_shard_hashes=shards,
    )


def base(failures: list[Json]) -> Json:
    """Readiness begins at zero until every owned check completes."""
    return dict(
        experiment_id=8000,
        task_id=TASK,
        milestone="2026.10.693",
        run_date="20261001",
        execution_date="20261001",
        schema="carnot.delayed_confidence.v1",
        honest_verdict="complete_blocked_delayed_confidence"
        if failures
        else "complete_null_delayed_confidence",
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
        confidence_measurement_ready_score=0,
        confidence_benefit_score=0,
        flagged_adversarial=False,
        verifier_is_oracle=False,
        config=m.CONFIG,
        recurrence_definition=m.RECURRENCE,
        rows=[
            dict(
                r,
                numerator=0,
                denominator=1,
                eligibility=False,
                failure_status=False,
                censor_status=True,
            )
            for r in failures
        ],
        issued_state_rows=[],
        pending_feedback={},
        coverage_windows=[],
        restart_rows=[],
        point_brier_parity={},
        delay_sensitivity=[],
        point_probability_hash=None,
        acceptance_gate_results={},
        positive_control_results={},
        claim_scope="finite development-stream diagnostic; fallible human source-support annotations",
        interval_scope="paired nonoverlapping20-slot block bootstrap; dependent adaptive stream; no deployment theorem",
        cited_upstream_artifacts=[],
        raw_shard_hashes=[],
        preconditions_checked=[],
        validation_receipts=[],
        coverage_statement_counts={},
        code_config_hashes={},
        validation_command_manifest_path=None,
        terminal_validation_sidecar_path=None,
        duration_s=0.0,
        phase_spans=[],
        random_seed=69300,
        reproducibility_checksum=None,
        sample_size_budget={},
        methodology_note="Repeated arms and bootstrap draws are not independent sources.",
    )


def finish_measurement(value: Json, bundle: Json, raw: Path, fixture: bool) -> None:
    """Seal primitive inputs so cold replay checks both sets and reductions."""
    atomic_json(
        raw / "point_predictions.json",
        dict(calibration=bundle["calibration"], stream=bundle["stream"]),
    )
    value["point_prediction_seal"] = custody.reference(raw / "point_predictions.json")
    if "targets" not in bundle:
        bundle["targets"] = {
            r["family_id"]: r["y"]
            for r in json.loads(custody.checked(bundle["stream_target_ref"]).read_text())["rows"]
        }
    atomic_json(raw / "primitive_bundle.json", bundle)
    value["primitive_bundle"] = custody.reference(raw / "primitive_bundle.json")
    result = m.measure(bundle)
    value.update(result)
    atomic_json(raw / "slot128_states.json", dict(states=value.pop("restart_states")))
    value["restart_state_checkpoint"] = custody.reference(raw / "slot128_states.json")
    value["verifier_is_oracle"] = fixture
    value["verdict_class"] = (
        "circular_positive"
        if fixture
        else ("positive" if value["confidence_benefit_score"] else "null")
    )
    value["honest_verdict"] = "complete_" + value["verdict_class"] + "_delayed_confidence"
    if fixture:
        value["confidence_benefit_score"] = 0
    n = len(bundle["stream"])
    eligible = sum(r["p"] is not None and r["status"] == "completed" for r in bundle["stream"])
    completed = sum(
        r["eligibility"] for r in value["rows"] if r["arm"] == "scalar" and r["delay"] == 20
    )
    value["sample_size_budget"] = dict(
        intended=n,
        eligible=eligible,
        started=eligible,
        completed=completed,
        excluded=n - eligible,
        failed=sum(r["status"] == "failed" for r in bundle["stream"]),
        censored=eligible - completed,
        independent=len({r["source_cluster_id"] for r in bundle["stream"] if r["p"] is not None}),
        primary_evaluated_groups=value["delay_sensitivity"][0]["eligible_groups"],
        primary_complete_blocks=value["delay_sensitivity"][0]["complete_blocks"],
        interpretation="descriptive only when support fails; seeds never multiply groups",
    )
    value["raw_shard_hashes"] = bundle.get("raw_shard_hashes", [])
    value["trained_head_specs"] = [
        dict(
            kind="historical_scalar_gibbs",
            fitted_current=False,
            parameter_count=33,
            producer=7972,
            seed=69101,
        ),
        dict(
            kind="calibration_temperature",
            candidates=[0.5, 1.0, 2.0],
            selected=m.calibrate(bundle["calibration"])[0],
            fitted_current=True,
            coefficient_refit_steps=0,
        ),
    ]


def apply_validation(value: Json, receipts: list[Json], counts: Json) -> None:
    """A failed owned command disqualifies; repository health stays diagnostic."""
    value["validation_receipts"], value["coverage_statement_counts"] = receipts, counts
    passed = (
        bool(receipts)
        and all(r["passed"] for r in receipts if r["required"])
        and (
            set(counts) == set(OWNED)
            and all(c["num_statements"] > 0 and c["missing_lines"] == 0 for c in counts.values())
        )
    )
    passed = (
        passed
        and bool(value.get("primitive_bundle"))
        and value["positive_control_results"]["passed"]
        and value["point_brier_parity"]["passed"]
        and all(r["passed"] for r in value["restart_rows"])
        if value["verdict_class"] != "blocked"
        else passed
    )
    value["repository_health"] = [r for r in receipts if not r["required"]]
    if value["verdict_class"] == "blocked":
        return
    if not passed:
        value.update(
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_delayed_confidence",
            confidence_benefit_score=0,
            confidence_measurement_ready_score=0,
        )
    else:
        value["confidence_measurement_ready_score"] = int(not value["verifier_is_oracle"])


def replay(value: Json) -> Json:
    """Reconstruction rejects changed issue states, feedback or aggregate windows."""
    if value["verdict_class"] == "blocked":
        if value["confidence_measurement_ready_score"]:
            raise ValueError("blocked_readiness")
        return dict(passed=True, blocked=True)
    bundle = json.loads(custody.checked(value["primitive_bundle"]).read_text())
    reconstructed = m.measure(bundle)
    for key in (
        "issued_state_rows",
        "rows",
        "pending_feedback",
        "coverage_windows",
        "restart_rows",
        "point_brier_parity",
        "point_probability_hash",
        "delay_sensitivity",
        "acceptance_gate_results",
    ):
        if canonical_hash(value[key]) != canonical_hash(reconstructed[key]):
            raise ValueError("reduction_drift:" + key)
    saved = json.loads(custody.checked(value["restart_state_checkpoint"]).read_text())["states"]
    for key, state in saved.items():
        continued = m.resume(bundle, state, 129)
        arm, delay = key.rsplit("-", 1)
        original = m.run(bundle, arm, int(delay))
        if canonical_hash(continued["pending"]) != canonical_hash(
            original["pending"]
        ) or canonical_hash(continued["feedback"]) != canonical_hash(original["feedback"]):
            raise ValueError("cold_restart_drift")
    return dict(passed=True, replay_passed=True, cold_restart_slot=128)


def terminal_check(path: Path) -> Json:
    """Existing validators judge exact candidate bytes without changing their rules."""
    from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands

    receipts = run_commands(
        ROOT,
        [
            CommandSpec(
                name,
                (str(ROOT / ".venv/bin/python"), "-u", str(ROOT / script), flag, str(path)),
                "terminal",
                60,
            )
            for name, script, flag in (
                ("adversarial", "scripts/adversarial_verify.py", "--json"),
                ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
            )
        ],
        log_dir=path.parent / "terminal_logs",
        heartbeat_s=15,
    )
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def publish(output: Path, value: Json, scratch: Path) -> None:
    """Both conductor readers select a private candidate before checked publication."""
    candidate = scratch / "reader_check" / output.name
    atomic_json(candidate, value)
    if not terminal_check(candidate)["passed"]:
        raise ValueError("candidate_rejected")
    selected = reader_receipt(
        TASK,
        candidate.parent,
        field="confidence_measurement_ready_score",
        expected=value["confidence_measurement_ready_score"],
    )
    if not selected["passed"]:
        raise ValueError("reader_mismatch")
    value["primary_resolution_receipt"] = selected
    atomic_json(candidate, value)
    publication = publish_primary(output, value, terminal_check)
    atomic_json(output.parent / "raw" / output.stem / "terminal_validation.json", publication)
    print(f"[exp8000] published={output} sha256={publication['primary_sha256']}", flush=True)


def main(argv: list[str] | None = None) -> int:
    """Private worker paths run the same numerical core without recursive validation."""
    from carnot.reporting import delayed_confidence_validation_8000 as validation

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", choices=["20261001"], default="20261001")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument("--validation-worker", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    started = time.monotonic()
    print("[exp8000] phase=preconditions model_loads=0 generation_calls=0", flush=True)
    if args.cold_replay:
        try:
            print(json.dumps(replay(json.loads(args.cold_replay.read_text()))), flush=True)
            return 0
        except (OSError, ValueError, KeyError) as exc:
            print(f"[exp8000] replay_failed={exc}", flush=True)
            return 1
    scratch = Path(tempfile.mkdtemp(prefix="carnot-8000-"))
    raw = scratch / "evidence"
    raw.mkdir()
    output = args.output or args.root / "results" / (NAME + ".json")
    failures, plan = (
        ([], dict(checks=[], refs=[])) if args.fixture_input else authenticate(args.root)
    )
    value = base(failures)
    value["cited_upstream_artifacts"], value["preconditions_checked"] = plan["refs"], plan["checks"]
    manifest = validation.freeze(raw, scratch)
    value["validation_command_manifest_path"] = str(raw / "validation_manifest.json")
    value["code_config_hashes"] = {p: sha256_file(ROOT / p) for p in OWNED + TESTS}
    value["code_config_hashes"]["config"] = canonical_hash(m.CONFIG)
    print("[exp8000] phase=configuration_frozen", flush=True)
    if not failures:
        bundle = json.loads(args.fixture_input.read_text()) if args.fixture_input else load(plan)
        finish_measurement(value, bundle, raw, bool(args.fixture_input))
    measured = time.monotonic()
    value["phase_spans"].append(
        dict(name="preconditions_and_replay", duration_s=measured - started)
    )
    if not args.validation_worker:
        print("[exp8000] phase=validation_begin", flush=True)
        receipts = validation.execute(manifest, raw)
        apply_validation(value, receipts, validation.coverage_counts(scratch))
    value["phase_spans"].append(dict(name="validation", duration_s=time.monotonic() - measured))
    value["duration_s"] = time.monotonic() - started
    value["reproducibility_checksum"] = canonical_hash(
        dict(
            config=m.CONFIG,
            code=value["code_config_hashes"],
            upstream=value["cited_upstream_artifacts"],
            point_probability_hash=value["point_probability_hash"],
        )
    )
    value["terminal_validation_sidecar_path"] = str(
        output.parent / "raw" / output.stem / "terminal_validation.json"
    )
    print("[exp8000] phase=final_byte_validation", flush=True)
    publish(output, value, scratch)
    return 0
