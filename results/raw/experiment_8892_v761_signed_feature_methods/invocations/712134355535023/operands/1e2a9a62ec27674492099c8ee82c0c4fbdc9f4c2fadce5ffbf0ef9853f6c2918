"""REQ-VERIFY-8828: bound original replay and publish only measured qualification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import signal_replay_qualification_8828 as q
from carnot.reporting.current_work_receipt import (
    build_current_work_receipt,
    validate_current_work_receipt,
)
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]
ROOT, CLI = q.ROOT, q.CLI


def partition(raw: Path, mode: str, ref: Json | None = None, index: int = 0) -> Json:
    """Use the existing output-forwarding supervisor with its original deadline."""
    argv = [
        "env",
        "-u",
        "PYTHONPATH",
        "-C",
        "/tmp",
        str(ROOT / ".venv/bin/python"),
        "-u",
        str(ROOT / CLI),
        "--child",
        mode,
        "--raw-dir",
        str(raw),
        "--index",
        str(index),
    ]
    if ref is not None:
        argv += ["--snapshot", ref["path"], "--snapshot-hash", ref["sha256"]]
    receipt = q.old.commands([(mode, argv, "owned", 600)], raw / "logs")[0]
    q.atomic_json(raw / "receipt.json", receipt)
    q.d.require(receipt["passed"], "child_failed:" + mode)
    return receipt


def execute(raw: Path) -> Json:
    """Pilot the authentication and each objective before completing the inventory."""
    auth = partition(raw, "authenticate")
    ref = q.d.reference(raw / "snapshot.json")
    snapshot = q.load_snapshot(ref)
    partitions, fits = [], {}
    for index in [0, 3, 6]:
        target = raw / "partitions" / f"fit-{index:02}"
        receipt = partition(target, f"fit-{index:02}", ref, index)
        partitions.append(dict(index=index, **receipt))
        fits[index] = q.d.load(q.d.reference(target / f"fit-{index:02}.json"))["value"]
    projected = q.projection(auth["duration_s"], [r["duration_s"] for r in partitions])
    q.atomic_json(raw / "replay_projection.json", projected)
    q.d.require(projected["feasible"], "replay_projection_infeasible")
    for index in range(45):
        if index in fits:
            continue
        target = raw / "partitions" / f"fit-{index:02}"
        receipt = partition(target, f"fit-{index:02}", ref, index)
        partitions.append(dict(index=index, **receipt))
        fits[index] = q.d.load(q.d.reference(target / f"fit-{index:02}.json"))["value"]
    control = partition(raw / "partitions/controls", "controls", ref)
    controls = q.d.load(q.d.reference(raw / "partitions/controls/controls.json"))["value"]
    q.bindings(snapshot["inputs"], "operand_hash")
    evidence = {
        k: snapshot[k]
        for k in ("diagnostic_rows", "development_rows", "split_manifest", "source_artifact_hashes")
    }
    evidence.update(
        fits=[fits[i] for i in range(45)],
        controls=controls,
        snapshot=ref,
        partition_rows=sorted(partitions, key=lambda r: r["index"]),
        authentication_receipt=auth,
        control_receipt=control,
        replay_projection=projected,
    )
    q.atomic_json(raw / "evidence.json", evidence)
    return evidence


def replay(path: Path) -> Json:
    """Fresh replay checks admitted raw operands and all exact original fit outputs."""
    value: Json = json.loads(path.read_bytes())
    q.bindings(value["code_config_hashes"], "code_hash")
    for receipt in value["validation_receipts"]:
        log = Path(receipt["log_path"])
        if not log.is_absolute():
            log = ROOT / log
        q.d.require(q.sha256_file(log) == receipt["log_sha256"], "validation_log")
    if not value["raw_shard_hashes"] or value["verdict_class"] == "blocked":
        q.d.require(
            value["diagnosis_rows_ready_score"] == value["objective_candidate_ready_score"] == 0,
            "blocked_readiness",
        )
        return dict(passed=True, replay_passed=True, scope="blocked_no_measurement")
    evidence = q.d.load(value["raw_shard_hashes"][0])
    snapshot = q.load_snapshot(evidence["snapshot"], verify=True)
    q.d.require(value["code_config_hashes"] == snapshot["code"], "code_inventory")
    q.d.require(value["original_inventory"] == q.inventory(), "original_inventory")
    for key in ("partition_rows", "replay_projection"):
        q.d.require(value[key] == evidence[key], "reduction_drift:" + key)
    q.d.require(
        value["linear_comparator_rows"] == value["objective_selection"]["paired_rows"],
        "linear_drift",
    )
    q.d.require(
        len(evidence["fits"]) == 45
        and [
            {k: row[k] for k in ("seed", "objective", "lr", "updates")} for row in evidence["fits"]
        ]
        == q.inventory(),
        "fit_inventory",
    )
    target = Path(evidence["snapshot"]["path"]).parent / "replay" / str(time.time_ns())
    for index, expected in enumerate(evidence["fits"]):
        work = target / str(index)
        partition(work, f"fit-{index:02}", evidence["snapshot"], index)
        actual = q.d.load(q.d.reference(work / f"fit-{index:02}.json"))["value"]
        q.d.require(actual == expected, "fit_drift")
    partition(target / "controls", "controls", evidence["snapshot"])
    actual_controls = q.d.load(q.d.reference(target / "controls/controls.json"))["value"]
    q.d.require(actual_controls == evidence["controls"], "control_drift")
    q.d.require(
        all(
            evidence[k] == snapshot[k]
            for k in (
                "diagnostic_rows",
                "development_rows",
                "split_manifest",
                "source_artifact_hashes",
            )
        ),
        "evidence_drift",
    )
    for key, expected in q.old.reduce(evidence).items():
        if key.endswith("_ready_score") and value["verdict_class"] == "disqualified":
            expected = 0
        q.d.require(value[key] == expected, "reduction_drift:" + key)
    q.bindings(snapshot["inputs"], "operand_hash")
    return dict(
        passed=True, replay_passed=True, fits=45, updates=18000, primary_sha256=q.sha256_file(path)
    )


def validate(raw: Path) -> Json:
    """Reuse scoped validation, original negative tests, private E2E and health separation."""
    with patch.multiple(q.old, OWNED=q.OWNED, TESTS=q.TESTS + q.old.TESTS, CLI=CLI):
        return q.old.validate(raw)


def terminal(candidate: Path, raw: Path) -> Json:
    """Keep the unchanged scanners and the 600 second outside-checkout replay."""
    with patch.object(q.old, "CLI", CLI):
        return q.old.terminal(candidate, raw)


def gate(error: Exception) -> Json:
    """Name the exact external absence separately from an internal code failure."""
    missing = isinstance(error, (FileNotFoundError, KeyError))
    path = Path(getattr(error, "filename", None) or q.HISTORICAL)
    return dict(
        check="historical_operands" if missing else "owned_execution",
        upstream="exp8815-learning-signal-diagnosis",
        artifact_field=str(error),
        expected="readable hash-bound complete original operands",
        observed=str(error),
        path=str(path),
        hash=q.sha256_file(path) if path.is_file() else None,
        status="missing_file"
        if isinstance(error, FileNotFoundError)
        else "missing_field"
        if isinstance(error, KeyError)
        else "rejected_producer",
    )


def main(argv: list[str] | None = None) -> int:
    """Output custody and replay flags cannot change the registered experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261105")
    parser.add_argument(
        "--output", "--private-output", type=Path, default=ROOT / "results" / (q.NAME + ".json")
    )
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument(
        "--child", choices=["authenticate", "controls"] + [f"fit-{i:02}" for i in range(45)]
    )
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--snapshot-hash")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args(argv)
    if args.replay or args.child:
        try:
            if args.replay:
                print(json.dumps(replay(args.replay)), flush=True)
            else:
                q.d.require(args.raw_dir is not None, "missing_raw_dir")
                ref = (
                    dict(path=str(args.snapshot), sha256=args.snapshot_hash)
                    if args.snapshot
                    else None
                )
                q.child(args.raw_dir, args.child, ref, args.index)
            return 0
        except (ValueError, OSError, KeyError, IndexError) as error:
            print(f"replay_rejected:{error}", flush=True)
            return 1
    if args.date != "20261105":
        print("date_rejected: expected 20261105", flush=True)
        return 1
    start = time.monotonic_ns()
    output = args.output.absolute()
    raw = (
        args.raw_dir or output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    ).absolute()
    evidence, preconditions, gates = None, {}, []
    verdict, stage = "null", "preconditions"
    q.d.progress("preconditions_before", 0, 1)
    try:
        preconditions = q.preconditions(raw)
        q.d.progress("preconditions_after", 1, 0)
        stage = "execution"
        evidence = execute(raw)
        value = q.old.reduce(evidence)
        verdict = "circular_positive" if value["objective_candidate_ready_score"] else "null"
    except (ValueError, OSError, KeyError, IndexError) as error:
        verdict = (
            "blocked"
            if stage == "preconditions" and isinstance(error, (FileNotFoundError, KeyError))
            else "disqualified"
        )
        gates.append(gate(error))
        rows = [
            dict(
                unit_id=f"fit-{i:02}",
                cluster_id="original-development",
                arm=c["objective"],
                seed=c["seed"],
                metric=None,
                denominator=1,
                status="censored",
                missing_reason=str(error),
            )
            for i, c in enumerate(q.inventory())
        ]
        value = dict(
            rows=rows,
            intended_count=45,
            completed_count=0,
            failed_count=0,
            censored_count=45,
            excluded_count=0,
            independent_count=0,
            diagnosis_rows_ready_score=0,
            objective_candidate_ready_score=0,
            objective_selection={},
            selected_configuration_hash=None,
        )
    checked = validate(raw)
    if not checked["passed"]:
        verdict = "disqualified"
        value.update(diagnosis_rows_ready_score=0, objective_candidate_ready_score=0)
    end = time.monotonic_ns()
    spans = [
        dict(
            phase="authentication_original_fits_and_validation",
            started_monotonic_ns=start,
            ended_monotonic_ns=end,
            duration_s=(end - start) / 1e9,
        )
    ]
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="deterministic_verifier",
        inference_substrate_details=dict(
            backend="JAX CPU", architecture="original Gibbs9x4 and linear"
        ),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        phase_spans=spans,
        small_ebm_training=dict(performed=evidence is not None),
    )
    q.atomic_json(raw / "current_work_receipt.json", receipt)
    value.update(
        experiment_id=8828,
        task_id="exp8828-signal-replay-qualification",
        milestone="2026.11.756",
        run_date=args.date,
        honest_verdict="complete_" + verdict + "_signal_replay_qualification",
        verdict_class=verdict,
        inference_substrate="deterministic_verifier",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=20001 if evidence else 0,
        verifier_is_oracle=True,
        exposure_scope="Exposed synthetic development and retrospective V754 truth; no future stream.",
        independent_generalization_score=0,
        duration_s=(end - start) / 1e9,
        random_seed=8815,
        reproducibility_checksum=q.canonical_hash(
            {
                k: evidence[k]
                for k in (
                    "diagnostic_rows",
                    "development_rows",
                    "split_manifest",
                    "fits",
                    "controls",
                    "source_artifact_hashes",
                )
            }
            if evidence
            else None
        ),
        source_artifact_hashes=evidence["source_artifact_hashes"] if evidence else [],
        code_config_hashes=preconditions.get(
            "code", [q.d.reference(ROOT / p) for p in q.OWNED + q.TESTS]
        ),
        raw_shard_hashes=([q.d.reference(raw / "evidence.json")] if evidence else []),
        current_work_receipt=q.d.reference(raw / "current_work_receipt.json"),
        phase_spans=spans,
        preconditions_checked=preconditions,
        gate_check_summary=gates,
        required_checks_passed=checked["passed"],
        validation_receipts=checked["receipts"],
        owned_statement_coverage=checked["coverage"],
        original_inventory=q.inventory(),
        partition_rows=evidence["partition_rows"] if evidence else [],
        replay_projection=evidence["replay_projection"] if evidence else {},
        linear_comparator_rows=value["objective_selection"].get("paired_rows", []),
        repository_health_debt=[
            r for r in checked["receipts"] if r["scope"] == "repository_health"
        ],
        predecessor_disposition="Exp8815 remains disqualified. Development selection is not future learning.",
        methodology_note="One complete unchanged audit; immutable primitive snapshot;45 exact original fits;400 updates;40/20 split;original rates;linear comparison retained.",
        acceptance_gates=dict(
            evidence_readiness=dict(
                passed=value["diagnosis_rows_ready_score"] == 1,
                principle="Complete original replay and owned validation are required.",
            ),
            scientific_benefit=dict(
                passed=value["objective_candidate_ready_score"] == 1,
                threshold=">=0.01 development Brier gain in >=4/5 seeds plus controls and disjointness",
                principle="Development selection licenses only the finite categorical candidate.",
            ),
        ),
    )
    value["field_principles"] = {
        k: "Exact invocation evidence preserves the original finite comparison; it establishes no future generalization."
        for k in value
    }
    candidate = raw / "candidate.json"
    q.atomic_json(candidate, value)
    terminal_start = time.monotonic_ns()
    final = terminal(candidate, raw)
    final["candidate_sha256"] = q.sha256_file(candidate)
    q.atomic_json(raw / "terminal_validation.json", final)
    if not final["passed"]:
        value.update(
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_terminal_check",
            required_checks_passed=False,
            diagnosis_rows_ready_score=0,
            objective_candidate_ready_score=0,
        )
        for item in value["acceptance_gates"].values():
            item["passed"] = False
    terminal_end = time.monotonic_ns()
    spans.append(
        dict(
            phase="terminal_scanners_and_original_replay",
            started_monotonic_ns=terminal_start,
            ended_monotonic_ns=terminal_end,
            duration_s=(terminal_end - terminal_start) / 1e9,
        )
    )
    receipt.update(
        ended_monotonic_ns=terminal_end, duration_s=(terminal_end - start) / 1e9, phase_spans=spans
    )
    q.d.require(not validate_current_work_receipt(receipt), "current_work_receipt")
    q.atomic_json(raw / "current_work_receipt.json", receipt)
    value.update(
        duration_s=(terminal_end - start) / 1e9,
        current_work_receipt=q.d.reference(raw / "current_work_receipt.json"),
        terminal_validation=q.d.reference(raw / "terminal_validation.json"),
    )
    value["validation_receipts"].extend(final["receipts"])
    replay_files = list(raw.glob("replay/**/fit-*.json"))
    replay_controls = list(raw.glob("replay/**/controls.json"))
    value["numerical_training_calls"] += 400 * len(replay_files) + 2001 * len(replay_controls)
    value["numerical_training_scope"] = (
        "Completed scientific SGD updates in owned execution and cold replay; validation test work is separate."
    )
    value["field_principles"].update(
        {
            "terminal_validation": "Bind actual scanner and outside-checkout replay logs to their prospective candidate bytes.",
            "numerical_training_scope": "CPU numerical updates are separate from zero language-model invocations.",
        }
    )
    publish_primary(
        output,
        value,
        lambda path: dict(
            passed=q.canonical_hash(json.loads(path.read_bytes())) == q.canonical_hash(value),
            qualification=q.d.reference(raw / "terminal_validation.json"),
            candidate_sha256=q.sha256_file(path),
        ),
    )
    q.d.progress("published_8828", 1, 0)
    return 0
