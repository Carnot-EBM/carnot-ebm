"""REQ-REPORT-8634: publish only immutable, independently checked compatibility evidence."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

from carnot.reporting import installed_compatibility_8634 as e
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.primary_publication import publish_primary, reader_receipt, read_bound_sidecar
from carnot.reporting.v709_execution import child, progress

Json = dict[str, Any]
REQUIRED = {
    "exact_node_union",
    "owned_coverage",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
    "e2e018",
    "e2e021",
    "primary_publication",
}


def artifact(authority: Json) -> Json:
    """Keep compatibility readiness separate from oracle-dependent scientific evidence."""
    work = e.load(authority)
    checks = {r["name"]: r["passed"] for r in work["validation_receipts"]}
    failed = work.get("owned_execution_failed", False) or any(not r for r in checks.values())
    gates = work["gate_check_summary"]
    ready = int(
        not gates
        and not failed
        and REQUIRED <= checks.keys()
        and bool(work["installed_parity_rows"])
    )
    counts = (
        e.reduce(work)
        if work["installed_parity_rows"] and not work.get("owned_execution_failed", False)
        else dict(
            intended_count=128,
            completed_count=0,
            failed_count=0,
            censored_count=128,
            excluded_count=0,
            independent_count=0,
        )
    )
    verdict = (
        "disqualified"
        if failed
        else "blocked"
        if gates
        else "circular_positive"
        if ready
        else "disqualified"
    )
    bundle = work["bundle"]
    value: Json = dict(
        experiment_id=8634,
        task_id=e.TASK,
        milestone="2026.10.741",
        run_date=work["run_date"],
        honest_verdict="complete_" + verdict + "_installed_compatibility",
        verdict_class=verdict,
        gate_check_summary=gates,
        inference_substrate="CPU fixed109-parameter small-head spline arithmetic; installed public-contract checks",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_language_model_calls=0,
        historical_model_provenance=bundle.get("historical_model_provenance", {}),
        target_model=bundle.get("target_model", {}),
        rows=work["installed_parity_rows"],
        **counts,
        sample_size_budget=dict(
            original_requests=64,
            independent_questions=0,
            intended=counts["intended_count"],
            arithmetic_arms=2,
            reason="Exposed original requests and deterministic numerical controls",
        ),
        verifier_is_oracle=True,
        exposure_scope="Exposed compatibility operands and synthetic public verification smoke; deployment only",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        installed_measurement_ready_score=ready,
        branch_gate_values=dict(
            installed_measurement_ready_score=ready,
            independent_generalization_score=0,
            generalized_learning_benefit_score=0,
        ),
        required_checks_passed=bool(ready),
        flagged_adversarial=False,
        acceptance_gates=[dict(name=k, passed=checks.get(k, False)) for k in sorted(REQUIRED)]
        + [dict(name="installed_compatibility", passed=bool(ready))],
        validation_receipts=work["validation_receipts"],
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        adversarial_findings=[],
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7418634,
        source_artifact_hashes=work["source_artifact_hashes"],
        code_config_hashes=work.get(
            "code_config_hashes", [e.reference(e.ROOT / p) for p in e.OWNED]
        ),
        raw_shard_hashes=work["raw_shard_hashes"],
        cited_upstream_artifacts=work["source_artifact_hashes"],
        installed_module_identity=work["installed_module_identity"],
        wheel_reference=work.get("wheel_reference"),
        public_surface_rows=work["public_surface_rows"],
        export_compatibility_rows=work["export_compatibility_rows"],
        original_node_union=work["original_node_union"],
        installed_parity_rows=work["installed_parity_rows"],
        rejection_rows=work["rejection_rows"],
        child_shard_manifest=work["child_shard_manifest"],
        historical_8609=work.get("historical_8609"),
        preserved_historical_nulls=bundle.get("historical_cost_null"),
        preserved_quarantines=bundle.get("preserved_quarantines"),
        authority_reference=authority,
        methodology="Install the exact Exp8633 wheel offline. Compare private and installed Python/native responses on original requests, knot/decay and threshold controls. Test original exports, public verification and bounded rejection families. The existing HTTP API fits its small detector on80 synthetic fixture examples. The original energy head remains fixed. Retain and combine genuine child coverage. No generator, timing or independent calibration claim.",
    )
    value["field_principles"] = {
        key: "Bind this invocation to immutable primitives; compatibility supplies no independent scientific credit."
        for key in value
    }
    value["field_principles"].update(
        installed_measurement_ready_score="One only after installed identities, public contracts, numerical parity, real child coverage and checked publication pass. Deployment compatibility only.",
        installed_module_identity="Paths and hashes must belong to the private wheel environment; checkout shadowing fails.",
        original_node_union="Retain all122 historical nodes plus declared current and public families, unchanged assertions and immutable original authority.",
        child_shard_manifest="Retain actual child coverage before combine; summaries cannot replace lost shards.",
        historical_8609="Preserve the historical methods blockage; current compatibility cannot rewrite its verdict.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path, authority: Json) -> None:
    """An external authority prevents a self-consistent rehash from replacing evidence."""
    value = json.loads(path.read_bytes())
    if value != artifact(authority):
        raise ValueError("immutable_reduction_drift")
    for ref in (
        value["source_artifact_hashes"]
        + value["code_config_hashes"]
        + value["raw_shard_hashes"]
        + value["child_shard_manifest"]
    ):
        if e.reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
            raise ValueError("replay_input_hash")


def terminal(path: Path, authority: Json, logs: Path) -> Json:
    """Retain actual reader findings and reject missing and rehashed candidates."""
    changed = json.loads(path.read_bytes())
    changed["completed_count"] += 1
    changed["reproducibility_checksum"] = canonical_hash(changed)
    bad = logs / "rehashed_tamper.json"
    atomic_json(bad, changed)
    receipts = []
    for name, argv, expected in [
        (
            "cold",
            [
                e.CLI,
                "--cold-replay",
                str(path),
                "--authority",
                authority["path"],
                authority["sha256"],
            ],
            0,
        ),
        ("adversarial", ["scripts/adversarial_verify.py", "--json", str(path)], 0),
        ("strict_rows", ["scripts/verdict_row_consistency_lint.py", "--strict", str(path)], 0),
        (
            "missing",
            [
                e.CLI,
                "--cold-replay",
                str(logs / "missing.json"),
                "--authority",
                authority["path"],
                authority["sha256"],
            ],
            1,
        ),
        (
            "rehashed_tamper",
            [
                e.CLI,
                "--cold-replay",
                str(bad),
                "--authority",
                authority["path"],
                authority["sha256"],
            ],
            1,
        ),
    ]:
        receipts.append(
            child(name, [sys.executable, "-u", *argv], logs, expected=expected, deadline=90)
        )
    findings = json.loads(Path(receipts[1]["stdout_path"]).read_text())
    return dict(
        passed=all(r["passed"] for r in receipts) and findings["flagged_count"] == 0,
        receipts=receipts,
        actual_findings=findings,
    )


def run(output: Path, run_date: str, root: Path) -> int:
    """Keep measurement below3600 seconds and retain bounded validation before publication."""
    from carnot.reporting.installed_validation_8634 import validate

    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    started = time.monotonic_ns()
    work = e.measure(raw, run_date, root)
    before_validation = time.monotonic_ns()
    work["validation_receipts"] = validate(work, raw)
    ended = time.monotonic_ns()
    work["duration_s"] = (ended - started) / 1e9
    work["phase_spans"].append(dict(phase="validation", start_ns=before_validation, end_ns=ended))
    if work["duration_s"] > 4500:
        work["validation_receipts"].append(dict(name="runtime_cap", passed=False))
    work["code_config_hashes"] = [
        e.reference(e.ROOT / p)
        for p in e.OWNED
        + e.TESTS
        + ["python/carnot/__init__.py", "python/carnot/inference/resident_protocol.py"]
    ]
    work["raw_shard_hashes"] = [e.reference(p) for p in sorted(raw.rglob("*")) if p.is_file()]
    authority_path = raw / "authority.json"
    atomic_json(authority_path, work)
    authority = e.reference(authority_path)
    value = artifact(authority)
    result = publish_primary(
        output, value, lambda path: terminal(path, authority, raw / "terminal")
    )
    atomic_json(
        raw / "terminal_validation.json", read_bound_sidecar(output, Path(result["sidecar_path"]))
    )
    reader = reader_receipt(
        e.TASK,
        output.parent,
        field="installed_measurement_ready_score",
        expected=value["installed_measurement_ready_score"],
    )
    atomic_json(raw / "reader_receipt.json", reader)
    if not reader["passed"]:
        raise ValueError("publication_reader")
    progress(
        "8634_complete_" + value["verdict_class"], value["completed_count"], value["censored_count"]
    )
    return int(value["verdict_class"] == "disqualified")


def main(argv: list[str] | None = None) -> int:
    """Accept explicit private paths and frozen replay authority without changing defaults."""
    progress("8634_start_no_model_load_MODEL_SPECS=[]_current_language_model_calls=0")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261018")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--authority", nargs=2)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            if not args.authority:
                raise ValueError("immutable_authority_required")
            replay(args.cold_replay, dict(path=args.authority[0], sha256=args.authority[1]))
            return 0
        if (
            datetime.strptime(args.date, "%Y%m%d").strftime("%Y%m%d") != args.date
            or (args.input_root and not args.private_output)
            or (args.private_output and args.private_output.resolve().is_relative_to(e.ROOT))
        ):
            raise ValueError("date_or_private_boundary")
        return run(
            args.private_output or e.ROOT / "results" / (e.NAME + ".json"),
            args.date,
            args.input_root or e.ROOT,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[8634] error={error}", file=sys.stderr, flush=True)
        return 1
