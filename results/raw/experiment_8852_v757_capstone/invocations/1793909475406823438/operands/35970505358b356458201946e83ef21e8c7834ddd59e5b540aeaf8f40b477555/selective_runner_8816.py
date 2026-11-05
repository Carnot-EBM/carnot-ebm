"""REQ-VERIFY-8816: publish only hash-bound private protocol evidence with actual checks."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from typing import Any

from carnot.reporting import selective_evidence_8816 as m
from carnot.reporting import selective_protocol_8816 as p
from carnot.reporting import evidence_evaluation_runner_8779 as shared
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    build_current_work_receipt,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]


def stream(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
    """Forward child streams while its existing fixed deadline continues to run."""
    offsets = {"stdout": 0, "stderr": 0}
    started = last = time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(child, name, argv, logs, **kwargs)
        while True:
            for suffix in offsets:
                path = logs / (name + "." + suffix)
                if path.exists():
                    with path.open() as reader:
                        reader.seek(offsets[suffix])
                        text = reader.read()
                        offsets[suffix] = reader.tell()
                    if text:
                        print(text, end="", flush=True)
            if pending.done():
                return dict(pending.result())
            if time.monotonic() - last >= kwargs.get("heartbeat", 30):
                m.progress(f"child_{name}_elapsed_{time.monotonic() - started:.0f}s", 0, 1)
                last = time.monotonic()
            time.sleep(0.05)


def unqualified() -> Json:
    """A private fixture cannot grant itself production validation."""
    return dict(
        receipts=[dict(name="private_unqualified", passed=False)],
        union=dict(passed=False),
        coverage=dict(complete=False, percent=0, files=[]),
        raw_references=[],
    )


def validate(raw: Path, private: Path) -> Json:
    """Reuse exact-node collection, actual unit/child coverage and scoped lint checks."""
    validator = namespace(shared, "selective8816_validation")
    validator.e = SimpleNamespace(OWNED=m.OWNED, TESTS=m.TESTS, ROOT=m.ROOT)
    validator.original = SimpleNamespace(
        CONSUMERS=[
            "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
            "tests/python/test_restricted_decision_audit_8210.py",
        ]
    )
    validator.child = stream
    return dict(validator.validate(raw, private))


def prepare(raw: Path) -> Json:
    """Seal inputs and private labels without running any compared policy or optional branch."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw.chmod(0o700)
    diagnostics, sources = [], []
    for path, pin in m.HISTORY.items():
        source = m.ROOT / path
        status = (
            "missing_file"
            if not source.is_file()
            else "admitted"
            if sha256_file(source) == pin
            else "rejected_producer"
        )
        row = dict(
            check="immutable_V754_authority",
            upstream=path,
            artifact_field="sha256",
            expected=pin,
            observed=sha256_file(source) if source.is_file() else None,
            path=str(source),
            sha256=sha256_file(source) if source.is_file() else None,
            status=status,
        )
        sources.append(row)
        if status != "admitted":
            diagnostics.append(row)
    m.progress("before_world_seal", 0, 372)
    cohort = m.freeze()
    atomic_json(raw / "cohort.json", cohort)
    (raw / "cohort.json").chmod(0o600)
    atomic_json(raw / "inputs.json", dict(rows=cohort["inputs"]))
    atomic_json(raw / "private_labels.json", dict(rows=cohort["labels"]))
    (raw / "private_labels.json").chmod(0o600)
    protocol = p.protocol()
    atomic_json(raw / "protocol.json", protocol)
    controls = m.controls(cohort)
    end = time.monotonic_ns()
    receipt = build_current_work_receipt(
        events=[],
        run_id=str(start),
        owner_pid=os.getpid(),
        inference_substrate="deterministic_local",
        inference_substrate_details=dict(scope="constructed protocol seal; no policy evaluation"),
        inference_substrate_class="cpu_exact_solver_or_simulator",
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        phase_spans=[dict(phase="seal", start_s=0.0, end_s=(end - start) / 1e9)],
    )
    m.progress("after_world_seal", len(cohort["worlds"]), 372 - len(cohort["worlds"]))
    return dict(
        cohort_status=cohort["status"],
        manifest={k: v for k, v in cohort.items() if k not in {"inputs", "labels"}},
        cohort_reference=reference(raw / "cohort.json"),
        controls=controls,
        source_artifact_hashes=sources,
        gate_check_summary=diagnostics,
        preconditions_checked=dict(
            passed=not diagnostics,
            private_output_writable=True,
            prospective_only=True,
            optional_branches_required=False,
        ),
        raw_shard_hashes=[
            reference(raw / name)
            for name in ["cohort.json", "inputs.json", "private_labels.json", "protocol.json"]
        ],
        code_config_hashes=[reference(m.ROOT / path) for path in m.OWNED + m.TESTS],
        protocol=protocol,
        current_work_receipt=receipt,
    )


def build(work: Json, validation: Json) -> Json:
    """Readiness qualifies frozen evidence only; exact reference success is circular."""
    qualified = bool(
        validation["union"]["passed"]
        and validation["coverage"]["complete"]
        and all(x["passed"] for x in validation["receipts"])
    )
    external = work["preconditions_checked"]["passed"] and work["cohort_status"] == "sealed"
    controls = all(x["passed"] for x in work["controls"])
    ready = int(qualified and external and controls)
    verdict = (
        "blocked"
        if not external
        else "disqualified"
        if not qualified or not controls
        else "circular_positive"
    )
    manifest, protocol, receipt = work["manifest"], work["protocol"], work["current_work_receipt"]
    rows = [
        dict(
            unit_id=w["world_id"],
            cluster_id=w["world_id"],
            arm="protocol_seal",
            seed=m.SEED,
            metric="independent_topology_sealed",
            denominator=372,
            status="completed",
            missing_reason=None,
        )
        for w in manifest["worlds"]
    ]
    value = dict(
        experiment_id=8816,
        task_id=m.TASK,
        milestone="2026.11.755",
        run_date="20261104",
        honest_verdict=f"complete_{verdict}_selective_evidence_protocol",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=0,
        verifier_is_oracle=True,
        duration_s=receipt["duration_s"],
        random_seed=m.SEED,
        source_artifact_hashes=work["source_artifact_hashes"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=work["raw_shard_hashes"],
        phase_spans=receipt["phase_spans"],
        preconditions_checked=work["preconditions_checked"],
        rows=rows,
        intended_count=372,
        completed_count=len(rows),
        failed_count=0,
        censored_count=372 - len(rows),
        excluded_count=0,
        independent_count=len(rows),
        gate_check_summary=work["gate_check_summary"],
        acceptance_gates=dict(
            protocol_qualified=bool(ready),
            scientific_benefit_measured=False,
            V754_gates_unchanged=True,
            evidence_protocol=bool(ready),
            consensus_protocol=bool(ready),
        ),
        required_checks_passed=qualified,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        exposure_scope="constructed six-vertex signed graphs only; private evaluation labels unopened by compared policies",
        independent_generalization_score=0,
        cited_upstream_artifacts=work["source_artifact_hashes"],
        evidence_protocol_ready_score=ready,
        consensus_protocol_ready_score=ready,
        canary_protocol_ready_score=ready,
        split_manifest=manifest,
        split_manifest_hash=canonical_hash(manifest),
        structure_disjointness_rows=manifest["structure_disjointness_rows"],
        protocol_hashes={
            name: canonical_hash(protocol["estimands"][name]) for name in ["exp8818", "exp8819"]
        },
        threshold_menu=protocol["threshold_menu"],
        estimands=protocol["estimands"],
        multiplicity_families=protocol["multiplicity_families"],
        label_custody=protocol["label_custody"],
        calibration_selection_rule=protocol["calibration_selection_rule"],
        protocol=protocol,
        mutation_controls=work["controls"],
        current_work_receipt=receipt,
        calibration_selection_status="prospectively frozen; Exp8818/8819 must select before evaluation; no study predictions in this slot",
        exact_reference_cost_scope="two exact label evaluators; no model calls; never charged as a policy comparator",
        historical_diagnosis=dict(
            exp8805=dict(
                false_accepts=0,
                denominator=1440,
                abstention_count=2880,
                scheduled_denominator=2880,
                decisive_coverage=0,
                reason="zero false-accept headroom against abstaining local baseline",
            ),
            exp8806=dict(coverage_delta=-0.11358, reason="lower Brier with lost decisive coverage"),
            old_conjunctive_gates="failed and unchanged",
        ),
        methodology_note="Exact-oracle construction and mutation controls qualify infrastructure only. Readiness is valid evidence, not measured benefit. No natural-text or hidden-game generalization.",
        global_test_spec_debt="global suite outcome is recorded separately from scoped owned qualification",
    )
    value["field_principles"] = {
        k: "Bind exact prospective evidence; preserve denominators and V754 nulls; selection precedes untouched evaluation."
        for k in value
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Recompute frozen semantics as well as hashes so rehashed aggregate forgeries fail."""
    try:
        value = json.loads(path.read_bytes())
        work = load(value["work_reference"])
        validation = load(value["validation_reference"])
        refs = work["raw_shard_hashes"] + work["code_config_hashes"] + validation["raw_references"]
        if any(sha256_file(Path(x["path"])) != x["sha256"] for x in refs):
            return False
        if load(work["cohort_reference"]) != m.freeze() or work["protocol"] != p.protocol():
            return False
        if work["controls"] != m.controls(load(work["cohort_reference"])):
            return False
        expected = build(work, validation)
        expected.update(
            work_reference=value["work_reference"],
            validation_reference=value["validation_reference"],
        )
        expected["field_principles"].update(
            work_reference="Immutable work custody",
            validation_reference="Actual owned check custody",
        )
        expected["reproducibility_checksum"] = canonical_hash(
            {k: v for k, v in expected.items() if k != "reproducibility_checksum"}
        )
        return bool(value == expected)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main(argv: list[str] | None = None) -> int:
    """Private runs remain unqualified; production publishes only checked candidate bytes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261104"], default="20261104")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--private-output", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = replay(args.cold_replay)
        m.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    output = args.output.absolute()
    if args.private_output and output.resolve().is_relative_to(m.ROOT):
        parser.error("private outputs must remain outside checkout")
    raw = output.parent / "raw" / m.NAME / "invocations" / str(time.time_ns())
    m.progress("start_no_model_load", 0, 372)
    work = prepare(raw)
    atomic_json(raw / "work.json", work)
    with TemporaryDirectory(prefix="exp8816-") as temp:
        m.progress("before_owned_validation", 0, 1)
        validation = (
            unqualified() if args.private_output else validate(raw / "validation", Path(temp))
        )
        m.progress("after_owned_validation", 1, 0)
    atomic_json(raw / "validation.json", validation)
    value = build(work, validation)
    value.update(
        work_reference=reference(raw / "work.json"),
        validation_reference=reference(raw / "validation.json"),
    )
    value["field_principles"].update(
        work_reference="Immutable work custody", validation_reference="Actual owned check custody"
    )
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    m.progress("before_terminal_publication", 0, 1)
    publish_primary(output, value, lambda candidate: terminal(candidate, raw))
    m.progress("complete", 1, 0)
    return 0


def terminal(candidate: Path, raw: Path) -> Json:
    """Both existing independent artifact checkers run on the exact candidate before publication."""
    receipts = [
        stream(
            name,
            [str(m.ROOT / ".venv/bin/python"), "-u", script, str(candidate), *extra],
            raw / "terminal_checks",
            deadline=120,
        )
        for name, script, extra in [
            ("adversarial", "scripts/adversarial_verify.py", []),
            ("rows_strict", "scripts/verdict_row_consistency_lint.py", ["--strict"]),
        ]
    ]
    return dict(passed=replay(candidate) and all(x["passed"] for x in receipts), receipts=receipts)
