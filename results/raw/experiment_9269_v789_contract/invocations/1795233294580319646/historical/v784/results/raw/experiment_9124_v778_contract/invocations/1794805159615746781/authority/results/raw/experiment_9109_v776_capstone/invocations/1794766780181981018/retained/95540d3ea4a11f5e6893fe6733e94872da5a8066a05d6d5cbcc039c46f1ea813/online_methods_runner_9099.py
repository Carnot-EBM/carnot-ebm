"""REQ-REPORT-9099-V776: publish frozen human mechanics after measured owned checks."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.memory import human_online_9099 as learner
from carnot.reporting import online_methods_9099 as d
from carnot.reporting import mdh_human_release_runner_9056 as prior
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    validate_current_work_receipt,
)
from carnot.reporting.natural_label_runner_8975 import read_ref
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]
DIRECT = [
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_current_work_receipt.py",
    "tests/python/test_online_spline_methods_8988.py",
    "tests/python/test_spline_basis_methods_8986.py",
    "tests/python/test_human_authority_9097.py",
]
REUSED = [
    *prior.REUSED,
    "python/carnot/autoresearch/spline_basis_8986.py",
    "python/carnot/verify/block_feedback_state_8433.py",
    "python/carnot/reporting/human_methods_9084.py",
    d.PROTOCOL,
]
CHECKS = [*prior.CHECKS, "e2e020"]


def scope_manifest(commands: list[Json]) -> Json:
    """Reuse strict receipt checks while freezing this task's owned files and direct consumers."""
    with patch.multiple(prior, e=d, DIRECT=DIRECT, REUSED=REUSED, CHECKS=CHECKS):
        return prior.scope_manifest(commands)


def receipts_pass(scope: Json, validation: Json) -> bool:
    """Only real matching commands, pinned logs and complete SQLite coverage qualify readiness."""
    with patch.multiple(prior, e=d, CHECKS=CHECKS):
        return prior.receipts_pass(scope, validation)


def measure(root: Path, raw: Path, date: str, scope: Json) -> Json:
    """Freeze design and rosters before fixtures, with no model load or human benefit measurement."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    d.progress("authenticate_before")
    evidence = d.authenticate(root)
    storage = d.gate(raw, "mode", 0o700, raw.stat().st_mode & 0o777, "private_storage")
    evidence["gate_check_summary"].insert(0, storage)
    if not storage["passed"]:
        evidence["blocker"] = "private_storage"
    atomic_json(raw / "protocol.json", d.protocol())
    atomic_json(raw / "rosters.json", evidence["roster"])
    retained = []
    for reference in evidence["source_references"]:
        if reference["sha256"] is not None:
            target = raw / "inputs" / reference["sha256"].split(":")[1]
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(reference["path"], target)
            target.chmod(0o400)
            retained.append(d.ref(target))
    for name in ("protocol", "rosters"):
        (raw / (name + ".json")).chmod(0o400)
    d.progress(
        "protocol_rosters_sealed", len(evidence["roster"].get("initial_group_roster", [])), 0
    )
    result = {} if evidence["blocker"] else d.evaluate(raw / "fixtures")
    fixture_references = [
        d.ref(p)
        for p in sorted((raw / "fixtures").rglob("*"))
        if p.is_file() and p.name != ".state.tmp"
    ]
    changes = (
        {}
        if evidence["blocker"]
        else d.donor_changes(
            json.loads((root / d.AUTHORITY).read_bytes())["rows"], evidence["roster"]
        )
    )
    end = time.monotonic_ns()
    current = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        inference_substrate_details=dict(
            work="CPU constructed human coefficient mechanics only", human_benefit_measured=False
        ),
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
    )
    atomic_json(raw / "current_work.json", current)
    work = dict(
        root=str(root),
        raw=str(raw),
        run_date=date,
        evidence=evidence,
        result=result,
        donor_changes=changes,
        retained_references=retained,
        fixture_references=fixture_references,
        protocol_reference=d.ref(raw / "protocol.json"),
        roster_reference=d.ref(raw / "rosters.json"),
        current_work_reference=d.ref(raw / "current_work.json"),
        validation_scope_manifest=scope,
        duration_s=(end - start) / 1e9,
    )
    atomic_json(raw / "work.json", work)
    (raw / "work.json").chmod(0o400)
    return d.ref(raw / "work.json")


def build(reference: Json, validation: Json) -> Json:
    """Administrative mechanics readiness is separate from all six unmeasured scientific bounds."""
    work = read_ref(reference)
    evidence, result = work["evidence"], work["result"]
    passed = receipts_pass(work["validation_scope_manifest"], validation)
    mechanics = (
        bool(result)
        and result["exact_resume"]
        and all(r["passed"] for r in result["causal_controls"])
    )
    blocked = evidence["blocker"] is not None
    disqualified = not passed or bool(result) and not mechanics
    cls = "disqualified" if disqualified else "blocked" if blocked else "null"
    ready = int(passed and mechanics and not blocked)
    rows = result.get("rows", [])
    contract = read_ref(work["protocol_reference"])
    roster = read_ref(work["roster_reference"])
    value: Json = dict(
        result,
        experiment_id=9099,
        task_id=d.TASK,
        milestone="2026.11.776",
        run_date=work["run_date"],
        honest_verdict="complete_disqualified_owned_validation"
        if disqualified
        else "complete_blocked_" + evidence["blocker"]
        if blocked
        else "complete_null_online_methods_qualified",
        verdict_class=cls,
        required_checks_passed=passed,
        flagged_adversarial=disqualified,
        online_methods_ready_score=ready,
        learning_signal_score=0,
        scientific_bounds_status="frozen_unmeasured",
        verifier_is_oracle=False,
        independent_generalization_score=0,
        claim_scope="Frozen causal human coefficient methods and exact CPU-fixture recovery only; no measured human learning benefit or current ARC solve.",
        exposure_scope="LPHB exposed development; original300 qids/900 siblings; CPU fixtures establish mechanics only.",
        MODEL_SPECS=[],
        model_specs=[],
        inference_substrate_class="no_model_load",
        inference_substrate="verifier_ensemble_against_cached_candidates",
        current_llm_invocations=0,
        model_invocation_counts=dict(model_loads=0, generations=0, embedding_calls=0, total=0),
        execution_venue="host",
        numerical_training_scope="small convex coefficient fits on CPU fixtures only; generator weights unchanged",
        rows=rows,
        intended_count=150,
        attempted_count=len(rows),
        completed_count=len(rows),
        failed_count=0,
        censored_count=150 - len(rows),
        excluded_count=0,
        independent_count=0,
        effective_independent_groups=0,
        sample_size_budget=dict(
            online_qids=50,
            streams=10,
            retention_qids=100,
            future_scored_qids=40,
            fixture_independent_groups=0,
        ),
        online_protocol_reference=work["protocol_reference"],
        **{
            k: roster.get(k, [])
            for k in ("initial_group_roster", "stream_roster", "guard_roster", "retention_roster")
        },
        donor_map=roster.get("donor_map", {}),
        sibling_roster=roster.get("sibling_roster", {}),
        original_group_ids=roster.get("original_group_ids", {}),
        donor_changed_label_fraction=work["donor_changes"].get("fraction"),
        donor_label_change_rows=work["donor_changes"].get("rows", []),
        donor_changed_label_pair_count=work["donor_changes"].get("numerator"),
        donor_observed_label_pair_count=work["donor_changes"].get("denominator"),
        donor_change_status="descriptive original-label diagnostic after map freeze; no future truth reaches a learner",
        arm_roster=contract["arms"],
        update_equations=contract["update_equations"],
        six_directional_bounds=contract["benefit_bounds"],
        rollback_checks=result.get("rollback_checks", []),
        hard_exit_receipts=result.get("hard_exit_receipts", []),
        exact_resume=result.get("exact_resume", False),
        gate_check_summary=evidence["gate_check_summary"],
        source_references=evidence["source_references"],
        primitive_reference=reference,
        current_work_receipts=[work["current_work_reference"]],
        raw_shard_hashes=[
            reference,
            work["protocol_reference"],
            work["roster_reference"],
            work["current_work_reference"],
            *work["retained_references"],
            *work["fixture_references"],
        ],
        code_config_hashes=work["validation_scope_manifest"].get("code_test_hashes", []),
        validation_scope_manifest=work["validation_scope_manifest"],
        validation_receipts=validation.get("checks", []),
        owned_statement_coverage=validation.get("owned_statement_coverage", {}),
        repository_health=validation.get("repository_health", {}),
        duration_s=work["duration_s"],
        random_seed=776099,
        acceptance_gates=dict(
            authenticated_inputs=not blocked,
            owned_checks=passed,
            causal_mechanics=mechanics,
            exact_recovery=result.get("exact_resume", False),
        ),
        methodology_note="Separate exploratory online estimand; transforms refit on100 initialization qids only. Fixed five arms, delayed whole-sibling release, CAL-only guards and evaluator-only retention. No human representation is fabricated when embeddings are unavailable. CPU fixture wins establish no scientific benefit.",
        reconciliation="Conductor owns doc/status/traceability reconciliation; no public-default change or external publication.",
    )
    value["field_principles"] = principles(value)
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def principles(value: Json) -> Json:
    """Explain distinct limits so a methods receipt cannot silently become a science claim."""
    explained = {
        "honest_verdict": "A terminal outcome prevents repeated work on unchanged missing inputs.",
        "verdict_class": "The closed class separates an observed scientific win from administrative mechanics.",
        "gate_check_summary": "Expected and observed typed operands distinguish missing_file, missing_field and observed zero.",
        "rows": "Every retained sibling/arm observation exposes the numerator and denominator used in aggregates.",
        "sample_size_budget": "The50 online qids form10 reset streams; repeated100-qid retention never creates1000 independent units.",
        "effective_independent_groups": "Constructed CPU fixtures contribute zero independent human evidence.",
        "online_methods_ready_score": "Integer1 requires authentic inputs, causal controls, exact recovery and every owned check, not a favorable scientific result.",
        "learning_signal_score": "Unmeasured future human benefit remains zero regardless of fixture improvement.",
        "online_protocol_reference": "Exact frozen bytes prevent outcome-dependent changes to cohorts, arms or thresholds.",
        "initial_group_roster": "Only100 original TRAIN qids may determine preprocessing and initial coefficients.",
        "stream_roster": "Fifty disjoint held-back TRAIN qids define a new exploratory estimand without rewriting old evidence.",
        "guard_roster": "Original CAL50 is an operational safety set, separate from research retention.",
        "retention_roster": "Original TEST100 labels remain outside all updates and promotion decisions.",
        "donor_map": "Seeded initialization-only donors preserve sibling positions without future-target access.",
        "donor_changed_label_fraction": "The evaluator counts actual changed original sibling labels after freezing the initialization-only map; this diagnostic never enters learner updates.",
        "donor_changed_label_pair_count": "Exact integer numerator counts original sibling-label pairs with different observed A1 labels.",
        "donor_observed_label_pair_count": "Exact integer denominator counts observed original sibling-label pairs; these are distinct from constructed fixture prediction rows.",
        "arm_roster": "Frozen and restricted-update controls isolate coefficient adaptation from bias and architecture.",
        "update_equations": "One-vector proper-loss gradients, clipping and AdaGrad are mathematical operands, not metadata changes.",
        "rollback_checks": "Rejecting a proposal restores both parameters and optimizer memory while consuming feedback once.",
        "hard_exit_receipts": "Observed child exits test process failure rather than Python exception handling.",
        "exact_resume": "Complete-state equality checks predictions, coefficients, accumulators, pending work and acknowledgments.",
        "six_directional_bounds": "Fixed directional multiplicity bounds and small-sample caution precede all real-data outcomes.",
        "verifier_is_oracle": "Human factuality targets are distinct from cheap exact-oracle verification.",
        "claim_scope": "Methods qualification alone proves neither human benefit nor a current ARC solve.",
        "exposure_scope": "LPHB is exposed development; retained labels cannot establish independent generalization.",
        "independent_generalization_score": "Exposure and CPU fixtures force independent generalization to zero.",
        "MODEL_SPECS": "No generator is authorized or loaded in this coefficient-methods task.",
        "inference_substrate_class": "no_model_load distinguishes current CPU mechanics from historical model provenance.",
        "current_llm_invocations": "Current model calls remain zero even when upstream artifacts cite historical generators.",
        "model_invocation_counts": "Loads, generations and embeddings are counted separately from small coefficient fitting.",
        "duration_s": "Monotonic elapsed time describes actual work without duration padding.",
        "random_seed": "A fixed seed makes donor assignment and constructed operands repeatable.",
        "reproducibility_checksum": "The full reduced artifact is bound; primitive replay still rejects rehashed false summaries.",
        "code_config_hashes": "Executed new code, preserved consumers and protocol bytes bind validation to this implementation.",
        "source_references": "Exact upstream hashes authenticate original authority without repinning historical producers.",
        "current_work_receipts": "An owned empty model-event ledger cannot absorb historical model activity.",
        "validation_receipts": "Normal exits, frozen commands, live log hashes and SQLite execution evidence establish actual checks.",
        "required_checks_passed": "Failed affected validation disqualifies methods regardless of external evidence status.",
        "acceptance_gates": "Authentication, causality, recovery and owned validation remain individually inspectable.",
        "repository_health": "Broader historical failures remain separate from newly owned checks.",
    }
    return {
        k: explained.get(
            k,
            "Retain this contextual operand without expanding the declared exposed-development methods claim.",
        )
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }


def replay(path: Path) -> Json:
    """Recompute trajectories and the entire reduction so rehashed summaries cannot acquire readiness."""
    value = json.loads(path.read_bytes())
    work = read_ref(value["primitive_reference"])
    for reference in (
        value["source_references"] + value["raw_shard_hashes"] + value["code_config_hashes"]
    ):
        if d.ref(Path(reference["path"])) != reference:
            raise ValueError("operand_hash")
    if read_ref(work["protocol_reference"]) != d.protocol():
        raise ValueError("protocol_drift")
    fresh = d.authenticate(Path(work["root"]))
    fresh["gate_check_summary"].insert(0, work["evidence"]["gate_check_summary"][0])
    if work["evidence"]["blocker"] == "private_storage":
        fresh["blocker"] = "private_storage"
    if fresh != work["evidence"]:
        raise ValueError("source_reduction_drift")
    if read_ref(work["roster_reference"]) != fresh["roster"]:
        raise ValueError("roster_drift")
    expected_changes = (
        {}
        if fresh["blocker"]
        else d.donor_changes(
            json.loads((Path(work["root"]) / d.AUTHORITY).read_bytes())["rows"], fresh["roster"]
        )
    )
    if expected_changes != work["donor_changes"]:
        raise ValueError("donor_change_drift")
    if work["result"] and d.reduce(Path(work["raw"]) / "fixtures") != work["result"]:
        raise ValueError("fixture_reduction_drift")
    if validate_current_work_receipt(read_ref(work["current_work_reference"]), []):
        raise ValueError("current_work_drift")
    validation = dict(
        checks=value["validation_receipts"],
        owned_statement_coverage=value["owned_statement_coverage"],
        repository_health=value["repository_health"],
    )
    if build(value["primitive_reference"], validation) != value:
        raise ValueError("reduction_drift")
    return dict(passed=True, replay_passed=True, primary_sha256=d.ref(path)["sha256"])


def validate(candidate: Path, logs: Path) -> Json:
    """Reuse unchanged scanners and a real cold replay without PYTHONPATH before publication."""
    with patch.object(prior, "e", d):
        return prior.validate(candidate, logs)


def main(argv: list[str] | None = None) -> int:
    """Keep failed checks private; publish only the completely validated candidate bytes."""
    d.progress("start_no_model_load")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261115")
    parser.add_argument("--root", type=Path, default=d.ROOT)
    parser.add_argument("--output", type=Path, default=d.ROOT / "results" / (d.NAME + ".json"))
    parser.add_argument(
        "--scope", type=Path, default=d.DEFAULT_INPUT / "validation_scope_manifest.json"
    )
    parser.add_argument("--receipts", type=Path, default=d.DEFAULT_INPUT / "validation.json")
    parser.add_argument("--work", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--worker", type=Path, nargs=3)
    parser.add_argument(
        "--crash",
        default="",
        choices=["", "before_prediction_fsync", "after_feedback_receipt", "during_atomic_replace"],
    )
    args = parser.parse_args(argv)
    try:
        if args.date != "20261115":
            raise ValueError("run_date")
        if args.worker:
            request, raw, labels = args.worker
            learner.drive(json.loads(request.read_bytes()), raw, labels, crash=args.crash)
            return 0
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        scope = json.loads(args.scope.read_bytes()) if args.scope.is_file() else scope_manifest([])
        validation = json.loads(args.receipts.read_bytes()) if args.receipts.is_file() else {}
        raw = Path(tempfile.mkdtemp(prefix="carnot-9099-work-"))
        reference = (
            json.loads(args.work.read_bytes())
            if args.work
            else measure(args.root, raw, args.date, scope)
        )
        value = build(reference, validation)
        atomic_json(raw / "candidate.json", value)
        if not value["required_checks_passed"]:
            d.progress("owned_validation_failed_primary_preserved")
            return 1
        d.progress("publication_before")
        print(
            json.dumps(
                publish_primary(args.output, value, lambda p: validate(p, raw / "terminal"))
            ),
            flush=True,
        )
        d.progress("complete", len(value["rows"]), 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        d.progress("refused:" + str(error))
        return 1
