"""REQ-REPORT-9138-V779: publish a checked administrative contract with no model work."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import logit_methods_runner_9126 as lifecycle
from carnot.reporting import v779_contract_evidence as e
from carnot.reporting.current_work_receipt import atomic_json as atomic_json
from carnot.reporting.current_work_receipt import (
    build_current_work_receipt,
    canonical_hash,
    validate_current_work_receipt,
)
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]


def checksum(value: Json) -> str:
    """A checksum cannot authenticate invented rows; replay also rebuilds the reduction."""
    return canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"})


def measure(root: Path, raw: Path, date: str) -> Json:
    """Freeze commands and resource limits before importing any historical outcomes."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    with patch.object(lifecycle, "d", e):
        scope = lifecycle.scope(raw)
    scope["commands"][0]["argv"].insert(
        2, "COVERAGE_RCFILE=" + scope["coverage_config_reference"]["path"]
    )
    scope["code_test_hashes"].append(e.ref(e.ROOT / e.CONFIG))
    scope.update(random_seed=e.FROZEN["seed"], exclusions=[], owned_paths=e.OWNED)
    e.progress("resources_before")
    limit = resource.getrlimit(resource.RLIMIT_AS)[0]
    resources = [
        e.check(
            raw,
            "free_disk_bytes>=2GiB",
            True,
            shutil.disk_usage(raw).free >= 2 * 1024**3,
            "resource_disk",
        ),
        e.check(
            raw,
            "address_space_at_least_256MiB",
            True,
            limit == resource.RLIM_INFINITY or limit >= 256 * 1024**2,
            "resource_memory",
        ),
        e.check(raw, "python_virtualenv", str(e.ROOT / ".venv"), sys.prefix, "resource_import"),
    ]
    atomic_json(raw / "frozen_scope.json", scope)
    e.progress("resources_after")
    inspected = e.inspect(root)
    inspected["checks"].extend(resources)
    ended = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="cpu_cached_authority_and_receipt_replay",
        inference_substrate_details=dict(
            work="hash authentication, archived receipt replay and task binding; no model, fit or generation"
        ),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=ended,
    )
    atomic_json(raw / "current_work.json", receipt)
    work = dict(
        root=str(root),
        raw=str(raw),
        run_date=date,
        scope=scope,
        inspection=inspected,
        current_work_reference=e.ref(raw / "current_work.json"),
        duration_s=(ended - start) / 1e9,
        start_ns=start,
    )
    atomic_json(raw / "work.json", work)
    return work


def receipts_pass(work: Json, validation: Json) -> bool:
    """Reuse real log, exact command and SQLite coverage validation without self-reported passes."""
    with patch.object(lifecycle, "d", e):
        return bool(lifecycle.receipts_pass(work, validation))


def qualify(work: Json) -> Json:
    """The reused supervisor runs owned checks and the broad suite once with heartbeats."""
    with patch.object(lifecycle, "d", e):
        return dict(lifecycle.qualify(work))


def build(work: Json, validation: Json) -> Json:
    """Recompute task counts and keep every readiness gate independent of scientific benefit."""
    inspected = work["inspection"]
    contract = inspected["contract"]
    passed = receipts_pass(work, validation)
    blocked = (
        not contract["ready"]
        or not all(inspected[k] for k in ["source_ready", "static_ready", "online_ready"])
        or not all(r["passed"] for r in inspected["checks"])
    )
    cls = "disqualified" if not passed and not blocked else "blocked" if blocked else "null"
    rows = deepcopy(contract["rows"])
    if not rows:
        rows = [
            dict(
                unit_id=f"exp{n}",
                qid=None,
                group_id=f"exp{n}",
                sibling=None,
                arm="full_task_contract",
                condition="exact_authority",
                seed=e.FROZEN["seed"],
                metric="ordered_full_object_match",
                absolute_metric=None,
                numerator=None,
                denominator=0,
                status="censored",
                exclusion_reason=None,
                censor_reason="missing_contract_authority",
            )
            for n in range(9138, 9152)
        ]
    completed = sum(r["status"] == "completed" for r in rows)
    failed = sum(r["status"] == "failed" for r in rows)
    censored = sum(r["status"] == "censored" for r in rows)
    value: Json = dict(
        schema="carnot.v779_contract.v1",
        experiment_id=9138,
        task_id=e.TASK,
        milestone="2026.11.779",
        run_date=work["run_date"],
        honest_verdict=f"complete_{cls}_v779_contract",
        verdict_class=cls,
        required_checks_passed=passed,
        flagged_adversarial=False,
        contract_ready_score=int(passed and contract["ready"]),
        source_ready_score=int(passed and inspected["source_ready"]),
        static_methods_ready_score=int(passed and inspected["static_ready"]),
        online_methods_ready_score=int(passed and inspected["online_ready"]),
        rows=rows,
        intended_count=14,
        attempted_count=len(rows) - censored,
        completed_count=completed,
        failed_count=failed,
        censored_count=censored,
        excluded_count=0,
        effective_independent_groups=0,
        independent_generalization_score=0,
        sample_size_budget=dict(
            intended=14,
            attempted=len(rows) - censored,
            completed=completed,
            failed=failed,
            censored=censored,
            effective_independent_groups=0,
        ),
        verifier_is_oracle=False,
        claim_scope=dict(
            scope="Administrative contract and original methods only; no measured human benefit or ARC solve. Cannot close GAP-ORACLE-DISTINCT or GAP-DETECTOR-AUROC-4208.",
            exposure_scope="Original LPHB900 answers/300 qids remain exposed development",
            independent_generalization_score=0,
        ),
        exposure_scope="exposed_development",
        MODEL_SPECS=[],
        requested_inference_substrate_class="no_model_load",
        inference_substrate_class="blocked_no_run" if blocked else "no_model_load",
        inference_substrate="cpu_cached_authority_and_receipt_replay",
        current_llm_invocations=0,
        generated_tokens=0,
        model_invocation_counts=dict(
            current_llm_invocations=0,
            loads_attempted=0,
            loads_completed=0,
            loads_failed=0,
            forward_calls=0,
            generated_tokens=0,
        ),
        duration_s=work["duration_s"],
        random_seed=e.FROZEN["seed"],
        gate_check_summary=inspected["checks"],
        full_task_objects=contract["tasks"],
        canonical_task_hash=canonical_hash(contract["tasks"]),
        previous_outcome_rows=inspected["previous_outcome_rows"],
        protocol_references=inspected["protocol_references"],
        source_references=inspected["source_references"],
        code_config_hashes=work["scope"]["code_test_hashes"],
        current_work_receipts=[work["current_work_reference"]],
        validation_receipts=validation.get("checks", []),
        owned_statement_coverage=validation.get("owned_statement_coverage", {}),
        repository_health=validation.get("repository_health", {}),
        validation_scope_manifest=work["scope"],
        retired_qualification=inspected["retired_qualification"],
        scheduling_repair_revision=e.FROZEN["scheduling_repair_revision"],
        archive_lag=inspected["archive_lag"],
        source_summary=inspected["source"],
        static_protocol=inspected["static_protocol"],
        online_protocol=inspected["online_protocol"],
        method_receipt_replays=inspected["method_receipt_replays"],
        historical_model_provenance=inspected.get("historical_model_provenance", {}),
        activation_authorized=False,
        external_publication_authorized=False,
        work_reference=e.ref(Path(work["raw"]) / "work.json"),
    )
    value["acceptance_gates"] = dict(
        contract="14 exact ordered objects including prompts, gates, deliverables and model metadata; independent of scientific availability",
        source="900 answers/300 qids with original roles, labels and lexical shards",
        static="Original Exp9126 five arms, formulas, transforms and receipt replay",
        online="Original Exp9132 six arms, delayed updates, recovery fixtures and seven bounds",
        validation="All affected checks and 100 percent owned statements with zero exclusions",
        scientific_benefit="unmeasured; fixtures and historical receipts grant no independent benefit",
    )
    value.update(inspected["protocol_references"])
    value["field_principles"] = {
        k: "Immutable operands and executed checks bind this administrative field; no scientific benefit is inferred."
        for k in value
    }
    value["field_principles"].update(
        contract_ready_score="Exact task equality is independent of scientific availability.",
        source_ready_score="Original evaluator labels and all question roles must authenticate.",
        static_methods_ready_score="Five static arms retain original authority and measured checks.",
        online_methods_ready_score="Six delayed arms and all seven bounds remain unchanged.",
        retired_qualification="New scheduling work never changes historical receipt readiness or quarantine bytes.",
        model_invocation_counts="Empty current event ledgers cannot inherit historical model activity.",
        sample_size_budget="Administrative slots and repeated siblings provide zero independent scientific groups.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Reauthenticate original operands and rebuild every aggregate, even after rehashing."""
    value = json.loads(path.read_bytes())
    reference = value["work_reference"]
    if e.ref(Path(reference["path"])) != reference:
        raise ValueError("work_hash")
    work = json.loads(Path(reference["path"]).read_bytes())
    fresh = e.inspect(Path(work["root"]))
    if fresh != {k: v for k, v in work["inspection"].items() if k != "checks"} | dict(
        checks=work["inspection"]["checks"][:-3]
    ):
        raise ValueError("source_drift")
    if e.ref(Path(work["current_work_reference"]["path"])) != work["current_work_reference"]:
        raise ValueError("current_work_hash")
    if validate_current_work_receipt(
        json.loads(Path(work["current_work_reference"]["path"]).read_bytes()), []
    ):
        raise ValueError("current_work_drift")
    validation = dict(
        checks=value["validation_receipts"],
        owned_statement_coverage=value["owned_statement_coverage"],
        repository_health=value["repository_health"],
    )
    if build(work, validation) != value:
        raise ValueError("reduction_drift")
    return dict(passed=True, replay_passed=True)


def terminal(candidate: Path, logs: Path) -> Json:
    """Unchanged checkers and a foreign cold replay must accept bytes before publication."""
    with patch.object(lifecycle, "d", e):
        return dict(lifecycle.terminal(candidate, logs))


def main(argv: list[str] | None = None) -> int:
    """Private replay cannot replace a live primary; production runs actual owned checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261116")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--from-work", type=Path)
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args(argv)
    e.progress("start_no_model_load")
    try:
        if args.date != "20261116":
            raise ValueError("run_date")
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        if args.from_work and args.output.absolute().is_relative_to(e.ROOT):
            raise ValueError("private_work_requires_external_output")
        raw = args.output.absolute().parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
        work = (
            json.loads(args.from_work.read_bytes())
            if args.from_work
            else measure(args.root, raw, args.date)
        )
        validation: Json = {}
        if not args.from_work and not args.checks_only:
            if (time.monotonic_ns() - work["start_ns"]) / 1e9 >= 3300:
                raise TimeoutError("measurement_deadline")
            e.progress("validation_before")
            validation = qualify(work)
            e.progress("validation_after")
            atomic_json(raw / "validation.json", validation)
            if (time.monotonic_ns() - work["start_ns"]) / 1e9 >= 4800:
                raise TimeoutError("hard_deadline")
        value = build(work, validation)
        atomic_json(raw / "candidate.json", value)
        if args.checks_only:
            print(json.dumps(dict(candidate=str(raw / "candidate.json"))), flush=True)
            return 0
        if not args.from_work and not value["required_checks_passed"]:
            e.progress("owned_validation_failed")
            return 1
        e.progress("publication_before")
        receipt = publish_primary(args.output, value, lambda path: terminal(path, raw / "terminal"))
        atomic_json(raw / "publication.json", receipt)
        print(json.dumps(dict(receipt, honest_verdict=value["honest_verdict"])), flush=True)
        e.progress("publication_after")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        e.progress("refused:" + str(error))
        return 1
