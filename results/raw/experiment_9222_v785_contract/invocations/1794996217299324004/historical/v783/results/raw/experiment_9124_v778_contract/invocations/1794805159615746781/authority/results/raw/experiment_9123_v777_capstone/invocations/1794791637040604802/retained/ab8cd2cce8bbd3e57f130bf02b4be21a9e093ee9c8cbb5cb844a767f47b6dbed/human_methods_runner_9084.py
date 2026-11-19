"""REQ-REPORT-9084-V775: publish only qualified methods and honest source refusals."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import human_methods_9084 as d
from carnot.reporting import mdh_human_release_runner_9056 as prior
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
)
from carnot.reporting.natural_label_runner_8975 import read_ref
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]
DIRECT = [
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_current_work_receipt.py",
    "tests/python/test_human_label_custody_8389.py",
    "tests/python/test_released_question_identity_8403.py",
    "tests/python/test_retrospective_label_criterion_8409.py",
]
REUSED = [
    *prior.REUSED,
    "python/carnot/reporting/mdh_human_release_runner_9056.py",
    "python/carnot/reporting/human_label_custody_8389.py",
    "python/carnot/reporting/spline_basis_methods_8986.py",
    "python/carnot/autoresearch/spline_basis_8986.py",
]


def scope_manifest(commands: list[Json]) -> Json:
    """Reuse strict receipt validation while freezing this task's owned and consumer paths."""
    with patch.multiple(prior, e=d, DIRECT=DIRECT, REUSED=REUSED):
        return prior.scope_manifest(commands)


def receipts_pass(scope: Json, validation: Json) -> bool:
    """Only actual frozen commands, durable log hashes and complete SQLite coverage qualify."""
    with patch.object(prior, "e", d):
        return prior.receipts_pass(scope, validation)


def measure(root: Path, raw: Path, date: str, scope: Json) -> Json:
    """Seal protocol before independent source inventory; never load a model or fit a head."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    storage = d.gate(raw, "mode", 0o700, raw.stat().st_mode & 0o777, "private_storage")
    atomic_json(raw / "protocol.json", d.protocol())
    reduced = d.inspect(root if storage["passed"] else raw / "unavailable")
    reduced["gate_check_summary"].insert(0, storage)
    if not storage["passed"]:
        reduced["blocker"] = "private_storage"
    end = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details=dict(
            work="original source inventory and frozen design",
            embedding_calls=0,
            head_training_runs=0,
        ),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
    )
    atomic_json(raw / "current_work.json", receipt)
    work = dict(
        root=str(root),
        raw=str(raw),
        run_date=date,
        reduction=reduced,
        duration_s=(end - start) / 1e9,
        validation_scope_manifest=scope,
        protocol_reference=d.ref(raw / "protocol.json"),
        current_work_reference=d.ref(raw / "current_work.json"),
    )
    atomic_json(raw / "work.json", work)
    for name in ["protocol", "current_work", "work"]:
        (raw / (name + ".json")).chmod(0o400)
    d.progress("protocol_and_inventory_frozen", len(reduced["rows"]), 0)
    return d.ref(raw / "work.json")


def build(reference: Json, validation: Json) -> Json:
    """Methods readiness measures authentic inputs and owned checks, never a favorable outcome."""
    work = read_ref(reference)
    value: Json = deepcopy(work["reduction"])
    passed = receipts_pass(work["validation_scope_manifest"], validation)
    blocked = value["blocker"] is not None
    ready = int(passed and not blocked)
    verdict = "disqualified" if not passed else "blocked" if blocked else "positive"
    contract = read_ref(work["protocol_reference"])
    value.update(
        schema="carnot.exposed_human_methods.v1",
        experiment_id=9084,
        task_id=d.TASK,
        milestone="2026.11.775",
        honest_verdict="complete_disqualified_owned_validation"
        if not passed
        else "complete_blocked_" + value["blocker"]
        if blocked
        else "complete_positive_exposed_human_methods",
        verdict_class=verdict,
        required_checks_passed=passed,
        flagged_adversarial=not passed,
        verifier_is_oracle=False,
        claim_scope="Original exposed LPHB inventory and frozen embedding comparison protocol only; no detector benefit, independent source qualification or MDHJudgments repair.",
        exposure_scope="All900 released answers/300 original qids are exposed development through Exp8389/8403/8409 and Exp8429-8435; earlier unknown overlap remains unresolved.",
        independent_generalization_score=0,
        independent_eval_ready_score=0,
        human_methods_ready_score=ready,
        MODEL_SPECS=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        current_llm_invocations=0,
        model_invocation_counts=dict(
            model_loads=0, generations=0, embedding_calls=0, head_training_runs=0, total=0
        ),
        historical_model_provenance=dict(
            scope="cited source provenance only; not current execution",
            references=value["source_references"],
        ),
        human_protocol_reference=work["protocol_reference"],
        protocol=contract,
        prohibited_feature_columns=d.PROHIBITED,
        arm_roster=contract["arms"],
        criterion_definitions=dict(
            factuality="Original A1 factual correctness, 1=hallucination/error; A2 sensitivity on observed subset",
            reference_faithfulness="Agreement with author reference only; separate context, never learned features or replacement target",
            missing="-1 remains missing; disagreement never forces consensus",
        ),
        current_work_receipts=[work["current_work_reference"]],
        intended_count=len(value["rows"]),
        attempted_count=len(value["rows"]),
        completed_count=len(value["rows"]),
        failed_count=0,
        censored_count=0,
        independent_count=value["question_count"],
        sample_size_budget=dict(
            intended_responses=900,
            intended_question_groups=300,
            effective_independent_group_count=value["question_count"],
            independent_eval_groups=0,
            seeds_and_predictions_multiply_units=False,
        ),
        head_comparison_status="protocol_only_not_measured",
        per_arm_metrics=None,
        coefficient_hashes=None,
        acceptance_gates=dict(
            authenticated_sources=not blocked, owned_validation=passed, frozen_design=True
        ),
        run_date=work["run_date"],
        duration_s=work["duration_s"],
        random_seed=775084,
        primitive_reference=reference,
        raw_shard_hashes=[reference, work["protocol_reference"], work["current_work_reference"]],
        code_config_hashes=work["validation_scope_manifest"].get("code_test_hashes", []),
        validation_scope_manifest=work["validation_scope_manifest"],
        validation_receipts=validation.get("checks", []),
        owned_statement_coverage=validation.get("owned_statement_coverage", {}),
        repository_health=validation.get(
            "repository_health", dict(status="not_supplied", passed=False)
        ),
        methodology_note="Authenticated original CSV/dictionary, unchanged V726 groups and all known exposure. Protocol frozen before any extraction. Zero current model loads, embeddings, fits or benefit measurements. Historical capacity null and independent-source failures remain unchanged.",
        reconciliation="Conductor owns ops/status/traceability updates; no roadmap activation or external publication.",
    )
    value["field_principles"] = {
        k: "Bind to retained source or owned receipt; exposed methods do not establish scientific benefit."
        for k in value
    }
    value["field_principles"].update(
        human_methods_ready_score="Integer1 only for authenticated exposed source, frozen design and all owned checks; independent readiness remains0.",
        rows="One original response per row; sibling responses share their original qid unit.",
        gate_check_summary="Explicit expected and observed operands distinguish missing_file, missing_field and observed zero.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> Json:
    """Recompute original rows and the entire terminal report; rehashed summaries have no authority."""
    value = json.loads(path.read_bytes())
    for reference in (
        value["source_references"] + value["raw_shard_hashes"] + value["code_config_hashes"]
    ):
        if d.ref(Path(reference["path"])) != reference:
            raise ValueError("operand_hash")
    work = read_ref(value["primitive_reference"])
    if read_ref(work["protocol_reference"]) != d.protocol():
        raise ValueError("protocol_drift")
    fresh = d.inspect(Path(work["root"]))
    fresh["gate_check_summary"].insert(0, work["reduction"]["gate_check_summary"][0])
    if fresh != work["reduction"]:
        raise ValueError("source_reduction_drift")
    validation = dict(
        checks=value["validation_receipts"],
        owned_statement_coverage=value["owned_statement_coverage"],
        repository_health=value["repository_health"],
    )
    if build(value["primitive_reference"], validation) != value:
        raise ValueError("reduction_drift")
    return dict(passed=True, replay_passed=True, primary_sha256=d.ref(path)["sha256"])


def validate(candidate: Path, logs: Path) -> Json:
    """Reuse unchanged scanners and foreign-directory cold replay before atomic publication."""
    with patch.object(prior, "e", d):
        return prior.validate(candidate, logs)


def main(argv: list[str] | None = None) -> int:
    """Publish qualified terminal methods; failed owned checks leave any primary untouched."""
    d.progress("start_no_model_load")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261115")
    parser.add_argument("--root", type=Path, default=d.ROOT)
    parser.add_argument("--output", type=Path, default=d.ROOT / "results" / (d.NAME + ".json"))
    parser.add_argument(
        "--scope", type=Path, default=d.DEFAULT_INPUT / "validation_scope_manifest.json"
    )
    parser.add_argument("--receipts", type=Path, default=d.DEFAULT_INPUT / "validation.json")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.date != "20261115":
            raise ValueError("run_date")
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        scope = json.loads(args.scope.read_bytes()) if args.scope.is_file() else scope_manifest([])
        validation = json.loads(args.receipts.read_bytes()) if args.receipts.is_file() else {}
        raw = Path(tempfile.mkdtemp(prefix="carnot-9084-work-"))
        reference = measure(args.root, raw, args.date, scope)
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
