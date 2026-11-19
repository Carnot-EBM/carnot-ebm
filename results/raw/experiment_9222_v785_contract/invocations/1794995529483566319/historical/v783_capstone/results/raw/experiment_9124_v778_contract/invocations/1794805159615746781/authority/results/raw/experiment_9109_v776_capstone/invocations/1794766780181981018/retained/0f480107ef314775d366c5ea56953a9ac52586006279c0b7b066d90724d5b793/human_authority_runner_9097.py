"""REQ-REPORT-9097-V776: qualify exposed custody separately from scientific benefit."""

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

from carnot.reporting import human_authority_9097 as d
from carnot.reporting import human_methods_runner_9084 as previous
from carnot.reporting import mdh_human_release_runner_9056 as prior
from carnot.reporting.current_work_receipt import build_current_work_receipt, canonical_hash
from carnot.reporting.natural_label_runner_8975 import read_ref
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]
DIRECT = [*previous.DIRECT, "tests/python/test_human_methods_9084.py"]
REUSED = [
    *previous.REUSED,
    *d.methods.OWNED[:2],
    "python/carnot/reporting/human_methods_runner_9084.py",
]


def scope_manifest(commands: list[Json]) -> Json:
    """Freeze actual owned and consumer command vectors before any outcome is reduced."""
    with patch.multiple(prior, e=d, DIRECT=DIRECT, REUSED=REUSED):
        return prior.scope_manifest(commands)


def receipts_pass(scope: Json, validation: Json) -> bool:
    """Require normal exits, durable original logs and real SQLite statement coverage."""
    with patch.object(prior, "e", d):
        return prior.receipts_pass(scope, validation)


def measure(root: Path, raw: Path, date: str, scope: Json) -> Json:
    """Freeze methods before authenticating bytes; no model or fitted head is involved."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    protocol_ref = d.seal(raw / "protocol.json", d.methods.protocol())
    private = raw.stat().st_mode & 0o777 == 0o700
    inspected_root = root if private else raw / "unavailable"
    reduced = d.inspect(inspected_root, raw / "source")
    reduced["gate_check_summary"].insert(
        0, d.gate(raw, "mode", 0o700, raw.stat().st_mode & 0o777, "private_storage")
    )
    if not private:
        reduced["blocker"] = "private_storage"
    end = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details=dict(
            work="historical byte authentication and exposed source sealing", model_loads=0
        ),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
    )
    current = d.seal(raw / "current_work.json", receipt)
    return d.seal(
        raw / "work.json",
        dict(
            root=str(root),
            inspected_root=str(inspected_root),
            raw=str(raw),
            run_date=date,
            reduction=reduced,
            duration_s=(end - start) / 1e9,
            validation_scope_manifest=scope,
            static_protocol_reference=protocol_ref,
            current_work_reference=current,
        ),
    )


def build(reference: Json, validation: Json) -> Json:
    """Readiness requires authentic exposed inputs and every affected check, not a benefit claim."""
    work = read_ref(reference)
    value: Json = deepcopy(work["reduction"])
    passed = receipts_pass(work["validation_scope_manifest"], validation)
    blocked = value["blocker"] is not None
    ready = int(passed and not blocked)
    value.update(
        schema="carnot.historical_human_authority.v1",
        experiment_id=9097,
        task_id=d.TASK,
        milestone="2026.11.776",
        honest_verdict="complete_disqualified_owned_validation"
        if not passed
        else "complete_blocked_" + value["blocker"]
        if blocked
        else "complete_positive_exposed_human_authority",
        verdict_class="disqualified" if not passed else "blocked" if blocked else "positive",
        required_checks_passed=passed,
        flagged_adversarial=not passed,
        source_ready_score=ready,
        independent_eval_ready_score=0,
        independent_generalization_score=0,
        verifier_is_oracle=False,
        claim_scope="Authenticated original producer, exposed label custody, retained roles and sealed input views only. Administrative readiness establishes no scientific benefit; no current ARC solve is claimed.",
        exposure_scope="LPHB is exposed development: all900 archived answers/300 original qids retain Exp8389/8403/8409 and Exp8429-8435 exposure. Earlier unknown overlap remains unresolved.",
        MODEL_SPECS=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        current_llm_invocations=0,
        model_invocation_counts=dict(
            model_loads=0, generations=0, embedding_calls=0, head_training_runs=0, total=0
        ),
        historical_model_provenance=dict(
            scope="original upstream evidence only; separate from zero current model work"
        ),
        static_protocol_reference=work["static_protocol_reference"],
        current_work_receipts=[work["current_work_reference"]],
        intended_count=900,
        attempted_count=len(value["rows"]),
        completed_count=len(value["rows"]),
        failed_count=0,
        censored_count=900 - len(value["rows"]),
        effective_independent_groups=dict(
            exposed_development_count=value["question_count"],
            independent_evaluation_count=0,
            unit="original_qid",
        ),
        independent_count=value["question_count"],
        sample_size_budget=dict(
            intended_responses=900,
            intended_question_groups=300,
            effective_independent_group_count=value["question_count"],
            independent_eval_groups=0,
            siblings_and_seeds_multiply_units=False,
        ),
        acceptance_gates=dict(
            authenticated_closure=not blocked,
            owned_validation=passed,
            unchanged_static_protocol=True,
        ),
        run_date=work["run_date"],
        random_seed=7769097,
        duration_s=work["duration_s"],
        primitive_reference=reference,
        raw_shard_hashes=[
            reference,
            work["static_protocol_reference"],
            work["current_work_reference"],
        ],
        code_config_hashes=work["validation_scope_manifest"].get("code_test_hashes", []),
        validation_scope_manifest=work["validation_scope_manifest"],
        validation_receipts=validation.get("checks", []),
        owned_statement_coverage=validation.get("owned_statement_coverage", {}),
        repository_health=validation.get(
            "repository_health", dict(passed=False, status="not_supplied")
        ),
        methodology_note="Exact pinned upstream bytes precede historical replay. Independent author CSV/dictionary and original V726 roles agree on every item and both annotators. Features contain question and answer text only; evaluator labels and shard rosters are separate immutable private files. No model, extraction, fitting or scientific benefit measurement. Prior capacity-benefit null is unchanged.",
    )
    principles = [
        (
            "honest_verdict verdict_class",
            "Terminal classification distinguishes external blockage, owned failure and administrative qualification.",
        ),
        (
            "source_ready_score required_checks_passed acceptance_gates",
            "Only complete authentic closure and affected checks qualify exposed inputs.",
        ),
        (
            "independent_eval_ready_score independent_generalization_score claim_scope exposure_scope exposure_ledger",
            "Known exposure cannot become independent evidence or a scientific benefit claim.",
        ),
        (
            "rows role_counts intended_count attempted_count completed_count failed_count censored_count effective_independent_groups independent_count sample_size_budget",
            "Recompute support from original per-response rows; siblings and seeds never add independent questions.",
        ),
        (
            "MODEL_SPECS inference_substrate inference_substrate_class current_llm_invocations model_invocation_counts current_work_receipts historical_model_provenance",
            "Separate current owned execution from historical model activity.",
        ),
        (
            "historical_producer_snapshot_reference replay_receipt source_manifest_reference label_authority_reference source_references",
            "Immutable original bytes and isolated execution establish custody, not a locally rehashed substitute.",
        ),
        (
            "original_group_partition_reference static_protocol static_protocol_reference shard_a_roster_reference shard_b_roster_reference evaluator_label_reference",
            "Freeze methods, roles and learner/evaluator boundaries before later outcomes.",
        ),
        (
            "validation_scope_manifest validation_receipts owned_statement_coverage repository_health",
            "Actual command exits and measured statements qualify affected work; broad historical debt stays separate.",
        ),
        (
            "run_date random_seed duration_s reproducibility_checksum code_config_hashes primitive_reference raw_shard_hashes",
            "Exact code, operands and owned timing make the result reproducible.",
        ),
        (
            "original_capacity_null_reference",
            "Recovered custody cannot reverse a prior capacity-benefit null.",
        ),
    ]
    value["field_principles"] = {k: why for keys, why in principles for k in keys.split()}
    value["field_principles"]["gate_check_summary"] = (
        "Explicit expected and observed operands distinguish missing_file, missing_field and observed zero."
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> Json:
    """Recompute every source row and sealed boundary so rehashing cannot bless tampering."""
    value = json.loads(path.read_bytes())
    work = read_ref(value["primitive_reference"])
    for reference in (
        value["source_references"] + value["raw_shard_hashes"] + value["code_config_hashes"]
    ):
        if d.ref(Path(reference["path"])) != reference:
            raise ValueError("operand_hash")
    if read_ref(work["static_protocol_reference"]) != d.methods.protocol():
        raise ValueError("protocol_drift")
    fresh = d.inspect(Path(work["inspected_root"]), Path(work["raw"]) / "source")
    fresh["gate_check_summary"].insert(0, work["reduction"]["gate_check_summary"][0])
    if work["reduction"]["blocker"] == "private_storage":
        fresh["blocker"] = "private_storage"
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
    """Unchanged scanners and actual outside-checkout replay must accept the candidate."""
    with patch.object(prior, "e", d):
        return prior.validate(candidate, logs)


def main(argv: list[str] | None = None) -> int:
    """Publish only after owned checks pass; preserve existing primary on any refusal."""
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
        raw = Path(tempfile.mkdtemp(prefix="carnot-9097-work-"))
        reference = measure(args.root, raw, args.date, scope)
        value = build(reference, validation)
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
