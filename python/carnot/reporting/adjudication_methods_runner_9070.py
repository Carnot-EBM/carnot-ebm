"""REQ-REPORT-9070-V774: publish a reproducible authority refusal and frozen methods.

Qualified checks certify this producer's work. They cannot supply missing author
lineage or turn retained anonymous adjudications into an independent truth source.
"""

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

from carnot.reporting import adjudication_methods_9070 as d
from carnot.reporting import mdh_human_release_runner_9056 as prior
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.natural_label_runner_8975 import read_ref
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]
DIRECT = [
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_current_work_receipt.py",
    "tests/python/test_mdh_human_release_9056.py",
]
REUSED = [
    *prior.REUSED,
    "python/carnot/reporting/mdh_human_release_runner_9056.py",
    "python/carnot/reporting/spline_basis_methods_8986.py",
    "python/carnot/reporting/natural_annotation_authority_8989.py",
    "python/carnot/reporting/human_disagreement_source_9015.py",
    "docs/research-notes/v774-adjudication-methods.md",
]


def scope_manifest(commands: list[Json]) -> Json:
    """Keep the qualified receipt policy while freezing this producer and its consumers."""
    with patch.multiple(prior, e=d, DIRECT=DIRECT, REUSED=REUSED):
        return prior.scope_manifest(commands)


def receipts_pass(scope: Json, validation: Json) -> bool:
    """Use the existing hash, normal-exit and real SQLite coverage checks unchanged."""
    with patch.object(prior, "e", d):
        return prior.receipts_pass(scope, validation)


def measure(inputs: Path, raw: Path, date: str, scope: Json) -> Json:
    """Record actual private work and source hashes without projecting target values."""
    d.progress("preconditions_before")
    start = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    storage = d.gate(
        raw, "private_storage_mode", 0o700, raw.stat().st_mode & 0o777, "private_storage"
    )
    reduced = d.inspect(inputs if storage["passed"] else raw / "unavailable")
    reduced["gate_check_summary"].insert(0, storage)
    if not storage["passed"]:
        reduced["blocker"] = "private_storage"
    atomic_json(raw / "reduction.json", reduced)
    elapsed = time.monotonic() - start
    health_path = d.DEFAULT_INPUT / "repository_health.json"
    work = dict(
        inputs=str(inputs),
        raw=str(raw),
        reduction_reference=d.ref(raw / "reduction.json"),
        run_date=date,
        duration_s=elapsed,
        validation_scope_manifest=scope,
        phase_spans=[dict(phase="authenticated_metadata_join", start_s=0, end_s=elapsed)],
        repository_health=json.loads(health_path.read_bytes())
        if health_path.is_file()
        else dict(
            passed=False, scope="broader suite", status="missing_file", reference=d.ref(health_path)
        ),
        preconditions_checked=dict(
            private_storage=d.ref(raw),
            private_mode=raw.stat().st_mode & 0o777,
            disk_free_bytes=shutil.disk_usage(raw).free,
            pinned_revision=d.REVISION,
            ledgers_read_before_acquisition=True,
            source_access_reference=d.ref(inputs / "access.json"),
            network_limits=dict(timeout_s=30, attempts=2, total_s=300),
            current_model_calls=0,
            protected_labels_opened=False,
        ),
    )
    atomic_json(raw / "work.json", work)
    d.progress("measurement_complete", len(reduced["rows"]), 0)
    return d.ref(raw / "work.json")


def build(reference: Json, validation: Json) -> Json:
    """External authority blocks stay distinct from failed owned implementation checks."""
    work = read_ref(reference)
    reduced = read_ref(work["reduction_reference"])
    value: Json = deepcopy(reduced)
    scope = work["validation_scope_manifest"]
    passed = receipts_pass(scope, validation)
    contract = d.protocol()
    rows = reduced["rows"]
    value.update(
        experiment_id=9070,
        task_id="exp9070-adjudication-methods",
        milestone="2026.11.774",
        honest_verdict="complete_blocked_" + reduced["blocker"].replace(":", "_")
        if passed
        else "complete_disqualified_owned_validation",
        verdict_class="blocked" if passed else "disqualified",
        required_checks_passed=passed,
        flagged_adversarial=not passed,
        verifier_is_oracle=False,
        independent_generalization_score=0,
        claim_scope="Pinned MDHJudgments authority audit and frozen human-target methods; no fitted head, medical correctness or general verifier moat.",
        exposure_scope="Five-release exact response/section join; original labels sealed; base-family construction and prior-exposure closure unknown.",
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        trained_head_specs=[],
        model_invocation_counts=dict(
            model_loads=0,
            generations=0,
            embedding_calls=0,
            forward_calls=0,
            small_head_training_runs=0,
            total=0,
        ),
        cited_upstream_artifacts=[
            dict(
                experiment_id=9056,
                sha256=d.HISTORY.get("results/experiment_9056_v773_human_release_authority.json"),
                fields_imported=["honest_verdict", "verdict_class", "gate_check_summary"],
                scope="historical failure only; MDH-Bench stays closed",
            )
        ],
        intended_count=len(rows),
        completed_count=0,
        failed_count=0,
        censored_count=0,
        excluded_count=len(rows),
        independent_count=0,
        human_source_ready_score=0,
        human_methods_ready_score=0,
        artifact_resolution_ready_score=0,
        release_revision=d.REVISION,
        rights_reference=dict(
            reference=d.ref(Path(work["inputs"]) / "DATA_LICENSE"),
            scope="CC-BY-4.0 dataset except third-party evidence excerpts; code MIT; retain excerpts privately",
        ),
        original_label_mapping=contract["mapping"],
        feature_protocol=contract["features"],
        head_protocol=contract["heads"],
        human_bound_contract=contract["bound"],
        fold_protocol=contract["folds"],
        target_panel=contract["panel"],
        target_custody_reference=dict(
            references=reduced["release_hashes"],
            outcomes_opened=False,
            learner_access=False,
            unique_people="unknown",
            judgment_field=contract["judgment_field"],
        ),
        schema_mutation_rows=[
            dict(control=c, passed=passed, evidence="owned_tests validation receipt")
            for c in ["denied_label", "mismatched_ID", "duplicated_section", "generator_metadata"]
        ],
        run_date=work["run_date"],
        duration_s=work["duration_s"],
        random_seed=774070,
        phase_spans=work["phase_spans"],
        preconditions_checked=work["preconditions_checked"],
        primitive_reference=reference,
        raw_shard_hashes=[reference, work["reduction_reference"]],
        code_config_hashes=scope.get("code_test_hashes", []),
        validation_scope_manifest=scope,
        validation_receipts=validation.get("checks", []),
        owned_statement_coverage=validation.get("owned_statement_coverage", {}),
        repository_health=work["repository_health"],
        methodology_note="No numerical feasibility measurement is claimed: independent family authority is unknown. Missing authority blocks; a genuinely measured insufficient cohort would be terminal null. Neither the failed 400-group scope nor composite adjudication labels are reused.",
        reconciliation="Ops/status/traceability delegated to conductor by explicit task instruction",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> Json:
    """Recompute metadata from exact originals so rehashing a summary cannot invent authority."""
    value = json.loads(path.read_bytes())
    for reference in (
        value["source_artifact_hashes"] + value["raw_shard_hashes"] + value["code_config_hashes"]
    ):
        if d.ref(Path(reference["path"])) != reference:
            raise ValueError("operand_hash")
    work = read_ref(value["primitive_reference"])
    fresh = d.inspect(Path(work["inputs"]))
    retained = read_ref(work["reduction_reference"])
    fresh["gate_check_summary"].insert(0, retained["gate_check_summary"][0])
    if fresh != retained:
        raise ValueError("source_reduction_drift")
    expected = build(
        value["primitive_reference"],
        dict(
            checks=value["validation_receipts"],
            owned_statement_coverage=value["owned_statement_coverage"],
        ),
    )
    if expected != value:
        raise ValueError("reduction_drift")
    return dict(passed=True, replay_passed=True, primary_sha256=d.ref(path)["sha256"])


def validate(candidate: Path, logs: Path) -> Json:
    """Reuse unchanged adversarial/row scanners and the actual foreign-directory CLI."""
    with patch.object(prior, "e", d):
        return prior.validate(candidate, logs)


def main(argv: list[str] | None = None) -> int:
    """Only checked terminal bytes reach the primary; all retained targets stay private."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    d.progress("start_no_model_load")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261113")
    parser.add_argument("--inputs", type=Path, default=d.DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=d.ROOT / "results" / (d.NAME + ".json"))
    parser.add_argument(
        "--scope", type=Path, default=d.DEFAULT_INPUT / "validation_scope_manifest.json"
    )
    parser.add_argument("--receipts", type=Path, default=d.DEFAULT_INPUT / "validation.json")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.date != "20261113":
            raise ValueError("run_date")
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        scope = json.loads(args.scope.read_bytes()) if args.scope.is_file() else scope_manifest([])
        validation = json.loads(args.receipts.read_bytes()) if args.receipts.is_file() else {}
        raw = Path(tempfile.mkdtemp(prefix="carnot-9070-work-"))
        reference = measure(args.inputs, raw, args.date, scope)
        value = build(reference, validation)
        atomic_json(raw / "candidate.json", value)
        if not value["required_checks_passed"]:
            d.progress("owned_checks_failed_primary_preserved")
            return 1
        d.progress("publication_before")
        receipt = publish_primary(args.output, value, lambda p: validate(p, raw / "terminal"))
        print(json.dumps(receipt), flush=True)
        d.progress("complete", len(value["rows"]), 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        d.progress("refused:" + str(error))
        return 1
