"""REQ-REPORT-9056-AUTHORITY: publish only source-bound authority audits.

A completed audit can report unavailable external authority. Current owned checks
still have to pass, and their failure cannot become a scientific null finding.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

from carnot.reporting import mdh_human_release_9056 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.natural_label_runner_8975 import read_ref
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child
from carnot.reporting.v756_branch_qualification import verify_measured

Json = dict[str, Any]
CHECKS = [
    "owned_tests",
    "direct_consumers",
    "e2e018",
    "e2e021",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
    "owned_coverage",
]
DIRECT = [
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_current_work_receipt.py",
    "tests/python/test_annotation_feasibility_9044.py",
]
REUSED = [
    "python/carnot/reporting/" + name + ".py"
    for name in [
        "primary_publication",
        "current_work_receipt",
        "v709_execution",
        "v756_branch_qualification",
        "human_source_authority_9001",
        "natural_label_eligibility_8975",
        "natural_label_runner_8975",
    ]
] + ["scripts/adversarial_verify.py", "scripts/verdict_row_consistency_lint.py"]


def scope_manifest(commands: list[Json]) -> Json:
    """Freeze owned paths and exact checks before source measurements begin."""
    return dict(
        commands=commands,
        owned_paths=e.OWNED,
        code_test_hashes=[e.ref(e.ROOT / p) for p in [*e.OWNED, e.TEST, *DIRECT, *REUSED]],
        frozen_ns=time.monotonic_ns(),
        required_check_names=CHECKS,
        statement_percent=100,
        excluded_statements=0,
        measurement_cap_s=3300,
        validation_reserve_s=1200,
        hard_cap_s=4800,
    )


def receipts_pass(scope: Json, validation: Json) -> bool:
    """Require real normal exits, original commands, current timing and measured SQLite coverage."""
    try:
        expected = {c["name"]: c["argv"] for c in scope["commands"]}
        checks = validation["checks"]
        return bool(
            sorted(expected) == sorted(CHECKS)
            and sorted(c["name"] for c in checks) == sorted(CHECKS)
            and all(e.ref(Path(r["path"])) == r for r in scope["code_test_hashes"])
            and all(
                c["passed"] is True
                and c["normal_exit"] is True
                and c["timed_out"] is False
                and c["exit_code"] == 0
                and c["started_monotonic_ns"] >= scope["frozen_ns"]
                and c["ended_monotonic_ns"] >= c["started_monotonic_ns"]
                and c["scope_sha256"] == canonical_hash(scope)
                and c["argv"] == expected[c["name"]]
                and all(
                    e.ref(Path(c[s + "_path"]))["sha256"] == c[s + "_sha256"]
                    and c[s + "_sha256"] is not None
                    for s in ["stdout", "stderr"]
                )
                for c in checks
            )
            and verify_measured(validation["owned_statement_coverage"], e.OWNED)
        )
    except (KeyError, ValueError, TypeError, OSError):
        return False


def measure(inputs: Path, output: Path, date: str, scope: Json) -> Json:
    """Check environment and free resources before authenticated metadata projection."""
    e.progress("preconditions_before")
    start = time.monotonic_ns()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    atomic_json(raw / "validation_scope_manifest.json", scope)
    disk = shutil.disk_usage(raw).free
    memory = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    resources = (
        min(disk, memory) >= 300_000_000
        and Path(sys.prefix).resolve() == (e.ROOT / ".venv").resolve()
    )
    e.progress("preconditions_after", int(resources), int(not resources))
    inspected = inputs if resources else raw / "unavailable_inputs"
    reduced = e.inspect(inspected)
    reduced["gate_check_summary"].insert(
        0, e.gate("virtualenv_and_resources", True, resources, e.ref(inputs))
    )
    atomic_json(
        raw / "primitive_rows.json",
        {
            k: reduced[k]
            for k in [
                "records",
                "annotation_authority_rows",
                "group_rows",
                "exposure_join_rows",
                "workbook_schema_rows",
                "workbook_member_rows",
            ]
        },
    )
    end = time.monotonic_ns()
    primitive = dict(
        inputs=str(inputs.absolute()),
        inspected_inputs=str(inspected.absolute()),
        reduction=reduced,
        raw_rows=e.ref(raw / "primitive_rows.json"),
        run_date=date,
        duration_s=(end - start) / 1e9,
        phase_spans=[
            dict(phase="authenticated_release_audit", start_s=0, end_s=(end - start) / 1e9)
        ],
        validation_scope_manifest=scope,
        preconditions_checked=dict(
            python_executable=sys.executable,
            virtualenv=sys.prefix,
            free_disk_bytes=disk,
            available_memory_bytes=memory,
            resource_floor_bytes=300_000_000,
            MODEL_SPECS=[],
            current_model_calls=0,
            network_acquisition="retained pinned author bytes only; no retries",
            acquisition_limits=dict(
                max_bytes=100000000, max_requests=12, timeout_s=25, total_s=300
            ),
            current_gate_fields=[
                "original_independent_human_judgments_per_response",
                "original_factual_error_criterion",
                "source_family_and_exposure_closure",
                "independently_authenticated_unexposed_prompt_groups",
            ],
            current_gate_operands=reduced["gate_check_summary"],
        ),
    )
    atomic_json(raw / "primitive.json", primitive)
    e.progress("measurement_complete", reduced["completed_count"], 0)
    return e.ref(raw / "primitive.json")


def build(reference: Json, validation: Json) -> Json:
    """Separate an externally blocked study from owned qualification and historical debt."""
    p = read_ref(reference)
    value: Json = deepcopy(p["reduction"])
    value.pop("records")
    value.pop("group_rows")
    passed = receipts_pass(p["validation_scope_manifest"], validation)
    blockers = [g for g in value["gate_check_summary"] if not g["passed"]]
    value.update(
        experiment_id=9056,
        task_id="exp9056-human-release-authority",
        milestone="2026.11.773",
        honest_verdict="complete_disqualified_owned_validation"
        if not passed
        else "complete_blocked_" + blockers[0]["check"].replace(":", "_")
        if blockers
        else "complete_null_no_eligible_pilot",
        verdict_class="disqualified" if not passed else "blocked" if blockers else "null",
        required_checks_passed=passed,
        flagged_adversarial=not passed,
        verifier_is_oracle=False,
        claim_scope="Original MDH release authority audit only; no verifier, detector, factual oracle or generalization result.",
        exposure_scope="All retained release prompt/response slots projected by allowlist; source families and historical overlap remain unknown. Original targets and class counts withheld.",
        independent_generalization_score=0,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        trained_head_specs=[],
        model_invocation_counts=dict(
            model_loads=0, generations=0, embedding_calls=0, forward_calls=0, total=0
        ),
        run_date=p["run_date"],
        duration_s=p["duration_s"],
        random_seed=7739056,
        phase_spans=p["phase_spans"],
        primitive_reference=reference,
        raw_shard_hashes=[reference, p["raw_rows"]],
        code_config_hashes=p["validation_scope_manifest"]["code_test_hashes"],
        preconditions_checked=p["preconditions_checked"],
        validation_scope_manifest=p["validation_scope_manifest"],
        validation_receipts=validation.get("checks", []),
        owned_statement_coverage=validation.get("owned_statement_coverage", {}),
        repository_health=dict(
            passed=False,
            scope="historical broad suite only; not rerun per task",
            references=[e.ref(e.ROOT / name) for name in e.HISTORY if name.startswith("results/")],
        ),
        group_manifest_reference=p["raw_rows"],
        protected_label_reference=dict(
            references=value["release_hashes"],
            field_policy="Only allowlisted source/text cells decoded; all other values retained in original workbook bytes",
            outcomes_reduced=False,
            class_counts="protected_not_evaluated",
            mode="0600",
        ),
        release_date_reconciliation=dict(
            release_revision=e.REVISION,
            release_last_modified="2026-03-09T15:46:46.000Z",
            manuscript_publication="2026-10-15",
            manuscript_claims=dict(
                prompts=400,
                annotators=3,
                authority="task context; article access refused at Nature identity redirect",
            ),
            release_card="paper under review/forthcoming",
            manuscript_release_identity="unresolved; publication claims do not authenticate rows",
            viewer_rows=28,
            viewer_unit="configuration_text_lines_not_responses",
            advertised_configs={
                name: json.loads((Path(p["inputs"]) / name).read_bytes())
                for name in e.PINS
                if name.startswith("config") and (Path(p["inputs"]) / name).is_file()
            },
            runs="three response worksheets and a cross-run sheet per workbook; sheet position does not authenticate generation or original annotators",
        ),
        criterion_mapping=dict(
            original_factual_error_criterion="unknown_original_instructions_and_individual_votes",
            aggregate_correctness="protected; not a substitute for original per-person factual-error judgments",
            R_Score="excluded_composite",
            automatic_labels="excluded",
            disagreement="unknown",
            missingness="unknown_original_person_level_judgments",
        ),
        pilot_contract=dict(
            salt="v773-mdh-v1",
            canonical_rank="SHA256(salt + ':' + canonical_group_id)",
            role_counts=dict(FIT=160, CAL=40, TEST=200),
            minimum_distinct_groups_per_class_per_role=10,
            roles_frozen=False,
            selected_subset=False,
            trusted_evaluator_readiness=0,
            source_family_complete=False,
        ),
        schema_mutation_rows=[
            dict(
                scenario="SCENARIO-REPORT-9056-SCHEMA",
                evidence="private owned pytest receipt",
                passed=passed,
            )
        ],
        cited_upstream_artifacts=[
            dict(
                experiment_id=int(Path(name).name.split("_")[1]),
                sha256="sha256:" + pin,
                fields_imported=[
                    "honest_verdict",
                    "verdict_class",
                    "source_identity_ready_score",
                    "annotation_structure_ready_score",
                ],
                historical_model_receipts="reference_only",
            )
            for name, pin in e.HISTORY.items()
            if name.startswith("results/")
        ],
        methodology_note="Authenticate pinned release, metadata and preserved historical design/queue bytes, inspect safe ZIP/XML members, project source/text allowlist, cluster prompt IDs and token-set similarity transitively before labels, join frozen historical exposure keys. Missing attributable original human votes and source-family closure block eligibility. No model, training, fitted head or outcome reduction.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> Json:
    """Recompute from pinned originals so rehashed summaries cannot invent authority."""
    value = json.loads(path.read_bytes())
    p = read_ref(value["primitive_reference"])
    for reference in (
        value["source_artifact_hashes"] + value["raw_shard_hashes"] + value["code_config_hashes"]
    ):
        if e.ref(Path(reference["path"])) != reference:
            raise ValueError("operand_hash")
    fresh = e.inspect(Path(p["inspected_inputs"]))
    fresh["gate_check_summary"].insert(0, p["reduction"]["gate_check_summary"][0])
    if fresh != p["reduction"]:
        raise ValueError("source_reduction_drift")
    if read_ref(p["raw_rows"]) != {
        k: fresh[k]
        for k in [
            "records",
            "annotation_authority_rows",
            "group_rows",
            "exposure_join_rows",
            "workbook_schema_rows",
            "workbook_member_rows",
        ]
    }:
        raise ValueError("primitive_rows_drift")
    expected = build(
        value["primitive_reference"],
        dict(
            checks=value["validation_receipts"],
            owned_statement_coverage=value["owned_statement_coverage"],
        ),
    )
    if expected != value:
        raise ValueError("reduction_drift")
    return dict(passed=True, replay_passed=True, primary_sha256=e.ref(path)["sha256"])


def validate(candidate: Path, logs: Path) -> Json:
    """Run unchanged scanners and actual foreign-directory replay before atomic publication."""
    python = str(e.ROOT / ".venv/bin/python")
    commands = [
        (
            "cold_replay",
            [
                "env",
                "-u",
                "PYTHONPATH",
                "-C",
                tempfile.gettempdir(),
                python,
                "-u",
                str(e.ROOT / e.CLI),
                "--cold-replay",
                str(candidate),
            ],
        ),
        (
            "adversarial",
            [python, "-u", str(e.ROOT / "scripts/adversarial_verify.py"), "--json", str(candidate)],
        ),
        (
            "strict_rows",
            [
                python,
                "-u",
                str(e.ROOT / "scripts/verdict_row_consistency_lint.py"),
                "--strict",
                str(candidate),
            ],
        ),
    ]
    checks = [child(name, argv, logs, deadline=120, heartbeat=20) for name, argv in commands]
    return dict(passed=all(c["passed"] and c["normal_exit"] for c in checks), checks=checks)


def main(argv: list[str] | None = None) -> int:
    """Retained release bytes support replay without new acquisition or regenerated responses."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    e.progress("start_no_model_load")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261112")
    parser.add_argument("--inputs", type=Path, default=e.DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--scope", type=Path)
    parser.add_argument("--receipts", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.date != "20261112":
            raise ValueError("run_date")
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        raw = args.output.parent / "raw" / e.NAME
        scope_path = args.scope or raw / "validation_scope_manifest.json"
        receipts_path = args.receipts or raw / "validation.json"
        scope = json.loads(scope_path.read_bytes()) if scope_path.is_file() else scope_manifest([])
        validation = json.loads(receipts_path.read_bytes()) if receipts_path.is_file() else {}
        reference = measure(args.inputs, args.output, args.date, scope)
        value = build(reference, validation)
        publication = publish_primary(
            args.output, value, lambda p: validate(p, Path(reference["path"]).parent / "terminal")
        )
        print(json.dumps(publication), flush=True)
        e.progress("complete", value["completed_count"], 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        e.progress("refused:" + str(error))
        return 1
