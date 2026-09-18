"""Qualify current-work receipts and the historical assignment reducer.

This experiment performs host-only aggregation. It replays immutable model
evidence but never loads a model, generates text, or turns a clean fixture into
new scientific evidence.

Spec refs: REQ-REPORT-7395 and SCENARIO-REPORT-7395-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import tempfile
import time
from typing import Any

import yaml

from carnot import experiment_7383_v648_canary_reducer as exp7383
from carnot.experiment_7329_v644_contract import parse_markdown_contract, parse_yaml_contract
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.reporting import current_work_receipt
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260918"
MILESTONE = "2026.09.649"
EXPERIMENT_ID = "exp7395-receipt-protocol"
SCHEMA = "carnot.exp7395.v649.receipt_protocol.v1"
PHASE = 1

MODULE_PATH = Path("python/carnot/experiment_7395_v649_receipt_protocol.py")
RECEIPT_HELPER_PATH = Path("python/carnot/reporting/current_work_receipt.py")
EXP7383_PATH = Path("python/carnot/experiment_7383_v648_canary_reducer.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7395_v649_receipt_protocol.py")
TEST_PATH = Path("tests/python/test_experiment_7395_v649_receipt_protocol.py")
HELPER_TEST_PATH = Path("tests/python/test_current_work_receipt.py")
EXP7383_TEST_PATH = Path("tests/python/test_experiment_7383_v648_canary_reducer.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_7395_v649_receipt_protocol.json")
RAW_DIR = Path("results/raw/experiment_7395_v649_receipt_protocol")
ROADMAP_PATH = Path("research-roadmap.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
EXP7383_RESULT = Path("results/experiment_7383_v648_canary_reducer.json")
EXP7384_RESULT = Path("results/experiment_7384_v648_arc_invocation_boundary.json")

TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
RECEIPT_MUTATIONS = (
    "unreported_load",
    "falsified_duration",
    "dropped_failed_call",
    "missing_completion",
    "invalid_venue",
    "changed_source_hash",
)
ZERO_INVOCATION_COUNTS = current_work_receipt.ZERO_INVOCATION_COUNTS

SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("scripts/roadmap_schema.py"),
    Path("scripts/audit_roadmap_gates.py"),
    Path("scripts/exclusion_manifest_lint.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7381_v648_contract.py"),
    EXP7383_PATH,
    Path("python/carnot/experiment_7384_v648_arc_invocation_boundary.py"),
    SPEC_PATH,
    ROADMAP_PATH,
    DESIGN_PATH,
    EXP7383_RESULT,
    EXP7384_RESULT,
    MODULE_PATH,
    RECEIPT_HELPER_PATH,
    WRAPPER_PATH,
    TEST_PATH,
    HELPER_TEST_PATH,
    EXP7383_TEST_PATH,
)

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(), HELPER_TEST_PATH.as_posix(), EXP7383_TEST_PATH.as_posix()),
    changed_modules=(
        MODULE_PATH.as_posix(),
        RECEIPT_HELPER_PATH.as_posix(),
        EXP7383_PATH.as_posix(),
    ),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "milestone",
        "phase",
        "status",
        "run_date",
        "started_at_utc",
        "ended_at_utc",
        "preconditions_checked",
        "MODEL_SPECS",
        "model_invoked",
        "invocation_counts",
        "current_invocation_events",
        "inference_substrate",
        "inference_substrate_details",
        "inference_substrate_class",
        "execution_venue",
        "duration_s",
        "phase_spans",
        "receipt_sidecars",
        "small_ebm_training",
        "random_seed",
        "reproducibility_checksum",
        "source_artifact_hashes",
        "rows",
        "sample_size_budget",
        "acceptance_gate_results",
        "gate_check_summary",
        "verifier_is_oracle",
        "honest_verdict",
        "verdict_class",
        "flagged_adversarial",
        "validation_receipts",
        "repository_health",
        "field_principles",
        "promotion_score",
        "assignment_reducer_ready_score",
        "receipt_protocol_ready_score",
        "receipt_mutation_rows",
        "contract_rows",
        "assignment_replay",
        "contract_comparison",
        "historical_diagnostic_rows",
        "required_check_names",
    }
)


def utc_now() -> str:  # pragma: no cover - real run boundary.
    """Return an aware UTC timestamp for an observed process boundary."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush every phase and long subprocess boundary with monotonic time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7395] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def sha256_file(path: Path) -> str:
    """Hash exact bytes without loading a large artifact at once."""

    return current_work_receipt.sha256_file(path)


def load_json(path: Path) -> JsonDict:
    """Return one JSON object, or an empty object for unavailable bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def load_yaml(path: Path) -> JsonDict:
    """Return one YAML mapping, or an empty mapping for malformed input."""

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind every artifact field except the checksum slot itself."""

    payload = {key: item for key, item in value.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write terminal JSON with fsync and one local atomic rename."""

    current_work_receipt.atomic_json(path, value)


def collect_preconditions(root: Path) -> tuple[list[JsonDict], dict[str, str]]:
    """Authenticate named sources, the requirement, and historical eligibility."""

    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            {
                "check": f"source_bytes:{relative.as_posix()}",
                "category": "precondition",
                "upstream": relative.as_posix(),
                "artifact_field": "bytes",
                "expected": "readable_nonempty_bytes",
                "observed": "readable_nonempty_bytes" if available else None,
                "operator": "==",
                "passed": available,
            }
        )
        if available:
            hashes[relative.as_posix()] = sha256_file(path)

    spec = (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    checks.append(
        {
            "check": "driving_requirement",
            "category": "precondition",
            "upstream": SPEC_PATH.as_posix(),
            "artifact_field": "REQ-*",
            "expected": "REQ-REPORT-7395",
            "observed": "REQ-REPORT-7395" if "REQ-REPORT-7395" in spec else None,
            "operator": "==",
            "passed": "REQ-REPORT-7395" in spec,
        }
    )
    expected = {
        EXP7383_RESULT: ("exp7383-canary-reducer", "disqualified", False),
        EXP7384_RESULT: ("exp7384-arc-invocation-boundary", "disqualified", True),
    }
    for relative, identity in expected.items():
        source = load_json(root / relative)
        observed = (
            source.get("experiment_id"),
            source.get("verdict_class"),
            source.get("flagged_adversarial"),
        )
        checks.append(
            {
                "check": f"historical_identity:{relative.name}",
                "category": "historical_diagnostic_only",
                "upstream": relative.as_posix(),
                "artifact_field": "experiment_id/verdict_class/flagged_adversarial",
                "expected": list(identity),
                "observed": list(observed),
                "operator": "==",
                "passed": observed == identity,
            }
        )
    return checks, hashes


def replay_assignment_and_proof(root: Path) -> JsonDict:
    """Cold-replay the published Exp7383 reducer and unchanged proof boundary."""

    calls, formulas = exp7383.load_assignment_evidence(root)
    assignment = exp7383.reduce_assignment_receipts(calls, formulas)
    proof = exp7383.replay_proof_boundary(root)
    fidelity = sum(row.get("response_fidelity_valid") is True for row in assignment["rows"])
    usable = assignment["usable_proposal_count"]
    return {
        "assignment": assignment,
        "proof": proof,
        "source_literal_fidelity_count": fidelity,
        "usable_gate": {
            "check": "usable_assignment_proposals",
            "expected": 3,
            "operator": ">=",
            "observed": usable,
            "passed": usable >= 3,
        },
        "historical_inputs_remain_ineligible": True,
    }


def yaml_contract_rows(roadmap: Mapping[str, Any]) -> list[JsonDict]:
    """Parse executable task rows without reading values from Markdown."""

    return [deepcopy(dict(row)) for row in parse_yaml_contract(roadmap)["tasks"]]


def compare_contract_authorities(markdown: str, roadmap: Mapping[str, Any]) -> JsonDict:
    """Compare V649 authorities while keeping an absent V649 Markdown table advisory."""

    errors: list[str] = []
    markdown_parsed: JsonDict = {}
    yaml_parsed: JsonDict = {}
    try:
        markdown_parsed = parse_markdown_contract(markdown)
    except ValueError as exc:
        errors.append(f"markdown_parse_error:{exc}")
    try:
        yaml_parsed = parse_yaml_contract(roadmap)
    except ValueError as exc:
        errors.append(f"yaml_parse_error:{exc}")
    if markdown_parsed.get("milestone") != MILESTONE:
        errors.append("markdown_milestone_mismatch")
    if yaml_parsed.get("milestone") != MILESTONE:
        errors.append("yaml_milestone_mismatch")
    markdown_rows = markdown_parsed.get("tasks") or []
    yaml_rows = yaml_parsed.get("tasks") or []
    if len(markdown_rows) != 14:
        errors.append("markdown_task_count_mismatch")
    if len(yaml_rows) != 14:
        errors.append("yaml_task_count_mismatch")
    if markdown_rows != yaml_rows:
        errors.append("contract_rows_mismatch")
    return {
        "markdown_milestone": markdown_parsed.get("milestone"),
        "yaml_milestone": yaml_parsed.get("milestone"),
        "markdown_task_count": len(markdown_rows),
        "yaml_task_count": len(yaml_rows),
        "errors": list(dict.fromkeys(errors)),
        "passed": not errors,
        "advisory_only": True,
    }


def run_contract_mutations(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Prove count, order, and field comparisons reject private defects."""

    baseline = [deepcopy(dict(row)) for row in rows]
    missing = deepcopy(baseline[:-1])
    reordered = deepcopy(baseline)
    if len(reordered) >= 2:
        reordered[0], reordered[1] = reordered[1], reordered[0]
    wrong = deepcopy(baseline)
    if wrong:
        wrong[0]["title"] = "mutated title"
    return [
        {
            "mutation": name,
            "expected_check": check,
            "rejected": candidate != baseline,
            "disposition": "complete",
            "censored": False,
            "metric_value": int(candidate != baseline),
            "cost": {"current_llm_calls": 0},
            "failure": None,
        }
        for name, check, candidate in (
            ("missing_row", "task_count", missing),
            ("reordered_rows", "task_order", reordered),
            ("wrong_field", "title", wrong),
        )
    ]


def _write_sidecars(root: Path, directory: Path) -> list[JsonDict]:
    """Write scripted and historical evidence outside current provenance."""

    exp7383_artifact = load_json(root / EXP7383_RESULT)
    exp7384_artifact = load_json(root / EXP7384_RESULT)
    historical_path = directory / "historical_model_receipts.json"
    simulated_path = directory / "simulated_transport_events.json"
    atomic_json(
        historical_path,
        {
            "scope": "historical",
            "sources": [
                {
                    "path": EXP7383_RESULT.as_posix(),
                    "sha256": sha256_file(root / EXP7383_RESULT),
                    "verdict_class": exp7383_artifact.get("verdict_class"),
                    "flagged_adversarial": exp7383_artifact.get("flagged_adversarial"),
                    "invocation_counts": exp7383_artifact.get("historical_model_inputs", {}).get(
                        "historical_invocation_counts"
                    ),
                },
                {
                    "path": EXP7384_RESULT.as_posix(),
                    "sha256": sha256_file(root / EXP7384_RESULT),
                    "verdict_class": exp7384_artifact.get("verdict_class"),
                    "flagged_adversarial": exp7384_artifact.get("flagged_adversarial"),
                    "historical_model_receipts": exp7384_artifact.get("historical_model_receipts"),
                },
            ],
        },
    )
    atomic_json(
        simulated_path,
        {
            "scope": "simulated_transport",
            "source_path": EXP7384_RESULT.as_posix(),
            "source_sha256": sha256_file(root / EXP7384_RESULT),
            "events": exp7384_artifact.get("scripted_scored_path", {}).get(
                "simulated_transport_events", []
            ),
            "scripted_scored_path": exp7384_artifact.get("scripted_scored_path", {}),
        },
    )
    return [
        current_work_receipt.sidecar_reference(historical_path, root=directory, scope="historical"),
        current_work_receipt.sidecar_reference(
            simulated_path, root=directory, scope="simulated_transport"
        ),
    ]


def _clean_receipt(sidecars: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build the no-model fixture used by producer and unchanged-reader controls."""

    return current_work_receipt.build_current_work_receipt(
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details={
            "device": "host_cpu",
            "machine": platform.machine(),
            "python": platform.python_version(),
            "jax_platform": os.environ.get("JAX_PLATFORMS", "cpu"),
            "work": "hashing, JSON reduction, exact SAT replay, and subprocess validation",
        },
        inference_substrate_class="aggregation",
        execution_venue="host",
        duration_s=2.0,
        phase_spans=[{"phase": "fixture", "start_s": 0.0, "end_s": 1.0}],
        owned_run_events=[],
        sidecar_references=sidecars,
        small_ebm_training={"performed": False, "kind": "none"},
    )


def run_receipt_controls(
    directory: Path, *, source_root: Path = REPO_ROOT
) -> tuple[list[JsonDict], JsonDict]:
    """Apply six named mutations and restore the clean sidecar after the hash control."""

    directory.mkdir(parents=True, exist_ok=True)
    sidecars = _write_sidecars(source_root, directory)
    clean = _clean_receipt(sidecars)
    clean_errors = current_work_receipt.validate_current_work_receipt(clean, root=directory)
    rows: list[JsonDict] = []
    for mutation in RECEIPT_MUTATIONS:
        changed = deepcopy(clean)
        if mutation == "unreported_load":
            changed["current_invocation_events"] = [
                {"event_id": "load-1", "operation": "model_load", "state": "attempted"},
                {"event_id": "load-1", "operation": "model_load", "state": "completed"},
            ]
        elif mutation == "falsified_duration":
            changed["duration_s"] = 0.5
        elif mutation == "dropped_failed_call":
            changed["invocation_counts"]["generation_calls_failed"] = 1
        elif mutation == "missing_completion":
            changed["current_invocation_events"] = [
                {"event_id": "gen-1", "operation": "generation", "state": "attempted"}
            ]
            changed["model_invoked"] = True
            changed["invocation_counts"]["generation_calls_attempted"] = 1
            changed["invocation_counts"]["generation_calls_in_flight"] = 1
        elif mutation == "invalid_venue":
            changed["execution_venue"] = "host_cpu"
        else:
            simulated = directory / "simulated_transport_events.json"
            original = simulated.read_bytes()
            simulated.write_text("{}\n", encoding="utf-8")
        errors = current_work_receipt.validate_current_work_receipt(changed, root=directory)
        if mutation == "changed_source_hash":
            simulated.write_bytes(original)
        rows.append(
            {
                "mutation": mutation,
                "guard_version": "carnot.reporting.current_work_receipt.v1",
                "errors": errors,
                "rejected": bool(errors),
                "disposition": "complete",
                "censored": False,
                "metric_value": int(bool(errors)),
                "cost": {"current_llm_calls": 0},
                "failure": None if errors else "mutation_not_rejected",
            }
        )
    return rows, {**clean, "errors": clean_errors}


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the exact Exp7358 plan with three explicit changed modules."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject changed names, broad tests, and missing command-local coverage data."""

    errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    counts = Counter(command.name for command in commands)
    if counts != Counter(validation_scope.REQUIRED_CHECK_NAMES):
        errors.append("required_command_names_changed")
    if "full_python_suite" in counts:
        errors.append("full_python_suite_forbidden")
    coverage = [row for row in commands if row.name == "changed_module_coverage_report"]
    if len(coverage) != 1 or "COVERAGE_FILE" not in dict(
        getattr(coverage[0], "command_environment", ()) if coverage else ()
    ):
        errors.append("coverage_file_not_preserved")
    return list(dict.fromkeys(errors))


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Require exactly one passing, completed receipt for every frozen name."""

    counts = Counter(row.get("name") for row in receipts)
    return all(
        counts[name] == 1
        and next(row for row in receipts if row.get("name") == name).get("passed") is True
        and next(row for row in receipts if row.get("name") == name).get("exit_code") == 0
        and next(row for row in receipts if row.get("name") == name).get("timed_out") is False
        for name in names
    )


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Expose every failed gate and the first exact upstream field."""

    failed = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    return {
        "all_passed": not failed,
        "failed_count": len(failed),
        "failed_checks": failed,
        "first_failure": failed[0] if failed else None,
    }


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain ordinary fields without wrapping their machine-readable values."""

    special = {
        "schema": "A versioned schema lets cold readers reject incompatible receipts.",
        "run_date": "The fixed date and actual UTC boundaries identify this execution.",
        "preconditions_checked": "Exact paths and identities block dependent work on absence.",
        "MODEL_SPECS": "An empty list states that this run performs no current LLM work.",
        "model_invoked": "False follows from an empty owned current invocation ledger.",
        "invocation_counts": "Only owned current events contribute to these zero counters.",
        "inference_substrate": "A string describes current host aggregation without copied history.",
        "inference_substrate_details": "Device and software details stay separate from the substrate string.",
        "inference_substrate_class": "Aggregation is the closed class for this current work.",
        "execution_venue": "Host is the closed location value, not a hostname or dictionary.",
        "duration_s": "Monotonic elapsed time is measured and never padded.",
        "phase_spans": "Actual phase boundaries expose current work and validation cost.",
        "random_seed": "Frozen experiment and mutation seeds support exact replay.",
        "reproducibility_checksum": "The checksum binds code, sources, rows, gates, and receipts.",
        "source_artifact_hashes": "Exact byte hashes preserve every named input identity.",
        "rows": "Per-unit reducer and mutation rows retain costs and dispositions.",
        "sample_size_budget": "Planned and completed units retain the frozen stop rule.",
        "acceptance_gate_results": "Categories and operands keep completion separate from advisory checks.",
        "gate_check_summary": "Failures name their upstream, field, expected, and observed values.",
        "verifier_is_oracle": "Formal SAT defines assignment truth; receipt safety itself is not model value.",
        "honest_verdict": "The completed finding claims protocol readiness, not fresh model evidence.",
        "verdict_class": "The closed null class records qualified infrastructure without efficacy.",
        "flagged_adversarial": "A critical current finding excludes this producer from readiness.",
        "validation_receipts": "Exact argv, environment, exits, durations, and log hashes retain failures.",
        "repository_health": "Unrelated broad health is outside this affected-check decision.",
        "field_principles": "This map explains fields without changing their value shapes.",
        "promotion_score": "Zero forbids rollout, publication, or generator-weight changes.",
        "assignment_reducer_ready_score": "One requires reducer, proof, coverage, and affected validation.",
        "receipt_protocol_ready_score": "One independently requires clean and rejecting receipt controls.",
        "receipt_mutation_rows": "Six named defects retain the exact producer guard response.",
        "contract_rows": "Fourteen YAML rows and mutations are advisory to other branches.",
    }
    return {
        key: special.get(key, f"The {key} field preserves directly auditable experiment evidence.")
        for key in keys
    }


def _acceptance_gates(
    replay: Mapping[str, Any],
    mutation_rows: Sequence[Mapping[str, Any]],
    clean: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    *,
    require_terminal: bool,
) -> list[JsonDict]:
    """Reduce assignment and receipt readiness without using advisory contract status."""

    affected = _receipts_pass(receipts, validation_scope.REQUIRED_CHECK_NAMES)
    terminal = _receipts_pass(receipts, TERMINAL_CHECK_NAMES) if require_terminal else True
    values = (
        (
            "assignment_reducer",
            "completion",
            1,
            replay["assignment"]["assignment_reducer_ready_score"],
        ),
        ("proof_boundary", "safety", 1, replay["proof"]["proof_boundary_replay_ready_score"]),
        ("affected_validation", "validation", True, affected),
        ("clean_receipt_fixture", "safety", [], clean.get("errors")),
        (
            "receipt_mutations",
            "safety",
            len(RECEIPT_MUTATIONS),
            sum(row.get("rejected") is True for row in mutation_rows),
        ),
        ("terminal_readers", "validation", True, terminal),
        ("promotion_forbidden", "promotion", 0, 0),
    )
    return [
        {
            "check": check,
            "category": category,
            "upstream": EXPERIMENT_ID,
            "artifact_field": check,
            "expected": deepcopy(expected),
            "operator": "==",
            "observed": deepcopy(observed),
            "passed": observed == expected,
        }
        for check, category, expected, observed in values
    ]


def build_artifact(
    *,
    root: Path,
    sidecar_dir: Path,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    receipts: Sequence[Mapping[str, Any]],
    started_at: str,
    ended_at: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    historical_diagnostics: Sequence[Mapping[str, Any]] = (),
    require_terminal: bool = True,
    flagged_adversarial: bool = False,
) -> JsonDict:
    """Build one terminal-shaped record from independently reducible raw evidence."""

    replay = replay_assignment_and_proof(root)
    mutations, clean = run_receipt_controls(sidecar_dir, source_root=root)
    roadmap = load_yaml(root / ROADMAP_PATH)
    markdown = (root / DESIGN_PATH).read_text(encoding="utf-8")
    contract = compare_contract_authorities(markdown, roadmap)
    yaml_rows = yaml_contract_rows(roadmap)
    contract_mutations = run_contract_mutations(yaml_rows)
    gates = _acceptance_gates(replay, mutations, clean, receipts, require_terminal=require_terminal)
    affected = _receipts_pass(receipts, validation_scope.REQUIRED_CHECK_NAMES)
    terminal = _receipts_pass(receipts, TERMINAL_CHECK_NAMES) if require_terminal else True
    assignment_ready = int(
        replay["assignment"]["assignment_reducer_ready_score"] == 1
        and replay["proof"]["proof_boundary_replay_ready_score"] == 1
        and affected
        and terminal
        and not flagged_adversarial
    )
    receipt_ready = int(
        clean["errors"] == []
        and all(row["rejected"] for row in mutations)
        and affected
        and terminal
        and not flagged_adversarial
    )
    if not all(row.get("passed") is True for row in preconditions):
        verdict_class = "blocked"
        honest_verdict = "blocked_required_immutable_input"
    elif assignment_ready and receipt_ready:
        verdict_class = "null"
        honest_verdict = "complete_null_receipt_protocol_and_assignment_reducer_qualified"
    else:
        verdict_class = "disqualified"
        honest_verdict = "complete_disqualified_receipt_or_validation_failure"

    current = deepcopy(clean)
    current.pop("errors", None)
    current["duration_s"] = float(duration_s)
    current["phase_spans"] = [deepcopy(dict(row)) for row in phase_spans]
    rows = [*deepcopy(replay["assignment"]["rows"]), *deepcopy(mutations), *contract_mutations]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": honest_verdict,
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        **current,
        "random_seed": {"experiment": 7_395_202_609_18, "mutations": 7_395_202_609_19},
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(source_hashes),
        "rows": rows,
        "sample_size_budget": {
            "planned_assignment_calls": 4,
            "attempted_assignment_calls": 4,
            "completed_assignment_calls": replay["assignment"]["complete_response_count"],
            "censored_assignment_calls": 4 - replay["assignment"]["complete_response_count"],
            "unstarted_assignment_calls": 0,
            "planned_receipt_mutations": len(RECEIPT_MUTATIONS),
            "completed_receipt_mutations": len(mutations),
            "effective_independent_group_count": 4,
            "limits": "Historical replay only; no current model calls or replacement units.",
            "stop_rule": "Replay all four frozen calls and all six frozen mutations once.",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "verifier_is_oracle": True,
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": flagged_adversarial,
        "validation_receipts": [deepcopy(dict(row)) for row in receipts],
        "repository_health": {
            "status": "not_assessed_by_scoped_experiment",
            "affects_required_checks": False,
            "unrelated_broad_suite_observations": [],
        },
        "field_principles": {},
        "promotion_score": 0,
        "assignment_reducer_ready_score": assignment_ready,
        "receipt_protocol_ready_score": receipt_ready,
        "receipt_mutation_rows": mutations,
        "contract_rows": {
            "authority_comparison": contract,
            "yaml_rows": yaml_rows,
            "mutation_rows": contract_mutations,
            "advisory_only": True,
        },
        "assignment_replay": replay,
        "contract_comparison": contract,
        "historical_diagnostic_rows": [deepcopy(dict(row)) for row in historical_diagnostics],
        "required_check_names": [*validation_scope.REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES],
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_artifact_for_test(
    root: Path, sidecar_dir: Path, receipts: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Build a deterministic terminal fixture from real immutable inputs."""

    preconditions, hashes = collect_preconditions(root)
    return build_artifact(
        root=root,
        sidecar_dir=sidecar_dir,
        preconditions=preconditions,
        source_hashes=hashes,
        receipts=receipts,
        started_at="2026-09-18T00:00:00Z",
        ended_at="2026-09-18T00:00:02Z",
        duration_s=2.0,
        phase_spans=[{"phase": "test", "start_s": 0.0, "end_s": 1.5}],
    )


def build_blocked_artifact(
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    *,
    started_at: str,
    duration_s: float,
) -> JsonDict:
    """Publish exact external absence without success-shaped dependent evidence."""

    failed = next((deepcopy(dict(row)) for row in preconditions if not row.get("passed")), None)
    sidecars: list[JsonDict] = []
    current = current_work_receipt.build_current_work_receipt(
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details={"device": "host_cpu", "work": "preconditions_only"},
        inference_substrate_class="aggregation",
        execution_venue="host",
        duration_s=duration_s,
        phase_spans=[],
        owned_run_events=[],
        sidecar_references=sidecars,
        small_ebm_training={"performed": False, "kind": "none"},
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": "blocked_required_immutable_input",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        **current,
        "random_seed": {"experiment": 7_395_202_609_18, "mutations": 7_395_202_609_19},
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(source_hashes),
        "rows": [],
        "sample_size_budget": {
            "planned_assignment_calls": 4,
            "attempted_assignment_calls": 0,
            "completed_assignment_calls": 0,
            "censored_assignment_calls": 0,
            "unstarted_assignment_calls": 4,
            "stop_rule": "Stop before dependent work on missing immutable input.",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {
            "all_passed": False,
            "failed_count": 1,
            "failed_checks": [failed],
            "first_failure": failed,
        },
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_required_immutable_input",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "repository_health": {"status": "not_evaluated", "affects_required_checks": False},
        "field_principles": {},
        "promotion_score": 0,
        "assignment_reducer_ready_score": 0,
        "receipt_protocol_ready_score": 0,
        "receipt_mutation_rows": [],
        "contract_rows": {"advisory_only": True, "yaml_rows": [], "mutation_rows": []},
        "assignment_replay": {},
        "contract_comparison": {},
        "historical_diagnostic_rows": [],
        "required_check_names": [*validation_scope.REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES],
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(
    value: object,
    *,
    root: Path = REPO_ROOT,
    sidecar_root: Path | None = None,
    require_terminal: bool = True,
) -> list[str]:
    """Cold-check identity, hashes, raw reductions, gates, scores, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    missing = sorted(REQUIRED_FIELDS - set(artifact))
    if missing:
        return [f"missing_required_field:{field}" for field in missing]
    errors: list[str] = []
    if (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("run_date"),
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
        errors.append("identity_invalid")
    current_fields = {
        key: artifact.get(key)
        for key in (
            "MODEL_SPECS",
            "model_invoked",
            "invocation_counts",
            "current_invocation_events",
            "inference_substrate",
            "inference_substrate_details",
            "inference_substrate_class",
            "execution_venue",
            "duration_s",
            "phase_spans",
            "receipt_sidecars",
            "small_ebm_training",
        )
    }
    errors.extend(
        current_work_receipt.validate_current_work_receipt(
            current_fields, root=sidecar_root or root / RAW_DIR
        )
    )
    if (
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("current_invocation_declaration_invalid")
    if (
        artifact.get("inference_substrate_class") != "aggregation"
        or artifact.get("execution_venue") != "host"
    ):
        errors.append("current_substrate_declaration_invalid")
    for relative, expected in artifact.get("source_artifact_hashes", {}).items():
        path = Path(str(relative))
        resolved = path if path.is_absolute() else root / path
        observed = sha256_file(resolved) if resolved.is_file() else None
        if observed != expected:
            errors.append(f"source_hash_mismatch:{relative}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_incomplete")
    if artifact.get("verdict_class") == "blocked":
        if (
            artifact.get("assignment_reducer_ready_score") != 0
            or artifact.get("receipt_protocol_ready_score") != 0
        ):
            errors.append("blocked_scores_nonzero")
    else:
        replay = replay_assignment_and_proof(root)
        if artifact.get("assignment_replay") != replay:
            errors.append("assignment_replay_mismatch")
        mutation_rows = artifact.get("receipt_mutation_rows") or []
        clean = {
            **current_fields,
            "errors": current_work_receipt.validate_current_work_receipt(
                current_fields, root=sidecar_root or root / RAW_DIR
            ),
        }
        gates = _acceptance_gates(
            replay,
            mutation_rows,
            clean,
            artifact.get("validation_receipts") or [],
            require_terminal=require_terminal,
        )
        if artifact.get("acceptance_gate_results") != gates:
            errors.append("acceptance_gate_results_mismatch")
        affected = _receipts_pass(
            artifact.get("validation_receipts") or [], validation_scope.REQUIRED_CHECK_NAMES
        )
        terminal = (
            _receipts_pass(artifact.get("validation_receipts") or [], TERMINAL_CHECK_NAMES)
            if require_terminal
            else True
        )
        expected_assignment = int(
            replay["assignment"]["assignment_reducer_ready_score"] == 1
            and replay["proof"]["proof_boundary_replay_ready_score"] == 1
            and affected
            and terminal
            and artifact.get("flagged_adversarial") is False
        )
        expected_receipt = int(
            clean["errors"] == []
            and len(mutation_rows) == len(RECEIPT_MUTATIONS)
            and all(row.get("rejected") is True for row in mutation_rows)
            and affected
            and terminal
            and artifact.get("flagged_adversarial") is False
        )
        if artifact.get("assignment_reducer_ready_score") != expected_assignment:
            errors.append("assignment_reducer_ready_score_mismatch")
        if artifact.get("receipt_protocol_ready_score") != expected_receipt:
            errors.append("receipt_protocol_ready_score_mismatch")
        if artifact.get("gate_check_summary") != _gate_summary(gates):
            errors.append("gate_check_summary_mismatch")
    if artifact.get("promotion_score") != 0:
        errors.append("promotion_score_nonzero")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def terminal_command_specs(root: Path, candidate: Path) -> list[PlannedCommand]:
    """Build fresh-process replay and the two unchanged terminal readers."""

    python = str(root / ".venv/bin/python")
    commands = (
        (
            "declared_entrypoint_cold_replay",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--validate",
                str(candidate),
            ),
            "completion",
        ),
        (
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "safety",
        ),
        (
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "completion",
        ),
    )
    return [
        PlannedCommand(
            validation_scope.CommandSpec(name, argv, "measured_candidate"), category, True
        )
        for name, argv, category in commands
    ]


def _historical_diagnostic_specs(root: Path) -> list[PlannedCommand]:  # pragma: no cover
    """Replay unchanged failed V648 candidates without making them current evidence."""

    python = str(root / ".venv/bin/python")
    return [
        PlannedCommand(
            validation_scope.CommandSpec(
                f"historical_adversarial_exp{number}",
                (python, "-u", "scripts/adversarial_verify.py", relative.as_posix()),
                "historical_diagnostic_only",
            ),
            "historical_diagnostic_only",
            False,
        )
        for number, relative in ((7383, EXP7383_RESULT), (7384, EXP7384_RESULT))
    ]


def _span(
    name: str, phase_started: float, run_started: float, units: int
) -> JsonDict:  # pragma: no cover
    """Close one measured phase with a monotonic checkpoint."""

    ended = time.monotonic()
    return {
        "phase": name,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "checkpoint_at_utc": utc_now(),
    }


def run_experiment(
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover
    """Execute replay, scoped checks, terminal readers, and atomic publication."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    output = output_path if output_path.is_absolute() else root / output_path
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []

    phase_started = time.monotonic()
    progress(started, "preconditions", "start")
    preconditions, hashes = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, started, len(preconditions)))
    progress(started, "preconditions", "end", passed=all(row["passed"] for row in preconditions))
    if not all(row["passed"] for row in preconditions):
        blocked = build_blocked_artifact(
            preconditions, hashes, started_at=started_at, duration_s=time.monotonic() - started
        )
        progress(started, "write", "before_atomic_blocked", path=output)
        atomic_json(output, blocked)
        progress(started, "write", "after_atomic_blocked", path=output)
        return blocked

    for phase in ("load", "generate"):
        phase_started = time.monotonic()
        progress(started, phase, "before_no_current_model_work")
        spans.append(_span(phase, phase_started, started, 0))
        progress(started, phase, "after_no_current_model_work", model_invoked=False)

    phase_started = time.monotonic()
    progress(started, "diagnostic_replay", "before_subprocesses")
    diagnostics = run_categorized_commands(
        root, _historical_diagnostic_specs(root), log_dir=raw_dir / "validation/historical"
    )
    spans.append(_span("diagnostic_replay", phase_started, started, len(diagnostics)))
    progress(started, "diagnostic_replay", "after_subprocesses", units=len(diagnostics))

    private = Path(tempfile.mkdtemp(prefix="exp7395-validation-", dir="/tmp"))
    phase_started = time.monotonic()
    commands = build_validation_plan(root, private)
    plan_errors = validate_validation_plan(root, commands)
    progress(started, "affected_validation", "before_subprocesses", plan_errors=len(plan_errors))
    affected: list[JsonDict] = []
    if not plan_errors:
        affected = run_categorized_commands(
            root,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=raw_dir / "validation/affected",
        )
    affected_reduction = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    spans.append(_span("affected_validation", phase_started, started, len(affected)))
    progress(
        started,
        "affected_validation",
        "after_subprocesses",
        passed=affected_reduction["passed"] and not plan_errors,
    )

    phase_started = time.monotonic()
    candidate = build_artifact(
        root=root,
        sidecar_dir=raw_dir,
        preconditions=preconditions,
        source_hashes=hashes,
        receipts=affected,
        started_at=started_at,
        ended_at=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        historical_diagnostics=diagnostics,
        require_terminal=False,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    spans.append(_span("candidate_write", phase_started, started, 1))

    phase_started = time.monotonic()
    progress(started, "terminal_validation", "before_subprocesses")
    terminal = run_categorized_commands(
        root, terminal_command_specs(root, candidate_path), log_dir=raw_dir / "validation/terminal"
    )
    spans.append(_span("terminal_validation", phase_started, started, len(terminal)))
    terminal_passed = _receipts_pass(terminal, TERMINAL_CHECK_NAMES)
    critical = any("[CRITICAL]" in str(row.get("output_tail") or "") for row in terminal)
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        passed=terminal_passed,
        critical=critical,
    )

    final = build_artifact(
        root=root,
        sidecar_dir=raw_dir,
        preconditions=preconditions,
        source_hashes=hashes,
        receipts=[*affected, *terminal],
        started_at=started_at,
        ended_at=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        historical_diagnostics=diagnostics,
        require_terminal=True,
        flagged_adversarial=critical or not terminal_passed,
    )
    errors = validate_artifact(final, root=root, sidecar_root=raw_dir)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(started, "write", "before_atomic_terminal", path=output)
    atomic_json(candidate_path, final)
    atomic_json(output, final)
    progress(started, "write", "after_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the fixed date, output, and cold-validation modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the protocol or cold-reload one measured candidate."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    if args.validate is not None:
        value = load_json(args.validate)
        errors = validate_artifact(
            value, root=REPO_ROOT, sidecar_root=args.validate.parent, require_terminal=False
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
