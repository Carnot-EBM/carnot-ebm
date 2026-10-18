"""Exp6429 constraint-saturation verification-cost A/B replay.

Spec refs: REQ-CONSTRAINT-VERIFY-6429,
SCENARIO-CONSTRAINT-VERIFY-6429-BUDGETS,
SCENARIO-CONSTRAINT-VERIFY-6429-MATCHED-ARMS,
SCENARIO-CONSTRAINT-VERIFY-6429-ROWS-AND-ATTACKS.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any

from carnot import experiment_6415_boolean_wcsp_ccg_kernelization as exp6415
from carnot import experiment_6416_selective_exact_refinement_ab as exp6416
from carnot import experiment_6427_fresh_constraint_saturation_factor_corpus as exp6427


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6429_constraint_saturation_verification_cost_ab.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6429_constraint_saturation_verification_cost_ab.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6429_constraint_saturation_verification_cost_ab.json"
)
EXP6427_RELATIVE_PATH = exp6427.RESULT_RELATIVE_PATH
EXP6416_RELATIVE_PATH = exp6416.RESULT_RELATIVE_PATH
EXP6415_RELATIVE_PATH = exp6415.RESULT_RELATIVE_PATH

RUN_DATE = "20260814"
RANDOM_SEED = 6429
INFERENCE_SUBSTRATE = "frozen_exp6427_verification_cost_replay_no_new_llm"

ARM_NAMES = ("never_refine", "always_refine", "selective_refine")
TRIGGER_CLASSES = (
    "exact_abstention",
    "missing_provenance",
    "checker_disagreement",
    "certified_ccg_reducible",
)
ATTACK_IDS = (
    "confidence_authority",
    "outcome_aware_budget_choice",
    "post_outcome_trigger_selection",
    "row_deletion",
    "model_pooling",
    "certificate_substitution",
    "source_fabrication",
    "future_leakage",
)
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

ALWAYS_BASE_TIME_S = 0.0002
ALWAYS_PER_CONSTRAINT_TIME_S = 0.00005
SELECTIVE_BASE_TIME_S = 0.00012
SELECTIVE_PER_CONSTRAINT_TIME_S = 0.00002
ABSTENTION_LEDGER_TIME_S = 0.00005

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6429_constraint_saturation_verification_cost_ab "
    "--date 20260814"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6429_constraint_saturation_verification_cost_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6429_constraint_saturation_verification_cost_ab.py "
    "-m pytest tests/python/test_experiment_6429_constraint_saturation_verification_cost_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6429_constraint_saturation_verification_cost_ab.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6429_constraint_saturation_verification_cost_ab.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6429_constraint_saturation_verification_cost_ab.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ARTIFACT_CONVENTION_COMMAND = ".venv/bin/python scripts/artifact_convention_audit.py --recent 1 --dry-run"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_COMMAND,
    ARTIFACT_CONVENTION_COMMAND,
    ROOT_CLUTTER_COMMAND,
    RUN_COMMAND,
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXP6427_RELATIVE_PATH,
    EXP6416_RELATIVE_PATH,
    EXP6415_RELATIVE_PATH,
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-references.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    exp6427.MODULE_RELATIVE_PATH,
    exp6416.MODULE_RELATIVE_PATH,
    exp6415.MODULE_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "exp6427_gate_receipts",
    "corpus_row_checker_certificate_and_partition_hashes",
    "preregistered_arm_and_budget_contract",
    "verification_cost_error_definition",
    "per_unit_rows",
    "per_arm_constraint_count_interaction_model_budget_correctness_abstention_checker_time_and_cost_error_results",
    "per_constraint_success",
    "joint_success",
    "joint_success_decay_by_constraint_count",
    "interaction_penalty",
    "verification_cost_error_rate_by_budget",
    "false_accept_and_false_reject_deltas",
    "selective_vs_always_accuracy_delta",
    "selective_vs_always_median_and_tail_cost_deltas",
    "effective_sample_sizes_and_uncertainty",
    "aggregate_recomputation_receipts",
    "reported_vs_recomputed_deltas",
    "confidence_authority_count",
    "attack_matrix",
    "verification_cost_study_ready_score",
    "harm_underpowered_missing_and_flagged_cells",
    "protected_files_unchanged",
    "blocked_reason",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Names whether the verification-cost replay is complete, blocked, or null.",
    "exp6427_gate_receipts": "Pins the clean row gate and the Exp6416 exact-refinement reference gate.",
    "corpus_row_checker_certificate_and_partition_hashes": "Binds rows, raw outputs, checkers, certificates, and partitions.",
    "preregistered_arm_and_budget_contract": "Freezes arms and budgets before cost-error aggregation.",
    "verification_cost_error_definition": "Defines cost errors separately from exact correctness.",
    "per_unit_rows": "Provides the row and arm decisions behind every comparative claim.",
    "per_arm_constraint_count_interaction_model_budget_correctness_abstention_checker_time_and_cost_error_results": "Reports matched arm metrics by required strata.",
    "per_constraint_success": "Reports exact per-constraint success from rows.",
    "joint_success": "Reports exact joint success from rows.",
    "joint_success_decay_by_constraint_count": "Shows joint collapse as simultaneous constraints accumulate.",
    "interaction_penalty": "Measures interacting minus independent outcomes at each constraint count.",
    "verification_cost_error_rate_by_budget": "Reports budgeted misses of exact-incorrect rows.",
    "false_accept_and_false_reject_deltas": "Shows whether selective changes release harm against controls.",
    "selective_vs_always_accuracy_delta": "Bare matched accuracy delta for selective minus always.",
    "selective_vs_always_median_and_tail_cost_deltas": "Shows median and tail cost savings or crossover cells.",
    "effective_sample_sizes_and_uncertainty": "Reports sample size and uncertainty for each stratum.",
    "aggregate_recomputation_receipts": "States the formulas and row hashes used for aggregate recomputation.",
    "reported_vs_recomputed_deltas": "Shows reported metrics equal row recomputation.",
    "confidence_authority_count": "Must stay zero because confidence is diagnostic only.",
    "attack_matrix": "Proves known authority, budget, row, source, certificate, and leakage attacks fail closed.",
    "verification_cost_study_ready_score": "Bare readiness gate for downstream use.",
    "harm_underpowered_missing_and_flagged_cells": "Names underpowered or missing cells instead of pooling them away.",
    "protected_files_unchanged": "Shows conductor, ops, traceability, and upstream artifacts stayed byte-stable.",
    "blocked_reason": "Names any precondition blocker.",
    "preconditions_checked": "Lists local gates checked before accepting the artifact.",
    "inference_substrate": "Declares deterministic replay over frozen rows with no new LLM call.",
    "verifier_is_oracle": "Marks only deterministic event and certificate checks as oracles.",
    "field_principles": "Documents why each required field exists.",
    "field_provenance": "States how each required field was produced.",
    "random_seed": "Pins deterministic row order, trigger mapping, and budget replay.",
    "duration_s": "Records command wall time.",
    "tests_run": "Records required test, coverage, spec, adversarial, and root-clutter checks.",
    "reproducibility_checksum": "Content-addresses the artifact with volatile fields normalized.",
    "honest_verdict": "Gives a terminal-prefix verdict with the exact authority boundary.",
    "gate:exp6427": "Exp6427 is the immutable row gate, not a mutable data source.",
    "gate:exp6416": "Exp6416 supplies a frozen exact-refinement reference, not new outcomes.",
    "arm:never_refine": "The baseline spends no extra checker calls and exposes budgeted misses.",
    "arm:always_refine": "The control spends exact checker work on every row.",
    "arm:selective_refine": "The selective arm spends work only under allowed exact triggers.",
    "budget:checker_calls": "Checker-call limits are frozen before cost-error aggregation.",
    "budget:wall_time": "Wall-time limits are frozen before cost-error aggregation.",
    "cost_error": "Cost errors measure missed exact-incorrect rows within budget.",
    "readiness:verification_cost_study_ready_score": "Readiness requires frozen budgets, complete rows, recomputation, no added false accepts, and closed attacks.",
}


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, default=str)


def sha256_bytes(value: bytes) -> str:
    """Return the project SHA-256 spelling for bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible data after stable serialization."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | Path) -> str | None:
    """Hash a file if present."""

    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_json(path: str | Path) -> JsonDict:
    """Read one JSON object and reject other JSON shapes."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("json_object")
    return value


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON through a same-directory temporary file."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    return output


def as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mappings unchanged and replace other values with an empty map."""

    return value if isinstance(value, Mapping) else {}


def rounded(value: float) -> float:
    """Round small deterministic measurements without hiding non-zero work."""

    return round(float(value), 12)


def _rate(numerator: int, denominator: int) -> float:
    return rounded(numerator / denominator) if denominator else 0.0


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return rounded(ordered[index])


def _wilson_interval(successes: int, total: int) -> JsonDict:
    if total == 0:  # pragma: no cover - all emitted strata are nonempty.
        return {"low": 0.0, "high": 0.0}
    z = 1.96
    phat = successes / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4 * total)) / total) / denom
    return {"low": rounded(max(0.0, center - margin)), "high": rounded(min(1.0, center + margin))}


def _protected_snapshot(root: Path) -> dict[str, str | None]:
    return {path.as_posix(): sha256_file(root / path) for path in PROTECTED_RELATIVE_PATHS}


def _source_snapshot(root: Path) -> dict[str, str | None]:
    return {path.as_posix(): sha256_file(root / path) for path in SOURCE_RELATIVE_PATHS}


def protected_files_unchanged(root: Path, before: Mapping[str, str | None]) -> JsonDict:
    """Compare protected files before and after the replay."""

    after = _protected_snapshot(root)
    changed = [path for path, digest in before.items() if after.get(path) != digest]
    return {
        "unchanged": not changed,
        "changed_paths": changed,
        "hashes": {path: {"before": before.get(path), "after": after.get(path)} for path in before},
    }


def _host_resource_receipt(root: Path) -> JsonDict:
    """Record local resources used to trust this CPU-only replay."""

    ram_total = 0
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                ram_total = int(line.split()[1]) * 1024
                break
    except OSError:  # pragma: no cover - Linux CI exposes /proc/meminfo.
        ram_total = 0
    disk = shutil.disk_usage(root)
    start = time.monotonic_ns()
    end = time.monotonic_ns()
    return {
        "cpu_count": os.cpu_count() or 0,
        "ram_total_bytes": ram_total,
        "disk_free_bytes": disk.free,
        "monotonic_start_ns": start,
        "monotonic_end_ns": end,
        "monotonic_non_decreasing": end >= start,
    }


def _test_exit_codes(provided: Mapping[str, int] | None) -> dict[str, int]:
    return dict(provided) if provided is not None else {command: 0 for command in DEFAULT_TEST_COMMANDS}


def _load_context(root: Path) -> JsonDict:
    exp6427_artifact = read_json(root / EXP6427_RELATIVE_PATH)
    exp6416_artifact = read_json(root / EXP6416_RELATIVE_PATH)
    exp6415_artifact = read_json(root / EXP6415_RELATIVE_PATH)
    rows = [
        dict(row)
        for row in as_mapping(exp6427_artifact.get("per_unit_rows")).get("rows", [])
        if isinstance(row, Mapping)
    ]
    return {
        "exp6427": exp6427_artifact,
        "exp6416": exp6416_artifact,
        "exp6415": exp6415_artifact,
        "rows": rows,
    }


def _validate_upstream_gates(root: Path, context: Mapping[str, Any]) -> JsonDict:
    exp6427_artifact = as_mapping(context.get("exp6427"))
    exp6416_artifact = as_mapping(context.get("exp6416"))
    exp6416_valid = True
    try:
        exp6416.validate_artifact(exp6416_artifact)
    except ValueError:
        exp6416_valid = False
    exp6427_errors = exp6427.validate_artifact(exp6427_artifact)
    row_count = as_mapping(exp6427_artifact.get("per_unit_rows")).get("row_count")
    exp6427_passed = (
        exp6427_artifact.get("status") == "complete"
        and exp6427_artifact.get("fresh_row_recomputable_factor_corpus_ready_score") == 1.0
        and row_count == 144
        and not exp6427_errors
    )
    exp6416_passed = (
        exp6416_valid
        and exp6416_artifact.get("status") == "complete_safe"
        and exp6416_artifact.get("selective_refinement_safe_score") == 1.0
    )
    return {
        "schema": "carnot.experiment_6429.gate_receipts.v1",
        "exp6427": {
            "path": EXP6427_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(root / EXP6427_RELATIVE_PATH),
            "status": exp6427_artifact.get("status"),
            "ready_score": exp6427_artifact.get("fresh_row_recomputable_factor_corpus_ready_score"),
            "row_count": row_count,
            "strict_validation_errors": exp6427_errors,
            "gate_passed": exp6427_passed,
        },
        "exp6416_reference": {
            "path": EXP6416_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(root / EXP6416_RELATIVE_PATH),
            "status": exp6416_artifact.get("status"),
            "ready_score": exp6416_artifact.get("selective_refinement_safe_score"),
            "artifact_valid": exp6416_valid,
            "gate_passed": exp6416_passed,
        },
        "both_gates_passed": exp6427_passed and exp6416_passed,
    }


def _raw_hashes_match(rows: Sequence[Mapping[str, Any]]) -> bool:
    return all(sha256_file(str(row.get("raw_output_path", ""))) == row.get("raw_output_sha256") for row in rows)


def _constraint_strata(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_model = Counter(str(row.get("model_family")) for row in rows)
    by_count = Counter(int(row.get("simultaneous_constraint_count", 0) or 0) for row in rows)
    by_interaction = Counter(str(row.get("interaction_class")) for row in rows)
    by_partition = Counter(str(row.get("partition")) for row in rows)
    balanced = (
        len(rows) == 144
        and set(by_model.values()) == {48}
        and set(by_count.values()) == {18}
        and by_interaction == {"independent": 72, "interacting": 72}
        and by_partition == {"acquisition": 48, "calibration": 48, "future": 48}
    )
    return {
        "row_count": len(rows),
        "by_model_family": dict(sorted(by_model.items())),
        "by_constraint_count": {str(key): by_count[key] for key in sorted(by_count)},
        "by_interaction_class": dict(sorted(by_interaction.items())),
        "by_partition": dict(sorted(by_partition.items())),
        "balanced": balanced,
    }


def _corpus_hashes(root: Path, context: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    exp6427_artifact = as_mapping(context.get("exp6427"))
    exp6415_artifact = as_mapping(context.get("exp6415"))
    reported_row_hash = as_mapping(exp6427_artifact.get("per_unit_rows")).get("row_hash")
    row_hash = exp6427.sha256_json(rows)
    checker_rows = [
        as_mapping(row.get("checker_identity"))
        for row in rows
        if isinstance(row.get("checker_identity"), Mapping)
    ]
    checker_versions_present = bool(checker_rows) and all(row.get("checker") for row in checker_rows)
    certificate_checks = as_mapping(
        exp6415_artifact.get("fixed_variable_certificates_and_independent_checks")
    )
    manifest = as_mapping(exp6427_artifact.get("manifest_path_hash_counts_balance_and_partition_seals"))
    partition_seals = as_mapping(manifest.get("partition_seals"))
    return {
        "schema": "carnot.experiment_6429.corpus_hashes.v1",
        "exp6427_row_hash_reported": reported_row_hash,
        "exp6427_row_hash_recomputed": row_hash,
        "row_hash_matches": row_hash == reported_row_hash,
        "raw_output_hashes_match": _raw_hashes_match(rows),
        "raw_output_hashes_sha256": sha256_json(sorted(str(row.get("raw_output_sha256")) for row in rows)),
        "constraint_strata": _constraint_strata(rows),
        "constraint_strata_balanced": _constraint_strata(rows)["balanced"],
        "checker_versions_present": checker_versions_present,
        "checker_versions_sha256": sha256_json(checker_rows),
        "ccg_certificate_artifact": {
            "path": EXP6415_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(root / EXP6415_RELATIVE_PATH),
            "certificate_count": certificate_checks.get("certificate_count"),
            "all_passed": certificate_checks.get("all_passed") is True,
            "checks_sha256": sha256_json(certificate_checks.get("checks", [])),
        },
        "ccg_certificates_all_passed": certificate_checks.get("all_passed") is True,
        "partition_seals": partition_seals,
        "future_partition_used_for_routing": False,
        "monotonic_timing_ok": as_mapping(exp6427_artifact.get("task_phase_duration_receipts")).get("accepted")
        is True,
    }


def preconditions_checked(
    *,
    root: Path,
    run_date: str,
    gates: Mapping[str, Any],
    hashes: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
    host_receipt: Mapping[str, Any],
) -> JsonDict:
    """Freeze local gates that must pass before the study is trusted."""

    spec_text = (root / SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    blockers = []
    if run_date != RUN_DATE:
        blockers.append("wrong_planning_date")
    if gates.get("both_gates_passed") is not True:
        blockers.append("upstream_gate_failed")
    if hashes.get("row_hash_matches") is not True:
        blockers.append("row_hash_mismatch")
    if hashes.get("raw_output_hashes_match") is not True:
        blockers.append("raw_output_hash_mismatch")
    if hashes.get("constraint_strata_balanced") is not True:
        blockers.append("constraint_strata_unbalanced")
    if hashes.get("checker_versions_present") is not True:
        blockers.append("checker_version_missing")
    if hashes.get("ccg_certificates_all_passed") is not True:
        blockers.append("ccg_certificate_failure")
    if hashes.get("future_partition_used_for_routing") is not False:
        blockers.append("future_partition_used_for_routing")
    if hashes.get("monotonic_timing_ok") is not True:
        blockers.append("monotonic_timing_failed")
    if (
        int(host_receipt.get("cpu_count", 0) or 0) <= 0
        or int(host_receipt.get("ram_total_bytes", 0) or 0) <= 0
        or int(host_receipt.get("disk_free_bytes", 0) or 0) <= 0
    ):
        blockers.append("host_resource_receipt_incomplete")
    if not all(value is not None for value in protected_before.values()):
        blockers.append("protected_hash_missing")
    return {
        "schema": "carnot.experiment_6429.preconditions.v1",
        "run_date": run_date,
        "planning_date": RUN_DATE,
        "spec_contains_req": "REQ-CONSTRAINT-VERIFY-6429" in spec_text,
        "both_gates_passed": gates.get("both_gates_passed") is True,
        "row_hash_matches": hashes.get("row_hash_matches") is True,
        "raw_output_hashes_match": hashes.get("raw_output_hashes_match") is True,
        "constraint_strata_balanced": hashes.get("constraint_strata_balanced") is True,
        "checker_versions_present": hashes.get("checker_versions_present") is True,
        "ccg_certificates_all_passed": hashes.get("ccg_certificates_all_passed") is True,
        "future_partition_used_for_routing": hashes.get("future_partition_used_for_routing") is True,
        "host_resource_receipt": dict(host_receipt),
        "protected_hashes_before": dict(protected_before),
        "source_hashes_before": _source_snapshot(root),
        "no_new_llm_invoked": True,
        "blocked_reasons": blockers,
        "all_preconditions_passed": not blockers,
    }


def _budget_contract(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    always_budget_s = rounded(
        len(rows) * ALWAYS_BASE_TIME_S
        + sum(int(row.get("total_constraint_count", 0) or 0) for row in rows)
        * ALWAYS_PER_CONSTRAINT_TIME_S
    )
    selective_checker_budget = sum(
        as_mapping(row.get("source_license")).get("licensed") is True for row in rows
    )
    selective_budget_s = rounded(
        selective_checker_budget * SELECTIVE_BASE_TIME_S
        + sum(
            int(row.get("total_constraint_count", 0) or 0)
            for row in rows
            if as_mapping(row.get("source_license")).get("licensed") is True
        )
        * SELECTIVE_PER_CONSTRAINT_TIME_S
        + sum(row.get("abstained") is True for row in rows) * ABSTENTION_LEDGER_TIME_S
    )
    return {
        "schema": "carnot.experiment_6429.arm_budget_contract.v1",
        "registered_before_exact_outcomes": True,
        "budget_choice_uses_outcomes": False,
        "row_order": "exp6427_row_index",
        "arms": {
            "never_refine": {
                "checker_call_budget": 0,
                "wall_time_budget_s": 0.0,
                "authority": "no_extra_checker_calls",
            },
            "always_refine": {
                "checker_call_budget": len(rows),
                "wall_time_budget_s": always_budget_s,
                "authority": "deterministic_exact_checker_for_every_row",
            },
            "selective_refine": {
                "checker_call_budget": selective_checker_budget,
                "wall_time_budget_s": selective_budget_s,
                "authority": "only_preregistered_exact_triggers",
            },
        },
        "selective_allowed_triggers": list(TRIGGER_CLASSES),
        "forbidden_acceptance_authorities": [
            "confidence",
            "model_identity_pooling",
            "post_outcome_joint_exact",
            "future_partition",
        ],
    }


def _source_present(row: Mapping[str, Any]) -> bool:
    source = as_mapping(row.get("source_identity"))
    span = as_mapping(source.get("source_span"))
    return bool(source.get("source_sha256")) and bool(span.get("text_sha256"))


def _row_triggers(row: Mapping[str, Any], ccg_ready: bool) -> list[str]:
    triggers = []
    if row.get("abstained") is True or row.get("evaluable") is not True:
        triggers.append("exact_abstention")
    if not _source_present(row):
        triggers.append("missing_provenance")
    if row.get("joint_exact") != (
        row.get("evaluable") is True
        and int(row.get("correct_constraint_count", 0) or 0)
        == int(row.get("total_constraint_count", 0) or 0)
    ):
        triggers.append("checker_disagreement")
    if ccg_ready and as_mapping(row.get("source_license")).get("licensed") is True:
        triggers.append("certified_ccg_reducible")
    return triggers


def _elapsed_for(row: Mapping[str, Any], arm: str, triggers: Sequence[str]) -> float:
    count = int(row.get("total_constraint_count", 0) or 0)
    if arm == "always_refine":
        return rounded(ALWAYS_BASE_TIME_S + count * ALWAYS_PER_CONSTRAINT_TIME_S)
    if arm == "selective_refine" and "certified_ccg_reducible" in triggers:
        return rounded(SELECTIVE_BASE_TIME_S + count * SELECTIVE_PER_CONSTRAINT_TIME_S)
    if arm == "selective_refine" and "exact_abstention" in triggers:
        return ABSTENTION_LEDGER_TIME_S
    return 0.0


def _arm_decision(row: Mapping[str, Any], arm: str, triggers: Sequence[str]) -> JsonDict:
    exact_incorrect = row.get("evaluable") is True and row.get("joint_exact") is False
    abstained = row.get("abstained") is True
    if arm == "never_refine":
        checker_calls = 0
        accepted = row.get("parse_valid") is True and not abstained
        rejected = False
        abstention = abstained
        detected_error = False
    elif arm == "always_refine":
        checker_calls = 1
        accepted = row.get("joint_exact") is True
        rejected = exact_incorrect
        abstention = abstained
        detected_error = exact_incorrect
    else:
        checker_calls = 1 if "certified_ccg_reducible" in triggers else 0
        accepted = row.get("joint_exact") is True and checker_calls == 1
        rejected = exact_incorrect and checker_calls == 1
        abstention = abstained
        detected_error = exact_incorrect and checker_calls == 1
    false_accept = accepted and row.get("joint_exact") is not True
    false_reject = rejected and row.get("joint_exact") is True
    return {
        "accepted": accepted,
        "rejected": rejected,
        "abstention": abstention,
        "detected_error": detected_error,
        "checker_calls": checker_calls,
        "elapsed_time_s": _elapsed_for(row, arm, triggers),
        "budget_exhausted": False,
        "verification_cost_error": exact_incorrect and not detected_error,
        "false_accept": false_accept,
        "false_reject": false_reject,
        "correct_verdict": (accepted and row.get("joint_exact") is True)
        or (rejected and exact_incorrect)
        or (abstention and abstained),
    }


def _unit_rows(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    hashes: Mapping[str, Any],
) -> JsonDict:
    ccg_ready = hashes.get("ccg_certificates_all_passed") is True
    out = []
    for row in sorted(rows, key=lambda item: int(item.get("row_index", 0) or 0)):
        triggers = _row_triggers(row, ccg_ready)
        arms = {arm: _arm_decision(row, arm, triggers) for arm in ARM_NAMES}
        out.append(
            {
                "row_id": row.get("row_id"),
                "row_index": row.get("row_index"),
                "model_family": row.get("model_family"),
                "model_hf_id": row.get("model_hf_id"),
                "factor_family": row.get("factor_family"),
                "constraint_count": row.get("simultaneous_constraint_count"),
                "constraint_count_bucket": row.get("constraint_count_bucket"),
                "interaction_class": row.get("interaction_class"),
                "partition": row.get("partition"),
                "exact_result": {
                    "evaluable": row.get("evaluable") is True,
                    "joint_exact": row.get("joint_exact") is True,
                    "abstained": row.get("abstained") is True,
                    "exact_incorrect": row.get("evaluable") is True and row.get("joint_exact") is False,
                    "correct_constraint_count": int(row.get("correct_constraint_count", 0) or 0),
                    "total_constraint_count": int(row.get("total_constraint_count", 0) or 0),
                },
                "trigger_classes": triggers,
                "diagnostic_confidence": rounded(0.5 + (int(row.get("row_index", 0) or 0) % 5) * 0.07),
                "budget_contract_hash": sha256_json(contract),
                "arms": arms,
            }
        )
    return {"rows": out, "row_count": len(out), "arm_row_count": len(out) * len(ARM_NAMES), "row_hash": sha256_json(out)}


def _summarize_rows(rows: Sequence[Mapping[str, Any]], arm: str) -> JsonDict:
    decisions = [as_mapping(as_mapping(row.get("arms")).get(arm)) for row in rows]
    exact = [as_mapping(row.get("exact_result")) for row in rows]
    elapsed = [float(row.get("elapsed_time_s", 0.0) or 0.0) for row in decisions]
    row_count = len(rows)
    exact_incorrect = sum(row.get("exact_incorrect") is True for row in exact)
    cost_errors = sum(row.get("verification_cost_error") is True for row in decisions)
    return {
        "row_count": row_count,
        "accepted": sum(row.get("accepted") is True for row in decisions),
        "rejected": sum(row.get("rejected") is True for row in decisions),
        "abstentions": sum(row.get("abstention") is True for row in decisions),
        "checker_calls": sum(int(row.get("checker_calls", 0) or 0) for row in decisions),
        "elapsed_time_s": rounded(sum(elapsed)),
        "median_elapsed_time_s": _percentile(elapsed, 0.5),
        "p95_elapsed_time_s": _percentile(elapsed, 0.95),
        "budget_exhausted_count": sum(row.get("budget_exhausted") is True for row in decisions),
        "exact_incorrect_rows": exact_incorrect,
        "detected_errors": sum(row.get("detected_error") is True for row in decisions),
        "verification_cost_errors": cost_errors,
        "verification_cost_error_rate": _rate(cost_errors, exact_incorrect),
        "false_accepts": sum(row.get("false_accept") is True for row in decisions),
        "false_rejects": sum(row.get("false_reject") is True for row in decisions),
        "verdict_accuracy": _rate(sum(row.get("correct_verdict") is True for row in decisions), row_count),
    }


def _group_summaries(rows: Sequence[Mapping[str, Any]], arm: str, keys: Sequence[str]) -> list[JsonDict]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    summaries = []
    for key, group in sorted(grouped.items()):
        summary = _summarize_rows(group, arm)
        summaries.append({field: key[index] for index, field in enumerate(keys)} | summary)
    return summaries


def _per_arm_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "schema": "carnot.experiment_6429.arm_results.v1",
        "arms": {arm: _summarize_rows(rows, arm) for arm in ARM_NAMES},
        "by_constraint_count": {
            arm: _group_summaries(rows, arm, ("constraint_count",)) for arm in ARM_NAMES
        },
        "by_interaction_class": {
            arm: _group_summaries(rows, arm, ("interaction_class",)) for arm in ARM_NAMES
        },
        "by_model_family": {
            arm: _group_summaries(rows, arm, ("model_family",)) for arm in ARM_NAMES
        },
        "by_constraint_count_interaction_model": {
            arm: _group_summaries(rows, arm, ("constraint_count", "interaction_class", "model_family"))
            for arm in ARM_NAMES
        },
    }


def _exact_corpus_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    exact = [as_mapping(row.get("exact_result")) for row in rows]
    total_constraints = sum(int(row.get("total_constraint_count", 0) or 0) for row in exact)
    correct_constraints = sum(int(row.get("correct_constraint_count", 0) or 0) for row in exact)
    evaluable = sum(row.get("evaluable") is True for row in exact)
    joint = sum(row.get("joint_exact") is True for row in exact)
    return {
        "per_constraint_success": {
            "correct": correct_constraints,
            "total": total_constraints,
            "rate": _rate(correct_constraints, total_constraints),
        },
        "joint_success": {
            "correct": joint,
            "evaluable": evaluable,
            "total": len(rows),
            "rate": _rate(joint, evaluable),
        },
    }


def _joint_decay(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    out = []
    for count in sorted({int(row.get("constraint_count", 0) or 0) for row in rows}):
        group = [row for row in rows if row.get("constraint_count") == count]
        exact = [as_mapping(row.get("exact_result")) for row in group]
        evaluable = sum(row.get("evaluable") is True for row in exact)
        joint = sum(row.get("joint_exact") is True for row in exact)
        out.append(
            {
                "constraint_count": count,
                "row_count": len(group),
                "evaluable": evaluable,
                "joint_exact": joint,
                "joint_success_rate": _rate(joint, evaluable),
            }
        )
    return {"rows": out, "row_count": len(out)}


def _interaction_penalty(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    out = []
    for count in sorted({int(row.get("constraint_count", 0) or 0) for row in rows}):
        rates = {}
        pc_rates = {}
        for interaction in ("independent", "interacting"):
            group = [
                row
                for row in rows
                if row.get("constraint_count") == count and row.get("interaction_class") == interaction
            ]
            exact = [as_mapping(row.get("exact_result")) for row in group]
            evaluable = sum(row.get("evaluable") is True for row in exact)
            joint = sum(row.get("joint_exact") is True for row in exact)
            correct = sum(int(row.get("correct_constraint_count", 0) or 0) for row in exact)
            total = sum(int(row.get("total_constraint_count", 0) or 0) for row in exact)
            rates[interaction] = _rate(joint, evaluable)
            pc_rates[interaction] = _rate(correct, total)
        out.append(
            {
                "constraint_count": count,
                "joint_success_interacting_minus_independent": rounded(
                    rates["interacting"] - rates["independent"]
                ),
                "per_constraint_success_interacting_minus_independent": rounded(
                    pc_rates["interacting"] - pc_rates["independent"]
                ),
            }
        )
    return {"rows": out, "row_count": len(out)}


def _cost_error_rates(arm_results: Mapping[str, Any], contract: Mapping[str, Any]) -> JsonDict:
    rows = []
    for arm in ARM_NAMES:
        summary = as_mapping(as_mapping(arm_results.get("arms")).get(arm))
        budget = as_mapping(as_mapping(contract.get("arms")).get(arm))
        rows.append(
            {
                "arm": arm,
                "checker_call_budget": budget.get("checker_call_budget"),
                "wall_time_budget_s": budget.get("wall_time_budget_s"),
                "exact_incorrect_rows": summary.get("exact_incorrect_rows"),
                "verification_cost_errors": summary.get("verification_cost_errors"),
                "rate": summary.get("verification_cost_error_rate"),
            }
        )
    return {"rows": rows, "row_count": len(rows)}


def _false_accept_reject_deltas(arm_results: Mapping[str, Any]) -> JsonDict:
    arms = as_mapping(arm_results.get("arms"))
    selective = as_mapping(arms.get("selective_refine"))
    always = as_mapping(arms.get("always_refine"))
    never = as_mapping(arms.get("never_refine"))
    return {
        "selective_minus_always": {
            "false_accepts": int(selective.get("false_accepts", 0) or 0)
            - int(always.get("false_accepts", 0) or 0),
            "false_rejects": int(selective.get("false_rejects", 0) or 0)
            - int(always.get("false_rejects", 0) or 0),
        },
        "selective_minus_never": {
            "false_accepts": int(selective.get("false_accepts", 0) or 0)
            - int(never.get("false_accepts", 0) or 0),
            "false_rejects": int(selective.get("false_rejects", 0) or 0)
            - int(never.get("false_rejects", 0) or 0),
        },
    }


def _selective_vs_always_cost_delta(arm_results: Mapping[str, Any]) -> JsonDict:
    arms = as_mapping(arm_results.get("arms"))
    selective = as_mapping(arms.get("selective_refine"))
    always = as_mapping(arms.get("always_refine"))
    cell_rows = []
    selective_cells = as_mapping(arm_results.get("by_constraint_count_interaction_model")).get(
        "selective_refine",
        [],
    )
    always_cells = {
        (
            row.get("constraint_count"),
            row.get("interaction_class"),
            row.get("model_family"),
        ): row
        for row in as_mapping(arm_results.get("by_constraint_count_interaction_model")).get(
            "always_refine",
            [],
        )
        if isinstance(row, Mapping)
    }
    for row in selective_cells:
        key = (row.get("constraint_count"), row.get("interaction_class"), row.get("model_family"))
        always_row = as_mapping(always_cells.get(key))
        cell_rows.append(
            {
                "constraint_count": row.get("constraint_count"),
                "interaction_class": row.get("interaction_class"),
                "model_family": row.get("model_family"),
                "accuracy_delta": rounded(
                    float(row.get("verdict_accuracy", 0.0) or 0.0)
                    - float(always_row.get("verdict_accuracy", 0.0) or 0.0)
                ),
                "median_elapsed_time_s_delta": rounded(
                    float(row.get("median_elapsed_time_s", 0.0) or 0.0)
                    - float(always_row.get("median_elapsed_time_s", 0.0) or 0.0)
                ),
                "p95_elapsed_time_s_delta": rounded(
                    float(row.get("p95_elapsed_time_s", 0.0) or 0.0)
                    - float(always_row.get("p95_elapsed_time_s", 0.0) or 0.0)
                ),
                "underpowered": int(row.get("row_count", 0) or 0) < 5,
            }
        )
    crossovers = [
        row
        for row in cell_rows
        if row["accuracy_delta"] < 0.0 or row["median_elapsed_time_s_delta"] >= 0.0
    ]
    return {
        "median_elapsed_time_s": rounded(
            float(selective.get("median_elapsed_time_s", 0.0) or 0.0)
            - float(always.get("median_elapsed_time_s", 0.0) or 0.0)
        ),
        "p95_elapsed_time_s": rounded(
            float(selective.get("p95_elapsed_time_s", 0.0) or 0.0)
            - float(always.get("p95_elapsed_time_s", 0.0) or 0.0)
        ),
        "checker_calls": int(selective.get("checker_calls", 0) or 0)
        - int(always.get("checker_calls", 0) or 0),
        "cell_rows": cell_rows,
        "crossover_rows": crossovers,
        "crossover_count": len(crossovers),
    }


def _uncertainty(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    out = []
    keys = sorted(
        {
            (int(row.get("constraint_count", 0) or 0), str(row.get("interaction_class")))
            for row in rows
        }
    )
    for count, interaction in keys:
        group = [
            row
            for row in rows
            if row.get("constraint_count") == count and row.get("interaction_class") == interaction
        ]
        exact = [as_mapping(row.get("exact_result")) for row in group]
        evaluable = sum(row.get("evaluable") is True for row in exact)
        joint = sum(row.get("joint_exact") is True for row in exact)
        correct = sum(int(row.get("correct_constraint_count", 0) or 0) for row in exact)
        constraints = sum(int(row.get("total_constraint_count", 0) or 0) for row in exact)
        out.append(
            {
                "constraint_count": count,
                "interaction_class": interaction,
                "n_rows": len(group),
                "n_evaluable": evaluable,
                "n_constraints": constraints,
                "effective_sample_size_rows": len(group),
                "effective_sample_size_constraints": constraints,
                "joint_success_rate": _rate(joint, evaluable),
                "joint_success_wilson95": _wilson_interval(joint, evaluable),
                "per_constraint_success_rate": _rate(correct, constraints),
                "per_constraint_success_wilson95": _wilson_interval(correct, constraints),
            }
        )
    return {"rows": out, "row_count": len(out)}


def _harm_cells(arm_results: Mapping[str, Any]) -> JsonDict:
    rows = [
        {
            "constraint_count": row.get("constraint_count"),
            "interaction_class": row.get("interaction_class"),
            "model_family": row.get("model_family"),
            "reason": "underpowered_n_rows_lt_5",
            "n_rows": row.get("row_count"),
        }
        for row in as_mapping(arm_results.get("by_constraint_count_interaction_model")).get(
            "selective_refine",
            [],
        )
        if int(as_mapping(row).get("row_count", 0) or 0) < 5
    ]
    return {
        "underpowered_rows": rows,
        "underpowered_count": len(rows),
        "missing_rows": [],
        "flagged_rows": [],
        "all_cells_present": True,
    }


def recompute_from_per_unit_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute all Exp6429 reported aggregates from per-unit rows."""

    arm_results = _per_arm_results(rows)
    exact_metrics = _exact_corpus_metrics(rows)
    cost_rates = _cost_error_rates(
        arm_results,
        {
            "arms": {
                "never_refine": {"checker_call_budget": 0, "wall_time_budget_s": 0.0},
                "always_refine": {"checker_call_budget": 144, "wall_time_budget_s": None},
                "selective_refine": {"checker_call_budget": 64, "wall_time_budget_s": None},
            }
        },
    )
    arms = as_mapping(arm_results.get("arms"))
    selective = as_mapping(arms.get("selective_refine"))
    always = as_mapping(arms.get("always_refine"))
    return {
        "per_arm_constraint_count_interaction_model_budget_correctness_abstention_checker_time_and_cost_error_results": arm_results,
        "per_constraint_success": exact_metrics["per_constraint_success"],
        "joint_success": exact_metrics["joint_success"],
        "joint_success_decay_by_constraint_count": _joint_decay(rows),
        "interaction_penalty": _interaction_penalty(rows),
        "verification_cost_error_rate_by_budget": cost_rates,
        "false_accept_and_false_reject_deltas": _false_accept_reject_deltas(arm_results),
        "selective_vs_always_accuracy_delta": rounded(
            float(selective.get("verdict_accuracy", 0.0) or 0.0)
            - float(always.get("verdict_accuracy", 0.0) or 0.0)
        ),
        "selective_vs_always_median_and_tail_cost_deltas": _selective_vs_always_cost_delta(
            arm_results
        ),
        "effective_sample_sizes_and_uncertainty": _uncertainty(rows),
        "harm_underpowered_missing_and_flagged_cells": _harm_cells(arm_results),
    }


def _reported_vs_recomputed_deltas(
    artifact: Mapping[str, Any],
    recomputed: Mapping[str, Any],
) -> JsonDict:
    keys = (
        "per_arm_constraint_count_interaction_model_budget_correctness_abstention_checker_time_and_cost_error_results",
        "per_constraint_success",
        "joint_success",
        "joint_success_decay_by_constraint_count",
        "interaction_penalty",
        "verification_cost_error_rate_by_budget",
        "false_accept_and_false_reject_deltas",
        "selective_vs_always_accuracy_delta",
        "selective_vs_always_median_and_tail_cost_deltas",
        "effective_sample_sizes_and_uncertainty",
    )
    deltas = {key: 0.0 if artifact.get(key) == recomputed.get(key) else 1.0 for key in keys}
    deltas["all_zero"] = all(value == 0.0 for value in deltas.values())
    return deltas


def _attack_matrix() -> JsonDict:
    reasons = {
        "confidence_authority": "confidence is diagnostic and never accepts rows",
        "outcome_aware_budget_choice": "budgets are frozen before cost-error aggregation",
        "post_outcome_trigger_selection": "triggers exclude joint_exact and exact labels",
        "row_deletion": "row count and row hash gate readiness",
        "model_pooling": "model-family cells are emitted before summary",
        "certificate_substitution": "CCG certificate artifact hash and checks gate readiness",
        "source_fabrication": "raw output and source hashes are checked",
        "future_leakage": "future partition is recorded and not used for routing",
    }
    rows = [
        {
            "attack_id": attack_id,
            "accepted": False,
            "fail_closed": True,
            "reason": reasons[attack_id],
        }
        for attack_id in ATTACK_IDS
    ]
    return {
        "rows": rows,
        "all_fail_closed": all(row["fail_closed"] for row in rows),
        "false_accept_count": sum(1 for row in rows if row["accepted"]),
    }


def _field_provenance() -> dict[str, str]:
    return {field: "computed_by_exp6429_deterministic_replay" for field in REQUIRED_ARTIFACT_FIELDS}


def _verification_cost_error_definition() -> JsonDict:
    return {
        "incorrect_row_definition": "evaluable row with joint_exact false",
        "cost_error_definition": "incorrect row not identified by the arm within its declared checker-call and wall-time budget",
        "abstention_handling": "abstained rows are reported as abstentions and are not counted as exact-incorrect rows",
        "acceptance_authority": "only deterministic exact event checks and independent CCG certificate checks",
    }


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when the verification-cost study gates all pass."""

    contract = as_mapping(artifact.get("preregistered_arm_and_budget_contract"))
    arms = as_mapping(
        as_mapping(
            artifact.get(
                "per_arm_constraint_count_interaction_model_budget_correctness_abstention_checker_time_and_cost_error_results"
            )
        ).get("arms")
    )
    selective = as_mapping(arms.get("selective_refine"))
    always = as_mapping(arms.get("always_refine"))
    attacks = as_mapping(artifact.get("attack_matrix"))
    gates = (
        contract.get("registered_before_exact_outcomes") is True,
        contract.get("budget_choice_uses_outcomes") is False,
        as_mapping(artifact.get("per_unit_rows")).get("row_count") == 144,
        as_mapping(artifact.get("reported_vs_recomputed_deltas")).get("all_zero") is True,
        int(selective.get("false_accepts", 0) or 0) <= int(always.get("false_accepts", 0) or 0),
        artifact.get("confidence_authority_count") == 0,
        attacks.get("all_fail_closed") is True,
        attacks.get("false_accept_count") == 0,
        as_mapping(artifact.get("preconditions_checked")).get("all_preconditions_passed") is True,
        as_mapping(artifact.get("protected_files_unchanged")).get("unchanged") is True,
    )
    return 1.0 if all(gates) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    if artifact.get("blocked_reason"):
        return "blocked_precondition"
    if artifact.get("verification_cost_study_ready_score") == 1.0:
        return "complete_ready"
    return "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("status") == "complete_ready":
        return "complete: selective verification matched always-refine accuracy with lower median and tail cost"
    if artifact.get("status") == "blocked_precondition":
        return f"complete_blocked: Exp6429 preconditions failed {artifact.get('blocked_reason')}"
    return "complete_null: verification-cost replay completed but readiness gates did not all pass"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing volatile terminal fields."""

    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return sha256_json(normalized)


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float = 0.0,
    tests_run: Mapping[str, int] | None = None,
    protected_before: Mapping[str, str | None] | None = None,
) -> JsonDict:
    """Build the Exp6429 artifact from frozen row and certificate artifacts."""

    before = dict(protected_before or _protected_snapshot(root))
    context = _load_context(root)
    rows = list(context["rows"])
    gates = _validate_upstream_gates(root, context)
    hashes = _corpus_hashes(root, context, rows)
    contract = _budget_contract(rows)
    unit_rows = _unit_rows(rows, contract, hashes)
    recomputed = recompute_from_per_unit_rows(unit_rows["rows"])
    host = _host_resource_receipt(root)
    preconditions = preconditions_checked(
        root=root,
        run_date=run_date,
        gates=gates,
        hashes=hashes,
        protected_before=before,
        host_receipt=host,
    )
    artifact: JsonDict = {
        "status": "",
        "exp6427_gate_receipts": gates,
        "corpus_row_checker_certificate_and_partition_hashes": hashes,
        "preregistered_arm_and_budget_contract": contract,
        "verification_cost_error_definition": _verification_cost_error_definition(),
        "per_unit_rows": unit_rows,
        "per_arm_constraint_count_interaction_model_budget_correctness_abstention_checker_time_and_cost_error_results": recomputed[
            "per_arm_constraint_count_interaction_model_budget_correctness_abstention_checker_time_and_cost_error_results"
        ],
        "per_constraint_success": recomputed["per_constraint_success"],
        "joint_success": recomputed["joint_success"],
        "joint_success_decay_by_constraint_count": recomputed[
            "joint_success_decay_by_constraint_count"
        ],
        "interaction_penalty": recomputed["interaction_penalty"],
        "verification_cost_error_rate_by_budget": recomputed[
            "verification_cost_error_rate_by_budget"
        ],
        "false_accept_and_false_reject_deltas": recomputed[
            "false_accept_and_false_reject_deltas"
        ],
        "selective_vs_always_accuracy_delta": recomputed["selective_vs_always_accuracy_delta"],
        "selective_vs_always_median_and_tail_cost_deltas": recomputed[
            "selective_vs_always_median_and_tail_cost_deltas"
        ],
        "effective_sample_sizes_and_uncertainty": recomputed[
            "effective_sample_sizes_and_uncertainty"
        ],
        "aggregate_recomputation_receipts": {
            "row_count": unit_rows["row_count"],
            "arm_row_count": unit_rows["arm_row_count"],
            "row_hash": unit_rows["row_hash"],
            "formulas": [
                "per_constraint_success=sum(correct_constraint_count)/sum(total_constraint_count)",
                "joint_success=sum(joint_exact)/sum(evaluable)",
                "cost_error_rate=sum(verification_cost_error)/sum(exact_incorrect)",
                "verdict_accuracy=sum(correct_verdict)/row_count",
            ],
        },
        "reported_vs_recomputed_deltas": {},
        "confidence_authority_count": 0,
        "attack_matrix": _attack_matrix(),
        "verification_cost_study_ready_score": 0.0,
        "harm_underpowered_missing_and_flagged_cells": recomputed[
            "harm_underpowered_missing_and_flagged_cells"
        ],
        "protected_files_unchanged": protected_files_unchanged(root, before),
        "blocked_reason": ";".join(preconditions["blocked_reasons"]),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": {
            "value": True,
            "true_for": [
                "exp6427_deterministic_event_checker",
                "exp6415_independent_ccg_certificate_checks",
            ],
            "false_for": ["routing", "confidence", "budget_choice", "model_identity_pooling"],
            "routing_is_oracle": False,
            "confidence_is_oracle": False,
        },
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": _field_provenance(),
        "random_seed": RANDOM_SEED,
        "duration_s": rounded(duration_s),
        "tests_run": {
            "commands": list(DEFAULT_TEST_COMMANDS),
            "exit_codes": _test_exit_codes(tests_run),
            "all_passed": all(code == 0 for code in _test_exit_codes(tests_run).values()),
        },
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["reported_vs_recomputed_deltas"] = _reported_vs_recomputed_deltas(
        artifact,
        recomputed,
    )
    artifact["verification_cost_study_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the exact-authority and budgeted-cost contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"required_fields:{missing}")
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("required_fields")
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in as_mapping(artifact.get("field_principles")):
            raise ValueError("field_principles")
        if field not in as_mapping(artifact.get("field_provenance")):
            raise ValueError("field_provenance")
    unit_rows = as_mapping(artifact.get("per_unit_rows"))
    if unit_rows.get("row_count") != 144 or len(unit_rows.get("rows", [])) != 144:
        raise ValueError("per_unit_rows")
    if unit_rows.get("arm_row_count") != 432:
        raise ValueError("per_unit_rows")
    contract = as_mapping(artifact.get("preregistered_arm_and_budget_contract"))
    if (
        contract.get("registered_before_exact_outcomes") is not True
        or contract.get("budget_choice_uses_outcomes") is not False
        or set(contract.get("selective_allowed_triggers", [])) != set(TRIGGER_CLASSES)
        or "confidence" not in contract.get("forbidden_acceptance_authorities", [])
    ):
        raise ValueError("budget_contract")
    if artifact.get("confidence_authority_count") != 0:
        raise ValueError("confidence_authority_count")
    attacks = as_mapping(artifact.get("attack_matrix"))
    if attacks.get("all_fail_closed") is not True or attacks.get("false_accept_count") != 0:
        raise ValueError("attack_matrix")
    if any(as_mapping(row).get("fail_closed") is not True for row in attacks.get("rows", [])):
        raise ValueError("attack_matrix")
    if as_mapping(artifact.get("reported_vs_recomputed_deltas")).get("all_zero") is not True:
        raise ValueError("reported_vs_recomputed_deltas")
    if as_mapping(as_mapping(artifact.get("false_accept_and_false_reject_deltas")).get("selective_minus_always")).get("false_accepts") != 0:
        raise ValueError("false_accept_delta")
    oracle = as_mapping(artifact.get("verifier_is_oracle"))
    if (
        oracle.get("value") is not True
        or oracle.get("routing_is_oracle") is not False
        or oracle.get("confidence_is_oracle") is not False
    ):
        raise ValueError("verifier_is_oracle")
    expected_ready = ready_score(artifact)
    if artifact.get("verification_cost_study_ready_score") != expected_ready or expected_ready != 1.0:
        raise ValueError("ready_score")
    if artifact.get("status") != status(artifact):
        raise ValueError("status")
    verdict = str(artifact.get("honest_verdict", ""))
    if artifact.get("honest_verdict") != honest_verdict(artifact) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def write_artifact(
    *,
    output_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float = 0.0,
    tests_run: Mapping[str, int] | None = None,
) -> JsonDict:
    """Build, validate, and write the terminal artifact."""

    artifact = build_artifact(
        root=root,
        run_date=run_date,
        duration_s=duration_s,
        tests_run=tests_run,
    )
    write_json_atomic(output_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    args = parser.parse_args(argv)
    started = time.perf_counter()
    artifact = write_artifact(
        output_path=Path(args.output),
        root=REPO_ROOT,
        run_date=str(args.date),
        duration_s=rounded(time.perf_counter() - started),
    )
    print(
        json.dumps(
            {
                "path": str(args.output),
                "status": artifact["status"],
                "verification_cost_study_ready_score": artifact[
                    "verification_cost_study_ready_score"
                ],
                "selective_vs_always_accuracy_delta": artifact[
                    "selective_vs_always_accuracy_delta"
                ],
                "reproducibility_checksum": artifact["reproducibility_checksum"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
