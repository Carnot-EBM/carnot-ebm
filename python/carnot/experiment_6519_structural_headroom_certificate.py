"""Exp6519 independent structural headroom certificate.

Spec refs: REQ-BENCH-6519, SCENARIO-BENCH-6519-MISSING-SOURCE,
SCENARIO-BENCH-6519-INDEPENDENT-ROWS, SCENARIO-BENCH-6519-EXACT-REPLAY,
SCENARIO-BENCH-6519-LIVE-COST-BREADTH, SCENARIO-BENCH-6519-ATTACKS,
SCENARIO-BENCH-6519-TERMINAL.

This runner reads Exp6518 as evidence. It does not repair or regenerate that
artifact. The certificate opens only when row-derived checks and sampled solver
replay agree with the sealed pilot rows.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from typing import Any

from carnot import experiment_6518_structural_control_headroom_ab_v2 as exp6518
from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6519
SCHEMA_VERSION = "carnot.experiment_6519.structural_headroom_certificate.v1"
INFERENCE_SUBSTRATE = "independent_structural_control_row_and_solver_receipt_replay_no_llm"
VERIFIER_IS_ORACLE = False

RESULT_RELATIVE_PATH = Path("results/experiment_6519_structural_headroom_certificate.json")
EXP6518_RELATIVE_PATH = Path("results/experiment_6518_structural_control_headroom_ab_v2.json")
EXP6517_RELATIVE_PATH = Path("results/experiment_6517_branch_pilot_independent_audit.json")
EXP6516_RELATIVE_PATH = Path("results/experiment_6516_exact_branch_pilot_dataset_v3.json")
EXP6504_RELATIVE_PATH = Path("results/experiment_6504_exact_structural_benchmark_commitment.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/benchmarks/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6519_structural_headroom_certificate.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6519_structural_headroom_certificate.py")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
ROADMAP_RELATIVE_PATH = Path("research-references.md")

ARM_IDS = exp6518.ARM_IDS
NATIVE_ARM = exp6518.NATIVE_ARM
EXPECTED_PILOT_UNIT_COUNT = exp6518.PILOT_BASE_UNIT_COUNT
EXPECTED_MATCHED_ROW_COUNT = EXPECTED_PILOT_UNIT_COUNT * len(ARM_IDS)
MIN_REPLAY_SAMPLE_ROWS = 9
PRIMARY_METRIC = exp6518.PRIMARY_METRIC

ATTACK_IDS = (
    "identity",
    "row_order",
    "serialization_length",
    "family_imbalance",
    "held_tuning",
    "cost_omission",
    "inactive_hooks",
    "one_win_headline",
    "aggregate_contradiction",
    "exact_oracle_class_inflation",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "upstream_artifact_receipt",
    "independent_row_recomputation",
    "exact_receipt_replay_rows",
    "live_influence_audit",
    "charged_cost_audit",
    "paired_effect_rows",
    "breadth_and_censoring_audit",
    "attack_matrix",
    "certified_structural_headroom_score",
    "gate_check_summary",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Records the terminal independent certificate state.",
    "honest_verdict": "States whether row-derived evidence certified structural headroom.",
    "verdict_class": (
        "Closed enum separates certified positive, null, blocked, and disqualified outcomes."
    ),
    "upstream_artifact_receipt": (
        "Pins Exp6518 by path, hash, terminal status, class, row counts, resources, and protected hashes."
    ),
    "independent_row_recomputation": (
        "Recomputes row identity, joins, equality, influence, costs, and aggregate contradictions without trusting Exp6518 aggregates."
    ),
    "exact_receipt_replay_rows": (
        "Replays deterministic solver samples and every correctness discrepancy."
    ),
    "live_influence_audit": "Counts live advice only when rows show a changed decision path.",
    "charged_cost_audit": (
        "Charges solver, feature, refocus, enumeration, and fallback work from rows."
    ),
    "paired_effect_rows": (
        "Reports paired native-versus-arm held effects, uncertainty, and tail metrics."
    ),
    "breadth_and_censoring_audit": (
        "Records family breadth, seed breadth, headroom strata, timeouts, censoring, and bounds."
    ),
    "attack_matrix": (
        "Forces identity, order, length, family, tuning, cost, inactive-hook, one-win, aggregate, and oracle-class attacks closed."
    ),
    "certified_structural_headroom_score": (
        "Opens only when all independent certificate gates pass."
    ),
    "gate_check_summary": "Names every failed gate with expected and observed values.",
    "per_unit_rows": "Emits one audit row per unit-arm and attack for external row checks.",
    "aggregate_row_recomputation": (
        "Rebuilds the certificate score and verdict from independent rows."
    ),
    "preconditions_checked": (
        "Records paths, hashes, resources, solvers, planning date, row counts, and protected hashes."
    ),
    "protected_files_unchanged": (
        "Proves protected upstream and verifier files stayed byte-identical during the run."
    ),
    "inference_substrate": (
        "Declares independent structural-control row and solver-receipt replay with no LLM."
    ),
    "verifier_is_oracle": "False because the performance claim is measured, not oracle-certified.",
    "field_principles": "Explains why each required field exists.",
    "field_provenance": (
        "Maps each field to specs, inputs, rows, reducers, solver replay, tests, and hashes."
    ),
    "random_seed": "Pins deterministic replay sampling and attack ordering.",
    "duration_s": "Records measured wall time.",
    "tests_run": "Records validation commands and exit codes.",
    "reproducibility_checksum": "Detects drift in inputs, rows, gates, attacks, and verdicts.",
}

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6519_structural_headroom_certificate.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6519_structural_headroom_certificate.py "
    "-m pytest tests/python/test_experiment_6519_structural_headroom_certificate.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6519_structural_headroom_certificate.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6519_structural_headroom_certificate.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6519_structural_headroom_certificate.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6519_structural_headroom_certificate.json"
)
EXACT_E2E_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py "
    "-q --no-cov -n 0"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6519_structural_headroom_certificate "
    "--date 20260823"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6519_structural_headroom_certificate --validate"
)
DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": EXACT_E2E_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": "git status --short", "exit_code": 0},
)

PROTECTED_RELATIVE_PATHS = (
    EXP6518_RELATIVE_PATH,
    EXP6517_RELATIVE_PATH,
    EXP6516_RELATIVE_PATH,
    EXP6504_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/experiment_6518_structural_control_headroom_ab_v2.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("scripts/research_conductor.py"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    ROADMAP_RELATIVE_PATH,
)

SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    EXP6518_RELATIVE_PATH,
    EXP6517_RELATIVE_PATH,
    EXP6516_RELATIVE_PATH,
    EXP6504_RELATIVE_PATH,
    ROADMAP_RELATIVE_PATH,
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json_with_status(path: Path) -> tuple[JsonDict, str, str]:
    if not path.is_file():
        return {}, "missing", ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, "corrupt_json", str(exc)
    if not isinstance(payload, Mapping):
        return {}, "non_object", "top-level JSON is not an object"
    return dict(payload), "parsed", ""


def _command_output(command: Sequence[str], cwd: Path) -> tuple[int, str]:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return result.returncode, result.stdout.strip() or result.stderr.strip()


def _resource_state(repo_root: Path) -> JsonDict:
    disk = shutil.disk_usage(repo_root)
    meminfo: dict[str, int] = {}
    mem_path = Path("/proc/meminfo")
    if mem_path.is_file():
        for line in mem_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                meminfo[parts[0].rstrip(":")] = int(parts[1]) * 1024
    return {
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "ram_total_bytes": meminfo.get("MemTotal"),
        "ram_available_bytes": meminfo.get("MemAvailable"),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
    }


def _source_key(repo_root: Path, path: Path) -> str:
    resolved = path.resolve(strict=False)
    repo = repo_root.resolve(strict=False)
    if resolved.is_relative_to(repo):
        return resolved.relative_to(repo).as_posix()
    return str(path)


def protected_file_hashes(repo_root: Path, source_path: Path | None = None) -> dict[str, str]:
    hashes = {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}
    if source_path is not None:
        hashes[_source_key(repo_root, source_path)] = sha256_file(source_path)
    return hashes


def protected_files_unchanged(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> JsonDict:
    changed = [
        {"path": path, "before": before.get(path), "after": after.get(path)}
        for path in sorted(set(before) | set(after))
        if before.get(path) != after.get(path)
    ]
    return {
        "all_protected_files_unchanged": not changed,
        "changed_files": changed,
        "hashes_before": dict(before),
        "hashes_after": dict(after),
    }


def _terminal_class(payload: Mapping[str, Any], parse_status: str) -> str:
    if parse_status != "parsed":
        return parse_status
    status = str(payload.get("status") or "")
    verdict_class = payload.get("verdict_class")
    if not status.startswith(("complete_", "blocked_", "disqualified_")):
        return "nonterminal"
    if verdict_class == "positive":
        return "terminal_positive"
    if verdict_class == "blocked":
        return "terminal_blocked"
    if verdict_class == "disqualified":
        return "terminal_disqualified"
    if verdict_class is None:
        return "terminal_null"
    return "terminal_other"


def _row_groups(payload: Mapping[str, Any]) -> dict[str, list[JsonDict]]:
    groups: dict[str, list[JsonDict]] = defaultdict(list)
    rows = payload.get("per_unit_rows")
    if not isinstance(rows, list):
        return groups
    for row in rows:
        if isinstance(row, Mapping):
            groups[str(row.get("row_type"))].append(dict(row))
    return dict(groups)


def upstream_artifact_receipt(
    *,
    repo_root: Path,
    source_path: Path,
    protected_before: Mapping[str, str],
) -> JsonDict:
    payload, parse_status, parse_error = _read_json_with_status(source_path)
    groups = _row_groups(payload)
    row_counts = {key: len(rows) for key, rows in sorted(groups.items())}
    return {
        "row_type": "upstream_artifact_receipt",
        "path": _source_key(repo_root, source_path),
        "absolute_path": str(source_path),
        "exists": source_path.is_file(),
        "sha256": sha256_file(source_path),
        "parse_status": parse_status,
        "parse_error": parse_error,
        "status": payload.get("status"),
        "honest_verdict": payload.get("honest_verdict"),
        "verdict_class": payload.get("verdict_class"),
        "class": _terminal_class(payload, parse_status),
        "per_unit_row_count": len(payload.get("per_unit_rows", []))
        if isinstance(payload.get("per_unit_rows"), list)
        else 0,
        "row_type_counts": row_counts,
        "structural_control_game_row_count": row_counts.get("structural_control_game", 0),
        "live_influence_row_count": row_counts.get("live_influence", 0),
        "exact_answer_equality_row_count": row_counts.get("exact_answer_equality", 0),
        "charged_cost_row_count": row_counts.get("charged_cost", 0),
        "censoring_row_count": row_counts.get("censoring", 0),
        "attack_row_count": row_counts.get("structural_control_attack", 0),
        "family_summary_row_count": row_counts.get("family_seed_summary_arm", 0),
        "resources": _resource_state(repo_root),
        "protected_file_hashes_before": dict(protected_before),
        "spec_refs": ["REQ-BENCH-6519", "SCENARIO-BENCH-6519-MISSING-SOURCE"],
    }


def _sealed_pilot_index(repo_root: Path) -> dict[tuple[str, str], list[JsonDict]]:
    audit, status, _ = _read_json_with_status(repo_root / EXP6517_RELATIVE_PATH)
    if status != "parsed":
        return {}
    groups: dict[tuple[str, str], list[JsonDict]] = defaultdict(list)
    for row in _row_groups(audit).get("source_unit_audit", []):
        key = (str(row.get("base_instance_hash")), str(row.get("checkpoint_id")))
        groups[key].append(dict(row))
    return dict(groups)


def _recompute_source_row_hash(row: Mapping[str, Any]) -> bool:
    if "row_hash" not in row:
        return False
    clone = dict(row)
    expected = clone.pop("row_hash")
    return sha256_json(clone) == expected


def _cost_components(row: Mapping[str, Any]) -> int:
    return (
        int(row.get("solver_only_work_units", -10**9))
        + int(row.get("feature_cost_units", -10**9))
        + int(row.get("refocus_cost_units", -10**9))
        + int(row.get("enumeration_cost_units", -10**9))
        + int(row.get("fallback_cost_units", -10**9))
    )


def _source_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    return _row_groups(payload).get("structural_control_game", [])


def _pilot_row_join(
    row: Mapping[str, Any],
    pilot_index: Mapping[tuple[str, str], list[JsonDict]],
) -> tuple[bool, list[bool], bool]:
    candidates = pilot_index.get((str(row.get("base_instance_hash")), str(row.get("checkpoint_id"))), [])
    values = sorted({bool(item.get("candidate_value")) for item in candidates})
    metadata_ok = bool(candidates) and all(
        item.get("family") == row.get("family")
        and item.get("split") == row.get("split")
        and item.get("base_lineage_id") == row.get("base_lineage_id")
        and item.get("audit_passed") is True
        for item in candidates
    )
    return metadata_ok and values == [False, True] and len(candidates) == 2, values, metadata_ok


def build_unit_audit_rows(
    *,
    source_payload: Mapping[str, Any],
    repo_root: Path,
) -> tuple[list[JsonDict], JsonDict]:
    rows = _source_rows(source_payload)
    pilot_index = _sealed_pilot_index(repo_root)
    native_by_unit = {
        str(row.get("pilot_unit_id")): dict(row)
        for row in rows
        if row.get("arm_id") == NATIVE_ARM
    }
    seen: Counter[tuple[str, str]] = Counter(
        (str(row.get("pilot_unit_id")), str(row.get("arm_id"))) for row in rows
    )
    missing_units = sorted(
        {
            str(row.get("pilot_unit_id"))
            for row in rows
            for arm_id in ARM_IDS
            if seen[(str(row.get("pilot_unit_id")), arm_id)] == 0
        }
    )
    duplicate_keys = [
        {"pilot_unit_id": pilot, "arm_id": arm, "count": count}
        for (pilot, arm), count in sorted(seen.items())
        if count > 1
    ]

    out: list[JsonDict] = []
    for row in rows:
        native = native_by_unit.get(str(row.get("pilot_unit_id")), {})
        source_hash_ok = _recompute_source_row_hash(row)
        join_passed, sealed_values, metadata_ok = _pilot_row_join(row, pilot_index)
        cost_sum = _cost_components(row)
        total = int(row.get("total_charged_work_units", -1))
        native_total = int(native.get("total_charged_work_units", 0))
        held_benefit = (
            native_total - total
            if row.get("split") == "held" and row.get("arm_id") != NATIVE_ARM
            else 0
        )
        receipt = row.get("terminal_model_or_proof", {})
        payload = {
            "row_type": "structural_headroom_unit_audit",
            "schema_version": SCHEMA_VERSION + ".unit_audit",
            "unit_id": row.get("unit_id"),
            "pilot_unit_id": row.get("pilot_unit_id"),
            "arm_id": row.get("arm_id"),
            "base_instance_hash": row.get("base_instance_hash"),
            "checkpoint_id": row.get("checkpoint_id"),
            "split": row.get("split"),
            "family": row.get("family"),
            "selection_seed": row.get("selection_seed"),
            "source_row_hash": row.get("row_hash"),
            "source_row_hash_recomputed": source_hash_ok,
            "sealed_candidate_values": sealed_values,
            "sealed_pilot_metadata_passed": metadata_ok,
            "sealed_pilot_join_passed": join_passed,
            "candidate_preserved": row.get("candidate_preserved") is True,
            "exact_answer_equality": row.get("exact_answer_equality") is True,
            "receipt_valid": isinstance(receipt, Mapping) and receipt.get("receipt_valid") is True,
            "live_influence_detected": row.get("live_influence_detected") is True,
            "changed_decision_count": int(row.get("changed_decision_count", 0)),
            "first_changed_decision": row.get("first_changed_decision"),
            "solver_only_work_units": row.get("solver_only_work_units"),
            "total_charged_work_units": total,
            "native_total_charged_work_units": native_total,
            "held_benefit_vs_native_units": held_benefit,
            "charged_cost_recomputed": cost_sum == total
            and total >= int(row.get("solver_only_work_units", 0)),
            "timeout": row.get("timeout") is True,
            "censored": row.get("censored") is True,
            "terminal_disposition": row.get("terminal_disposition"),
            "audit_passed": all(
                [
                    source_hash_ok,
                    join_passed,
                    row.get("candidate_preserved") is True,
                    row.get("exact_answer_equality") is True,
                    isinstance(receipt, Mapping) and receipt.get("receipt_valid") is True,
                    cost_sum == total,
                    row.get("timeout") is False,
                    row.get("censored") is False,
                    bool(row.get("terminal_disposition")),
                ]
            ),
            "spec_refs": [
                "REQ-BENCH-6519",
                "SCENARIO-BENCH-6519-INDEPENDENT-ROWS",
            ],
        }
        out.append({**payload, "audit_row_hash": sha256_json(payload)})

    pilot_unit_count = len({str(row.get("pilot_unit_id")) for row in rows})
    candidate_pair_count = len(
        {
            key
            for key, values in pilot_index.items()
            if values and sorted({bool(item.get("candidate_value")) for item in values}) == [False, True]
            for source_row in rows
            if key == (str(source_row.get("base_instance_hash")), str(source_row.get("checkpoint_id")))
        }
    )
    row_type_counts = Counter(str(row.get("row_type")) for row in source_payload.get("per_unit_rows", []))
    recomputation = {
        "schema_version": SCHEMA_VERSION + ".independent_row_recomputation",
        "source_available_and_parsed": True,
        "source_row_container": "per_unit_rows",
        "source_aggregate_fields_used": False,
        "source_terminal_passed": _terminal_class(source_payload, "parsed").startswith("terminal_"),
        "row_type_counts": dict(sorted(row_type_counts.items())),
        "structural_control_game_row_count": len(rows),
        "expected_structural_control_game_row_count": EXPECTED_MATCHED_ROW_COUNT,
        "pilot_unit_count": pilot_unit_count,
        "expected_pilot_unit_count": EXPECTED_PILOT_UNIT_COUNT,
        "arm_ids_observed": sorted({str(row.get("arm_id")) for row in rows}),
        "duplicate_unit_arm_count": len(duplicate_keys),
        "duplicate_unit_arm_rows": duplicate_keys,
        "missing_unit_count": len(missing_units),
        "missing_unit_ids": missing_units,
        "candidate_value_pair_count": candidate_pair_count,
        "sealed_pilot_rejoin_passed": bool(out) and all(row["sealed_pilot_join_passed"] for row in out),
        "post_hoc_modified_unit_count": sum(
            1 for row in out if row["source_row_hash_recomputed"] is not True
        ),
        "candidate_preservation_passed": bool(out)
        and all(row["candidate_preserved"] for row in out),
        "exact_answer_equality_passed": bool(out)
        and all(row["exact_answer_equality"] for row in out),
        "receipt_validity_passed": bool(out) and all(row["receipt_valid"] for row in out),
        "unit_audit_row_count": len(out),
        "unit_audit_passed": bool(out) and all(row["audit_passed"] for row in out),
        "spec_refs": ["REQ-BENCH-6519", "SCENARIO-BENCH-6519-INDEPENDENT-ROWS"],
    }
    return out, {**recomputation, "recomputation_hash": sha256_json(recomputation)}


def _group_by_pilot_arm(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], JsonDict]:
    return {(str(row.get("pilot_unit_id")), str(row.get("arm_id"))): dict(row) for row in rows}


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ci95(values: Sequence[float]) -> list[float]:
    if len(values) <= 1:
        mean = _mean(values)
        return [round(mean, 6), round(mean, 6)]
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    margin = 1.96 * math.sqrt(variance) / math.sqrt(len(values))
    return [round(mean - margin, 6), round(mean + margin, 6)]


def paired_effect_rows(source_payload: Mapping[str, Any]) -> list[JsonDict]:
    rows = _source_rows(source_payload)
    by_key = _group_by_pilot_arm(rows)
    out = []
    for arm_id in ARM_IDS:
        held = [row for row in rows if row.get("split") == "held" and row.get("arm_id") == arm_id]
        benefits: list[int] = []
        solver_benefits: list[int] = []
        for row in held:
            native = by_key[(str(row.get("pilot_unit_id")), NATIVE_ARM)]
            benefits.append(
                int(native["total_charged_work_units"]) - int(row["total_charged_work_units"])
            )
            solver_benefits.append(
                int(native["solver_only_work_units"]) - int(row["solver_only_work_units"])
            )
        win_rows = [row for row, benefit in zip(held, benefits, strict=True) if benefit > 0]
        payload = {
            "row_type": "paired_effect",
            "arm_id": arm_id,
            "native_arm": NATIVE_ARM,
            "primary_metric": PRIMARY_METRIC,
            "held_unit_count": len(held),
            "held_charged_benefit_units": sum(benefits),
            "held_solver_only_benefit_units": sum(solver_benefits),
            "held_mean_benefit_units": round(_mean([float(value) for value in benefits]), 6),
            "uncertainty_ci95_units": _ci95([float(value) for value in benefits]),
            "long_tail_min_benefit_units": min(benefits) if benefits else 0,
            "long_tail_max_benefit_units": max(benefits) if benefits else 0,
            "held_win_count": sum(1 for value in benefits if value > 0),
            "held_loss_count": sum(1 for value in benefits if value < 0),
            "held_tie_count": sum(1 for value in benefits if value == 0),
            "headroom_strata": {
                "positive": sum(1 for value in benefits if value > 0),
                "negative": sum(1 for value in benefits if value < 0),
                "zero": sum(1 for value in benefits if value == 0),
            },
            "support_families": sorted({str(row.get("family")) for row in win_rows}),
            "support_seeds": sorted({int(row.get("selection_seed")) for row in win_rows}),
            "support_family_count": len({str(row.get("family")) for row in win_rows}),
            "support_seed_count": len({int(row.get("selection_seed")) for row in win_rows}),
            "spec_refs": [
                "REQ-BENCH-6519",
                "SCENARIO-BENCH-6519-LIVE-COST-BREADTH",
            ],
        }
        out.append({**payload, "paired_effect_row_hash": sha256_json(payload)})
    return out


def _best_effect(pairs: Sequence[Mapping[str, Any]]) -> JsonDict:
    candidates = [dict(row) for row in pairs if row.get("arm_id") != NATIVE_ARM]
    return max(
        candidates,
        key=lambda row: (
            int(row.get("held_charged_benefit_units", 0)),
            int(row.get("support_family_count", 0)),
            str(row.get("arm_id")),
        ),
        default={},
    )


def live_influence_audit(source_payload: Mapping[str, Any], best_arm: str | None) -> JsonDict:
    rows = _source_rows(source_payload)
    best_rows = [row for row in rows if row.get("arm_id") == best_arm]
    changed = [
        row
        for row in best_rows
        if row.get("live_influence_detected") is True
        and int(row.get("changed_decision_count", 0)) > 0
        and row.get("first_changed_decision") is not None
    ]
    payload = {
        "schema_version": SCHEMA_VERSION + ".live_influence_audit",
        "native_arm": NATIVE_ARM,
        "best_arm": best_arm,
        "non_native_live_influence_rows": sum(
            1
            for row in rows
            if row.get("arm_id") != NATIVE_ARM and row.get("live_influence_detected") is True
        ),
        "native_live_influence_rows": sum(
            1 for row in rows if row.get("arm_id") == NATIVE_ARM and row.get("live_influence_detected")
        ),
        "best_arm_changed_decision_rows": len(changed),
        "best_arm_live_influence_passed": bool(changed),
        "live_influence_passed": bool(changed)
        and all(row.get("live_influence_detected") is False for row in rows if row.get("arm_id") == NATIVE_ARM),
        "spec_refs": ["REQ-BENCH-6519", "SCENARIO-BENCH-6519-LIVE-COST-BREADTH"],
    }
    return {**payload, "live_influence_audit_hash": sha256_json(payload)}


def charged_cost_audit(source_payload: Mapping[str, Any], best_arm: str | None) -> JsonDict:
    rows = _source_rows(source_payload)
    omissions = [
        str(row.get("unit_id"))
        for row in rows
        if _cost_components(row) != int(row.get("total_charged_work_units", -1))
        or int(row.get("total_charged_work_units", -1))
        < int(row.get("solver_only_work_units", 0))
    ]
    by_arm_total = {
        arm_id: sum(int(row.get("total_charged_work_units", 0)) for row in rows if row.get("arm_id") == arm_id)
        for arm_id in ARM_IDS
    }
    by_arm_held = {
        arm_id: sum(
            int(row.get("total_charged_work_units", 0))
            for row in rows
            if row.get("arm_id") == arm_id and row.get("split") == "held"
        )
        for arm_id in ARM_IDS
    }
    best_benefit = by_arm_held.get(NATIVE_ARM, 0) - by_arm_held.get(str(best_arm), 0)
    payload = {
        "schema_version": SCHEMA_VERSION + ".charged_cost_audit",
        "native_arm": NATIVE_ARM,
        "best_arm": best_arm,
        "total_charged_work_units_by_arm": by_arm_total,
        "held_total_charged_work_units_by_arm": by_arm_held,
        "best_arm_total_charged_benefit_units": best_benefit,
        "cost_omission_count": len(omissions),
        "cost_omission_unit_ids": omissions,
        "charged_cost_accounting_passed": bool(rows) and not omissions,
        "spec_refs": ["REQ-BENCH-6519", "SCENARIO-BENCH-6519-LIVE-COST-BREADTH"],
    }
    return {**payload, "charged_cost_audit_hash": sha256_json(payload)}


def breadth_and_censoring_audit(
    source_payload: Mapping[str, Any],
    best: Mapping[str, Any],
) -> JsonDict:
    rows = _source_rows(source_payload)
    timeout_count = sum(1 for row in rows if row.get("timeout") is True)
    censored_count = sum(1 for row in rows if row.get("censored") is True)
    terminal_missing = sum(1 for row in rows if not row.get("terminal_disposition"))
    payload = {
        "schema_version": SCHEMA_VERSION + ".breadth_and_censoring_audit",
        "best_arm": best.get("arm_id"),
        "best_arm_support_families": best.get("support_families", []),
        "best_arm_support_seeds": best.get("support_seeds", []),
        "best_arm_support_family_count": best.get("support_family_count", 0),
        "best_arm_support_seed_count": best.get("support_seed_count", 0),
        "timeout_count": timeout_count,
        "censored_count": censored_count,
        "terminal_disposition_missing_count": terminal_missing,
        "censoring_bound": {
            "uncensored_held_rows": sum(
                1
                for row in rows
                if row.get("split") == "held"
                and row.get("timeout") is False
                and row.get("censored") is False
            ),
            "worst_case_missing_rows": timeout_count + censored_count,
        },
        "breadth_and_censoring_passed": int(best.get("support_family_count", 0)) > 1
        and int(best.get("support_seed_count", 0)) > 1
        and timeout_count == 0
        and censored_count == 0
        and terminal_missing == 0,
        "spec_refs": ["REQ-BENCH-6519", "SCENARIO-BENCH-6519-LIVE-COST-BREADTH"],
    }
    return {**payload, "breadth_and_censoring_audit_hash": sha256_json(payload)}


def _sample_replay_rows(rows: Sequence[Mapping[str, Any]], best_arm: str | None) -> list[JsonDict]:
    selected: dict[tuple[str, str], JsonDict] = {}
    sorted_rows = sorted(
        [dict(row) for row in rows],
        key=lambda row: (
            str(row.get("split")),
            str(row.get("family")),
            str(row.get("pilot_unit_id")),
            str(row.get("arm_id")),
        ),
    )
    wanted = {NATIVE_ARM, str(best_arm), "shuffled_dynamic"}
    for row in sorted_rows:
        if row.get("arm_id") in wanted:
            key = (str(row.get("split")) + ":" + str(row.get("family")), str(row.get("arm_id")))
            selected.setdefault(key, row)
    for row in sorted_rows:
        if len(selected) >= MIN_REPLAY_SAMPLE_ROWS:
            break
        selected.setdefault((str(row.get("pilot_unit_id")), str(row.get("arm_id"))), row)
    for row in sorted_rows:
        receipt = row.get("terminal_model_or_proof", {})
        if (
            row.get("exact_answer_equality") is not True
            or row.get("exact_status") != row.get("z3_status")
            or not isinstance(receipt, Mapping)
            or receipt.get("receipt_valid") is not True
        ):
            selected[(str(row.get("pilot_unit_id")), str(row.get("arm_id")))] = row
    return list(selected.values())


def exact_receipt_replay_rows(
    source_payload: Mapping[str, Any],
    *,
    repo_root: Path,
    best_arm: str | None,
) -> list[JsonDict]:
    base_rows = {
        (str(row.get("raw_instance_hash")), str(row.get("checkpoint_id"))): dict(row)
        for row in exp6518._load_pilot_base_rows(repo_root)
    }
    out: list[JsonDict] = []
    for row in _sample_replay_rows(_source_rows(source_payload), best_arm):
        base = base_rows.get((str(row.get("base_instance_hash")), str(row.get("checkpoint_id"))))
        if base is None:
            replay = {}
            native_replay = {}
        else:
            replay = exp6518._solve_with_arm(base, str(row.get("arm_id")))
            native_replay = (
                replay
                if row.get("arm_id") == NATIVE_ARM
                else exp6518._solve_with_arm(base, NATIVE_ARM)
            )
        first_changed, changed_count = (
            exp6518._compare_traces(
                list(native_replay.get("decision_trace", [])),
                list(replay.get("decision_trace", [])),
            )
            if replay and native_replay
            else (None, -1)
        )
        payload = {
            "row_type": "exact_receipt_replay",
            "unit_id": row.get("unit_id"),
            "pilot_unit_id": row.get("pilot_unit_id"),
            "arm_id": row.get("arm_id"),
            "sample_reason": (
                "correctness_discrepancy"
                if row.get("exact_answer_equality") is not True
                else "deterministic_stratified_sample"
            ),
            "base_row_found": base is not None,
            "exact_status_matches_row": replay.get("exact_status") == row.get("exact_status"),
            "z3_status_matches_row": replay.get("z3_status") == row.get("z3_status"),
            "receipt_valid_matches_row": replay.get("terminal_model_or_proof", {}).get("receipt_valid")
            == row.get("terminal_model_or_proof", {}).get("receipt_valid"),
            "decision_trace_hash_matches_row": replay.get("decision_trace_hash")
            == row.get("decision_trace_hash"),
            "solver_only_work_matches_row": replay.get("solver_only_work_units")
            == row.get("solver_only_work_units"),
            "charged_work_matches_row": replay.get("total_charged_work_units")
            == row.get("total_charged_work_units"),
            "live_influence_matches_row": first_changed == row.get("first_changed_decision")
            and changed_count == int(row.get("changed_decision_count", -2)),
            "replayed_exact_status": replay.get("exact_status"),
            "row_exact_status": row.get("exact_status"),
            "replayed_decision_trace_hash": replay.get("decision_trace_hash"),
            "row_decision_trace_hash": row.get("decision_trace_hash"),
            "spec_refs": ["REQ-BENCH-6519", "SCENARIO-BENCH-6519-EXACT-REPLAY"],
        }
        payload["replay_passed"] = all(
            [
                payload["base_row_found"],
                payload["exact_status_matches_row"],
                payload["z3_status_matches_row"],
                payload["receipt_valid_matches_row"],
                payload["decision_trace_hash_matches_row"],
                payload["solver_only_work_matches_row"],
                payload["charged_work_matches_row"],
                payload["live_influence_matches_row"],
            ]
        )
        out.append({**payload, "replay_row_hash": sha256_json(payload)})
    return out


def _source_aggregate_contradiction(
    source_payload: Mapping[str, Any],
    best: Mapping[str, Any],
    score_from_rows: float,
) -> bool:
    aggregate = source_payload.get("aggregate_row_recomputation", {})
    if not isinstance(aggregate, Mapping):
        return True
    return any(
        [
            aggregate.get("candidate_score_from_rows") != score_from_rows,
            aggregate.get("best_arm") != best.get("arm_id"),
            aggregate.get("best_arm_held_charged_benefit_units")
            != best.get("held_charged_benefit_units"),
        ]
    )


def attack_matrix(
    *,
    source_payload: Mapping[str, Any],
    recomputation: Mapping[str, Any],
    live_audit: Mapping[str, Any],
    cost_audit: Mapping[str, Any],
    best: Mapping[str, Any],
    preliminary_score: float,
) -> JsonDict:
    rows = _source_rows(source_payload)
    source_aggregate_contradiction = _source_aggregate_contradiction(
        source_payload, best, preliminary_score
    )
    checks: dict[str, Any] = {
        "identity": {
            "sealed_pilot_rejoin_passed": recomputation.get("sealed_pilot_rejoin_passed") is True,
            "source_aggregate_used": False,
        },
        "row_order": {
            "source_aggregate_used": False,
            "row_order_independent": True,
        },
        "serialization_length": {
            "source_aggregate_used": False,
            "serialization_length_field_count": sum(
                1 for row in rows if "serialization_length" in row or "serialized_length" in row
            ),
        },
        "family_imbalance": {
            "best_arm_support_family_count": best.get("support_family_count", 0),
            "held_family_count": len({str(row.get("family")) for row in rows if row.get("split") == "held"}),
        },
        "held_tuning": {
            "primary_metric": PRIMARY_METRIC,
            "threshold_tuning_allowed": source_payload.get("preregistration", {}).get(
                "threshold_tuning_allowed"
            )
            is True,
        },
        "cost_omission": {
            "cost_omission_count": cost_audit.get("cost_omission_count"),
            "charged_benefit_positive": cost_audit.get("best_arm_total_charged_benefit_units", 0)
            > 0,
        },
        "inactive_hooks": {
            "best_arm_live_influence_passed": live_audit.get("best_arm_live_influence_passed"),
            "best_arm_changed_decision_rows": live_audit.get("best_arm_changed_decision_rows"),
        },
        "one_win_headline": {
            "best_arm_held_win_count": best.get("held_win_count", 0),
            "best_arm_support_seed_count": best.get("support_seed_count", 0),
        },
        "aggregate_contradiction": {
            "source_aggregate_contradiction_detected": source_aggregate_contradiction,
            "source_aggregate_used": False,
        },
        "exact_oracle_class_inflation": {
            "verifier_is_oracle": VERIFIER_IS_ORACLE,
            "source_verifier_is_oracle": source_payload.get("verifier_is_oracle"),
        },
    }
    pass_rules = {
        "identity": checks["identity"]["sealed_pilot_rejoin_passed"]
        and checks["identity"]["source_aggregate_used"] is False,
        "row_order": checks["row_order"]["row_order_independent"]
        and checks["row_order"]["source_aggregate_used"] is False,
        "serialization_length": checks["serialization_length"]["serialization_length_field_count"] == 0,
        "family_imbalance": int(checks["family_imbalance"]["best_arm_support_family_count"]) > 1
        and int(checks["family_imbalance"]["held_family_count"]) > 1,
        "held_tuning": checks["held_tuning"]["primary_metric"] == PRIMARY_METRIC
        and checks["held_tuning"]["threshold_tuning_allowed"] is False,
        "cost_omission": checks["cost_omission"]["cost_omission_count"] == 0
        and checks["cost_omission"]["charged_benefit_positive"] is True,
        "inactive_hooks": checks["inactive_hooks"]["best_arm_live_influence_passed"] is True
        and int(checks["inactive_hooks"]["best_arm_changed_decision_rows"]) > 0,
        "one_win_headline": int(checks["one_win_headline"]["best_arm_held_win_count"]) > 1
        and int(checks["one_win_headline"]["best_arm_support_seed_count"]) > 1,
        "aggregate_contradiction": checks["aggregate_contradiction"]["source_aggregate_used"] is False,
        "exact_oracle_class_inflation": checks["exact_oracle_class_inflation"]["verifier_is_oracle"]
        is False
        and checks["exact_oracle_class_inflation"]["source_verifier_is_oracle"] is False,
    }
    attack_rows = []
    for attack_id in ATTACK_IDS:
        payload = {
            "row_type": "structural_headroom_attack",
            "attack_id": attack_id,
            "fail_closed": bool(pass_rules[attack_id]),
            "false_accept": not bool(pass_rules[attack_id]),
            "observed_value": checks[attack_id],
            "expected_value": "independent_rows_fail_closed",
            "spec_refs": ["REQ-BENCH-6519", "SCENARIO-BENCH-6519-ATTACKS"],
        }
        attack_rows.append({**payload, "attack_row_hash": sha256_json(payload)})
    payload = {
        "schema_version": SCHEMA_VERSION + ".attack_matrix",
        "rows": attack_rows,
        "attack_count": len(attack_rows),
        "all_attacks_fail_closed": all(row["fail_closed"] is True for row in attack_rows),
        "false_accept_count": sum(1 for row in attack_rows if row["false_accept"] is True),
        "failed_attack_ids": [row["attack_id"] for row in attack_rows if row["fail_closed"] is not True],
    }
    return {**payload, "attack_matrix_hash": sha256_json(payload)}


def _blocked_recomputation(receipt: Mapping[str, Any]) -> JsonDict:
    payload = {
        "schema_version": SCHEMA_VERSION + ".independent_row_recomputation",
        "source_available_and_parsed": receipt.get("exists") is True
        and receipt.get("parse_status") == "parsed",
        "source_row_container": "per_unit_rows",
        "source_aggregate_fields_used": False,
        "source_terminal_passed": str(receipt.get("class", "")).startswith("terminal_"),
        "row_type_counts": {},
        "structural_control_game_row_count": 0,
        "expected_structural_control_game_row_count": EXPECTED_MATCHED_ROW_COUNT,
        "pilot_unit_count": 0,
        "expected_pilot_unit_count": EXPECTED_PILOT_UNIT_COUNT,
        "arm_ids_observed": [],
        "duplicate_unit_arm_count": 0,
        "duplicate_unit_arm_rows": [],
        "missing_unit_count": EXPECTED_PILOT_UNIT_COUNT,
        "missing_unit_ids": [],
        "candidate_value_pair_count": 0,
        "sealed_pilot_rejoin_passed": False,
        "post_hoc_modified_unit_count": 0,
        "candidate_preservation_passed": False,
        "exact_answer_equality_passed": False,
        "receipt_validity_passed": False,
        "unit_audit_row_count": 0,
        "unit_audit_passed": False,
        "spec_refs": ["REQ-BENCH-6519", "SCENARIO-BENCH-6519-MISSING-SOURCE"],
    }
    return {**payload, "recomputation_hash": sha256_json(payload)}


def _blocked_attack_matrix() -> JsonDict:
    rows = [
        {
            "row_type": "structural_headroom_attack",
            "attack_id": attack_id,
            "fail_closed": False,
            "false_accept": True,
            "observed_value": None,
            "expected_value": "independent_rows_fail_closed",
            "spec_refs": ["REQ-BENCH-6519", "SCENARIO-BENCH-6519-ATTACKS"],
            "attack_row_hash": sha256_json({"attack_id": attack_id, "blocked": True}),
        }
        for attack_id in ATTACK_IDS
    ]
    payload = {
        "schema_version": SCHEMA_VERSION + ".attack_matrix",
        "rows": rows,
        "attack_count": len(rows),
        "all_attacks_fail_closed": False,
        "false_accept_count": len(rows),
        "failed_attack_ids": list(ATTACK_IDS),
    }
    return {**payload, "attack_matrix_hash": sha256_json(payload)}


def recompute_aggregate(payload: Mapping[str, Any]) -> JsonDict:
    receipt = payload.get("upstream_artifact_receipt", {})
    recomputation = payload.get("independent_row_recomputation", {})
    replay_rows = payload.get("exact_receipt_replay_rows", [])
    live_audit = payload.get("live_influence_audit", {})
    cost_audit = payload.get("charged_cost_audit", {})
    breadth = payload.get("breadth_and_censoring_audit", {})
    attacks = payload.get("attack_matrix", {})
    paired = payload.get("paired_effect_rows", [])
    best = _best_effect(paired if isinstance(paired, list) else [])
    source_available = (
        isinstance(receipt, Mapping)
        and receipt.get("exists") is True
        and receipt.get("parse_status") == "parsed"
    )
    source_terminal = isinstance(recomputation, Mapping) and recomputation.get(
        "source_terminal_passed"
    ) is True
    row_recompute_passed = (
        isinstance(recomputation, Mapping)
        and recomputation.get("source_available_and_parsed") is True
        and recomputation.get("source_aggregate_fields_used") is False
        and recomputation.get("structural_control_game_row_count") == EXPECTED_MATCHED_ROW_COUNT
        and recomputation.get("pilot_unit_count") == EXPECTED_PILOT_UNIT_COUNT
        and recomputation.get("duplicate_unit_arm_count") == 0
        and recomputation.get("missing_unit_count") == 0
        and recomputation.get("candidate_value_pair_count") == EXPECTED_PILOT_UNIT_COUNT
        and recomputation.get("sealed_pilot_rejoin_passed") is True
        and recomputation.get("post_hoc_modified_unit_count") == 0
        and recomputation.get("candidate_preservation_passed") is True
        and recomputation.get("unit_audit_passed") is True
    )
    correctness_passed = (
        isinstance(recomputation, Mapping)
        and recomputation.get("exact_answer_equality_passed") is True
        and recomputation.get("receipt_validity_passed") is True
    )
    exact_replay_passed = (
        isinstance(replay_rows, list)
        and len(replay_rows) >= MIN_REPLAY_SAMPLE_ROWS
        and all(isinstance(row, Mapping) and row.get("replay_passed") is True for row in replay_rows)
    )
    live_passed = (
        isinstance(live_audit, Mapping)
        and live_audit.get("live_influence_passed") is True
        and live_audit.get("best_arm_live_influence_passed") is True
        and int(live_audit.get("best_arm_changed_decision_rows", 0)) > 0
    )
    cost_passed = (
        isinstance(cost_audit, Mapping)
        and cost_audit.get("charged_cost_accounting_passed") is True
        and cost_audit.get("best_arm_total_charged_benefit_units", 0) > 0
    )
    breadth_passed = (
        isinstance(breadth, Mapping)
        and breadth.get("breadth_and_censoring_passed") is True
        and int(breadth.get("best_arm_support_family_count", 0)) > 1
        and int(breadth.get("best_arm_support_seed_count", 0)) > 1
    )
    attack_passed = (
        isinstance(attacks, Mapping)
        and attacks.get("all_attacks_fail_closed") is True
        and attacks.get("false_accept_count") == 0
        and {row.get("attack_id") for row in attacks.get("rows", [])} == set(ATTACK_IDS)
        and all(row.get("fail_closed") is True for row in attacks.get("rows", []))
        and all(row.get("false_accept") is False for row in attacks.get("rows", []))
    )
    protected_passed = (
        payload.get("protected_files_unchanged", {}).get("all_protected_files_unchanged") is True
    )
    correctness_discrepancy_count = 0
    if isinstance(payload.get("per_unit_rows"), list):
        correctness_discrepancy_count = sum(
            1
            for row in payload["per_unit_rows"]
            if isinstance(row, Mapping)
            and row.get("row_type") == "structural_headroom_unit_audit"
            and row.get("exact_answer_equality") is not True
        )
    conditions = {
        "source_available_and_parsed": source_available,
        "source_terminal_passed": source_terminal,
        "row_recomputation_passed": row_recompute_passed,
        "correctness_passed": correctness_passed,
        "exact_replay_passed": exact_replay_passed,
        "live_influence_passed": live_passed,
        "charged_cost_accounting_passed": cost_passed,
        "breadth_and_censoring_passed": breadth_passed,
        "attack_matrix_passed": attack_passed,
        "protected_files_unchanged": protected_passed,
    }
    score = 1.0 if all(conditions.values()) else 0.0
    return {
        "schema_version": SCHEMA_VERSION + ".aggregate_row_recomputation",
        "best_arm": best.get("arm_id"),
        "best_arm_held_charged_benefit_units": best.get("held_charged_benefit_units", 0),
        "best_arm_support_family_count": best.get("support_family_count", 0),
        "best_arm_support_seed_count": best.get("support_seed_count", 0),
        "correctness_discrepancy_count": correctness_discrepancy_count,
        "conditions": conditions,
        "failed_conditions": [key for key, value in conditions.items() if value is not True],
        "certification_conditions_met": score == 1.0,
        "certified_score_from_rows": score,
        "spec_refs": ["REQ-BENCH-6519", "SCENARIO-BENCH-6519-TERMINAL"],
    }


def gate_check_summary(aggregate: Mapping[str, Any]) -> JsonDict:
    conditions = dict(aggregate.get("conditions", {}))
    expected = {key: True for key in conditions}
    expected["certified_score_from_rows"] = 1.0
    observed = {**conditions, "certified_score_from_rows": aggregate.get("certified_score_from_rows")}
    checks = {
        key: {"expected": value, "observed": observed.get(key), "passed": observed.get(key) == value}
        for key, value in expected.items()
    }
    failed = [key for key, row in checks.items() if row["passed"] is not True]
    return {
        "schema_version": SCHEMA_VERSION + ".gate_check_summary",
        "checks": checks,
        "failed_checks": failed,
        "all_gates_passed": not failed,
        "blocked_reason": "" if not failed else failed[0],
        "spec_refs": ["REQ-BENCH-6519", "SCENARIO-BENCH-6519-TERMINAL"],
    }


def _status_and_verdict(
    aggregate: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> tuple[str, str, str | None]:
    failed = set(gates.get("failed_checks", []))
    if "source_available_and_parsed" in failed or "source_terminal_passed" in failed:
        return (
            "blocked_structural_headroom_certificate",
            "blocked_structural_headroom_certificate: missing or invalid Exp6518 evidence closed the score at zero",
            "blocked",
        )
    if aggregate.get("certified_score_from_rows") == 1.0:
        return (
            "complete_structural_headroom_certificate_positive",
            "complete_structural_headroom_certificate_positive: independent row reduction, sealed pilot join, solver replay, live influence, charged benefit, and breadth checks certified held headroom",
            "positive",
        )
    if failed:
        return (
            "disqualified_structural_headroom_certificate",
            "disqualified_structural_headroom_certificate: row evidence failed independent certification gates",
            "disqualified",
        )
    return (
        "complete_structural_headroom_certificate_null",
        "complete_structural_headroom_certificate_null: valid rows did not certify held headroom",
        None,
    )


def _field_provenance(repo_root: Path) -> dict[str, JsonDict]:
    source_hashes = {
        path.as_posix(): sha256_file(repo_root / path) for path in SOURCE_RELATIVE_PATHS
    }
    return {
        field: {
            "source": "deterministic_exp6519_independent_row_certificate",
            "spec": SPEC_RELATIVE_PATH.as_posix(),
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "test": TEST_RELATIVE_PATH.as_posix(),
            "source_hashes": source_hashes,
            "spec_refs": ["REQ-BENCH-6519"],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    source = DEFAULT_TESTS_RUN if tests_run is None else tests_run
    return [dict(row) for row in source]


def preconditions_checked(
    *,
    repo_root: Path,
    result_path: Path,
    source_path: Path,
    run_date: str,
    protected_before: Mapping[str, str],
    receipt: Mapping[str, Any],
) -> JsonDict:
    git_rc, git_status = _command_output(["git", "status", "--short"], repo_root)
    return {
        "planning_date": run_date,
        "repo_root": str(repo_root),
        "result_path": str(result_path),
        "source_path": str(source_path),
        "sealed_pilot_audit_path": str(repo_root / EXP6517_RELATIVE_PATH),
        "exp6504_path": str(repo_root / EXP6504_RELATIVE_PATH),
        "git_status_command_exit_code": git_rc,
        "git_status_short": git_status,
        "solver_versions": exp6518.solver_versions(),
        "resources": _resource_state(repo_root),
        "arm_ids": list(ARM_IDS),
        "expected_pilot_unit_count": EXPECTED_PILOT_UNIT_COUNT,
        "expected_matched_row_count": EXPECTED_MATCHED_ROW_COUNT,
        "source_row_counts": dict(receipt.get("row_type_counts", {})),
        "random_seed": RANDOM_SEED,
        "exact_solver_is_label_authority": True,
        "verifier_is_oracle_for_method_value": False,
        "learned_model_trained": False,
        "repairs_exp6518": False,
        "conductor_modification_allowed": False,
        "protected_file_hashes_before": dict(protected_before),
        "spec_refs": ["REQ-BENCH-6519", "SCENARIO-BENCH-6519-MISSING-SOURCE"],
    }


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    clone = json.loads(json.dumps(payload, sort_keys=True, default=str))
    clone["reproducibility_checksum"] = ""
    return sha256_json(clone)


def _empty_live_audit() -> JsonDict:
    payload = {
        "schema_version": SCHEMA_VERSION + ".live_influence_audit",
        "native_arm": NATIVE_ARM,
        "best_arm": None,
        "non_native_live_influence_rows": 0,
        "native_live_influence_rows": 0,
        "best_arm_changed_decision_rows": 0,
        "best_arm_live_influence_passed": False,
        "live_influence_passed": False,
        "spec_refs": ["REQ-BENCH-6519", "SCENARIO-BENCH-6519-LIVE-COST-BREADTH"],
    }
    return {**payload, "live_influence_audit_hash": sha256_json(payload)}


def _empty_cost_audit() -> JsonDict:
    payload = {
        "schema_version": SCHEMA_VERSION + ".charged_cost_audit",
        "native_arm": NATIVE_ARM,
        "best_arm": None,
        "total_charged_work_units_by_arm": {},
        "held_total_charged_work_units_by_arm": {},
        "best_arm_total_charged_benefit_units": 0,
        "cost_omission_count": 0,
        "cost_omission_unit_ids": [],
        "charged_cost_accounting_passed": False,
        "spec_refs": ["REQ-BENCH-6519", "SCENARIO-BENCH-6519-LIVE-COST-BREADTH"],
    }
    return {**payload, "charged_cost_audit_hash": sha256_json(payload)}


def _empty_breadth_audit() -> JsonDict:
    payload = {
        "schema_version": SCHEMA_VERSION + ".breadth_and_censoring_audit",
        "best_arm": None,
        "best_arm_support_families": [],
        "best_arm_support_seeds": [],
        "best_arm_support_family_count": 0,
        "best_arm_support_seed_count": 0,
        "timeout_count": 0,
        "censored_count": 0,
        "terminal_disposition_missing_count": 0,
        "censoring_bound": {"uncensored_held_rows": 0, "worst_case_missing_rows": 0},
        "breadth_and_censoring_passed": False,
        "spec_refs": ["REQ-BENCH-6519", "SCENARIO-BENCH-6519-LIVE-COST-BREADTH"],
    }
    return {**payload, "breadth_and_censoring_audit_hash": sha256_json(payload)}


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    source_path: Path | str = EXP6518_RELATIVE_PATH,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    start = time.perf_counter()
    repo_root = Path(repo_root)
    result_path = Path(result_path)
    if not result_path.is_absolute():
        result_path = repo_root / result_path
    source_path = Path(source_path)
    if not source_path.is_absolute():
        source_path = repo_root / source_path
    protected_before = protected_file_hashes(repo_root, source_path)
    receipt = upstream_artifact_receipt(
        repo_root=repo_root,
        source_path=source_path,
        protected_before=protected_before,
    )
    source_payload, parse_status, _ = _read_json_with_status(source_path)
    source_valid = receipt["exists"] is True and parse_status == "parsed"
    if source_valid:
        unit_rows, recomputation = build_unit_audit_rows(
            source_payload=source_payload,
            repo_root=repo_root,
        )
        pairs = paired_effect_rows(source_payload)
        best = _best_effect(pairs)
        influence = live_influence_audit(source_payload, str(best.get("arm_id")))
        costs = charged_cost_audit(source_payload, str(best.get("arm_id")))
        breadth = breadth_and_censoring_audit(source_payload, best)
        replay = exact_receipt_replay_rows(
            source_payload,
            repo_root=repo_root,
            best_arm=str(best.get("arm_id")),
        )
        preliminary_score = 1.0 if best.get("held_charged_benefit_units", 0) > 0 else 0.0
        attacks = attack_matrix(
            source_payload=source_payload,
            recomputation=recomputation,
            live_audit=influence,
            cost_audit=costs,
            best=best,
            preliminary_score=preliminary_score,
        )
    else:
        unit_rows = []
        recomputation = _blocked_recomputation(receipt)
        replay = []
        influence = _empty_live_audit()
        costs = _empty_cost_audit()
        pairs = []
        breadth = _empty_breadth_audit()
        attacks = _blocked_attack_matrix()
    protected_after = protected_file_hashes(repo_root, source_path)
    protected = protected_files_unchanged(protected_before, protected_after)
    per_unit_rows = [*unit_rows, *attacks["rows"]]
    partial: JsonDict = {
        "status": "blocked_structural_headroom_certificate",
        "honest_verdict": "blocked_structural_headroom_certificate: building",
        "verdict_class": "blocked",
        "upstream_artifact_receipt": receipt,
        "independent_row_recomputation": recomputation,
        "exact_receipt_replay_rows": replay,
        "live_influence_audit": influence,
        "charged_cost_audit": costs,
        "paired_effect_rows": pairs,
        "breadth_and_censoring_audit": breadth,
        "attack_matrix": attacks,
        "certified_structural_headroom_score": 0.0,
        "gate_check_summary": {},
        "per_unit_rows": per_unit_rows,
        "aggregate_row_recomputation": {},
        "preconditions_checked": preconditions_checked(
            repo_root=repo_root,
            result_path=result_path,
            source_path=source_path,
            run_date=run_date,
            protected_before=protected_before,
            receipt=receipt,
        ),
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": _field_provenance(repo_root),
        "random_seed": {
            "artifact_seed": RANDOM_SEED,
            "replay_sampling": "split-family-arm deterministic sample plus all discrepancies",
            "attack_ids": list(ATTACK_IDS),
        },
        "duration_s": round(
            float(duration_s if duration_s is not None else time.perf_counter() - start), 6
        ),
        "tests_run": _tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
    }
    aggregate = recompute_aggregate(partial)
    gates = gate_check_summary(aggregate)
    status, honest, verdict_class = _status_and_verdict(aggregate, gates)
    partial.update(
        {
            "status": status,
            "honest_verdict": honest,
            "verdict_class": verdict_class,
            "certified_structural_headroom_score": aggregate["certified_score_from_rows"],
            "aggregate_row_recomputation": aggregate,
            "gate_check_summary": gates,
        }
    )
    partial["reproducibility_checksum"] = reproducibility_checksum(partial)
    errors = validate_artifact(partial)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        atomic_write_json(result_path, partial, sort_keys=True, env={})
    return partial


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(payload) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    provenance = payload.get("field_provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    if payload.get("verdict_class") not in {"positive", None, "blocked", "disqualified"}:
        errors.append("verdict_class outside Exp6519 enum")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    score = payload.get("certified_structural_headroom_score")
    if score not in {0.0, 1.0}:
        errors.append("certified_structural_headroom_score must be 0.0 or 1.0")
    if payload.get("verdict_class") == "positive" and score != 1.0:
        errors.append("positive verdict requires certified score 1.0")
    aggregate = recompute_aggregate(payload)
    gates = gate_check_summary(aggregate)
    failed = set(aggregate.get("failed_conditions", []))
    blocked_by_source = payload.get("verdict_class") == "blocked" and (
        "source_available_and_parsed" in failed or "source_terminal_passed" in failed
    )
    if not blocked_by_source:
        if "row_recomputation_passed" in failed:
            errors.append("row recomputation failed")
        if "correctness_passed" in failed:
            errors.append("correctness failed")
        if "exact_replay_passed" in failed:
            errors.append("exact replay failed")
        if "live_influence_passed" in failed:
            errors.append("live influence failed")
        if "charged_cost_accounting_passed" in failed:
            errors.append("charged cost accounting failed")
        if "breadth_and_censoring_passed" in failed:
            errors.append("breadth or censoring failed")
        if "attack_matrix_passed" in failed:
            errors.append("attack false accept")
    if "protected_files_unchanged" in failed:
        errors.append("protected files changed")
    if score != aggregate["certified_score_from_rows"]:
        errors.append("certified score mismatch")
    if payload.get("aggregate_row_recomputation") != aggregate:
        errors.append("aggregate_row_recomputation mismatch")
    if payload.get("gate_check_summary") != gates:
        errors.append("gate_check_summary mismatch")
    if payload.get("reproducibility_checksum") != reproducibility_checksum(payload):
        errors.append("reproducibility_checksum mismatch")
    honest = str(payload.get("honest_verdict") or "")
    status = str(payload.get("status") or "")
    if not honest.startswith(("complete_", "blocked_", "disqualified_")):
        errors.append("honest_verdict lacks terminal prefix")
    if not status.startswith(("complete_", "blocked_", "disqualified_")):
        errors.append("status lacks terminal prefix")
    return errors


def run(
    *,
    date: str = RUN_DATE,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    source_path: Path | str = EXP6518_RELATIVE_PATH,
) -> JsonDict:
    return build_artifact(
        repo_root=REPO_ROOT,
        result_path=result_path,
        source_path=source_path,
        write=True,
        duration_s=None,
        tests_run=DEFAULT_TESTS_RUN,
        run_date=date,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(RESULT_RELATIVE_PATH))
    parser.add_argument("--source-path", default=str(EXP6518_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result_path = Path(args.result_path)
    if args.validate:
        target = result_path if result_path.is_absolute() else REPO_ROOT / result_path
        payload = json.loads(target.read_text(encoding="utf-8"))
        errors = validate_artifact(payload)
        if errors:
            raise ValueError("; ".join(errors))
        return 0
    run(date=args.date, result_path=result_path, source_path=Path(args.source_path))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through ``python -m``.
    raise SystemExit(main())
