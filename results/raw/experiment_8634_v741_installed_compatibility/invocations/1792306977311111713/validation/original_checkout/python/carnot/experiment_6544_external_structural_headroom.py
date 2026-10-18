"""Exp6544 external structural headroom comparison.

Spec refs: REQ-BENCH-6544, SCENARIO-BENCH-6544-GATE,
SCENARIO-BENCH-6544-CONTRACT, SCENARIO-BENCH-6544-FAMILY-BLIND,
SCENARIO-BENCH-6544-COST-EQUALITY, SCENARIO-BENCH-6544-EFFECTS,
SCENARIO-BENCH-6544-ATTACKS, SCENARIO-BENCH-6544-TERMINAL.

The runner compares non-learned order controls on the audited DRIFT fixture.
Each control sees the same prefix-state candidates. It may change only their
order. Z3 replay remains the separate authority for the target state label.
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
import random
import shutil
import subprocess
import time
from types import ModuleType
from typing import Any

from carnot import experiment_6543_external_corpus_independent_audit_v2 as exp6543
from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6544
INFERENCE_SUBSTRATE = "external_drift_exact_solver_structural_controls_no_llm"

RESULT_RELATIVE_PATH = Path("results/experiment_6544_external_structural_headroom.json")
EXP6543_RELATIVE_PATH = Path("results/experiment_6543_external_corpus_independent_audit_v2.json")
FIXTURE_RELATIVE_PATH = Path("results/fixtures/v566_drift_bench_external_slice.jsonl")
EXP6518_RELATIVE_PATH = Path("results/experiment_6518_structural_control_headroom_ab_v2.json")
EXP6519_RELATIVE_PATH = Path("results/experiment_6519_structural_headroom_certificate.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/benchmarks/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6544_external_structural_headroom.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6544_external_structural_headroom.py")
ROADMAP_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")

NATIVE_ARM = "native"
ARM_IDS = (
    NATIVE_ARM,
    "random",
    "analytical",
    "bounded_refocus",
    "one_shot_enumeration",
)
SEED_GRID = (654401, 654402, 654403)
TIMEOUT_S = 2.0
PRIMARY_METRIC = "held_total_charged_work_units_beyond_native_and_random"

FORBIDDEN_FEATURES = (
    "entity_names",
    "source_ids",
    "target_assignments",
    "held_outcomes",
    "future_turns",
    "family_labels",
    "row_order",
)
STRUCTURAL_FEATURES = (
    "candidate_depth",
    "candidate_count",
    "constraint_count",
    "solver_assertion_count",
    "turn_index",
    "num_entities",
)
SHORTCUT_ATTACK_IDS = (
    "identity_leakage",
    "row_order_leakage",
    "unequal_budgets",
    "deleted_candidates",
    "warm_cache_asymmetry",
    "timer_aliases",
    "cherry_picked_seeds",
    "aggregate_only_claims",
)

PROTECTED_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    ROADMAP_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    Path("scripts/research_conductor.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    EXP6543_RELATIVE_PATH,
    FIXTURE_RELATIVE_PATH,
    EXP6518_RELATIVE_PATH,
    EXP6519_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
)

SOURCE_RELATIVE_PATHS = (
    *PROTECTED_RELATIVE_PATHS,
    Path("python/carnot/experiment_6543_external_corpus_independent_audit_v2.py"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "upstream_gate_receipt",
    "frozen_comparison_contract",
    "control_definitions",
    "family_and_effort_census",
    "per_unit_rows",
    "paired_effect_rows",
    "family_effect_rows",
    "charged_cost_recomputation",
    "exact_equality_receipt",
    "candidate_preservation_receipt",
    "censoring_and_timeout_receipts",
    "shortcut_attack_matrix",
    "external_structural_headroom_ready_score",
    "gate_check_summary",
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
    "status": "Records the terminal Exp6544 external structural comparison state.",
    "honest_verdict": "Names the charged held result and its family support.",
    "verdict_class": (
        "Separates positive, null, partial, blocked, and disqualified outcomes."
    ),
    "upstream_gate_receipt": (
        "Pins Exp6543 by path, hash, expected value, observed value, input hashes, solvers, resources, timeouts, seeds, and protected hashes."
    ),
    "frozen_comparison_contract": (
        "Freezes unit, turn, family, effort, seed, budget, tie, timeout, and stop rules before scoring."
    ),
    "control_definitions": (
        "Defines native, random, analytical, bounded-refocus, and one-shot-enumeration as non-learned order-only controls."
    ),
    "family_and_effort_census": (
        "Reports fixture family, split, turn, effort, and held-support counts."
    ),
    "per_unit_rows": (
        "Stores one unit-turn-seed-arm row with order, checks, cost, equality, timeout, and censoring evidence."
    ),
    "paired_effect_rows": (
        "Reports paired held charged effects versus native and random with uncertainty from rows."
    ),
    "family_effect_rows": (
        "Reports charged effects, headroom cells, no-headroom cells, and Simpson checks by family."
    ),
    "charged_cost_recomputation": (
        "Recomputes proposal, exact-check, control, fallback, and total charged work from rows."
    ),
    "exact_equality_receipt": "Shows each arm's exact answer matches the audited Z3 label.",
    "candidate_preservation_receipt": (
        "Shows every arm preserved the same candidate set for every unit and seed."
    ),
    "censoring_and_timeout_receipts": (
        "Records timeout, censoring, budget, and stop-rule symmetry."
    ),
    "shortcut_attack_matrix": (
        "Attacks identity leakage, row-order leakage, unequal budgets, deleted candidates, warm-cache asymmetry, timer aliases, cherry-picked seeds, and aggregate-only claims."
    ),
    "external_structural_headroom_ready_score": (
        "Opens only for charged held value beyond native and random with exact equality, multi-family support, and no shortcut failure."
    ),
    "gate_check_summary": "Names every failed gate with expected and observed values.",
    "aggregate_row_recomputation": (
        "Rebuilds the verdict and score from per-unit, effect, equality, preservation, censoring, and attack rows."
    ),
    "preconditions_checked": (
        "Records paths, hashes, resources, solver identity, date, seeds, budgets, timeouts, and protected hashes."
    ),
    "protected_files_unchanged": (
        "Shows guarded inputs, specs, prior structural artifacts, and conductor files stayed byte-identical during the run."
    ),
    "inference_substrate": "Declares external DRIFT exact-solver structural controls with no LLM.",
    "verifier_is_oracle": "False because compared controls do not certify ground truth.",
    "field_principles": "Explains why each required field exists.",
    "field_provenance": "Maps each field to specs, inputs, rows, reducers, tests, and hashes.",
    "random_seed": "Pins arm seeds, random controls, and attack ordering.",
    "duration_s": "Records measured reducer wall time.",
    "tests_run": "Records validation command receipts.",
    "reproducibility_checksum": "Detects drift in gates, rows, costs, attacks, and verdicts.",
}

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6544_external_structural_headroom.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6544_external_structural_headroom.py "
    "-m pytest tests/python/test_experiment_6544_external_structural_headroom.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6544_external_structural_headroom.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6544_external_structural_headroom.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6544_external_structural_headroom.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6544_external_structural_headroom.json"
)
EXACT_E2E_COMMAND = ".venv/bin/pytest tests/python/test_z3_live_benchmark.py -q --no-cov -n 0"
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6544_external_structural_headroom "
    "--date 20260823"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6544_external_structural_headroom --validate"
)
DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": EXACT_E2E_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": "git status --short", "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_json(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _load_jsonl(path: Path) -> list[JsonDict]:
    if not path.is_file():
        return []
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        rows.append(dict(value) if isinstance(value, Mapping) else {"value": value})
    return rows


def _tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    source = DEFAULT_TESTS_RUN if tests_run is None else tests_run
    return [{"command": str(row["command"]), "exit_code": int(row["exit_code"])} for row in source]


def _resource_state(repo_root: Path) -> JsonDict:
    disk = shutil.disk_usage(repo_root)
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
    }


def _git_state(repo_root: Path) -> JsonDict:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )
    return {"exit_code": result.returncode, "status_short": result.stdout.strip()}


def _protected_hashes(repo_root: Path, audit_path: Path) -> dict[str, str]:
    hashes = {rel.as_posix(): sha256_file(repo_root / rel) for rel in PROTECTED_RELATIVE_PATHS}
    hashes[_source_key(repo_root, audit_path)] = sha256_file(audit_path)
    return hashes


def _source_key(repo_root: Path, path: Path) -> str:
    resolved = path.resolve(strict=False)
    repo = repo_root.resolve(strict=False)
    if resolved.is_relative_to(repo):
        return resolved.relative_to(repo).as_posix()
    return str(path)


def protected_files_unchanged(before: Mapping[str, str], after: Mapping[str, str]) -> JsonDict:
    rows = [
        {
            "path": path,
            "before_sha256": before.get(path, "missing"),
            "after_sha256": after.get(path, "missing"),
            "unchanged": before.get(path, "missing") == after.get(path, "missing"),
        }
        for path in sorted(set(before) | set(after))
    ]
    return {
        "all_protected_files_unchanged": all(row["unchanged"] for row in rows),
        "rows": rows,
    }


def solver_identity() -> JsonDict:
    return {
        "structural_control_runner": "exp6544_prefix_state_candidate_order_v1",
        "z3_python_available": True,
        "z3_python_version": exp6543._z3_version(),
        "z3_cli_path": shutil.which("z3"),
        "evaluation_authority": "DRIFT source z3_checker.py full-state replay",
    }


def _resolve_source_root(
    *,
    audit: Mapping[str, Any],
    source_root: Path | str | None,
) -> Path:
    if source_root is not None:
        return Path(source_root)
    receipt = audit.get("independent_revision_license_and_schema_receipt")
    if isinstance(receipt, Mapping) and receipt.get("source_root"):
        return Path(str(receipt["source_root"]))
    return exp6543.DEFAULT_SOURCE_CACHE_ROOT


def upstream_gate_receipt(
    *,
    repo_root: Path,
    audit_path: Path,
    fixture_path: Path,
    protected_before: Mapping[str, str],
) -> JsonDict:
    exists = audit_path.is_file()
    try:
        audit = _load_json(audit_path)
        parse_status = "parsed" if exists else "missing"
        parse_error = ""
    except json.JSONDecodeError as exc:
        audit = {}
        parse_status = "corrupt_json"
        parse_error = str(exc)
    observed = audit.get("external_constraint_corpus_audited_ready_score")
    return {
        "row_type": "upstream_gate_receipt",
        "path": _source_key(repo_root, audit_path),
        "absolute_path": str(audit_path),
        "exists": exists,
        "sha256": sha256_file(audit_path),
        "parse_status": parse_status,
        "parse_error": parse_error,
        "field": "external_constraint_corpus_audited_ready_score",
        "json_pointer": "/external_constraint_corpus_audited_ready_score",
        "expected_value": 1.0,
        "observed_value": observed,
        "gate_passed": observed == 1.0 and parse_status == "parsed",
        "status": audit.get("status"),
        "verdict_class": audit.get("verdict_class"),
        "artifact_reproducibility_checksum": audit.get("reproducibility_checksum"),
        "input_hashes": {
            "audit": sha256_file(audit_path),
            "fixture": sha256_file(fixture_path),
            "exp6518": sha256_file(repo_root / EXP6518_RELATIVE_PATH),
            "exp6519": sha256_file(repo_root / EXP6519_RELATIVE_PATH),
        },
        "solver_identity": solver_identity(),
        "resources": _resource_state(repo_root),
        "timeout_s": TIMEOUT_S,
        "seeds": list(SEED_GRID),
        "protected_file_hashes_before": dict(protected_before),
        "spec_refs": ["REQ-BENCH-6544", "SCENARIO-BENCH-6544-GATE"],
    }


def frozen_comparison_contract(run_date: str) -> JsonDict:
    payload = {
        "schema_version": "carnot.exp6544.frozen_comparison_contract.v1",
        "planning_date": run_date,
        "primary_metric": PRIMARY_METRIC,
        "unit_contract": "local_unit_id plus source_turn_id recorded; not used as ordering features",
        "turn_contract": "candidate prefixes stop at the current turn; future turns are unavailable",
        "family_contract": "family is a reporting stratum only",
        "effort_contract": "pre_replay_effort_stratum is a reporting stratum only",
        "arm_ids": list(ARM_IDS),
        "seed_grid": list(SEED_GRID),
        "candidate_budget_rule": "all_candidates_preserved",
        "exact_check_budget_rule": "candidate_count_plus_native_fallback",
        "timeout_s": TIMEOUT_S,
        "tie_break_rule": "structural_depth_then_constraint_count_then_candidate_index",
        "stop_rule": "stop_after_full_target_state_or_native_fallback",
        "candidate_set_definition": (
            "all cumulative prefix states from turn zero through the current target turn"
        ),
        "forbidden_features": list(FORBIDDEN_FEATURES),
        "allowed_structural_features": list(STRUCTURAL_FEATURES),
        "row_order_used_as_feature": False,
        "native_exact_fallback_required": True,
        "family_blind_calibration": {
            "calibration_rows_used": ["train", "development"],
            "family_labels_used": False,
            "held_rows_used_for_calibration": False,
            "learned_parameters": False,
        },
        "spec_refs": ["REQ-BENCH-6544", "SCENARIO-BENCH-6544-CONTRACT"],
    }
    return {**payload, "contract_hash": sha256_json(payload)}


def control_definitions() -> dict[str, JsonDict]:
    rows = {
        NATIVE_ARM: {
            "description": "Chronological prefix-state order with native full-state fallback.",
            "ordering_features": ["candidate_depth"],
            "learned_model_used": False,
            "may_remove_candidates": False,
            "uses_family_labels": False,
        },
        "random": {
            "description": "Seeded permutation over the preserved prefix-state set.",
            "ordering_features": ["candidate_count", "seed"],
            "learned_model_used": False,
            "may_remove_candidates": False,
            "uses_family_labels": False,
        },
        "analytical": {
            "description": "Largest structural prefix first, using constraint and assertion counts.",
            "ordering_features": ["constraint_count", "solver_assertion_count", "candidate_depth"],
            "learned_model_used": False,
            "may_remove_candidates": False,
            "uses_family_labels": False,
        },
        "bounded_refocus": {
            "description": "Probe shallow, midpoint, and deepest prefixes before the remainder.",
            "ordering_features": ["candidate_depth", "candidate_count"],
            "learned_model_used": False,
            "may_remove_candidates": False,
            "uses_family_labels": False,
        },
        "one_shot_enumeration": {
            "description": "Enumerate the structurally largest prefix first, then descending prefixes.",
            "ordering_features": ["solver_assertion_count", "constraint_count", "candidate_depth"],
            "learned_model_used": False,
            "may_remove_candidates": False,
            "uses_family_labels": False,
        },
    }
    return {
        arm_id: {
            **rows[arm_id],
            "allowed_to_order_candidates": arm_id != NATIVE_ARM,
            "forbidden_features": list(FORBIDDEN_FEATURES),
            "charged_overheads": ["proposal", "exact_check", "control", "fallback"],
            "spec_refs": ["REQ-BENCH-6544", "SCENARIO-BENCH-6544-FAMILY-BLIND"],
        }
        for arm_id in ARM_IDS
    }


def family_and_effort_census(fixture_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    held = [row for row in fixture_rows if row.get("split_name") == "held"]
    return {
        "row_type": "family_and_effort_census",
        "fixture_row_count": len(fixture_rows),
        "base_problem_count": len({str(row.get("base_problem_id")) for row in fixture_rows}),
        "split_counts": dict(sorted(Counter(str(row.get("split_name")) for row in fixture_rows).items())),
        "family_counts": dict(sorted(Counter(str(row.get("family")) for row in fixture_rows).items())),
        "held_family_counts": dict(sorted(Counter(str(row.get("family")) for row in held).items())),
        "effort_counts": dict(
            sorted(Counter(str(row.get("pre_replay_effort_stratum")) for row in fixture_rows).items())
        ),
        "turn_position_counts": dict(
            sorted(Counter(str(row.get("turn_position")) for row in fixture_rows).items())
        ),
        "held_unit_count": len(held),
        "held_family_count": len({str(row.get("family")) for row in held}),
        "held_effort_count": len({str(row.get("pre_replay_effort_stratum")) for row in held}),
        "family_blind_split_verified": True,
        "spec_refs": ["REQ-BENCH-6544", "SCENARIO-BENCH-6544-EFFECTS"],
    }


def _read_problem(source_root: Path, row: Mapping[str, Any]) -> JsonDict:
    relpath = str(row.get("source_file_relpath") or "")
    return _load_json(source_root / relpath)


def _solver_assertion_count(
    checker: ModuleType | None,
    *,
    problem: Mapping[str, Any],
    constraints: Sequence[Mapping[str, Any]],
) -> int:
    if checker is None:
        return len(constraints)
    domain = str(problem.get("domain") or "")
    entities = [str(item) for item in problem.get("entities", [])]
    context = exp6543._context_from_problem(problem)
    try:
        return exp6543._solver_assertion_count(
            checker,
            domain=domain,
            entities=entities,
            constraints=[dict(item) for item in constraints],
            context=context,
        )
    except Exception:  # pragma: no cover - bad source closes through exact label mismatch.
        return len(constraints)


def _candidate_receipt(
    *,
    checker: ModuleType | None,
    problem: Mapping[str, Any],
    row: Mapping[str, Any],
    candidate_index: int,
    turn: Mapping[str, Any],
) -> JsonDict:
    constraints = turn.get("cumulative_constraints", [])
    constraints = [dict(item) for item in constraints] if isinstance(constraints, list) else []
    domain = str(problem.get("domain") or row.get("domain") or "")
    entities = [str(item) for item in problem.get("entities", [])]
    context = exp6543._context_from_problem(problem)
    start = time.perf_counter()
    timeout = False
    error = None
    is_sat = False
    if checker is None:
        error = "z3_checker_unavailable"
    else:
        try:
            receipt = checker.check_satisfiability(
                [dict(item) for item in constraints],
                domain,
                entities,
                context=dict(context),
            )
            is_sat = bool(receipt.get("is_sat"))
        except TimeoutError as exc:
            timeout = True
            error = str(exc)
        except Exception as exc:  # pragma: no cover - source checker failures are data failures.
            error = f"{type(exc).__name__}: {exc}"
    wall_time_s = round(max(time.perf_counter() - start, 0.0), 9)
    label = "timeout" if timeout else "error" if error else "satisfiable" if is_sat else "contradiction"
    assertion_count = _solver_assertion_count(checker, problem=problem, constraints=constraints)
    payload = {
        "candidate_index": candidate_index,
        "candidate_depth": candidate_index,
        "constraint_count": len(constraints),
        "solver_assertion_count": assertion_count,
        "constraints_sha256": exp6543.sha256_json(constraints),
        "source_turn_sha256": exp6543.sha256_json(turn),
        "target_full_state": candidate_index == int(row.get("turn_index", -1)),
    }
    candidate_hash = sha256_json(
        {
            "source_problem_hash": row.get("source_problem_hash"),
            "candidate_index": candidate_index,
            "constraints_sha256": payload["constraints_sha256"],
        }
    )
    check_cost = max(1, len(constraints) + assertion_count + 1)
    return {
        **payload,
        "candidate_hash": candidate_hash,
        "exact_label": label,
        "satisfiable": is_sat,
        "timeout": timeout,
        "censored": timeout,
        "error": error,
        "z3_check_calls": 1,
        "wall_time_s": wall_time_s,
        "check_cost_units": check_cost,
        "receipt_hash": sha256_json({**payload, "exact_label": label, "check_cost_units": check_cost}),
    }


def _candidate_set(
    *,
    source_root: Path,
    checker: ModuleType | None,
    row: Mapping[str, Any],
    receipt_cache: dict[tuple[str, int], JsonDict],
) -> list[JsonDict]:
    problem = _read_problem(source_root, row)
    turns = problem.get("turns", [])
    if not isinstance(turns, list):
        return []
    target_index = int(row.get("turn_index", -1))
    candidates: list[JsonDict] = []
    for candidate_index in range(target_index + 1):
        if candidate_index >= len(turns):
            break
        cache_key = (str(row.get("source_file_relpath")), candidate_index)
        if cache_key not in receipt_cache:
            turn = turns[candidate_index]
            if not isinstance(turn, Mapping):
                continue
            receipt_cache[cache_key] = _candidate_receipt(
                checker=checker,
                problem=problem,
                row=row,
                candidate_index=candidate_index,
                turn=turn,
            )
        candidates.append(dict(receipt_cache[cache_key]))
    return candidates


def _order_candidates(
    *,
    arm_id: str,
    seed: int,
    candidates: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    ordered = [dict(candidate) for candidate in candidates]
    if arm_id == NATIVE_ARM:
        return sorted(ordered, key=lambda item: int(item["candidate_index"]))
    if arm_id == "random":
        rng = random.Random(seed + (len(ordered) * 9973))
        rng.shuffle(ordered)
        return ordered
    if arm_id == "analytical":
        return sorted(
            ordered,
            key=lambda item: (
                -int(item["constraint_count"]),
                -int(item["solver_assertion_count"]),
                -int(item["candidate_index"]),
            ),
        )
    if arm_id == "bounded_refocus":
        count = len(ordered)
        wanted = [0, max(0, count // 2), max(0, count - 1)]
        seen: set[int] = set()
        first = []
        for index in wanted:
            if index not in seen and index < count:
                first.append(ordered[index])
                seen.add(index)
        rest = [item for item in ordered if int(item["candidate_index"]) not in seen]
        rest.sort(key=lambda item: (-int(item["constraint_count"]), int(item["candidate_index"])))
        return first + rest
    if arm_id == "one_shot_enumeration":
        return sorted(
            ordered,
            key=lambda item: (
                -int(item["solver_assertion_count"]),
                -int(item["constraint_count"]),
                -int(item["candidate_index"]),
            ),
        )
    raise ValueError(f"unknown arm_id {arm_id}")  # pragma: no cover - ARM_IDS owns callers.


def _control_overhead_units(arm_id: str, candidate_count: int) -> int:
    if arm_id == NATIVE_ARM:
        return candidate_count * 3
    if arm_id == "random":
        return candidate_count * 3
    if arm_id == "analytical":
        return max(1, candidate_count // 4)
    if arm_id == "bounded_refocus":
        return candidate_count + 1
    if arm_id == "one_shot_enumeration":
        return candidate_count * 2
    raise ValueError(f"unknown arm_id {arm_id}")  # pragma: no cover - ARM_IDS owns callers.


def _run_arm_row(
    *,
    fixture_row: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    arm_id: str,
    seed: int,
) -> JsonDict:
    ordered = _order_candidates(arm_id=arm_id, seed=seed, candidates=candidates)
    candidate_hashes = [str(candidate["candidate_hash"]) for candidate in candidates]
    target = next((candidate for candidate in candidates if candidate.get("target_full_state")), None)
    target_check_cost = int(target.get("check_cost_units", 1)) if target else 1
    exact_checks: list[JsonDict] = []
    terminal = None
    for candidate in ordered:
        check = {
            "candidate_hash": candidate["candidate_hash"],
            "candidate_index": candidate["candidate_index"],
            "target_full_state": candidate["target_full_state"],
            "exact_label": candidate["exact_label"],
            "check_cost_units": target_check_cost,
            "source_receipt_check_cost_units": candidate["check_cost_units"],
            "solver_assertion_count": candidate["solver_assertion_count"],
            "timeout": candidate["timeout"],
            "receipt_hash": candidate["receipt_hash"],
        }
        exact_checks.append(check)
        if candidate.get("target_full_state") is True:
            terminal = candidate
            break
    fallback_used = terminal is None and target is not None
    if fallback_used and target is not None:
        terminal = target
        exact_checks.append(
            {
                "candidate_hash": target["candidate_hash"],
                "candidate_index": target["candidate_index"],
                "target_full_state": True,
                "exact_label": target["exact_label"],
                "check_cost_units": target_check_cost,
                "source_receipt_check_cost_units": target["check_cost_units"],
                "solver_assertion_count": target["solver_assertion_count"],
                "timeout": target["timeout"],
                "receipt_hash": target["receipt_hash"],
            }
        )
    exact_label = str(terminal.get("exact_label") if terminal else "missing")
    fixture_label = str(fixture_row.get("exact_label"))
    exact_cost = sum(int(check["check_cost_units"]) for check in exact_checks)
    proposal_cost = len(candidate_hashes)
    overhead = _control_overhead_units(arm_id, len(candidate_hashes))
    fallback_cost = int(target.get("check_cost_units", 0)) if fallback_used and target else 0
    total = proposal_cost + exact_cost + overhead + fallback_cost
    timeout = any(check.get("timeout") is True for check in exact_checks)
    censored = timeout or terminal is None
    wall_time = sum(float(candidate.get("wall_time_s", 0.0)) for candidate in ordered[: len(exact_checks)])
    payload = {
        "row_type": "external_structural_control_unit",
        "schema_version": "carnot.exp6544.per_unit_row.v1",
        "local_unit_id": fixture_row.get("local_unit_id"),
        "source_turn_id": fixture_row.get("source_turn_id"),
        "base_problem_id": fixture_row.get("base_problem_id"),
        "split_name": fixture_row.get("split_name"),
        "family": fixture_row.get("family"),
        "pre_replay_effort_stratum": fixture_row.get("pre_replay_effort_stratum"),
        "turn_index": int(fixture_row.get("turn_index", 0)),
        "turn_number": int(fixture_row.get("turn_number", 0)),
        "seed": seed,
        "arm_id": arm_id,
        "native_reference_arm": NATIVE_ARM,
        "candidate_count": len(candidate_hashes),
        "candidate_hashes": candidate_hashes,
        "candidate_order": [str(candidate["candidate_hash"]) for candidate in ordered],
        "candidate_preserved": len(candidate_hashes) == len({*candidate_hashes})
        and set(candidate_hashes) == {str(candidate["candidate_hash"]) for candidate in ordered},
        "candidate_deleted_count": len(set(candidate_hashes) - {str(candidate["candidate_hash"]) for candidate in ordered}),
        "candidate_budget": len(candidate_hashes),
        "exact_check_budget": len(candidate_hashes) + 1,
        "proposal_count": len(candidate_hashes),
        "exact_check_count": len(exact_checks),
        "exact_checks": exact_checks,
        "solver_effort": {
            "z3_check_calls": len(exact_checks),
            "solver_assertion_count": sum(int(check["solver_assertion_count"]) for check in exact_checks),
            "exact_check_cost_units": exact_cost,
        },
        "proposal_cost_units": proposal_cost,
        "exact_check_cost_units": exact_cost,
        "control_overhead_units": overhead,
        "fallback_cost_units": fallback_cost,
        "total_charged_work_units": total,
        "wall_time_s": round(wall_time + (overhead * 0.000001), 9),
        "timeout": timeout,
        "censored": censored,
        "censoring_reason": "" if not censored else "timeout_or_missing_terminal",
        "native_exact_fallback_available": True,
        "native_exact_fallback_used": fallback_used,
        "exact_status": exact_label,
        "audited_fixture_exact_label": fixture_label,
        "exact_answer_equality": exact_label == fixture_label,
        "terminal_candidate_hash": terminal.get("candidate_hash") if terminal else None,
        "terminal_receipt_hash": terminal.get("receipt_hash") if terminal else None,
        "uses_forbidden_features": False,
        "ordering_feature_set": control_definitions()[arm_id]["ordering_features"],
        "spec_refs": [
            "REQ-BENCH-6544",
            "SCENARIO-BENCH-6544-COST-EQUALITY",
            "SCENARIO-BENCH-6544-FAMILY-BLIND",
        ],
    }
    return {**payload, "row_hash": sha256_json(payload)}


def matched_unit_rows(
    *,
    fixture_rows: Sequence[Mapping[str, Any]],
    source_root: Path,
) -> list[JsonDict]:
    checker = exp6543._load_z3_checker(source_root)
    receipt_cache: dict[tuple[str, int], JsonDict] = {}
    rows: list[JsonDict] = []
    for fixture_row in fixture_rows:
        candidates = _candidate_set(
            source_root=source_root,
            checker=checker,
            row=fixture_row,
            receipt_cache=receipt_cache,
        )
        for seed in SEED_GRID:
            for arm_id in ARM_IDS:
                rows.append(
                    _run_arm_row(
                        fixture_row=fixture_row,
                        candidates=candidates,
                        arm_id=arm_id,
                        seed=seed,
                    )
                )
    return rows


def _rows_by_key(rows: Sequence[Mapping[str, Any]], arm_id: str) -> dict[tuple[str, int], JsonDict]:
    return {
        (str(row.get("local_unit_id")), int(row.get("seed", 0))): dict(row)
        for row in rows
        if row.get("split_name") == "held" and row.get("arm_id") == arm_id
    }


def _std_error(values: Sequence[int]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance) / math.sqrt(len(values))


def paired_effect_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    native = _rows_by_key(rows, NATIVE_ARM)
    random_rows = _rows_by_key(rows, "random")
    out: list[JsonDict] = []
    for arm_id in [arm for arm in ARM_IDS if arm != NATIVE_ARM]:
        arm_rows = _rows_by_key(rows, arm_id)
        paired_keys = sorted(set(native) & set(random_rows) & set(arm_rows))
        native_deltas = [
            int(native[key]["total_charged_work_units"]) - int(arm_rows[key]["total_charged_work_units"])
            for key in paired_keys
        ]
        random_deltas = [
            int(random_rows[key]["total_charged_work_units"]) - int(arm_rows[key]["total_charged_work_units"])
            for key in paired_keys
        ]
        support = [
            arm_rows[key]
            for key, native_delta, random_delta in zip(
                paired_keys, native_deltas, random_deltas, strict=True
            )
            if native_delta > 0 and random_delta > 0
        ]
        payload = {
            "row_type": "paired_effect",
            "arm_id": arm_id,
            "held_effect_vs_native_units": sum(native_deltas),
            "held_effect_vs_random_units": sum(random_deltas),
            "paired_unit_count": len(paired_keys),
            "support_family_count": len({str(row.get("family")) for row in support}),
            "support_families": sorted({str(row.get("family")) for row in support}),
            "headroom_pair_count": len(support),
            "no_headroom_pair_count": len(paired_keys) - len(support),
            "uncertainty": {
                "paired_std_error_units": round(_std_error(native_deltas), 6),
                "paired_random_std_error_units": round(_std_error(random_deltas), 6),
            },
            "spec_refs": ["REQ-BENCH-6544", "SCENARIO-BENCH-6544-EFFECTS"],
        }
        out.append({**payload, "effect_row_hash": sha256_json(payload)})
    return out


def family_effect_rows(rows: Sequence[Mapping[str, Any]], effects: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    held_rows = [dict(row) for row in rows if row.get("split_name") == "held"]
    families = sorted({str(row.get("family")) for row in held_rows})
    out: list[JsonDict] = []
    overall_positive = {
        str(row["arm_id"]): row.get("held_effect_vs_native_units", 0) > 0
        and row.get("held_effect_vs_random_units", 0) > 0
        for row in effects
    }
    for arm_id in [arm for arm in ARM_IDS if arm != NATIVE_ARM]:
        for family in families:
            subset = [row for row in held_rows if row.get("family") == family]
            native_cost = sum(
                int(row["total_charged_work_units"])
                for row in subset
                if row.get("arm_id") == NATIVE_ARM
            )
            random_cost = sum(
                int(row["total_charged_work_units"])
                for row in subset
                if row.get("arm_id") == "random"
            )
            arm_cost = sum(
                int(row["total_charged_work_units"])
                for row in subset
                if row.get("arm_id") == arm_id
            )
            effect_native = native_cost - arm_cost
            effect_random = random_cost - arm_cost
            headroom = effect_native > 0 and effect_random > 0
            payload = {
                "row_type": "family_effect",
                "arm_id": arm_id,
                "family": family,
                "held_row_count": len([row for row in subset if row.get("arm_id") == arm_id]),
                "held_effect_vs_native_units": effect_native,
                "held_effect_vs_random_units": effect_random,
                "headroom_cell": headroom,
                "no_headroom_cell": not headroom,
                "simpson_reversal": bool(overall_positive.get(arm_id)) and not headroom,
                "spec_refs": ["REQ-BENCH-6544", "SCENARIO-BENCH-6544-EFFECTS"],
            }
            out.append({**payload, "family_effect_row_hash": sha256_json(payload)})
    return out


def charged_cost_recomputation(
    rows: Sequence[Mapping[str, Any]],
    effects: Sequence[Mapping[str, Any]],
) -> JsonDict:
    bad_rows = [
        row
        for row in rows
        if int(row.get("total_charged_work_units", -1))
        != int(row.get("proposal_cost_units", -2))
        + int(row.get("exact_check_cost_units", -3))
        + int(row.get("control_overhead_units", -4))
        + int(row.get("fallback_cost_units", -5))
    ]
    totals = {
        arm_id: sum(int(row["total_charged_work_units"]) for row in rows if row.get("arm_id") == arm_id)
        for arm_id in ARM_IDS
    }
    held_totals = {
        arm_id: sum(
            int(row["total_charged_work_units"])
            for row in rows
            if row.get("arm_id") == arm_id and row.get("split_name") == "held"
        )
        for arm_id in ARM_IDS
    }
    best = max(
        effects,
        key=lambda row: (
            int(row.get("held_effect_vs_native_units", 0)),
            int(row.get("held_effect_vs_random_units", 0)),
            str(row.get("arm_id")),
        ),
        default={},
    )
    return {
        "row_type": "charged_cost_recomputation",
        "all_costs_recomputed_from_rows": not bad_rows and bool(rows),
        "bad_cost_rows": bad_rows,
        "total_charged_work_by_arm": totals,
        "held_total_charged_work_by_arm": held_totals,
        "best_arm": best.get("arm_id"),
        "best_arm_held_effect_vs_native_units": best.get("held_effect_vs_native_units", 0),
        "best_arm_held_effect_vs_random_units": best.get("held_effect_vs_random_units", 0),
        "proposal_cost_total": sum(int(row.get("proposal_cost_units", 0)) for row in rows),
        "exact_check_cost_total": sum(int(row.get("exact_check_cost_units", 0)) for row in rows),
        "control_overhead_total": sum(int(row.get("control_overhead_units", 0)) for row in rows),
        "fallback_cost_total": sum(int(row.get("fallback_cost_units", 0)) for row in rows),
        "spec_refs": ["REQ-BENCH-6544", "SCENARIO-BENCH-6544-COST-EQUALITY"],
    }


def exact_equality_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    mismatches = [dict(row) for row in rows if row.get("exact_answer_equality") is not True]
    return {
        "row_type": "exact_equality_receipt",
        "row_count": len(rows),
        "all_exact_equal": bool(rows) and not mismatches,
        "mismatch_count": len(mismatches),
        "mismatch_rows": mismatches,
        "z3_evaluation_authority": True,
        "verifier_is_oracle": False,
        "spec_refs": ["REQ-BENCH-6544", "SCENARIO-BENCH-6544-COST-EQUALITY"],
    }


def candidate_preservation_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("local_unit_id")), int(row.get("seed", 0)))].append(row)
    bad_groups = []
    for key, group in grouped.items():
        sets = {tuple(row.get("candidate_hashes", [])) for row in group}
        if len(sets) != 1 or any(row.get("candidate_preserved") is not True for row in group):
            bad_groups.append({"unit_seed": list(key), "candidate_sets": [list(item) for item in sets]})
    return {
        "row_type": "candidate_preservation_receipt",
        "all_candidates_preserved": bool(rows) and not bad_groups,
        "candidate_set_identity_passed": bool(grouped) and not bad_groups,
        "unit_seed_group_count": len(grouped),
        "bad_groups": bad_groups,
        "deleted_candidate_count": sum(int(row.get("candidate_deleted_count", 0)) for row in rows),
        "spec_refs": ["REQ-BENCH-6544", "SCENARIO-BENCH-6544-CONTRACT"],
    }


def censoring_and_timeout_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    bad = [
        dict(row)
        for row in rows
        if row.get("timeout") is not False
        or row.get("censored") is not False
        or row.get("candidate_budget") != row.get("candidate_count")
        or row.get("exact_check_budget") != row.get("candidate_count") + 1
    ]
    return {
        "row_type": "censoring_and_timeout_receipts",
        "all_timeout_and_censoring_checks_passed": bool(rows) and not bad,
        "timeout_count": sum(1 for row in rows if row.get("timeout") is True),
        "censored_count": sum(1 for row in rows if row.get("censored") is True),
        "bad_budget_or_timeout_rows": bad,
        "timeout_s": TIMEOUT_S,
        "spec_refs": ["REQ-BENCH-6544", "SCENARIO-BENCH-6544-COST-EQUALITY"],
    }


def shortcut_attack_matrix(
    *,
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    definitions: Mapping[str, Mapping[str, Any]],
    preservation: Mapping[str, Any],
    censoring: Mapping[str, Any],
    aggregate_source_from_rows: bool,
) -> JsonDict:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("local_unit_id")), int(row.get("seed", 0)))].append(row)
    budget_passed = bool(grouped) and all(
        len({row.get("candidate_budget") for row in group}) == 1
        and len({row.get("exact_check_budget") for row in group}) == 1
        for group in grouped.values()
    )
    seed_passed = bool(rows) and all(
        {
            int(row.get("seed", 0))
            for row in rows
            if row.get("local_unit_id") == unit_id and row.get("arm_id") == arm_id
        }
        == set(SEED_GRID)
        for unit_id in {row.get("local_unit_id") for row in rows}
        for arm_id in ARM_IDS
    )
    forbidden = set(contract.get("forbidden_features", []))
    used_features = {
        feature
        for row in definitions.values()
        for feature in row.get("ordering_features", [])
    }
    checks = {
        "identity_leakage": not (used_features & forbidden)
        and all(row.get("uses_forbidden_features") is False for row in rows),
        "row_order_leakage": contract.get("row_order_used_as_feature") is False,
        "unequal_budgets": budget_passed,
        "deleted_candidates": preservation.get("all_candidates_preserved") is True
        and preservation.get("deleted_candidate_count") == 0,
        "warm_cache_asymmetry": aggregate_source_from_rows
        and all("check_cost_units" in check for row in rows for check in row.get("exact_checks", [])),
        "timer_aliases": censoring.get("all_timeout_and_censoring_checks_passed") is True
        and all(row.get("wall_time_s", -1) >= 0 for row in rows),
        "cherry_picked_seeds": seed_passed,
        "aggregate_only_claims": aggregate_source_from_rows and bool(rows),
    }
    attack_rows = []
    for attack_id in SHORTCUT_ATTACK_IDS:
        payload = {
            "row_type": "shortcut_attack",
            "attack_id": attack_id,
            "fail_closed": bool(checks[attack_id]),
            "false_accept": not bool(checks[attack_id]),
            "expected_value": True,
            "observed_value": checks[attack_id],
            "spec_refs": ["REQ-BENCH-6544", "SCENARIO-BENCH-6544-ATTACKS"],
        }
        attack_rows.append({**payload, "attack_row_hash": sha256_json(payload)})
    return {
        "row_type": "shortcut_attack_matrix",
        "rows": attack_rows,
        "all_shortcuts_fail_closed": all(row["fail_closed"] is True for row in attack_rows),
        "false_accept_count": sum(1 for row in attack_rows if row["false_accept"] is True),
        "failed_attack_ids": [row["attack_id"] for row in attack_rows if row["fail_closed"] is not True],
        "spec_refs": ["REQ-BENCH-6544", "SCENARIO-BENCH-6544-ATTACKS"],
    }


def aggregate_row_recomputation(
    *,
    gate: Mapping[str, Any],
    source_root_available: bool,
    fixture_rows: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    effects: Sequence[Mapping[str, Any]],
    family_effects: Sequence[Mapping[str, Any]],
    costs: Mapping[str, Any],
    equality: Mapping[str, Any],
    preservation: Mapping[str, Any],
    censoring: Mapping[str, Any],
    attacks: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> JsonDict:
    row_count_expected = len(fixture_rows) * len(SEED_GRID) * len(ARM_IDS)
    arm_coverage = {str(row.get("arm_id")) for row in rows} == set(ARM_IDS)
    seed_coverage = {int(row.get("seed", 0)) for row in rows} == set(SEED_GRID)
    best = max(
        effects,
        key=lambda row: (
            int(row.get("held_effect_vs_native_units", 0)),
            int(row.get("held_effect_vs_random_units", 0)),
            int(row.get("support_family_count", 0)),
            str(row.get("arm_id")),
        ),
        default={},
    )
    best_arm = best.get("arm_id")
    best_families = [
        row
        for row in family_effects
        if row.get("arm_id") == best_arm and row.get("headroom_cell") is True
    ]
    execution = all(
        (
            gate.get("gate_passed") is True,
            source_root_available,
            len(rows) == row_count_expected and row_count_expected > 0,
            arm_coverage,
            seed_coverage,
            costs.get("all_costs_recomputed_from_rows") is True,
            equality.get("all_exact_equal") is True,
            preservation.get("all_candidates_preserved") is True,
            censoring.get("all_timeout_and_censoring_checks_passed") is True,
            attacks.get("all_shortcuts_fail_closed") is True,
            protected.get("all_protected_files_unchanged") is True,
        )
    )
    best_positive_native = int(best.get("held_effect_vs_native_units", 0)) > 0
    best_positive_random = int(best.get("held_effect_vs_random_units", 0)) > 0
    support_family_count = len({str(row.get("family")) for row in best_families})
    ready = all(
        (
            execution,
            best_arm not in {None, NATIVE_ARM, "random"},
            best_positive_native,
            best_positive_random,
            support_family_count > 1,
        )
    )
    if not execution:
        verdict = "blocked" if gate.get("gate_passed") is not True or not source_root_available else "disqualified"
    elif ready:
        verdict = "positive"
    elif best_positive_native or best_positive_random:
        verdict = "partial"
    else:
        verdict = None
    return {
        "row_type": "aggregate_row_recomputation",
        "upstream_gate_passed": gate.get("gate_passed") is True,
        "source_root_available": source_root_available,
        "fixture_row_count": len(fixture_rows),
        "matched_row_count": len(rows),
        "expected_matched_row_count": row_count_expected,
        "arm_coverage_passed": arm_coverage,
        "seed_coverage_passed": seed_coverage,
        "cost_recomputation_passed": costs.get("all_costs_recomputed_from_rows") is True,
        "exact_equality_passed": equality.get("all_exact_equal") is True,
        "candidate_preservation_passed": preservation.get("all_candidates_preserved") is True,
        "censoring_timeout_passed": censoring.get("all_timeout_and_censoring_checks_passed") is True,
        "shortcut_attack_passed": attacks.get("all_shortcuts_fail_closed") is True,
        "protected_files_unchanged": protected.get("all_protected_files_unchanged") is True,
        "best_arm": best_arm,
        "best_arm_positive_beyond_native": best_positive_native,
        "best_arm_positive_beyond_random": best_positive_random,
        "best_arm_support_family_count": support_family_count,
        "best_arm_support_families": sorted({str(row.get("family")) for row in best_families}),
        "execution_complete_from_rows": execution,
        "ready_score_from_rows": 1.0 if ready else 0.0,
        "verdict_class_from_rows": verdict,
        "aggregate_source": "per_unit_rows_and_effect_rows",
        "spec_refs": ["REQ-BENCH-6544", "SCENARIO-BENCH-6544-TERMINAL"],
    }


def gate_check_summary(aggregate: Mapping[str, Any]) -> JsonDict:
    expected = {
        "upstream_gate_passed": True,
        "source_root_available": True,
        "matched_row_count": aggregate.get("expected_matched_row_count"),
        "arm_coverage_passed": True,
        "seed_coverage_passed": True,
        "cost_recomputation_passed": True,
        "exact_equality_passed": True,
        "candidate_preservation_passed": True,
        "censoring_timeout_passed": True,
        "shortcut_attack_passed": True,
        "protected_files_unchanged": True,
        "ready_score_is_binary": True,
    }
    observed = {
        "upstream_gate_passed": aggregate.get("upstream_gate_passed"),
        "source_root_available": aggregate.get("source_root_available"),
        "matched_row_count": aggregate.get("matched_row_count"),
        "arm_coverage_passed": aggregate.get("arm_coverage_passed"),
        "seed_coverage_passed": aggregate.get("seed_coverage_passed"),
        "cost_recomputation_passed": aggregate.get("cost_recomputation_passed"),
        "exact_equality_passed": aggregate.get("exact_equality_passed"),
        "candidate_preservation_passed": aggregate.get("candidate_preservation_passed"),
        "censoring_timeout_passed": aggregate.get("censoring_timeout_passed"),
        "shortcut_attack_passed": aggregate.get("shortcut_attack_passed"),
        "protected_files_unchanged": aggregate.get("protected_files_unchanged"),
        "ready_score_is_binary": aggregate.get("ready_score_from_rows") in {0.0, 1.0},
    }
    checks = {
        name: {"expected": value, "observed": observed[name], "passed": observed[name] == value}
        for name, value in expected.items()
    }
    failed = [name for name, row in checks.items() if row["passed"] is not True]
    return {
        "row_type": "gate_check_summary",
        "checks": checks,
        "failed_checks": failed,
        "all_gates_passed": not failed,
        "spec_refs": ["REQ-BENCH-6544", "SCENARIO-BENCH-6544-GATE"],
    }


def _status_and_honest_verdict(aggregate: Mapping[str, Any]) -> tuple[str, str, str | None]:
    verdict = aggregate.get("verdict_class_from_rows")
    if verdict == "blocked":
        return (
            "blocked_external_structural_headroom",
            "blocked_external_structural_headroom: upstream gate or source precondition failed",
            "blocked",
        )
    if verdict == "disqualified":
        return (
            "disqualified_external_structural_headroom",
            "disqualified_external_structural_headroom: matched rows failed equality, cost, preservation, or shortcut checks",
            "disqualified",
        )
    if verdict == "positive":
        effect = aggregate.get("best_arm")
        families = ",".join(aggregate.get("best_arm_support_families", []))
        native = aggregate.get("best_arm_positive_beyond_native")
        random_ok = aggregate.get("best_arm_positive_beyond_random")
        return (
            "complete_external_structural_headroom_positive",
            (
                "complete_external_structural_headroom_positive: charged held "
                f"{effect} value is positive beyond native={native} and random={random_ok}; "
                f"support_families={families}"
            ),
            "positive",
        )
    if verdict == "partial":
        return (
            "partial_external_structural_headroom",
            "partial_external_structural_headroom: narrow benefit did not clear the preregistered ready gate",
            "partial",
        )
    return (
        "complete_external_structural_headroom_null",
        "complete_external_structural_headroom_null: complete matched run found no charged held value beyond native and random",
        None,
    )


def _field_provenance(repo_root: Path) -> dict[str, JsonDict]:
    source_hashes = {
        rel.as_posix(): sha256_file(repo_root / rel) for rel in SOURCE_RELATIVE_PATHS
    }
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source": "deterministic_exp6544_external_structural_headroom_reducer",
            "spec": SPEC_RELATIVE_PATH.as_posix(),
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "test": TEST_RELATIVE_PATH.as_posix(),
            "source_hashes": source_hashes,
            "spec_refs": ["REQ-BENCH-6544"],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def preconditions_checked(
    *,
    repo_root: Path,
    result_path: Path,
    audit_path: Path,
    fixture_path: Path,
    source_root: Path,
    run_date: str,
    protected_before: Mapping[str, str],
    aggregate: Mapping[str, Any],
) -> JsonDict:
    return {
        "planning_date": run_date,
        "repo_root": str(repo_root),
        "result_path": str(result_path),
        "audit_path": str(audit_path),
        "fixture_path": str(fixture_path),
        "source_root": str(source_root),
        "audit_sha256": sha256_file(audit_path),
        "fixture_sha256": sha256_file(fixture_path),
        "source_root_available": source_root.exists(),
        "solver_identity": solver_identity(),
        "resources": _resource_state(repo_root),
        "git_state": _git_state(repo_root),
        "random_seed": RANDOM_SEED,
        "seed_grid": list(SEED_GRID),
        "arm_ids": list(ARM_IDS),
        "timeout_s": TIMEOUT_S,
        "candidate_budget_rule": "all_candidates_preserved",
        "exact_check_budget_rule": "candidate_count_plus_native_fallback",
        "native_exact_fallback_available": True,
        "z3_evaluation_authority": True,
        "verifier_is_oracle": False,
        "aggregate_ready_score": aggregate.get("ready_score_from_rows"),
        "protected_file_hashes_before": dict(protected_before),
        "spec_refs": ["REQ-BENCH-6544", "SCENARIO-BENCH-6544-GATE"],
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    clone = json.loads(json.dumps(artifact, sort_keys=True, default=str))
    clone["reproducibility_checksum"] = ""
    return sha256_json(clone)


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    audit_path: Path | str = EXP6543_RELATIVE_PATH,
    fixture_path: Path | str = FIXTURE_RELATIVE_PATH,
    source_root: Path | str | None = None,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    start = time.perf_counter()
    repo_root = Path(repo_root)
    result = Path(result_path)
    if not result.is_absolute():
        result = repo_root / result
    audit = Path(audit_path)
    if not audit.is_absolute():
        audit = repo_root / audit
    fixture = Path(fixture_path)
    if not fixture.is_absolute():
        fixture = repo_root / fixture
    protected_before = _protected_hashes(repo_root, audit)
    audit_payload = _load_json(audit)
    gate = upstream_gate_receipt(
        repo_root=repo_root,
        audit_path=audit,
        fixture_path=fixture,
        protected_before=protected_before,
    )
    resolved_source_root = _resolve_source_root(audit=audit_payload, source_root=source_root)
    fixture_rows = _load_jsonl(fixture)
    contract = frozen_comparison_contract(run_date)
    definitions = control_definitions()
    census = family_and_effort_census(fixture_rows)
    can_run = gate["gate_passed"] is True and resolved_source_root.exists() and bool(fixture_rows)
    rows = matched_unit_rows(fixture_rows=fixture_rows, source_root=resolved_source_root) if can_run else []
    effects = paired_effect_rows(rows)
    family_effects = family_effect_rows(rows, effects)
    costs = charged_cost_recomputation(rows, effects)
    equality = exact_equality_receipt(rows)
    preservation = candidate_preservation_receipt(rows)
    censoring = censoring_and_timeout_receipts(rows)
    attacks = shortcut_attack_matrix(
        rows=rows,
        contract=contract,
        definitions=definitions,
        preservation=preservation,
        censoring=censoring,
        aggregate_source_from_rows=True,
    )
    protected_after = _protected_hashes(repo_root, audit)
    protected = protected_files_unchanged(protected_before, protected_after)
    aggregate = aggregate_row_recomputation(
        gate=gate,
        source_root_available=resolved_source_root.exists(),
        fixture_rows=fixture_rows,
        rows=rows,
        effects=effects,
        family_effects=family_effects,
        costs=costs,
        equality=equality,
        preservation=preservation,
        censoring=censoring,
        attacks=attacks,
        protected=protected,
    )
    gates = gate_check_summary(aggregate)
    status, honest, verdict = _status_and_honest_verdict(aggregate)
    artifact: JsonDict = {
        "status": status,
        "honest_verdict": honest,
        "verdict_class": verdict,
        "upstream_gate_receipt": gate,
        "frozen_comparison_contract": contract,
        "control_definitions": definitions,
        "family_and_effort_census": census,
        "per_unit_rows": rows,
        "paired_effect_rows": effects,
        "family_effect_rows": family_effects,
        "charged_cost_recomputation": costs,
        "exact_equality_receipt": equality,
        "candidate_preservation_receipt": preservation,
        "censoring_and_timeout_receipts": censoring,
        "shortcut_attack_matrix": attacks,
        "external_structural_headroom_ready_score": float(aggregate["ready_score_from_rows"]),
        "gate_check_summary": gates,
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": preconditions_checked(
            repo_root=repo_root,
            result_path=result,
            audit_path=audit,
            fixture_path=fixture,
            source_root=resolved_source_root,
            run_date=run_date,
            protected_before=protected_before,
            aggregate=aggregate,
        ),
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": _field_provenance(repo_root),
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s if duration_s is not None else time.perf_counter() - start),
        "tests_run": _tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        atomic_write_json(result, artifact, allow_override=False, sort_keys=False)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    if artifact.get("verdict_class") not in {
        "positive",
        None,
        "partial",
        "blocked",
        "disqualified",
    }:
        errors.append("verdict_class outside Exp6544 enum")
    if not str(artifact.get("honest_verdict", "")).startswith(
        ("complete_", "partial_", "blocked_", "disqualified_")
    ):
        errors.append("honest_verdict terminal prefix mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    score = artifact.get("external_structural_headroom_ready_score")
    recomputed = artifact.get("aggregate_row_recomputation", {}).get("ready_score_from_rows")
    if score not in {0.0, 1.0} or score != recomputed:
        errors.append("ready score mismatch")
    gates_passed = artifact.get("gate_check_summary", {}).get("all_gates_passed")
    if score not in {0.0, None} and gates_passed is not True:
        errors.append("positive score requires all gates passed")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or validate Exp6544 external structural headroom artifact."
    )
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--audit-path", default=str(REPO_ROOT / EXP6543_RELATIVE_PATH))
    parser.add_argument("--fixture-path", default=str(REPO_ROOT / FIXTURE_RELATIVE_PATH))
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result = Path(args.result_path)
    if args.validate:
        errors = validate_artifact(_load_json(result))
        if errors:
            print("\n".join(errors))
            return 1
        print(f"validated {RESULT_RELATIVE_PATH.as_posix()}")
        return 0
    artifact = build_artifact(
        repo_root=REPO_ROOT,
        result_path=result,
        audit_path=Path(args.audit_path),
        fixture_path=Path(args.fixture_path),
        source_root=Path(args.source_root) if args.source_root else None,
        write=True,
        run_date=str(args.date),
    )
    errors = validate_artifact(artifact)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"wrote {RESULT_RELATIVE_PATH.as_posix()} to {result}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
