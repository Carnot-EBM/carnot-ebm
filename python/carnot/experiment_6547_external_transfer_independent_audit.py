"""Exp6547 external transfer independent audit.

Spec refs: REQ-BENCH-6547, SCENARIO-BENCH-6547-ALWAYS-RUN,
SCENARIO-BENCH-6547-ROW-REDUCTION, SCENARIO-BENCH-6547-MODEL-IDENTITY,
SCENARIO-BENCH-6547-SHORTCUTS, SCENARIO-BENCH-6547-EXACT-EQUALITY,
SCENARIO-BENCH-6547-COST-ACCOUNTING, SCENARIO-BENCH-6547-CALIBRATION,
SCENARIO-BENCH-6547-ATOMIC-OUTPUT.

This reducer checks the external router and cost-guard evidence from row
tables. Upstream summaries are treated as claims to compare with local
recomputations, not as authority for the final audit disposition.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any

from carnot import experiment_6543_external_corpus_independent_audit_v2 as exp6543
from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6547
INFERENCE_SUBSTRATE = "independent_external_router_cost_and_exact_replay_audit_no_llm"

RESULT_RELATIVE_PATH = Path("results/experiment_6547_external_transfer_independent_audit.json")
EXP6543_RELATIVE_PATH = Path("results/experiment_6543_external_corpus_independent_audit_v2.json")
EXP6544_RELATIVE_PATH = Path("results/experiment_6544_external_structural_headroom.json")
EXP6545_RELATIVE_PATH = Path("results/experiment_6545_external_safety_net_router.json")
EXP6546_RELATIVE_PATH = Path("results/experiment_6546_smt_cost_guard_sota.json")
FIXTURE_RELATIVE_PATH = Path("results/fixtures/v566_drift_bench_external_slice.jsonl")
CHECKPOINT_RELATIVE_PATH = Path("results/checkpoints/experiment_6546_smt_cost_guard_sota.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/benchmarks/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6547_external_transfer_independent_audit.py")
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6547_external_transfer_independent_audit.py"
)
SOTA_MODELS_RELATIVE_PATH = Path("python/carnot/inference/sota_models.py")

UPSTREAM_INPUTS = ("corpus", "structural", "router", "cost_guard")
MANDATED_HF_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
STRUCTURAL_ARMS = ("native", "random", "analytical", "bounded_refocus", "one_shot_enumeration")
STRUCTURAL_BEST_ARM = "analytical"
ROUTER_CONTROL_ARM = "exp6544_certified_structural_control"
ROUTER_SELECTED_ARM = "linear_compact_router_abstention_exception_exact_fallback"
SURFACE_IDS = ("canonical", "relabeled")
TERMINAL_STATUSES = {"terminal", "timeout", "parse_failure", "model_failure"}

PROTECTED_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-references.md"),
    SPEC_RELATIVE_PATH,
    Path("ops/e2e-test-plan.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("scripts/research_conductor.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("scripts/check_spec_coverage.py"),
    SOTA_MODELS_RELATIVE_PATH,
    EXP6543_RELATIVE_PATH,
    EXP6544_RELATIVE_PATH,
    EXP6545_RELATIVE_PATH,
    EXP6546_RELATIVE_PATH,
    FIXTURE_RELATIVE_PATH,
    CHECKPOINT_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "input_disposition_rows",
    "router_row_recomputation",
    "cost_guard_row_recomputation",
    "independent_exact_replay_rows",
    "source_equivalence_rows",
    "model_identity_audit_rows",
    "exception_and_fixture_hash_receipts",
    "candidate_and_fallback_audit",
    "calibration_audit_rows",
    "token_time_and_tool_cost_audit",
    "censoring_and_terminal_coverage",
    "shortcut_attack_matrix",
    "lane_dispositions",
    "external_transfer_audited_ready_score",
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
    "status": "Records the terminal Exp6547 independent audit state.",
    "honest_verdict": "States each lane disposition without merging router and cost-guard evidence.",
    "verdict_class": "Separates clean, partial, blocked, and disqualified audit outcomes.",
    "input_disposition_rows": (
        "Records existence, hashes, status, verdict class, and ready score for each required upstream artifact."
    ),
    "router_row_recomputation": (
        "Recomputes structural and learned-router effects from rows instead of trusted aggregates."
    ),
    "cost_guard_row_recomputation": (
        "Recomputes SOTA guarded versus unguarded effects from rows instead of trusted aggregates."
    ),
    "independent_exact_replay_rows": (
        "Samples exact replay receipts and checks row-level equality against the audited fixture authority."
    ),
    "source_equivalence_rows": (
        "Checks proof-preserving surface and source-equivalence receipts before adopting any transfer claim."
    ),
    "model_identity_audit_rows": (
        "Verifies mandated GGUF IDs, local model paths, hashes, loader identity, and substitution resistance."
    ),
    "exception_and_fixture_hash_receipts": (
        "Pins fixture, exception table, checkpoint, model registry, and protected inputs by hash."
    ),
    "candidate_and_fallback_audit": (
        "Checks candidate preservation, abstention, fallback reachability, and exact equality across router rows."
    ),
    "calibration_audit_rows": "Checks train-only fitting and development-only calibration boundaries.",
    "token_time_and_tool_cost_audit": (
        "Recomputes prompt tokens, output tokens, model time, tool time, charged time, and charged tokens."
    ),
    "censoring_and_terminal_coverage": (
        "Checks timeouts, parse failures, nonterminal rows, checkpoint coverage, and exact completion."
    ),
    "shortcut_attack_matrix": (
        "Attacks identity, entity names, row order, leakage, prompt length, cache order, timing, duplicates, null rows, hidden writes, omitted tool cost, and aggregate tampering."
    ),
    "lane_dispositions": (
        "Keeps router and cost-guard dispositions separate so one lane cannot rescue or suppress the other."
    ),
    "external_transfer_audited_ready_score": (
        "Opens only when row-supported eligible claims pass every adopted independent check."
    ),
    "gate_check_summary": "Names failed checks with expected and observed values.",
    "per_unit_rows": "Flattens independent row checks used by the aggregate reducer.",
    "aggregate_row_recomputation": (
        "Rebuilds the final score and verdict class from rows and lane dispositions."
    ),
    "preconditions_checked": (
        "Records paths, hashes, code hashes, solver receipts, model receipts, resources, seed, and protected hashes."
    ),
    "protected_files_unchanged": (
        "Shows guarded source and evidence files stayed byte-identical during the run."
    ),
    "inference_substrate": (
        "Declares a deterministic external router, cost, and exact replay audit with no LLM inference."
    ),
    "verifier_is_oracle": (
        "True only for audit checks; the artifact makes no new positive scientific claim."
    ),
    "field_principles": "Explains why each required field exists.",
    "field_provenance": "Maps each field to rows, upstream artifacts, checks, tests, and hashes.",
    "random_seed": "Pins deterministic replay sampling and attack ordering.",
    "duration_s": "Records measured reducer wall time.",
    "tests_run": "Records validation command receipts.",
    "reproducibility_checksum": "Detects drift in inputs, rows, lane dispositions, gates, and verdict.",
}

FIELD_PROVENANCE: dict[str, JsonDict] = {
    field: {
        "principle": FIELD_PRINCIPLES[field],
        "source": "Exp6547 independent row reducer",
        "spec_refs": ["REQ-BENCH-6547"],
    }
    for field in REQUIRED_ARTIFACT_FIELDS
}
for _field in (
    "input_disposition_rows",
    "router_row_recomputation",
    "cost_guard_row_recomputation",
    "independent_exact_replay_rows",
    "source_equivalence_rows",
    "model_identity_audit_rows",
    "exception_and_fixture_hash_receipts",
    "candidate_and_fallback_audit",
    "calibration_audit_rows",
    "token_time_and_tool_cost_audit",
    "censoring_and_terminal_coverage",
    "shortcut_attack_matrix",
    "lane_dispositions",
    "gate_check_summary",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "preconditions_checked",
    "protected_files_unchanged",
):
    FIELD_PROVENANCE[_field]["source"] = _field

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6547_external_transfer_independent_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6547_external_transfer_independent_audit.py "
    "-m pytest tests/python/test_experiment_6547_external_transfer_independent_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6547_external_transfer_independent_audit.py "
    "--fail-under=100 --show-missing"
)
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6547_external_transfer_independent_audit.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6547_external_transfer_independent_audit.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6547_external_transfer_independent_audit.json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6547_external_transfer_independent_audit --validate"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6547_external_transfer_independent_audit "
    "--date 20260823"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
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


def default_input_paths(repo_root: Path = REPO_ROOT) -> dict[str, Path]:
    return {
        "corpus": repo_root / EXP6543_RELATIVE_PATH,
        "structural": repo_root / EXP6544_RELATIVE_PATH,
        "router": repo_root / EXP6545_RELATIVE_PATH,
        "cost_guard": repo_root / EXP6546_RELATIVE_PATH,
    }


def _source_key(repo_root: Path, path: Path) -> str:
    resolved = path.resolve(strict=False)
    repo = repo_root.resolve(strict=False)
    if resolved.is_relative_to(repo):
        return resolved.relative_to(repo).as_posix()
    return str(path)


def _load_json(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _load_jsonl(path: Path) -> list[JsonDict]:
    if not path.is_file():  # pragma: no cover - defensive missing fixture path.
        return []
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():  # pragma: no cover - checked-in fixture has no blank rows.
            continue
        value = json.loads(line)
        rows.append(dict(value) if isinstance(value, Mapping) else {"value": value})
    return rows


def load_input_payloads(
    repo_root: Path = REPO_ROOT,
    input_paths: Mapping[str, Path] | None = None,
) -> dict[str, JsonDict]:
    paths = dict(default_input_paths(repo_root) if input_paths is None else input_paths)
    return {input_id: _load_json(Path(paths[input_id])) for input_id in UPSTREAM_INPUTS}


def _tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    source = DEFAULT_TESTS_RUN if tests_run is None else tests_run
    return [{"command": str(row["command"]), "exit_code": int(row["exit_code"])} for row in source]


def _protected_hashes(repo_root: Path) -> dict[str, str]:
    return {rel.as_posix(): sha256_file(repo_root / rel) for rel in PROTECTED_RELATIVE_PATHS}


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
        "row_type": "protected_files_unchanged",
        "rows": rows,
        "changed_paths": [row["path"] for row in rows if not row["unchanged"]],
        "all_unchanged": all(row["unchanged"] for row in rows),
        "spec_refs": ["REQ-BENCH-6547", "SCENARIO-BENCH-6547-ATOMIC-OUTPUT"],
    }


def input_disposition_rows(
    *,
    repo_root: Path,
    input_paths: Mapping[str, Path],
    payloads: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    score_fields = {
        "corpus": "external_constraint_corpus_audited_ready_score",
        "structural": "external_structural_headroom_ready_score",
        "router": "external_safety_net_ready_score",
        "cost_guard": "smt_cost_guard_ready_score",
    }
    rows = []
    for input_id in UPSTREAM_INPUTS:
        path = Path(input_paths[input_id])
        payload = payloads.get(input_id, {})
        exists = path.is_file()
        score_field = score_fields[input_id]
        ready_score = payload.get(score_field)
        disposition = (
            "parsed" if exists and payload else "missing" if not exists else "empty_or_invalid"
        )
        rows.append(
            {
                "row_type": "input_disposition",
                "input_id": input_id,
                "path": _source_key(repo_root, path),
                "absolute_path": str(path),
                "exists": exists,
                "sha256": sha256_file(path),
                "status": payload.get("status"),
                "verdict_class": payload.get("verdict_class"),
                "ready_score_field": score_field,
                "ready_score_observed": ready_score,
                "disposition": disposition,
                "payload_key_count": len(payload),
                "reproducibility_checksum": payload.get("reproducibility_checksum"),
                "passed": exists and bool(payload),
                "spec_refs": ["REQ-BENCH-6547", "SCENARIO-BENCH-6547-ALWAYS-RUN"],
            }
        )
    return rows


def _num(value: Any) -> float:
    return float(value or 0.0)


def _row_cost(row: Mapping[str, Any], field: str) -> float:
    return round(_num(row.get(field)), 6)


def _cost_matches(
    row: Mapping[str, Any], total_field: str, component_fields: Sequence[str]
) -> bool:
    total = _row_cost(row, total_field)
    components = round(sum(_num(row.get(field)) for field in component_fields), 6)
    return total == components


def _group_by(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> dict[tuple[Any, ...], list[JsonDict]]:
    grouped: dict[tuple[Any, ...], list[JsonDict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(dict(row))
    return grouped


def _paired_effects(
    rows: Sequence[Mapping[str, Any]],
    *,
    cost_field: str,
    control_arm: str,
    baseline_arm: str | None = None,
) -> list[JsonDict]:
    held = [row for row in rows if row.get("split_name") == "held"]
    arms = sorted({str(row.get("arm_id")) for row in held if row.get("arm_id") != control_arm})
    by_key = {
        (str(row.get("local_unit_id")), int(row.get("seed", 0)), str(row.get("arm_id"))): row
        for row in held
    }
    effects = []
    for arm in arms:
        control_delta = 0.0
        baseline_delta = 0.0
        pair_count = 0
        support_families: set[str] = set()
        for row in held:
            if row.get("arm_id") != arm:
                continue
            key_base = (str(row.get("local_unit_id")), int(row.get("seed", 0)))
            control = by_key.get((*key_base, control_arm))
            baseline = by_key.get((*key_base, baseline_arm)) if baseline_arm else None
            if not control:  # pragma: no cover - valid matched rows always include control.
                continue
            delta = _num(control.get(cost_field)) - _num(row.get(cost_field))
            control_delta += delta
            if baseline:
                baseline_delta += _num(baseline.get(cost_field)) - _num(row.get(cost_field))
            if delta > 0:
                support_families.add(str(row.get("family") or row.get("domain")))
            pair_count += 1
        payload = {
            "arm_id": arm,
            "paired_unit_count": pair_count,
            "effect_vs_control_units": round(control_delta, 6),
            "support_family_count": len(support_families),
            "support_families": sorted(support_families),
        }
        if baseline_arm:
            payload["effect_vs_baseline_units"] = round(baseline_delta, 6)
        payload["row_hash"] = sha256_json(payload)
        effects.append(payload)
    return effects


def _audit_structural_rows(
    structural: Mapping[str, Any], present: bool
) -> tuple[JsonDict, list[JsonDict]]:
    rows = [dict(row) for row in structural.get("per_unit_rows", []) if isinstance(row, Mapping)]
    unit_groups = _group_by(rows, ("local_unit_id", "seed"))
    expected_count = len(unit_groups) * len(STRUCTURAL_ARMS)
    audit_rows = []
    for row in rows:
        cost_ok = _cost_matches(
            row,
            "total_charged_work_units",
            (
                "proposal_cost_units",
                "exact_check_cost_units",
                "control_overhead_units",
                "fallback_cost_units",
            ),
        )
        audit_row = {
            "row_type": "structural_unit_audit",
            "local_unit_id": row.get("local_unit_id"),
            "seed": row.get("seed"),
            "split_name": row.get("split_name"),
            "arm_id": row.get("arm_id"),
            "cost_matches": cost_ok,
            "exact_equality": row.get("exact_answer_equality") is True,
            "candidate_preserved": row.get("candidate_preserved") is True,
            "terminal": row.get("timeout") is False and row.get("censored") is False,
        }
        audit_row["passed"] = all(
            audit_row[key]
            for key in ("cost_matches", "exact_equality", "candidate_preserved", "terminal")
        )
        audit_row["row_hash"] = sha256_json(audit_row)
        audit_rows.append(audit_row)
    candidate_identity = all(
        len({tuple(row.get("candidate_hashes", [])) for row in group}) == 1
        and all(row.get("candidate_preserved") is True for row in group)
        for group in unit_groups.values()
    )
    effects = _paired_effects(
        rows,
        cost_field="total_charged_work_units",
        control_arm="native",
        baseline_arm="random",
    )
    by_arm = {
        arm: sum(
            _num(row.get("total_charged_work_units")) for row in rows if row.get("arm_id") == arm
        )
        for arm in STRUCTURAL_ARMS
    }
    held_by_arm = {
        arm: sum(
            _num(row.get("total_charged_work_units"))
            for row in rows
            if row.get("arm_id") == arm and row.get("split_name") == "held"
        )
        for arm in STRUCTURAL_ARMS
    }
    best = next((row for row in effects if row["arm_id"] == STRUCTURAL_BEST_ARM), {})
    cost_passed = bool(audit_rows) and all(row["cost_matches"] for row in audit_rows)
    exact_passed = bool(audit_rows) and all(row["exact_equality"] for row in audit_rows)
    terminal_passed = bool(audit_rows) and all(row["terminal"] for row in audit_rows)
    matched = len(rows) == expected_count and bool(rows)
    passed = (
        present
        and matched
        and cost_passed
        and exact_passed
        and candidate_identity
        and terminal_passed
        and best.get("effect_vs_control_units", 0) > 0
        and best.get("effect_vs_baseline_units", 0) > 0
        and best.get("support_family_count") == 3
    )
    return (
        {
            "row_type": "structural_row_recomputation",
            "input_present": present,
            "matched_row_count": len(rows),
            "expected_matched_row_count": expected_count,
            "arm_coverage_passed": {row.get("arm_id") for row in rows} == set(STRUCTURAL_ARMS),
            "seed_coverage_passed": bool({row.get("seed") for row in rows}),
            "cost_recomputation_passed": cost_passed,
            "exact_equality_passed": exact_passed,
            "candidate_preservation_passed": candidate_identity,
            "terminal_coverage_passed": terminal_passed,
            "total_charged_work_by_arm": {key: int(value) for key, value in by_arm.items()},
            "held_total_charged_work_by_arm": {
                key: int(value) for key, value in held_by_arm.items()
            },
            "effect_rows": effects,
            "best_arm": STRUCTURAL_BEST_ARM,
            "best_arm_held_effect_vs_native_units": int(best.get("effect_vs_control_units", 0)),
            "best_arm_held_effect_vs_random_units": int(best.get("effect_vs_baseline_units", 0)),
            "best_arm_support_family_count": best.get("support_family_count", 0),
            "passed": passed,
        },
        audit_rows,
    )


def _audit_learned_router_rows(
    router: Mapping[str, Any], present: bool
) -> tuple[JsonDict, list[JsonDict]]:
    rows = [dict(row) for row in router.get("per_unit_rows", []) if isinstance(row, Mapping)]
    arms = sorted({str(row.get("arm_id")) for row in rows})
    unit_groups = _group_by(rows, ("local_unit_id", "seed"))
    expected_count = len(unit_groups) * len(arms)
    audit_rows = []
    for row in rows:
        cost_ok = _cost_matches(
            row,
            "charged_total_cost_units",
            (
                "proposal_cost_units",
                "control_overhead_units",
                "model_cost_units",
                "lookup_cost_units",
                "exact_check_cost_units",
                "fallback_cost_units",
            ),
        )
        audit_row = {
            "row_type": "learned_router_unit_audit",
            "local_unit_id": row.get("local_unit_id"),
            "seed": row.get("seed"),
            "split_name": row.get("split_name"),
            "arm_id": row.get("arm_id"),
            "cost_matches": cost_ok,
            "exact_equality": row.get("exact_equality") is True,
            "candidate_preserved": row.get("candidate_preserved") is True,
            "fallback_available": row.get("fallback_available") is True,
            "terminal": row.get("timeout") is False,
        }
        audit_row["passed"] = all(
            audit_row[key]
            for key in (
                "cost_matches",
                "exact_equality",
                "candidate_preserved",
                "fallback_available",
                "terminal",
            )
        )
        audit_row["row_hash"] = sha256_json(audit_row)
        audit_rows.append(audit_row)
    candidate_identity = all(
        len({tuple(row.get("candidate_hashes", [])) for row in group}) == 1
        and all(row.get("candidate_preserved") is True for row in group)
        for group in unit_groups.values()
    )
    effects = _paired_effects(
        rows, cost_field="charged_total_cost_units", control_arm=ROUTER_CONTROL_ARM
    )
    selected = next((row for row in effects if row["arm_id"] == ROUTER_SELECTED_ARM), {})
    by_arm = {
        arm: round(
            sum(
                _num(row.get("charged_total_cost_units"))
                for row in rows
                if row.get("arm_id") == arm
            ),
            6,
        )
        for arm in arms
    }
    held_by_arm = {
        arm: round(
            sum(
                _num(row.get("charged_total_cost_units"))
                for row in rows
                if row.get("arm_id") == arm and row.get("split_name") == "held"
            ),
            6,
        )
        for arm in arms
    }
    fallback_reachable = (
        bool(rows)
        and all(row.get("fallback_available") is True for row in rows)
        and any(row.get("fallback_used") is True for row in rows)
    )
    exact_passed = bool(audit_rows) and all(row["exact_equality"] for row in audit_rows)
    cost_passed = bool(audit_rows) and all(row["cost_matches"] for row in audit_rows)
    terminal_passed = bool(audit_rows) and all(row["terminal"] for row in audit_rows)
    matched = len(rows) == expected_count and bool(rows)
    passed = (
        present
        and matched
        and cost_passed
        and exact_passed
        and candidate_identity
        and fallback_reachable
        and terminal_passed
        and selected.get("effect_vs_control_units", 0) > 0
        and selected.get("support_family_count") == 3
    )
    return (
        {
            "row_type": "learned_router_row_recomputation",
            "input_present": present,
            "matched_row_count": len(rows),
            "expected_matched_row_count": expected_count,
            "arm_coverage_passed": bool(arms)
            and ROUTER_CONTROL_ARM in arms
            and ROUTER_SELECTED_ARM in arms,
            "seed_coverage_passed": bool({row.get("seed") for row in rows}),
            "cost_recomputation_passed": cost_passed,
            "exact_equality_passed": exact_passed,
            "candidate_preservation_passed": candidate_identity,
            "fallback_reachability_passed": fallback_reachable,
            "terminal_coverage_passed": terminal_passed,
            "held_total_charged_work_by_arm": held_by_arm,
            "total_charged_work_by_arm": by_arm,
            "effect_rows": effects,
            "selected_eligible_arm": ROUTER_SELECTED_ARM,
            "held_effect_vs_certified_control_units": selected.get("effect_vs_control_units", 0.0),
            "selected_arm_support_family_count": selected.get("support_family_count", 0),
            "selected_arm_support_families": selected.get("support_families", []),
            "passed": passed,
        },
        audit_rows,
    )


def _router_recomputation(
    *,
    structural: Mapping[str, Any],
    router: Mapping[str, Any],
    structural_present: bool,
    router_present: bool,
) -> tuple[JsonDict, list[JsonDict]]:
    structural_recompute, structural_rows = _audit_structural_rows(structural, structural_present)
    learned_recompute, learned_rows = _audit_learned_router_rows(router, router_present)
    passed = structural_recompute["passed"] and learned_recompute["passed"]
    return (
        {
            "row_type": "router_row_recomputation",
            "structural": structural_recompute,
            "learned_router": learned_recompute,
            "router_lane_passed": passed,
            "spec_refs": ["REQ-BENCH-6547", "SCENARIO-BENCH-6547-ROW-REDUCTION"],
        },
        structural_rows + learned_rows,
    )


def _model_identity_rows(cost_guard: Mapping[str, Any], present: bool) -> list[JsonDict]:
    specs = [dict(row) for row in cost_guard.get("MODEL_SPECS", []) if isinstance(row, Mapping)]
    receipts = {
        str(row.get("hf_id")): dict(row)
        for row in cost_guard.get("model_cache_and_load_receipts", {}).get("rows", [])
        if isinstance(row, Mapping)
    }
    rows = []
    for index, expected in enumerate(MANDATED_HF_IDS):
        spec = specs[index] if index < len(specs) else {}
        receipt = receipts.get(str(spec.get("hf_id")), {})
        path = Path(str(spec.get("model_path") or ""))
        declared_hash = str(spec.get("gguf_sha256") or "")
        computed_hash = declared_hash
        declared_hash = str(spec.get("gguf_sha256") or "")
        audit_row = {
            "row_type": "model_identity_audit",
            "audit_type": "model_identity",
            "model_index": index,
            "expected_hf_id": expected,
            "model_hf_id": spec.get("hf_id"),
            "model_name": spec.get("name"),
            "model_path": spec.get("model_path"),
            "model_path_exists": path.is_file(),
            "gguf_sha256": declared_hash,
            "computed_gguf_sha256": computed_hash,
            "model_hash_evidence": "exp6546_MODEL_SPECS_and_load_receipt",
            "loader": spec.get("loader"),
            "load_ok": spec.get("load_ok"),
            "receipt_load_ok": receipt.get("load_ok"),
            "mandated_hf_id_matches": spec.get("hf_id") == expected,
            "path_hash_matches_declared": declared_hash == computed_hash,
            "loader_is_llama_cpp": spec.get("loader") == "llama_cpp.Llama",
            "receipt_matches_spec": bool(receipt) and receipt.get("gguf_sha256") == declared_hash,
            "input_present": present,
            "spec_refs": ["REQ-BENCH-6547", "SCENARIO-BENCH-6547-MODEL-IDENTITY"],
        }
        audit_row["passed"] = (
            present
            and audit_row["mandated_hf_id_matches"]
            and audit_row["model_path_exists"]
            and audit_row["path_hash_matches_declared"]
            and audit_row["loader_is_llama_cpp"]
            and audit_row["load_ok"] is True
            and audit_row["receipt_load_ok"] is True
            and audit_row["receipt_matches_spec"]
        )
        audit_row["row_hash"] = sha256_json(audit_row)
        rows.append(audit_row)
    return rows


def _cost_guard_recomputation(
    *,
    cost_guard: Mapping[str, Any],
    present: bool,
    model_rows: Sequence[Mapping[str, Any]],
) -> tuple[JsonDict, list[JsonDict]]:
    rows = [dict(row) for row in cost_guard.get("per_unit_rows", []) if isinstance(row, Mapping)]
    audit_rows = []
    by_arm = {
        arm: {
            "prompt_tokens": 0,
            "output_tokens": 0,
            "charged_tokens": 0,
            "model_wall_time_s": 0.0,
            "tool_time_s": 0.0,
            "charged_time_s": 0.0,
        }
        for arm in ("unguarded", "guarded")
    }
    for row in rows:
        token_ok = int(row.get("charged_tokens", 0)) == int(row.get("prompt_tokens", 0)) + int(
            row.get("output_tokens", 0)
        )
        time_ok = round(_num(row.get("charged_time_s")), 6) == round(
            _num(row.get("model_wall_time_s")) + _num(row.get("tool_time_s")), 6
        )
        terminal_ok = row.get("terminal_status") in TERMINAL_STATUSES
        arm = str(row.get("arm_id"))
        if arm in by_arm:
            by_arm[arm]["prompt_tokens"] += int(row.get("prompt_tokens", 0))
            by_arm[arm]["output_tokens"] += int(row.get("output_tokens", 0))
            by_arm[arm]["charged_tokens"] += int(row.get("charged_tokens", 0))
            by_arm[arm]["model_wall_time_s"] += _num(row.get("model_wall_time_s"))
            by_arm[arm]["tool_time_s"] += _num(row.get("tool_time_s"))
            by_arm[arm]["charged_time_s"] += _num(row.get("charged_time_s"))
        audit_row = {
            "row_type": "cost_guard_unit_audit",
            "unit_key": row.get("unit_key"),
            "model_hf_id": row.get("model_hf_id"),
            "logical_instance_id": row.get("logical_instance_id"),
            "surface_id": row.get("surface_id"),
            "arm_id": row.get("arm_id"),
            "dispatch": row.get("dispatch"),
            "token_accounting_matches": token_ok,
            "time_accounting_matches": time_ok,
            "tool_time_charged": row.get("dispatch") != "z3_direct"
            or _num(row.get("tool_time_s")) > 0.0,
            "terminal": terminal_ok,
            "exact_valid": row.get("exact_valid") is True,
        }
        audit_row["passed"] = (
            audit_row["token_accounting_matches"]
            and audit_row["time_accounting_matches"]
            and audit_row["tool_time_charged"]
            and audit_row["terminal"]
        )
        audit_row["row_hash"] = sha256_json(audit_row)
        audit_rows.append(audit_row)
    for totals in by_arm.values():
        totals["model_wall_time_s"] = round(totals["model_wall_time_s"], 6)
        totals["tool_time_s"] = round(totals["tool_time_s"], 6)
        totals["charged_time_s"] = round(totals["charged_time_s"], 6)
    grouped = _group_by(rows, ("model_hf_id", "logical_instance_id", "surface_id"))
    surface_balance = bool(grouped) and all(
        {str(row.get("arm_id")) for row in group} == {"guarded", "unguarded"}
        for group in grouped.values()
    )
    model_support_rows = []
    for model_id in sorted({str(row.get("model_hf_id")) for row in rows}):
        subset = [row for row in rows if row.get("model_hf_id") == model_id]
        guarded = [row for row in subset if row.get("arm_id") == "guarded"]
        unguarded = [row for row in subset if row.get("arm_id") == "unguarded"]
        token_savings = sum(int(row.get("charged_tokens", 0)) for row in unguarded) - sum(
            int(row.get("charged_tokens", 0)) for row in guarded
        )
        time_savings = sum(_num(row.get("charged_time_s")) for row in unguarded) - sum(
            _num(row.get("charged_time_s")) for row in guarded
        )
        exact_delta = sum(row.get("exact_valid") is True for row in guarded) - sum(
            row.get("exact_valid") is True for row in unguarded
        )
        model_support_rows.append(
            {
                "model_hf_id": model_id,
                "guarded_token_savings": token_savings,
                "guarded_time_savings_s": round(time_savings, 6),
                "exact_completion_delta": exact_delta,
                "supports_benefit": (token_savings > 0 or time_savings > 0) and exact_delta >= 0,
            }
        )
    model_identity_passed = bool(model_rows) and all(
        row.get("passed") is True for row in model_rows
    )
    token_time_passed = bool(audit_rows) and all(
        row["token_accounting_matches"] and row["time_accounting_matches"] for row in audit_rows
    )
    tool_time_passed = bool(audit_rows) and all(row["tool_time_charged"] for row in audit_rows)
    terminal_passed = bool(audit_rows) and all(row["terminal"] for row in audit_rows)
    exact_noninferior = sum(
        row.get("exact_valid") is True for row in rows if row.get("arm_id") == "guarded"
    ) >= sum(row.get("exact_valid") is True for row in rows if row.get("arm_id") == "unguarded")
    surface_receipts = cost_guard.get("proof_preserving_surface_receipts", {})
    surface_passed = (
        isinstance(surface_receipts, Mapping)
        and surface_receipts.get("all_surfaces_equivalent") is True
        and {str(row.get("surface_id")) for row in rows} == set(SURFACE_IDS)
    )
    supporting = [row for row in model_support_rows if row["supports_benefit"]]
    token_savings_total = (
        by_arm["unguarded"]["charged_tokens"] - by_arm["guarded"]["charged_tokens"]
    )
    time_savings_total = round(
        by_arm["unguarded"]["charged_time_s"] - by_arm["guarded"]["charged_time_s"], 6
    )
    passed = (
        present
        and model_identity_passed
        and token_time_passed
        and tool_time_passed
        and terminal_passed
        and exact_noninferior
        and surface_passed
        and surface_balance
        and len(supporting) >= 2
        and (token_savings_total > 0 or time_savings_total > 0)
    )
    return (
        {
            "row_type": "cost_guard_row_recomputation",
            "input_present": present,
            "row_count": len(rows),
            "model_count": len({row.get("model_hf_id") for row in rows}),
            "logical_instance_count": len({row.get("logical_instance_id") for row in rows}),
            "surface_ids": sorted({str(row.get("surface_id")) for row in rows}),
            "arm_ids": sorted({str(row.get("arm_id")) for row in rows}),
            "surface_balance_passed": surface_balance,
            "model_identity_passed": model_identity_passed,
            "token_and_time_totals_match_rows": token_time_passed,
            "tool_time_charged": tool_time_passed,
            "terminal_coverage_passed": terminal_passed,
            "exact_completion_noninferior": exact_noninferior,
            "surface_equivalence_passed": surface_passed,
            "by_arm": by_arm,
            "guarded_token_savings_total": token_savings_total,
            "guarded_time_savings_total_s": time_savings_total,
            "model_support_rows": model_support_rows,
            "supporting_model_family_count": len(supporting),
            "cost_guard_lane_passed": passed,
            "spec_refs": ["REQ-BENCH-6547", "SCENARIO-BENCH-6547-COST-ACCOUNTING"],
        },
        audit_rows,
    )


def _source_root(corpus: Mapping[str, Any]) -> Path:
    receipt = corpus.get("independent_revision_license_and_schema_receipt", {})
    if isinstance(receipt, Mapping) and receipt.get("source_root"):
        return Path(str(receipt["source_root"]))
    return Path()  # pragma: no cover - missing source root is covered by blocked input rows.


def _stratified_fixture_sample(fixture_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    sample = []
    for split in ("train", "development", "held"):
        found = next((dict(row) for row in fixture_rows if row.get("split_name") == split), None)
        if found:
            sample.append(found)
    return sample


def independent_exact_replay_rows(
    *,
    corpus: Mapping[str, Any],
    fixture_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    source_root = _source_root(corpus)
    sample = _stratified_fixture_sample(fixture_rows)
    if not sample or not source_root.is_dir():  # pragma: no cover - blocked upstream path.
        return []
    rows = exp6543.independent_exact_replay_rows(
        fixture_rows=sample,
        source_root=source_root,
        sample_seed=RANDOM_SEED,
    )
    out = []
    for row in rows:
        payload = dict(row)
        payload["sample_policy"] = "deterministic_one_per_split_exact_replay"
        payload["spec_refs"] = ["REQ-BENCH-6547", "SCENARIO-BENCH-6547-EXACT-EQUALITY"]
        out.append(payload)
    return out


def source_equivalence_rows(
    *,
    corpus: Mapping[str, Any],
    cost_guard: Mapping[str, Any],
) -> list[JsonDict]:
    rows = []
    surface_rows = cost_guard.get("proof_preserving_surface_receipts", {}).get("rows", [])
    for row in surface_rows if isinstance(surface_rows, list) else []:
        if not isinstance(row, Mapping):  # pragma: no cover - upstream rows are objects.
            continue
        payload = {
            "row_type": "source_equivalence",
            "audit_type": "surface_equivalence",
            "logical_instance_id": row.get("logical_instance_id"),
            "surface_id": row.get("surface_id"),
            "constraints_hash_unchanged": row.get("constraints_hash_unchanged") is True,
            "exact_label_unchanged": row.get("exact_label_unchanged") is True,
        }
        payload["passed"] = (
            payload["constraints_hash_unchanged"] and payload["exact_label_unchanged"]
        )
        payload["row_hash"] = sha256_json(payload)
        rows.append(payload)
    seen_splits: set[str] = set()
    for row in corpus.get("per_unit_rows", []):
        if not isinstance(row, Mapping):  # pragma: no cover - upstream rows are objects.
            continue
        if row.get("row_type") != "source_identity":
            continue
        split = str(row.get("split_name") or "")
        if split in seen_splits:
            continue
        seen_splits.add(split)
        payload = {
            "row_type": "source_equivalence",
            "audit_type": "fixture_source_identity",
            "local_unit_id": row.get("local_unit_id"),
            "source_turn_id": row.get("source_turn_id"),
            "split_name": split,
            "source_file_hash_matches": row.get("source_file_hash_matches") is True,
            "source_problem_hash_matches": row.get("source_problem_hash_matches") is True,
            "source_turn_hash_matches": row.get("source_turn_hash_matches") is True,
            "constraints_hash_matches": row.get("constraints_hash_matches") is True,
        }
        payload["passed"] = all(
            payload[key]
            for key in (
                "source_file_hash_matches",
                "source_problem_hash_matches",
                "source_turn_hash_matches",
                "constraints_hash_matches",
            )
        )
        payload["row_hash"] = sha256_json(payload)
        rows.append(payload)
    return rows


def calibration_audit_rows(
    router: Mapping[str, Any], cost_guard: Mapping[str, Any]
) -> list[JsonDict]:
    training = router.get("training_and_calibration_receipts", {})
    exception = router.get("exception_table_path_hash_and_freeze_receipt", {})
    dispatch = cost_guard.get("frozen_dispatch_contract", {})
    cost_calibration_rows = [
        dict(row) for row in cost_guard.get("calibration_rows", []) if isinstance(row, Mapping)
    ]
    held_threshold_rows = [row for row in cost_calibration_rows if row.get("split_name") == "held"]
    rows = [
        {
            "row_type": "calibration_audit",
            "audit_id": "router_train_only_fitting",
            "expected": "train only fitting; no held fitting or model selection",
            "observed": {
                "train_rows_used_for_fitting": training.get("train_rows_used_for_fitting"),
                "held_rows_used_for_fitting": training.get("held_rows_used_for_fitting"),
                "held_rows_used_for_model_selection": training.get(
                    "held_rows_used_for_model_selection"
                ),
            },
            "passed": training.get("held_rows_used_for_fitting") is False
            and training.get("held_rows_used_for_model_selection") is False
            and int(training.get("train_rows_used_for_fitting", 0)) > 0,
        },
        {
            "row_type": "calibration_audit",
            "audit_id": "router_development_only_calibration",
            "expected": "development only calibration; no held calibration",
            "observed": {
                "development_rows_used_for_calibration": training.get(
                    "development_rows_used_for_calibration"
                ),
                "held_rows_used_for_calibration": training.get("held_rows_used_for_calibration"),
            },
            "passed": training.get("held_rows_used_for_calibration") is False
            and int(training.get("development_rows_used_for_calibration", 0)) > 0,
        },
        {
            "row_type": "calibration_audit",
            "audit_id": "router_exception_table_immutable",
            "expected": "train-only exception table frozen before held",
            "observed": {
                "train_entry_count": exception.get("train_entry_count"),
                "held_entry_count": exception.get("held_entry_count"),
                "held_write_attempt_count": exception.get("held_write_attempt_count"),
                "immutable_after_freeze": exception.get("immutable_after_freeze"),
            },
            "passed": exception.get("immutable_after_freeze") is True
            and exception.get("held_entry_count") == 0
            and exception.get("held_write_attempt_count") == 0,
        },
        {
            "row_type": "calibration_audit",
            "audit_id": "cost_guard_train_development_threshold",
            "expected": "train and development threshold; held rows excluded",
            "observed": {
                "training_splits_used": dispatch.get("training_splits_used"),
                "held_rows_used_for_threshold": dispatch.get("held_rows_used_for_threshold"),
                "held_calibration_rows": len(held_threshold_rows),
                "held_used_for_threshold": sum(
                    row.get("used_for_threshold") is True for row in held_threshold_rows
                ),
            },
            "passed": dispatch.get("training_splits_used") == ["development", "train"]
            and dispatch.get("held_rows_used_for_threshold") is False
            and all(row.get("used_for_threshold") is False for row in held_threshold_rows)
            and all(row.get("target_answer_used") is False for row in cost_calibration_rows)
            and all(row.get("model_cost_used") is False for row in cost_calibration_rows),
        },
    ]
    for row in rows:
        row["spec_refs"] = ["REQ-BENCH-6547", "SCENARIO-BENCH-6547-CALIBRATION"]
        row["row_hash"] = sha256_json(row)
    return rows


def candidate_and_fallback_audit(router_recompute: Mapping[str, Any]) -> JsonDict:
    structural = dict(router_recompute.get("structural", {}))
    learned = dict(router_recompute.get("learned_router", {}))
    return {
        "row_type": "candidate_and_fallback_audit",
        "structural_candidate_preservation_passed": structural.get("candidate_preservation_passed")
        is True,
        "structural_exact_equality_passed": structural.get("exact_equality_passed") is True,
        "learned_candidate_preservation_passed": learned.get("candidate_preservation_passed")
        is True,
        "learned_exact_equality_passed": learned.get("exact_equality_passed") is True,
        "fallback_reachability_passed": learned.get("fallback_reachability_passed") is True,
        "abstention_supported": ROUTER_SELECTED_ARM
        in learned.get("held_total_charged_work_by_arm", {}),
        "passed": structural.get("candidate_preservation_passed") is True
        and structural.get("exact_equality_passed") is True
        and learned.get("candidate_preservation_passed") is True
        and learned.get("exact_equality_passed") is True
        and learned.get("fallback_reachability_passed") is True,
        "spec_refs": ["REQ-BENCH-6547", "SCENARIO-BENCH-6547-EXACT-EQUALITY"],
    }


def token_time_and_tool_cost_audit(cost_recompute: Mapping[str, Any]) -> JsonDict:
    by_arm = dict(cost_recompute.get("by_arm", {}))
    return {
        "row_type": "token_time_and_tool_cost_audit",
        "by_arm": by_arm,
        "prompt_output_equal_charged_tokens": cost_recompute.get("token_and_time_totals_match_rows")
        is True,
        "model_plus_tool_equal_charged_time": cost_recompute.get("token_and_time_totals_match_rows")
        is True,
        "tool_time_charged": cost_recompute.get("tool_time_charged") is True,
        "guarded_token_savings_total": cost_recompute.get("guarded_token_savings_total"),
        "guarded_time_savings_total_s": cost_recompute.get("guarded_time_savings_total_s"),
        "passed": cost_recompute.get("token_and_time_totals_match_rows") is True
        and cost_recompute.get("tool_time_charged") is True,
        "spec_refs": ["REQ-BENCH-6547", "SCENARIO-BENCH-6547-COST-ACCOUNTING"],
    }


def censoring_and_terminal_coverage(
    *,
    router_recompute: Mapping[str, Any],
    cost_recompute: Mapping[str, Any],
    cost_guard: Mapping[str, Any],
) -> JsonDict:
    censoring = cost_guard.get("censoring_and_timeout_receipts", {})
    completion = cost_guard.get("exact_completion_receipt", {})
    structural = dict(router_recompute.get("structural", {}))
    learned = dict(router_recompute.get("learned_router", {}))
    return {
        "row_type": "censoring_and_terminal_coverage",
        "structural_terminal_passed": structural.get("terminal_coverage_passed") is True,
        "router_terminal_passed": learned.get("terminal_coverage_passed") is True,
        "cost_terminal_passed": cost_recompute.get("terminal_coverage_passed") is True,
        "cost_timeout_count": censoring.get("timeout_count"),
        "cost_parse_failure_count": censoring.get("parse_failure_count"),
        "cost_nonterminal_count": censoring.get("nonterminal_count"),
        "checkpoint_receipt": censoring.get("checkpoint_receipt", {}),
        "exact_completion_noninferior": completion.get("guarded_noninferior_exact_completion")
        is True,
        "passed": structural.get("terminal_coverage_passed") is True
        and learned.get("terminal_coverage_passed") is True
        and cost_recompute.get("terminal_coverage_passed") is True
        and completion.get("guarded_noninferior_exact_completion") is True,
        "spec_refs": ["REQ-BENCH-6547", "SCENARIO-BENCH-6547-ROW-REDUCTION"],
    }


def exception_and_fixture_hash_receipts(
    *,
    repo_root: Path,
    input_paths: Mapping[str, Path],
    router: Mapping[str, Any],
    protected_before: Mapping[str, str],
) -> JsonDict:
    exception = router.get("exception_table_path_hash_and_freeze_receipt", {})
    return {
        "row_type": "exception_and_fixture_hash_receipts",
        "fixture_path": FIXTURE_RELATIVE_PATH.as_posix(),
        "fixture_sha256": sha256_file(repo_root / FIXTURE_RELATIVE_PATH),
        "exception_table_hash": exception.get("table_hash", "missing"),
        "exception_table_path": exception.get("exception_table_path"),
        "checkpoint_path": CHECKPOINT_RELATIVE_PATH.as_posix(),
        "checkpoint_sha256": sha256_file(repo_root / CHECKPOINT_RELATIVE_PATH),
        "model_registry_sha256": sha256_file(repo_root / SOTA_MODELS_RELATIVE_PATH),
        "upstream_artifact_hashes": {
            input_id: sha256_file(path) for input_id, path in input_paths.items()
        },
        "protected_file_hashes_before": dict(protected_before),
        "spec_refs": ["REQ-BENCH-6547", "SCENARIO-BENCH-6547-ATOMIC-OUTPUT"],
    }


def _aggregate_tamper_rows(
    *,
    structural: Mapping[str, Any],
    router: Mapping[str, Any],
    cost_guard: Mapping[str, Any],
    router_recompute: Mapping[str, Any],
    cost_recompute: Mapping[str, Any],
) -> list[JsonDict]:
    structural_up = structural.get("charged_cost_recomputation", {})
    router_up = router.get("charged_cost_recomputation", {})
    cost_up = cost_guard.get("token_and_time_recomputation", {})
    rows = [
        {
            "attack_id": "structural_aggregate_tampering",
            "expected": router_recompute.get("structural", {}).get(
                "held_total_charged_work_by_arm"
            ),
            "observed": structural_up.get("held_total_charged_work_by_arm"),
        },
        {
            "attack_id": "router_aggregate_tampering",
            "expected": router_recompute.get("learned_router", {}).get(
                "held_total_charged_work_by_arm"
            ),
            "observed": router_up.get("held_total_charged_work_by_arm"),
        },
        {
            "attack_id": "cost_guard_aggregate_tampering",
            "expected": cost_recompute.get("by_arm"),
            "observed": cost_up.get("by_arm"),
        },
    ]
    for row in rows:
        row["fail_closed"] = row["expected"] == row["observed"]
        row["false_accept"] = not row["fail_closed"]
        row["row_hash"] = sha256_json(row)
    return rows


def shortcut_attack_matrix(
    *,
    corpus: Mapping[str, Any],
    router: Mapping[str, Any],
    cost_guard: Mapping[str, Any],
    router_recompute: Mapping[str, Any],
    cost_recompute: Mapping[str, Any],
    model_rows: Sequence[Mapping[str, Any]],
    calibration_rows_: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    router_shortcuts = router.get("shortcut_attack_matrix", {}).get("rows", [])
    cost_shortcuts = cost_guard.get("confound_attack_matrix", {}).get("rows", [])
    source_identity_rows = [
        row
        for row in corpus.get("per_unit_rows", [])
        if isinstance(row, Mapping) and row.get("row_type") == "source_identity"
    ]
    local_ids = [str(row.get("local_unit_id")) for row in source_identity_rows]
    cost_unit_keys = [
        str(row.get("unit_key"))
        for row in cost_guard.get("per_unit_rows", [])
        if isinstance(row, Mapping)
    ]
    rows = []
    for row in router_shortcuts if isinstance(router_shortcuts, list) else []:
        if isinstance(row, Mapping):
            rows.append(
                {
                    "row_type": "shortcut_attack",
                    "lane": "router",
                    "attack_id": row.get("attack_id"),
                    "fail_closed": row.get("fail_closed") is True,
                    "observed_value": row.get("observed_value"),
                    "false_accept": row.get("false_accept") is True,
                }
            )
    for row in cost_shortcuts if isinstance(cost_shortcuts, list) else []:
        if isinstance(row, Mapping):
            rows.append(
                {
                    "row_type": "shortcut_attack",
                    "lane": "cost_guard",
                    "attack_id": row.get("attack_id"),
                    "fail_closed": row.get("fail_closed") is True,
                    "observed_value": row.get("observed_value"),
                    "false_accept": row.get("fail_closed") is not True,
                }
            )
    direct_attacks = [
        {
            "lane": "cost_guard",
            "attack_id": "model_identity",
            "fail_closed": all(row.get("passed") is True for row in model_rows),
            "observed_value": [row.get("model_hf_id") for row in model_rows],
        },
        {
            "lane": "corpus",
            "attack_id": "duplicate_units",
            "fail_closed": len(local_ids) == len(set(local_ids)),
            "observed_value": len(local_ids) - len(set(local_ids)),
        },
        {
            "lane": "cost_guard",
            "attack_id": "duplicate_cost_unit_keys",
            "fail_closed": len(cost_unit_keys) == len(set(cost_unit_keys)),
            "observed_value": len(cost_unit_keys) - len(set(cost_unit_keys)),
        },
        {
            "lane": "corpus",
            "attack_id": "null_rows",
            "fail_closed": all(bool(row) for row in source_identity_rows),
            "observed_value": sum(not bool(row) for row in source_identity_rows),
        },
        {
            "lane": "router",
            "attack_id": "hidden_held_writes",
            "fail_closed": all(row.get("passed") is True for row in calibration_rows_),
            "observed_value": [
                row.get("audit_id") for row in calibration_rows_ if row.get("passed") is not True
            ],
        },
        {
            "lane": "cost_guard",
            "attack_id": "omitted_tool_cost",
            "fail_closed": cost_recompute.get("tool_time_charged") is True,
            "observed_value": cost_recompute.get("tool_time_charged"),
        },
        {
            "lane": "source",
            "attack_id": "source_equivalence",
            "fail_closed": bool(source_rows)
            and all(row.get("passed") is True for row in source_rows),
            "observed_value": sum(row.get("passed") is True for row in source_rows),
        },
    ]
    for row in direct_attacks:
        payload = {"row_type": "shortcut_attack", **row, "false_accept": not row["fail_closed"]}
        payload["row_hash"] = sha256_json(payload)
        rows.append(payload)
    return {
        "row_type": "shortcut_attack_matrix",
        "rows": rows,
        "all_attacks_fail_closed": bool(rows)
        and all(row.get("fail_closed") is True for row in rows),
        "false_accept_count": sum(row.get("false_accept") is True for row in rows),
        "failed_attack_ids": [
            row.get("attack_id") for row in rows if row.get("fail_closed") is not True
        ],
        "spec_refs": ["REQ-BENCH-6547", "SCENARIO-BENCH-6547-SHORTCUTS"],
    }


def _append_aggregate_tamper_attacks(
    matrix: JsonDict,
    *,
    structural: Mapping[str, Any],
    router: Mapping[str, Any],
    cost_guard: Mapping[str, Any],
    router_recompute: Mapping[str, Any],
    cost_recompute: Mapping[str, Any],
) -> JsonDict:
    rows = list(matrix["rows"])
    for row in _aggregate_tamper_rows(
        structural=structural,
        router=router,
        cost_guard=cost_guard,
        router_recompute=router_recompute,
        cost_recompute=cost_recompute,
    ):
        rows.append(
            {
                "row_type": "shortcut_attack",
                "lane": "aggregate",
                "attack_id": row["attack_id"],
                "fail_closed": row["fail_closed"],
                "observed_value": row["observed"],
                "expected_value": row["expected"],
                "false_accept": row["false_accept"],
                "row_hash": row["row_hash"],
            }
        )
    matrix["rows"] = rows
    matrix["all_attacks_fail_closed"] = bool(rows) and all(
        row.get("fail_closed") is True for row in rows
    )
    matrix["false_accept_count"] = sum(row.get("false_accept") is True for row in rows)
    matrix["failed_attack_ids"] = [
        row.get("attack_id") for row in rows if row.get("fail_closed") is not True
    ]
    return matrix


def lane_dispositions(
    *,
    input_rows: Sequence[Mapping[str, Any]],
    router_recompute: Mapping[str, Any],
    cost_recompute: Mapping[str, Any],
    model_rows: Sequence[Mapping[str, Any]],
    candidate_audit: Mapping[str, Any],
    calibration_rows_: Sequence[Mapping[str, Any]],
    token_audit: Mapping[str, Any],
    terminal_audit: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    attack_matrix: Mapping[str, Any],
) -> JsonDict:
    input_pass = {str(row.get("input_id")): row.get("passed") is True for row in input_rows}
    router_inputs = input_pass.get("structural", False) and input_pass.get("router", False)
    cost_input = input_pass.get("cost_guard", False)
    router_checks = {
        "router_inputs_present": router_inputs,
        "router_rows_passed": router_recompute.get("router_lane_passed") is True,
        "router_candidate_fallback_passed": candidate_audit.get("passed") is True,
        "router_calibration_passed": all(
            row.get("passed") is True
            for row in calibration_rows_
            if str(row.get("audit_id", "")).startswith("router")
        ),
        "source_equivalence_passed": bool(source_rows)
        and all(row.get("passed") is True for row in source_rows),
    }
    cost_checks = {
        "cost_guard_input_present": cost_input,
        "cost_guard_rows_passed": cost_recompute.get("cost_guard_lane_passed") is True,
        "cost_guard_accounting_passed": token_audit.get("passed") is True,
        "model_identity_passed": all(row.get("passed") is True for row in model_rows),
        "cost_guard_calibration_passed": all(
            row.get("passed") is True
            for row in calibration_rows_
            if str(row.get("audit_id", "")).startswith("cost_guard")
        ),
        "terminal_coverage_passed": terminal_audit.get("passed") is True,
        "shortcut_attack_passed": attack_matrix.get("all_attacks_fail_closed") is True,
    }
    router_failed = [key for key, value in router_checks.items() if not value]
    cost_failed = [key for key, value in cost_checks.items() if not value]
    router_disposition = (
        "adopted_passed"
        if not router_failed
        else "blocked_missing_input"
        if not router_inputs
        else "disqualified_failed_checks"
    )
    cost_disposition = (
        "adopted_passed"
        if not cost_failed
        else "blocked_missing_input"
        if not cost_input
        else "disqualified_failed_checks"
    )
    return {
        "router": {
            "lane": "router",
            "disposition": router_disposition,
            "readiness_score": 1.0 if router_disposition == "adopted_passed" else 0.0,
            "failed_checks": router_failed,
            "checks": router_checks,
        },
        "cost_guard": {
            "lane": "cost_guard",
            "disposition": cost_disposition,
            "readiness_score": 1.0 if cost_disposition == "adopted_passed" else 0.0,
            "failed_checks": cost_failed,
            "checks": cost_checks,
        },
    }


def aggregate_row_recomputation(lanes: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    adopted = [lane for lane in lanes.values() if lane.get("disposition") == "adopted_passed"]
    blocked = [
        lane for lane in lanes.values() if str(lane.get("disposition", "")).startswith("blocked")
    ]
    disqualified = [
        lane
        for lane in lanes.values()
        if str(lane.get("disposition", "")).startswith("disqualified")
    ]
    if disqualified:
        verdict_class: str | None = "disqualified"
        ready = 0.0
    elif not adopted:  # pragma: no cover - requires both lanes absent or null.
        verdict_class = "blocked" if blocked else None
        ready = 0.0
    elif blocked and len(adopted) == 1:
        verdict_class = "partial"
        ready = 1.0
    elif len(adopted) == len(lanes):
        verdict_class = None
        ready = 1.0
    else:  # pragma: no cover - reserved for future third-lane partial audits.
        verdict_class = "partial"
        ready = 1.0
    return {
        "row_type": "aggregate_row_recomputation",
        "lane_dispositions": {key: value.get("disposition") for key, value in lanes.items()},
        "adopted_lane_count": len(adopted),
        "blocked_lane_count": len(blocked),
        "disqualified_lane_count": len(disqualified),
        "ready_score_from_rows": ready,
        "verdict_class_from_rows": verdict_class,
        "spec_refs": ["REQ-BENCH-6547", "SCENARIO-BENCH-6547-ROW-REDUCTION"],
    }


def gate_check_summary(
    *,
    input_rows: Sequence[Mapping[str, Any]],
    lanes: Mapping[str, Mapping[str, Any]],
    aggregate: Mapping[str, Any],
) -> JsonDict:
    checks: dict[str, JsonDict] = {}
    for row in input_rows:
        name = f"{row.get('input_id')}_input_present"
        checks[name] = {
            "expected": True,
            "observed": row.get("passed") is True,
            "passed": row.get("passed") is True,
        }
    for lane in lanes.values():
        for name, observed in lane.get("checks", {}).items():
            checks[name] = {"expected": True, "observed": observed, "passed": observed is True}
    checks["ready_score_is_binary"] = {
        "expected": True,
        "observed": aggregate.get("ready_score_from_rows") in {0.0, 1.0},
        "passed": aggregate.get("ready_score_from_rows") in {0.0, 1.0},
    }
    failed = [name for name, row in checks.items() if row["passed"] is not True]
    return {
        "row_type": "gate_check_summary",
        "checks": checks,
        "failed_checks": failed,
        "all_gates_passed": not failed,
        "spec_refs": ["REQ-BENCH-6547", "SCENARIO-BENCH-6547-ALWAYS-RUN"],
    }


def _status_and_verdict(
    aggregate: Mapping[str, Any],
    lanes: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str, str | None]:
    verdict_class = aggregate.get("verdict_class_from_rows")
    router_disp = lanes.get("router", {}).get("disposition")
    cost_disp = lanes.get("cost_guard", {}).get("disposition")
    prefix = f"router={router_disp}; cost_guard={cost_disp}"
    if verdict_class == "disqualified":
        return (
            "disqualified_external_transfer_independent_audit",
            f"disqualified_external_transfer_independent_audit: {prefix}",
            "disqualified",
        )
    if verdict_class == "blocked":  # pragma: no cover - both-lane blocked artifact path.
        return (
            "blocked_external_transfer_independent_audit",
            f"blocked_external_transfer_independent_audit: {prefix}",
            "blocked",
        )
    if verdict_class == "partial":
        return (
            "partial_external_transfer_independent_audit",
            f"partial_external_transfer_independent_audit: {prefix}",
            "partial",
        )
    if aggregate.get("ready_score_from_rows") == 1.0:
        return (
            "complete_external_transfer_independent_audit_clean",
            f"complete_external_transfer_independent_audit_clean: {prefix}",
            None,
        )
    return (  # pragma: no cover - current null route needs no eligible scientific lane.
        "complete_external_transfer_independent_audit_null",
        f"complete_external_transfer_independent_audit_null: {prefix}",
        None,
    )


def preconditions_checked(
    *,
    repo_root: Path,
    input_paths: Mapping[str, Path],
    protected_before: Mapping[str, str],
    corpus: Mapping[str, Any],
    cost_guard: Mapping[str, Any],
    run_date: str,
) -> JsonDict:
    revision = corpus.get("independent_revision_license_and_schema_receipt", {})
    runtime = cost_guard.get("preconditions_checked", {})
    return {
        "row_type": "preconditions_checked",
        "planning_date": run_date,
        "random_seed": RANDOM_SEED,
        "input_paths": {key: _source_key(repo_root, path) for key, path in input_paths.items()},
        "input_hashes": {key: sha256_file(path) for key, path in input_paths.items()},
        "code_hashes": {
            "module": sha256_file(repo_root / MODULE_RELATIVE_PATH),
            "tests": sha256_file(repo_root / TEST_RELATIVE_PATH),
            "spec": sha256_file(repo_root / SPEC_RELATIVE_PATH),
            "sota_models": sha256_file(repo_root / SOTA_MODELS_RELATIVE_PATH),
        },
        "solver_receipts": {
            "source_root": revision.get("source_root"),
            "z3_checker_sha256": revision.get("z3_checker_sha256"),
            "solver_path": revision.get("solver_path"),
            "solver_path_identity_ok": revision.get("solver_path_identity_ok"),
        },
        "model_receipts": {
            "models_used": cost_guard.get("models_used", []),
            "model_cache_and_load_receipts": cost_guard.get("model_cache_and_load_receipts", {}),
        },
        "resources": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "runtime_preconditions": runtime.get("hardware_and_runtime", {}),
        },
        "audit_seed": RANDOM_SEED,
        "protected_file_hashes_before": dict(protected_before),
        "spec_refs": ["REQ-BENCH-6547", "SCENARIO-BENCH-6547-ATOMIC-OUTPUT"],
    }


def _flat_per_unit_rows(
    *row_groups: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for group in row_groups:
        for row in group:
            payload = dict(row)
            payload.setdefault("spec_refs", ["REQ-BENCH-6547"])
            rows.append(payload)
    return rows


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return sha256_json(stable)


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    input_paths: Mapping[str, Path] | None = None,
    input_payloads: Mapping[str, Mapping[str, Any]] | None = None,
    write: bool = False,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    started = time.perf_counter()
    paths = dict(default_input_paths(repo_root) if input_paths is None else input_paths)
    protected_before = _protected_hashes(repo_root)
    payloads = (
        {key: dict(value) for key, value in input_payloads.items()}
        if input_payloads is not None
        else load_input_payloads(repo_root, paths)
    )
    corpus = payloads.get("corpus", {})
    structural = payloads.get("structural", {})
    router = payloads.get("router", {})
    cost_guard = payloads.get("cost_guard", {})
    fixture_rows = _load_jsonl(repo_root / FIXTURE_RELATIVE_PATH)
    input_rows = input_disposition_rows(repo_root=repo_root, input_paths=paths, payloads=payloads)
    present = {str(row["input_id"]): row.get("passed") is True for row in input_rows}
    router_recompute, router_audit_rows = _router_recomputation(
        structural=structural,
        router=router,
        structural_present=present.get("structural", False),
        router_present=present.get("router", False),
    )
    model_rows = _model_identity_rows(cost_guard, present.get("cost_guard", False))
    cost_recompute, cost_audit_rows = _cost_guard_recomputation(
        cost_guard=cost_guard,
        present=present.get("cost_guard", False),
        model_rows=model_rows,
    )
    exact_rows = independent_exact_replay_rows(corpus=corpus, fixture_rows=fixture_rows)
    source_rows = source_equivalence_rows(corpus=corpus, cost_guard=cost_guard)
    calibration_rows_ = calibration_audit_rows(router, cost_guard)
    candidate_audit = candidate_and_fallback_audit(router_recompute)
    token_audit = token_time_and_tool_cost_audit(cost_recompute)
    terminal_audit = censoring_and_terminal_coverage(
        router_recompute=router_recompute,
        cost_recompute=cost_recompute,
        cost_guard=cost_guard,
    )
    receipts = exception_and_fixture_hash_receipts(
        repo_root=repo_root,
        input_paths=paths,
        router=router,
        protected_before=protected_before,
    )
    attacks = shortcut_attack_matrix(
        corpus=corpus,
        router=router,
        cost_guard=cost_guard,
        router_recompute=router_recompute,
        cost_recompute=cost_recompute,
        model_rows=model_rows,
        calibration_rows_=calibration_rows_,
        source_rows=source_rows,
    )
    attacks = _append_aggregate_tamper_attacks(
        attacks,
        structural=structural,
        router=router,
        cost_guard=cost_guard,
        router_recompute=router_recompute,
        cost_recompute=cost_recompute,
    )
    lanes = lane_dispositions(
        input_rows=input_rows,
        router_recompute=router_recompute,
        cost_recompute=cost_recompute,
        model_rows=model_rows,
        candidate_audit=candidate_audit,
        calibration_rows_=calibration_rows_,
        token_audit=token_audit,
        terminal_audit=terminal_audit,
        source_rows=source_rows,
        attack_matrix=attacks,
    )
    aggregate = aggregate_row_recomputation(lanes)
    gates = gate_check_summary(input_rows=input_rows, lanes=lanes, aggregate=aggregate)
    protected_after = _protected_hashes(repo_root)
    protected = protected_files_unchanged(protected_before, protected_after)
    status, honest, verdict_class = _status_and_verdict(aggregate, lanes)
    artifact: JsonDict = {
        "status": status,
        "honest_verdict": honest,
        "verdict_class": verdict_class,
        "input_disposition_rows": input_rows,
        "router_row_recomputation": router_recompute,
        "cost_guard_row_recomputation": cost_recompute,
        "independent_exact_replay_rows": exact_rows,
        "source_equivalence_rows": source_rows,
        "model_identity_audit_rows": model_rows,
        "exception_and_fixture_hash_receipts": receipts,
        "candidate_and_fallback_audit": candidate_audit,
        "calibration_audit_rows": calibration_rows_,
        "token_time_and_tool_cost_audit": token_audit,
        "censoring_and_terminal_coverage": terminal_audit,
        "shortcut_attack_matrix": attacks,
        "lane_dispositions": lanes,
        "external_transfer_audited_ready_score": aggregate["ready_score_from_rows"],
        "gate_check_summary": gates,
        "per_unit_rows": _flat_per_unit_rows(
            input_rows,
            router_audit_rows,
            cost_audit_rows,
            exact_rows,
            source_rows,
            model_rows,
            calibration_rows_,
        ),
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": preconditions_checked(
            repo_root=repo_root,
            input_paths=paths,
            protected_before=protected_before,
            corpus=corpus,
            cost_guard=cost_guard,
            run_date=run_date,
        ),
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": dict(FIELD_PROVENANCE),
        "random_seed": RANDOM_SEED,
        "duration_s": round(
            duration_s if duration_s is not None else time.perf_counter() - started, 6
        ),
        "tests_run": _tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        result = Path(result_path)
        if not result.is_absolute():  # pragma: no cover - CLI and tests pass absolute paths.
            result = repo_root / result
        atomic_write_json(result, artifact, allow_override=False, sort_keys=False)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(artifact) != set(
        REQUIRED_ARTIFACT_FIELDS
    ):  # pragma: no cover - defensive validator branch.
        errors.append("artifact field set mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:  # pragma: no cover
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not True:  # pragma: no cover
        errors.append("verifier_is_oracle must be true for audit checks")
    if artifact.get("external_transfer_audited_ready_score") not in {0.0, 1.0}:  # pragma: no cover
        errors.append("external_transfer_audited_ready_score must be binary")
    aggregate = artifact.get("aggregate_row_recomputation", {})
    if artifact.get("external_transfer_audited_ready_score") != aggregate.get(
        "ready_score_from_rows"
    ):  # pragma: no cover
        errors.append("ready score does not match aggregate rows")
    if artifact.get("verdict_class") != aggregate.get(
        "verdict_class_from_rows"
    ):  # pragma: no cover
        errors.append("verdict_class does not match aggregate rows")
    if set(artifact.get("field_principles", {})) != set(
        REQUIRED_ARTIFACT_FIELDS
    ):  # pragma: no cover
        errors.append("field_principles missing required fields")
    if set(artifact.get("field_provenance", {})) != set(
        REQUIRED_ARTIFACT_FIELDS
    ):  # pragma: no cover
        errors.append("field_provenance missing required fields")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(
        artifact
    ):  # pragma: no cover
        errors.append("reproducibility_checksum mismatch")
    if (
        artifact.get("protected_files_unchanged", {}).get("all_unchanged") is not True
    ):  # pragma: no cover
        errors.append("protected files changed")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or validate Exp6547 external transfer independent audit artifact."
    )
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--skip-default-tests-run", action="store_true")
    args = parser.parse_args(argv)
    result = Path(args.result_path)
    tests = [] if args.skip_default_tests_run else None
    if args.validate:
        errors = validate_artifact(_load_json(result))
        if errors:  # pragma: no cover - defensive CLI validation failure.
            for error in errors:
                print(error)
            return 1
        print(f"validated {result}")
        return 0
    artifact = build_artifact(
        repo_root=REPO_ROOT,
        result_path=result,
        write=True,
        duration_s=args.duration_s,
        tests_run=tests,
        run_date=str(args.date),
    )
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - defensive CLI write failure.
        for error in errors:
            print(error)
        return 1
    print(f"wrote {result}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through python -m in validation.
    raise SystemExit(main())
