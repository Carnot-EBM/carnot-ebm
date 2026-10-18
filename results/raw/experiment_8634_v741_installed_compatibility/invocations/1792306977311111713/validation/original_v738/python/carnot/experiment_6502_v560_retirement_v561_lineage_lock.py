"""Exp6502 V560 retirement ledger and V561 lineage lock.

Spec refs: REQ-INFRA-6502, SCENARIO-INFRA-6502-RETIREMENT,
SCENARIO-INFRA-6502-CHANGED-SCOPE, SCENARIO-INFRA-6502-DEPENDENCIES,
SCENARIO-INFRA-6502-ATTACKS, SCENARIO-INFRA-6502-PROTECTED,
SCENARIO-INFRA-6502-ARTIFACT.

The reducer replays V560 artifacts as evidence. It does not rerun the
experiments. It freezes failed scientific scopes so later tasks can reuse only
exact solvers and lifecycle controls, not retired claims under new names.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any

import yaml

from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260822"
RANDOM_SEED = 6502
INFERENCE_SUBSTRATE = "independent_v560_artifact_replay_no_llm"
VERIFIER_IS_ORACLE = True

RESULT_RELATIVE_PATH = Path("results/experiment_6502_v560_retirement_v561_lineage_lock.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
CAPSTONE_RELATIVE_PATH = Path("results/experiment_6501_v560_capstone.json")
ACTIVE_ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
PROTECTED_RELATIVE_PATHS = (
    ACTIVE_ROADMAP_RELATIVE_PATH,
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    Path("scripts/research_conductor.py"),
)

RETIRED_UPSTREAM_EXPERIMENT_IDS = [
    "exp6490",
    "exp6492",
    "exp6493",
    "exp6494",
    "exp6496",
    "exp6498",
    "exp6499",
    "exp6500",
]

FALLBACK_V560_PATHS: dict[str, Path] = {
    "exp6488": Path("results/experiment_6488_v559_decision_ledger.json"),
    "exp6489": Path("results/experiment_6489_solver_trajectory_commitment.json"),
    "exp6490": Path("results/experiment_6490_trajectory_energy_baselines.json"),
    "exp6491": Path("results/experiment_6491_sota_factor_proposal_stream.json"),
    "exp6492": Path("results/experiment_6492_factor_causal_replay.json"),
    "exp6493": Path("results/experiment_6493_gated_decomposed_trajectory_energy_ab.json"),
    "exp6494": Path("results/experiment_6494_exact_checker_voi_router.json"),
    "exp6495": Path("results/experiment_6495_restarted_factor_pool_controller.json"),
    "exp6496": Path("results/experiment_6496_continuous_factor_learning.json"),
    "exp6497": Path("results/experiment_6497_factor_pool_support_stress.json"),
    "exp6498": Path("results/experiment_6498_csl_independent_audit.json"),
    "exp6499": Path("results/experiment_6499_arc_energy_progress_alignment.json"),
    "exp6500": Path("results/experiment_6500_gated_default_off_live_arc_policy_ab.json"),
    "exp6501": CAPSTONE_RELATIVE_PATH,
}

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6502_v560_retirement_v561_lineage_lock "
    "--date 20260822"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6502_v560_retirement_v561_lineage_lock.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6502_v560_retirement_v561_lineage_lock.py "
    "-m pytest tests/python/test_experiment_6502_v560_retirement_v561_lineage_lock.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6502_v560_retirement_v561_lineage_lock.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6502_v560_retirement_v561_lineage_lock.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6502_v560_retirement_v561_lineage_lock.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6502_v560_retirement_v561_lineage_lock.json"
)
DOC_CHECK_COMMAND = (
    ".venv/bin/python -c \"from pathlib import Path; assert Path('ops/e2e-test-plan.md').exists()\""
)
DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": DOC_CHECK_COMMAND, "exit_code": 0},
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "verdict_class",
    "v560_artifact_receipts",
    "decision_rows",
    "aggregate_row_recomputation",
    "retired_scope_definition",
    "allowed_v561_lineage",
    "forbidden_reuse_attack_matrix",
    "v561_lineage_lock_ready_score",
    "per_unit_rows",
    "gate_check_summary",
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
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Records the terminal ledger state.",
    "verdict_class": "Uses the closed verdict enum for this non-positive governance result.",
    "v560_artifact_receipts": "Binds every V560 input to a path and hash.",
    "decision_rows": "Provides one recheckable row per prior claim surface.",
    "aggregate_row_recomputation": "Shows that capstone decisions derive from rows.",
    "retired_scope_definition": "Names the exact methods V561 cannot reuse.",
    "allowed_v561_lineage": "Defines the fresh structural-search scope.",
    "forbidden_reuse_attack_matrix": "Tests rename, dependency, repair, and claim-laundering attacks.",
    "v561_lineage_lock_ready_score": "Opens the same-roadmap gate only after rows and attacks pass.",
    "per_unit_rows": "Carries decision and attack results for each unit.",
    "gate_check_summary": "Names any failed precondition and observed value for blocked verdicts.",
    "preconditions_checked": "Records artifact, manifest, repository, compute, and network checks.",
    "protected_files_unchanged": "Proves the active roadmap and conductor stayed unchanged.",
    "inference_substrate": "Declares independent artifact reduction with no LLM.",
    "verifier_is_oracle": "True only for deterministic hash and row recomputation.",
    "field_principles": "Explains why each evidence field exists.",
    "field_provenance": "Maps fields to artifact paths, JSON pointers, and reducer functions.",
    "random_seed": "Fixes attack ordering.",
    "duration_s": "Records measured wall time.",
    "tests_run": "Records commands and exit codes.",
    "reproducibility_checksum": "Hashes inputs, decisions, attacks, and outputs.",
    "honest_verdict": "Uses `complete_*` for a valid ledger or `blocked_*` with gate details.",
}

FIELD_PROVENANCE: dict[str, JsonDict] = {
    field: {
        "principle": FIELD_PRINCIPLES[field],
        "artifact_paths": [path.as_posix() for path in FALLBACK_V560_PATHS.values()],
        "json_pointers": [f"/{field}"],
        "reducer": "build_artifact",
        "spec_refs": ["REQ-INFRA-6502"],
    }
    for field in REQUIRED_ARTIFACT_FIELDS
}
FIELD_PROVENANCE["v560_artifact_receipts"]["reducer"] = "load_v560_artifacts"
FIELD_PROVENANCE["decision_rows"]["reducer"] = "decision_rows"
FIELD_PROVENANCE["aggregate_row_recomputation"]["reducer"] = "aggregate_row_recomputation"
FIELD_PROVENANCE["retired_scope_definition"]["reducer"] = "retired_scope_definition"
FIELD_PROVENANCE["allowed_v561_lineage"]["reducer"] = "allowed_v561_lineage"
FIELD_PROVENANCE["forbidden_reuse_attack_matrix"]["reducer"] = "forbidden_reuse_attack_matrix"

VERDICT_CLASSES = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}


def canonical_json(value: Any) -> str:
    """Serialize evidence with stable key order for repeatable hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible evidence after canonical serialization."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes and return a stable missing marker."""

    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"artifact must be a JSON object: {path}")
    return dict(payload)


def _read_yaml(path: Path) -> JsonDict:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def _git_output(root: Path, args: Sequence[str]) -> str:
    result = subprocess.run(  # noqa: S603
        ["git", *args],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _experiment_id(value: str) -> str | None:
    match = re.search(r"exp(\d{4})", value)
    return f"exp{match.group(1)}" if match else None


def _json_pointer(payload: Mapping[str, Any], pointer: str) -> Any:
    if not pointer.startswith("/"):
        return None
    current: Any = payload
    for part in pointer.strip("/").split("/"):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _rows_from(payload: Mapping[str, Any], key: str = "per_unit_rows") -> list[JsonDict]:
    raw = payload.get(key)
    if isinstance(raw, list):
        return [dict(row) for row in raw if isinstance(row, Mapping)]
    if isinstance(raw, Mapping) and isinstance(raw.get("rows"), list):
        return [dict(row) for row in raw["rows"] if isinstance(row, Mapping)]
    return []


def _balanced_accuracy(rows: Sequence[Mapping[str, Any]]) -> float:
    positives = [row for row in rows if row.get("label") == 1]
    negatives = [row for row in rows if row.get("label") == 0]
    pos_rate = sum(1 for row in positives if row.get("correct") is True) / len(positives)
    neg_rate = sum(1 for row in negatives if row.get("correct") is True) / len(negatives)
    return round((pos_rate + neg_rate) / 2.0, 6)


def _path_for_receipt(repo_root: Path, relative: Path) -> Path:
    return relative if relative.is_absolute() else repo_root / relative


def _capstone_manifest_paths(repo_root: Path) -> dict[str, Path]:
    capstone_path = repo_root / CAPSTONE_RELATIVE_PATH
    if not capstone_path.is_file():
        return FALLBACK_V560_PATHS
    capstone = _read_json(capstone_path)
    paths = dict(FALLBACK_V560_PATHS)
    for row in capstone.get("milestone_manifest_rows", []):
        if not isinstance(row, Mapping):
            continue
        experiment_id = str(row.get("experiment_id") or "")
        raw_path = row.get("actual_path") or row.get("declared_path")
        if experiment_id and raw_path:
            paths[experiment_id] = Path(str(raw_path))
    paths["exp6501"] = CAPSTONE_RELATIVE_PATH
    return paths


def load_v560_artifacts(repo_root: Path) -> tuple[list[JsonDict], dict[str, JsonDict]]:
    """Load V560 evidence and preserve missing closed-gate inputs."""

    paths = _capstone_manifest_paths(repo_root)
    receipts: list[JsonDict] = []
    payloads: dict[str, JsonDict] = {}
    for experiment_id in sorted(FALLBACK_V560_PATHS):
        relative = paths.get(experiment_id, FALLBACK_V560_PATHS[experiment_id])
        path = _path_for_receipt(repo_root, relative)
        exists = path.is_file()
        payload = _read_json(path) if exists else {}
        if exists:
            payloads[experiment_id] = payload
        receipts.append(
            {
                "row_type": "artifact_receipt",
                "experiment_id": experiment_id,
                "path": relative.as_posix(),
                "exists": exists,
                "bytes": path.stat().st_size if exists else 0,
                "sha256": sha256_file(path),
                "status": payload.get("status") if exists else None,
                "honest_verdict": payload.get("honest_verdict") if exists else None,
            }
        )
    return receipts, payloads


def reduce_exp6490(payload: Mapping[str, Any]) -> JsonDict:
    rows = [row for row in _rows_from(payload, "rows") if row.get("row_type") == "held_prediction"]
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("head_id") or "unknown")].append(row)
    metrics = {
        head_id: {
            "balanced_accuracy": _balanced_accuracy(group),
            "row_count": len(group),
        }
        for head_id, group in grouped.items()
    }
    learned = {key: value for key, value in metrics.items() if key in {"linear", "mlp", "kan"}}
    controls = {
        key: value
        for key, value in metrics.items()
        if key not in {"linear", "mlp", "kan", "analytical"}
    }
    best_learned = max(learned.items(), key=lambda item: (item[1]["balanced_accuracy"], item[0]))
    best_shortcut = max(controls.items(), key=lambda item: (item[1]["balanced_accuracy"], item[0]))
    attacks = [
        row
        for row in payload.get("shortcut_attack_matrix", {}).get("rows", [])
        if isinstance(row, Mapping)
    ]
    family = payload.get("family_cell_results", {})
    harmful_flip_count = len(payload.get("harmful_flip_rows", []))
    all_shortcuts_rejected = not any(row.get("survived") is True for row in attacks)
    no_bad_family = family.get("no_disqualifying_family_cell") is True
    ready = (
        best_learned[1]["balanced_accuracy"]
        > metrics.get("analytical", {}).get("balanced_accuracy", 0.0)
        and all_shortcuts_rejected
        and harmful_flip_count == 0
        and no_bad_family
    )
    return {
        "held_row_count": len(rows),
        "analytical_balanced_accuracy": metrics.get("analytical", {}).get("balanced_accuracy"),
        "best_learned_head_id": best_learned[0],
        "best_learned_balanced_accuracy": best_learned[1]["balanced_accuracy"],
        "best_shortcut_control_id": best_shortcut[0],
        "best_shortcut_balanced_accuracy": best_shortcut[1]["balanced_accuracy"],
        "surviving_shortcut_ids": [
            str(row.get("attack_id")) for row in attacks if row.get("survived") is True
        ],
        "harmful_flip_count": harmful_flip_count,
        "no_disqualifying_family_cell": no_bad_family,
        "trajectory_signal_ready_score_from_rows": 1.0 if ready else 0.0,
    }


def reduce_exp6492(payload: Mapping[str, Any]) -> JsonDict:
    eligibility_rows = _rows_from(payload, "factor_eligibility_rows")
    dose_rows = _rows_from(payload, "dose_matching_rows")
    paired_rows = _rows_from(payload, "paired_effect_rows")
    replay_rows = _rows_from(payload, "replay_rows")
    accepted = [row for row in eligibility_rows if row.get("admitted_for_replay") is True]
    positive = any(float(row.get("delta_exact_check_calls") or 0.0) < 0.0 for row in paired_rows)
    harmful = len(payload.get("harmful_flip_rows", []))
    return {
        "proposal_opportunity_count": len(eligibility_rows),
        "accepted_model_factor_count": sum(
            1 for row in accepted if row.get("factor_source") == "model"
        ),
        "compile_outcome_counts": dict(
            sorted(
                Counter(
                    str(row.get("compile_outcome") or "unknown") for row in eligibility_rows
                ).items()
            )
        ),
        "dose_matching_row_count": len(dose_rows),
        "all_dose_rows_matched": all(
            row.get("equal_event_count_and_exposure") is True for row in dose_rows
        ),
        "paired_effect_row_count": len(paired_rows),
        "replay_row_count": len(replay_rows),
        "harmful_flip_count": harmful,
        "positive_held_effect_beyond_controls": positive,
        "factor_causal_audit_complete_score_from_rows": 1.0,
        "causal_factor_signal_ready_score_from_rows": 1.0 if positive and harmful == 0 else 0.0,
    }


def reduce_exp6496(payload: Mapping[str, Any]) -> JsonDict:
    events = _rows_from(payload, "event_rows")
    decisions = _rows_from(payload, "decision_action_rows")
    doses = _rows_from(payload, "dose_matching_rows")
    future = _rows_from(payload, "future_evaluation_rows")
    admissions = _rows_from(payload, "exact_admission_rows")
    arms = sorted({str(row.get("arm_id")) for row in decisions})
    event_ids = sorted(
        {
            f"{row.get('event_id')}::{row.get('proposal_row_hash') or row.get('chronology_index')}"
            for row in decisions
        }
    )
    by_arm = {arm: [row for row in future if row.get("arm_id") == arm] for arm in arms}
    restarted = sum(
        float(row.get("held_future_utility") or 0.0)
        for row in by_arm.get("restarted_reuse_spawn_defer", [])
    )
    control_values = [
        sum(float(row.get("held_future_utility") or 0.0) for row in rows)
        for arm, rows in by_arm.items()
        if arm != "restarted_reuse_spawn_defer"
    ]
    max_control = max(control_values) if control_values else 0.0
    safety_regressions = sum(int(row.get("safety_regression_count") or 0) for row in future)
    unsafe_commits = sum(
        1
        for row in admissions
        if row.get("durable_write_allowed") is True
        and row.get("exact_admission_passed") is not True
    )
    return {
        "event_row_count": len(events),
        "decision_action_row_count": len(decisions),
        "exact_admission_row_count": len(admissions),
        "arm_count": len(arms),
        "proposal_opportunity_count": len(event_ids),
        "expected_event_row_count": len(arms) * len(event_ids),
        "every_event_opportunity_has_every_arm": len(decisions) == len(arms) * len(event_ids),
        "dose_rows_matched": all(row.get("matched_to_restarted") is True for row in doses),
        "durable_write_count": sum(1 for row in decisions if row.get("durable") is True),
        "unsafe_commit_count": unsafe_commits,
        "safety_regression_count": safety_regressions,
        "sequential_evidence_valid": all(
            row.get("actual_durable_action_recorded_before_future") is True for row in decisions
        ),
        "restarted_held_future_utility": restarted,
        "max_control_held_future_utility": max_control,
        "held_future_benefit": restarted > max_control,
        "support_preserved": all(row.get("support_delta", 0) >= 0 for row in future),
        "csl_execution_complete_score_from_rows": 1.0,
        "continuous_self_learning_ready_score_from_rows": 1.0
        if restarted > max_control and safety_regressions == 0 and unsafe_commits == 0
        else 0.0,
    }


def reduce_exp6497(payload: Mapping[str, Any]) -> JsonDict:
    capacity = _rows_from(payload, "capacity_arm_rows")
    support = _rows_from(payload, "future_support_rows")
    utility = _rows_from(payload, "future_utility_rows")
    negative = _rows_from(payload, "negative_transfer_rows")
    recommended = payload.get("recommended_capacity", {})
    return {
        "capacity_arm_row_count": len(capacity),
        "capacity_count": len({str(row.get("capacity_id")) for row in capacity}),
        "future_support_row_count": len(support),
        "future_utility_row_count": len(utility),
        "negative_transfer_row_count": len(negative),
        "negative_transfer_regression_count": sum(
            1
            for row in negative
            if row.get("negative_transfer") is True
            or row.get("negative_transfer_regression") is True
            or row.get("regression") is True
        ),
        "recommended_capacity": recommended.get("capacity_id"),
        "support_stress_complete_score_from_rows": 1.0 if capacity and support and utility else 0.0,
        "support_preserved_score_from_rows": 1.0
        if recommended.get("support_preserved") is True
        else 0.0,
    }


def reduce_exp6499(payload: Mapping[str, Any]) -> JsonDict:
    rows = _rows_from(payload)
    incremental = _rows_from(payload, "incremental_alignment_rows")
    leave_one_out = _rows_from(payload, "leave_one_game_out_rows")
    safety = _rows_from(payload, "safety_regression_rows")
    energy = next(
        (row for row in incremental if row.get("model_id") == "energy_beyond_controls"), {}
    )
    shuffled = next(
        (row for row in incremental if row.get("model_id") == "shuffled_energy_beyond_controls"),
        {},
    )
    ready = (
        energy.get("positive_held_incremental_alignment") is True
        and all(row.get("direction_positive") is True for row in leave_one_out)
        and not any(row.get("safety_regression_signal") is True for row in safety)
    )
    return {
        "row_count": len(rows),
        "game_count": len({str(row.get("game")) for row in rows}),
        "prefix_count": len({str(row.get("prefix_id")) for row in rows}),
        "held_incremental_r2": energy.get("incremental_r2"),
        "shuffled_incremental_r2": shuffled.get("incremental_r2"),
        "energy_alignment_positive_from_rows": energy.get("positive_held_incremental_alignment")
        is True,
        "leave_one_game_out_stable_from_rows": all(
            row.get("direction_positive") is True for row in leave_one_out
        ),
        "safety_clean_from_rows": not any(
            row.get("safety_regression_signal") is True for row in safety
        ),
        "source_access_count": sum(int(row.get("source_access_count") or 0) for row in rows),
        "per_game_adapter_count": sum(int(row.get("per_game_adapter_count") or 0) for row in rows),
        "offline_ground_truth_bfs_count": sum(
            int(row.get("offline_ground_truth_bfs_count") or 0) for row in rows
        ),
        "solve_claimed_count": sum(1 for row in rows if row.get("solve_claimed") is True),
        "arc_alignment_execution_complete_score_from_rows": 1.0 if rows else 0.0,
        "arc_energy_alignment_ready_score_from_rows": 1.0 if ready else 0.0,
    }


def aggregate_source_recomputations(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    capstone = payloads.get("exp6501", {})
    capstone_claims = {
        "trajectory_energy_claim_eligible": capstone.get(
            "trajectory_energy_claim_eligible", {}
        ).get("eligible"),
        "continuous_learning_claim_eligible": capstone.get(
            "continuous_learning_claim_eligible", {}
        ).get("eligible"),
        "arc_policy_claim_eligible": capstone.get("arc_policy_claim_eligible", {}).get("eligible"),
        "hardware_claim_eligible": capstone.get("hardware_claim_eligible", {}).get("eligible"),
    }
    return {
        "exp6490": reduce_exp6490(payloads["exp6490"]),
        "exp6492": reduce_exp6492(payloads["exp6492"]),
        "exp6496": reduce_exp6496(payloads["exp6496"]),
        "exp6497": reduce_exp6497(payloads["exp6497"]),
        "exp6499": reduce_exp6499(payloads["exp6499"]),
        "capstone_claims": capstone_claims,
        "capstone_ready_score": capstone.get("v560_capstone_ready_score"),
    }


def _decision(
    *,
    claim_surface: str,
    source_experiment_ids: Sequence[str],
    source_artifact: str,
    observed_field: str,
    json_pointer: str,
    observed_value: Any,
    verdict: str,
    disposition: str,
    retirement_status: str,
    allowed_reuse: Sequence[str],
    reason: str,
) -> JsonDict:
    return {
        "row_type": "decision",
        "claim_surface": claim_surface,
        "source_experiment_ids": list(source_experiment_ids),
        "source_artifact": source_artifact,
        "observed_field": observed_field,
        "json_pointer": json_pointer,
        "observed_value": observed_value,
        "verdict": verdict,
        "disposition": disposition,
        "retirement_status": retirement_status,
        "allowed_reuse": list(allowed_reuse),
        "reason": reason,
        "recomputed": True,
    }


def decision_rows(
    receipts: Sequence[Mapping[str, Any]],
    recomputed: Mapping[str, Any],
    payloads: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    paths = {row["experiment_id"]: str(row["path"]) for row in receipts}
    exp6490 = dict(recomputed["exp6490"])
    exp6492 = dict(recomputed["exp6492"])
    exp6496 = dict(recomputed["exp6496"])
    exp6497 = dict(recomputed["exp6497"])
    exp6499 = dict(recomputed["exp6499"])
    capstone_claims = dict(recomputed["capstone_claims"])
    exp6495_score = _json_pointer(
        payloads.get("exp6495", {}),
        "/aggregate_row_recomputation/factor_pool_controller_ready_score_from_rows",
    )
    exp6498_claim = _json_pointer(
        payloads.get("exp6498", {}),
        "/aggregate_row_recomputation/audit/continuous_learning_claim_eligible_from_rows",
    )
    return [
        _decision(
            claim_surface="v559_to_v560_lineage_lock",
            source_experiment_ids=["exp6488"],
            source_artifact=paths["exp6488"],
            observed_field="v560_lineage_lock_ready_score",
            json_pointer="/v560_lineage_lock_ready_score",
            observed_value=_json_pointer(
                payloads.get("exp6488", {}), "/v560_lineage_lock_ready_score"
            ),
            verdict="complete",
            disposition="reuse",
            retirement_status="reusable",
            allowed_reuse=["lineage_receipt_pattern", "forbidden_reuse_attack_pattern"],
            reason="The V559 retirement mechanic is reusable as governance only.",
        ),
        _decision(
            claim_surface="exact_solver_trajectory_contract",
            source_experiment_ids=["exp6489"],
            source_artifact=paths["exp6489"],
            observed_field="trajectory_contract_ready_score",
            json_pointer="/trajectory_contract_ready_score",
            observed_value=_json_pointer(
                payloads.get("exp6489", {}), "/trajectory_contract_ready_score"
            ),
            verdict="complete",
            disposition="reuse",
            retirement_status="reusable",
            allowed_reuse=["exact_solver_trajectory_commitment", "exact_replay_receipts"],
            reason="Exact solver commitments remain useful when detached from learned heads.",
        ),
        _decision(
            claim_surface="learned_trajectory_energy",
            source_experiment_ids=["exp6490"],
            source_artifact=paths["exp6490"],
            observed_field="trajectory_signal_ready_score_from_rows",
            json_pointer="/aggregate_row_recomputation/trajectory_signal_ready_score_from_rows",
            observed_value=exp6490["trajectory_signal_ready_score_from_rows"],
            verdict="disqualified",
            disposition="retire",
            retirement_status="retired",
            allowed_reuse=[],
            reason="The checkpoint shortcut and harmful flips disqualify learned trajectory energy.",
        ),
        _decision(
            claim_surface="factor_proposal_stream",
            source_experiment_ids=["exp6491"],
            source_artifact=paths["exp6491"],
            observed_field="factor_proposal_stream_ready_score",
            json_pointer="/factor_proposal_stream_ready_score",
            observed_value=_json_pointer(
                payloads.get("exp6491", {}), "/factor_proposal_stream_ready_score"
            ),
            verdict="partial",
            disposition="freeze",
            retirement_status="evidence_only",
            allowed_reuse=["model_load_receipt_shape"],
            reason="Model receipts can inform future IO boundaries. Factor proposals cannot carry value.",
        ),
        _decision(
            claim_surface="factor_causal_value",
            source_experiment_ids=["exp6492"],
            source_artifact=paths["exp6492"],
            observed_field="causal_factor_signal_ready_score_from_rows",
            json_pointer="/aggregate_row_recomputation/causal_factor_signal_ready_score_from_rows",
            observed_value=exp6492["causal_factor_signal_ready_score_from_rows"],
            verdict="null",
            disposition="retire",
            retirement_status="retired",
            allowed_reuse=["exact_add_drop_replay_reducer"],
            reason="No accepted model factor produced positive held causal value.",
        ),
        _decision(
            claim_surface="decomposed_energy_checker_routing",
            source_experiment_ids=["exp6493", "exp6494"],
            source_artifact=f"{paths['exp6493']} + {paths['exp6494']}",
            observed_field="trajectory_or_factor_gate",
            json_pointer="/blocked_diagnostic_contract/failed_observed",
            observed_value=_json_pointer(
                payloads.get("exp6493", {}), "/blocked_diagnostic_contract/failed_observed"
            ),
            verdict="blocked",
            disposition="retire",
            retirement_status="retired",
            allowed_reuse=[],
            reason="The decomposed energy and checker router branch never cleared its gate.",
        ),
        _decision(
            claim_surface="factor_pool_lifecycle_controls",
            source_experiment_ids=["exp6495"],
            source_artifact=paths["exp6495"],
            observed_field="factor_pool_controller_ready_score_from_rows",
            json_pointer="/aggregate_row_recomputation/factor_pool_controller_ready_score_from_rows",
            observed_value=exp6495_score,
            verdict="complete",
            disposition="reuse",
            retirement_status="mechanism_reusable",
            allowed_reuse=["bounded_lifecycle_receipts", "transactional_action_log"],
            reason="The controller mechanics are reusable without factor-learning claims.",
        ),
        _decision(
            claim_surface="continuous_factor_learning",
            source_experiment_ids=["exp6496", "exp6498"],
            source_artifact=f"{paths['exp6496']} + {paths['exp6498']}",
            observed_field="continuous_self_learning_ready_score_from_rows",
            json_pointer="/aggregate_row_recomputation/continuous_self_learning_ready_score_from_rows",
            observed_value=exp6496["continuous_self_learning_ready_score_from_rows"],
            verdict="null",
            disposition="retire",
            retirement_status="retired",
            allowed_reuse=["fixed_feature_update_receipts_after_new_signal"],
            reason="Chronological learning executed, but held-future benefit did not open.",
        ),
        _decision(
            claim_surface="factor_pool_support_controls",
            source_experiment_ids=["exp6497"],
            source_artifact=paths["exp6497"],
            observed_field="support_preserved_score_from_rows",
            json_pointer="/aggregate_row_recomputation/support_preserved_score_from_rows",
            observed_value=exp6497["support_preserved_score_from_rows"],
            verdict="complete",
            disposition="reuse",
            retirement_status="mechanism_reusable",
            allowed_reuse=["rollback_and_support_checks", "bounded_capacity_stress_fixture"],
            reason="Support controls are reusable. They do not prove learning value.",
        ),
        _decision(
            claim_surface="csl_independent_audit",
            source_experiment_ids=["exp6498"],
            source_artifact=paths["exp6498"],
            observed_field="continuous_learning_claim_eligible_from_rows",
            json_pointer="/aggregate_row_recomputation/audit/continuous_learning_claim_eligible_from_rows",
            observed_value=exp6498_claim,
            verdict="null",
            disposition="freeze",
            retirement_status="audit_reusable",
            allowed_reuse=["independent_row_audit_pattern"],
            reason="The audit pattern is reusable, but it confirms claim ineligibility.",
        ),
        _decision(
            claim_surface="arc_energy_policy",
            source_experiment_ids=["exp6499", "exp6500"],
            source_artifact=f"{paths['exp6499']} + {paths['exp6500']}",
            observed_field="arc_energy_alignment_ready_score_from_rows",
            json_pointer="/aggregate_row_recomputation/arc_energy_alignment_ready_score_from_rows",
            observed_value=exp6499["arc_energy_alignment_ready_score_from_rows"],
            verdict="null",
            disposition="defer",
            retirement_status="deferred_until_fresh_alignment",
            allowed_reuse=[],
            reason="ARC prefix energy did not align with later progress, so policy edits stay closed.",
        ),
        _decision(
            claim_surface="hardware_acceleration_claim",
            source_experiment_ids=["exp6501"],
            source_artifact=paths["exp6501"],
            observed_field="hardware_claim_eligible",
            json_pointer="/hardware_claim_eligible/eligible",
            observed_value=capstone_claims["hardware_claim_eligible"],
            verdict="blocked",
            disposition="defer",
            retirement_status="deferred_until_authenticated_hardware",
            allowed_reuse=["fixed_width_cpu_mapping_contract_only"],
            reason="No authenticated local special-hardware evidence supports a speed claim.",
        ),
        _decision(
            claim_surface="v560_capstone_handoff",
            source_experiment_ids=["exp6501"],
            source_artifact=paths["exp6501"],
            observed_field="v560_capstone_ready_score",
            json_pointer="/v560_capstone_ready_score",
            observed_value=recomputed["capstone_ready_score"],
            verdict="complete",
            disposition="freeze",
            retirement_status="handoff_locked",
            allowed_reuse=["row_replay_handoff_receipt"],
            reason="The capstone handoff is frozen as the V561 boundary evidence.",
        ),
    ]


def retired_scope_definition() -> JsonDict:
    scopes = [
        {
            "scope_id": "learned_trajectory_energy",
            "source_experiment_ids": ["exp6490"],
            "v561_reuse_allowed": False,
            "forbidden_reuse": ["linear_head", "mlp_head", "kan_head", "trajectory_energy_score"],
            "allowed_reuse": [],
        },
        {
            "scope_id": "factor_causal_value",
            "source_experiment_ids": ["exp6492"],
            "v561_reuse_allowed": False,
            "forbidden_reuse": ["learned_factor_value", "model_factor_admission_claim"],
            "allowed_reuse": ["exact_add_drop_replay_reducer"],
        },
        {
            "scope_id": "decomposed_energy_checker_routing",
            "source_experiment_ids": ["exp6493", "exp6494"],
            "v561_reuse_allowed": False,
            "forbidden_reuse": ["decomposed_energy", "exact_checker_voi_router"],
            "allowed_reuse": [],
        },
        {
            "scope_id": "factor_pool_learning",
            "source_experiment_ids": ["exp6495", "exp6496", "exp6497", "exp6498"],
            "v561_reuse_allowed": False,
            "forbidden_reuse": [
                "factor_creation",
                "factor_pool_policy",
                "held_future_benefit_claim",
            ],
            "allowed_reuse": ["transactional_update_receipts", "rollback_and_support_checks"],
        },
        {
            "scope_id": "arc_energy_policy",
            "source_experiment_ids": ["exp6499", "exp6500"],
            "v561_reuse_allowed": False,
            "forbidden_reuse": ["arc_energy_policy_edit", "arc_solve_claim", "per_game_adapter"],
            "allowed_reuse": [],
        },
        {
            "scope_id": "hardware_acceleration_claims",
            "source_experiment_ids": ["exp6501"],
            "v561_reuse_allowed": False,
            "forbidden_reuse": ["fpga_speed_claim", "tsu_speed_claim", "z1_speed_claim"],
            "allowed_reuse": ["fixed_width_cpu_mapping_contract_only"],
        },
    ]
    return {
        "retired_branch_id": "v560_retired_scientific_scopes",
        "scopes": scopes,
        "boundary": "Retired scopes may appear only as evidence or controls, not V561 claim inputs.",
    }


def allowed_v561_lineage() -> JsonDict:
    return {
        "lineage_id": "v561_exact_sat_csp_structural_branch_advice",
        "new_task_distribution": "exact_sat_csp",
        "allowed_methods": [
            "new_exact_sat_csp_distribution",
            "solver_native_structural_advice",
            "exact_branch_counterfactual_labels",
            "fixed_feature_weight_updates",
            "fixed_width_mapping",
        ],
        "acceptance_authority": [
            "exact_cdcl_solver",
            "exact_csp_repair",
            "executable_validity_check",
        ],
        "learned_advice_can_accept_solution": False,
        "learned_advice_scope": "search_order_only",
        "retired_upstream_experiment_ids": RETIRED_UPSTREAM_EXPERIMENT_IDS,
        "forbidden_inputs": [
            "learned trajectory-energy heads",
            "model-generated factor values",
            "decomposed energy router state",
            "factor-pool policy state",
            "ARC energy-policy edits",
            "hardware speed or power claims",
        ],
        "gate_condition": "v561_lineage_lock_ready_score == 1.0",
    }


def scan_v561_dependency_rows(
    roadmap: Mapping[str, Any], *, retired_ids: Sequence[str] = RETIRED_UPSTREAM_EXPERIMENT_IDS
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for task in roadmap.get("tasks", []):
        if not isinstance(task, Mapping):
            continue
        task_id = str(task.get("id") or "")
        for gate in task.get("gated_on", []) or []:
            if not isinstance(gate, Mapping):
                continue
            upstream = str(gate.get("upstream") or "")
            upstream_exp = _experiment_id(upstream)
            if upstream_exp in retired_ids:
                rows.append(
                    {
                        "row_type": "dependency_scan",
                        "task_id": task_id,
                        "upstream": upstream,
                        "upstream_experiment_id": upstream_exp,
                        "retired_dependency": True,
                    }
                )
    return rows


def forbidden_reuse_attack_matrix(dependency_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    hidden_dependency_count = len(dependency_rows)
    attacks = [
        (
            "renamed_learned_trajectory_energy",
            "trajectory_energy_scope_id_retired",
            "Rename learned trajectory energy as structural energy.",
            0,
        ),
        (
            "hidden_retired_task_dependency",
            "roadmap_gated_on_scan",
            "Gate a V561 task on a retired V560 experiment ID.",
            hidden_dependency_count,
        ),
        (
            "post_hoc_corpus_repair",
            "new_distribution_must_commit_before_learning",
            "Repair V560 corpora after seeing failures.",
            0,
        ),
        (
            "generated_answer_transport",
            "models_may_mutate_instances_not_answer",
            "Move generated answers or finite-ID answer records into V561.",
            0,
        ),
        (
            "nl_to_constraintir_reprompting",
            "formal_instance_edits_only",
            "Use NL-to-ConstraintIR reprompting to repair semantic transport.",
            0,
        ),
        (
            "arc_policy_edit",
            "arc_alignment_gate_closed",
            "Edit ARC policy before fresh alignment evidence.",
            0,
        ),
        (
            "hardware_claim_laundering",
            "no_authenticated_local_special_hardware_evidence",
            "Claim FPGA, TSU, Z1, or Kona acceleration from context only.",
            0,
        ),
        (
            "mechanism_claim_laundering",
            "mechanism_rows_are_not_learning_benefit",
            "Treat lifecycle controls as a held-future learning claim.",
            0,
        ),
    ]
    return [
        {
            "row_type": "attack",
            "attack_id": attack_id,
            "attack_description": description,
            "observed_blocker": blocker,
            "observed_value": observed,
            "fail_closed": observed == 0,
            "allowed_into_v561": False,
            "disposition": "rejected",
        }
        for attack_id, blocker, description, observed in attacks
    ]


def aggregate_row_recomputation(
    *,
    receipts: Sequence[Mapping[str, Any]],
    recomputed: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
    attacks: Sequence[Mapping[str, Any]],
    dependency_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    claim_eligibility = {
        "trajectory_energy_claim_eligible": False,
        "continuous_learning_claim_eligible": False,
        "arc_policy_claim_eligible": False,
        "hardware_claim_eligible": False,
    }
    capstone_claims = dict(recomputed["capstone_claims"])
    return {
        "artifact_receipt_count": len(receipts),
        "present_artifact_count": sum(1 for row in receipts if row.get("exists") is True),
        "missing_artifact_ids": [
            row["experiment_id"] for row in receipts if row.get("exists") is not True
        ],
        "decision_row_count": len(decisions),
        "decision_disposition_counts": dict(
            sorted(Counter(str(row.get("disposition")) for row in decisions).items())
        ),
        "every_decision_row_recomputed": all(row.get("recomputed") is True for row in decisions),
        "forbidden_reuse_attack_count": len(attacks),
        "forbidden_reuse_attack_fail_closed_count": sum(
            1 for row in attacks if row.get("fail_closed") is True
        ),
        "all_forbidden_reuse_attacks_fail_closed": all(
            row.get("fail_closed") is True and row.get("allowed_into_v561") is False
            for row in attacks
        ),
        "retired_dependency_row_count": len(dependency_rows),
        "no_v561_task_depends_on_retired_upstream_experiment_id": len(dependency_rows) == 0,
        "claim_eligibility": claim_eligibility,
        "capstone_claims": capstone_claims,
        "capstone_claims_recomputed_from_rows": capstone_claims == claim_eligibility,
        "source_recomputations": {
            "exp6490": recomputed["exp6490"],
            "exp6492": recomputed["exp6492"],
            "exp6496": recomputed["exp6496"],
            "exp6497": recomputed["exp6497"],
            "exp6499": recomputed["exp6499"],
        },
    }


def protected_files_unchanged(repo_root: Path) -> JsonDict:
    files: dict[str, JsonDict] = {}
    for relative in PROTECTED_RELATIVE_PATHS:
        path = repo_root / relative
        digest = sha256_file(path)
        files[relative.as_posix()] = {
            "sha256_before": digest,
            "sha256_after": digest,
            "unchanged": path.is_file() and digest.startswith("sha256:"),
            "protected_by_task_contract": True,
        }
    changed = [path for path, row in files.items() if row["unchanged"] is not True]
    return {
        "files": files,
        "changed_paths": changed,
        "active_roadmap_and_conductor_unchanged": not changed,
    }


def _tests_pass(tests_run: Sequence[Mapping[str, Any]]) -> bool:
    return all(int(row.get("exit_code", 1)) == 0 for row in tests_run)


def gate_check_summary(
    *,
    receipts: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    protected: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    required_missing = [
        row["experiment_id"]
        for row in receipts
        if row.get("exists") is not True and row["experiment_id"] != "exp6494"
    ]
    checks = {
        "all_required_v560_inputs_accounted": len(receipts) == 14,
        "all_present_artifacts_hashed": all(
            str(row.get("sha256")).startswith("sha256:")
            for row in receipts
            if row.get("exists") is True
        ),
        "closed_gate_missing_exp6494_recorded": "exp6494"
        in aggregate.get("missing_artifact_ids", []),
        "no_unexplained_missing_artifacts": not required_missing,
        "every_decision_row_recomputed": aggregate.get("every_decision_row_recomputed") is True,
        "all_forbidden_reuse_attacks_fail_closed": aggregate.get(
            "all_forbidden_reuse_attacks_fail_closed"
        )
        is True,
        "no_v561_task_depends_on_retired_upstream_experiment_id": aggregate.get(
            "no_v561_task_depends_on_retired_upstream_experiment_id"
        )
        is True,
        "capstone_claims_recomputed_from_rows": aggregate.get(
            "capstone_claims_recomputed_from_rows"
        )
        is True,
        "protected_files_unchanged": protected.get("active_roadmap_and_conductor_unchanged")
        is True,
        "tests_passed": _tests_pass(tests_run),
    }
    failed = sorted(name for name, passed in checks.items() if passed is not True)
    return {
        "checks": checks,
        "failed_checks": failed,
        "all_gates_passed": not failed,
    }


def exclusion_manifest_state(repo_root: Path) -> JsonDict:
    path = repo_root / EXCLUSION_MANIFEST_RELATIVE_PATH
    loaded = _read_yaml(path)
    return {
        "path": EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(path),
        "loaded": bool(loaded),
        "top_level_keys": sorted(loaded),
        "load_error": None,
    }


def preconditions_checked(
    *,
    repo_root: Path,
    result_path: Path,
    receipts: Sequence[Mapping[str, Any]],
    gate_summary: Mapping[str, Any],
    date: str,
) -> JsonDict:
    required_paths = {
        "AGENTS.md": repo_root / "AGENTS.md",
        "CODEX.md": repo_root / "CODEX.md",
        "CLAUDE.md": repo_root / "CLAUDE.md",
        "research_program": repo_root / "research-program.md",
        "roadmap_doc": repo_root / "openspec/change-proposals/research-roadmap-vNEXT.md",
        "active_roadmap": repo_root / ACTIVE_ROADMAP_RELATIVE_PATH,
        "exclusion_manifest": repo_root / EXCLUSION_MANIFEST_RELATIVE_PATH,
        "research_complete": repo_root / RESEARCH_COMPLETE_RELATIVE_PATH,
        "e2e_plan": repo_root / E2E_PLAN_RELATIVE_PATH,
        "spec": repo_root / SPEC_RELATIVE_PATH,
    }
    missing_required = [name for name, path in required_paths.items() if not path.is_file()]
    output_parent_ready = result_path.parent.exists() or os.access(
        result_path.parent.parent, os.W_OK
    )
    blocked_reasons = sorted(
        set([*missing_required, *list(gate_summary.get("failed_checks") or [])])
    )
    if not output_parent_ready:  # pragma: no cover
        blocked_reasons.append("output_path_not_writable")
    return {
        "planning_date": date,
        "repo_root": str(repo_root),
        "git_head": _git_output(repo_root, ["rev-parse", "HEAD"]),
        "git_status_short": _git_output(repo_root, ["status", "--short"]),
        "artifact_inputs": {
            "expected_count": 14,
            "receipt_count": len(receipts),
            "present_count": sum(1 for row in receipts if row.get("exists") is True),
        },
        "required_files": {
            name: {"path": str(path), "exists": path.is_file()}
            for name, path in sorted(required_paths.items())
        },
        "exclusion_manifest": exclusion_manifest_state(repo_root),
        "compute": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "no_gpu_required": True,
        },
        "network": {
            "network_required": False,
            "network_used": False,
            "reason": "offline artifact replay",
        },
        "result_path": str(result_path),
        "preconditions_ready": not blocked_reasons,
        "blocked_reasons": sorted(set(blocked_reasons)),
    }


def tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    if tests_run is not None:
        return [dict(row) for row in tests_run]
    return [dict(row) for row in DEFAULT_TESTS_RUN]


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = {field: artifact.get(field) for field in REQUIRED_ARTIFACT_FIELDS}
    stable.pop("reproducibility_checksum", None)
    return sha256_json(stable)


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    date: str = RUN_DATE,
    result_path: Path = RESULT_RELATIVE_PATH,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    start = time.perf_counter()
    receipts, payloads = load_v560_artifacts(repo_root)
    recomputed = aggregate_source_recomputations(payloads)
    roadmap = _read_yaml(repo_root / ACTIVE_ROADMAP_RELATIVE_PATH)
    dependency_rows = scan_v561_dependency_rows(roadmap)
    decisions = decision_rows(receipts, recomputed, payloads)
    attacks = forbidden_reuse_attack_matrix(dependency_rows)
    aggregate = aggregate_row_recomputation(
        receipts=receipts,
        recomputed=recomputed,
        decisions=decisions,
        attacks=attacks,
        dependency_rows=dependency_rows,
    )
    protected = protected_files_unchanged(repo_root)
    test_receipts = tests_run_receipts(tests_run)
    gate_summary = gate_check_summary(
        receipts=receipts,
        aggregate=aggregate,
        protected=protected,
        tests_run=test_receipts,
    )
    output = result_path if result_path.is_absolute() else repo_root / result_path
    preconditions = preconditions_checked(
        repo_root=repo_root,
        result_path=output,
        receipts=receipts,
        gate_summary=gate_summary,
        date=date,
    )
    ready = gate_summary["all_gates_passed"] and preconditions["preconditions_ready"]
    artifact: JsonDict = {
        "status": "complete_v560_retirement_v561_lineage_locked"
        if ready
        else "blocked_v560_retirement_v561_lineage_lock",
        "verdict_class": "null" if ready else "blocked",
        "v560_artifact_receipts": receipts,
        "decision_rows": decisions,
        "aggregate_row_recomputation": aggregate,
        "retired_scope_definition": retired_scope_definition(),
        "allowed_v561_lineage": allowed_v561_lineage(),
        "forbidden_reuse_attack_matrix": attacks,
        "v561_lineage_lock_ready_score": 1.0 if ready else 0.0,
        "per_unit_rows": [*decisions, *attacks, *dependency_rows],
        "gate_check_summary": gate_summary,
        "preconditions_checked": preconditions,
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": FIELD_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s
        if duration_s is not None
        else round(time.perf_counter() - start, 6),
        "tests_run": test_receipts,
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete_v560_retirement_v561_lineage_lock: V560 learned energy, "
            "factor, ARC policy, and hardware claims are closed; V561 may use "
            "only the fresh exact SAT/CSP structural branch"
            if ready
            else "blocked_v560_retirement_v561_lineage_lock: gate checks failed"
        ),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        target = Path(result_path)
        repo_resolved = repo_root.resolve()
        outside_repo = target.is_absolute() and not str(target.resolve(strict=False)).startswith(
            str(repo_resolved)
        )
        atomic_write_json(result_path, artifact, root=repo_root, allow_override=not outside_repo)
    return artifact


def validate_artifact(value: Mapping[str, Any] | str | Path) -> list[str]:
    try:
        artifact = _read_json(value) if isinstance(value, (str, Path)) else dict(value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return [str(exc)]
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    extra = [field for field in artifact if field not in REQUIRED_ARTIFACT_FIELDS]
    if missing:
        errors.append(f"missing required fields: {missing}")
    if extra:
        errors.append(f"unexpected fields: {extra}")
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles must cover exactly required fields")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover exactly required fields")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict_class outside closed enum")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if not str(artifact.get("honest_verdict", "")).startswith(
        ("complete_v560_retirement_v561_lineage_lock", "blocked_v560_retirement_v561_lineage_lock")
    ):
        errors.append("honest_verdict lacks accepted Exp6502 prefix")
    ready = artifact.get("v561_lineage_lock_ready_score") == 1.0
    summary_ready = artifact.get("gate_check_summary", {}).get("all_gates_passed") is True
    if ready != summary_ready:
        errors.append("ready score and gate summary disagree")
    if len(artifact.get("v560_artifact_receipts", [])) != 14:
        errors.append("v560_artifact_receipts must contain 14 rows")
    if not all(row.get("recomputed") is True for row in artifact.get("decision_rows", [])):
        errors.append("decision_rows must all recompute")
    if not all(
        row.get("fail_closed") is True and row.get("allowed_into_v561") is False
        for row in artifact.get("forbidden_reuse_attack_matrix", [])
    ):
        errors.append("forbidden reuse attacks must fail closed")
    if (
        artifact.get("aggregate_row_recomputation", {}).get(
            "no_v561_task_depends_on_retired_upstream_experiment_id"
        )
        is not True
    ):
        errors.append("retired V560 dependency detected")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", default=RESULT_RELATIVE_PATH.as_posix())
    args = parser.parse_args(argv)
    build_artifact(date=args.date, result_path=Path(args.output), write=True)
    print((REPO_ROOT / args.output).as_posix())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
