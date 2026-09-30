"""Reuse qualified authorities and source bytes without granting science credit.

REQ-REPORT-7903-V686. Missing design authority remains an outside prerequisite;
private oracle fixtures demonstrate mechanics without repairing that absence.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

import yaml

from carnot.reporting import source_boundary_7892 as boundary
from carnot.reporting import v685_authority_lifecycle as lifecycle
from carnot.reporting.current_work_receipt import (
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.verify import source_projection

MILESTONE = "2026.09.686"
SOURCE_SHA256 = "sha256:528a2243693f2543f43bbf4f0b2ce789f2b01ccf14d795260c7e15029230ca14"
FIXTURE_DIGEST = "1c2b048911268230ee424964a8c85281ea6a901ce59199e56e84de62438cddb9"


def operand(
    path: Path, field: str, expected: Any, observed: Any, upstream: str = "exp7892-source-boundary"
) -> dict[str, Any]:
    """Name the failed input so a missing value differs from a wrong value."""
    return {
        "upstream_id": upstream,
        "path": str(path),
        "hash": sha256_file(path) if path.is_file() else None,
        "artifact_field": field,
        "op": "==",
        "expected": expected,
        "observed": observed,
    }


def assess(design: Path, staged: Path, active: Path, snapshots: Path) -> dict[str, Any]:
    """Require real design authority before asking the versioned lifecycle to bind it."""
    text = design.read_text() if design.is_file() else ""
    required = {
        "task_table": "## Exact task contract",
        "machine_contract": "V686_TASK_CONTRACT_START",
        "canonical_tasks_sha256": r"Canonical full-task SHA-256: `([0-9a-f]{64})`",
    }
    failures = [
        operand(design, key, "present in design", None, "V686_authority")
        for key, pattern in required.items()
        if re.search(pattern, text) is None
    ]
    if not failures:
        try:
            result = lifecycle.assess_authorities(
                design, staged, active, snapshots, milestone=MILESTONE, first_id=7903
            )
            _, machine = lifecycle.parse_design(text, milestone=MILESTONE)
            if lifecycle.tasks_digest(machine) != result["canonical_tasks_sha256"]:
                result["gate_check_summary"].append(
                    operand(
                        design,
                        "design_tasks_sha256",
                        result["canonical_tasks_sha256"],
                        lifecycle.tasks_digest(machine),
                        "V686_authority",
                    )
                )
                result["activated"] = False
            for failure in result["gate_check_summary"]:
                failure["upstream_id"] = "V686_authority"
            return result
        except (ValueError, IndexError, KeyError) as error:
            failures.append(
                operand(
                    design,
                    "design_schema",
                    "complete parseable contract",
                    str(error),
                    "V686_authority",
                )
            )
    observed = {}
    values = {}
    for role, path in (("design", design), ("staged", staged), ("active", active)):
        raw = path.read_bytes() if path.is_file() else None
        observed[role] = lifecycle._snapshot(path, raw, snapshots, role)
        values[role] = yaml.safe_load(raw) if raw is not None and role != "design" else None
    active_value = values["active"] or {}
    tasks = active_value.get("tasks", [])
    rows = [
        {
            "family": task.get("id"),
            "unit_id": task.get("id"),
            "order": i,
            "arm": "unconfirmed_active_contract",
            "seed": None,
            "status": "completed",
            "matched": False,
            "checks": {"design_contract_present": False},
            "absolute_metric": 0,
            "raw_numerator": 0,
            "raw_denominator": 1,
            "censored": False,
            "excluded": False,
            "effective_independent_groups": 0,
        }
        for i, task in enumerate(tasks[:12], 1)
    ]
    while len(rows) < 12:
        rows.append(
            {
                "unit_id": f"unconfirmed-{len(rows) + 1}",
                "order": len(rows) + 1,
                "status": "unstarted",
                "matched": False,
                "checks": {"design_contract_present": False},
                "absolute_metric": 0,
            }
        )
    return {
        "activated": False,
        "planning_matched": False,
        "canonical_tasks_sha256": None,
        "active_tasks_sha256": lifecycle.tasks_digest(tasks),
        "staged_tasks_sha256": None,
        "contract_rows": rows,
        "authority_snapshots": observed,
        "gate_check_summary": failures,
    }


def _source_custody(receipt: Path, expected_hash: str) -> dict[str, Any]:
    """Authenticate the frozen receipt and raw joins while preserving every excluded row."""
    hashes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    def check(path: Path, field: str, expected: Any, observed: Any) -> None:
        if observed != expected:
            failures.append(operand(path, field, expected, observed))

    def authenticate(path: Path, expected: str, role: str) -> bool:
        observed = sha256_file(path) if path.is_file() else None
        hashes.append(
            {
                "path": str(path),
                "sha256": observed,
                "expected_sha256": expected,
                "role": role,
                "exposure": "exposed_development",
            }
        )
        check(path, "sha256", expected, observed)
        return observed == expected

    if authenticate(receipt, expected_hash, "qualified_source_receipt"):
        value = json.loads(receipt.read_text())
        for field, expected in {
            **boundary.IDENTITY,
            "source_boundary_ready_score": 1,
            "flagged_adversarial": False,
            "verdict_class": "circular_positive",
        }.items():
            check(receipt, field, expected, value.get(field))
        manifest_path = Path(value["cohort_manifest_path"])
        if authenticate(manifest_path, value["cohort_manifest_sha256"], "cohort_manifest"):
            manifest = json.loads(manifest_path.read_text())
            rows = manifest["rows"]
            check(manifest_path, "rows", value["rows"], rows)
            check(
                manifest_path,
                "budget",
                {
                    "intended": 640,
                    "eligible": 604,
                    "started": 640,
                    "completed": 604,
                    "failed": 0,
                    "censored": 0,
                    "excluded": 36,
                    "independent": 640,
                },
                boundary.budget(rows, 640),
            )
            check(
                manifest_path,
                "role_counts",
                boundary.EIGHT_ROLES,
                dict(Counter(row["role"] for row in rows)),
            )
            for kind in ("public_shards", "evaluator_shards", "feature_shards"):
                check(manifest_path, kind, value[kind], manifest.get(kind))
                check(receipt, f"{kind}.count", 2, len(value[kind]))
                joined = []
                for reference in value[kind]:
                    path = Path(reference["path"])
                    if authenticate(path, reference["sha256"], kind):
                        check(path, "below_50_MiB", True, path.stat().st_size < 50 * 1024**2)
                        joined.extend(source_projection.read_jsonl(path))
                check(
                    receipt,
                    f"{kind}.family_ids",
                    [row["family_id"] for row in rows],
                    [row.get("family_id") for row in joined],
                )
                for i, item in enumerate(joined):
                    if kind == "public_shards":
                        check(
                            receipt,
                            f"public[{i}].keys",
                            sorted(source_projection.PUBLIC_KEYS),
                            sorted(item),
                        )
                        check(
                            receipt,
                            f"public[{i}].source_sha256",
                            rows[i]["source_sha256"],
                            "sha256:"
                            + hashlib.sha256(bytes.fromhex(item["source_bytes"])).hexdigest(),
                        )
                    elif kind == "evaluator_shards":
                        check(receipt, f"evaluator[{i}].role", rows[i]["role"], item["role"])
                        check(
                            receipt, f"evaluator[{i}].label_scope", "response", item["label_scope"]
                        )
                    else:
                        check(
                            receipt,
                            f"features[{i}].hash",
                            rows[i]["feature_hash"],
                            item["feature_hash"],
                        )
                        check(
                            receipt,
                            f"features[{i}].excluded",
                            rows[i]["status"] == "excluded",
                            bool(item["abstention"]),
                        )
        check(receipt, "validation_receipts.nonempty", True, bool(value["validation_receipts"]))
        for row in value["validation_receipts"]:
            check(receipt, f"validation_receipts.{row['name']}.passed", True, row["passed"])
            authenticate(Path(row["log_path"]), row["log_sha256"], "qualified_source_validation")
    return {
        "ready": not failures,
        "rows": rows,
        "hashes": hashes,
        "budget": boundary.budget(rows, 640),
        "gate_check_summary": failures,
    }


def source_custody(receipt: Path, expected_hash: str) -> dict[str, Any]:
    """Return a terminal operand failure when an authenticated source schema is malformed."""
    try:
        return _source_custody(receipt, expected_hash)
    except (ValueError, KeyError, TypeError, IndexError) as error:
        return {
            "ready": False,
            "rows": [],
            "hashes": [],
            "budget": boundary.budget([], 640),
            "gate_check_summary": [
                operand(
                    receipt,
                    "source_schema",
                    "complete parseable receipt",
                    f"{type(error).__name__}:{error}",
                )
            ],
        }


def mutations(design: Path, active: Path, source: Path, private: Path) -> list[dict[str, Any]]:
    """Challenge private copies; passing oracle mechanics never changes live authority."""
    started = time.monotonic()
    baseline = yaml.safe_load(active.read_text())
    names = (
        "consumed_staging",
        "later_staging",
        "stale_active",
        "prompt_edit",
        "prior_failure_deletion",
        "wrong_model",
        "wrong_class",
        "order",
        "gate",
        "phase",
        "deliverable",
        "source_hash_drift",
    )
    rows = []
    for index, name in enumerate(names, 1):
        directory = private / name
        directory.mkdir(parents=True, exist_ok=True)
        value = deepcopy(baseline)
        staged_path, active_path = directory / "stage.yaml", directory / "active.yaml"
        expected = name in {"consumed_staging", "later_staging"}
        if name == "source_hash_drift":
            observed = source_custody(source, "sha256:drift")["ready"]
        else:
            if name == "later_staging":
                staged_path.write_text("milestone: 2026.10.687\ntasks: []\n")
            elif name == "stale_active":
                value["milestone"] = "2026.09.685"
            elif name == "order":
                value["tasks"].reverse()
            elif name == "prior_failure_deletion":
                value["tasks"][0]["prior_failures"].pop()
            elif name not in {"consumed_staging", "later_staging"}:
                key = {
                    "prompt_edit": "prompt",
                    "wrong_model": "MODEL_SPECS",
                    "wrong_class": "inference_substrate_class",
                    "gate": "gated_on",
                }.get(name, name)
                value["tasks"][0][key] = "changed"
            active_path.write_text(yaml.safe_dump(value, sort_keys=False))
            observed = assess(design, staged_path, active_path, directory / "snapshots")[
                "activated"
            ]
        rows.append(
            {
                "unit_id": name,
                "arm": "private_oracle_mutation",
                "status": "completed",
                "expected_activation": expected,
                "observed_activation": observed,
                "passed": observed is expected,
                "claim_scope": "circular_positive",
            }
        )
        print(
            f"[exp7903] phase=mutations elapsed_s={time.monotonic() - started:.3f} completed_units={index}/12",
            flush=True,
        )
    return rows


def method_freeze(root: Path) -> dict[str, Any]:
    """Freeze design methods without claiming that future experiments have executed."""
    import gzip

    tasks = yaml.safe_load(
        gzip.decompress((root / "tests/fixtures/v686/active.yaml.gz").read_bytes())
    )["tasks"]
    return {
        "authority_status": "active_only_unconfirmed_until_design_binding",
        "source_sha256": SOURCE_SHA256,
        "source_roles": boundary.EIGHT_ROLES,
        "training": {
            "seeds": [67801, 67802, 67803],
            "arms": 9,
            "heads": 27,
            "width": 16,
            "max_parameters": 4096,
            "learning_rate": 0.01,
            "epochs": 16,
            "features": 132,
            "complete_static_predicates": 16,
            "deadline_s": 3000,
            "max_windows": 128,
            "max_answer_units": 16,
            "kl_tolerance": 0.01,
            "alternate_ce_limit": 0.70,
            "dual_step": 0.01,
            "dual_clip": [0, 10],
            "temperature_grid": {"count": 17, "range": [0.25, 4], "role": "tune"},
        },
        "decisions": {
            "costs": [5, 1, 0.25],
            "cost_gain_min": 0.02,
            "bootstrap_draws": 10000,
            "bootstrap_unit": "source_cluster",
            "seed_reduction": "mean_within_family",
            "multiple_tests": "Holm_six_Brier_cost",
            "coverage_min": 0.20,
            "equal_coverage_target": 0.50,
            "equal_coverage_tolerance": 0.05,
        },
        "qwen": {
            "model": "unsloth/Qwen3.8-27B-GGUF",
            "families": 48,
            "calls_per_family": 4,
            "tokens_per_call": 128,
            "total_tokens": 24576,
            "n_ctx": 8192,
            "temperature": 0,
            "seed": 67801,
            "load_timeout_s": 300,
            "call_timeout_s": 120,
            "stop_starting_s": 2400,
            "deadline_s": 3000,
            "minimum_complete_families": 32,
            "claim": "sensitivity_only_without_sentence_labels",
        },
        "learning": {
            "blocks": 8,
            "update_slots_per_block": 12,
            "admission_slots_per_block": 8,
            "feedback_delay_blocks": 1,
            "max_admissions_per_block": 1,
            "max_admissions": 8,
            "admission_brier_gain_min": 0.01,
            "crash_resume_after_block": 4,
            "cost_gain_min": 0.02,
            "retention_cost_increase_ci95_upper_max": 0.01,
            "controls": ["frozen", "complete_static", "no_write", "shuffled_released_past"],
        },
        "delayed_aci": {
            "tau": [1, 4, 16],
            "gamma": 0.01,
            "initial_alpha": 0.10,
            "target_coverage": 0.90,
            "nonconformity": "1-p_y",
            "update": "alpha[t+tau]=issued_alpha[t]+.01*(.10-miss[t])",
            "rolling_released_window": 32,
            "bootstrap_draws": 10000,
            "bootstrap_unit": "contiguous_16_event_block",
            "minimum_windows": 4,
            "minimum_past_labels_for_memory": 64,
            "local_error_gain_min": 0.02,
            "set_size_increase_max": 0.10,
            "multiple_tests": "Holm_six_local_error",
            "alpha_below_zero": "full_binary_set",
            "alpha_above_one": "empty_set",
            "trajectory": "fixed_predictions_and_bank",
            "pending_tails": "preserve",
        },
        "service_cost": {
            "denominators": ["processed", "accepted", "caught_error"],
            "include": ["host", "transport", "updates", "abstention", "false_accepts"],
        },
        "group_tests": {
            "science_independent": False,
            "exposure": "exposed_development",
            "oracle_distinct_gap_closed": False,
        },
        "future_validation_scopes": [
            {
                "task_id": task["id"],
                "deliverable": task["deliverable"],
                "MODEL_SPECS": task["MODEL_SPECS"],
                "inference_substrate_class": task["inference_substrate_class"],
                "gated_on": task.get("gated_on", []),
                "prompt_sha256": canonical_hash(task["prompt"]),
                "frozen_validation_and_methods": task["prompt"],
                "prospective_code_closure_required": True,
            }
            for task in tasks
        ],
        "literature_adoption_decisions": [
            {
                "source": "https://arxiv.org/abs/2609.07251",
                "decision": "adapt",
                "task": "exp7910",
                "method": "issued-state delayed ACI with past-only memory and set-size controls",
                "limit": "binary empirical adaptation; no forecasting theorem",
                "measured_benefit": None,
            },
            {
                "source": "https://arxiv.org/abs/2609.35028",
                "decision": "adapt",
                "task": "exp7912",
                "method": "processed, accepted and caught-error cost denominators",
                "limit": "no LLM-judge truth oracle",
                "measured_benefit": None,
            },
            {
                "source": "https://arxiv.org/abs/2604.23987",
                "decision": "adapt",
                "task": "exp7909/7910",
                "method": "separate calibration and constraint feedback",
                "measured_benefit": None,
            },
            {
                "source": "https://arxiv.org/abs/2605.12306",
                "decision": "defer",
                "method": "sparse updates and retention; no unchanged KAN sweep",
                "measured_benefit": None,
            },
            {
                "source": "https://arxiv.org/abs/2602.15985",
                "decision": "adapt",
                "task": "exp7912/7913",
                "method": "include preprocessing, host work and transport",
                "measured_benefit": None,
            },
            {
                "source": "https://extropic.ai/writing/z1t",
                "decision": "defer",
                "method": "vendor estimates cannot establish local speedup",
                "measured_benefit": None,
            },
            {
                "source": "https://arxiv.org/abs/2609.19515",
                "decision": "defer",
                "method": "repair after useful verification",
                "measured_benefit": None,
            },
            {
                "source": "https://arxiv.org/abs/2609.35362",
                "decision": "defer",
                "method": "distillation requires different generator scope",
                "measured_benefit": None,
            },
        ],
        "primary_rechecks": {
            "date": "20260930",
            "request_count": 2,
            "bounded_requests": True,
            "sources": ["https://arxiv.org/abs/2609.07251", "https://arxiv.org/abs/2609.35028"],
        },
    }


def verdict(activated: bool, source_ready: bool, checks_passed: bool) -> tuple[str, str, int]:
    """Keep outside incompleteness terminal and owned validation failures disqualified."""
    if not checks_passed:
        return "complete_disqualified_required_validation", "disqualified", 0
    if not activated:
        return "complete_blocked_authority", "blocked", 0
    if not source_ready:
        return "complete_blocked_source_custody", "blocked", 0
    return "complete_circular_positive_contract_methods", "circular_positive", 1


def contract_budget(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count contract attempts at their own unit, separate from source families."""
    completed = sum(row["status"] == "completed" for row in rows)
    return {
        "intended": 12,
        "eligible": 12,
        "started": completed,
        "completed": completed,
        "failed": sum(row["status"] == "completed" and not row["matched"] for row in rows),
        "censored": 0,
        "excluded": 0,
        "independent": 0,
    }


def candidate(
    root: Path,
    assessment: dict[str, Any],
    custody: dict[str, Any],
    freeze: dict[str, Any],
    controls: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
    manifest_path: Path,
    started_ns: int,
    ended_ns: int,
    *,
    fixture: bool = False,
) -> dict[str, Any]:
    """Build the full terminal schema while keeping historical failure receipts separate."""
    import os

    required = [row for row in receipts if row.get("classification") != "diagnostic"]
    passed = bool(required) and all(row["passed"] for row in required + controls)
    honest, verdict_class, ready = verdict(assessment["activated"], custody["ready"], passed)
    if fixture:
        honest, verdict_class, ready = "partial_private_unvalidated", "partial", 0
    failures = assessment["gate_check_summary"] + custody["gate_check_summary"]
    failures += [
        operand(
            Path(row.get("log_path", manifest_path)),
            "required_check.passed",
            True,
            row["passed"],
            "exp7903-owned-validation",
        )
        for row in required + controls
        if not row["passed"]
    ]
    source_history = json.loads(
        (root / "results/experiment_7892_v685_source_boundary.json").read_text()
    )
    history = list(source_history["historical_required_failures"])
    for name in (
        "experiment_7893_v685_intervention_protocol.json",
        "experiment_7894_v685_energy_fit.json",
        "experiment_7901_v685_hardware_evidence.json",
        "experiment_7902_v685_capstone.json",
    ):
        path = root / "results" / name
        old = json.loads(path.read_text())
        history.append(
            {
                "experiment_id": old["experiment_id"],
                "honest_verdict": old["honest_verdict"],
                "path": str(path),
                "sha256": sha256_file(path),
                "resolved": False,
                "required_failures": old.get("gate_check_summary", []),
            }
        )
    receipt = build_current_work_receipt(
        run_id="exp7903-20260930",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details={"models": "none"},
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=ended_ns,
        phase_spans=[
            {
                "phase": "authority_source_and_validation",
                "start_s": 0,
                "end_s": (ended_ns - started_ns) / 1e9,
                "completed_units": 12 + len(receipts),
            }
        ],
    )
    sources = custody["hashes"] + [
        {
            "path": row["source_path"],
            "sha256": row["sha256"],
            "role": role,
            "exposure": "administrative_authority",
        }
        for role, row in assessment["authority_snapshots"].items()
    ]
    sources.append(
        {
            "path": str(root / "research-references.md"),
            "sha256": sha256_file(root / "research-references.md"),
            "role": "exposed_method_review",
        }
    )
    value = {
        **receipt,
        "experiment_id": 7903,
        "task_id": "exp7903-contract-methods",
        "milestone": MILESTONE,
        "run_date": "20260930",
        "honest_verdict": honest,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "rows": assessment["contract_rows"],
        "contract_rows": assessment["contract_rows"],
        "sample_size_budget": contract_budget(assessment["contract_rows"]),
        "source_sample_size_budget": custody["budget"],
        "source_custody_rows": custody["rows"],
        "source_custody_ready": custody["ready"],
        "acceptance_gate_results": {
            "validity": bool(passed),
            "readiness": ready,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "random_seed": 6867903,
        "reproducibility_checksum": canonical_hash(
            {
                "manifest": sha256_file(manifest_path),
                "sources": sources,
                "methods": canonical_hash(freeze),
            }
        )[7:23],
        "source_artifact_hashes": sources,
        "preconditions_checked": {
            "activation_confirmed": assessment["activated"],
            "source_custody_passed": custody["ready"],
        },
        "resolved_imports": {
            "carnot.reporting.current_work_receipt": str(
                Path(build_current_work_receipt.__code__.co_filename).resolve()
            ),
            "carnot.reporting.v686_contract_methods": str(Path(__file__).resolve()),
            "carnot.reporting.v685_authority_lifecycle": str(Path(lifecycle.__file__).resolve()),
            "carnot.reporting.source_boundary_7892": str(Path(boundary.__file__).resolve()),
            "carnot.verify.source_projection": str(Path(source_projection.__file__).resolve()),
        },
        "validation_receipts": receipts,
        "validation_command_manifest_path": str(manifest_path),
        "validation_command_manifest_sha256": sha256_file(manifest_path),
        "observed_child_commands": [row["argv"] for row in receipts],
        "historical_required_failures": history,
        "repository_health": {
            "status": "degraded_open",
            "affects_required_checks": False,
            "historical": source_history["repository_health"],
            "current_full_suite": [
                row for row in receipts if row.get("name") == "repository_full_suite"
            ],
        },
        "verifier_is_oracle": True,
        "claim_scope": {
            "authority": "administrative_contract",
            "science": "unmeasured",
            "exposure": "exposed_development",
            "independent_benefit": False,
        },
        "model_specs": [],
        "target_model": "none",
        "model_invocation_counts": {"loads": 0, "calls": 0, "tokens": 0},
        "trained_head_specs": [],
        "contract_ready_score": ready,
        "mutation_rows": controls,
        "authority_snapshots": assessment["authority_snapshots"],
        "canonical_tasks_sha256": assessment["canonical_tasks_sha256"],
        "observed_active_tasks_sha256": assessment["active_tasks_sha256"],
        "activation_confirmed": assessment["activated"],
        "activation_observation": "confirmed" if assessment["activated"] else "unconfirmed",
        "planning_matched": assessment["planning_matched"],
        "required_checks_passed": passed,
        "method_freeze": freeze,
        "literature_adoption_decisions": freeze["literature_adoption_decisions"],
        "retire_if_same_verdict": {
            "required": True,
            "identical_prior_failure": False,
            "action": "preserve_prior_failures; this is new versioned authority",
        },
    }
    value["field_principles"] = {
        key: "Record task-owned custody and mechanics; reused or planned evidence earns no scientific benefit."
        for key in value
    }
    value["field_principles"].update(
        {
            "canonical_tasks_sha256": "Only a complete-task digest from design can confirm activation; absent design means null.",
            "sample_size_budget": "Count twelve contract tasks; oracle comparisons have zero independent scientific units.",
            "source_sample_size_budget": "Count source families separately and retain exclusions.",
            "source_custody_rows": "Preserve every upstream role, exclusion reason and primitive family identity.",
            "field_principles": "Explain each owned field and gate.",
            **{
                f"acceptance_gate_results.{key}": "Readiness requires matching activation and owned checks; unmeasured science gates remain null."
                for key in value["acceptance_gate_results"]
            },
        }
    )
    return value


def primitive_rows(value: dict[str, Any]) -> dict[str, Any]:
    """Seal reduction operands rather than trusting aggregate counters on replay."""
    return {
        key: value[key]
        for key in (
            "rows",
            "source_custody_rows",
            "activation_confirmed",
            "source_custody_ready",
            "required_checks_passed",
            "mutation_rows",
        )
    }


def cold_replay(path: Path, raw: Path) -> bool:
    """Reduce saved primitive checks and family counts without rerunning producers."""
    if not path.is_file() or not raw.is_file():
        return False
    value, primitive = json.loads(path.read_text()), json.loads(raw.read_text())
    for snapshot in value.get("authority_snapshots", {}).values():
        if snapshot["exists"]:
            saved = Path(snapshot["snapshot_path"])
            if not saved.is_file() or sha256_file(saved) != snapshot["sha256"]:
                return False
    return (
        value.get("experiment_id") == 7903
        and value.get("task_id") == "exp7903-contract-methods"
        and primitive_rows(value) == primitive
        and len(value["rows"]) == 12
        and value["contract_rows"] == value["rows"]
        and all(row["absolute_metric"] == int(all(row["checks"].values())) for row in value["rows"])
        and value["sample_size_budget"] == contract_budget(primitive["rows"])
        and value["source_sample_size_budget"]
        == boundary.budget(primitive["source_custody_rows"], 640)
        and value["contract_ready_score"]
        == int(
            value["activation_confirmed"]
            and value["source_custody_ready"]
            and value["required_checks_passed"]
            and value["verdict_class"] == "circular_positive"
        )
    )
