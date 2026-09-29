"""Reduce current V684 bytes without invoking historical experiment runners.

The administrative ledger keeps missing science visible. A valid audit is not
evidence that a source decision, a learned constraint, or a device improved.
REQ-REPORT-7890-V684.
"""

from __future__ import annotations

from collections import defaultdict
import importlib
import json
from pathlib import Path
import random
import time
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.roadmap_contract import parse_design
from scripts import publication_gate


DESIGN = "openspec/change-proposals/research-roadmap-vNEXT.md"
STAGED = "research-roadmap-next.yaml"
ACTIVE = "research-roadmap.yaml"
SCIENCE = {7882, 7883, 7884, 7885, 7886, 7888}
QUALIFIED = {"positive", "circular_positive", "null"}
METRICS = (
    "probability",
    "probability_unsupported",
    "label",
    "original_human_label",
    "cost",
    "brier",
    "syntax_valid",
    "source_byte_fidelity",
    "semantic_sensitivity",
    "prediction_step",
    "feedback_step",
    "release_step",
    "admitted",
    "constraint_effect",
    "no_write_decision",
    "retention",
    "calibration_brier",
    "latency_ms",
    "service_work_ms",
    "traffic_bytes",
    "current_hardware_execution",
    "new_level_solve",
    "checkpoint_hash",
    "source_hash",
    "label_authority",
)


def _read(path: Path) -> dict[str, Any] | None:
    """Return no producer for missing or malformed external bytes."""
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _failure(
    number: int, path: Path, field: str, expected: Any, observed: Any, op: str = "=="
) -> dict[str, Any]:
    """Preserve one gate operand with the exact upstream byte identity."""
    return {
        "upstream_id": f"Exp{number}",
        "path": str(path),
        "hash": sha256_file(path) if path.is_file() else None,
        "artifact_field": field,
        "op": op,
        "expected": expected,
        "observed": observed,
    }


def resolve_imports(root: Path) -> dict[str, str]:
    """Check both actual package roots; scripts is a valid root of its own."""
    names = ("carnot.reporting.v684_capstone", "scripts.publication_gate")
    found = {name: str(Path(importlib.import_module(name).__file__).resolve()) for name in names}
    if not Path(found[names[0]]).is_relative_to((root / "python/carnot").resolve()):
        raise ValueError("carnot_import_root_changed")
    if not Path(found[names[1]]).is_relative_to((root / "scripts").resolve()):
        raise ValueError("scripts_import_root_changed")
    return found


def reduce_primitive(
    rows: list[dict[str, Any]], *, seed: int = 7890, draws: int = 10000
) -> dict[str, Any]:
    """Average repeated seeds before comparing the same source families."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("malformed_primitive_row")
        if "original_human_label" in row and row.get("label") != row["original_human_label"]:
            raise ValueError("human_label_changed")
        if row.get("completed") == 1 and row.get("eligible") == 0:
            raise ValueError("denominator_conflict")
        if row.get("status") == "completed" and row.get("family_id") is not None:
            groups[(str(row["family_id"]), str(row.get("arm")))].append(row)
    family_rows: list[dict[str, Any]] = []
    for (family, arm), repeats in sorted(groups.items()):
        values: dict[str, Any] = {"family_id": family, "arm": arm, "seed_count": len(repeats)}
        for metric in ("probability", "cost", "latency_ms", "service_work_ms", "traffic_bytes"):
            measured = [float(r[metric]) for r in repeats if type(r.get(metric)) in (int, float)]
            values[metric] = sum(measured) / len(measured) if measured else None
        labels = {r.get("label") for r in repeats if r.get("label") in (0, 1)}
        if len(labels) > 1:
            raise ValueError("family_label_conflict")
        label = next(iter(labels)) if labels else None
        values["label"] = label
        risks = [
            (float(r.get("probability", r.get("probability_unsupported"))) - label) ** 2
            for r in repeats
            if label is not None
            and type(r.get("probability", r.get("probability_unsupported"))) in (int, float)
        ]
        values["brier"] = sum(risks) / len(risks) if risks else None
        family_rows.append(values)
    arms = sorted(
        {row["arm"] for row in family_rows},
        key=lambda arm: ("control" in arm or arm == "no_write", arm),
    )
    by_arm = {(row["family_id"], row["arm"]): row for row in family_rows}
    paired: dict[str, Any] = {}
    for left_index, left in enumerate(arms):
        for right in arms[left_index + 1 :]:
            families = sorted(
                {f for f, arm in by_arm if arm == left} & {f for f, arm in by_arm if arm == right}
            )
            costs = [
                by_arm[(f, right)]["cost"] - by_arm[(f, left)]["cost"]
                for f in families
                if by_arm[(f, left)]["cost"] is not None and by_arm[(f, right)]["cost"] is not None
            ]
            briers = [
                by_arm[(f, right)]["brier"] - by_arm[(f, left)]["brier"]
                for f in families
                if by_arm[(f, left)]["brier"] is not None
                and by_arm[(f, right)]["brier"] is not None
            ]
            rng = random.Random(canonical_hash([seed, left, right]))
            sample = (
                sorted(
                    sum(costs[rng.randrange(len(costs))] for _ in costs) / len(costs)
                    for _ in range(draws)
                )
                if costs
                else []
            )
            paired[f"{left}:{right}"] = {
                "equal_coverage_families": len(families),
                "cost_n": len(costs),
                "brier_n": len(briers),
                "cost_gain": sum(costs) / len(costs) if costs else None,
                "brier_gain": sum(briers) / len(briers) if briers else None,
                "cost_gain_ci95": [
                    sample[int(0.025 * (len(sample) - 1))],
                    sample[int(0.975 * (len(sample) - 1))],
                ]
                if sample
                else None,
            }
    return {
        "independent_family_count": len({f for f, _ in groups}),
        "seed_count": sum(len(group) for group in groups.values()),
        "family_rows": family_rows,
        "paired": paired,
        "syntax_valid_count": sum(r.get("syntax_valid") is True for r in rows),
        "source_fidelity_count": sum(r.get("source_byte_fidelity") is True for r in rows),
        "source_sensitivity_count": sum(r.get("semantic_sensitivity") is True for r in rows),
        "release_violations": sum(
            type(r.get("prediction_step")) is int
            and type(r.get("release_step", r.get("feedback_step"))) is int
            and r.get("release_step", r.get("feedback_step")) <= r["prediction_step"]
            for r in rows
        ),
        "admitted_count": sum(r.get("admitted") is True for r in rows),
        "constraint_effect_count": sum(r.get("constraint_effect") is True for r in rows),
        "no_write_changes": sum(r.get("no_write_decision") is True for r in rows),
        "retention_count": sum(r.get("retention") is True for r in rows),
        "new_arc_solve_count": sum(r.get("new_level_solve") is True for r in rows),
        "device_execution_count": sum(r.get("current_hardware_execution") is True for r in rows),
    }


def collect(root: Path, date: str) -> dict[str, Any]:
    """Read all declared producer paths once, retaining every failed operand."""
    if date != "20260929":
        raise ValueError("v684_date_changed")
    design = root / DESIGN
    _, tasks = parse_design(design.read_text(), milestone="2026.09.684")
    if len(tasks) != 12 or [t["id"].split("-", 1)[0] for t in tasks] != [
        f"exp{n}" for n in range(7879, 7891)
    ]:
        raise ValueError("v684_contract_order_changed")
    sources: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    historical: list[dict[str, Any]] = []
    by_id = {task["id"]: task for task in tasks}
    for label, role in (
        (DESIGN, "design_contract"),
        (STAGED, "staged_authority"),
        (ACTIVE, "active_authority"),
    ):
        path = root / label
        sources.append(
            {
                "path": str(path),
                "sha256": sha256_file(path) if path.is_file() else None,
                "date": date if path.is_file() else None,
                "role": role,
                "exposure_status": "current" if path.is_file() else "missing",
            }
        )
        if role == "staged_authority" and not path.is_file():
            failures.append(_failure(7879, path, "authority_exists", True, False))
    for number, task in zip(range(7879, 7891), tasks, strict=True):
        path = root / task["deliverable"]
        own = number == 7890
        data = None if own else _read(path)
        digest = sha256_file(path) if path.is_file() and not own else None
        skip_paths = [
            p for p in sorted((root / "results").glob(f"experiment_{number}_*.json")) if p != path
        ]
        skips = [
            {"path": str(p), "hash": sha256_file(p), "role": "conductor_skip_receipt"}
            for p in skip_paths
        ]
        state = (
            "own_administrative_row"
            if own
            else "missing_producer"
            if not path.is_file()
            else "invalid_json"
            if data is None
            else str(data.get("verdict_class", "missing_field"))
        )
        if not own:
            if data is None:
                failures.append(_failure(number, path, "producer_exists", True, path.is_file()))
            else:
                for field, expected in (
                    ("experiment_id", number),
                    ("task_id", task["id"]),
                    ("milestone", "2026.09.684"),
                    ("run_date", date),
                    ("flagged_adversarial", False),
                ):
                    if data.get(field, "missing_field") != expected:
                        failures.append(
                            _failure(
                                number, path, field, expected, data.get(field, "missing_field")
                            )
                        )
                if data.get("verdict_class", "missing_field") not in QUALIFIED:
                    failures.append(
                        _failure(
                            number,
                            path,
                            "verdict_class",
                            sorted(QUALIFIED),
                            data.get("verdict_class", "missing_field"),
                            "in",
                        )
                    )
                checks = data.get("validation_receipts", [])
                checks = checks.get("checks", []) if isinstance(checks, dict) else checks
                if isinstance(checks, list):
                    for check in checks:
                        if (
                            isinstance(check, dict)
                            and check.get("classification", check.get("scope")) == "required"
                            and (
                                check.get("passed") is False
                                or check.get("exit_code") != 0
                                or check.get("timed_out")
                            )
                        ):
                            historical.append(
                                {
                                    "upstream_id": f"Exp{number}",
                                    "path": str(path),
                                    "name": check.get("name"),
                                    "exit_code": check.get("exit_code"),
                                    "log_path": check.get("log_path"),
                                    "log_sha256": check.get("log_sha256"),
                                }
                            )
                            failures.append(
                                _failure(
                                    number,
                                    path,
                                    f"validation_receipts.{check.get('name', 'unnamed')}",
                                    0,
                                    check.get("exit_code"),
                                )
                            )
            for gate in task["gated_on"]:
                upstream = by_id[gate["upstream"]]
                upstream_path = root / upstream["deliverable"]
                upstream_data = _read(upstream_path)
                observed = (
                    upstream_data.get(gate["artifact_field"], "missing_field")
                    if upstream_data is not None
                    else "missing_source"
                )
                passed = (
                    observed == gate["value"] if gate["op"] == "==" else observed in gate["value"]
                )
                if not passed:
                    failures.append(
                        _failure(
                            int(gate["upstream"][3:7]),
                            upstream_path,
                            gate["artifact_field"],
                            gate["value"],
                            observed,
                            gate["op"],
                        )
                    )
        own_failures = any(
            f["upstream_id"] == f"Exp{number}" and f["path"] == str(path) for f in failures
        )
        eligible = bool(data) and not own_failures and data.get("verdict_class") in QUALIFIED
        role = (
            "own_administrative"
            if own
            else "scientific_producer"
            if number in SCIENCE
            else "administrative_or_continuity"
        )
        exposure = "exposed_development" if number in SCIENCE else "administrative"
        evidence.append(
            {
                "upstream_id": f"Exp{number}",
                "task_id": task["id"],
                "path": str(path),
                "hash": digest,
                "status": state,
                "eligible": eligible,
                "role": role,
                "source_exposure": exposure,
                "label_authority": "original_human_label_only" if number in SCIENCE else "none",
                "skip_receipts": skips,
            }
        )
        sources.append(
            {
                "path": str(path),
                "sha256": digest,
                "date": data.get("run_date") if data else None,
                "role": role,
                "exposure_status": exposure,
            }
        )
        sources.extend(
            {
                "path": item["path"],
                "sha256": item["hash"],
                "date": None,
                "role": "conductor_skip_receipt",
                "exposure_status": "administrative",
            }
            for item in skips
        )
        raw = data.get("rows", []) if data else []
        raw = raw if isinstance(raw, list) else []
        if not raw:
            units.append(
                {
                    "upstream_id": f"Exp{number}",
                    "arm": task["id"],
                    "family_id": None,
                    "seed": None,
                    "status": state,
                    "metric": {},
                    "eligible": 0,
                    "started": 0,
                    "completed": 0,
                    "censored": 0,
                    "excluded": int(not own),
                    "independent": 0,
                }
            )
        projected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, row in enumerate(raw):
            row = row if isinstance(row, dict) else {"status": "malformed"}
            family = row.get("source_family", row.get("family_id", row.get("family")))
            status = str(row.get("status", "unknown"))
            unit = {
                "upstream_id": f"Exp{number}",
                "primitive_index": index,
                "arm": row.get("arm"),
                "family_id": family,
                "seed": row.get("seed"),
                "status": status,
                "metric": {key: row[key] for key in METRICS if key in row},
                "eligible": int(eligible and status != "excluded"),
                "started": int(status not in {"unstarted", "excluded"}),
                "completed": int(status == "completed"),
                "censored": int(status == "censored"),
                "excluded": int(status == "excluded"),
                "independent": int(eligible and family is not None and str(family) not in seen),
            }
            if family is not None:
                seen.add(str(family))
            units.append(unit)
            projected.append(
                {
                    **unit["metric"],
                    "family_id": family,
                    "arm": unit["arm"],
                    "seed": unit["seed"],
                    "status": status,
                }
            )
        summary = reduce_primitive(projected)
        comparisons.append(
            {
                "upstream_id": f"Exp{number}",
                "path": str(path),
                "hash": digest,
                "row_count": len(raw),
                "source_exposure": exposure,
                **summary,
            }
        )
    for number in (7877, 7878):
        paths = sorted((root / "results").glob(f"experiment_{number}_v683_*.json"))
        for path in paths:
            data = _read(path) or {}
            checks = data.get("validation_receipts", [])
            checks = checks.get("checks", []) if isinstance(checks, dict) else checks
            for check in checks if isinstance(checks, list) else []:
                if (
                    isinstance(check, dict)
                    and check.get("classification", check.get("scope")) == "required"
                    and (check.get("passed") is False or check.get("exit_code") != 0)
                ):
                    historical.append(
                        {
                            "upstream_id": f"Exp{number}",
                            "path": str(path),
                            "name": check.get("name"),
                            "exit_code": check.get("exit_code"),
                            "log_path": check.get("log_path"),
                            "log_sha256": check.get("log_sha256"),
                        }
                    )
    prior_attempt = root / "results/raw/experiment_7890_v684_capstone/attempt1_disqualified.json"
    if prior_attempt.is_file():
        prior = _read(prior_attempt) or {}
        sources.append(
            {
                "path": str(prior_attempt),
                "sha256": sha256_file(prior_attempt),
                "date": prior.get("run_date"),
                "role": "prior_owned_failed_attempt",
                "exposure_status": "administrative",
            }
        )
        for check in prior.get("validation_receipts", []):
            if (
                isinstance(check, dict)
                and check.get("classification") == "required"
                and check.get("passed") is False
            ):
                historical.append(
                    {
                        "upstream_id": "Exp7890-attempt1",
                        "path": str(prior_attempt),
                        "name": check.get("name"),
                        "exit_code": check.get("exit_code"),
                        "log_path": check.get("log_path"),
                        "log_sha256": check.get("log_sha256"),
                    }
                )
    return {
        "task_evidence_rows": evidence,
        "gate_check_summary": failures,
        "source_artifact_hashes": sources,
        "rows": units,
        "recomputed_comparison_rows": comparisons,
        "historical_required_failures": historical,
    }


def build_candidate(root: Path, date: str) -> dict[str, Any]:
    """Keep administrative completion separate from scientific qualification."""
    started = time.monotonic()
    ledger = collect(root, date)
    audit_done = time.monotonic()
    science_ready = all(
        row["eligible"]
        for row in ledger["task_evidence_rows"]
        if int(row["upstream_id"][3:]) in SCIENCE
    )
    budget = {
        key: sum(int(row[key]) for row in ledger["rows"])
        for key in ("eligible", "started", "completed", "censored", "excluded", "independent")
    }
    budget.update(
        {"intended": len(ledger["rows"]), "independent_unit": "source_family_not_seed_or_view"}
    )
    pub = publication_gate.evaluate()
    publication_done = time.monotonic()
    # A missing comparison is an unknown effect, never a zero effect.
    gaps = {
        "FR-12": {
            "decision": "blocked" if not science_ready else "null_pending_independent_holdout",
            "required_producers": [7882, 7883, 7884],
            "probability_quality": None,
            "decision_benefit": None,
            "authority": "original human labels; exposed development",
            "next_question": "Do paired family Brier and decision cost gains survive independent hidden sources?",
        },
        "FR-11": {
            "decision": "blocked" if not science_ready else "null_pending_causal_replication",
            "required_producers": [7885, 7886],
            "retention": None,
            "next_question": "Does an admitted constraint change later decisions against no-write without old-task damage?",
        },
        "FR-05/FR-08/NFR-01": {
            "decision": "blocked" if not science_ready else "null_pending_live_deployment",
            "required_producers": [7887, 7888, 7889],
            "efficiency": None,
            "next_question": "Does complete service work and a new live board receipt improve useful throughput?",
        },
    }
    verdict = (
        "complete_blocked_required_v684_science"
        if not science_ready
        else "complete_null_independent_benefit_unshown"
    )
    result: dict[str, Any] = {
        "experiment_id": 7890,
        "task_id": "exp7890-capstone",
        "milestone": "2026.09.684",
        "run_date": date,
        "honest_verdict": verdict,
        "verdict_class": "blocked" if not science_ready else "null",
        "flagged_adversarial": False,
        **ledger,
        "sample_size_budget": budget,
        "acceptance_gate_results": {
            "validity": None,
            "readiness": 0,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": time.monotonic() - started,
        "phase_spans": [
            {
                "phase": "preconditions_and_independent_rows",
                "duration_s": audit_done - started,
                "completed_units": 12,
            },
            {
                "phase": "publication_gate_and_decisions",
                "duration_s": publication_done - audit_done,
                "completed_units": 3,
            },
        ],
        "random_seed": 7890,
        "preconditions_checked": [
            {
                "path": item["path"],
                "hash": item["sha256"],
                "date": item["date"],
                "source_role": item["role"],
                "exposure_status": item["exposure_status"],
                "observed": item["sha256"] is not None,
            }
            for item in ledger["source_artifact_hashes"]
        ],
        "validation_receipts": [],
        "validation_command_manifest_path": None,
        "observed_child_commands": [],
        "repository_health": {"status": "unresolved_historical_debt"},
        "verifier_is_oracle": False,
        "claim_scope": {
            "natural_annotations": "exposed_development_only",
            "fixture_agreement": "circular_only",
            "fresh_holdout_generalization": False,
            "gap_oracle_distinct": "open",
            "september_28_retractions": [4245, 5160, 5171],
            "diffusiongemma": "not_promoted",
            "new_arc_solve": False,
            "new_board_execution": False,
            "llm_weight_training": False,
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "MODEL_SPECS": [],
        "model_specs": [],
        "target_model": "none (no pretrained model)",
        "trained_head_specs": [],
        "model_invocation_counts": {"loads": 0, "calls": 0, "tokens": 0, "model_file_hashes": []},
        "capstone_execution_ready_score": 0,
        "milestone_evidence_complete_score": int(science_ready),
        "milestone_benefit_score": 0,
        "publication_gate_results": pub,
        "paper_ready": pub.get("paper_ready"),
        "unmet_gates": pub.get("unmet_gates"),
        "prd_gap_decisions": gaps,
        "retirement_decisions": [
            {
                "scope": "V684 current mechanisms",
                "decision": "no_same_scope_retirement_established",
                "reason": "Missing current science cannot repeat an identical prior verdict.",
            }
        ],
        "literature_adoption_decisions": {
            "source_sufficiency": "continue with held-out family labels before adoption",
            "causal_feedback": "continue only after qualified delayed-feedback and no-write comparison",
            "continual_calibration": "distinct estimand from skipped scheduling experiment",
            "whole_service": "measure complete service before hardware placement",
            "diffusiongemma": "defer until an independent scorer and local control pass",
        },
        "report_path": "docs/research-notes/milestone-2026.09.684-decisions.md",
        "resolved_imports": resolve_imports(Path(__file__).resolve().parents[3]),
    }
    result["reproducibility_checksum"] = canonical_hash(
        {
            "source_hashes": ledger["source_artifact_hashes"],
            "seed": 7890,
            "configuration": "v684_independent_capstone_v1",
            "rows": ledger["rows"],
        }
    )
    result["field_principles"] = {
        key: "Bind this audit field to current bytes, measured work, and its claim limit."
        for key in result
    }
    result["field_principles"]["acceptance_gate_results"] = {
        key: "Unmeasured science remains null; validation controls only owned readiness."
        for key in result["acceptance_gate_results"]
    }
    result["field_principles"]["field_principles"] = (
        "Explain why every task-owned field and gate exists."
    )
    return result


def cold_replay(path: Path, root: Path) -> list[str]:
    """Recompute current source and row identity before trusting a candidate."""
    value = _read(path)
    if value is None:
        return ["candidate_unreadable"]
    expected = build_candidate(root, "20260929")
    errors: list[str] = []
    for source in value.get("source_artifact_hashes", []):
        if source.get("role") == "own_administrative":
            continue
        candidate = Path(source["path"])
        digest = sha256_file(candidate) if candidate.is_file() else None
        if digest != source["sha256"]:
            errors.append("source_bytes_changed")
    for key in (
        "task_evidence_rows",
        "rows",
        "recomputed_comparison_rows",
        "sample_size_budget",
        "reproducibility_checksum",
        "milestone_evidence_complete_score",
    ):
        if value.get(key) != expected[key]:
            errors.append(f"{key}_changed")
    if (
        value.get("gate_check_summary", [])[: len(expected["gate_check_summary"])]
        != expected["gate_check_summary"]
    ):
        errors.append("gate_operands_changed")
    for item in value.get("validation_receipts", []):
        log = Path(item["log_path"])
        if not log.is_file() or sha256_file(log) != item["log_sha256"]:
            errors.append("validation_log_changed")
    return sorted(set(errors))
