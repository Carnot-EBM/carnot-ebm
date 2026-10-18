"""Reduce V674 task evidence without promoting missing science (REQ-REPORT-7752)."""

from __future__ import annotations

from pathlib import Path
import json
import os
import socket
from typing import Any

from carnot.experiment_7739_v674_contract_methods import (
    DESIGN,
    compare_contract,
    resolve_authority,
)
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path("results/experiment_7752_v674_capstone.json")
RAW = Path("results/raw/experiment_7752_v674_capstone")
MODULE = Path("python/carnot/experiment_7752_v674_capstone.py")
CLI = Path("scripts/experiments/experiment_7752_v674_capstone.py")
TEST = Path("tests/python/test_experiment_7752_v674_capstone.py")
REQUIRED = (7744, 7746, 7747)
ELIGIBLE = {"null", "positive", "circular_positive"}
PRINCIPLES = {
    "honest_verdict": "Completion and scientific benefit are different facts.",
    "verdict_class": "Unchanged external failure is blocked, never partial.",
    "flagged_adversarial": "Flagged evidence opens no gate.",
    "gate_check_summary": "A missing field is a broken contract, not a scientific null.",
    "acceptance_gate_results": "Negative science may be valid while invalid execution never qualifies.",
    "rows": "Every comparison must be independently recomputable.",
    "sample_size_budget": "Seeds, sentences and actions do not multiply families or games.",
    "claim_scope": "A new split cannot erase historical exposure.",
    "inference_substrate": "Cached scoring cannot claim live generation.",
    "inference_substrate_class": "Duration checks must match the work performed.",
    "MODEL_SPECS": "The experimental model is distinct from the coding backend.",
    "model_invoked": "Model strings alone are not invocation evidence.",
    "execution_venue": "Host work is not fabric work.",
    "phase_spans": "Real timings and bounded silence expose stalled work.",
    "random_seed": "Independent replay needs fixed inputs.",
    "source_artifact_hashes": "An artifact cannot authenticate itself as upstream.",
    "preconditions_checked": "Missing access must block before expensive work.",
    "validation_receipts": "Only registered checks qualify the current result.",
    "verifier_is_oracle": "Circular success cannot establish independent verification value.",
    "field_principles": "The rationale travels with the contract.",
    "capstone_complete_score": "A complete accounting of failure is not incomplete owned execution.",
    "task_dispositions": "The document and actual work have the same roster.",
    "prd_gap_findings": "Administrative readiness is not research progress.",
    "publication_gate_results": "Publication readiness is computed, never declared by narrative.",
    "next_question": "Repeated failed scope requires a changed prerequisite.",
}


def failed(number: int, path: str, field: str, expected: Any, observed: Any) -> dict[str, Any]:
    """Name the exact upstream operand that prevents a scientific claim."""
    return {
        "check": "required_source_eligible",
        "upstream_id": f"Exp{number}",
        "artifact_path": path,
        "field": field,
        "op": "==",
        "expected": expected,
        "observed": observed,
    }


def authority(root: Path) -> dict[str, Any]:
    """Read independent roadmap and design bytes under the requested root."""
    path, roadmap, candidates = resolve_authority(root)
    comparison = compare_contract((root / DESIGN).read_text(), roadmap)
    return {
        "tasks": roadmap["tasks"],
        "comparison": comparison,
        "candidates": candidates,
        "hashes": {
            path.relative_to(root).as_posix(): sha256_file(path),
            DESIGN.as_posix(): sha256_file(root / DESIGN),
        },
    }


def account(
    root: Path, tasks: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Cold-read each producer and raw row file, retaining all terminal nulls."""
    if len(tasks) != 14 or any(
        not task["id"].startswith(f"exp{7739 + i}-") for i, task in enumerate(tasks)
    ):
        raise ValueError("V674 fourteen-task order required")
    hashes: dict[str, Any] = {
        name: []
        for name in (
            "eligible_producers",
            "historical_disqualified_sources",
            "pre_gate_receipts",
            "missing_inputs",
        )
    }
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        number = 7739 + index
        planned = task["deliverable"]
        alternate = planned.replace("_v674_", "_")
        label = planned if (root / planned).is_file() else alternate
        path = root / label
        value: dict[str, Any] = {}
        state = "planned_output" if number == 7752 else "absent"
        if number != 7752 and path.is_file():
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError(f"producer object required: {label}")
            state = (
                "pre_gate_receipt" if value.get("schema") == "blocked_gate_check_v1" else "producer"
            )
            if label == alternate and state != "pre_gate_receipt":
                raise ValueError(f"pre-gate schema required: {label}")
        raw_dir = planned.removesuffix(".json").replace("results/", "results/raw/")
        raw_label = next(
            (
                f"{raw_dir}/{name}"
                for name in (
                    "rows.json",
                    "rows.jsonl",
                    "independent_reduction.json",
                    "sentence_protocol_manifest.json",
                )
                if (root / raw_dir / name).is_file()
            ),
            None,
        )
        raw = root / raw_label if raw_label else None
        raw_hash = sha256_file(raw) if number != 7752 and raw is not None else None
        source_hash = sha256_file(path) if state not in {"absent", "planned_output"} else None
        if number != 7752:
            bucket = (
                "missing_inputs"
                if state == "absent"
                else "pre_gate_receipts"
                if state == "pre_gate_receipt"
                else "eligible_producers"
                if value.get("verdict_class") in ELIGIBLE
                and value.get("flagged_adversarial") is False
                and str(value.get("honest_verdict", "")).startswith("complete_")
                else "historical_disqualified_sources"
            )
            hashes[bucket].append(
                {
                    "experiment_id": number,
                    "path": label if source_hash else planned,
                    "sha256": source_hash,
                    "raw_rows_path": raw_label if raw_hash else None,
                    "raw_rows_sha256": raw_hash,
                }
            )
        row = {
            "unit_id": task["id"],
            "task_id": task["id"],
            "experiment_id": number,
            "order": index + 1,
            "arm": "task_accounting",
            "planned_path": planned,
            "evidence_path": label if source_hash else None,
            "availability": state,
            "verdict_class": value.get("verdict_class"),
            "honest_verdict": value.get("honest_verdict"),
            "flagged_adversarial": value.get("flagged_adversarial"),
            "raw_metrics": {
                "verdict_class": value.get("verdict_class"),
                "honest_verdict": value.get("honest_verdict"),
            },
            "numerators": {},
            "denominators": {
                "independent_tasks": 1,
                "independent_families": None,
                "independent_games": None,
            },
            "exclusions": [state] if state != "producer" else [],
            "censored": state in {"absent", "pre_gate_receipt"},
            "input_hashes": {label: source_hash} if source_hash else {},
            "raw_rows_sha256": raw_hash,
        }
        rows.append(row)
        if number in REQUIRED:
            if state != "producer":
                failures.append(failed(number, planned, "exists", True, False))
            elif value.get("flagged_adversarial") is not False:
                failures.append(
                    failed(
                        number,
                        label,
                        "flagged_adversarial",
                        False,
                        value.get("flagged_adversarial"),
                    )
                )
            elif value.get("verdict_class") not in ELIGIBLE or not str(
                value.get("honest_verdict", "")
            ).startswith("complete_"):
                failures.append(
                    failed(
                        number, label, "verdict_class", sorted(ELIGIBLE), value.get("verdict_class")
                    )
                )
    return rows, hashes, failures


def build_artifact(
    root: Path,
    publication: dict[str, Any],
    receipts: list[dict[str, Any]] | None = None,
    spans: list[dict[str, Any]] | None = None,
    duration_s: float = 0.0,
) -> dict[str, Any]:
    """Make one bounded result from source custody and measured validation."""
    root = root.resolve()
    source = authority(root)
    rows, buckets, failures = account(root, source["tasks"])
    if not source["comparison"]["passed"]:
        failures.append(
            {
                "check": "independent_contract",
                "upstream_id": "Exp7739",
                "artifact_path": str(DESIGN),
                "field": "table_json_yaml_match",
                "op": "==",
                "expected": True,
                "observed": source["comparison"]["errors"],
            }
        )
    checked = all(receipt.get("passed") is True for receipt in receipts or [])
    valid = source["comparison"]["passed"] and checked
    verdict = "disqualified" if not valid else "blocked" if failures else "null"
    honest = (
        "complete_disqualified_v674_capstone_validation"
        if not valid
        else "complete_blocked_required_v674_evidence"
        if failures
        else "complete_null_v674_scientific_accounting"
    )
    rows[-1]["verdict_class"] = verdict
    rows[-1]["honest_verdict"] = honest
    rows[-1]["raw_metrics"] = {"verdict_class": verdict, "honest_verdict": honest}
    dispositions = [
        {
            "experiment_id": row["experiment_id"],
            "task_id": row["task_id"],
            "availability": row["availability"],
            "verdict_class": row["verdict_class"],
            "honest_verdict": row["honest_verdict"],
            "evidence_path": row["evidence_path"],
            "producer_eligible": row["verdict_class"] in ELIGIBLE
            and row["flagged_adversarial"] is False,
            "minimum_changed_prerequisite": (
                "none for this accounting slot"
                if row["experiment_id"] == 7752
                else "new qualified producer at planned path"
                if row["availability"] == "absent"
                else "repair upstream registered gate"
                if row["availability"] == "pre_gate_receipt"
                else "repair recorded disqualification in same scope"
                if row["verdict_class"] == "disqualified"
                else "supply required scientific source"
                if row["verdict_class"] == "blocked"
                else "none"
            ),
        }
        for row in rows
    ]
    audit_path = root / source["tasks"][7747 - 7739]["deliverable"]
    audit = json.loads(audit_path.read_text()) if audit_path.is_file() else {}
    static_eligible = audit.get("independent_static_eligible")
    online_eligible = audit.get("independent_online_eligible")
    gates = {
        "validity": valid,
        "readiness": None,
        "probability_quality": None,
        "decision_benefit": None,
        "retention": None,
        "efficiency": None,
    }
    findings = {
        "local_supervision_effect": None,
        "source_dependence": None,
        "decision_value": None,
        "retained_admissions": None,
        "complete_static_comparison": None,
        "retention": None,
        "independent_static_eligible": static_eligible,
        "independent_online_eligible": online_eligible,
    }
    qwen = rows[7745 - 7739]
    arc = rows[7749 - 7739]
    service = rows[7750 - 7739]
    field_principles: dict[str, Any] = {}
    artifact: dict[str, Any] = {
        "schema": "carnot.exp7752.v674.capstone.v1",
        "experiment_id": 7752,
        "milestone": "2026.09.674",
        "run_date": "20260927",
        "honest_verdict": honest,
        "verdict_class": verdict,
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "acceptance_gate_results": gates,
        "rows": rows,
        "task_dispositions": dispositions,
        "sample_size_budget": {
            "intended": 14,
            "started": sum(r["availability"] == "producer" for r in rows),
            "completed": sum(r["verdict_class"] is not None for r in rows),
            "eligible": len(buckets["eligible_producers"]),
            "excluded": len(buckets["historical_disqualified_sources"]),
            "censored": sum(r["censored"] for r in rows),
            "effective_independent_N": None,
            "independent_families": None,
            "independent_games": None,
        },
        "claim_scope": {
            "RAGTruth": "development_only",
            "constructed_truth": "fixture_only",
            "ARC": "adapter_withheld_public",
            "fresh_generalization_eligible": False,
        },
        "fresh_generalization_eligible": False,
        "inference_substrate": "cpu_aggregation",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": {
            key: 0
            for key in (
                "loads",
                "forwards",
                "generations",
                "input_tokens",
                "output_tokens",
                "failures",
                "cancellations",
            )
        },
        "execution_venue": "host",
        "execution_venue_details": {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "gpu_uuid": None,
        },
        "phase_spans": spans or [],
        "duration_s": duration_s,
        "random_seed": {"seeds": [], "purpose": "deterministic_aggregation"},
        "source_artifact_hashes": {
            **buckets,
            "authority": source["hashes"],
            "planned_output_is_input": False,
        },
        "preconditions_checked": {
            "absolute_root": str(root),
            "authority_match": source["comparison"]["passed"],
            "authority_errors": source["comparison"]["errors"],
            "owned_output_parent_exists": (root / OUTPUT.parent).is_dir(),
            "effective_coding_backend": "codex",
            "experimental_model_load": False,
        },
        "validation_receipts": {
            "frozen_scope": {
                "changed_modules": [str(MODULE)],
                "static_paths": [str(CLI)],
                "tests": [str(TEST)],
                "requirements": ["REQ-REPORT-7752"],
            },
            "commands": receipts or [],
            "required_checks_passed": checked,
            "e2e": "SCENARIO-REPORT-7752-REPLAY",
            "global_suite_debt": None,
        },
        "verifier_is_oracle": False,
        "capstone_complete_score": int(valid and not failures),
        "scientific_findings": findings,
        "domain_results": {
            "qwen": {
                "availability": qwen["availability"],
                "verdict_class": qwen["verdict_class"],
                "claim": "development_only",
            },
            "ARC": {
                "availability": arc["availability"],
                "verdict_class": arc["verdict_class"],
                "claim": "adapter_withheld_public",
            },
            "efficiency": {
                "availability": service["availability"],
                "verdict_class": service["verdict_class"],
            },
        },
        "prd_gap_findings": [
            {"gap": "useful_localized_verification", "qualified": None},
            {"gap": "retained_online_acquisition", "qualified": None},
            {"gap": "live_path_value", "qualified": None},
        ],
        "publication_gate_results": publication,
        "publication_evidence_disposition": "draft_only_no_submission",
        "established_FoVer_AUROC": 0.9131,
        "next_question": "Repair the Exp7740 protocol validation before measuring paired family decisions.",
        "live_ARC_selector_enabled": False,
        "retirement_applied": [],
        "publication_performed": False,
        "production_defaults_changed": False,
        "generator_training_performed": False,
    }
    artifact["reproducibility_checksum"] = canonical_hash(
        {
            "authority": source["hashes"],
            "sources": buckets,
            "reducer": sha256_file(ROOT / MODULE),
            "parameters": ["aggregation", []],
        }
    )
    field_principles.update(
        {key: PRINCIPLES.get(key, "Measured evidence bounds this field.") for key in artifact}
    )
    field_principles["acceptance_gates"] = {
        key: PRINCIPLES["acceptance_gate_results"] for key in gates
    }
    field_principles["field_principles"] = PRINCIPLES["field_principles"]
    artifact["field_principles"] = field_principles
    return artifact


def cold_replay(value: dict[str, Any], root: Path) -> list[str]:
    """Reopen source bytes and independently compare the immutable conclusions."""
    expected = build_artifact(
        root,
        value.get("publication_gate_results", {}),
        value.get("validation_receipts", {}).get("commands", []),
        value.get("phase_spans", []),
        value.get("duration_s", 0.0),
    )
    keys = (
        "rows",
        "task_dispositions",
        "source_artifact_hashes",
        "gate_check_summary",
        "sample_size_budget",
        "acceptance_gate_results",
        "scientific_findings",
        "domain_results",
        "prd_gap_findings",
        "capstone_complete_score",
        "reproducibility_checksum",
        "honest_verdict",
        "verdict_class",
    )
    return [key for key in keys if canonical_hash(value.get(key)) != canonical_hash(expected[key])]
