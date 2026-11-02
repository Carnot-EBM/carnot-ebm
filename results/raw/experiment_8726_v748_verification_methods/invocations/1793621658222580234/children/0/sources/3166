"""Reduce V675 sources without promoting queue receipts into scientific results.

REQ-REPORT-7766; SCENARIO-REPORT-7766-CONTRACT/BLOCKED/REPLAY.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.experiment_7753_v675_contract_methods import (
    DESIGN,
    compare_contract,
    resolve_authority,
)
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path("results/experiment_7766_v675_capstone.json")
RAW = Path("results/raw/experiment_7766_v675_capstone")
MODULE = Path("python/carnot/experiment_7766_v675_capstone.py")
CLI = Path("scripts/experiments/experiment_7766_v675_capstone.py")
TEST = Path("tests/python/test_experiment_7766_v675_capstone.py")
ELIGIBLE = {"null", "positive", "circular_positive"}
METRICS = (
    "sentence_protocol_ready_score",
    "training_runtime_ready_score",
    "evidence_view_ready_score",
    "fit_ready_score",
    "decision_evidence_ready_score",
    "pilot_evidence_ready_score",
    "online_runtime_ready_score",
    "online_learning_ready_score",
    "independent_static_eligible",
    "independent_online_eligible",
    "organic_runner_ready_score",
    "organic_selector_ready_score",
    "service_evidence_ready_score",
)
PRINCIPLES = {
    "experiment_id": "An artifact must have a unique current owner.",
    "milestone": "An artifact must have a unique current owner.",
    "run_date": "An artifact must have a unique current owner.",
    "honest_verdict": "A terminal record must not waste attempts on unchanged inputs.",
    "verdict_class": "The claim class travels with the evidence.",
    "flagged_adversarial": "Invalid evidence must not open downstream gates.",
    "gate_check_summary": "Missing producers and failed scientific thresholds are different causes.",
    "rows": "Aggregates must be recomputable without rerunning science.",
    "acceptance_gate_results": "A working protocol is not evidence of benefit.",
    "duration_s": "Duration must describe actual work without padding.",
    "phase_spans": "Duration must describe actual work without padding.",
    "random_seed": "A third party needs the same experiment inputs.",
    "reproducibility_checksum": "A third party needs the same experiment inputs.",
    "sample_size_budget": "Repeated views and seeds do not increase independent family count.",
    "source_artifact_hashes": "A missing producer cannot be replaced with an old result.",
    "preconditions_checked": "Access and validity must be established before expensive work.",
    "validation_receipts": "All registered checks must pass before readiness opens.",
    "verifier_is_oracle": "Execution truth and independent semantic verification differ.",
    "claim_scope": "Natural annotation comparisons remain exposed development evidence.",
    "inference_substrate": "Duration floors must match the invoked substrate.",
    "inference_substrate_class": "Duration floors must match the invoked substrate.",
    "MODEL_SPECS": "An upstream model is not a current invocation.",
    "model_specs": "An upstream model is not a current invocation.",
    "model_invocation_counts": "An upstream model is not a current invocation.",
    "capstone_complete_score": "Bookkeeping cannot substitute for completed science.",
    "task_dispositions": "The design and executed roster must agree.",
    "prd_gap_findings": "The milestone must advance or bound three gaps.",
    "publication_gate_results": "Publication readiness keeps its fixed historical definition.",
    "next_question": "Repeated scope requires a changed prerequisite.",
}


def authority(root: Path) -> dict[str, Any]:
    """Bind the literal design and the exact matching active or staged roadmap."""
    path, roadmap, candidates = resolve_authority(root)
    comparison = compare_contract((root / DESIGN).read_text(), roadmap)
    return {
        "tasks": roadmap["tasks"],
        "roadmap": roadmap,
        "comparison": comparison,
        "candidates": candidates,
        "design_path": str(DESIGN),
        "roadmap_path": str(path.relative_to(root)),
        "hashes": {
            str(DESIGN): sha256_file(root / DESIGN),
            str(path.relative_to(root)): sha256_file(path),
        },
    }


def _failure(
    number: int, path: str, digest: str | None, field: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Keep the exact failing operand so a later run can identify a change."""
    return {
        "upstream_id": f"Exp{number}",
        "artifact_path": path,
        "artifact_hash": digest,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
    }


def account(
    root: Path, tasks: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Read every declared producer and any separate conductor receipt."""
    if len(tasks) != 14 or [t["id"].split("-")[0] for t in tasks] != [
        f"exp{number}" for number in range(7753, 7767)
    ]:
        raise ValueError("V675 fourteen-task order required")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        number = 7753 + index
        planned = task["deliverable"]
        producer = root / planned
        receipt_label = (
            f"results/experiment_{number}_{task['id'].split('-', 1)[1].replace('-', '_')}.json"
        )
        receipt = root / receipt_label
        has_producer = number != 7766 and producer.is_file()
        has_receipt = number != 7766 and receipt.is_file() and receipt != producer
        value = json.loads(producer.read_text()) if has_producer else {}
        pre_gate = json.loads(receipt.read_text()) if has_receipt else {}
        producer_hash = sha256_file(producer) if has_producer else None
        receipt_hash = sha256_file(receipt) if has_receipt else None
        state = (
            "planned_output"
            if number == 7766
            else "producer"
            if has_producer
            else "pre_gate_receipt"
            if has_receipt
            else "absent"
        )
        raw_dir = root / planned.removesuffix(".json").replace("results/", "results/raw/")
        raw_paths = [
            p.relative_to(root).as_posix()
            for p in (
                raw_dir / "rows.json",
                raw_dir / "rows.jsonl",
                raw_dir / "independent_reduction.json",
            )
            if number != 7766 and p.is_file()
        ]
        raw_hashes = {p: sha256_file(root / p) for p in raw_paths}
        eligible = (
            state == "producer"
            and value.get("verdict_class") in ELIGIBLE
            and value.get("flagged_adversarial") is False
            and str(value.get("honest_verdict", "")).startswith("complete_")
        )
        metrics = {key: value.get(key) for key in METRICS if key in value}
        row = {
            "unit_id": task["id"],
            "task_id": task["id"],
            "experiment_id": number,
            "order": index + 1,
            "arm": "task_accounting",
            "producer_path": planned,
            "producer_hash": producer_hash,
            "pre_gate_receipt_path": receipt_label if has_receipt else None,
            "pre_gate_receipt_hash": receipt_hash,
            "pre_gate_schema": pre_gate.get("schema"),
            "availability": state,
            "verdict_class": value.get("verdict_class"),
            "honest_verdict": value.get("honest_verdict"),
            "flagged_adversarial": value.get("flagged_adversarial"),
            "producer_eligible": eligible,
            "raw_paths": raw_paths,
            "input_hashes": raw_hashes,
            "raw_metrics": metrics,
            "exclusions": [] if eligible else [state],
            "censored": state in {"absent", "pre_gate_receipt"},
            "effective_independent_N": None,
        }
        rows.append(row)
        if number == 7766:
            continue
        sources.append(
            {
                "upstream_id": f"Exp{number}",
                "path": planned,
                "sha256": producer_hash,
                "date": value.get("run_date"),
                "imported_fields": ["verdict_class", "honest_verdict", *metrics],
                "eligible": eligible,
                "pre_gate_path": receipt_label if has_receipt else None,
                "pre_gate_sha256": receipt_hash,
                "raw_paths": raw_hashes,
            }
        )
        if not has_producer:
            failures.append(_failure(number, planned, None, "producer_exists", True, False))
        elif not eligible:
            failures.append(
                _failure(
                    number,
                    planned,
                    producer_hash,
                    "producer_eligible",
                    True,
                    {
                        "verdict_class": value.get("verdict_class"),
                        "flagged_adversarial": value.get("flagged_adversarial"),
                    },
                )
            )
        for check in value.get("gate_check_summary", []):
            failures.append(
                _failure(
                    number,
                    planned,
                    producer_hash,
                    str(check.get("field", check.get("check", "upstream_gate"))),
                    check.get("expected", True),
                    check.get("observed", False),
                )
            )
        if number == 7762:
            for key in ("independent_static_eligible", "independent_online_eligible"):
                if value.get(key) is not True:
                    failures.append(
                        _failure(number, planned, producer_hash, key, True, value.get(key))
                    )
    return rows, sources, failures


def build_artifact(
    root: Path,
    publication: dict[str, Any],
    receipts: list[dict[str, Any]] | None = None,
    spans: list[dict[str, Any]] | None = None,
    duration_s: float = 0.0,
) -> dict[str, Any]:
    """Keep scientific, validation, and old publication gates separate."""
    root = root.resolve()
    source = authority(root)
    rows, sources, failures = account(root, source["tasks"])
    if not source["comparison"]["passed"]:
        failures.append(
            _failure(
                7753,
                str(DESIGN),
                source["hashes"][str(DESIGN)],
                "table_json_yaml_match",
                True,
                source["comparison"]["errors"],
            )
        )
    for receipt in receipts or []:
        if receipt.get("passed") is not True:
            failures.append(
                _failure(
                    7766,
                    str(receipt.get("log_path")),
                    receipt.get("log_sha256"),
                    f"validation.{receipt.get('name')}.exit_code",
                    0,
                    receipt.get("exit_code"),
                )
            )
    checks_passed = bool(receipts) and all(r.get("passed") is True for r in receipts)
    validity = source["comparison"]["passed"] and checks_passed
    verdict_class = "disqualified" if not validity else "blocked" if failures else "null"
    honest_verdict = (
        "complete_disqualified_v675_capstone_validation"
        if not validity
        else "complete_blocked_required_v675_evidence"
        if failures
        else "complete_null_v675_scientific_accounting"
    )
    rows[-1].update(
        verdict_class=verdict_class,
        honest_verdict=honest_verdict,
        raw_metrics={"capstone_complete_score": int(validity and not failures)},
    )
    dispositions = [
        {
            "experiment_id": row["experiment_id"],
            "task_id": row["task_id"],
            "availability": row["availability"],
            "producer_path": row["producer_path"],
            "producer_hash": row["producer_hash"],
            "pre_gate_receipt_path": row["pre_gate_receipt_path"],
            "pre_gate_receipt_hash": row["pre_gate_receipt_hash"],
            "verdict_class": row["verdict_class"],
            "honest_verdict": row["honest_verdict"],
            "producer_eligible": row["producer_eligible"],
            "minimum_changed_prerequisite": (
                "none for this accounting slot"
                if row["experiment_id"] == 7766
                else "qualify declared producer; receipt is queue evidence only"
                if row["availability"] == "pre_gate_receipt"
                else "produce declared scientific artifact"
                if row["availability"] == "absent"
                else "repair failed registered validation and rerun exact producer"
                if not row["producer_eligible"]
                else "none"
            ),
        }
        for row in rows
    ]
    audit = rows[7762 - 7753]
    static = (
        audit["producer_eligible"]
        and audit["raw_metrics"].get("independent_static_eligible") is True
    )
    online = (
        audit["producer_eligible"]
        and audit["raw_metrics"].get("independent_online_eligible") is True
    )
    arc = rows[7764 - 7753]["producer_eligible"]
    service = rows[7765 - 7753]["producer_eligible"]
    service_value = json.loads((root / rows[7765 - 7753]["producer_path"]).read_text())
    hardware = service_value.get("hardware_continuity", {})
    gates = {
        "validity": validity,
        "readiness": 0 if failures or not validity else 1,
        "probability_quality": None,
        "decision_benefit": None,
        "retention": None,
        "efficiency": None,
    }
    artifact: dict[str, Any] = {
        "schema": "carnot.exp7766.v675.capstone.v1",
        "experiment_id": 7766,
        "milestone": "2026.09.675",
        "run_date": "20260927",
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "rows": rows,
        "task_dispositions": dispositions,
        "acceptance_gate_results": gates,
        "sample_size_budget": {
            "intended": 14,
            "eligible": sum(r["producer_eligible"] for r in rows),
            "started": sum(r["availability"] == "producer" for r in rows),
            "completed": sum(r["verdict_class"] is not None for r in rows),
            "excluded": sum(
                r["availability"] == "producer" and not r["producer_eligible"] for r in rows
            ),
            "censored": sum(r["censored"] for r in rows),
            "effective_independent_N": None,
            "independent_families": None,
            "independent_games": None,
        },
        "source_artifact_hashes": sources
        + [
            {
                "upstream_id": "V675 authority",
                "path": path,
                "sha256": digest,
                "date": "20260927",
                "imported_fields": ["exact contract bytes"],
                "eligible": True,
            }
            for path, digest in source["hashes"].items()
        ],
        "preconditions_checked": {
            "root": str(root),
            "authority_candidates": source["candidates"],
            "authority_match": source["comparison"]["passed"],
            "authority_errors": source["comparison"]["errors"],
            "output_parent_exists": (root / OUTPUT.parent).is_dir(),
            "backend": "codex",
            "experimental_model_load": False,
            "resource_observation": "host aggregation; no board or LLM invocation",
        },
        "validation_receipts": {
            "frozen_scope_path": str(RAW / "frozen_affected_scope.json"),
            "frozen_scope_sha256": sha256_file(root / RAW / "frozen_affected_scope.json"),
            "commands": receipts or [],
            "required_checks_passed": checks_passed,
            "e2e": "SCENARIO-REPORT-7766-REPLAY",
        },
        "verifier_is_oracle": False,
        "claim_scope": {
            "RAGTruth": "exposed_development_only",
            "constructed_truth": "fixture_only",
            "ARC": "adapter_withheld_public",
            "fresh_generalization_eligible": False,
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {
            "loads": 0,
            "forwards": 0,
            "generations": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "failures": 0,
            "cancellations": 0,
            "loaded_files": [],
        },
        "model_invoked": False,
        "random_seed": {"seeds": [], "purpose": "deterministic aggregation"},
        "phase_spans": spans or [],
        "duration_s": duration_s,
        "capstone_complete_score": int(validity and not failures),
        "prd_gap_findings": {
            "verification": {
                "qualified": static,
                "independent_static_eligible": audit["raw_metrics"].get(
                    "independent_static_eligible"
                ),
                "probability_quality": None,
                "decision_benefit": None,
                "source": audit["producer_path"],
            },
            "retained_learning": {
                "qualified": online,
                "independent_online_eligible": audit["raw_metrics"].get(
                    "independent_online_eligible"
                ),
                "retention": None,
                "source": audit["producer_path"],
            },
            "live_path_efficiency": {
                "qualified": arc and service,
                "arc_live_path_eligible": arc,
                "service_cost_eligible": service,
                "source_paths": [rows[11]["producer_path"], rows[12]["producer_path"]],
            },
            "qwen_diagnostic": {
                "producer_eligible": rows[6]["producer_eligible"],
                "rescues_learned_head": False,
            },
        },
        "domain_results": {
            "ARC": {"availability": rows[11]["availability"], "effect": None},
            "service": {"availability": rows[12]["availability"], "cost_eligible": service},
            "qwen": {"verdict_class": rows[6]["verdict_class"], "diagnostic_only": True},
        },
        "hardware_continuity": hardware,
        "hardware_speedup_claimed": False,
        "publication_gate_results": publication,
        "established_FoVer_AUROC": 0.9131,
        "publication_evidence_disposition": "older_headline_only_draft_no_submission",
        "publication_performed": False,
        "production_defaults_changed": False,
        "generator_training_performed": False,
        "retirement_applied": [
            d["task_id"]
            for d, task in zip(dispositions[:-1], source["tasks"][:-1])
            if any(
                prior.get("retire_if_same_verdict") is True
                and prior.get("verdict") == d["honest_verdict"]
                for prior in task.get("prior_failures", [])
            )
        ],
        "next_question": "Resolve the existing full-suite collection errors, then rerun the first failed V675 producer validation before restaging this science.",
    }
    artifact["reproducibility_checksum"] = canonical_hash(
        {
            "authority": source["hashes"],
            "sources": sources,
            "code": {str(p): sha256_file(ROOT / p) for p in (MODULE, CLI)},
            "roles": [t["id"] for t in source["tasks"]],
            "parameters": {"aggregation": True, "model_specs": [], "seeds": []},
        }
    )
    artifact["field_principles"] = {
        key: PRINCIPLES.get(key, "Source evidence bounds this field.") for key in artifact
    }
    artifact["field_principles"]["acceptance_gates"] = {
        key: PRINCIPLES["acceptance_gate_results"] for key in gates
    }
    return artifact


def cold_replay(value: dict[str, Any], root: Path) -> list[str]:
    """Reopen source bytes and compare immutable conclusions in a fresh process."""
    expected = build_artifact(
        root,
        value["publication_gate_results"],
        value["validation_receipts"]["commands"],
        value["phase_spans"],
        value["duration_s"],
    )
    keys = (
        "rows",
        "task_dispositions",
        "source_artifact_hashes",
        "gate_check_summary",
        "sample_size_budget",
        "acceptance_gate_results",
        "prd_gap_findings",
        "domain_results",
        "hardware_continuity",
        "capstone_complete_score",
        "reproducibility_checksum",
        "honest_verdict",
        "verdict_class",
    )
    return [key for key in keys if canonical_hash(value.get(key)) != canonical_hash(expected[key])]
