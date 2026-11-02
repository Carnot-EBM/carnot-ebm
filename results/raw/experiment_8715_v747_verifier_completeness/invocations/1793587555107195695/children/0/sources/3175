"""Reconcile V676 source evidence without turning accounting into science.

REQ-REPORT-7780; SCENARIO-REPORT-7780-CUSTODY/GATES/REPLAY.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.experiment_7767_v676_contract_methods import DESIGN, compare_contract, resolve_authority
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path("results/experiment_7780_v676_capstone.json")
RAW = Path("results/raw/experiment_7780_v676_capstone")
MODULE = Path("python/carnot/experiment_7780_v676_capstone.py")
CLI = Path("scripts/experiments/experiment_7780_v676_capstone.py")
TEST = Path("tests/python/test_experiment_7780_v676_capstone.py")
ELIGIBLE = {"null", "positive", "circular_positive"}
PRINCIPLES = {
    "experiment_id": "An artifact must have a unique current owner.",
    "honest_verdict": "A terminal record must not waste attempts on unchanged inputs.",
    "verdict_class": "The claim class travels with the evidence.",
    "flagged_adversarial": "Invalid evidence must not open downstream gates.",
    "gate_check_summary": "Missing producers and failed scientific thresholds are different causes.",
    "rows": "Aggregates must be recomputable without rerunning science.",
    "acceptance_gate_results": "A working protocol is not evidence of benefit.",
    "duration_s": "Duration must describe actual work without padding.",
    "reproducibility_checksum": "A third party needs the same experiment inputs.",
    "sample_size_budget": "Repeated views and seeds do not increase independent family count.",
    "source_artifact_hashes": "A missing producer cannot be replaced with a convenient old result.",
    "preconditions_checked": "Access and validity must be established before expensive work.",
    "validation_receipts": "All registered checks must pass before readiness opens.",
    "verifier_is_oracle": "Execution truth and independent semantic verification are distinct claims.",
    "inference_substrate_class": "Duration floors must match the invoked substrate.",
    "MODEL_SPECS": "A cited upstream model is not a current model invocation.",
    "capstone_complete_score": "Completed bookkeeping cannot substitute for completed science.",
    "task_dispositions": "The design and actual executed roster must agree.",
    "prd_gap_findings": "The milestone must advance or bound the three stated gaps.",
    "publication_gate_results": "Publication readiness keeps its fixed historical definition.",
    "next_question": "Repeated scope requires a reason beyond another version number.",
}


def failure(
    number: int,
    path: str,
    digest: str | None,
    field: str,
    expected: Any,
    observed: Any,
    operator: str = "==",
) -> dict[str, Any]:
    """Preserve one exact failed operand and the bytes it describes."""
    return {
        "upstream_id": f"Exp{number}",
        "artifact_path": path,
        "artifact_hash": digest,
        "field": field,
        "operator": operator,
        "expected": expected,
        "observed": observed,
    }


def authority(root: Path) -> dict[str, Any]:
    """Compare independently written design table, JSON block and active YAML."""
    path, roadmap, candidates = resolve_authority(root)
    comparison = compare_contract((root / DESIGN).read_text(), roadmap)
    labels = [DESIGN.as_posix(), path.relative_to(root).as_posix()]
    return {
        "tasks": roadmap["tasks"],
        "roadmap": roadmap,
        "comparison": comparison,
        "candidates": candidates,
        "design_path": labels[0],
        "roadmap_path": labels[1],
        "hashes": {label: sha256_file(root / label) for label in labels},
    }


def account(
    root: Path, tasks: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Cold-read all declared outcomes, raw files and separate queue receipts."""
    if len(tasks) != 14 or [task["id"].split("-")[0] for task in tasks] != [
        f"exp{n}" for n in range(7767, 7781)
    ]:
        raise ValueError("V676 fourteen-task order required")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    values: dict[str, dict[str, Any]] = {}
    for index, task in enumerate(tasks):
        number = 7767 + index
        planned = task["deliverable"]
        producer = root / planned
        queue_label = (
            f"results/experiment_{number}_{task['id'].split('-', 1)[1].replace('-', '_')}.json"
        )
        queue = root / queue_label
        exists = number != 7780 and producer.is_file()
        has_queue = number != 7780 and queue != producer and queue.is_file()
        value = json.loads(producer.read_text()) if exists else {}
        receipt = json.loads(queue.read_text()) if has_queue else {}
        digest = sha256_file(producer) if exists else None
        queue_hash = sha256_file(queue) if has_queue else None
        state = (
            "planned_output"
            if number == 7780
            else "producer"
            if exists
            else "pre_gate_receipt"
            if has_queue
            else "absent"
        )
        raw_dir = root / planned.removesuffix(".json").replace("results/", "results/raw/")
        raw_paths = (
            sorted(p.relative_to(root).as_posix() for p in raw_dir.rglob("*") if p.is_file())
            if number != 7780 and raw_dir.is_dir()
            else []
        )
        raw_hashes = {label: sha256_file(root / label) for label in raw_paths}
        eligible = (
            exists
            and value.get("verdict_class") in ELIGIBLE
            and value.get("flagged_adversarial") is False
            and str(value.get("honest_verdict", "")).startswith("complete_")
        )
        metrics = {
            key: item
            for key, item in value.items()
            if key.endswith("_score") or key.endswith("_eligible")
        }
        row = {
            "unit_id": task["id"],
            "task_id": task["id"],
            "experiment_id": number,
            "order": index + 1,
            "arm": "task_accounting",
            "producer_path": planned,
            "producer_hash": digest,
            "pre_gate_receipt_path": queue_label if has_queue else None,
            "pre_gate_receipt_hash": queue_hash,
            "pre_gate_schema": receipt.get("schema"),
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
        values[task["id"]] = value
        if number == 7780:
            continue
        sources.append(
            {
                "upstream_id": f"Exp{number}",
                "path": planned,
                "sha256": digest,
                "date": value.get("run_date"),
                "imported_fields": ["verdict_class", "honest_verdict", *metrics],
                "eligible": eligible,
                "pre_gate_path": queue_label if has_queue else None,
                "pre_gate_sha256": queue_hash,
                "raw_paths": raw_hashes,
            }
        )
        if not exists:
            failures.append(failure(number, planned, None, "producer_exists", True, False))
        elif not eligible:
            failures.append(
                failure(
                    number,
                    planned,
                    digest,
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
                failure(
                    number,
                    planned,
                    digest,
                    str(check.get("field", "upstream_gate")),
                    check.get("expected", True),
                    check.get("observed", False),
                    str(check.get("operator", "==")),
                )
            )
    for task in tasks:
        for gate in task.get("gated_on", []):
            upstream = gate["upstream"]
            number = int(upstream.split("-", 1)[0].removeprefix("exp"))
            source = rows[number - 7767]
            observed = values[upstream].get(gate["artifact_field"])
            passed = observed == gate["value"] if gate["op"] == "==" else observed in gate["value"]
            if not passed:
                failures.append(
                    failure(
                        number,
                        source["producer_path"],
                        source["producer_hash"],
                        gate["artifact_field"],
                        gate["value"],
                        observed,
                        gate["op"],
                    )
                )
    return rows, sources, failures


def historical_boundaries(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pin old Qwen and online claims to immutable result bytes, not prose."""
    paths = {
        7759: "results/experiment_7759_v675_qwen_evidence_views.json",
        7760: "results/experiment_7760_v675_online_runner.json",
    }
    old = {n: json.loads((root / path).read_text()) for n, path in paths.items()}
    arms = old[7759]["paired_family_results"]["by_arm"]
    full = old[7760]["validation_receipts"]["full_python_suite"]
    boundary = {
        "Exp7759": {
            "artifact_path": paths[7759],
            "artifact_hash": sha256_file(root / paths[7759]),
            "canonical_parsed": arms["canonical"]["parse_valid"],
            "alternate_parsed": arms["paired"]["parse_valid"],
            "denominator_families": arms["canonical"]["denominator"],
            "claim": "diagnostic_only_low_parse",
        },
        "Exp7760": {
            "artifact_path": paths[7760],
            "artifact_hash": sha256_file(root / paths[7760]),
            "reported_verdict": old[7760].get("honest_verdict"),
            "full_python_suite_exits": [r.get("exit_code") for r in full],
            "spec_mismatch": any(r.get("exit_code") != 0 for r in full),
            "claim": "fixture_only_historical_spec_mismatch",
        },
    }
    sources = [
        {
            "upstream_id": f"Exp{n}",
            "path": path,
            "sha256": sha256_file(root / path),
            "date": old[n].get("run_date"),
            "imported_fields": ["historical_boundary"],
            "eligible": False,
        }
        for n, path in paths.items()
    ]
    return boundary, sources


def build_artifact(
    root: Path,
    publication: dict[str, Any],
    receipts: list[dict[str, Any]] | None = None,
    spans: list[dict[str, Any]] | None = None,
    duration_s: float = 0.0,
) -> dict[str, Any]:
    """Keep own validation separate from missing external scientific evidence."""
    root = root.resolve()
    contract = authority(root)
    rows, sources, failures = account(root, contract["tasks"])
    old, historical_sources = historical_boundaries(root)
    if not contract["comparison"]["passed"]:
        failures.append(
            failure(
                7767,
                str(DESIGN),
                contract["hashes"][str(DESIGN)],
                "table_json_yaml_match",
                True,
                contract["comparison"]["errors"],
            )
        )
    for receipt in receipts or []:
        if receipt.get("passed") is not True:
            failures.append(
                failure(
                    7780,
                    str(receipt.get("log_path")),
                    receipt.get("log_sha256"),
                    f"validation.{receipt.get('name')}.exit_code",
                    0,
                    receipt.get("exit_code"),
                )
            )
    checks_passed = bool(receipts) and all(r.get("passed") is True for r in receipts)
    valid = contract["comparison"]["passed"] and checks_passed
    verdict = "disqualified" if not valid else "blocked" if failures else "null"
    honest = (
        "complete_disqualified_v676_capstone_validation"
        if not valid
        else "complete_blocked_required_v676_evidence"
        if failures
        else "complete_null_v676_scientific_accounting"
    )
    rows[-1].update(
        verdict_class=verdict,
        honest_verdict=honest,
        raw_metrics={"capstone_complete_score": int(valid and not failures)},
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
                "none for current accounting"
                if row["experiment_id"] == 7780
                else "qualify declared producer; queue receipt is separate"
                if row["availability"] == "pre_gate_receipt"
                else "produce declared scientific artifact"
                if row["availability"] == "absent"
                else "repair required producer validation"
                if not row["producer_eligible"]
                else "none"
            ),
        }
        for row in rows
    ]
    audit = rows[7775 - 7767]
    static = (
        audit["producer_eligible"]
        and audit["raw_metrics"].get("independent_static_eligible") is True
    )
    online = (
        audit["producer_eligible"]
        and audit["raw_metrics"].get("independent_online_eligible") is True
    )
    arc = rows[7776 - 7767]["producer_eligible"] and rows[7777 - 7767]["producer_eligible"]
    service = rows[7778 - 7767]["producer_eligible"] and rows[7779 - 7767]["producer_eligible"]
    gates = {
        "validity": valid,
        "readiness": int(valid and not failures),
        "probability_quality": None,
        "decision_benefit": None,
        "retention": None,
        "efficiency": None,
    }
    scope_path = root / RAW / "frozen_affected_scope.json"
    all_sources = (
        sources
        + historical_sources
        + [
            {
                "upstream_id": "V676 authority",
                "path": path,
                "sha256": digest,
                "date": "20260927",
                "imported_fields": ["exact contract bytes"],
                "eligible": True,
            }
            for path, digest in contract["hashes"].items()
        ]
    )
    artifact: dict[str, Any] = {
        "schema": "carnot.exp7780.v676.capstone.v1",
        "experiment_id": 7780,
        "milestone": "2026.09.676",
        "run_date": "20260927",
        "honest_verdict": honest,
        "verdict_class": verdict,
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
        "source_artifact_hashes": all_sources,
        "preconditions_checked": {
            "root": str(root),
            "authority_candidates": contract["candidates"],
            "authority_match": contract["comparison"]["passed"],
            "authority_errors": contract["comparison"]["errors"],
            "output_parent_exists": (root / OUTPUT.parent).is_dir(),
            "backend": "codex",
            "experimental_model_load": False,
            "resource_observation": "host aggregation; no model or board invocation",
        },
        "validation_receipts": {
            "frozen_scope_path": str(RAW / "frozen_affected_scope.json"),
            "frozen_scope_sha256": sha256_file(scope_path),
            "commands": receipts or [],
            "required_checks_passed": checks_passed,
            "repository_collection_healthy": None,
            "e2e": "SCENARIO-REPORT-7780-REPLAY",
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
        }
        | {"loaded_files": []},
        "model_invoked": False,
        "random_seed": {"seeds": [], "purpose": "deterministic aggregation"},
        "phase_spans": spans or [],
        "duration_s": duration_s,
        "capstone_complete_score": int(valid and not failures),
        "historical_boundaries": old,
        "prd_gap_findings": {
            "verification": {
                "qualified": static,
                "probability_quality": None,
                "decision_benefit": None,
                "source": audit["producer_path"],
                "source_eligible": audit["producer_eligible"],
            },
            "retained_learning": {
                "qualified": online,
                "retention": None,
                "source": audit["producer_path"],
                "source_eligible": audit["producer_eligible"],
            },
            "live_path_efficiency": {
                "qualified": arc and service,
                "efficiency": None,
                "source_paths": [
                    rows[10]["producer_path"],
                    rows[11]["producer_path"],
                    rows[12]["producer_path"],
                ],
                "arc_eligible": arc,
                "service_eligible": service,
            },
        },
        "publication_gate_results": publication,
        "established_FoVer_AUROC": 0.9131,
        "publication_evidence_disposition": "older_FoVer_headline_only_draft_no_submission",
        "publication_performed": False,
        "production_defaults_changed": False,
        "roadmap_activated": False,
        "retirement_applied": [
            d["task_id"]
            for d, task in zip(dispositions[:-1], contract["tasks"][:-1])
            if any(
                prior.get("retire_if_same_verdict") is True
                and prior.get("verdict") == d["honest_verdict"]
                for prior in task.get("prior_failures", [])
            )
        ],
        "next_question": "Qualify Exp7768 and Exp7769 affected validation, then produce Exp7771 before retrying decision or learning measurements.",
    }
    artifact["reproducibility_checksum"] = canonical_hash(
        {
            "authority": contract["hashes"],
            "sources": all_sources,
            "code": {str(p): sha256_file(root / p) for p in (MODULE, CLI)},
            "roles": [task["id"] for task in contract["tasks"]],
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
    """Reopen input bytes and independently reduce immutable conclusions."""
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
        "historical_boundaries",
        "capstone_complete_score",
        "reproducibility_checksum",
        "honest_verdict",
        "verdict_class",
    )
    return [key for key in keys if canonical_hash(value.get(key)) != canonical_hash(expected[key])]
