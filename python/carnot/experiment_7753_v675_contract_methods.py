"""Bind V675 planning sources while keeping scientific evidence unpromoted.

REQ-REPORT-7753; SCENARIO-REPORT-7753-CONTRACT/CUSTODY/TERMINAL.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

import yaml

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file

ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.09.675"
DESIGN = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
PRESERVED = Path("openspec/change-proposals/research-roadmap-v674-preserved-20260927.md")
RESULT = Path("results/experiment_7753_v675_contract_methods.json")
RAW = Path("results/raw/experiment_7753_v675_contract_methods")
METHOD = Path("docs/research-notes/v675-method-map.md")
MODULE = Path("python/carnot/experiment_7753_v675_contract_methods.py")
TEST = Path("tests/python/test_experiment_7753_v675_contract_methods.py")
CLI = Path("scripts/experiments/experiment_7753_v675_contract_methods.py")
FIELDS = (
    "id",
    "title",
    "phase",
    "deliverable",
    "inference_substrate_class",
    "MODEL_SPECS",
    "gated_on",
)


def resolve_authority(root: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    """Select a staged roadmap when present and matching, else matching active."""
    candidates = []
    for name in ("research-roadmap-next.yaml", "research-roadmap.yaml"):
        path = root / name
        value = yaml.safe_load(path.read_text()) if path.is_file() else None
        milestone = value.get("milestone") if isinstance(value, dict) else None
        candidates.append({"path": name, "exists": path.is_file(), "milestone": milestone})
        if milestone == MILESTONE:
            from scripts.roadmap_schema import Roadmap

            Roadmap.model_validate(value)
            return path, value, candidates
    raise ValueError("matching V675 authority missing")


def parse_design(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Parse the visible table and JSON independently, preserving errors."""
    parts = text.split("## Exact Task Contract", 1)
    if len(parts) != 2:
        return [], [], ["design_section_missing"]
    table: list[dict[str, Any]] = []
    errors: list[str] = []
    for line in parts[1].splitlines():
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if len(cells) != 8 or not cells[0].isdigit():
            continue
        try:
            table.append(
                {
                    "order": int(cells[0]),
                    "id": cells[1],
                    "phase": int(cells[2]),
                    "title": cells[3],
                    "deliverable": cells[4],
                    "inference_substrate_class": cells[5],
                    "MODEL_SPECS": json.loads(cells[6]),
                    "gated_on": json.loads(cells[7]),
                }
            )
        except (ValueError, TypeError):
            errors.append("design_table_invalid")
    block = re.search(r"```json\s*(.*?)\s*```", parts[1], re.S)
    if block is None:
        return [], table, [*errors, "design_json_missing"]
    try:
        machine = json.loads(block.group(1))
        if machine.get("milestone") != MILESTONE or not isinstance(machine.get("tasks"), list):
            raise ValueError("wrong machine contract")
        return machine["tasks"], table, errors
    except (ValueError, TypeError, AttributeError):
        return [], table, [*errors, "design_json_invalid"]


def compare_contract(text: str, roadmap: dict[str, Any]) -> dict[str, Any]:
    """Compare every registered task field without deriving one authority from another."""
    machine, table, errors = parse_design(text)
    tasks = roadmap.get("tasks", [])
    if roadmap.get("milestone") != MILESTONE:
        errors.append("roadmap_milestone")
    if (len(machine), len(table), len(tasks)) != (14, 14, 14):
        errors.append("task_count")
    rows = []
    hashes = {"design": canonical_hash(text), "roadmap": canonical_hash(roadmap)}
    for index in range(14):
        expected = machine[index] if index < len(machine) else {}
        shown = table[index] if index < len(table) else {}
        actual = tasks[index] if index < len(tasks) else {}
        checks = {
            field: field in expected
            and field in shown
            and (field in actual or field == "gated_on")
            and actual.get(field, [] if field == "gated_on" else None)
            == expected[field]
            == shown[field]
            for field in FIELDS
        }
        checks["order"] = expected.get("order") == shown.get("order") == index + 1
        checks["sequence"] = str(actual.get("id", "")).startswith(f"exp{7753 + index}-")
        checks["milestone"] = actual.get("milestone") == MILESTONE
        prior = actual.get("prior_failures") or []
        checks["prior_fields"] = bool(prior) and all(
            set(item) >= {"experiment_id", "verdict", "addressed_by", "retire_if_same_verdict"}
            and bool(item["experiment_id"] and item["verdict"] and item["addressed_by"])
            for item in prior
        )
        checks["retirement"] = bool(prior) and all(
            item.get("retire_if_same_verdict") is True for item in prior
        )
        checks["producer_precedes"] = all(
            any(task.get("id") == gate.get("upstream") for task in tasks[:index])
            for gate in actual.get("gated_on") or []
        )
        checks["producer_field"] = all(
            any(
                task.get("id") == gate.get("upstream")
                and gate.get("artifact_field") in task.get("prompt", "")
                for task in tasks[:index]
            )
            for gate in actual.get("gated_on") or []
        )
        checks["gate_keys"] = all(
            set(gate) == {"upstream", "artifact_field", "op", "value"}
            for gate in actual.get("gated_on") or []
        )
        matched = all(checks.values())
        rows.append(
            {
                "unit_id": actual.get("id", f"missing-{index + 1}"),
                "order": index + 1,
                "arm": "three_source_contract",
                "checks": checks,
                "matched": matched,
                "raw_numerator": sum(checks.values()),
                "raw_denominator": len(checks),
                "absolute_metric": int(matched),
                "censored": False,
                "exclusions": [] if matched else ["contract_mismatch"],
                "raw_paths": [str(DESIGN), "selected_roadmap"],
                "input_hashes": hashes,
                "effective_independent_groups": 1,
            }
        )
    if not all(row["matched"] for row in rows):
        errors.append("row_mismatch")
    return {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "rows": rows,
        "table_count": len(table),
        "machine_count": len(machine),
        "roadmap_count": len(tasks),
    }


def mutate(roadmap: dict[str, Any], mutation: str) -> dict[str, Any]:
    """Change private task bytes to prove that each contract check can fail."""
    value = deepcopy(roadmap)
    tasks = value["tasks"]
    if mutation == "delete":
        tasks.pop()
    elif mutation == "reorder":
        tasks[0], tasks[1] = tasks[1], tasks[0]
    elif mutation == "title":
        tasks[0]["title"] = "Unregistered title"
    elif mutation == "producer_field":
        gate = next(task for task in tasks if task.get("gated_on"))["gated_on"][0]
        gate["artifact_field"] = "missing_ready_score"
    elif mutation == "substrate":
        tasks[6]["inference_substrate_class"] = "no_model_load"
    elif mutation == "prior_field":
        del tasks[0]["prior_failures"][0]["addressed_by"]
    elif mutation == "retirement":
        tasks[0]["prior_failures"][0]["retire_if_same_verdict"] = False
    else:
        raise ValueError(f"unknown mutation: {mutation}")
    return value


def cold_reduce(raw: Path, design: str, roadmap: dict[str, Any]) -> dict[str, Any]:
    """Recompute rows from independent authorities in the current process."""
    comparison = compare_contract(design, roadmap)
    if json.loads(raw.read_text()) != comparison["rows"]:
        raise ValueError("raw rows differ from independent reduction")
    return comparison


def v674_inventory(root: Path, design: str) -> list[dict[str, Any]]:
    """Keep absent scientific producers separate from conductor queue receipts."""
    section = design.split("## Exact Task Contract", 1)[1]
    block = re.search(r"```json\s*(.*?)\s*```", section, re.S)
    if block is None:
        raise ValueError("V674 machine contract missing")
    tasks = json.loads(block.group(1))["tasks"]
    if [task["id"].split("-")[0] for task in tasks] != [
        f"exp{index}" for index in range(7739, 7753)
    ]:
        raise ValueError("V674 historical task sequence changed")
    rows = []
    for number, task in enumerate(tasks, 7739):
        producer_path = Path(task["deliverable"])
        producer = root / producer_path
        receipt_path = (
            root
            / f"results/experiment_{number}_{task['id'].split('-', 1)[1].replace('-', '_')}.json"
        )
        receipt = receipt_path if receipt_path.is_file() and not producer.is_file() else None
        value = json.loads(producer.read_text()) if producer.is_file() else {}
        verdict_class = value.get("verdict_class")
        state = (
            "missing"
            if not producer.is_file()
            else "disqualified"
            if verdict_class == "disqualified"
            else "measured_null"
            if verdict_class == "null"
            else "blocked"
            if verdict_class == "blocked"
            else "circular_positive"
            if verdict_class == "circular_positive"
            else "other"
        )
        rows.append(
            {
                "experiment": number,
                "task_id": task["id"],
                "producer_path": str(producer_path),
                "producer_sha256": sha256_file(producer) if producer.is_file() else None,
                "producer_state": state,
                "verdict_class": verdict_class,
                "honest_verdict": value.get("honest_verdict"),
                "pre_gate_path": str(receipt.relative_to(root)) if receipt else None,
                "pre_gate_sha256": sha256_file(receipt) if receipt else None,
                "pre_gate_verdict": json.loads(receipt.read_text()).get("honest_verdict")
                if receipt
                else None,
            }
        )
    return rows


def input_inventory(root: Path, authority: Path) -> list[dict[str, Any]]:
    """Hash current read-first inputs and label future producers without opening them."""
    current = [
        authority.relative_to(root),
        DESIGN,
        PRESERVED,
        METHOD,
        MODULE,
        TEST,
        CLI,
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        Path("scripts/experiment_template.py"),
        Path("python/carnot/reporting/current_work_receipt.py"),
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        Path("openspec/capabilities/research-reporting/spec.md"),
        Path("research-references.md"),
        Path("research-complete.yaml"),
        Path("ops/conductor-log.md"),
        Path("scripts/roadmap_schema.py"),
        Path("scripts/audit_roadmap_gates.py"),
        Path("scripts/exclusion_manifest_lint.py"),
        Path("scripts/harness_fit_lint.py"),
        Path("python/carnot/experiment_7739_v674_contract_methods.py"),
        Path("results/experiment_7752_v674_capstone.json"),
        RAW / "frozen_affected_scope.json",
    ]
    rows = []
    for path in current:
        actual = root / path
        rows.append(
            {
                "path": str(path),
                "role": "current_input",
                "exists": actual.is_file(),
                "sha256": sha256_file(actual) if actual.is_file() else None,
                "date": "2026-09-27",
                "imported_fields": ["bytes"],
                "eligible": actual.is_file(),
            }
        )
    rows.append(
        {
            "path": "research-roadmap-next.yaml",
            "role": "absent_staging_candidate",
            "exists": (root / "research-roadmap-next.yaml").is_file(),
            "sha256": None,
            "date": "2026-09-27",
            "imported_fields": [],
            "eligible": False,
        }
    )
    return rows


def failed_checks(
    comparison: dict[str, Any], inventory: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Give every failed current check an operator and observed operand."""
    failures = []
    for row in inventory:
        if row["role"] == "current_input" and not row["exists"]:
            failures.append(
                {
                    "upstream_id": row["path"],
                    "artifact_path": row["path"],
                    "artifact_sha256": None,
                    "field": "exists",
                    "op": "==",
                    "expected": True,
                    "observed": False,
                }
            )
    for row in comparison["rows"]:
        for field, passed in row["checks"].items():
            if not passed:
                failures.append(
                    {
                        "upstream_id": row["unit_id"],
                        "artifact_path": str(DESIGN),
                        "artifact_sha256": next(
                            (i["sha256"] for i in inventory if i["path"] == str(DESIGN)), None
                        ),
                        "field": field,
                        "op": "==",
                        "expected": True,
                        "observed": False,
                    }
                )
    return failures


def checksum(value: dict[str, Any]) -> str:
    """Hash exact inputs, roles, rows and parameters without self-reference."""
    return canonical_hash(
        {key: item for key, item in value.items() if key != "reproducibility_checksum"}
    )


def cold_validate(value: dict[str, Any], root: Path, raw: Path) -> bool:
    """Reopen source bytes and independently reduce the exact raw row file."""
    if value.get("reproducibility_checksum") != checksum(value):
        return False
    authority = root / str(value.get("selected_roadmap_path"))
    if not authority.is_file() or not (root / DESIGN).is_file() or not raw.is_file():
        return False
    try:
        comparison = cold_reduce(
            raw, (root / DESIGN).read_text(), yaml.safe_load(authority.read_text())
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if value.get("rows") != comparison["rows"] or value.get("contract_comparison") != comparison:
        return False
    for row in value.get("source_artifact_hashes", []):
        if row["role"] != "current_input":
            continue
        path = root / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            return False
    return True


def build_artifact(
    root: Path,
    authority: Path,
    comparison: dict[str, Any],
    raw: Path,
    receipts: list[dict[str, Any]],
    spans: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a terminal administrative result without claiming scientific benefit."""
    inventory = [dict(row) for row in input_inventory(root, authority)]
    inventory.append(
        {
            "path": str(raw.relative_to(root)),
            "role": "raw_evidence",
            "exists": raw.is_file(),
            "sha256": sha256_file(raw) if raw.is_file() else None,
            "date": "2026-09-27",
            "imported_fields": ["rows"],
            "eligible": raw.is_file(),
        }
    )
    failures = failed_checks(comparison, inventory)
    validated = bool(receipts) and all(item.get("passed") is True for item in receipts)
    ready = comparison["passed"] and not failures and validated
    if failures:
        verdict, verdict_class = "complete_blocked_v675_contract_inputs", "blocked"
    elif ready:
        verdict, verdict_class = "complete_null_v675_contract_methods", "null"
    else:
        verdict, verdict_class = "complete_disqualified_v675_contract_validation", "disqualified"
    historical = v674_inventory(root, (root / PRESERVED).read_text())
    gates = {
        key: None for key in ("probability_quality", "decision_benefit", "retention", "efficiency")
    }
    gates.update({"validity": ready, "readiness": ready})
    value: dict[str, Any] = {
        "schema": "carnot.exp7753.v675.contract_methods.v1",
        "experiment_id": 7753,
        "milestone": MILESTONE,
        "run_date": "20260927",
        "status": "complete",
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": any(
            item.get("name") == "adversarial_verify" and item.get("passed") is not True
            for item in receipts
        ),
        "gate_check_summary": failures,
        "positive_claim": False,
        "rows": comparison["rows"],
        "contract_comparison": comparison,
        "acceptance_gate_results": gates,
        "contract_ready_score": int(ready),
        "sample_size_budget": {
            "intended": 14,
            "eligible": sum(r["matched"] for r in comparison["rows"]),
            "started": 14,
            "completed": len(comparison["rows"]),
            "excluded": sum(not r["matched"] for r in comparison["rows"]),
            "censored": 0,
            "effective_independent_n": 14,
            "unit": "administrative task",
        },
        "source_artifact_hashes": inventory,
        "v674_producer_inventory": historical,
        "source_artifact_categories": {
            "missing_scientific_producers": [
                r["producer_path"] for r in historical if r["producer_state"] == "missing"
            ],
            "pre_gate_receipts": [r["pre_gate_path"] for r in historical if r["pre_gate_path"]],
            "disqualified": [
                r["producer_path"] for r in historical if r["producer_state"] == "disqualified"
            ],
            "measured_null": [
                r["producer_path"] for r in historical if r["producer_state"] == "measured_null"
            ],
            "future_dependencies": [
                task["deliverable"] for task in yaml.safe_load(authority.read_text())["tasks"][1:]
            ],
        },
        "preconditions_checked": {
            "authority_candidates": candidates,
            "absolute_root": str(root.resolve()),
            "owned_output_parent_exists": (root / RESULT).parent.is_dir(),
            "process_exists": Path(f"/proc/{__import__('os').getpid()}").is_dir(),
            "backend": "host_cpu_aggregation",
            "gpu_used": False,
            "required_inputs": [
                {"path": item["path"], "exists": item["exists"], "sha256": item["sha256"]}
                for item in inventory
                if item["role"] == "current_input"
            ],
        },
        "validation_receipts": receipts,
        "affected_file_validation_manifest": {
            "path": str(RAW / "frozen_affected_scope.json"),
            "sha256": sha256_file(root / RAW / "frozen_affected_scope.json"),
            "frozen_before_implementation": True,
        },
        "verifier_is_oracle": False,
        "claim_scope": {
            "kind": "administrative_contract",
            "fresh_generalization_eligible": False,
            "science": "unmeasured",
            "RAGTruth": "development_only",
            "ARC": "adapter_withheld_public_unmeasured",
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "model_specs": [],
        "planned_MODEL_SPECS": [],
        "model_invocation_counts": {
            "loads": 0,
            "generations": 0,
            "forwards": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "loaded_files": [],
        },
        "model_invoked": False,
        "random_seed": 7753,
        "phase_spans": spans,
        "duration_s": sum(span["duration_s"] for span in spans),
        "method_map_path": str(METHOD),
        "selected_roadmap_path": str(authority.relative_to(root)),
        "preserved_v674_design_sha256": sha256_file(root / PRESERVED),
        "publication_gate_scope": "stable_FoVer_G1_G4_separate_from_new_science",
    }
    value["field_principles"] = {
        key: "Exact measured evidence limits downstream use of this field." for key in value
    }
    value["field_principles"].update(
        {
            "honest_verdict": "Terminal records do not retry unchanged external blocks.",
            "verdict_class": "The claim class travels with evidence.",
            "rows": "Aggregates must be recomputable from each task row.",
            "contract_ready_score": "Readiness certifies executable instructions, not science.",
            "source_artifact_hashes": "A missing producer cannot be replaced by an old result.",
            "method_map_path": "External work informs a falsifiable local question.",
            "phase_spans": "Duration is measured without padding.",
            "MODEL_SPECS": "An upstream model citation is not a current invocation.",
            "gate_check_summary": "Missing producers and failed thresholds have different causes.",
        }
    )
    value["field_principles"]["acceptance_gates"] = {
        key: "A working protocol is not evidence of benefit; unmeasured values stay null."
        for key in gates
    }
    value["reproducibility_checksum"] = checksum(value)
    return value
