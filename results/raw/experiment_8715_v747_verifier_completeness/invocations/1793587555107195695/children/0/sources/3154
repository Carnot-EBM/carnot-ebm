"""Certify V674 planning bytes without claiming scientific benefit.

REQ-REPORT-7739 and REQ-HARNESS-7739.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

import yaml

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)

ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.09.674"
RUN_DATE = "20260927"
DESIGN = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
RESULT = Path("results/experiment_7739_v674_contract_methods.json")
RAW = Path("results/raw/experiment_7739_v674_contract_methods")
METHOD = Path("docs/research-notes/v674-method-map.md")
MODULE = Path("python/carnot/experiment_7739_v674_contract_methods.py")
TEST = Path("tests/python/test_experiment_7739_v674_contract_methods.py")
CLI = Path("scripts/experiments/experiment_7739_v674_contract_methods.py")
FIELDS = (
    "id",
    "title",
    "phase",
    "deliverable",
    "inference_substrate_class",
    "MODEL_SPECS",
    "gated_on",
)


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Make long-running planning checks visible with measured elapsed time."""

    print(
        f"[exp7739] {phase} {event} elapsed_s={time.monotonic() - start:.3f} completed_units={units}",
        flush=True,
    )


def operand(
    check: str, upstream: str, path: str, field: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Keep a failed prerequisite specific enough to repair without guessing."""

    return {
        "check": check,
        "upstream": upstream,
        "artifact_path": path,
        "field": field,
        "op": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def resolve_authority(root: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    """Use only a staged or active roadmap whose milestone matches V674."""

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
    raise ValueError("matching V674 authority missing")


def parse_design(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Read the literal table and machine block independently of the roadmap."""

    body = text.split("## Exact Task Contract", 1)
    if len(body) != 2:
        return [], [], ["design_section_missing"]
    table = []
    errors = []
    for line in body[1].splitlines():
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
    block = re.search(r"```json\s*(.*?)\s*```", body[1], re.S)
    if block is None:
        return [], table, [*errors, "design_json_missing"]
    try:
        machine = json.loads(block.group(1))
        if machine.get("milestone") != MILESTONE or not isinstance(machine.get("tasks"), list):
            raise ValueError("wrong design milestone or tasks")
        return machine["tasks"], table, errors
    except (ValueError, AttributeError, TypeError):
        return [], table, [*errors, "design_json_invalid"]


def compare_contract(text: str, roadmap: dict[str, Any]) -> dict[str, Any]:
    """Compare each explicit field without letting one source fill another's gaps."""

    machine, table, errors = parse_design(text)
    tasks = roadmap.get("tasks", [])
    if roadmap.get("milestone") != MILESTONE:
        errors.append("roadmap_milestone")
    if (len(machine), len(table), len(tasks)) != (14, 14, 14):
        errors.append("task_count")
    hashes = {"design": canonical_hash(text), "roadmap": canonical_hash(roadmap)}
    rows = []
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
        checks["order"] = shown.get("order") == index + 1
        checks["sequence"] = str(actual.get("id", "")).startswith(f"exp{7739 + index}-")
        checks["milestone"] = actual.get("milestone") == MILESTONE
        for gate in actual.get("gated_on") or []:
            producer = next(
                (item for item in tasks[:index] if item.get("id") == gate.get("upstream")), None
            )
            checks.setdefault("producer_precedes", True)
            checks["producer_precedes"] &= producer is not None
            checks.setdefault("producer_field", True)
            checks["producer_field"] &= bool(
                producer and gate.get("artifact_field") in producer.get("prompt", "")
            )
            checks.setdefault("gate_keys", True)
            checks["gate_keys"] &= set(gate) == {"upstream", "artifact_field", "op", "value"}
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


def mutate_roadmap(roadmap: dict[str, Any], mutation: str) -> dict[str, Any]:
    """Test a defect on private bytes while leaving the selected authority alone."""

    changed = deepcopy(roadmap)
    tasks = changed["tasks"]
    if mutation == "delete":
        tasks.pop()
    elif mutation == "reorder":
        tasks[0], tasks[1] = tasks[1], tasks[0]
    elif mutation == "milestone":
        tasks[0]["milestone"] = "2026.09.673"
    elif mutation == "path":
        tasks[0]["deliverable"] = "results/stale.json"
    elif mutation == "model":
        next(item for item in tasks if item["MODEL_SPECS"])["MODEL_SPECS"] = []
    elif mutation == "gate_field":
        next(item for item in tasks if item.get("gated_on"))["gated_on"][0]["artifact_field"] = (
            "misspelled_ready"
        )
    else:
        raise ValueError(f"unknown mutation: {mutation}")
    return changed


def reduce_raw(path: Path, text: str, roadmap: dict[str, Any]) -> dict[str, Any]:
    """Recompute all rows so a changed raw metric cannot pass on its summary."""

    comparison = compare_contract(text, roadmap)
    if json.loads(path.read_text()) != comparison["rows"]:
        raise ValueError("raw rows differ from independent reduction")
    return comparison


def input_inventory(root: Path, authority: Path) -> list[dict[str, Any]]:
    """Keep current inputs and disqualified history in distinct custody roles."""

    current = [
        authority.relative_to(root),
        DESIGN,
        METHOD,
        MODULE,
        TEST,
        CLI,
        Path("research-references.md"),
        Path("research-studying.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        Path("openspec/capabilities/research-reporting/spec.md"),
        Path("openspec/capabilities/research-harnesses/spec.md"),
        Path("python/carnot/models/gibbs/__init__.py"),
    ]
    historical = [
        Path("results/experiment_7726_v673_contract_methods.json"),
        Path("openspec/change-proposals/research-roadmap-v673-preserved-20260926.md"),
    ]
    rows = []
    for path in (*current, *historical):
        exists = (root / path).is_file()
        rows.append(
            {
                "path": str(path),
                "role": "current_input" if path in current else "historical_disqualified",
                "exists": exists,
                "sha256": sha256_file(root / path) if exists else None,
            }
        )
    rows.append(
        {"path": str(RESULT), "role": "planned_output_not_input", "exists": False, "sha256": None}
    )
    return rows


def preconditions(
    root: Path, inventory: list[dict[str, Any]], comparison: dict[str, Any]
) -> list[dict[str, Any]]:
    """Fail before measurement when a required byte source or schema is absent."""

    checks = [
        operand(
            "required_input",
            row["path"],
            row["path"],
            "readable_nonempty_bytes",
            True,
            bool(row["exists"] and (root / row["path"]).stat().st_size),
        )
        for row in inventory
        if row["role"] != "planned_output_not_input"
    ]
    checks.extend(
        [
            operand(
                "design_schema",
                str(DESIGN),
                str(DESIGN),
                "table_tasks",
                14,
                comparison["table_count"],
            ),
            operand(
                "design_schema",
                str(DESIGN),
                str(DESIGN),
                "json_tasks",
                14,
                comparison["machine_count"],
            ),
            operand(
                "owned_resource",
                "host",
                str(root),
                "absolute_directory",
                True,
                root.is_absolute() and root.is_dir(),
            ),
            operand(
                "owned_resource",
                "process",
                f"/proc/{os.getpid()}",
                "exists",
                True,
                Path(f"/proc/{os.getpid()}").is_dir(),
            ),
        ]
    )
    return checks


def checksum(value: dict[str, Any]) -> str:
    """Bind stable fields without asking an artifact to hash itself."""

    return canonical_hash(
        {key: item for key, item in value.items() if key != "reproducibility_checksum"}
    )


def span(
    name: str, started: float, phase_start: float, units: int, checkpoint: str | None
) -> dict[str, Any]:
    """Measure a disjoint phase interval and its completed work."""

    end = time.monotonic()
    return {
        "phase": name,
        "started_offset_s": phase_start - started,
        "ended_offset_s": end - started,
        "duration_s": end - phase_start,
        "run_date": RUN_DATE,
        "completed_units": units,
        "heartbeat_times": [end - started],
        "checkpoint_sha256": checkpoint,
    }


def build_artifact(
    root: Path,
    authority: Path,
    comparison: dict[str, Any],
    raw: Path,
    receipts: list[dict[str, Any]],
    spans: list[dict[str, Any]],
) -> dict[str, Any]:
    """Carry administrative evidence while leaving science gates unmeasured."""

    inventory = input_inventory(root, authority)
    inventory.append(
        {
            "path": str(raw),
            "role": "raw_evidence",
            "exists": raw.is_file(),
            "sha256": sha256_file(raw) if raw.is_file() else None,
        }
    )
    checks = preconditions(root, inventory, comparison)
    blocked = [row for row in checks if not row["passed"]]
    validated = bool(receipts) and all(row.get("passed") is True for row in receipts)
    ready = comparison["passed"] and not blocked and validated
    if blocked:
        verdict, verdict_class = "complete_blocked_v674_required_input", "blocked"
    elif ready:
        verdict, verdict_class = "complete_null_v674_contract_methods", "null"
    else:
        verdict, verdict_class = "complete_disqualified_v674_contract_validation", "disqualified"
    old_path = root / "results/experiment_7726_v673_contract_methods.json"
    old = json.loads(old_path.read_text()) if old_path.is_file() else {}
    gates = {
        key: None for key in ("probability_quality", "decision_benefit", "retention", "efficiency")
    }
    gates.update({"validity": ready, "readiness": None})
    value: dict[str, Any] = {
        "schema": "carnot.exp7739.v674.contract_methods.v1",
        "experiment_id": "exp7739-contract-methods",
        "experiment": 7739,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": "complete",
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": any(
            row.get("name") == "adversarial_verify" and not row.get("passed") for row in receipts
        ),
        "positive_claim": False,
        "gate_check_summary": blocked,
        "acceptance_gate_results": gates,
        "contract_ready_score": int(ready),
        "rows": deepcopy(comparison["rows"]),
        "contract_comparison": deepcopy(comparison),
        "sample_size_budget": {
            "intended": 14,
            "started": 14,
            "completed": len(comparison["rows"]),
            "eligible": sum(row["matched"] for row in comparison["rows"]),
            "excluded": sum(not row["matched"] for row in comparison["rows"]),
            "censored": 0,
            "effective_independent_n": 14,
            "unit": "administrative task; no source families or games sampled",
        },
        "claim_scope": {
            "kind": "fixture_only",
            "fresh_generalization_eligible": False,
            "development_only": "exposed RAGTruth remains development-only",
            "adapter_withheld_public": "ARC comparison remains unmeasured",
        },
        "inference_substrate": "aggregation_from_planning_and_historical_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": {
            f"{kind}_{state}": 0
            for kind in ("loads", "forwards", "generations", "input_tokens", "output_tokens")
            for state in ("attempted", "completed", "failed", "cancelled")
        },
        "execution_venue": "host",
        "execution_venue_details": {
            "owned_pid": os.getpid(),
            "host": os.uname().nodename,
            "gpu_used": False,
            "gpu_uuid": None,
        },
        "phase_spans": deepcopy(spans),
        "duration_s": sum(item["duration_s"] for item in spans),
        "random_seed": {
            "value": 7739,
            "purpose": "deterministic private contract mutations; no sampling",
        },
        "source_artifact_hashes": inventory,
        "source_artifact_categories": {
            "eligible_producers": [],
            "historical_disqualified_sources": [str(old_path.relative_to(root))],
            "missing_inputs": [
                row["path"]
                for row in inventory
                if not row["exists"] and row["role"] != "planned_output_not_input"
            ],
            "pre_gate_receipts": [],
        },
        "preconditions_checked": checks,
        "validation_receipts": deepcopy(receipts),
        "verifier_is_oracle": False,
        "method_map_path": str(METHOD),
        "selected_roadmap_path": str(authority.relative_to(root)),
        "prior_v673_verdict_preserved": old.get("honest_verdict"),
        "archived_v673_design_sha256": sha256_file(
            root / "openspec/change-proposals/research-roadmap-v673-preserved-20260926.md"
        ),
        "execution_backend": {
            "requested": yaml.safe_load(authority.read_text())["tasks"][0].get("agent_type"),
            "effective": "codex"
            if os.getenv("CODEX_THREAD_ID")
            else "unknown_from_process_environment",
            "evidence": "CODEX_THREAD_ID presence in owned process",
        },
        "affected_file_validation_manifest": {
            "test_paths": [str(TEST)],
            "changed_modules": [str(MODULE)],
            "static_paths": [str(CLI)],
            "frozen_before_validation": True,
        },
        "repository_suite_debt": {
            "status": "recorded_separately",
            "required_for_admin_readiness": False,
        },
    }
    value["field_principles"] = {
        key: "Measured evidence bounds this field's claim and downstream use."
        for key in (*value, "field_principles", "reproducibility_checksum")
    }
    value["field_principles"].update(
        {
            "honest_verdict": "Completion and scientific benefit are different facts.",
            "verdict_class": "Unchanged external failures must not trigger repeated attempts.",
            "rows": "Every comparison must be independently recomputable.",
            "contract_ready_score": "Administrative completeness cannot manufacture science.",
            "validation_receipts": "A result is usable only after registered checks pass.",
            "method_map_path": "A paper must change a test or be explicitly deferred.",
        }
    )
    value["field_principles"].update(
        {f"acceptance_gate_results.{key}": "Unmeasured science stays null." for key in gates}
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def validate_candidate(value: dict[str, Any], root: Path, raw: Path) -> bool:
    """Cold-reduce rows and reject every changed referenced input byte."""

    if value.get("reproducibility_checksum") != checksum(value):
        return False
    authority = root / str(value.get("selected_roadmap_path"))
    if not authority.is_file() or not (root / DESIGN).is_file() or not raw.is_file():
        return False
    try:
        comparison = reduce_raw(
            raw, (root / DESIGN).read_text(), yaml.safe_load(authority.read_text())
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if value.get("rows") != comparison["rows"] or value.get("contract_comparison") != comparison:
        return False
    for row in value.get("source_artifact_hashes", []):
        if row["role"] == "planned_output_not_input":
            if row["sha256"] is not None:
                return False
            continue
        path = root / row["path"]
        if path.is_file() != row["exists"]:
            return False
        if path.is_file() and sha256_file(path) != row["sha256"]:
            return False
    return True


def validation_plan(root: Path, authority: Path, private: Path) -> list[CommandSpec]:
    """Freeze affected checks and the existing planning readers before running."""

    from carnot.experiment_7573_v662_contract_methods import build_repository_check_plan

    base = private / "basetemp"
    base.mkdir(parents=True, exist_ok=True)
    coverage = private / "coverage/.coverage"
    coverage.parent.mkdir(parents=True, exist_ok=True)
    scoped = build_scoped_commands(
        root,
        [str(TEST)],
        [str(MODULE)],
        static_paths=[str(CLI)],
        basetemp=base,
        coverage_file=coverage,
    )
    python = str(root / ".venv/bin/python")
    prompt = (
        "import json,sys;from pathlib import Path;"
        "from carnot.experiment_7643_v667_contract_methods import prompt_path_findings;"
        "v=prompt_path_findings(Path(sys.argv[1]),Path(sys.argv[2]));"
        "print(json.dumps(v),flush=True);sys.exit(bool(v))"
    )
    return [
        *scoped,
        *build_repository_check_plan(root, authority),
        CommandSpec(
            "arc_orphan_solver",
            (python, "-u", "scripts/arc_orphan_solver_lint.py"),
            "live_arc_path",
            900,
        ),
        CommandSpec(
            "prompt_path",
            (python, "-u", "-c", prompt, str(root), str(authority)),
            "selected_authority",
            900,
        ),
    ]


def terminal_plan(root: Path, candidate: Path, raw: Path) -> list[CommandSpec]:
    """Use fresh processes to inspect the candidate and its raw rows."""

    python = str(root / ".venv/bin/python")
    return [
        CommandSpec(
            "cold_cli_replay",
            (python, "-u", str(CLI), "--cold-validate", str(candidate), "--raw", str(raw)),
            "candidate",
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "candidate",
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "candidate",
        ),
    ]


def run_experiment(root: Path, run_date: str, output: Path) -> dict[str, Any]:
    """Validate registered checks, cold-replay, then publish one terminal JSON."""

    started = time.monotonic()
    progress(started, "preconditions", "before")
    if root.resolve() != ROOT or run_date != RUN_DATE or output != RESULT:
        raise ValueError("root, date, or result path differs from declared task")
    authority, roadmap, candidates = resolve_authority(root)
    design = (root / DESIGN).read_text()
    comparison = compare_contract(design, roadmap)
    inventory = input_inventory(root, authority)
    checks = preconditions(root, inventory, comparison)
    phases = [span("preconditions", started, started, len(checks), sha256_file(authority))]
    progress(started, "preconditions", "after", len(checks))
    raw_dir = root / RAW
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = raw_dir / "rows.json"
    atomic_json(raw, comparison["rows"])
    mutation_rows = [
        {
            "mutation": name,
            "rejected": not compare_contract(design, mutate_roadmap(roadmap, name))["passed"],
        }
        for name in ("delete", "reorder", "milestone", "path", "model", "gate_field")
    ]
    atomic_json(raw_dir / "mutations.json", mutation_rows)
    if not all(row["rejected"] for row in mutation_rows):
        raise RuntimeError("private mutation escaped contract reader")
    private = Path(tempfile.mkdtemp(prefix="exp7739-", dir="/tmp"))
    plan = validation_plan(root, authority, private)
    atomic_json(
        raw_dir / "validation_scope.json",
        {
            "authority": str(authority.relative_to(root)),
            "test_paths": [str(TEST)],
            "changed_modules": [str(MODULE)],
            "static_paths": [str(CLI)],
            "commands": [list(item.argv) for item in plan],
            "frozen_before_validation": True,
        },
    )
    progress(started, "validation", "before", 0)
    phase_start = time.monotonic()
    receipts = run_commands(root, plan, log_dir=raw_dir / "validation_logs", heartbeat_s=60)
    phases.append(span("validation", started, phase_start, len(receipts), canonical_hash(receipts)))
    progress(started, "validation", "after", len(receipts))
    candidate = private / "candidate.json"
    value = build_artifact(root, authority, comparison, raw, receipts, phases)
    value["roadmap_resolution_candidates"] = candidates
    value["contract_mutation_rows"] = mutation_rows
    value["reproducibility_checksum"] = checksum(value)
    atomic_json(candidate, value)
    if not validate_candidate(value, root, raw):
        raise RuntimeError("candidate failed cold reduction")
    progress(started, "terminal_readers", "before")
    phase_start = time.monotonic()
    terminal = run_commands(
        root, terminal_plan(root, candidate, raw), log_dir=raw_dir / "terminal_logs", heartbeat_s=60
    )
    phases.append(
        span("terminal_readers", started, phase_start, len(terminal), canonical_hash(terminal))
    )
    progress(started, "terminal_readers", "after", len(terminal))
    value = build_artifact(root, authority, comparison, raw, [*receipts, *terminal], phases)
    value["roadmap_resolution_candidates"] = candidates
    value["contract_mutation_rows"] = mutation_rows
    if not all(row["passed"] for row in terminal):
        value["honest_verdict"] = "complete_disqualified_v674_terminal_validation"
        value["verdict_class"] = "disqualified"
        value["contract_ready_score"] = 0
        value["acceptance_gate_results"]["validity"] = False
    value["reproducibility_checksum"] = checksum(value)
    atomic_json(candidate, value)
    if not validate_candidate(value, root, raw):
        raise RuntimeError("terminal candidate failed cold reduction")
    progress(started, "exact_terminal_replay", "before")
    exact = run_commands(
        root,
        terminal_plan(root, candidate, raw),
        log_dir=raw_dir / "exact_terminal_logs",
        heartbeat_s=60,
    )
    progress(started, "exact_terminal_replay", "after", len(exact))
    if [row["passed"] for row in terminal] != [row["passed"] for row in exact]:
        raise RuntimeError("exact terminal reader outcomes changed")
    atomic_json(raw_dir / "exact_reader_receipts.json", exact)
    atomic_json(root / output, value)
    progress(started, "publication", "after", 14)
    return value


def main(argv: list[str] | None = None) -> int:
    """Expose execution and a fresh-process raw-custody reader."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--cold-validate")
    parser.add_argument("--raw")
    parser.add_argument("--fixture-root")
    args = parser.parse_args(argv)
    if args.cold_validate:
        try:
            value = json.loads(Path(args.cold_validate).read_text())
            valid = validate_candidate(
                value,
                Path(args.fixture_root).resolve() if args.fixture_root else ROOT,
                Path(args.raw),
            )
            print(json.dumps({"valid": valid, "rows": len(value.get("rows", []))}), flush=True)
            return 0 if valid else 1
        except (OSError, ValueError, TypeError, KeyError):
            return 1
    run_experiment(ROOT, args.date, RESULT)
    return 0
