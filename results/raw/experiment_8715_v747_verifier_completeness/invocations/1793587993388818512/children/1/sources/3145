"""Certify the V673 administrative contract. REQ-REPORT-7726."""

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
MILESTONE = "2026.09.673"
RUN_DATE = "20260926"
DESIGN = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
RESULT = Path("results/experiment_7726_v673_contract_methods.json")
RAW = Path("results/raw/experiment_7726_v673_contract_methods")
MODULE = Path("python/carnot/experiment_7726_v673_contract_methods.py")
TEST = Path("tests/python/test_experiment_7726_v673_contract_methods.py")
CLI = Path("scripts/experiments/experiment_7726_v673_contract_methods.py")
METHOD = Path("docs/research-notes/v673-method-map.md")
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
    """Print each boundary with elapsed time and honest completed units."""

    print(
        f"[exp7726] {phase} {event} elapsed_s={time.monotonic() - start:.3f} completed_units={units}",
        flush=True,
    )


def operand(
    check: str, upstream: str, path: str, field: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Keep every field needed to explain a failed external input."""

    return {
        "check": check,
        "upstream": upstream,
        "artifact_path": path,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def resolve_authority(root: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    """Select only V673 staged bytes or the matching active authority."""

    candidates = []
    selected = None
    for name in ("research-roadmap-next.yaml", "research-roadmap.yaml"):
        path = root / name
        value = yaml.safe_load(path.read_text()) if path.is_file() else None
        observed = value.get("milestone") if isinstance(value, dict) else None
        candidates.append({"path": name, "exists": path.is_file(), "observed_milestone": observed})
        if selected is None and observed == MILESTONE:
            selected = (path, value)
    if selected is None:
        raise ValueError("matching V673 authority missing")
    return selected[0], selected[1], candidates


def parse_design(
    text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]], list[str]]:
    """Read the table, gate table and JSON block as separate sources."""

    section = text.split("## Exact Task Contract", 1)
    body = section[1] if len(section) == 2 else ""
    table = []
    gates = []
    for line in body.splitlines():
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if len(cells) == 5 and cells[0].isdigit():
            table.append(
                dict(
                    order=int(cells[0]),
                    id=cells[1],
                    phase=int(cells[2]),
                    title=cells[3],
                    deliverable=cells[4],
                )
            )
        if len(cells) == 3 and cells[0].startswith("Exp") and cells[1].startswith("Exp"):
            gates.append(dict(consumer=cells[0], producer=cells[1], condition=cells[2]))
    errors = []
    if not table:
        errors.append("design_table_missing")
    if not gates:
        errors.append("design_gate_table_missing")
    block = re.search(r"```json\s*(.*?)\s*```", body, re.S)
    if block is None:
        errors.append("design_json_missing")
        return [], table, gates, errors
    try:
        machine = json.loads(block.group(1))
        if machine.get("milestone") != MILESTONE or not isinstance(machine.get("tasks"), list):
            raise ValueError("wrong design milestone or tasks")
        return machine["tasks"], table, gates, errors
    except (ValueError, AttributeError):
        errors.append("design_json_invalid")
        return [], table, gates, errors


def compare_contract(text: str, roadmap: dict[str, Any]) -> dict[str, Any]:
    """Recompute thirteen independent task rows from all authorities."""

    machine, table, gate_table, errors = parse_design(text)
    tasks = roadmap.get("tasks", [])
    if roadmap.get("milestone") != MILESTONE:
        errors.append("roadmap_milestone")
    if len(machine) != 13 or len(table) != 13 or len(tasks) != 13:
        errors.append("task_count")
    rows = []
    hashes = {"design_text": canonical_hash(text), "selected_roadmap": canonical_hash(roadmap)}
    for index in range(13):
        expected = machine[index] if index < len(machine) else {}
        shown = table[index] if index < len(table) else {}
        actual = tasks[index] if index < len(tasks) else {}
        checks = {
            field: field in expected
            and actual.get(field, [] if field == "gated_on" else None) == expected.get(field)
            for field in FIELDS
        }
        checks.update(
            {
                field + "_table": actual.get(field) == shown.get(field)
                for field in ("id", "title", "phase", "deliverable")
            }
        )
        checks["order"] = shown.get("order") == index + 1
        checks["sequence"] = str(actual.get("id", "")).startswith(f"exp{7726 + index}-")
        checks["milestone"] = actual.get("milestone") == MILESTONE
        for gate in actual.get("gated_on") or []:
            producer = next(
                (task for task in tasks[:index] if task.get("id") == gate.get("upstream")), None
            )
            checks.setdefault("producer_precedes", True)
            checks["producer_precedes"] &= producer is not None
            checks.setdefault("producer_field", True)
            checks["producer_field"] &= bool(
                producer and gate.get("artifact_field") in producer.get("prompt", "")
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
                "input_hashes": hashes,
                "effective_independent_groups": 1,
            }
        )
    expected_gates = [
        (
            f"Exp{int(t['id'][3:7])}",
            f"Exp{int(g['upstream'][3:7])}",
            f"{g['artifact_field']} {g['op']} {json.dumps(g['value'])}",
        )
        for t in machine
        for g in t.get("gated_on", [])
        if g["artifact_field"] not in ("flagged_adversarial", "verdict_class")
    ]
    shown_gates = [(g["consumer"], g["producer"], g["condition"]) for g in gate_table]
    if expected_gates != shown_gates:
        errors.append("gate_table_mismatch")
    if not all(row["matched"] for row in rows):
        errors.append("row_mismatch")
    return {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "rows": rows,
        "table_count": len(table),
        "machine_count": len(machine),
        "gate_table_count": len(gate_table),
    }


def mutate_roadmap(roadmap: dict[str, Any], mutation: str) -> dict[str, Any]:
    """Change one private copy; never write an authority during measurement."""

    changed = deepcopy(roadmap)
    tasks = changed["tasks"]
    if mutation == "delete":
        tasks.pop()
    elif mutation == "reorder":
        tasks[0], tasks[1] = tasks[1], tasks[0]
    elif mutation == "stale":
        changed["milestone"] = "2026.09.672"
    elif mutation == "producer_field":
        next(task for task in tasks if task.get("gated_on"))["gated_on"][0]["artifact_field"] = (
            "wrong_producer_field"
        )
    elif mutation == "model":
        next(task for task in tasks if task["MODEL_SPECS"])["MODEL_SPECS"] = []
    else:
        raise ValueError(f"unknown mutation: {mutation}")
    return changed


def reduce_raw(path: Path, text: str, roadmap: dict[str, Any]) -> dict[str, Any]:
    """Rebuild comparison from source bytes and demand identical raw rows."""

    comparison = compare_contract(text, roadmap)
    if json.loads(path.read_text()) != comparison["rows"]:
        raise ValueError("raw rows differ from independent reduction")
    return comparison


def input_inventory(root: Path, authority: Path, raw: Path) -> list[dict[str, Any]]:
    """Separate actual inputs, old evidence and the planned own output."""

    raw_label = raw.relative_to(root) if raw.is_relative_to(root) else raw
    current = [
        authority.relative_to(root),
        DESIGN,
        METHOD,
        Path("research-references.md"),
        Path("research-studying.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("openspec/capabilities/research-reporting/spec.md"),
        Path("openspec/capabilities/research-harnesses/spec.md"),
        MODULE,
        CLI,
        TEST,
        raw_label,
    ]
    prior = [
        Path("results/experiment_7713_v672_contract_methods.json"),
        Path("results/experiment_7725_v672_capstone.json"),
    ]
    rows = []
    for path in (*current, *prior):
        exists = (root / path).is_file()
        rows.append(
            {
                "path": str(path),
                "role": "blocked_historical_v672" if path in prior else "current_input",
                "exists": exists,
                "sha256": sha256_file(root / path) if exists else None,
            }
        )
    rows.append(
        {"path": str(RESULT), "role": "planned_output_not_input", "exists": False, "sha256": None}
    )
    return rows


def preconditions(
    root: Path, comparison: dict[str, Any], inventory: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Record readable inputs, contract sections, and owned process resources."""

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
                13,
                comparison["table_count"],
            ),
            operand(
                "design_schema",
                str(DESIGN),
                str(DESIGN),
                "json_tasks",
                13,
                comparison["machine_count"],
            ),
            operand(
                "design_schema",
                str(DESIGN),
                str(DESIGN),
                "gate_table",
                7,
                comparison["gate_table_count"],
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
    """Bind all stable candidate fields without self hashing."""

    return canonical_hash(
        {key: item for key, item in value.items() if key != "reproducibility_checksum"}
    )


def build_artifact(
    root: Path,
    authority: Path,
    comparison: dict[str, Any],
    receipts: list[dict[str, Any]],
    raw: Path,
    spans: list[dict[str, Any]],
    started: float,
) -> dict[str, Any]:
    """Report administrative agreement without inventing scientific evidence."""

    inventory = input_inventory(root, authority, raw)
    checks = preconditions(root, comparison, inventory)
    blocked = [row for row in checks if not row["passed"]]
    validation_ok = bool(receipts) and all(row.get("passed") is True for row in receipts)
    ready = comparison["passed"] and not blocked and validation_ok
    if blocked:
        verdict, verdict_class = "complete_blocked_v673_required_input", "blocked"
    elif ready:
        verdict, verdict_class = "complete_null_v673_contract_methods", "null"
    else:
        verdict, verdict_class = "complete_disqualified_v673_contract_validation", "disqualified"
    roadmap = yaml.safe_load(authority.read_text())
    prior = {
        row["path"]: json.loads((root / row["path"]).read_text()).get("honest_verdict")
        for row in inventory
        if row["role"] == "blocked_historical_v672" and row["exists"]
    }
    gates = {
        key: None for key in ("brier_score", "decision_cost", "coverage", "retention", "efficiency")
    }
    gates.update({"validity": ready, "readiness": None})
    result: dict[str, Any] = {
        "schema": "carnot.exp7726.v673.contract_methods.v1",
        "experiment_id": "exp7726-contract-methods",
        "experiment": 7726,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": "complete",
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "positive_claim": False,
        "gate_check_summary": blocked,
        "acceptance_gate_results": gates,
        "contract_ready_score": int(ready),
        "rows": deepcopy(comparison["rows"]),
        "contract_comparison": deepcopy(comparison),
        "sample_size_budget": {
            "intended": 13,
            "observed": len(comparison["rows"]),
            "eligible": sum(row["matched"] for row in comparison["rows"]),
            "excluded": sum(not row["matched"] for row in comparison["rows"]),
            "censored": 0,
            "effective_independent_families": 0,
            "effective_independent_tasks": 13,
            "unit": "administrative task, not source family",
        },
        "claim_scope": {"kind": "fixture_only", "fresh_generalization_eligible": False},
        "inference_substrate": "aggregation_from_planning_and_historical_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_specs": [{"no_model": True}],
        "model_invoked": False,
        "invocation_counts": {
            f"{kind}_{state}": 0
            for kind in ("loads", "forwards", "generations", "input_tokens", "output_tokens")
            for state in ("attempted", "completed", "failed", "cancelled")
        },
        "execution_venue": "host",
        "execution_venue_details": {
            "host": os.uname().nodename,
            "owned_pid": os.getpid(),
            "gpu_used": False,
            "gpu_uuid": None,
        },
        "phase_spans": deepcopy(spans),
        "duration_s": max(0.0, time.monotonic() - started),
        "random_seed": {
            "value": 7726,
            "purpose": "deterministic private contract mutations; no sampling",
        },
        "source_artifact_hashes": inventory,
        "source_artifact_categories": {
            "eligible_producers": [],
            "flagged_historical_inputs": [],
            "pre_gate_receipts": [],
            "absent_required_sources": [
                row["path"]
                for row in inventory
                if not row["exists"] and row["role"] != "planned_output_not_input"
            ],
        },
        "preconditions_checked": checks,
        "validation_receipts": deepcopy(receipts),
        "verifier_is_oracle": False,
        "method_map_path": str(METHOD),
        "selected_roadmap_path": str(authority.relative_to(root)),
        "prior_v672_verdicts_preserved": prior,
        "execution_backend": {
            "requested": roadmap["tasks"][0].get("agent_type"),
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
    result["field_principles"] = {
        key: "Measured evidence bounds the claim and downstream use."
        for key in (*result, "field_principles", "reproducibility_checksum")
    }
    result["field_principles"].update(
        {
            "honest_verdict": "Terminal custody avoids retries of unchanged external failures.",
            "rows": "A comparison must be recomputable without repeating model work.",
            "validation_receipts": "A result is usable only after its registered checks pass.",
        }
    )
    result["field_principles"].update(
        {
            f"acceptance_gate_results.{name}": "Unmeasured science cannot open a readiness gate."
            for name in gates
        }
    )
    result["reproducibility_checksum"] = checksum(result)
    return result


def validate_candidate(value: dict[str, Any], root: Path, raw: Path) -> bool:
    """Cold-check rows, checksum, and exact referenced input bytes."""

    if value.get("reproducibility_checksum") != checksum(value):
        return False
    authority = root / str(value.get("selected_roadmap_path"))
    if not authority.is_file() or not (root / DESIGN).is_file() or not raw.is_file():
        return False
    comparison = reduce_raw(raw, (root / DESIGN).read_text(), yaml.safe_load(authority.read_text()))
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


def span(name: str, started: float, phase_started: float, units: int) -> dict[str, Any]:
    """Record a disjoint measured interval and its completed units."""

    ended = time.monotonic()
    return {
        "phase": name,
        "started_offset_s": phase_started - started,
        "ended_offset_s": ended - started,
        "duration_s": ended - phase_started,
        "run_date": RUN_DATE,
        "completed_units": units,
        "checkpoint_position": units,
        "heartbeat_times": [ended - started],
        "checkpoint_sha256": None,
    }


def validation_plan(root: Path, authority: Path, private: Path) -> list[CommandSpec]:
    """Freeze scoped checks and unchanged repository guards on selected bytes."""

    from carnot.experiment_7573_v662_contract_methods import build_repository_check_plan

    (private / "basetemp").mkdir(parents=True, exist_ok=True)
    (private / "coverage").mkdir(parents=True, exist_ok=True)
    scoped = build_scoped_commands(
        root,
        [str(TEST)],
        [str(MODULE)],
        static_paths=[str(CLI)],
        basetemp=private / "basetemp",
        coverage_file=private / "coverage/.coverage",
    )
    python = str(root / ".venv/bin/python")
    prompt = (
        "import sys;from pathlib import Path;"
        "from carnot.experiment_7643_v667_contract_methods import prompt_path_findings;"
        "v=prompt_path_findings(Path(sys.argv[1]),Path(sys.argv[2]));"
        "print(v,flush=True);sys.exit(bool(v))"
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
    """Read the candidate from independent fresh child processes."""

    python = str(root / ".venv/bin/python")
    wrapper = str(CLI)
    return [
        CommandSpec(
            "cold_cli_replay",
            (python, "-u", wrapper, "--cold-validate", str(candidate), "--raw", str(raw)),
            "candidate",
        ),
        CommandSpec(
            "cold_independent_reduction",
            (python, "-u", wrapper, "--independent-reduce", str(candidate), "--raw", str(raw)),
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
    """Run bounded gates, reduce raw rows and publish only a checked result."""

    started = time.monotonic()
    progress(started, "preconditions", "before")
    if root.resolve() != ROOT or run_date != RUN_DATE or output != RESULT:
        raise ValueError("root, date, or result path differs from the declared task")
    authority, roadmap, candidates = resolve_authority(root)
    design = (root / DESIGN).read_text()
    comparison = compare_contract(design, roadmap)
    spans = [span("preconditions", started, started, 13)]
    spans[0]["checkpoint_sha256"] = sha256_file(authority)
    progress(started, "preconditions", "after", 13)
    for phase in ("model_load", "generation", "benchmark"):
        phase_start = time.monotonic()
        progress(started, phase, "before")
        spans.append(span(phase, started, phase_start, 0))
        progress(started, phase, "after")
    (root / RAW).mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix="run-", dir=root / RAW))
    raw = private / "rows.json"
    atomic_json(raw, comparison["rows"])
    mutation_rows = [
        {
            "mutation": mutation,
            "rejected": not compare_contract(design, mutate_roadmap(roadmap, mutation))["passed"],
        }
        for mutation in ("delete", "reorder", "stale", "producer_field", "model")
    ]
    atomic_json(private / "mutations.json", mutation_rows)
    plan = validation_plan(root, authority, private)
    atomic_json(
        private / "validation_scope.json",
        {
            "authority": str(authority.relative_to(root)),
            "test_paths": [str(TEST)],
            "changed_modules": [str(MODULE)],
            "static_paths": [str(CLI)],
            "commands": [list(item.argv) for item in plan],
            "frozen_before_validation": True,
        },
    )
    phase_start = time.monotonic()
    progress(started, "validation", "before")
    receipts = run_commands(root, plan, log_dir=private / "logs", heartbeat_s=60)
    spans.append(span("validation", started, phase_start, len(receipts)))
    spans[-1]["checkpoint_sha256"] = canonical_hash(receipts)
    progress(started, "validation", "after", len(receipts))
    candidate = private / "candidate.json"
    value = build_artifact(root, authority, comparison, receipts, raw, spans, started)
    value["roadmap_resolution_candidates"] = candidates
    value["contract_mutation_rows"] = mutation_rows
    value["reproducibility_checksum"] = checksum(value)
    atomic_json(candidate, value)
    if not validate_candidate(value, root, raw):
        raise RuntimeError("raw candidate failed independent reduction")
    phase_start = time.monotonic()
    progress(started, "terminal_readers", "before")
    terminal = run_commands(
        root, terminal_plan(root, candidate, raw), log_dir=private / "terminal_logs", heartbeat_s=60
    )
    spans.append(span("terminal_readers", started, phase_start, len(terminal)))
    spans[-1]["checkpoint_sha256"] = canonical_hash(terminal)
    progress(started, "terminal_readers", "after", len(terminal))
    value = build_artifact(root, authority, comparison, [*receipts, *terminal], raw, spans, started)
    value["roadmap_resolution_candidates"] = candidates
    value["contract_mutation_rows"] = mutation_rows
    value["terminal_reader_outcomes"] = terminal
    value["flagged_adversarial"] = any(
        row["name"] == "adversarial_verify" and not row["passed"] for row in terminal
    )
    if not all(row["passed"] for row in terminal):
        value["honest_verdict"] = "complete_disqualified_v673_terminal_validation"
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
        log_dir=private / "exact_terminal_logs",
        heartbeat_s=60,
    )
    progress(started, "exact_terminal_replay", "after", len(exact))
    if [row["passed"] for row in terminal] != [row["passed"] for row in exact]:
        raise RuntimeError("exact terminal reader outcomes changed")
    atomic_json(private / "exact_reader_receipts.json", exact)
    atomic_json(root / output, value)
    progress(started, "publication", "after", 13)
    return value


def main(argv: list[str] | None = None) -> int:
    """Expose the live run and two fresh-process read modes."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--cold-validate")
    parser.add_argument("--independent-reduce")
    parser.add_argument("--raw")
    args = parser.parse_args(argv)
    candidate = args.cold_validate or args.independent_reduce
    if candidate:
        try:
            value = json.loads(Path(candidate).read_text())
            valid = validate_candidate(value, ROOT, Path(args.raw))
            print(json.dumps({"valid": valid, "rows": len(value.get("rows", []))}), flush=True)
            return 0 if valid else 1
        except (OSError, ValueError, TypeError, KeyError):
            return 1
    run_experiment(ROOT, args.date, RESULT)
    return 0
