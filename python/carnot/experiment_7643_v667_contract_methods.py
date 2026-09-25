"""Aggregate the V667 contract and literal V666 evidence without model inference.

The comparison measures custody of plans and old outcomes. It cannot measure
whether the proposed source witness improves a forecast or a decision.

Spec: REQ-REPORT-7643 and SCENARIO-REPORT-7643-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

import yaml

from carnot.experiment_7329_v644_contract import parse_markdown_contract
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)


ROOT = Path(__file__).resolve().parents[2]
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
RESULT_PATH = Path("results/experiment_7643_v667_contract_methods.json")
PRIOR_PATH = Path("results/experiment_7642_v666_capstone.json")
NOTE_PATH = Path("docs/research-notes/v667-method-map.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
MODULE_PATH = Path("python/carnot/experiment_7643_v667_contract_methods.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7643_v667_contract_methods.py")
TEST_PATH = Path("tests/python/test_experiment_7643_v667_contract_methods.py")
MILESTONE = "2026.09.667"
RUN_DATE = "20260925"
MODEL_SPECS: list[str] = []


def progress(started: float, phase: str, event: str) -> None:
    """Show each boundary so a long child process has visible ownership."""

    print(f"[exp7643] {phase} {event} elapsed_s={time.monotonic() - started:.3f}", flush=True)


def resolve_authority(root: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    """Use staged bytes when present, while rejecting a stale staged authority."""

    candidates = []
    for name in ("research-roadmap-next.yaml", "research-roadmap.yaml"):
        path = root / name
        value = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
        milestone = value.get("milestone") if isinstance(value, dict) else None
        candidates.append({"path": name, "exists": path.is_file(), "observed": milestone})
        if path.is_file():
            if milestone != MILESTONE:
                raise ValueError(f"stale roadmap authority: {name}")
            return path, value, candidates
    raise ValueError("V667 roadmap authority is unavailable")


def _machine_contract(text: str) -> list[dict[str, Any]]:
    """Parse the fenced machine list independently of executable YAML."""

    match = re.search(r"<!-- V667-TASK-CONTRACT-BEGIN -->\s*```json\s*(.*?)\s*```", text, re.S)
    if match is None:
        raise ValueError("V667 machine contract is missing")
    value = json.loads(match.group(1))
    if not isinstance(value, list):
        raise ValueError("V667 machine contract must be a list")
    return value


def compare_authorities(text: str, roadmap: dict[str, Any]) -> dict[str, Any]:
    """Require table, machine block, and executable roadmap to agree exactly."""

    machine = _machine_contract(text)
    table = parse_markdown_contract(text)
    tasks = roadmap.get("tasks", [])
    errors = []
    if table["milestone"] != MILESTONE or roadmap.get("milestone") != MILESTONE:
        errors.append("milestone")
    if len(machine) != 14 or len(table["tasks"]) != 14 or len(tasks) != 14:
        errors.append("task_count")
    rows = []
    for index in range(14):
        expected = machine[index] if index < len(machine) else {}
        displayed = table["tasks"][index] if index < len(table["tasks"]) else {}
        observed = tasks[index] if index < len(tasks) else {}
        task_id = expected.get("id", f"missing-{index}")
        expected_fields = {
            "id": expected.get("id"),
            "title": expected.get("title"),
            "phase": expected.get("phase"),
            "deliverable": expected.get("deliverable"),
            "inference_substrate_class": expected.get("inference_substrate_class"),
            "MODEL_SPECS": expected.get("MODEL_SPECS"),
            "gated_on": expected.get("gated_on"),
        }
        checks = {
            field: (observed.get(field, []) if field == "gated_on" else observed.get(field))
            == value
            for field, value in expected_fields.items()
        }
        checks["id_sequence"] = task_id.startswith(f"exp{7643 + index}-")
        checks["table"] = all(
            displayed.get(name) == expected_value
            for name, expected_value in (
                ("order", index + 1),
                ("id", expected.get("id")),
                ("title", expected.get("title")),
                ("phase", expected.get("phase")),
                ("deliverable", expected.get("deliverable")),
                ("substrate", expected.get("inference_substrate_class")),
                ("gates", expected.get("gated_on")),
            )
        )
        checks["no_self_input"] = (
            not any(gate.get("upstream") == task_id for gate in observed.get("gated_on", []))
            and observed.get("deliverable") != RESULT_PATH.as_posix()
            if index
            else True
        )
        matched = all(checks.values())
        rows.append(
            {
                "unit_id": task_id,
                "arm": "design_vs_authority",
                "order": index + 1,
                "expected": expected_fields,
                "observed": {
                    key: observed.get(key, []) if key == "gated_on" else observed.get(key)
                    for key in expected_fields
                },
                "checks": checks,
                "matched": matched,
                "absolute_metric": int(matched),
                "raw_numerator": sum(checks.values()),
                "raw_denominator": len(checks),
                "provenance": [DESIGN_PATH.as_posix(), "selected_roadmap"],
                "exclusions": [],
                "censored": False,
                "seed": None,
            }
        )
    if not all(row["matched"] for row in rows):
        errors.append("row_mismatch")
    return {"passed": not errors, "errors": errors, "rows": rows}


def mutate_authority(roadmap: dict[str, Any], name: str) -> dict[str, Any]:
    """Change one private copy to prove the comparator is fail closed."""

    changed = deepcopy(roadmap)
    tasks = changed["tasks"]
    if name == "missing_task":
        tasks.pop()
    elif name == "reordered_task":
        tasks[0], tasks[1] = tasks[1], tasks[0]
    elif name == "changed_producer":
        tasks[1]["deliverable"] = "results/private-corruption.json"
    elif name == "self_input":
        tasks[1]["gated_on"] = [
            {
                "upstream": tasks[1]["id"],
                "artifact_field": "flagged_adversarial",
                "op": "==",
                "value": False,
            }
        ]
    elif name == "changed_gate":
        next(task for task in tasks if task.get("gated_on"))["gated_on"][0]["artifact_field"] = (
            "wrong"
        )
    elif name == "changed_model":
        next(task for task in tasks if task["MODEL_SPECS"])["MODEL_SPECS"] = []
    else:
        raise ValueError(f"unknown authority mutation: {name}")
    return changed


def collect_prior_dispositions(root: Path) -> list[dict[str, Any]]:
    """Authenticate producer and pre-gate files while preserving absent paths."""

    capstone = json.loads((root / PRIOR_PATH).read_text(encoding="utf-8"))
    source = capstone["milestone_dispositions"]
    if len(source) != 14:
        raise ValueError("V666 disposition count is not fourteen")
    rows = []
    for index, original in enumerate(source):
        row = deepcopy(original)
        if row["order"] != index + 1 or not row["task_id"].startswith(f"exp{7629 + index}-"):
            raise ValueError("V666 disposition order changed")
        kind = row["custody_kind"]
        label = row.get("actual_path") or row.get("planned_path")
        path = root / (PRIOR_PATH if kind == "current_self" else Path(label))
        exists = path.is_file()
        actual_hash = sha256_file(path) if exists else None
        expected_hash = (
            sha256_file(root / PRIOR_PATH) if kind == "current_self" else row.get("sha256")
        )
        row["authentication_path"] = path.relative_to(root).as_posix()
        row["authentication_sha256"] = actual_hash
        row["authenticated"] = (
            kind == "missing_work" and not exists and expected_hash is None
        ) or (exists and actual_hash == expected_hash)
        rows.append(row)
    return rows


def input_check(
    check: str,
    upstream: str,
    field: str,
    operator: str,
    expected: object,
    observed: object,
) -> dict[str, Any]:
    """Keep complete operands so an upstream absence is actionable."""

    passed = expected == observed if operator in {"eq", "exists"} else expected != observed
    return {
        "check": check,
        "upstream": upstream,
        "path": upstream,
        "field": field,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": passed,
    }


def independent_reduce(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count independent task contracts, never seeds or repeated views."""

    return {"matched": sum(row["matched"] is True for row in rows), "total": len(rows)}


def classify(valid: bool, inputs_present: bool) -> tuple[str, str]:
    """Separate invalid current work from unchanged external absence."""

    if not valid:
        return "complete_disqualified_v667_validation", "disqualified"
    if not inputs_present:
        return "complete_blocked_v667_external_evidence", "blocked"
    return "complete_null_v667_contract_methods", "null"


def _source_hashes(
    root: Path, authority: Path, prior: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Distinguish true inputs, prior producers, pre-gates, and planned output."""

    paths = [
        authority.relative_to(root),
        DESIGN_PATH,
        PRIOR_PATH,
        SPEC_PATH,
        NOTE_PATH,
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        Path("research-references.md"),
        Path("research-studying.md"),
    ]
    rows = [
        {
            "path": item.as_posix(),
            "role": "current_input",
            "exists": True,
            "sha256": sha256_file(root / item),
        }
        for item in paths
    ]
    for row in prior:
        role = (
            "missing_input"
            if row["custody_kind"] == "missing_work"
            else (
                "pre_gate_receipt"
                if row["custody_kind"] == "conductor_pre_gate"
                else "producer_file"
            )
        )
        rows.append(
            {
                "path": row["authentication_path"],
                "role": role,
                "exists": row["custody_kind"] != "missing_work",
                "sha256": row["authentication_sha256"],
            }
        )
    rows.append(
        {
            "path": RESULT_PATH.as_posix(),
            "role": "planned_output_not_input",
            "exists": False,
            "sha256": None,
        }
    )
    return rows


def _preconditions(root: Path, authority: Path) -> list[dict[str, Any]]:
    """Check real inputs and owned CPU process, never the planned output."""

    names = [
        Path("AGENTS.md"),
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        Path("scripts/experiment_template.py"),
        Path("python/carnot/reporting/current_work_receipt.py"),
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        DESIGN_PATH,
        PRIOR_PATH,
        SPEC_PATH,
        NOTE_PATH,
        Path("research-references.md"),
        Path("research-studying.md"),
        authority.relative_to(root),
    ]
    rows = [
        input_check(
            "required_input",
            item.as_posix(),
            "readable_nonempty_bytes",
            "eq",
            True,
            (root / item).is_file() and (root / item).stat().st_size > 0,
        )
        for item in names
    ]
    rows.append(
        input_check(
            "absolute_repository_root",
            str(root),
            "is_absolute_directory",
            "eq",
            True,
            root.is_absolute() and root.is_dir(),
        )
    )
    rows.append(
        input_check(
            "process_resource_ownership",
            f"/proc/{os.getpid()}",
            "owned_pid_exists",
            "eq",
            True,
            Path(f"/proc/{os.getpid()}").is_dir(),
        )
    )
    return rows


def _gate(condition: str, observed: object, passed: bool, principle: str) -> dict[str, Any]:
    """Put the measured operand next to the rule it can actually support."""

    return {"condition": condition, "observed": observed, "passed": passed, "principle": principle}


def _checksum(value: dict[str, Any]) -> str:
    """Bind every published field except the checksum to stable JSON bytes."""

    return canonical_hash(
        {key: item for key, item in value.items() if key != "reproducibility_checksum"}
    )


def build_artifact(
    root: Path,
    authority: Path,
    candidates: list[dict[str, Any]],
    contract: dict[str, Any],
    prior: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
    started: float,
    spans: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reduce readiness from literal rows and current validation receipts."""

    checks = _preconditions(root, authority)
    mutation_rows = [
        {
            "mutation": name,
            "rejected": not compare_authorities(
                (root / DESIGN_PATH).read_text(encoding="utf-8"),
                mutate_authority(yaml.safe_load(authority.read_text(encoding="utf-8")), name),
            )["passed"],
        }
        for name in (
            "missing_task",
            "reordered_task",
            "changed_producer",
            "self_input",
            "changed_gate",
            "changed_model",
        )
    ]
    inputs_present = all(row["passed"] for row in checks) and all(
        row["authenticated"] for row in prior
    )
    required_passed = all(row.get("passed") is True for row in receipts)
    valid = contract["passed"] and all(row["rejected"] for row in mutation_rows)
    valid = valid and required_passed
    verdict, verdict_class = classify(valid, inputs_present)
    counts = Counter(row["custody_kind"] for row in prior)
    readiness = contract["passed"] and inputs_present
    gates = {
        "validity": _gate(
            "exact authority, custody and current checks",
            valid,
            valid,
            "Invalid required validation disqualifies current work.",
        ),
        "readiness": _gate(
            "fourteen authenticated task contracts",
            readiness,
            readiness,
            "Contract readiness is administrative and cannot show benefit.",
        ),
        "probability_benefit": _gate(
            "held-back proper-loss advantage",
            None,
            False,
            "Proper loss requires independent labeled source groups.",
        ),
        "utility": _gate(
            "held-back typed decision value",
            None,
            False,
            "Decision value must use its own frozen cost matrix.",
        ),
        "retention": _gate(
            "delayed update retained after restart",
            None,
            False,
            "An update is not evidence of durable learning.",
        ),
        "freshness": _gate(
            "unexposed source groups and live hidden-game windows",
            False,
            False,
            "Prior exposed artifacts cannot establish fresh gain.",
        ),
    }
    blocked = [row for row in checks if not row["passed"]]
    blocked.extend(
        input_check(
            "prior_authentication",
            row["authentication_path"],
            "sha256",
            "eq",
            row.get("sha256"),
            row["authentication_sha256"],
        )
        for row in prior
        if not row["authenticated"]
    )
    artifact: dict[str, Any] = {
        "schema": "carnot.exp7643.v667.contract_methods.v1",
        "experiment_id": "exp7643-contract-methods",
        "experiment": 7643,
        "title": "Bind fourteen tasks and source-witness method limits",
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": "complete",
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "positive_claim": False,
        "flagged_adversarial": False,
        "gate_check_summary": blocked,
        "acceptance_gate_results": gates,
        "contract_ready_score": int(readiness),
        "rows": deepcopy(contract["rows"]),
        "contract_comparison": deepcopy(contract),
        "contract_mutation_rows": mutation_rows,
        "prior_dispositions": deepcopy(prior),
        "prior_disposition_counts": dict(counts),
        "sample_size_budget": {
            "intended_independent_groups": 14,
            "observed_independent_groups": len(contract["rows"]),
            "eligible": independent_reduce(contract["rows"])["matched"],
            "excluded": 14 - independent_reduce(contract["rows"])["matched"],
            "censored": 0,
            "exposure_limit": "administrative task rows only",
        },
        "preconditions_checked": checks,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": {"loads": 0, "forwards": 0, "generations": 0, "tokens": 0},
        "historical_model_identity": "V666 planned Qwen work did not become current inference",
        "execution_venue": "host",
        "execution_venue_details": {
            "host": os.uname().nodename,
            "owned_pid": os.getpid(),
            "gpu_used": False,
        },
        "phase_spans": deepcopy(spans),
        "duration_s": time.monotonic() - started,
        "random_seed": {"contract_mutations": 7643, "purpose": "deterministic controls only"},
        "source_artifact_hashes": _source_hashes(root, authority, prior),
        "validation_receipts": deepcopy(receipts),
        "terminal_reader_outcomes": [],
        "verifier_is_oracle": False,
        "method_map_path": NOTE_PATH.as_posix(),
        "roadmap_resolution_candidates": candidates,
        "selected_roadmap_path": authority.relative_to(root).as_posix(),
        "affected_file_validation_manifest": {
            "test_paths": [TEST_PATH.as_posix()],
            "changed_modules": [MODULE_PATH.as_posix()],
            "static_paths": [WRAPPER_PATH.as_posix()],
            "spec_paths": [SPEC_PATH.as_posix()],
            "frozen_before_validation": True,
        },
        "publication_gates_unchanged": ["G1", "G2", "G3", "G4"],
        "v666_findings": {
            "gpu_capacity": "complete_blocked_owned_cuda_capacity",
            "arc_validation": "complete_disqualified_planner_goal_guard_validation_failed",
        },
    }
    artifact["field_principles"] = {
        key: "Report the observed scope; do not infer scientific benefit."
        for key in (*artifact, "field_principles", "reproducibility_checksum")
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def build_artifact_for_test(root: Path) -> dict[str, Any]:
    """Build a local pure fixture so tests can exercise cold reduction."""

    authority, roadmap, candidates = resolve_authority(root)
    contract = compare_authorities((root / DESIGN_PATH).read_text(encoding="utf-8"), roadmap)
    prior = collect_prior_dispositions(root)
    return build_artifact(root, authority, candidates, contract, prior, [], time.monotonic(), [])


def validate_artifact(value: dict[str, Any], root: Path) -> bool:
    """Cold-check raw authority rows, immutable bytes, and checksum."""

    if value.get("reproducibility_checksum") != _checksum(value):
        return False
    authority = root / str(value.get("selected_roadmap_path"))
    if not authority.is_file() or not (root / DESIGN_PATH).is_file():
        return False
    roadmap = yaml.safe_load(authority.read_text(encoding="utf-8"))
    expected = compare_authorities((root / DESIGN_PATH).read_text(encoding="utf-8"), roadmap)
    if value.get("rows") != expected["rows"]:
        return False
    if independent_reduce(value["rows"])["matched"] != value.get("sample_size_budget", {}).get(
        "eligible"
    ):
        return False
    for row in value.get("source_artifact_hashes", []):
        if row["role"] == "planned_output_not_input":
            if row.get("sha256") is not None:
                return False
            continue
        path = root / row["path"]
        if row["exists"] != path.is_file():
            return False
        if row["exists"] and sha256_file(path) != row["sha256"]:
            return False
    return len(value.get("prior_dispositions", [])) == 14


def cold_read(path: Path, root: Path) -> bool:
    """Read a candidate from disk so validation cannot trust in-memory values."""

    return validate_artifact(json.loads(path.read_text(encoding="utf-8")), root)


def _span(name: str, started: float, phase_start: float, units: int) -> dict[str, Any]:
    """Record disjoint current-work time and completed checkpoint position."""

    end = time.monotonic()
    return {
        "phase": name,
        "started_offset_s": phase_start - started,
        "ended_offset_s": end - started,
        "duration_s": end - phase_start,
        "planned_units": units,
        "completed_units": units,
        "checkpoint_position": units,
        "pending_operation": None,
    }


def prompt_path_findings(root: Path, authority: Path) -> list[str]:
    """Read prompt paths while recognizing the consumed staging alias."""

    from scripts.harness_consumer_checks import invented_prompt_paths

    findings = invented_prompt_paths(authority.read_text(encoding="utf-8"))
    consumed = (
        authority == root / "research-roadmap.yaml"
        and not (root / "research-roadmap-next.yaml").exists()
    )
    return [path for path in findings if not (consumed and path == "research-roadmap-next.yaml")]


def _validation_plan(root: Path, authority: Path, private: Path) -> list[CommandSpec]:
    """Freeze exact changed files and read-only repository readers."""

    from carnot.experiment_7573_v662_contract_methods import build_repository_check_plan

    base = private / "basetemp"
    base.mkdir(parents=True, exist_ok=True)
    coverage = private / "coverage" / ".coverage"
    coverage.parent.mkdir(parents=True, exist_ok=True)
    scoped = build_scoped_commands(
        root,
        [TEST_PATH.as_posix()],
        [MODULE_PATH.as_posix()],
        static_paths=[WRAPPER_PATH.as_posix()],
        basetemp=base,
        coverage_file=coverage,
    )
    repository = build_repository_check_plan(root, authority)
    python = str(root / ".venv/bin/python")
    prompt_reader = (
        "import json,sys;from pathlib import Path;"
        "from carnot.experiment_7643_v667_contract_methods import prompt_path_findings;"
        "findings=prompt_path_findings(Path(sys.argv[1]),Path(sys.argv[2]));"
        "print(json.dumps({'invented_paths':findings}),flush=True);"
        "sys.exit(1 if findings else 0)"
    )
    prompt = CommandSpec(
        "prompt_path",
        (python, "-u", "-c", prompt_reader, str(root), str(authority)),
        "selected_authority",
        900,
    )
    return [*scoped, *repository, prompt]


def _terminal_plan(root: Path, candidate: Path) -> list[CommandSpec]:
    """Read one frozen candidate in fresh processes with strict safety readers."""

    python = str(root / ".venv/bin/python")
    wrapper = WRAPPER_PATH.as_posix()
    return [
        CommandSpec(
            "cold_replay",
            (python, "-u", wrapper, "--cold-validate", str(candidate)),
            "exact_candidate",
            900,
        ),
        CommandSpec(
            "independent_reduction",
            (python, "-u", wrapper, "--independent-reduce", str(candidate)),
            "exact_candidate",
            900,
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_candidate",
            900,
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_candidate",
            900,
        ),
    ]


def run_experiment(root: Path, run_date: str, output: Path) -> dict[str, Any]:
    """Run bounded current checks, retain exact receipts, and publish atomically."""

    started = time.monotonic()
    progress(started, "preconditions", "before")
    if run_date != RUN_DATE or root.resolve() != ROOT or output != RESULT_PATH:
        raise ValueError("run date, absolute root, or deliverable changed")
    authority, roadmap, candidates = resolve_authority(root)
    contract = compare_authorities((root / DESIGN_PATH).read_text(encoding="utf-8"), roadmap)
    prior = collect_prior_dispositions(root)
    spans = [_span("preconditions", started, started, 14)]
    progress(started, "preconditions", "after")
    for phase in ("model_load", "generation", "benchmark"):
        phase_start = time.monotonic()
        progress(started, phase, "before")
        spans.append(_span(phase, started, phase_start, 0))
        progress(started, phase, "after")
    private = Path(tempfile.mkdtemp(prefix="carnot-exp7643-", dir="/tmp"))
    plan = _validation_plan(root, authority, private)
    atomic_json(
        private / "affected-file-manifest.json",
        {
            "tests": [TEST_PATH.as_posix()],
            "modules": [MODULE_PATH.as_posix()],
            "static": [WRAPPER_PATH.as_posix(), SPEC_PATH.as_posix(), NOTE_PATH.as_posix()],
            "frozen_before_validation": True,
        },
    )
    phase_start = time.monotonic()
    progress(started, "validation", "before")
    receipts = run_commands(root, plan, log_dir=private / "logs")
    for receipt in receipts:
        receipt["worktree"] = str(root)
    spans.append(_span("validation", started, phase_start, len(receipts)))
    progress(started, "validation", "after")
    candidate = build_artifact(
        root, authority, candidates, contract, prior, receipts, started, spans
    )
    candidate_path = private / "candidate.json"
    atomic_json(candidate_path, candidate)
    phase_start = time.monotonic()
    progress(started, "terminal_readers", "before")
    terminal = run_commands(
        root, _terminal_plan(root, candidate_path), log_dir=private / "terminal_logs"
    )
    for receipt in terminal:
        receipt["worktree"] = str(root)
    spans.append(_span("terminal_readers", started, phase_start, len(terminal)))
    progress(started, "terminal_readers", "after")
    final = build_artifact(
        root, authority, candidates, contract, prior, [*receipts, *terminal], started, spans
    )
    final["flagged_adversarial"] = next(
        receipt["passed"] is not True
        for receipt in terminal
        if receipt["name"] == "adversarial_verify"
    )
    final["terminal_reader_outcomes"] = [
        {
            "name": row["name"],
            "outcome": "passed" if row["passed"] else "failed",
            "exit_code": row["exit_code"],
            "log_sha256": row["log_sha256"],
        }
        for row in terminal
    ]
    final["reproducibility_checksum"] = _checksum(final)
    atomic_json(candidate_path, final)
    if not cold_read(candidate_path, root):
        raise RuntimeError("terminal candidate failed cold validation")
    progress(started, "exact_terminal_replay", "before")
    exact = run_commands(
        root, _terminal_plan(root, candidate_path), log_dir=private / "exact_terminal_logs"
    )
    progress(started, "exact_terminal_replay", "after")
    if [row["passed"] for row in exact] != [row["passed"] for row in terminal]:
        raise RuntimeError("exact terminal reader outcomes changed")
    atomic_json(root / RESULT_PATH, final)
    progress(started, "publication", "after")
    return final


def main(argv: list[str] | None = None) -> int:
    """Expose a thin declared run and two read-only fresh-process readers."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", default=RESULT_PATH.as_posix())
    parser.add_argument("--cold-validate", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    args = parser.parse_args(argv)
    if args.cold_validate is not None:
        return 0 if cold_read(args.cold_validate, ROOT) else 1
    if args.independent_reduce is not None:
        candidate = json.loads(args.independent_reduce.read_text(encoding="utf-8"))
        reduced = independent_reduce(candidate["rows"])
        print(json.dumps(reduced, sort_keys=True), flush=True)
        return (
            0
            if reduced == {"matched": 14, "total": 14} and cold_read(args.independent_reduce, ROOT)
            else 1
        )
    run_experiment(ROOT, args.date, Path(args.output))
    return 0
