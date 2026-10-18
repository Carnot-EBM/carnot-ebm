"""Run the V668 contract audit without model inference. REQ-REPORT-7657."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import yaml

from carnot.experiment_7573_v662_contract_methods import build_repository_check_plan
from carnot.experiment_7643_v667_contract_methods import prompt_path_findings
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    reduce_required_checks,
    run_commands,
)
from carnot.reporting import v668_contract as contract

ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = contract.RESULT_PATH
RUN_DATE = "20260925"
MODULE_PATH = Path("python/carnot/experiment_7657_v668_contract_methods.py")
CAPABILITY_PATH = Path("python/carnot/reporting/v668_contract.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7657_v668_contract_methods.py")
TEST_PATH = Path("tests/python/test_experiment_7657_v668_contract_methods.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
NOTE_PATH = Path("docs/research-notes/v668-method-map.md")
MODEL_SPECS: list[str] = []


def progress(started: float, phase: str, event: str) -> None:  # pragma: no cover
    """Print an owned, flushed phase boundary."""

    print(f"[exp7657] {phase} {event} elapsed_s={time.monotonic() - started:.3f}", flush=True)


def checksum(value: dict[str, Any]) -> str:
    """Bind the complete published candidate except its own checksum."""

    return canonical_hash(
        {key: item for key, item in value.items() if key != "reproducibility_checksum"}
    )


def gate(condition: str, observed: object, passed: bool, principle: str) -> dict[str, Any]:
    """Keep each acceptance operand and rule next to its result."""

    return {"condition": condition, "observed": observed, "passed": passed, "principle": principle}


def preconditions(root: Path, authority: Path) -> list[dict[str, Any]]:
    """Check current inputs and CPU ownership, never the planned output."""

    paths = (
        Path("AGENTS.md"),
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        Path("scripts/experiment_template.py"),
        Path("python/carnot/reporting/current_work_receipt.py"),
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        contract.DESIGN_PATH,
        contract.PRIOR_PATH,
        SPEC_PATH,
        NOTE_PATH,
        Path("research-references.md"),
        Path("research-studying.md"),
        authority.relative_to(root),
    )
    checks = [
        contract.input_check(
            "required_input",
            str(path),
            "readable_nonempty_bytes",
            "==",
            True,
            (root / path).is_file() and (root / path).stat().st_size > 0,
        )
        for path in paths
    ]
    checks.append(
        contract.input_check(
            "absolute_repository_root",
            str(root),
            "is_absolute_directory",
            "==",
            True,
            root.is_absolute() and root.is_dir(),
        )
    )
    checks.append(
        contract.input_check(
            "process_resource_ownership",
            f"/proc/{os.getpid()}",
            "owned_pid_exists",
            "==",
            True,
            Path(f"/proc/{os.getpid()}").is_dir(),
        )
    )
    return checks


def validation_plan(root: Path, authority: Path, private: Path) -> list[CommandSpec]:
    """Freeze focused validation and selected-roadmap readers before measurement."""

    basetemp = private / "basetemp"
    coverage = private / "coverage" / ".coverage"
    basetemp.mkdir(parents=True, exist_ok=True)
    coverage.parent.mkdir(parents=True, exist_ok=True)
    scoped = build_scoped_commands(
        root,
        [str(TEST_PATH)],
        [str(MODULE_PATH), str(CAPABILITY_PATH)],
        static_paths=[str(WRAPPER_PATH)],
        basetemp=basetemp,
        coverage_file=coverage,
    )
    python = str(root / ".venv/bin/python")
    prompt_reader = (
        "import sys;from pathlib import Path;"
        "from carnot.experiment_7643_v667_contract_methods import prompt_path_findings;"
        "v=prompt_path_findings(Path(sys.argv[1]),Path(sys.argv[2]));"
        "print(v,flush=True);sys.exit(bool(v))"
    )
    prompt = CommandSpec(
        "prompt_path",
        (python, "-u", "-c", prompt_reader, str(root), str(authority)),
        "selected_authority",
        900,
    )
    return [*scoped, *build_repository_check_plan(root, authority), prompt]


def terminal_plan(root: Path, candidate: Path) -> list[CommandSpec]:
    """Run cold reduction and strict readers on one candidate path."""

    python = str(root / ".venv/bin/python")
    wrapper = str(WRAPPER_PATH)
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


def source_hashes(root: Path, authority: Path, prior: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Distinguish immutable inputs from planned output and absent producers."""

    paths = [
        authority.relative_to(root),
        contract.DESIGN_PATH,
        contract.PRIOR_PATH,
        Path("openspec/change-proposals/research-roadmap-v667-preserved-20260925.md"),
        SPEC_PATH,
        NOTE_PATH,
        MODULE_PATH,
        CAPABILITY_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        Path("research-references.md"),
        Path("research-studying.md"),
    ]
    rows = [
        {
            "path": str(path),
            "role": "current_input",
            "exists": (root / path).is_file(),
            "sha256": sha256_file(root / path) if (root / path).is_file() else None,
        }
        for path in paths
    ]
    for row in prior:
        role = {
            "terminal_producer": "producer_file",
            "current_self": "producer_file",
            "conductor_pre_gate": "pre_gate_receipt",
            "missing_work": "missing_input",
        }[row["custody_kind"]]
        rows.append(
            {
                "path": row["authentication_path"],
                "role": role,
                "exists": role != "missing_input",
                "sha256": row["authentication_sha256"],
            }
        )
    rows.append(
        {
            "path": str(RESULT_PATH),
            "role": "planned_output_not_input",
            "exists": False,
            "sha256": None,
        }
    )
    return rows


def build_artifact(
    root: Path,
    authority: Path,
    candidates: list[dict[str, Any]],
    comparison: dict[str, Any],
    prior: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
    started: float,
    spans: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reduce administrative readiness without creating a scientific result."""

    checks = preconditions(root, authority)
    roadmap = yaml.safe_load(authority.read_text(encoding="utf-8"))
    design = (root / contract.DESIGN_PATH).read_text(encoding="utf-8")
    mutations = [
        {
            "mutation": name,
            "rejected": not contract.compare_authorities(
                design, contract.mutate_authority(roadmap, name)
            )["passed"],
        }
        for name in ("delete", "reorder", "gate", "self_input", "model", "stale")
    ]
    scoped = reduce_required_checks(receipts)
    required = bool(receipts) and all(row.get("passed") is True for row in receipts)
    required = required and scoped["required_checks_passed"]
    valid = comparison["passed"] and all(row["rejected"] for row in mutations) and required
    inputs_present = all(row["passed"] for row in checks) and all(
        row["authenticated"] for row in prior
    )
    verdict, verdict_class = contract.classify(valid, inputs_present)
    readiness = valid and inputs_present
    gates = {
        "validity": gate(
            "exact authority and required current validation",
            valid,
            valid,
            "Invalid required checks disqualify current work.",
        ),
        "readiness": gate(
            "fourteen authenticated contracts after validation",
            readiness,
            readiness,
            "Contract readiness is administrative only.",
        ),
        "coverage": gate(
            "real source predicates covered",
            0,
            False,
            "V667 zero checked predicates cannot establish coverage.",
        ),
        "probability_benefit": gate(
            "held-back proper-loss advantage",
            None,
            False,
            "Independent labels are required for probability value.",
        ),
        "utility": gate(
            "held-back typed decision value",
            None,
            False,
            "A frozen cost matrix and independent groups are required.",
        ),
        "retention": gate(
            "delayed update retained after restart", None, False, "No V667 learning producer ran."
        ),
        "freshness": gate(
            "unexposed source groups",
            False,
            False,
            "Prior exposed groups cannot establish fresh gain.",
        ),
    }
    blocked = [row for row in checks if not row["passed"]]
    blocked += [
        contract.input_check(
            "prior_authentication",
            row["authentication_path"],
            "sha256",
            "==",
            row.get("sha256"),
            row["authentication_sha256"],
        )
        for row in prior
        if not row["authenticated"]
    ]
    artifact: dict[str, Any] = {
        "schema": "carnot.exp7657.v668.contract_methods.v1",
        "experiment_id": "exp7657-contract-methods",
        "experiment": 7657,
        "title": "Bind fourteen tasks and distinguish evidence-format failure from scientific nulls",
        "milestone": contract.MILESTONE,
        "run_date": RUN_DATE,
        "status": "complete",
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "positive_claim": False,
        "flagged_adversarial": False,
        "gate_check_summary": blocked,
        "acceptance_gate_results": gates,
        "contract_ready_score": int(readiness),
        "rows": deepcopy(comparison["rows"]),
        "contract_comparison": deepcopy(comparison),
        "contract_mutation_rows": mutations,
        "prior_dispositions": deepcopy(prior),
        "sample_size_budget": {
            "intended_independent_groups": 14,
            "observed_independent_groups": len(comparison["rows"]),
            "eligible": sum(row["matched"] for row in comparison["rows"]),
            "excluded": sum(not row["matched"] for row in comparison["rows"]),
            "censored": 0,
            "prior_exposure": "administrative V668 task contracts",
            "claim_limit": "No source, probability, decision, retention or live ARC effect measured",
        },
        "preconditions_checked": checks,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_specs": [{"no_model": True}],
        "model_invoked": False,
        "invocation_counts": {
            key: 0
            for key in (
                "loads_attempted",
                "loads_completed",
                "loads_cancelled",
                "forwards_attempted",
                "forwards_completed",
                "forwards_cancelled",
                "generations_attempted",
                "generations_completed",
                "generations_cancelled",
                "tokens_attempted",
                "tokens_completed",
                "tokens_cancelled",
            )
        },
        "historical_model_provenance": "V667 Qwen pilots are historical, not current inference",
        "execution_venue": "host",
        "execution_venue_details": {
            "host": os.uname().nodename,
            "owned_pid": os.getpid(),
            "gpu_uuid": None,
            "gpu_used": False,
        },
        "phase_spans": deepcopy(spans),
        "duration_s": time.monotonic() - started,
        "random_seed": {"authority_mutations": 7657, "purpose": "deterministic controls"},
        "source_artifact_hashes": source_hashes(root, authority, prior),
        "validation_receipts": deepcopy(receipts),
        "terminal_reader_outcomes": [],
        "verifier_is_oracle": False,
        "method_map_path": str(NOTE_PATH),
        "selected_roadmap_path": authority.relative_to(root).as_posix(),
        "roadmap_resolution_candidates": candidates,
        "affected_file_validation_manifest": {
            "test_paths": [str(TEST_PATH)],
            "changed_modules": [str(MODULE_PATH), str(CAPABILITY_PATH)],
            "static_paths": [str(WRAPPER_PATH)],
            "spec_paths": [str(SPEC_PATH)],
            "frozen_before_validation": True,
        },
        "publication_gates_unchanged": ["G1", "G2", "G3", "G4"],
        "v667_findings": {
            "source_grammar_mismatch": "real formats escaped fixed fence and claim grammar",
            "checked_predicates": 0,
            "source_coverage_percent": 39,
            "full_suite_gate_contamination": "Exp7652 and Exp7653 added unrelated global suite failures",
            "arc_inductions_attempted": 6,
            "arc_inductions_accepted": 0,
            "unrun_learning_is_scientific_null": False,
        },
        "repository_suite_debt": {
            "status": "separate_from_required_checks",
            "source": "V667 full-suite receipts",
        },
    }
    artifact["field_principles"] = {
        key: "Report measured current scope; do not infer scientific benefit."
        for key in (*artifact, "field_principles", "reproducibility_checksum")
    }
    artifact["reproducibility_checksum"] = checksum(artifact)
    return artifact


def build_artifact_for_test(root: Path) -> dict[str, Any]:
    """Build a pure fixture with no fabricated validation receipts."""

    authority, roadmap, candidates = contract.resolve_authority(root)
    comparison = contract.compare_authorities(
        (root / contract.DESIGN_PATH).read_text(encoding="utf-8"), roadmap
    )
    prior = contract.collect_prior_dispositions(root)
    return build_artifact(root, authority, candidates, comparison, prior, [], time.monotonic(), [])


def validate_artifact(value: dict[str, Any], root: Path) -> bool:
    """Cold-check immutable inputs, task rows, custody and checksum."""

    if value.get("reproducibility_checksum") != checksum(value):
        return False
    authority = root / str(value.get("selected_roadmap_path"))
    if not authority.is_file() or not (root / contract.DESIGN_PATH).is_file():
        return False
    roadmap = yaml.safe_load(authority.read_text(encoding="utf-8"))
    expected = contract.compare_authorities(
        (root / contract.DESIGN_PATH).read_text(encoding="utf-8"), roadmap
    )
    if value.get("rows") != expected["rows"]:
        return False
    if value.get("sample_size_budget", {}).get("eligible") != sum(
        row["matched"] for row in expected["rows"]
    ):
        return False
    if len(value.get("prior_dispositions", [])) != 14:
        return False
    for row in value.get("source_artifact_hashes", []):
        if row["role"] == "planned_output_not_input":
            if row.get("sha256") is not None:
                return False
            continue
        path = root / row["path"]
        if path.is_file() != row["exists"]:
            return False
        if path.is_file() and sha256_file(path) != row["sha256"]:
            return False
    return True


def cold_read(path: Path, root: Path) -> bool:
    """Reload a disk candidate in a fresh reduction path."""

    return validate_artifact(json.loads(path.read_text(encoding="utf-8")), root)


def span(name: str, started: float, phase_start: float, units: int) -> dict[str, Any]:
    """Record one disjoint monotonic stage and checkpoint position."""

    ended = time.monotonic()
    return {
        "phase": name,
        "started_offset_s": phase_start - started,
        "ended_offset_s": ended - started,
        "duration_s": ended - phase_start,
        "planned_units": units,
        "completed_units": units,
        "checkpoint_position": units,
        "heartbeat_times": [],
        "pending_operation": None,
    }


def run_experiment(root: Path, run_date: str, output: Path) -> dict[str, Any]:
    """Run frozen checks, exact-candidate readers and atomic publication."""

    started = time.monotonic()
    progress(started, "preconditions", "before")
    if run_date != RUN_DATE or root.resolve() != ROOT or output != RESULT_PATH:
        raise ValueError("run date, absolute root, or deliverable changed")
    authority, roadmap, candidates = contract.resolve_authority(root)
    comparison = contract.compare_authorities(
        (root / contract.DESIGN_PATH).read_text(encoding="utf-8"), roadmap
    )
    prior = contract.collect_prior_dispositions(root)
    spans = [span("preconditions", started, started, 14)]
    progress(started, "preconditions", "after")
    for phase in ("model_load", "generation", "benchmark"):
        phase_start = time.monotonic()
        progress(started, phase, "before")
        spans.append(span(phase, started, phase_start, 0))
        progress(started, phase, "after")
    private = Path(tempfile.mkdtemp(prefix="carnot-exp7657-", dir="/tmp"))
    plan = validation_plan(root, authority, private)
    atomic_json(
        private / "affected-file-manifest.json",
        {
            "tests": [str(TEST_PATH)],
            "modules": [str(MODULE_PATH), str(CAPABILITY_PATH)],
            "static": [str(WRAPPER_PATH), str(SPEC_PATH), str(NOTE_PATH)],
            "frozen_before_validation": True,
        },
    )
    phase_start = time.monotonic()
    progress(started, "validation", "before")
    receipts = run_commands(root, plan, log_dir=private / "logs")
    spans.append(span("validation", started, phase_start, len(receipts)))
    progress(started, "validation", "after")
    candidate = build_artifact(
        root, authority, candidates, comparison, prior, receipts, started, spans
    )
    candidate_path = private / "candidate.json"
    atomic_json(candidate_path, candidate)
    phase_start = time.monotonic()
    progress(started, "terminal_readers", "before")
    terminal = run_commands(
        root, terminal_plan(root, candidate_path), log_dir=private / "terminal_logs"
    )
    spans.append(span("terminal_readers", started, phase_start, len(terminal)))
    progress(started, "terminal_readers", "after")
    final = build_artifact(
        root, authority, candidates, comparison, prior, [*receipts, *terminal], started, spans
    )
    final["flagged_adversarial"] = next(
        row["passed"] is not True for row in terminal if row["name"] == "adversarial_verify"
    )
    final["terminal_reader_outcomes"] = [
        {
            "name": row["name"],
            "exit_code": row["exit_code"],
            "log_sha256": row["log_sha256"],
            "passed": row["passed"],
        }
        for row in terminal
    ]
    if final["flagged_adversarial"] or not all(row["passed"] for row in terminal):
        final["honest_verdict"] = "complete_disqualified_v668_terminal_validation"
        final["verdict_class"] = "disqualified"
        final["contract_ready_score"] = 0
        final["acceptance_gate_results"]["readiness"]["passed"] = False
        final["acceptance_gate_results"]["readiness"]["observed"] = False
    final["reproducibility_checksum"] = checksum(final)
    atomic_json(candidate_path, final)
    if not cold_read(candidate_path, root):
        raise RuntimeError("terminal candidate failed cold validation")
    progress(started, "exact_terminal_replay", "before")
    exact = run_commands(
        root, terminal_plan(root, candidate_path), log_dir=private / "exact_terminal_logs"
    )
    progress(started, "exact_terminal_replay", "after")
    if [row["passed"] for row in exact] != [row["passed"] for row in terminal]:
        raise RuntimeError("exact terminal reader outcomes changed")
    atomic_json(root / RESULT_PATH, final)
    progress(started, "publication", "after")
    return final


def main(argv: list[str] | None = None) -> int:
    """Expose the declared run and two read-only fresh-process readers."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", default=str(RESULT_PATH))
    parser.add_argument("--cold-validate")
    parser.add_argument("--independent-reduce")
    args = parser.parse_args(argv)
    if args.cold_validate:
        return 0 if cold_read(Path(args.cold_validate), ROOT) else 1
    if args.independent_reduce:
        value = json.loads(Path(args.independent_reduce).read_text(encoding="utf-8"))
        return (
            0
            if validate_artifact(value, ROOT)
            and sum(row["matched"] for row in value["rows"])
            == value["sample_size_budget"]["eligible"]
            else 1
        )
    run_experiment(ROOT, args.date, Path(args.output))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
