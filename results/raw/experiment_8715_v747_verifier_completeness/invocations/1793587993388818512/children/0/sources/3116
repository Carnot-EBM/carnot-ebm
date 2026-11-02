"""Run the V669 accounting audit without model inference. REQ-REPORT-7671."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import yaml

from carnot.experiment_7573_v662_contract_methods import build_repository_check_plan
from carnot.reporting import v669_contract as contract
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    reduce_required_checks,
    run_commands,
)

ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = contract.RESULT_PATH
RUN_DATE = "20260926"
MODULE_PATH = Path("python/carnot/experiment_7671_v669_contract_methods.py")
CAPABILITY_PATH = Path("python/carnot/reporting/v669_contract.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7671_v669_contract_methods.py")
TEST_PATH = Path("tests/python/test_experiment_7671_v669_contract_methods.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
NOTE_PATH = Path("docs/research-notes/v669-method-map.md")
MODEL_SPECS: list[str] = []


def progress(started: float, phase: str, event: str) -> None:  # pragma: no cover
    """Keep an owned process visibly alive across bounded validation work."""

    print(f"[exp7671] {phase} {event} elapsed_s={time.monotonic() - started:.3f}", flush=True)


def checksum(value: dict[str, Any]) -> str:
    """Bind the candidate and every immutable-input digest except this field."""

    return canonical_hash(
        {key: item for key, item in value.items() if key != "reproducibility_checksum"}
    )


def gate(condition: str, observed: object, passed: bool, principle: str) -> dict[str, Any]:
    """Retain the measured operand beside the rule that interprets it."""

    return {"condition": condition, "observed": observed, "passed": passed, "principle": principle}


def preconditions(root: Path, authority: Path) -> list[dict[str, Any]]:
    """Check current resources and immutable inputs, never the planned output."""

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
        contract.PRIOR_DESIGN_PATH,
        contract.LOG_PATH,
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
    """Freeze affected checks and selected-roadmap readers before reduction."""

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
        (
            python,
            "-u",
            "-c",
            prompt_reader,
            str(root),
            str(authority),
        ),
        "selected_authority",
        900,
    )
    return [*scoped, *build_repository_check_plan(root, authority), prompt]


def terminal_plan(root: Path, candidate: Path) -> list[CommandSpec]:
    """Require two cold readers and both repository adversarial readers."""

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
            (
                python,
                "-u",
                wrapper,
                "--independent-reduce",
                str(candidate),
            ),
            "exact_candidate",
            900,
        ),
        CommandSpec(
            "adversarial_verify",
            (
                python,
                "-u",
                "scripts/adversarial_verify.py",
                str(candidate),
            ),
            "exact_candidate",
            900,
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "exact_candidate",
            900,
        ),
    ]


def source_hashes(root: Path, authority: Path, prior: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Distinguish immutable inputs, prior producers, log custody and output."""

    paths = [
        authority.relative_to(root),
        contract.DESIGN_PATH,
        contract.PRIOR_DESIGN_PATH,
        SPEC_PATH,
        NOTE_PATH,
        MODULE_PATH,
        CAPABILITY_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        Path("research-references.md"),
        Path("research-studying.md"),
        contract.LOG_PATH,
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
    rows.extend(
        {
            "path": row["planned_path"],
            "role": "producer_file" if row["custody_kind"] == "producer_file" else "missing_input",
            "exists": row["custody_kind"] == "producer_file",
            "sha256": row["authentication_sha256"]
            if row["custody_kind"] == "producer_file"
            else None,
        }
        for row in prior
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
    """Reduce contract validity while leaving every science gate closed."""

    checks = preconditions(root, authority)
    roadmap = yaml.safe_load(authority.read_text(encoding="utf-8"))
    design = (root / contract.DESIGN_PATH).read_text(encoding="utf-8")
    mutations = [
        {
            "mutation": name,
            "rejected": not contract.compare_authorities(
                design,
                contract.mutate_authority(roadmap, name),
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
            "exact contract and required current validation",
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
            "source group coverage",
            {"covered": 66, "total": 248},
            False,
            "Prior exposed coverage cannot prove fresh benefit.",
        ),
        "freshness": gate(
            "unexposed source groups", 0, False, "All 248 V668 groups were previously exposed."
        ),
        "probability": gate(
            "held-back proper-loss advantage", None, False, "Independent labels are required."
        ),
        "decision_utility": gate(
            "held-back typed decision value", None, False, "No new decision evaluation ran."
        ),
        "retention": gate(
            "fresh post-restart retained gain",
            None,
            False,
            "V668 learning used exposed groups and found no registered gain.",
        ),
        "efficiency": gate(
            "whole-service speed or cost", None, False, "The V668 cost producer is absent."
        ),
    }
    blocked = [row for row in checks if not row["passed"]]
    blocked.extend(
        contract.input_check(
            "prior_authentication",
            row["authentication_path"],
            "authenticated",
            "==",
            True,
            row["authenticated"],
        )
        for row in prior
        if not row["authenticated"]
    )
    from scripts import publication_gate

    artifact: dict[str, Any] = {
        "schema": "carnot.exp7671.v669.contract_methods.v1",
        "experiment_id": "exp7671-contract-methods",
        "experiment": 7671,
        "title": "Bind V669 contracts and preserve V668 evidence custody",
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
            "prior_exposure": "248 of 248 V668 source groups exposed",
            "effective_blocks": 14,
            "claim_limit": "Task contracts are independent accounting units, not source groups",
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
        "historical_model_provenance": "V668 Qwen pilot is inherited evidence only",
        "execution_venue": "host",
        "execution_venue_details": {
            "host": os.uname().nodename,
            "owned_pid": os.getpid(),
            "gpu_uuid": None,
            "gpu_used": False,
        },
        "phase_spans": deepcopy(spans),
        "duration_s": time.monotonic() - started,
        "random_seed": {"authority_mutations": 7671, "purpose": "deterministic controls"},
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
        "publication_gates": {
            **publication_gate.evaluate(),
            "headline_auroc": 0.9131,
            "missing_evidence": "No V669 fresh relation, decision, learning, live ARC or whole-service cost result",
        },
        "v668_findings": {
            "covered_groups": 66,
            "total_groups": 248,
            "prior_exposed_groups": 248,
            "decision_benefit": False,
            "learning_benefit": False,
            "qwen_exact_supported_proposals": 0,
            "unrun_science_is_null": False,
        },
        "repository_suite_debt": {
            "status": "separate_from_required_checks",
            "full_suite_attempt": {
                "command": ".venv/bin/pytest tests/python -q",
                "exit_code": 2,
                "interrupted_after": "4 failed, 894 passed, 5 skipped, 67 collection errors",
                "example": "pre-existing Qwen3.6-35B registry KeyError during collection",
                "part_of_acceptance_gate": False,
            },
        },
    }
    artifact["field_principles"] = {
        key: "Report measured current scope; do not infer scientific benefit."
        for key in (*artifact, "field_principles", "reproducibility_checksum")
    }
    artifact["reproducibility_checksum"] = checksum(artifact)
    return artifact


def build_artifact_for_test(root: Path) -> dict[str, Any]:
    """Build a fixture with no fabricated required-validation receipts."""

    authority, roadmap, candidates = contract.resolve_authority(root)
    comparison = contract.compare_authorities(
        (root / contract.DESIGN_PATH).read_text(encoding="utf-8"),
        roadmap,
    )
    prior = contract.collect_prior_dispositions(root)
    return build_artifact(root, authority, candidates, comparison, prior, [], time.monotonic(), [])


def validate_artifact(value: dict[str, Any], root: Path) -> bool:
    """Cold-check immutable bytes and independently recomputed contract rows."""

    if value.get("reproducibility_checksum") != checksum(value):
        return False
    authority = root / str(value.get("selected_roadmap_path"))
    if not authority.is_file() or not (root / contract.DESIGN_PATH).is_file():
        return False
    roadmap = yaml.safe_load(authority.read_text(encoding="utf-8"))
    expected = contract.compare_authorities(
        (root / contract.DESIGN_PATH).read_text(encoding="utf-8"),
        roadmap,
    )
    if value.get("rows") != expected["rows"]:
        return False
    if value.get("sample_size_budget", {}).get("eligible") != sum(
        row["matched"] for row in expected["rows"]
    ):
        return False
    if value.get("prior_dispositions") != contract.collect_prior_dispositions(root):
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
    """Reload a disk candidate in the child process, after serialization."""

    return validate_artifact(json.loads(path.read_text(encoding="utf-8")), root)


def span(name: str, started: float, phase_start: float, units: int) -> dict[str, Any]:
    """Record a disjoint monotonic phase and its completed checkpoint count."""

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
    """Run frozen checks, exact terminal readers and atomic publication."""

    started = time.monotonic()
    progress(started, "preconditions", "before")
    if run_date != RUN_DATE or root.resolve() != ROOT or output != RESULT_PATH:
        raise ValueError("run date, absolute root, or deliverable changed")
    authority, roadmap, candidates = contract.resolve_authority(root)
    comparison = contract.compare_authorities(
        (root / contract.DESIGN_PATH).read_text(encoding="utf-8"),
        roadmap,
    )
    prior = contract.collect_prior_dispositions(root)
    spans = [span("preconditions", started, started, 14)]
    progress(started, "preconditions", "after")
    for phase in ("model_load", "generation", "benchmark"):
        phase_start = time.monotonic()
        progress(started, phase, "before")
        spans.append(span(phase, started, phase_start, 0))
        progress(started, phase, "after")
    private = Path(tempfile.mkdtemp(prefix="carnot-exp7671-", dir="/tmp"))
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
        final["honest_verdict"] = "complete_disqualified_v669_terminal_validation"
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
    """Expose the declared run and two fresh-process read-only reductions."""

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
