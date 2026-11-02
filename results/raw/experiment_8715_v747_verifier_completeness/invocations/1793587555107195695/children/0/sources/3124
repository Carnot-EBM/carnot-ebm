"""Run V671 contract accounting without model inference. REQ-REPORT-7699."""

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
from carnot.reporting import v671_contract as contract
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    reduce_required_checks,
    run_commands,
)

ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260926"
RESULT_PATH = contract.RESULT_PATH
MODULE_PATH = Path("python/carnot/experiment_7699_v671_contract_methods.py")
CAPABILITY_PATH = Path("python/carnot/reporting/v671_contract.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7699_v671_contract_methods.py")
TEST_PATH = Path("tests/python/test_experiment_7699_v671_contract_methods.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
HARNESS_SPEC_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
NOTE_PATH = Path("docs/research-notes/v671-method-map.md")
MODEL_SPECS: list[str] = []


def progress(started: float, phase: str, event: str, units: int = 0) -> None:  # pragma: no cover
    """Keep a flushed heartbeat at every owned phase boundary."""

    print(
        f"[exp7699] {phase} {event} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def checksum(value: dict[str, Any]) -> str:
    """Bind exact candidate bytes except the checksum field itself."""

    return canonical_hash(
        {key: item for key, item in value.items() if key != "reproducibility_checksum"}
    )


def gate(condition: str, observed: object, passed: bool, principle: str) -> dict[str, Any]:
    """Keep the measured operand beside its governing rule."""

    return {"condition": condition, "observed": observed, "passed": passed, "principle": principle}


def preconditions(root: Path, authority: Path) -> list[dict[str, Any]]:
    """Authenticate current inputs and process availability, excluding output."""

    paths = (
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
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
        HARNESS_SPEC_PATH,
        NOTE_PATH,
        Path("research-references.md"),
        Path("research-studying.md"),
        Path("results/operational_retro_2026_09_670.json"),
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
            "owned_process",
            f"/proc/{os.getpid()}",
            "exists",
            "==",
            True,
            Path(f"/proc/{os.getpid()}").is_dir(),
        )
    )
    return checks


def validation_plan(root: Path, authority: Path, private: Path) -> list[CommandSpec]:
    """Freeze changed-file checks and selected-authority repository readers."""

    basetemp = private / "basetemp"
    coverage = private / "coverage/.coverage"
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
    """Name both cold readers and exact-candidate adversarial readers."""

    python = str(root / ".venv/bin/python")
    wrapper = str(WRAPPER_PATH)
    return [
        CommandSpec(
            "cold_replay",
            (python, "-u", wrapper, "--cold-validate", str(candidate)),
            "exact_candidate",
        ),
        CommandSpec(
            "independent_reduction",
            (python, "-u", wrapper, "--independent-reduce", str(candidate)),
            "exact_candidate",
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_candidate",
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_candidate",
        ),
    ]


def source_hashes(root: Path, authority: Path, prior: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Separate immutable inputs, missing producer custody and planned output."""

    paths = (
        authority.relative_to(root),
        contract.DESIGN_PATH,
        contract.PRIOR_DESIGN_PATH,
        contract.LOG_PATH,
        SPEC_PATH,
        HARNESS_SPEC_PATH,
        NOTE_PATH,
        MODULE_PATH,
        CAPABILITY_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        Path("research-references.md"),
        Path("research-studying.md"),
        Path("results/operational_retro_2026_09_670.json"),
    )
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
            "role": "missing_producer",
            "exists": False,
            "sha256": None,
            "pre_gate_path": row["authentication_path"],
            "pre_gate_sha256": row["authentication_sha256"],
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
    """Reduce only administrative readiness and leave science claims closed."""

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
        for name in ("delete", "reorder", "producer_field", "model", "stale")
    ]
    scoped = reduce_required_checks(receipts)
    required = (
        bool(receipts)
        and scoped["required_checks_passed"]
        and all(row.get("passed") is True for row in receipts)
    )
    valid = comparison["passed"] and all(row["rejected"] for row in mutations) and required
    inputs_present = all(row["passed"] for row in checks) and all(
        row["authenticated"] for row in prior
    )
    verdict, verdict_class = contract.classify(valid, inputs_present)
    readiness = valid and inputs_present
    gates = {
        "validity": gate(
            "exact contract and required checks", valid, valid, "Invalid evidence cannot propagate."
        ),
        "readiness": gate(
            "fourteen contracts and authenticated custody",
            readiness,
            readiness,
            "Readiness is administrative only.",
        ),
        "coverage": gate(
            "new verified source coverage",
            {"verified": 0, "eligible": 0},
            False,
            "No V670 source producer ran.",
        ),
        "freshness": gate(
            "unexposed current source families",
            {"observed": 0, "intended": None},
            False,
            "Prior exposure cannot prove fresh benefit.",
        ),
        "probability": gate(
            "held-back proper loss", None, False, "No independent probability result ran."
        ),
        "utility": gate("typed decision value", None, False, "No independent decision result ran."),
        "retention": gate(
            "post-restart retained gain",
            None,
            False,
            "Improvement requires nonforgetting evidence.",
        ),
        "efficiency": gate(
            "complete-service speed and cost",
            None,
            False,
            "Kernel time alone does not measure service cost.",
        ),
    }
    blocked = [row for row in checks if not row["passed"]]
    blocked.extend(
        contract.input_check(
            "prior_authentication",
            row["task_id"],
            "authenticated",
            "==",
            True,
            row["authenticated"],
        )
        | {"path": row["planned_path"]}
        for row in prior
        if not row["authenticated"]
    )
    requested = roadmap["tasks"][0].get("agent_type")
    forced = os.environ.get("CODEX_FORCE_EXPERIMENTS") == "1"
    effective = "codex" if forced and requested in {"claude", "gemini"} else requested
    invocation = {
        "backend": "codex",
        "kind": "current_api_assistant_invocation",
        "owned_pid": os.getpid(),
        "successful_current_execution": True,
        "future_quota_verified": False,
    }
    artifact: dict[str, Any] = {
        "schema": "carnot.exp7699.v671.contract_methods.v1",
        "experiment_id": "exp7699-contract-methods",
        "experiment": 7699,
        "title": roadmap["tasks"][0]["title"],
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
            "effective_blocks": 14,
            "prior_exposure": "V669 selected 480 families after excluding 429 previously exposed families; historical only",
            "inference_limit": "Task contracts are accounting units, not source samples",
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_specs": [{"no_model": True}],
        "model_invoked": False,
        "invocation_counts": {
            f"{op}_{state}": 0
            for op in ("loads", "forwards", "generations", "tokens")
            for state in ("attempted", "completed", "failed", "cancelled")
        },
        "historical_model_provenance": "V669 and V668 Qwen results are historical; no V670 model invocation",
        "execution_venue": "host",
        "execution_venue_details": {
            "host": os.uname().nodename,
            "owned_pid": os.getpid(),
            "gpu_uuid": None,
            "gpu_used": False,
        },
        "phase_spans": deepcopy(spans),
        "duration_s": time.monotonic() - started,
        "random_seed": {"value": 7699, "purpose": "deterministic private mutations"},
        "source_artifact_hashes": source_hashes(root, authority, prior),
        "preconditions_checked": checks,
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
            "spec_paths": [str(SPEC_PATH), str(HARNESS_SPEC_PATH)],
            "frozen_before_validation": True,
        },
        "execution_recovery_receipt": {
            "declared_backend": requested,
            "effective_backend": effective,
            "force_env_observed": forced,
            "dispatcher_source_path": "scripts/research_conductor.py",
            "dispatcher_source_sha256": sha256_file(root / "scripts/research_conductor.py"),
            "launch_receipt": {
                "kind": "current_invocation",
                "code_session_present": bool(os.environ.get("CODEX_SESSION_ID")),
                "code_thread_present": bool(os.environ.get("CODEX_THREAD_ID")),
            },
            "successful_invocation": invocation,
            "backend_recovery_claim": "current invocation only; no future quota guarantee",
        },
        "v669_historical_findings": {
            "previously_exposed_excluded_families": 429,
            "selected_families": 480,
            "v670_measurement": False,
        },
        "v670_operational_retro": {
            "path": "results/operational_retro_2026_09_670.json",
            "zero_reconstructed_commits": True,
            "compute_efficiency_measured": False,
        },
        "repository_suite_debt": {"status": "separate_from_required_checks"},
    }
    artifact["field_principles"] = {
        key: "Report exact current operands; administrative readiness does not establish scientific benefit."
        for key in (*artifact, "field_principles", "reproducibility_checksum")
    }
    artifact["field_principles"].update(
        {
            "honest_verdict": "A terminal disposition prevents retries of unchanged external blocks.",
            "verdict_class": "A closed enum carries claim eligibility into downstream readers.",
            "flagged_adversarial": "Disqualified evidence must not pass a downstream readiness gate.",
            "gate_check_summary": "Exact upstream fields and observed values distinguish a false gate from missing evidence.",
            "rows": "Independent unit observations permit recomputation without enlarging n by views or seeds.",
            "sample_size_budget": "The denominator and prior exposure limit every inference.",
            "inference_substrate": "The declared execution path must match real computation.",
            "inference_substrate_class": "Duration floors follow actual generation or no-generation work.",
            "MODEL_SPECS": "Experimental models match actual invocations, not the coding agent.",
            "model_invoked": "Invocation counts expose actual loads, calls and failures.",
            "execution_venue": "Host and device claims require current work receipts.",
            "phase_spans": "Monotonic spans and checkpoints bind current work timing.",
            "random_seed": "Independent replay requires declared random inputs.",
            "reproducibility_checksum": "Immutable sources, configuration and reducer code bind replay.",
            "source_artifact_hashes": "Producer, pre-gate and missing custody stay separate from planned output.",
            "preconditions_checked": "Planned output is not a prerequisite; current resources are measured.",
            "validation_receipts": "Frozen scope and exact reader exits determine validity.",
            "verifier_is_oracle": "Fixture truth cannot establish an oracle-distinct verifier advantage.",
            "contract_ready_score": "One means exact equality and valid readers, only administrative readiness.",
            "prior_dispositions": "Literal V670 log custody cannot invent scientific verdicts.",
            "method_map_path": "Method applicability and exclusions must remain reviewable.",
            "execution_recovery_receipt": "Effective backend evidence does not promise future quota.",
            "acceptance_gate_results": "Validity and readiness are separate from scientific quality thresholds.",
        }
    )
    artifact["field_principles"].update(
        {
            f"acceptance_gate_results.{name}": rule
            for name, rule in (
                ("validity", "Invalid evidence cannot propagate."),
                (
                    "readiness",
                    "Only exact contracts and valid readers allow administrative readiness.",
                ),
                ("coverage", "Unmeasured coverage cannot imply supported source relations."),
                ("freshness", "Prior exposure cannot become fresh evidence."),
                ("probability", "Proper loss needs held-back independent labels."),
                ("utility", "Typed decisions need independent measured value."),
                ("retention", "Improvement by forgetting fails the retention bound."),
                ("efficiency", "Complete orchestration cost governs speed claims."),
            )
        }
    )
    artifact["reproducibility_checksum"] = checksum(artifact)
    return artifact


def build_artifact_for_test(root: Path) -> dict[str, Any]:
    """Make a fixture without fabricating required validation receipts."""

    authority, roadmap, candidates = contract.resolve_authority(root)
    comparison = contract.compare_authorities(
        (root / contract.DESIGN_PATH).read_text(encoding="utf-8"), roadmap
    )
    return build_artifact(
        root,
        authority,
        candidates,
        comparison,
        contract.collect_prior_dispositions(root),
        [],
        time.monotonic(),
        [],
    )


def validate_artifact(value: dict[str, Any], root: Path) -> bool:
    """Cold-check source bytes, raw rows and custody without producer helpers."""

    if value.get("reproducibility_checksum") != checksum(value):
        return False
    authority = root / str(value.get("selected_roadmap_path"))
    if not authority.is_file() or not (root / contract.DESIGN_PATH).is_file():
        return False
    roadmap = yaml.safe_load(authority.read_text(encoding="utf-8"))
    expected = contract.compare_authorities(
        (root / contract.DESIGN_PATH).read_text(encoding="utf-8"), roadmap
    )
    if value.get("rows") != expected["rows"] or value.get(
        "prior_dispositions"
    ) != contract.collect_prior_dispositions(root):
        return False
    if value.get("sample_size_budget", {}).get("eligible") != sum(
        row["matched"] for row in expected["rows"]
    ):
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
    """Reload serialized bytes in a fresh process before reducing them."""

    return validate_artifact(json.loads(path.read_text(encoding="utf-8")), root)


def span(name: str, started: float, phase_start: float, units: int) -> dict[str, Any]:
    """Record one disjoint monotonic phase and its completed checkpoint."""

    ended = time.monotonic()
    return {
        "phase": name,
        "started_offset_s": phase_start - started,
        "ended_offset_s": ended - started,
        "duration_s": ended - phase_start,
        "planned_units": units,
        "completed_units": units,
        "checkpoint_position": units,
        "heartbeat_times": [ended - started],
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
        (root / contract.DESIGN_PATH).read_text(encoding="utf-8"), roadmap
    )
    prior = contract.collect_prior_dispositions(root)
    spans = [span("preconditions", started, started, 14)]
    progress(started, "preconditions", "after", 14)
    for phase in ("model_load", "generation", "benchmark"):
        phase_start = time.monotonic()
        progress(started, phase, "before")
        spans.append(span(phase, started, phase_start, 0))
        progress(started, phase, "after")
    private = Path(tempfile.mkdtemp(prefix="carnot-exp7699-", dir="/tmp"))
    plan = validation_plan(root, authority, private)
    atomic_json(
        private / "affected-file-manifest.json",
        {
            "tests": [str(TEST_PATH)],
            "modules": [str(MODULE_PATH), str(CAPABILITY_PATH)],
            "static": [str(WRAPPER_PATH), str(SPEC_PATH), str(HARNESS_SPEC_PATH), str(NOTE_PATH)],
            "frozen_before_validation": True,
        },
    )
    phase_start = time.monotonic()
    progress(started, "validation", "before")
    receipts = run_commands(root, plan, log_dir=private / "logs")
    spans.append(span("validation", started, phase_start, len(receipts)))
    progress(started, "validation", "after", len(receipts))
    candidate_path = private / "candidate.json"
    atomic_json(
        candidate_path,
        build_artifact(root, authority, candidates, comparison, prior, receipts, started, spans),
    )
    phase_start = time.monotonic()
    progress(started, "terminal_readers", "before")
    terminal = run_commands(
        root, terminal_plan(root, candidate_path), log_dir=private / "terminal_logs"
    )
    spans.append(span("terminal_readers", started, phase_start, len(terminal)))
    progress(started, "terminal_readers", "after", len(terminal))
    final = build_artifact(
        root, authority, candidates, comparison, prior, [*receipts, *terminal], started, spans
    )
    final["flagged_adversarial"] = any(
        not row["passed"] for row in terminal if row["name"] == "adversarial_verify"
    )
    final["terminal_reader_outcomes"] = [
        {key: row[key] for key in ("name", "exit_code", "log_sha256", "passed")} for row in terminal
    ]
    if not all(row["passed"] for row in terminal):
        final["honest_verdict"], final["verdict_class"] = (
            "complete_disqualified_v671_terminal_validation",
            "disqualified",
        )
        final["contract_ready_score"] = 0
        final["acceptance_gate_results"]["readiness"].update({"observed": False, "passed": False})
    final["reproducibility_checksum"] = checksum(final)
    atomic_json(candidate_path, final)
    if not cold_read(candidate_path, root):
        raise RuntimeError("terminal candidate failed cold validation")
    progress(started, "exact_terminal_replay", "before")
    exact = run_commands(
        root, terminal_plan(root, candidate_path), log_dir=private / "exact_terminal_logs"
    )
    progress(started, "exact_terminal_replay", "after", len(exact))
    if [row["passed"] for row in exact] != [row["passed"] for row in terminal]:
        raise RuntimeError("exact terminal outcomes changed")
    atomic_json(root / RESULT_PATH, final)
    progress(started, "publication", "after", 14)
    return final


def main(argv: list[str] | None = None) -> int:
    """Expose the run and fresh-process read-only reductions."""

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
