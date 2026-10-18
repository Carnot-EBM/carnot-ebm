"""Run the V672 administrative contract audit. REQ-REPORT-7713."""

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
from carnot.reporting import v672_contract as contract
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
RAW_DIR = Path("results/raw/experiment_7713_v672_contract_methods")
MODULE_PATH = Path("python/carnot/experiment_7713_v672_contract_methods.py")
CAPABILITY_PATH = Path("python/carnot/reporting/v672_contract.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7713_v672_contract_methods.py")
TEST_PATH = Path("tests/python/test_experiment_7713_v672_contract_methods.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
HARNESS_SPEC_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
NOTE_PATH = Path("docs/research-notes/v672-method-map.md")
MODEL_SPECS: list[str] = []


def progress(started: float, phase: str, event: str, units: int = 0) -> None:  # pragma: no cover
    """Print an owned heartbeat before or after every phase."""

    print(
        f"[exp7713] {phase} {event} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def checksum(value: dict[str, Any]) -> str:
    """Bind the reduced result, including source hashes and reader receipts."""

    return canonical_hash(
        {key: item for key, item in value.items() if key != "reproducibility_checksum"}
    )


def operand(
    check: str, path: str, field: str, expected: object, observed: object
) -> dict[str, Any]:
    """Name the exact value that blocked an external prerequisite."""

    return {
        "check": check,
        "upstream": path,
        "path": path,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def preconditions(root: Path, authority: Path, comparison: dict[str, Any]) -> list[dict[str, Any]]:
    """Measure current input bytes, contract sections, and owned host access."""

    paths = (
        contract.DESIGN_PATH,
        authority.relative_to(root),
        SPEC_PATH,
        HARNESS_SPEC_PATH,
        NOTE_PATH,
        Path("research-references.md"),
        Path("research-studying.md"),
        Path("research-complete.yaml"),
        Path("ops/conductor-log.md"),
        Path("results/experiment_7712_v671_capstone.json"),
    )
    checks = [
        operand(
            "required_input",
            str(path),
            "readable_nonempty_bytes",
            True,
            (root / path).is_file() and (root / path).stat().st_size > 0,
        )
        for path in paths
    ]
    checks.extend(
        [
            operand(
                "independent_contract",
                str(contract.DESIGN_PATH),
                "independent_table",
                13,
                0 if "design_table_missing" in comparison["errors"] else 13,
            ),
            operand(
                "independent_contract",
                str(contract.DESIGN_PATH),
                "independent_json_tasks",
                13,
                0 if "design_json_missing" in comparison["errors"] else 13,
            ),
            operand(
                "repository_root",
                str(root),
                "absolute_directory",
                True,
                root.is_absolute() and root.is_dir(),
            ),
            operand(
                "owned_process",
                f"/proc/{os.getpid()}",
                "exists",
                True,
                Path(f"/proc/{os.getpid()}").is_dir(),
            ),
        ]
    )
    return checks


def validation_plan(root: Path, authority: Path, private: Path) -> list[CommandSpec]:
    """Freeze affected Python checks and the selected-authority readers."""

    (private / "basetemp").mkdir(parents=True, exist_ok=True)
    (private / "coverage").mkdir(parents=True, exist_ok=True)
    scoped = build_scoped_commands(
        root,
        [str(TEST_PATH)],
        [str(MODULE_PATH), str(CAPABILITY_PATH)],
        static_paths=[str(WRAPPER_PATH)],
        basetemp=private / "basetemp",
        coverage_file=private / "coverage/.coverage",
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
    """Reopen the exact candidate through four independent processes."""

    python = str(root / ".venv/bin/python")
    wrapper = str(WRAPPER_PATH)
    return [
        CommandSpec(
            "cold_replay", (python, "-u", wrapper, "--cold-validate", str(candidate)), "candidate"
        ),
        CommandSpec(
            "independent_reduction",
            (python, "-u", wrapper, "--independent-reduce", str(candidate)),
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


def source_hashes(root: Path, authority: Path) -> list[dict[str, Any]]:
    """Separate current, historical and missing custody from planned output."""

    paths = (
        authority.relative_to(root),
        contract.DESIGN_PATH,
        SPEC_PATH,
        HARNESS_SPEC_PATH,
        NOTE_PATH,
        MODULE_PATH,
        CAPABILITY_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        Path("research-references.md"),
        Path("research-studying.md"),
        Path("results/experiment_7712_v671_capstone.json"),
    )
    result = [
        {
            "path": str(path),
            "role": "current_input",
            "exists": (root / path).is_file(),
            "sha256": sha256_file(root / path) if (root / path).is_file() else None,
        }
        for path in paths
    ]
    prior = json.loads((root / "results/experiment_7712_v671_capstone.json").read_text())
    for row in prior["prior_dispositions"]:
        if not row.get("evidence_path"):
            continue  # The prior capstone is already hashed above; never self-input.
        path = Path(row["evidence_path"])
        exists = (root / path).is_file()
        result.append(
            {
                "path": str(path),
                "role": "flagged_historical_evidence"
                if row["flagged_adversarial"]
                else "valid_prior_producer"
                if exists
                else "missing_prior_custody",
                "exists": exists,
                "sha256": sha256_file(root / path) if exists else None,
                "prior_verdict": row["honest_verdict"],
            }
        )
    result.append(
        {
            "path": "research-roadmap-next.yaml",
            "role": "missing_staged_authority",
            "exists": (root / "research-roadmap-next.yaml").is_file(),
            "sha256": None,
        }
    )
    result.append(
        {
            "path": str(RESULT_PATH),
            "role": "planned_output_not_input",
            "exists": False,
            "sha256": None,
        }
    )
    return result


def _gate(condition: str, observed: object, passed: bool) -> dict[str, Any]:
    """Keep an unmeasured science operand null rather than zero."""

    return {
        "condition": condition,
        "observed": observed,
        "passed": passed,
        "principle": "Only measured current evidence can satisfy this gate.",
    }


def build_artifact(
    root: Path,
    authority: Path,
    candidates: list[dict[str, Any]],
    comparison: dict[str, Any],
    receipts: list[dict[str, Any]],
    started: float,
    spans: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reduce administrative evidence while leaving all science gates null."""

    roadmap = yaml.safe_load(authority.read_text())
    checks = preconditions(root, authority, comparison)
    mutations = [
        {
            "mutation": name,
            "rejected": not contract.compare_authorities(
                (root / contract.DESIGN_PATH).read_text(), contract.mutate_authority(roadmap, name)
            )["passed"],
        }
        for name in ("delete", "reorder", "stale", "producer_field", "model")
    ]
    blocked = [row for row in checks if not row["passed"]]
    required = bool(receipts) and reduce_required_checks(receipts)["required_checks_passed"]
    required = required and all(row.get("passed") is True for row in receipts)
    inputs_present = not blocked
    verdict, verdict_class = contract.classify(comparison, inputs_present, required)
    if receipts and not required:
        verdict, verdict_class = "complete_disqualified_v672_required_validation", "disqualified"
    ready = inputs_present and comparison["passed"] and required
    prior = json.loads((root / "results/experiment_7712_v671_capstone.json").read_text())
    prior_rows = prior.get("prior_dispositions", [])
    gates = {
        name: _gate(name, None, False)
        for name in (
            "probability",
            "utility",
            "coverage",
            "source_dependence",
            "retention",
            "efficiency",
        )
    }
    gates["measured_validity"] = _gate(
        "contract, source bytes and required checks", comparison["passed"] and required, ready
    )
    gates["readiness"] = _gate("administrative agreement only", ready, ready)
    value: dict[str, Any] = {
        "schema": "carnot.exp7713.v672.contract_methods.v1",
        "experiment_id": "exp7713-contract-methods",
        "experiment": 7713,
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
        "contract_ready_score": int(ready),
        "rows": deepcopy(comparison["rows"]),
        "contract_comparison": deepcopy(comparison),
        "contract_mutation_rows": mutations,
        "sample_size_budget": {
            "intended_independent_groups": 13,
            "observed_independent_groups": len(comparison["rows"]),
            "eligible": sum(row["matched"] for row in comparison["rows"]),
            "excluded": sum(not row["matched"] for row in comparison["rows"]),
            "censored": 0,
            "effective_blocks": 13,
            "roles": ["administrative_task_contract"],
            "exposure": "no source-family experiment",
            "arms": ["design_vs_authority"],
        },
        "inference_substrate": "aggregation_from_planning_and_prior_artifacts",
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
        "execution_venue": "host",
        "execution_venue_details": {
            "host": os.uname().nodename,
            "owned_pid": os.getpid(),
            "gpu_uuid": None,
            "gpu_used": False,
        },
        "phase_spans": deepcopy(spans),
        "duration_s": time.monotonic() - started,
        "random_seed": {"value": 7713, "purpose": "deterministic private mutations; no sampling"},
        "source_artifact_hashes": source_hashes(root, authority),
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
            "frozen_before_validation": True,
        },
        "execution_backend": {
            "requested": roadmap["tasks"][0].get("agent_type"),
            "effective": "codex"
            if os.getenv("CODEX_SESSION_ID") and os.getenv("CODEX_THREAD_ID")
            else "unknown_from_process_environment",
            "evidence": "Current CODEX_SESSION_ID and CODEX_THREAD_ID presence; no future quota claim.",
        },
        "prior_v671_verdicts_preserved": {
            "source": "results/experiment_7712_v671_capstone.json",
            "prior_dispositions": prior_rows,
            "archive_lag": "completed archive ends at V670; V671 producer files and conductor log are newer",
        },
        "repository_suite_debt": {
            "status": "degraded_open_separate_from_required_checks",
            "command": ".venv/bin/pytest tests/python -q",
            "exit_code": 2,
            "observed": "2 failed, 1486 passed, 5 skipped, 63 collection errors; interrupted after failures",
            "representative_error": "retired unsloth/Qwen3.6-35B-A3B-GGUF missing from registry",
        },
    }
    value["field_principles"] = {
        key: "Measured evidence bounds the claim and prevents invalid downstream use."
        for key in (*value, "field_principles", "reproducibility_checksum")
    }
    value["field_principles"].update(
        {
            "honest_verdict": "Terminal custody prevents retries of unchanged external blocks.",
            "verdict_class": "Claim eligibility travels with the result.",
            "gate_check_summary": "Exact operands distinguish scientific failure from a broken interface.",
            "validation_receipts": "Required checks must pass before a result opens downstream execution.",
            "contract_ready_score": "Administrative agreement requires all three independent sources and readers.",
        }
    )
    value["field_principles"].update(
        {
            f"acceptance_gate_results.{name}": "Unmeasured science cannot open a readiness gate."
            for name in gates
        }
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def build_artifact_for_test(root: Path) -> dict[str, Any]:
    """Build a blocked fixture with no invented validation receipts."""

    authority, roadmap, candidates = contract.resolve_authority(root)
    comparison = contract.compare_authorities((root / contract.DESIGN_PATH).read_text(), roadmap)
    return build_artifact(root, authority, candidates, comparison, [], time.monotonic(), [])


def validate_artifact(value: dict[str, Any], root: Path) -> bool:
    """Cold-reduce task rows and exact source bytes without trusting summaries."""

    if value.get("reproducibility_checksum") != checksum(value):
        return False
    authority = root / str(value.get("selected_roadmap_path"))
    design = root / contract.DESIGN_PATH
    if not authority.is_file() or not design.is_file():
        return False
    roadmap = yaml.safe_load(authority.read_text())
    comparison = contract.compare_authorities(design.read_text(), roadmap)
    checks = preconditions(root, authority, comparison)
    ready = comparison["passed"] and all(row["passed"] for row in checks)
    ready = (
        ready
        and bool(value.get("validation_receipts"))
        and all(row.get("passed") is True for row in value["validation_receipts"])
    )
    if value.get("rows") != comparison["rows"] or value.get("contract_comparison") != comparison:
        return False
    if value.get("contract_ready_score") != int(ready):
        return False
    if value.get("sample_size_budget", {}).get("eligible") != sum(
        row["matched"] for row in comparison["rows"]
    ):
        return False
    if value.get("gate_check_summary") != [row for row in checks if not row["passed"]]:
        return False
    for row in value.get("source_artifact_hashes", []):
        if row["role"] == "planned_output_not_input":
            if row.get("sha256") is not None:
                return False
            continue
        path = root / row["path"]
        if path.is_file() != row["exists"]:
            return False
        if row["sha256"] is not None and sha256_file(path) != row["sha256"]:
            return False
    return True


def cold_read(path: Path, root: Path) -> bool:
    """Read serialized candidate bytes in a fresh process."""

    try:
        return validate_artifact(json.loads(path.read_text()), root)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def span(name: str, start: float, phase_start: float, units: int) -> dict[str, Any]:
    """Record disjoint owned time and a completed-unit checkpoint."""

    ended = time.monotonic()
    return {
        "phase": name,
        "started_offset_s": phase_start - start,
        "ended_offset_s": ended - start,
        "duration_s": ended - phase_start,
        "planned_units": units,
        "completed_units": units,
        "checkpoint_position": units,
        "heartbeat_times": [ended - start],
        "pending_operation": None,
    }


def run_experiment(root: Path, run_date: str, output: Path) -> dict[str, Any]:  # pragma: no cover
    """Run bounded validation, cold readers, then publish atomically."""

    started = time.monotonic()
    progress(started, "preconditions", "before")
    if run_date != RUN_DATE or root.resolve() != ROOT or output != RESULT_PATH:
        raise ValueError("run date, absolute root, or deliverable changed")
    authority, roadmap, candidates = contract.resolve_authority(root)
    comparison = contract.compare_authorities((root / contract.DESIGN_PATH).read_text(), roadmap)
    spans = [span("preconditions", started, started, 13)]
    progress(started, "preconditions", "after", 13)
    for phase in ("model_load", "generation", "benchmark"):
        phase_start = time.monotonic()
        progress(started, phase, "before")
        spans.append(span(phase, started, phase_start, 0))
        progress(started, phase, "after")
    (root / RAW_DIR).mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix="run-", dir=root / RAW_DIR))
    plan = validation_plan(root, authority, private)
    atomic_json(
        private / "validation_scope.json",
        {
            "tests": [str(TEST_PATH)],
            "modules": [str(MODULE_PATH), str(CAPABILITY_PATH)],
            "static": [str(WRAPPER_PATH)],
            "frozen_before_validation": True,
        },
    )
    phase_start = time.monotonic()
    progress(started, "validation", "before")
    receipts = run_commands(root, plan, log_dir=private / "logs", heartbeat_s=60)
    spans.append(span("validation", started, phase_start, len(receipts)))
    progress(started, "validation", "after", len(receipts))
    candidate = private / "candidate.json"
    atomic_json(
        candidate, build_artifact(root, authority, candidates, comparison, receipts, started, spans)
    )
    phase_start = time.monotonic()
    progress(started, "terminal_readers", "before")
    terminal = run_commands(
        root, terminal_plan(root, candidate), log_dir=private / "terminal_logs", heartbeat_s=60
    )
    spans.append(span("terminal_readers", started, phase_start, len(terminal)))
    progress(started, "terminal_readers", "after", len(terminal))
    final = build_artifact(
        root, authority, candidates, comparison, [*receipts, *terminal], started, spans
    )
    final["terminal_reader_outcomes"] = [
        {key: row[key] for key in ("name", "exit_code", "log_sha256", "passed")} for row in terminal
    ]
    final["flagged_adversarial"] = any(
        row["name"] == "adversarial_verify" and not row["passed"] for row in terminal
    )
    if not all(row["passed"] for row in terminal):
        final["honest_verdict"] = "complete_disqualified_v672_terminal_validation"
        final["verdict_class"] = "disqualified"
        final["contract_ready_score"] = 0
        final["acceptance_gate_results"]["readiness"] = _gate(
            "administrative agreement only", False, False
        )
    final["reproducibility_checksum"] = checksum(final)
    atomic_json(candidate, final)
    if not cold_read(candidate, root):
        raise RuntimeError("terminal candidate failed cold validation")
    progress(started, "exact_terminal_replay", "before")
    exact = run_commands(
        root,
        terminal_plan(root, candidate),
        log_dir=private / "exact_terminal_logs",
        heartbeat_s=60,
    )
    progress(started, "exact_terminal_replay", "after", len(exact))
    if [row["passed"] for row in exact] != [row["passed"] for row in terminal]:
        raise RuntimeError("exact terminal outcomes changed")
    atomic_json(root / output, final)
    progress(started, "publication", "after", 13)
    return final


def main(argv: list[str] | None = None) -> int:
    """Expose the required run and two fresh-process read modes."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--cold-validate")
    parser.add_argument("--independent-reduce")
    args = parser.parse_args(argv)
    if args.cold_validate:
        return 0 if cold_read(Path(args.cold_validate), ROOT) else 1
    if args.independent_reduce:
        return 0 if cold_read(Path(args.independent_reduce), ROOT) else 1
    run_experiment(ROOT, args.date, RESULT_PATH)
    return 0
