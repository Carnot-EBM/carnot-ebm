#!/usr/bin/env python3
"""Run V679's administrative contract; REQ-REPORT-7809."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import yaml  # noqa: E402
from carnot.experiment_7573_v662_contract_methods import build_repository_check_plan  # noqa: E402
from carnot.experiment_7809_v679_contract_methods import (  # noqa: E402
    DESIGN,
    DESIGN_SNAPSHOT,
    MANIFEST,
    METHOD,
    RAW,
    RESULT,
    ROOT,
    YAML_SNAPSHOT,
    cold_validate,
    compare_contract,
    mutate,
    prior_inventory,
    resolve_authority,
)
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file  # noqa: E402
from carnot.reporting.experiment_7303_validation_scope import (  # noqa: E402
    CommandSpec,
    build_scoped_commands,
    run_commands,
)

START = time.monotonic()
MANIFEST_SHA256 = "19aa0ef7ca0ad61a425ffb8a9ca714a32747ac8a502bbf640b033f7a762132ed"
MUTATIONS = (
    "drop",
    "reorder",
    "title",
    "phase",
    "deliverable",
    "model",
    "substrate",
    "unknown_producer",
    "gate_field",
    "prior_experiment_id",
    "prior_verdict",
    "prior_addressed_by",
    "prior_retirement",
)


def progress(phase: str, event: str, units: int) -> None:
    """Expose elapsed work at each boundary and child transition."""
    print(
        f"[exp7809] phase={phase} event={event} elapsed_s={time.monotonic() - START:.3f} completed_units={units}",
        flush=True,
    )


def load_manifest(path: Path) -> dict[str, Any]:
    """Reject a changed command plan before starting any child."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise ValueError("frozen manifest byte hash changed")
    return json.loads(path.read_text())


def runtime_plan(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Rebuild every command, including guards appended after the helper."""
    private = Path(manifest["private_root"])
    tests = manifest["affected_tests"]
    module, cli = manifest["changed_modules"]
    commands = build_scoped_commands(
        ROOT,
        tests,
        [module],
        static_paths=[cli],
        basetemp=private / "pytest",
        coverage_file=private / "coverage" / "shard.coverage",
    )
    include = module + "," + cli
    fixed = []
    for command in commands:
        argv = [
            arg if not arg.startswith("--include=") else "--include=" + include
            for arg in command.argv
        ]
        if command.name == "changed_module_coverage":
            argv.insert(2, "--branch")
        fixed.append(CommandSpec(command.name, tuple(argv), command.scope, command.timeout_s))
    fixed += build_repository_check_plan(ROOT, ROOT / "research-roadmap.yaml")
    python = str(ROOT / ".venv/bin/python")
    candidate = str(private / "candidate.json")
    rows = str(ROOT / RAW / "rows.json")
    fixed.append(
        CommandSpec(
            "repository_health_full_python_suite",
            (
                str(ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={private}/pytest/full",
                "tests/python",
                "-q",
            ),
            "diagnostic",
            120,
        )
    )
    fixed.extend(
        [
            CommandSpec(
                "real_entrypoint_e2e",
                (python, "-u", cli, "--check-authority"),
                "exact_candidate",
                120,
            ),
            CommandSpec(
                "cold_replay",
                (python, "-u", cli, "--cold-validate", candidate, "--raw", rows),
                "exact_candidate",
                120,
            ),
            CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", "--json", candidate),
                "exact_candidate",
                120,
            ),
            CommandSpec(
                "verdict_row_consistency_strict",
                (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", candidate),
                "exact_candidate",
                120,
            ),
        ]
    )
    phases = ["scoped"] * 8 + ["authority"] * 8 + ["diagnostic"] + ["terminal"] * 4
    return [
        {
            "phase": phase,
            "name": spec.name,
            "argv": list(spec.argv),
            "classification": "diagnostic" if phase == "diagnostic" else "required",
            "timeout_s": spec.timeout_s,
        }
        for phase, spec in zip(phases, fixed)
    ]


def dispatch(
    manifest: dict[str, Any], executor: Callable[[dict[str, Any]], dict[str, Any]]
) -> list[dict[str, Any]]:
    """Execute only the complete prospectively frozen child roster."""
    expected = runtime_plan(manifest)
    if manifest["commands"] != expected:
        raise ValueError("runtime command manifest mismatch")
    return [executor(command) for command in expected]


def require_pytest_parents(commands: list[dict[str, Any]]) -> None:
    """A nested pytest base cannot be created unless its parent exists."""
    for command in commands:
        for arg in command["argv"]:
            if arg.startswith("--basetemp=") and not Path(arg.split("=", 1)[1]).parent.is_dir():
                raise ValueError("pytest basetemp parent missing")


def seal_log(source: Path, durable: Path, attempt: str) -> dict[str, str]:
    """Copy completed bytes once to an attempt-specific content-addressed path."""
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    target = durable / attempt / digest / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError("sealed log already exists")
    shutil.copyfile(source, target)
    return {
        "path": str(target),
        "sha256": "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest(),
    }


def verify_seal(seal: dict[str, str]) -> bool:
    """Detect any later byte change in a cited log."""
    path = Path(seal["path"])
    return path.is_file() and sha256_file(path) == seal["sha256"]


def build_artifact(
    comparison: dict[str, Any],
    prior: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
    mutations: list[dict[str, Any]],
    phases: list[dict[str, Any]],
    authority: Path,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Report contract truth while keeping every science outcome unmeasured."""
    failed = []
    for source in sources:
        if not source["exists"]:
            failed.append(
                dict(
                    upstream_id=source["path"],
                    artifact_path=source["path"],
                    artifact_sha256=None,
                    field="exists",
                    op="==",
                    expected=True,
                    observed=False,
                )
            )
    for row in comparison["rows"]:
        for field, passed in row["checks"].items():
            if not passed:
                failed.append(
                    dict(
                        upstream_id=row["unit_id"],
                        artifact_path=str(DESIGN_SNAPSHOT),
                        artifact_sha256=sha256_file(ROOT / DESIGN_SNAPSHOT),
                        field=field,
                        op="==",
                        expected=True,
                        observed=False,
                    )
                )
    for receipt in receipts:
        if receipt["classification"] == "required" and not receipt["passed"]:
            failed.append(
                dict(
                    upstream_id=receipt["name"],
                    artifact_path=receipt["log_path"],
                    artifact_sha256=receipt["log_sha256"],
                    field="exit_code",
                    op="==",
                    expected=0,
                    observed=receipt["exit_code"],
                )
            )
    missing = any(not source["exists"] for source in sources)
    validated = all(r["passed"] for r in receipts if r["classification"] == "required")
    ready = (
        comparison["passed"] and all(m["rejected"] for m in mutations) and validated and not failed
    )
    verdict = (
        "complete_blocked_v679_contract_inputs"
        if missing
        else "complete_circular_positive_v679_contract_methods"
        if ready
        else "complete_disqualified_v679_contract_validation"
    )
    gates = {
        "validity": ready,
        "readiness": int(ready),
        "probability_quality": None,
        "decision_benefit": None,
        "retention": None,
        "efficiency": None,
    }
    value: dict[str, Any] = {
        "schema": "carnot.exp7809.v679.contract_methods.v1",
        "experiment_id": 7809,
        "milestone": "2026.09.679",
        "run_date": "20260928",
        "status": "complete",
        "honest_verdict": verdict,
        "verdict_class": "blocked" if missing else "circular_positive" if ready else "disqualified",
        "flagged_adversarial": any(
            r["name"] == "adversarial_verify" and not r["passed"] for r in receipts
        ),
        "gate_check_summary": failed,
        "positive_claim": False,
        "rows": comparison["rows"],
        "contract_comparison": comparison,
        "contract_ready_score": int(ready),
        "acceptance_gate_results": gates,
        "validation_receipts": receipts,
        "source_artifact_hashes": sources
        + [
            dict(
                path=r["producer_path"],
                role="v678_declared_producer",
                exists=r["producer_state"] != "missing",
                sha256=r["producer_sha256"],
                date="2026-09-28",
                imported_fields=["verdict_class", "honest_verdict"],
                eligible=r["producer_state"] in {"null", "circular_positive", "positive"},
            )
            for r in prior
        ],
        "v678_producer_inventory": prior,
        "source_artifact_categories": {
            "missing_scientific_producers": [
                r["producer_path"] for r in prior if r["producer_state"] == "missing"
            ],
            "pre_gate_receipts": [r["pre_gate_path"] for r in prior if r["pre_gate_path"]],
            "disqualified": [
                r["producer_path"] for r in prior if r["producer_state"] == "disqualified"
            ],
            "blocked": [r["producer_path"] for r in prior if r["producer_state"] == "blocked"],
        },
        "sample_size_budget": {
            "intended": 14,
            "eligible": sum(r["matched"] for r in comparison["rows"]),
            "started": 14,
            "completed": 14,
            "excluded": sum(not r["matched"] for r in comparison["rows"]),
            "censored": 0,
            "effective_independent_n": 14,
            "unit": "administrative task",
        },
        "preconditions_checked": {
            "authority_candidates": candidates,
            "backend": "host_cpu_aggregation",
            "gpu_used": False,
            "required_inputs": sources,
            "owned_output_parent_exists": (ROOT / RESULT).parent.is_dir(),
        },
        "mutations": mutations,
        "verifier_is_oracle": True,
        "claim_scope": {
            "kind": "administrative_exact_authority",
            "science": "unmeasured",
            "fresh_generalization_eligible": False,
            "RAGTruth": "all_640_families_exposed_development",
            "oracle_distinct_corrigendum": "2026-09-28; GAP-ORACLE-DISTINCT remains open",
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "actual_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {
            "loads": 0,
            "generations": 0,
            "forwards": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "loaded_files": [],
        },
        "model_invoked": False,
        "random_seed": 7809,
        "phase_spans": phases,
        "duration_s": time.monotonic() - START,
        "method_map_path": str(METHOD),
        "selected_roadmap_path": str(authority.relative_to(ROOT)),
        "authority_snapshot_paths": [
            {"path": str(p), "sha256": sha256_file(ROOT / p)}
            for p in (DESIGN_SNAPSHOT, YAML_SNAPSHOT)
        ],
        "validation_command_manifest_path": str(MANIFEST),
        "validation_command_manifest_sha256": sha256_file(ROOT / MANIFEST),
        "observed_child_commands": [
            {"name": r["name"], "argv": r["command_argv"], "classification": r["classification"]}
            for r in receipts
        ],
        "repository_health": next(
            (r for r in receipts if r["name"] == "repository_health_full_python_suite"), None
        ),
        "historical_failures": [
            {"failure": name, "repair": repair, "shipped": False}
            for name, repair in (
                ("candidate_roster_mismatch", "canonical family custody test"),
                ("five_stale_receipt_hashes", "immutable per-attempt log seal test"),
                ("ruff_format", "scoped format check"),
                ("terminated_broad_suite", "diagnostic bounded health check"),
                ("missing_pytest_parent", "private parent creation test"),
            )
        ],
        "prior_failure_retirement": [
            {
                "task_id": task["id"],
                "prior": item,
                "retired_if_same_verdict": task["id"].startswith("exp7809-")
                and item["retire_if_same_verdict"]
                and item["verdict"] == verdict,
            }
            for task in yaml.safe_load(authority.read_text())["tasks"]
            for item in task["prior_failures"]
        ],
    }
    principles = {
        "experiment_id": "Each result has one owner.",
        "milestone": "Each result has one owner.",
        "run_date": "Each result has one owner.",
        "honest_verdict": "External incompleteness cannot be fixed by retrying owned work.",
        "verdict_class": "Claim strength travels with the record.",
        "flagged_adversarial": "Invalid evidence cannot open a gate.",
        "gate_check_summary": "Missing evidence differs from a scientific null.",
        "rows": "Recompute every comparison from its units.",
        "acceptance_gate_results": "A working fixture proves no scientific gain.",
        "duration_s": "Duration reflects actual work.",
        "phase_spans": "Duration reflects actual work.",
        "random_seed": "Replay requires identical inputs.",
        "reproducibility_checksum": "Replay requires identical inputs.",
        "sample_size_budget": "Views and seeds are not new families.",
        "source_artifact_hashes": "Old files cannot replace missing current producers.",
        "preconditions_checked": "Cheap failures precede compute.",
        "validation_receipts": "Every required check must pass.",
        "verifier_is_oracle": "Fixture truth cannot show hidden generalization.",
        "claim_scope": "Fixture truth cannot show hidden generalization.",
        "inference_substrate": "Floors follow invoked work.",
        "inference_substrate_class": "Floors follow invoked work.",
        "MODEL_SPECS": "Citing a model is not invoking it.",
        "model_specs": "Citing a model is not invoking it.",
        "model_invocation_counts": "Citing a model is not invoking it.",
        "contract_ready_score": "Administrative agreement is not scientific benefit.",
        "authority_snapshot_paths": "Rollover must not invalidate past tests.",
        "method_map_path": "Method ownership must remain durable.",
        "validation_command_manifest_path": "The current dispatcher must retain its obligations.",
        "validation_command_manifest_sha256": "The current dispatcher must retain its obligations.",
        "observed_child_commands": "The actual commands must match the frozen plan.",
        "repository_health": "Broader failures remain visible without weakening current scope.",
    }
    value["field_principles"] = {
        key: principles.get(key, "Exact source evidence limits this administrative field.")
        for key in (*value, "reproducibility_checksum")
    }
    value["field_principles"]["acceptance_gates"] = {
        key: "A contract fixture does not measure scientific gain." for key in gates
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def run_experiment(run_date: str) -> dict[str, Any]:
    """Check inputs before work, then publish only after bounded validation."""
    if run_date != "20260928":
        raise ValueError("V679 contract run date must be 20260928")
    progress("preflight", "before", 0)
    phase_start = time.monotonic()
    authority, roadmap, candidates = resolve_authority(ROOT)
    design = (ROOT / DESIGN).read_text()
    comparison = compare_contract(design, roadmap)
    frozen = compare_contract(
        (ROOT / DESIGN_SNAPSHOT).read_text(), yaml.safe_load((ROOT / YAML_SNAPSHOT).read_text())
    )
    if (
        comparison != frozen
        or sha256_file(ROOT / DESIGN) != sha256_file(ROOT / DESIGN_SNAPSHOT)
        or sha256_file(authority) != sha256_file(ROOT / YAML_SNAPSHOT)
    ):
        raise ValueError("V679 live authorities differ from immutable snapshots")
    required = [
        authority.relative_to(ROOT),
        DESIGN,
        DESIGN_SNAPSHOT,
        YAML_SNAPSHOT,
        METHOD,
        MANIFEST,
        Path("AGENTS.md"),
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
        Path("docs/research-notes/v678-authority-snapshots/roadmap.yaml"),
        Path("results/experiment_7808_v678_capstone.json"),
    ]
    sources = [
        {
            "path": str(p),
            "role": "current_input",
            "exists": (ROOT / p).is_file(),
            "sha256": sha256_file(ROOT / p) if (ROOT / p).is_file() else None,
            "date": "2026-09-28",
            "imported_fields": ["bytes"] if (ROOT / p).is_file() else [],
            "eligible": (ROOT / p).is_file(),
        }
        for p in required
    ]
    prior = prior_inventory(ROOT)
    mutations = [
        {
            "mutation": name,
            "rejected": not compare_contract(design, mutate(roadmap, name))["passed"],
        }
        for name in MUTATIONS
    ]
    raw = ROOT / RAW
    raw.mkdir(parents=True, exist_ok=True)
    atomic_json(raw / "rows.json", comparison["rows"])
    atomic_json(raw / "mutations.json", mutations)
    phases = [
        {
            "phase": "preflight",
            "start_s": phase_start - START,
            "end_s": time.monotonic() - START,
            "duration_s": time.monotonic() - phase_start,
            "completed_units": 14,
            "run_date": run_date,
        }
    ]
    progress("preflight", "after", 14)

    manifest = load_manifest(ROOT / MANIFEST)
    private = Path(manifest["private_root"])
    private.mkdir(parents=True, exist_ok=True)
    for command in manifest["commands"]:
        for arg in command["argv"]:
            if arg.startswith("--basetemp="):
                Path(arg.split("=", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
    (private / "coverage").mkdir(parents=True, exist_ok=True)
    require_pytest_parents(manifest["commands"])
    candidate = private / "candidate.json"
    receipts: list[dict[str, Any]] = []
    current_phase = ""
    phase_start = time.monotonic()

    def execute(command: dict[str, Any]) -> dict[str, Any]:
        nonlocal current_phase, phase_start
        phase = command["phase"]
        if phase != current_phase:
            if current_phase:
                phases.append(
                    {
                        "phase": current_phase,
                        "start_s": phase_start - START,
                        "end_s": time.monotonic() - START,
                        "duration_s": time.monotonic() - phase_start,
                        "completed_units": sum(r["phase"] == current_phase for r in receipts),
                        "run_date": run_date,
                    }
                )
                progress(current_phase, "after", len(receipts))
            current_phase = phase
            phase_start = time.monotonic()
            progress(phase, "before", len(receipts))
            if phase == "terminal":
                atomic_json(
                    candidate,
                    build_artifact(
                        comparison,
                        prior,
                        sources,
                        receipts,
                        mutations,
                        phases,
                        authority,
                        candidates,
                    ),
                )
        index = len(receipts)
        spec = CommandSpec(command["name"], tuple(command["argv"]), phase, command["timeout_s"])
        attempt = f"attempt-{index + 1:02d}-{time.monotonic_ns()}"
        log_dir = raw / "validation" / attempt
        result = run_commands(ROOT, [spec], log_dir=log_dir, heartbeat_s=30)[0]
        seal = seal_log(ROOT / result["log_path"], raw / "sealed_logs", attempt)
        sealed = Path(seal["path"])
        result["log_path"] = str(
            sealed.relative_to(ROOT) if sealed.is_relative_to(ROOT) else sealed
        )
        result["log_sha256"] = seal["sha256"]
        result["classification"] = command["classification"]
        result["phase"] = phase
        receipts.append(result)
        return result

    dispatch(manifest, execute)
    phases.append(
        {
            "phase": current_phase,
            "start_s": phase_start - START,
            "end_s": time.monotonic() - START,
            "duration_s": time.monotonic() - phase_start,
            "completed_units": sum(r["phase"] == current_phase for r in receipts),
            "run_date": run_date,
        }
    )
    progress(current_phase, "after", len(receipts))
    value = build_artifact(
        comparison, prior, sources, receipts, mutations, phases, authority, candidates
    )
    atomic_json(ROOT / RESULT, value)
    progress("publication", "after", 14)
    return value


def main(argv: list[str] | None = None) -> int:
    """Expose the real run and a fresh-process cold authority reader."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260928")
    parser.add_argument("--check-authority", action="store_true")
    parser.add_argument("--cold-validate", type=Path)
    parser.add_argument("--raw", type=Path)
    args = parser.parse_args(argv)
    if args.check_authority:
        progress("e2e", "before", 0)
        frozen = compare_contract(
            (ROOT / DESIGN_SNAPSHOT).read_text(), yaml.safe_load((ROOT / YAML_SNAPSHOT).read_text())
        )
        progress("e2e", "after", len(frozen["rows"]))
        return 0 if frozen["passed"] else 1
    if args.cold_validate:
        progress("cold_replay", "before", 0)
        passed = bool(args.raw) and cold_validate(args.cold_validate, args.raw, ROOT)
        progress("cold_replay", "after", 14 if passed else 0)
        return 0 if passed else 1
    result = run_experiment(args.date)
    print(
        json.dumps(
            {"honest_verdict": result["honest_verdict"], "verdict_class": result["verdict_class"]}
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
