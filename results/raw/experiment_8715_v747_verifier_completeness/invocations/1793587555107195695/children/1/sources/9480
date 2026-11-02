#!/usr/bin/env python3
"""Run the V680 administrative contract; REQ-REPORT-7823."""

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
from carnot.experiment_7823_v680_contract_methods import (  # noqa: E402
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
MANIFEST_SHA256 = "b74b3cd1453b623152042fb3d39466b052e1fd7868334c87cdf7b07cfded9731"
MUTATIONS = (
    "drop",
    "reorder",
    "title",
    "phase",
    "deliverable",
    "model",
    "substrate",
    "gate_field",
    "unknown_producer",
    "prior_retirement",
)


def progress(phase: str, event: str, units: int) -> None:
    """Show real elapsed work at phase boundaries and child transitions."""
    print(
        f"[exp7823] phase={phase} event={event} elapsed_s={time.monotonic() - START:.3f} completed_units={units}",
        flush=True,
    )


def load_manifest(path: Path) -> dict[str, Any]:
    """Reject changes to the prospectively frozen child roster."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise ValueError("frozen manifest byte hash changed")
    return json.loads(path.read_text())


def runtime_plan(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Rebuild the complete plan, including commands after the scope helper."""
    private = Path(manifest["private_root"])
    module, cli = manifest["changed_modules"]
    tests = manifest["affected_tests"]
    commands = build_scoped_commands(
        ROOT,
        tests,
        [module],
        static_paths=[cli],
        basetemp=private / "pytest",
        coverage_file=private / "coverage" / "shard.coverage",
    )
    fixed = []
    for command in commands:
        argv = [
            arg.replace(
                "--include=*/experiment_7823_v680_contract_methods.py",
                "--include=*/experiment_7823_v680_contract_methods.py,*/experiment_7823_v680_contract_methods.py",
            )
            for arg in command.argv
        ]
        if command.name == "changed_module_coverage":
            argv.insert(2, "--branch")
        fixed.append(
            {
                "name": command.name,
                "argv": argv,
                "classification": "required",
                "phase": "scoped",
                "timeout_s": command.timeout_s,
            }
        )
    fixed.extend(
        {
            "name": command.name,
            "argv": list(command.argv),
            "classification": "required",
            "phase": "authority",
            "timeout_s": command.timeout_s,
        }
        for command in build_repository_check_plan(ROOT, ROOT / "research-roadmap.yaml")
    )
    python = str(ROOT / ".venv/bin/python")
    candidate = str(private / "candidate.json")
    rows = str(ROOT / RAW / "rows.json")
    fixed.append(
        {
            "name": "repository_health_full_python_suite",
            "argv": [
                str(ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={private}/pytest/full",
                "tests/python",
                "-q",
            ],
            "classification": "diagnostic",
            "phase": "diagnostic",
            "timeout_s": 180,
        }
    )
    for name, argv in (
        ("real_entrypoint_e2e", [python, "-u", cli, "--check-authority"]),
        ("cold_replay", [python, "-u", cli, "--cold-validate", candidate, "--raw", rows]),
        (
            "adversarial_verify",
            [python, "-u", "scripts/adversarial_verify.py", "--json", candidate],
        ),
        (
            "verdict_row_consistency_strict",
            [python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", candidate],
        ),
    ):
        fixed.append(
            {
                "name": name,
                "argv": argv,
                "classification": "required",
                "phase": "terminal",
                "timeout_s": 120,
            }
        )
    return fixed


def dispatch(
    manifest: dict[str, Any], executor: Callable[[dict[str, Any]], dict[str, Any]]
) -> list[dict[str, Any]]:
    """Run exactly the frozen names, arguments, order and classes."""
    expected = runtime_plan(manifest)
    if manifest["commands"] != expected:
        raise ValueError("runtime command manifest mismatch")
    return [executor(command) for command in expected]


def require_pytest_parents(commands: list[dict[str, Any]]) -> None:
    """Pytest cannot make a nested temporary base without its parent."""
    for command in commands:
        for arg in command["argv"]:
            if arg.startswith("--basetemp=") and not Path(arg.split("=", 1)[1]).parent.is_dir():
                raise ValueError("pytest basetemp parent missing")


def seal_log(source: Path, durable: Path, attempt: str) -> dict[str, str]:
    """Keep exited log bytes under an attempt and their content hash."""
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    target = durable / attempt / digest / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError("sealed log already exists")
    shutil.copyfile(source, target)
    return {"path": str(target), "sha256": sha256_file(target)}


def verify_seal(seal: dict[str, str]) -> bool:
    """Cold-check cited bytes after sealing and on later retries."""
    path = Path(seal["path"])
    return path.is_file() and sha256_file(path) == seal["sha256"]


def execute_plan(
    manifest: dict[str, Any], before_terminal: Callable[[list[dict[str, Any]]], None]
) -> list[dict[str, Any]]:
    """Run owned children with unique logs and return every real exit."""
    private = Path(manifest["private_root"])
    attempt = f"attempt-{time.monotonic_ns()}"
    (private / "coverage").mkdir(parents=True, exist_ok=True)
    for command in manifest["commands"]:
        for arg in command["argv"]:
            if arg.startswith("--basetemp="):
                Path(arg.split("=", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
    require_pytest_parents(manifest["commands"])
    receipts: list[dict[str, Any]] = []
    current_phase = ""

    def execute(command: dict[str, Any]) -> dict[str, Any]:
        nonlocal current_phase
        phase = command["phase"]
        if phase != current_phase:
            if current_phase:
                progress(current_phase, "after", len(receipts))
            current_phase = phase
            progress(phase, "before", len(receipts))
            if phase == "terminal":
                before_terminal(receipts)
        spec = CommandSpec(command["name"], tuple(command["argv"]), phase, command["timeout_s"])
        receipt = run_commands(ROOT, [spec], log_dir=private / attempt / "logs", heartbeat_s=60)[0]
        seal = seal_log(
            ROOT / receipt["log_path"]
            if not Path(receipt["log_path"]).is_absolute()
            else Path(receipt["log_path"]),
            ROOT / RAW / "sealed_logs",
            attempt,
        )
        sealed_path = Path(seal["path"])
        receipt.update(
            {
                "phase": phase,
                "classification": command["classification"],
                "log_path": str(
                    sealed_path.relative_to(ROOT)
                    if sealed_path.is_relative_to(ROOT)
                    else sealed_path
                ),
                "log_sha256": seal["sha256"],
            }
        )
        receipt["passed"] = receipt["passed"] and verify_seal(seal)
        receipts.append(receipt)
        return receipt

    dispatch(manifest, execute)
    progress(current_phase, "after", len(receipts))
    return receipts


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
    """Preserve exact administrative evidence and leave science unmeasured."""
    failed = [
        {
            "upstream_id": source["path"],
            "artifact_path": source["path"],
            "artifact_sha256": None,
            "field": "exists",
            "op": "==",
            "expected": True,
            "observed": False,
        }
        for source in sources
        if not source["exists"]
    ]
    for row in comparison["rows"]:
        for field, passed in row["checks"].items():
            if not passed:
                failed.append(
                    {
                        "upstream_id": row["unit_id"],
                        "artifact_path": str(DESIGN_SNAPSHOT),
                        "artifact_sha256": sha256_file(ROOT / DESIGN_SNAPSHOT),
                        "field": field,
                        "op": "==",
                        "expected": True,
                        "observed": False,
                    }
                )
    for receipt in receipts:
        if receipt["classification"] == "required" and not receipt["passed"]:
            failed.append(
                {
                    "upstream_id": receipt["name"],
                    "artifact_path": receipt["log_path"],
                    "artifact_sha256": receipt["log_sha256"],
                    "field": "exit_code",
                    "op": "==",
                    "expected": 0,
                    "observed": receipt["exit_code"],
                }
            )
    missing = any(not source["exists"] for source in sources)
    ready = comparison["passed"] and all(m["rejected"] for m in mutations) and not failed
    verdict = (
        "complete_blocked_v680_contract_inputs"
        if missing
        else "complete_circular_positive_v680_contract_methods"
        if ready
        else "complete_disqualified_v680_contract_validation"
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
        "schema": "carnot.exp7823.v680.contract_methods.v1",
        "experiment_id": 7823,
        "milestone": "2026.09.680",
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
            {
                "path": r["producer_path"],
                "role": "v679_declared_producer",
                "exists": r["producer_state"] != "missing",
                "sha256": r["producer_sha256"],
                "date": "2026-09-28",
                "imported_fields": ["verdict_class", "honest_verdict"],
                "eligible": r["producer_state"] in {"null", "circular_positive", "positive"},
            }
            for r in prior
        ],
        "v679_producer_inventory": prior,
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
        "random_seed": 7823,
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
            {
                "failure": "exp7809_ruff_format",
                "addressed_by": "exp7823 formatted CLI; exp7825 training CLI",
                "qualified": False,
            },
            {
                "failure": "exp7811_ruff_format",
                "addressed_by": "exp7825 training runtime",
                "qualified": False,
            },
            {
                "failure": "exp7814_coverage_combine",
                "addressed_by": "exp7828 explicit completed coverage files",
                "qualified": False,
            },
            {
                "failure": "exp7818_organic_selector",
                "addressed_by": "exp7831 new supervisor outcomes",
                "qualified": False,
            },
            {
                "failure": "five_missing_science_producers",
                "addressed_by": "exp7826, exp7827, exp7829, exp7830, exp7833",
                "qualified": False,
            },
        ],
        "prior_failure_retirement": [
            {
                "task_id": task["id"],
                "prior": item,
                "retired_if_same_verdict": task["id"].startswith("exp7823-")
                and item["retire_if_same_verdict"]
                and item["verdict"] == verdict,
            }
            for task in yaml.safe_load(authority.read_text())["tasks"]
            for item in task.get("prior_failures", [])
        ],
    }
    value["field_principles"] = {
        key: "Preserve the stated evidence boundary."
        for key in (*value, "reproducibility_checksum")
    }
    value["field_principles"]["acceptance_gates"] = {
        key: "Administrative agreement is not scientific evidence." for key in gates
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def run_experiment(run_date: str) -> dict[str, Any]:
    """Check cheap inputs first and publish only after owned validation."""
    if run_date != "20260928":
        raise ValueError("V680 contract run date must be 20260928")
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
        raise ValueError("V680 live authorities differ from immutable snapshots")
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
        Path("openspec/change-proposals/research-roadmap-v679-preserved-20260928.md"),
        Path("python/carnot/experiment_7809_v679_contract_methods.py"),
        Path("results/experiment_7822_v679_capstone.json"),
        Path("scripts/roadmap_schema.py"),
        Path("scripts/audit_roadmap_gates.py"),
        Path("scripts/exclusion_manifest_lint.py"),
        Path("docs/research-notes/v679-authority-snapshots/roadmap.yaml"),
        Path("python/carnot/experiment_7823_v680_contract_methods.py"),
        Path("scripts/experiments/experiment_7823_v680_contract_methods.py"),
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

    def before_terminal(receipts: list[dict[str, Any]]) -> None:
        """Give independent terminal readers the exact provisional bytes."""
        atomic_json(
            private / "candidate.json",
            build_artifact(
                comparison, prior, sources, receipts, mutations, phases, authority, candidates
            ),
        )

    validation_start = time.monotonic()
    receipts = execute_plan(manifest, before_terminal)
    phases.append(
        {
            "phase": "validation",
            "start_s": validation_start - START,
            "end_s": time.monotonic() - START,
            "duration_s": time.monotonic() - validation_start,
            "completed_units": len(receipts),
            "run_date": run_date,
        }
    )
    value = build_artifact(
        comparison, prior, sources, receipts, mutations, phases, authority, candidates
    )
    atomic_json(ROOT / RESULT, value)
    progress("publication", "after", len(receipts))
    return value


def main(argv: list[str] | None = None) -> int:
    """Expose live execution, authority check and frozen-byte cold replay."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260928")
    parser.add_argument("--check-authority", action="store_true")
    parser.add_argument("--cold-validate", type=Path)
    parser.add_argument("--raw", type=Path)
    args = parser.parse_args(argv)
    if args.check_authority:
        authority, roadmap, _ = resolve_authority(ROOT)
        result = compare_contract((ROOT / DESIGN).read_text(), roadmap)
        passed = (
            result["passed"]
            and sha256_file(ROOT / DESIGN) == sha256_file(ROOT / DESIGN_SNAPSHOT)
            and sha256_file(authority) == sha256_file(ROOT / YAML_SNAPSHOT)
        )
        print(json.dumps({"passed": passed, "task_count": len(result["rows"])}), flush=True)
        return 0 if passed else 1
    if args.cold_validate:
        passed = cold_validate(args.cold_validate, args.raw, ROOT)
        print(json.dumps({"passed": passed}), flush=True)
        return 0 if passed else 1
    value = run_experiment(args.date)
    print(
        json.dumps(
            {
                "honest_verdict": value["honest_verdict"],
                "contract_ready_score": value["contract_ready_score"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
