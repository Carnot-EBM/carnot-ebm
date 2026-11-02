#!/usr/bin/env python3
"""Run the V678 authority contract; REQ-REPORT-7795."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import yaml

from carnot.experiment_7795_v678_contract_methods import (
    CLI,
    DESIGN,
    DESIGN_SNAPSHOT,
    METHOD,
    MODULE,
    RAW,
    RESULT,
    ROOT,
    TEST,
    YAML_SNAPSHOT,
    cold_validate,
    compare_contract,
    mutate,
    prior_inventory,
    resolve_authority,
    source_hashes,
)
from carnot.experiment_7573_v662_contract_methods import build_repository_check_plan
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)

START = time.monotonic()


def progress(phase: str, event: str, units: int = 0) -> None:
    """Report real elapsed time at every boundary and child transition."""
    print(
        f"[exp7795] phase={phase} event={event} elapsed_s={time.monotonic() - START:.3f} completed_units={units}",
        flush=True,
    )


def span(phase: str, begin: float, units: int) -> dict:
    """Keep measured time; an administrative check has no duration floor."""
    end = time.monotonic()
    return {
        "phase": phase,
        "start_s": begin - START,
        "end_s": end - START,
        "duration_s": end - begin,
        "run_date": "20260928",
        "completed_units": units,
        "heartbeat_times": [end - START],
    }


def inputs(authority: Path) -> list[Path]:
    """Name resources before compute so missing bytes have exact operands."""
    return [
        authority.relative_to(ROOT),
        DESIGN,
        DESIGN_SNAPSHOT,
        YAML_SNAPSHOT,
        METHOD,
        MODULE,
        TEST,
        CLI,
        RAW / "frozen_affected_scope.json",
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
        Path("docs/research-notes/v677-authority-snapshots/roadmap.yaml"),
        Path("results/experiment_7794_v677_capstone.json"),
    ]


def failures_for(comparison: dict, sources: list[dict], receipts: list[dict]) -> list[dict]:
    """Expose the failed byte, field and operator for every refused check."""
    failures = []
    for source in sources:
        if not source["exists"]:
            failures.append(
                {
                    "upstream_id": source["path"],
                    "artifact_path": source["path"],
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
                        "artifact_path": str(DESIGN_SNAPSHOT),
                        "artifact_sha256": sha256_file(ROOT / DESIGN_SNAPSHOT),
                        "field": field,
                        "op": "==",
                        "expected": True,
                        "observed": False,
                    }
                )
    for receipt in receipts:
        if not receipt["passed"]:
            failures.append(
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
    return failures


def artifact(
    authority: Path,
    candidates: list[dict],
    comparison: dict,
    sources: list[dict],
    receipts: list[dict],
    phases: list[dict],
    mutations: list[dict],
) -> dict:
    """Build one complete record without converting contract truth into science."""
    failures = failures_for(comparison, sources, receipts)
    missing = any(not row["exists"] for row in sources)
    validated = bool(receipts) and all(row["passed"] for row in receipts)
    contract = comparison["passed"] and all(row["rejected"] for row in mutations)
    ready = contract and validated and not failures
    verdict = (
        "complete_blocked_v678_contract_inputs"
        if missing
        else "complete_circular_positive_v678_contract_methods"
        if ready
        else "complete_disqualified_v678_contract_validation"
    )
    verdict_class = "blocked" if missing else "circular_positive" if ready else "disqualified"
    prior = prior_inventory(ROOT)
    producer_sources = [
        {
            "path": row["producer_path"],
            "role": "v677_declared_producer",
            "exists": row["producer_state"] != "missing",
            "sha256": row["producer_sha256"],
            "date": "2026-09-28",
            "imported_fields": ["verdict_class", "honest_verdict"],
            "eligible": row["producer_state"] in {"null", "circular_positive", "positive"},
        }
        for row in prior
    ]
    gates = {
        "validity": ready,
        "readiness": int(ready),
        "probability_quality": None,
        "decision_benefit": None,
        "retention": None,
        "efficiency": None,
    }
    value = {
        "schema": "carnot.exp7795.v678.contract_methods.v1",
        "experiment_id": 7795,
        "milestone": "2026.09.678",
        "run_date": "20260928",
        "status": "complete",
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": any(
            row["name"] == "adversarial_verify" and not row["passed"] for row in receipts
        ),
        "gate_check_summary": failures,
        "positive_claim": False,
        "rows": comparison["rows"],
        "contract_comparison": comparison,
        "contract_ready_score": int(contract and validated),
        "acceptance_gate_results": gates,
        "source_artifact_hashes": sources + producer_sources,
        "v677_producer_inventory": prior,
        "source_artifact_categories": {
            "missing_scientific_producers": [
                row["producer_path"] for row in prior if row["producer_state"] == "missing"
            ],
            "pre_gate_receipts": [row["pre_gate_path"] for row in prior if row["pre_gate_path"]],
            "disqualified": [
                row["producer_path"] for row in prior if row["producer_state"] == "disqualified"
            ],
            "blocked": [
                row["producer_path"] for row in prior if row["producer_state"] == "blocked"
            ],
        },
        "sample_size_budget": {
            "intended": 14,
            "eligible": sum(row["matched"] for row in comparison["rows"]),
            "started": 14,
            "completed": 14,
            "excluded": sum(not row["matched"] for row in comparison["rows"]),
            "censored": 0,
            "effective_independent_n": 14,
            "unit": "administrative task",
        },
        "preconditions_checked": {
            "authority_candidates": candidates,
            "absolute_root": str(ROOT.resolve()),
            "owned_output_parent_exists": (ROOT / RESULT).parent.is_dir(),
            "backend": "host_cpu_aggregation",
            "gpu_used": False,
            "required_inputs": [
                {"path": row["path"], "exists": row["exists"], "sha256": row["sha256"]}
                for row in sources
            ],
        },
        "validation_receipts": receipts,
        "mutations": mutations,
        "frozen_affected_scope": {
            "path": str(RAW / "frozen_affected_scope.json"),
            "sha256": sha256_file(ROOT / RAW / "frozen_affected_scope.json"),
        },
        "verifier_is_oracle": True,
        "claim_scope": {
            "kind": "administrative_exact_authority",
            "science": "unmeasured",
            "fresh_generalization_eligible": False,
            "RAGTruth": "all_640_families_exposed_development",
            "ARC": "public_unmeasured",
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
        "random_seed": 7795,
        "phase_spans": phases,
        "duration_s": time.monotonic() - START,
        "method_map_path": str(METHOD),
        "selected_roadmap_path": str(authority.relative_to(ROOT)),
        "authority_snapshot_paths": [
            {"path": str(path), "sha256": sha256_file(ROOT / path)}
            for path in (DESIGN_SNAPSHOT, YAML_SNAPSHOT)
        ],
        "repository_collection_healthy": None,
        "prior_failure_retirement": [
            {
                "task_id": task["id"],
                "prior": item,
                "retired_if_same_verdict": item["retire_if_same_verdict"]
                and any(row["honest_verdict"] == item["verdict"] for row in prior),
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
        "verifier_is_oracle": "Fixture truth does not show hidden generalization.",
        "claim_scope": "Fixture truth does not show hidden generalization.",
        "inference_substrate": "Floors follow invoked work.",
        "inference_substrate_class": "Floors follow invoked work.",
        "MODEL_SPECS": "Citing a model is not invoking it.",
        "model_specs": "Citing a model is not invoking it.",
        "model_invocation_counts": "Citing a model is not invoking it.",
        "contract_ready_score": "Administrative agreement is not scientific benefit.",
        "authority_snapshot_paths": "Rollover must not invalidate past tests.",
        "method_map_path": "Method ownership and limits must remain durable.",
    }
    value["field_principles"] = {
        key: principles.get(key, "Exact source evidence limits this field.")
        for key in (*value, "reproducibility_checksum")
    }
    value["field_principles"]["acceptance_gates"] = {
        key: "Unmeasured science remains null; failed validation sets readiness to zero."
        for key in gates
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def terminal_plan(candidate: Path, raw: Path) -> list[CommandSpec]:
    """Run exact-candidate readers in separate bounded processes."""
    python = str(ROOT / ".venv/bin/python")
    return [
        CommandSpec(
            "real_entrypoint_e2e",
            (python, "-u", str(CLI), "--cold-validate", str(candidate), "--raw", str(raw)),
            "exact_candidate",
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", "--json", str(candidate)),
            "exact_candidate",
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_candidate",
        ),
    ]


def run_experiment(run_date: str) -> dict:
    """Use frozen scope, durable receipts and atomic output for one run."""
    progress("preflight", "before")
    if run_date != "20260928":
        raise ValueError("V678 contract run date must be 20260928")
    authority, roadmap, candidates = resolve_authority(ROOT)
    design = (ROOT / DESIGN).read_text()
    comparison = compare_contract(design, roadmap)
    snapshots = compare_contract(
        (ROOT / DESIGN_SNAPSHOT).read_text(), yaml.safe_load((ROOT / YAML_SNAPSHOT).read_text())
    )
    if (
        comparison != snapshots
        or sha256_file(ROOT / DESIGN) != sha256_file(ROOT / DESIGN_SNAPSHOT)
        or sha256_file(authority) != sha256_file(ROOT / YAML_SNAPSHOT)
    ):
        raise ValueError("V678 live authorities differ from frozen snapshots")
    sources = source_hashes(ROOT, inputs(authority))
    prior_inventory(ROOT)
    raw_dir = ROOT / RAW
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = raw_dir / "rows.json"
    atomic_json(raw, comparison["rows"])
    mutation_names = (
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
    mutations = [
        {
            "mutation": name,
            "rejected": not compare_contract(design, mutate(roadmap, name))["passed"],
        }
        for name in mutation_names
    ]
    atomic_json(raw_dir / "mutations.json", mutations)
    if not all(row["rejected"] for row in mutations):
        raise RuntimeError("private mutation escaped the contract reader")
    phases = [span("preflight", START, 14)]
    progress("preflight", "after", 14)

    private = Path(tempfile.mkdtemp(prefix="exp7795-", dir="/tmp"))
    Path("/tmp/exp7795-validation/pytest").mkdir(parents=True, exist_ok=True)
    Path("/tmp/exp7795-validation/coverage").mkdir(parents=True, exist_ok=True)
    frozen = json.loads((raw_dir / "frozen_affected_scope.json").read_text())
    plan = build_scoped_commands(
        ROOT,
        frozen["test_paths"],
        frozen["changed_modules"],
        static_paths=frozen["static_paths"],
        basetemp=Path("/tmp/exp7795-validation/pytest"),
        coverage_file=Path("/tmp/exp7795-validation/coverage/.coverage"),
    )
    plan += build_repository_check_plan(ROOT, authority)
    if [list(item.argv) for item in plan] != [row["argv"] for row in frozen["commands"]]:
        raise ValueError("frozen validation argv changed")
    atomic_json(raw_dir / "validation_plan.json", frozen)
    progress("validation", "before")
    begin = time.monotonic()
    receipts = run_commands(ROOT, plan, log_dir=raw_dir / "validation_logs", heartbeat_s=60)
    phases.append(span("validation", begin, len(receipts)))
    progress("validation", "after", len(receipts))

    candidate = private / "candidate.json"
    value = artifact(authority, candidates, comparison, sources, receipts, phases, mutations)
    atomic_json(candidate, value)
    progress("cold_replay", "before")
    begin = time.monotonic()
    if not cold_validate(candidate, raw, ROOT):
        raise RuntimeError("independent row reduction failed")
    phases.append(span("cold_replay", begin, 14))
    progress("cold_replay", "after", 14)
    progress("terminal", "before")
    begin = time.monotonic()
    terminal = run_commands(
        ROOT, terminal_plan(candidate, raw), log_dir=raw_dir / "terminal_logs", heartbeat_s=60
    )
    phases.append(span("terminal", begin, len(terminal)))
    progress("terminal", "after", len(terminal))
    value = artifact(
        authority, candidates, comparison, sources, receipts + terminal, phases, mutations
    )
    atomic_json(candidate, value)
    progress("exact_terminal", "before")
    exact = run_commands(
        ROOT, terminal_plan(candidate, raw), log_dir=raw_dir / "exact_terminal_logs", heartbeat_s=60
    )
    atomic_json(raw_dir / "exact_reader_receipts.json", exact)
    progress("exact_terminal", "after", len(exact))
    if not all(row["passed"] for row in exact):
        value["honest_verdict"] = "complete_disqualified_v678_terminal_validation"
        value["verdict_class"] = "disqualified"
        value["contract_ready_score"] = 0
        value["acceptance_gate_results"].update({"validity": False, "readiness": 0})
        value["flagged_adversarial"] = any(
            row["name"] == "adversarial_verify" and not row["passed"] for row in exact
        )
        value["reproducibility_checksum"] = canonical_hash(
            {key: item for key, item in value.items() if key != "reproducibility_checksum"}
        )
    atomic_json(ROOT / RESULT, value)
    progress("publication", "after", 14)
    return value


def main(argv: list[str] | None = None) -> int:
    """Expose the real run and a fresh-process historical reader."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260928")
    parser.add_argument("--cold-validate")
    parser.add_argument("--raw")
    args = parser.parse_args(argv)
    if args.cold_validate:
        valid = cold_validate(Path(args.cold_validate), Path(args.raw), ROOT)
        print(json.dumps({"valid": valid}), flush=True)
        return 0 if valid else 1
    run_experiment(args.date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
