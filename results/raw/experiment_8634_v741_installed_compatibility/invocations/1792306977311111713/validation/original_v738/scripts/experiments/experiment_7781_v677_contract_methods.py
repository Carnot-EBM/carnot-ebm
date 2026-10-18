#!/usr/bin/env python3
"""Run the V677 administrative contract (REQ-REPORT-7781)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import yaml

from carnot.experiment_7781_v677_contract_methods import (
    CLI,
    DESIGN,
    DESIGN_SNAPSHOT,
    METHOD,
    MODULE,
    PRESERVED,
    RAW,
    RESULT,
    ROOT,
    TEST,
    YAML_SNAPSHOT,
    cold_validate,
    compare_contract,
    failed_checks,
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


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Print the real phase clock before and after bounded work."""
    print(
        f"[exp7781] phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} "
        f"completed_units={units}",
        flush=True,
    )


def span(start: float, begin: float, phase: str, units: int, digest: str) -> dict:
    """Measure one disjoint phase with no synthetic duration floor."""
    end = time.monotonic()
    return {
        "phase": phase,
        "start_s": begin - start,
        "end_s": end - start,
        "duration_s": end - begin,
        "run_date": "20260927",
        "completed_units": units,
        "heartbeat_times": [end - start],
        "checkpoint_sha256": digest,
    }


def inputs(authority: Path) -> list[Path]:
    """Declare current bytes before validation so absent inputs are explicit."""
    return [
        authority.relative_to(ROOT),
        DESIGN,
        DESIGN_SNAPSHOT,
        YAML_SNAPSHOT,
        PRESERVED,
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
        Path("scripts/harness_fit_lint.py"),
        Path("python/carnot/experiment_7739_v674_contract_methods.py"),
        Path("results/experiment_7752_v674_capstone.json"),
        Path("python/carnot/experiment_7753_v675_contract_methods.py"),
        Path("results/experiment_7766_v675_capstone.json"),
        Path("openspec/change-proposals/research-roadmap-v675-preserved-20260927.md"),
        Path("python/carnot/experiment_7767_v676_contract_methods.py"),
        Path("results/experiment_7780_v676_capstone.json"),
    ]


def build_artifact(
    authority: Path,
    comparison: dict,
    receipts: list[dict],
    spans: list[dict],
    candidates: list[dict],
    sources: list[dict],
) -> dict:
    """Keep administrative success separate from scientific readiness."""
    failures = failed_checks(comparison, sources)
    required = [r for r in receipts if r["scope"] != "repository_diagnostic"]
    for receipt in required:
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
    validated = bool(required) and all(r["passed"] for r in required)
    ready = comparison["passed"] and not failures and validated
    if any(not row["exists"] for row in sources):
        verdict, verdict_class = "complete_blocked_v677_contract_inputs", "blocked"
    elif ready:
        verdict, verdict_class = (
            "complete_circular_positive_v677_contract_methods",
            "circular_positive",
        )
    else:
        verdict, verdict_class = "complete_disqualified_v677_contract_validation", "disqualified"
    prior = prior_inventory(ROOT)
    gates = {
        key: None for key in ("probability_quality", "decision_benefit", "retention", "efficiency")
    }
    gates.update({"validity": ready, "readiness": int(ready)})
    producer_sources = [
        {
            "path": r["producer_path"],
            "role": "v676_declared_producer",
            "exists": r["producer_state"] != "missing",
            "sha256": r["producer_sha256"],
            "date": "2026-09-27",
            "imported_fields": ["verdict_class", "honest_verdict"],
            "eligible": r["producer_state"] == "circular_positive",
        }
        for r in prior
    ]
    value = {
        "schema": "carnot.exp7781.v677.contract_methods.v1",
        "experiment_id": 7781,
        "milestone": "2026.09.677",
        "run_date": "20260927",
        "status": "complete",
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": any(
            r["name"] == "adversarial_verify" and not r["passed"] for r in receipts
        ),
        "gate_check_summary": failures,
        "positive_claim": False,
        "rows": comparison["rows"],
        "contract_comparison": comparison,
        "acceptance_gate_results": gates,
        "contract_ready_score": int(ready),
        "source_artifact_hashes": [*sources, *producer_sources],
        "v676_producer_inventory": prior,
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
            "absolute_root": str(ROOT.resolve()),
            "owned_output_parent_exists": (ROOT / RESULT).parent.is_dir(),
            "backend": "host_cpu_aggregation",
            "gpu_used": False,
            "required_inputs": [
                {"path": r["path"], "exists": r["exists"], "sha256": r["sha256"]} for r in sources
            ],
        },
        "validation_receipts": receipts,
        "affected_file_validation_manifest": {
            "path": str(RAW / "frozen_affected_scope.json"),
            "sha256": sha256_file(ROOT / RAW / "frozen_affected_scope.json"),
            "frozen_before_implementation": True,
        },
        "verifier_is_oracle": True,
        "claim_scope": {
            "kind": "administrative_exact_authority",
            "science": "unmeasured",
            "fresh_generalization_eligible": False,
            "RAGTruth": "exposed_development_unmeasured",
            "ARC": "public_unmeasured",
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "model_specs": [],
        "planned_MODEL_SPECS": [],
        "model_invocation_counts": {
            "loads": 0,
            "generations": 0,
            "forwards": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "loaded_files": [],
        },
        "model_invoked": False,
        "random_seed": 7781,
        "phase_spans": spans,
        "duration_s": sum(s["duration_s"] for s in spans),
        "method_map_path": str(METHOD),
        "selected_roadmap_path": str(authority.relative_to(ROOT)),
        "authority_snapshot_paths": [
            {"path": str(path), "sha256": sha256_file(ROOT / path)}
            for path in (DESIGN_SNAPSHOT, YAML_SNAPSHOT)
        ],
        "repository_collection_healthy": next(
            (r["passed"] for r in receipts if r["name"] == "repository_python_suite_diagnostic"),
            None,
        ),
        "scope_closure_rows": [
            {
                "scope": "exp7781",
                "changed_module": str(MODULE),
                "direct_test": str(TEST),
                "entrypoint": str(CLI),
                "transitive_consumers": [
                    "tests/python/test_experiment_7767_v676_contract_methods.py",
                    "tests/python/test_experiment_7780_v676_capstone.py",
                ],
                "state": "frozen_current_scope",
            }
        ],
    }
    principles = {
        "experiment_id": "Each record needs a unique owner.",
        "milestone": "Each record needs a unique owner.",
        "run_date": "Each record needs a unique owner.",
        "honest_verdict": "Unchanged external inputs must not consume retries.",
        "verdict_class": "Claim strength travels with the result.",
        "flagged_adversarial": "Invalid evidence cannot open a downstream gate.",
        "gate_check_summary": "A missing producer differs from a scientific null.",
        "rows": "A headline must be reproducible from individual units.",
        "acceptance_gate_results": "Working fixtures do not establish benefit.",
        "duration_s": "Real elapsed work determines substrate authenticity.",
        "phase_spans": "Real elapsed work determines substrate authenticity.",
        "random_seed": "Another process must recover the same inputs.",
        "reproducibility_checksum": "Another process must recover the same inputs.",
        "sample_size_budget": "Views and repeated seeds do not create independent families.",
        "source_artifact_hashes": "A convenient old file cannot replace a missing current producer.",
        "preconditions_checked": "Cheap failures should precede expensive work.",
        "validation_receipts": "Every registered requirement must pass before readiness.",
        "verifier_is_oracle": "Deterministic fixture truth is not independent semantic accuracy.",
        "claim_scope": "Deterministic fixture truth is not independent semantic accuracy.",
        "inference_substrate": "Duration floors follow invoked work, not names cited in a report.",
        "inference_substrate_class": "Duration floors follow invoked work, not names cited in a report.",
        "MODEL_SPECS": "Only current invocations belong in model metadata.",
        "model_specs": "Only current invocations belong in model metadata.",
        "model_invocation_counts": "Only current invocations belong in model metadata.",
        "contract_ready_score": "Administrative correctness does not prove a scientific effect.",
        "method_map_path": "A durable method note limits transfer claims.",
        "authority_snapshot_paths": "Later milestones must not invalidate historical tests.",
    }
    value["field_principles"] = {
        key: principles.get(key, "Exact source evidence limits this field.")
        for key in (*value, "reproducibility_checksum")
    }
    value["field_principles"]["acceptance_gates"] = {
        key: "Working fixtures do not establish benefit; unmeasured values stay null."
        for key in gates
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def verify_candidate(path: Path, raw: Path) -> bool:
    """Recompute candidate and snapshot rows in a fresh process."""
    value = json.loads(path.read_text())
    checksum = value.pop("reproducibility_checksum", None)
    sources = [
        row
        for row in value["source_artifact_hashes"]
        if row["path"] in {str(DESIGN_SNAPSHOT), str(YAML_SNAPSHOT)}
    ]
    return (
        checksum == canonical_hash(value)
        and cold_validate(ROOT, raw, sources)
        and value["rows"] == json.loads(raw.read_text())
    )


def terminal_plan(candidate: Path, raw: Path) -> list[CommandSpec]:
    """Declare three bounded external readers for the exact candidate bytes."""
    python = str(ROOT / ".venv/bin/python")
    return [
        CommandSpec(
            "real_entrypoint_e2e",
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


def run_experiment(run_date: str) -> dict:
    """Validate the frozen scope, then atomically publish one terminal record."""
    start = time.monotonic()
    progress(start, "preflight", "before")
    if run_date != "20260927":
        raise ValueError("V677 contract run date must be 20260927")
    authority, roadmap, candidates = resolve_authority(ROOT)
    comparison = compare_contract((ROOT / DESIGN).read_text(), roadmap)
    snapshot = compare_contract(
        (ROOT / DESIGN_SNAPSHOT).read_text(), yaml.safe_load((ROOT / YAML_SNAPSHOT).read_text())
    )
    if (
        comparison != snapshot
        or sha256_file(ROOT / DESIGN) != sha256_file(ROOT / DESIGN_SNAPSHOT)
        or sha256_file(authority) != sha256_file(ROOT / YAML_SNAPSHOT)
    ):
        raise ValueError("V677 live authorities differ from frozen snapshots")
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
        "unknown_producer",
        "gate_field",
        "model",
        "substrate",
        "prior_experiment_id",
        "prior_verdict",
        "prior_addressed_by",
        "prior_retirement",
    )
    mutations = [
        {
            "mutation": name,
            "rejected": not compare_contract((ROOT / DESIGN).read_text(), mutate(roadmap, name))[
                "passed"
            ],
        }
        for name in mutation_names
    ]
    atomic_json(raw_dir / "mutations.json", mutations)
    if not all(item["rejected"] for item in mutations):
        raise RuntimeError("private mutation escaped the contract reader")
    phases = [span(start, start, "preflight", 14, sha256_file(raw))]
    progress(start, "preflight", "after", 14)

    private = Path(tempfile.mkdtemp(prefix="exp7781-", dir="/tmp"))
    basetemp = private / "basetemp"
    for leaf in ("focused", "coverage", "repository"):
        (basetemp / leaf).parent.mkdir(parents=True, exist_ok=True)
    coverage = private / "coverage/.coverage"
    coverage.parent.mkdir(parents=True, exist_ok=True)
    tests = [
        str(TEST),
        "tests/python/test_experiment_7767_v676_contract_methods.py",
        "tests/python/test_experiment_7780_v676_capstone.py",
    ]
    scoped = build_scoped_commands(
        ROOT,
        tests,
        [str(MODULE)],
        static_paths=[str(CLI)],
        basetemp=basetemp,
        coverage_file=coverage,
    )
    plan = [*scoped, *build_repository_check_plan(ROOT, authority)]
    atomic_json(
        raw_dir / "validation_plan.json",
        {
            "frozen_scope_sha256": sha256_file(raw_dir / "frozen_affected_scope.json"),
            "authority": str(authority.relative_to(ROOT)),
            "commands": [list(item.argv) for item in plan],
        },
    )
    progress(start, "validation", "before")
    begin = time.monotonic()
    receipts = run_commands(ROOT, plan, log_dir=raw_dir / "validation_logs", heartbeat_s=60)
    phases.append(span(start, begin, "validation", len(receipts), canonical_hash(receipts)))
    progress(start, "validation", "after", len(receipts))

    progress(start, "repository_diagnostic", "before")
    begin = time.monotonic()
    diagnostic = run_commands(
        ROOT,
        [
            CommandSpec(
                "repository_python_suite_diagnostic",
                (
                    str(ROOT / ".venv/bin/pytest"),
                    "tests/python",
                    "-q",
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    f"--basetemp={basetemp / 'repository'}",
                ),
                "repository_diagnostic",
                900,
            )
        ],
        log_dir=raw_dir / "repository_logs",
        heartbeat_s=60,
    )
    phases.append(span(start, begin, "repository_diagnostic", 1, canonical_hash(diagnostic)))
    progress(start, "repository_diagnostic", "after", 1)

    candidate = private / "candidate.json"
    value = build_artifact(
        authority, comparison, [*receipts, *diagnostic], phases, candidates, sources
    )
    atomic_json(candidate, value)
    progress(start, "cold_replay", "before")
    begin = time.monotonic()
    if not verify_candidate(candidate, raw):
        raise RuntimeError("candidate failed independent row reduction")
    phases.append(span(start, begin, "cold_replay", 14, sha256_file(candidate)))
    progress(start, "cold_replay", "after", 14)

    progress(start, "terminal_readers", "before")
    begin = time.monotonic()
    terminal = run_commands(
        ROOT, terminal_plan(candidate, raw), log_dir=raw_dir / "terminal_logs", heartbeat_s=60
    )
    phases.append(span(start, begin, "terminal_readers", len(terminal), canonical_hash(terminal)))
    progress(start, "terminal_readers", "after", len(terminal))
    value = build_artifact(
        authority, comparison, [*receipts, *diagnostic, *terminal], phases, candidates, sources
    )
    atomic_json(candidate, value)
    progress(start, "exact_terminal_replay", "before")
    exact = run_commands(
        ROOT, terminal_plan(candidate, raw), log_dir=raw_dir / "exact_terminal_logs", heartbeat_s=60
    )
    progress(start, "exact_terminal_replay", "after", len(exact))
    atomic_json(raw_dir / "exact_reader_receipts.json", exact)
    if not all(item["passed"] for item in exact):
        value["honest_verdict"] = "complete_disqualified_v677_terminal_validation"
        value["verdict_class"] = "disqualified"
        value["contract_ready_score"] = 0
        value["acceptance_gate_results"].update({"validity": False, "readiness": 0})
        value["flagged_adversarial"] = any(
            item["name"] == "adversarial_verify" and not item["passed"] for item in exact
        )
        value["reproducibility_checksum"] = canonical_hash(
            {key: item for key, item in value.items() if key != "reproducibility_checksum"}
        )
    atomic_json(ROOT / RESULT, value)
    progress(start, "publication", "after", 14)
    return value


def main(argv: list[str] | None = None) -> int:
    """Expose a real run and a fresh-process cold reader."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260927")
    parser.add_argument("--cold-validate")
    parser.add_argument("--raw")
    args = parser.parse_args(argv)
    if args.cold_validate:
        valid = verify_candidate(Path(args.cold_validate), Path(args.raw))
        print(json.dumps({"valid": valid}), flush=True)
        return 0 if valid else 1
    run_experiment(args.date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
