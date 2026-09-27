#!/usr/bin/env python3
"""Run the V676 contract entrypoint (REQ-REPORT-7767)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import yaml

from carnot.experiment_7767_v676_contract_methods import (
    CLI,
    DESIGN,
    METHOD,
    MODULE,
    PRESERVED,
    RAW,
    RESULT,
    ROOT,
    TEST,
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
    """Expose real elapsed time and completed units at each phase boundary."""
    print(
        f"[exp7767] phase={phase} event={event} "
        f"elapsed_s={time.monotonic() - start:.3f} completed_units={units}",
        flush=True,
    )


def span(start: float, begin: float, phase: str, units: int, digest: str) -> dict:
    """Measure a disjoint phase without adding a synthetic duration."""
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
    """Hash exact current bytes; mutable documents are never given old digests."""
    return [
        DESIGN,
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
    ]


def build_artifact(
    authority: Path,
    comparison: dict,
    raw: Path,
    receipts: list[dict],
    spans: list[dict],
    candidates: list[dict],
    sources: list[dict],
) -> dict:
    """Build terminal administrative evidence with science gates left unmeasured."""
    failures = failed_checks(comparison, sources)
    required = [r for r in receipts if r["scope"] != "repository_diagnostic"]
    validated = bool(required) and all(r["passed"] for r in required)
    ready = comparison["passed"] and not failures and validated
    if failures:
        verdict, verdict_class = "complete_blocked_v676_contract_inputs", "blocked"
    elif ready:
        verdict, verdict_class = (
            "complete_circular_positive_v676_contract_methods",
            "circular_positive",
        )
    else:
        verdict, verdict_class = "complete_disqualified_v676_contract_validation", "disqualified"
    prior = prior_inventory(ROOT)
    gates = {
        key: None for key in ("probability_quality", "decision_benefit", "retention", "efficiency")
    }
    gates.update({"validity": ready, "readiness": int(ready)})
    value = {
        "schema": "carnot.exp7767.v676.contract_methods.v1",
        "experiment_id": 7767,
        "milestone": "2026.09.676",
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
        "repository_collection_healthy": False,
        "scope_closure_rows": [
            {
                "scope": "exp7767",
                "changed_module": str(MODULE),
                "direct_test": str(TEST),
                "entrypoint": str(CLI),
                "reused_reader_test": "tests/python/test_experiment_7753_v675_contract_methods.py",
                "transitive_consumers": [],
                "state": "frozen_current_scope",
            },
            *[
                {
                    "scope": task["id"],
                    "state": "future_producer_must_freeze_and_prove_closure",
                    "repository_collection_healthy": False,
                }
                for task in yaml.safe_load(authority.read_text())["tasks"][1:]
            ],
        ],
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
        "source_artifact_hashes": sources,
        "v675_producer_inventory": prior,
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
        "random_seed": 7767,
        "phase_spans": spans,
        "duration_s": sum(s["duration_s"] for s in spans),
        "method_map_path": str(METHOD),
        "selected_roadmap_path": str(authority.relative_to(ROOT)),
        "preserved_v675_design_sha256": sha256_file(ROOT / PRESERVED),
    }
    principles = {
        "experiment_id": "An artifact must have a unique current owner.",
        "milestone": "An artifact must have a unique current owner.",
        "run_date": "An artifact must have a unique current owner.",
        "honest_verdict": "A terminal record must not waste attempts on unchanged inputs.",
        "verdict_class": "The claim class travels with the evidence.",
        "flagged_adversarial": "Invalid evidence must not open downstream gates.",
        "gate_check_summary": "Missing producers and failed scientific thresholds are different causes.",
        "rows": "Aggregates must be recomputable without rerunning science.",
        "acceptance_gate_results": "A working protocol is not evidence of benefit.",
        "duration_s": "Duration must describe actual work without padding.",
        "phase_spans": "Duration must describe actual work without padding.",
        "random_seed": "A third party needs the same experiment inputs.",
        "reproducibility_checksum": "A third party needs the same experiment inputs.",
        "sample_size_budget": "Repeated views and seeds do not increase independent family count.",
        "source_artifact_hashes": "A missing producer cannot be replaced with a convenient old result.",
        "preconditions_checked": "Access and validity must be established before expensive work.",
        "validation_receipts": "All registered checks must pass before readiness opens.",
        "verifier_is_oracle": "Execution truth and independent semantic verification differ.",
        "claim_scope": "Execution truth and independent semantic verification differ.",
        "inference_substrate": "Duration floors must match the invoked substrate.",
        "inference_substrate_class": "Duration floors must match the invoked substrate.",
        "MODEL_SPECS": "An upstream model is not a current invocation.",
        "model_specs": "An upstream model is not a current invocation.",
        "model_invocation_counts": "An upstream model is not a current invocation.",
        "contract_ready_score": "Readiness certifies executable instructions, not science.",
        "method_map_path": "External work must inform a falsifiable local question.",
        "repository_collection_healthy": "A narrow scope cannot imply repository-wide health.",
        "scope_closure_rows": "A narrow scope cannot imply repository-wide health.",
    }
    value["field_principles"] = {
        key: principles.get(key, "Exact source evidence limits this field.") for key in value
    }
    value["field_principles"]["acceptance_gates"] = {
        key: "A working protocol is not evidence of benefit; unmeasured values stay null."
        for key in gates
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def verify_candidate(path: Path, raw: Path) -> bool:
    """Check exact candidate bytes against current sources and independent rows."""
    value = json.loads(path.read_text())
    checksum = value.pop("reproducibility_checksum", None)
    if checksum != canonical_hash(value):
        return False
    sources = [r for r in value["source_artifact_hashes"] if r["role"] == "current_input"]
    return cold_validate(ROOT, raw, sources) and value["rows"] == json.loads(raw.read_text())


def terminal_plan(candidate: Path, raw: Path) -> list[CommandSpec]:
    """Use fresh processes for replay and the two terminal readers."""
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
    """Validate the frozen current scope, then atomically publish one terminal result."""
    start = time.monotonic()
    progress(start, "preflight", "before")
    if run_date != "20260927":
        raise ValueError("V676 contract run date must be 20260927")
    authority, roadmap, candidates = resolve_authority(ROOT)
    comparison = compare_contract((ROOT / DESIGN).read_text(), roadmap)
    sources = source_hashes(ROOT, authority.relative_to(ROOT), inputs(authority))
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
    private = Path(tempfile.mkdtemp(prefix="exp7767-", dir="/tmp"))
    basetemp = private / "basetemp"
    for leaf in ("focused", "coverage", "repository"):
        (basetemp / leaf).parent.mkdir(parents=True, exist_ok=True)
    coverage = private / "coverage/.coverage"
    coverage.parent.mkdir(parents=True, exist_ok=True)
    scoped = build_scoped_commands(
        ROOT,
        [str(TEST), "tests/python/test_experiment_7753_v675_contract_methods.py"],
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
        authority, comparison, raw, [*receipts, *diagnostic], phases, candidates, sources
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
        authority, comparison, raw, [*receipts, *diagnostic, *terminal], phases, candidates, sources
    )
    atomic_json(candidate, value)
    progress(start, "exact_terminal_replay", "before")
    exact = run_commands(
        ROOT, terminal_plan(candidate, raw), log_dir=raw_dir / "exact_terminal_logs", heartbeat_s=60
    )
    progress(start, "exact_terminal_replay", "after", len(exact))
    atomic_json(raw_dir / "exact_reader_receipts.json", exact)
    if not all(item["passed"] for item in exact):
        value["honest_verdict"] = "complete_disqualified_v676_terminal_validation"
        value["verdict_class"] = "disqualified"
        value["contract_ready_score"] = 0
        value["acceptance_gate_results"]["validity"] = False
        value["acceptance_gate_results"]["readiness"] = 0
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
    """Expose a real run and a separate fresh-process candidate reader."""
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
