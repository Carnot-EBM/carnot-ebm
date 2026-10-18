#!/usr/bin/env python3
"""Run and cold-reduce the V675 contract receipt (REQ-REPORT-7753)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

from carnot.experiment_7753_v675_contract_methods import (
    CLI,
    DESIGN,
    MODULE,
    RAW,
    RESULT,
    ROOT,
    TEST,
    build_artifact,
    cold_validate,
    compare_contract,
    mutate,
    resolve_authority,
)
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)
from carnot.experiment_7573_v662_contract_methods import build_repository_check_plan


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Flush measured phase boundaries, including child and replay work."""
    print(
        f"[exp7753] {phase} {event} elapsed_s={time.monotonic() - start:.3f} "
        f"completed_units={units}",
        flush=True,
    )


def span(start: float, begin: float, phase: str, units: int, checkpoint: str) -> dict:
    """Record disjoint monotonic phase time without a synthetic floor."""
    end = time.monotonic()
    return {
        "phase": phase,
        "start_s": begin - start,
        "end_s": end - start,
        "duration_s": end - begin,
        "run_date": "20260927",
        "completed_units": units,
        "heartbeat_times": [end - start],
        "checkpoint_sha256": checkpoint,
    }


def terminal_plan(candidate: Path, raw: Path) -> list[CommandSpec]:
    """Check candidate bytes in three independent fresh processes."""
    python = str(ROOT / ".venv/bin/python")
    return [
        CommandSpec(
            "cold_replay",
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
    """Validate the registered scope and publish only complete terminal evidence."""
    start = time.monotonic()
    progress(start, "preflight", "before")
    if run_date != "20260927":
        raise ValueError("V675 contract run date must be 20260927")
    authority, roadmap, candidates = resolve_authority(ROOT)
    design = (ROOT / DESIGN).read_text()
    comparison = compare_contract(design, roadmap)
    raw_dir = ROOT / RAW
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = raw_dir / "rows.json"
    atomic_json(raw, comparison["rows"])
    mutations = [
        {
            "mutation": name,
            "rejected": not compare_contract(design, mutate(roadmap, name))["passed"],
        }
        for name in (
            "delete",
            "reorder",
            "title",
            "producer_field",
            "substrate",
            "prior_field",
            "retirement",
        )
    ]
    atomic_json(raw_dir / "mutations.json", mutations)
    if not all(item["rejected"] for item in mutations):
        raise RuntimeError("a private mutation escaped the contract reader")
    phases = [span(start, start, "preflight", 14, sha256_file(raw))]
    progress(start, "preflight", "after", 14)
    private = Path(tempfile.mkdtemp(prefix="exp7753-", dir="/tmp"))
    basetemp = private / "basetemp"
    for leaf in ("focused", "coverage"):
        (basetemp / leaf).parent.mkdir(parents=True, exist_ok=True)
    coverage = private / "coverage/.coverage"
    coverage.parent.mkdir(parents=True, exist_ok=True)
    scoped = build_scoped_commands(
        ROOT,
        [str(TEST)],
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
            "commands": [list(item.argv) for item in plan],
            "authority": str(authority.relative_to(ROOT)),
        },
    )
    progress(start, "validation", "before", 0)
    begin = time.monotonic()
    receipts = run_commands(ROOT, plan, log_dir=raw_dir / "validation_logs", heartbeat_s=60)
    phases.append(span(start, begin, "validation", len(receipts), canonical_hash(receipts)))
    progress(start, "validation", "after", len(receipts))
    candidate = private / "candidate.json"
    value = build_artifact(ROOT, authority, comparison, raw, receipts, phases, candidates)
    atomic_json(candidate, value)
    progress(start, "cold_replay", "before")
    begin = time.monotonic()
    if not cold_validate(value, ROOT, raw):
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
        ROOT, authority, comparison, raw, [*receipts, *terminal], phases, candidates
    )
    atomic_json(candidate, value)
    progress(start, "exact_terminal_replay", "before")
    exact = run_commands(
        ROOT, terminal_plan(candidate, raw), log_dir=raw_dir / "exact_terminal_logs", heartbeat_s=60
    )
    progress(start, "exact_terminal_replay", "after", len(exact))
    atomic_json(raw_dir / "exact_reader_receipts.json", exact)
    if [r["passed"] for r in terminal] != [r["passed"] for r in exact]:
        value["honest_verdict"] = "complete_disqualified_v675_terminal_validation"
        value["verdict_class"] = "disqualified"
        value["contract_ready_score"] = 0
        value["acceptance_gate_results"]["validity"] = False
        value["acceptance_gate_results"]["readiness"] = False
        value["reproducibility_checksum"] = canonical_hash(
            {k: v for k, v in value.items() if k != "reproducibility_checksum"}
        )
    atomic_json(ROOT / RESULT, value)
    progress(start, "publication", "after", 14)
    return value


def main(argv: list[str] | None = None) -> int:
    """Expose real execution and a fresh-process candidate reader."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260927")
    parser.add_argument("--cold-validate")
    parser.add_argument("--raw")
    args = parser.parse_args(argv)
    if args.cold_validate:
        value = json.loads(Path(args.cold_validate).read_text())
        valid = cold_validate(value, ROOT, Path(args.raw))
        print(json.dumps({"valid": valid, "rows": len(value.get("rows", []))}), flush=True)
        return 0 if valid else 1
    run_experiment(args.date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
