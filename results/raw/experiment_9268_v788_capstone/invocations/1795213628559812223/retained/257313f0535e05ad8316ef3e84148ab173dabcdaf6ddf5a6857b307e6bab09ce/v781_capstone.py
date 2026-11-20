"""REQ-REPORT-9179-V781: reconcile fourteen slots without claiming scientific success."""

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import scoped_provenance_9014 as runner
from carnot.reporting import v775_capstone as legacy
from carnot.reporting import v781_capstone_evidence as evidence
from carnot.reporting.current_work_receipt import atomic_json, build_current_work_receipt
from carnot.reporting.encoder_head_diagnosis_8972 import (
    checksum as checksum,
    reference as reference,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v763_capstone_reduction import require
from carnot.reporting.v764_capstone_sources import retain
from carnot.reporting.v781_capstone_reduction import reduce

Json = dict[str, Any]
ROOT = evidence.ROOT
NAME, TASK = "experiment_9179_v781_capstone", "exp9179-capstone"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v781_capstone_9179.py"
OWNED = [
    "python/carnot/reporting/v781_capstone.py",
    "python/carnot/reporting/v781_capstone_evidence.py",
    "python/carnot/reporting/v781_capstone_reduction.py",
    CLI,
]
DIRECT = legacy.DIRECT
REUSED = [
    *legacy.REUSED,
    *legacy.OWNED,
    "python/carnot/reporting/v781_contract_evidence.py",
    "openspec/change-proposals/v781-contract-inputs.json",
    evidence.CONFIG,
]
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries while the shared child supervisor reports blocked operations."""
    print(f"[exp9179] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def binding() -> Iterator[None]:
    """Reuse checked publication and validation without editing historical producers."""
    with patch.multiple(
        legacy,
        NAME=NAME,
        TASK=TASK,
        CLI=CLI,
        TEST=TEST,
        OWNED=OWNED,
        DIRECT=DIRECT,
        REUSED=REUSED,
        evidence=evidence,
        progress=progress,
        build=build,
        replay=replay,
    ):
        yield


def collect(raw: Path, run_date: str, root: Path = ROOT) -> Json:
    """Freeze commands and retain authenticated operands before reducing outcomes."""
    progress("preconditions_before")
    started = time.monotonic_ns()
    raw.mkdir(parents=True, mode=0o700, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix="carnot9179-", dir="/var/tmp"))
    with binding(), legacy.binding():
        manifest = runner.scope_manifest(private)
    work = evidence.authenticate(root, raw)
    work["checks"].extend(
        [
            evidence.check(raw, "run_date", "20261117", run_date, "date"),
            evidence.check(
                raw, "private_mode", 0o700, raw.stat().st_mode & 0o777, "private_storage"
            ),
        ]
    )
    work.update(
        raw=str(raw),
        private=str(private),
        run_date=run_date,
        started_ns=started,
        frozen_ns=time.monotonic_ns(),
        validation_scope_manifest=manifest,
        code=[retain(Path(r["path"]), raw, r["sha256"]) for r in manifest["code_test_hashes"]],
        validation=dict(receipts=[], coverage=dict(complete=False)),
        publication_receipts=[],
        publication_sources=[],
        repository_health_references=[],
    )
    health = ROOT / "results/raw" / NAME / "development/global_python_suite.receipt.json"
    if health.is_file():
        receipt = json.loads(health.read_bytes())
        work["repository_health_references"] = [
            retain(p, raw, reference(p)["sha256"])
            for p in [health, ROOT / receipt["stdout_path"], ROOT / receipt["stderr_path"]]
        ]
    with binding(), legacy.binding():
        receipt = runner.run_check(
            CommandSpec(
                "publication_gate",
                (str(ROOT / ".venv/bin/python"), "-u", "scripts/publication_gate.py", "--json"),
                "historical_FoVer",
                60,
            ),
            raw / "publication",
            manifest,
        )
    work["publication_receipts"] = [receipt]
    work["checks"].append(
        evidence.check(
            Path(receipt["log_path"]),
            "normal_exit",
            True,
            receipt["passed"] and receipt["normal_exit"],
            "publication_gate_execution",
        )
    )
    if receipt["passed"]:
        work["publication_gate"] = json.loads(Path(receipt["log_path"]).read_bytes())
        sources = [
            ROOT / "scripts/publication_gate.py",
            *(
                ROOT / "results" / g["source"]
                for g in work["publication_gate"]["gates"].values()
                if "source" in g
            ),
        ]
        work["publication_sources"] = [
            retain(p, raw, reference(p)["sha256"]) for p in dict.fromkeys(sources)
        ]
    progress(
        "preconditions_after", len(work["checks"]), sum(not r["passed"] for r in work["checks"])
    )
    if all(r["passed"] for r in work["checks"]):
        progress("primitive_reduction_before")
        reduce(work)
        progress("primitive_reduction_after", 13, 1)
    work["ended_ns"] = time.monotonic_ns()
    current = build_current_work_receipt(
        run_id=str(started),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details=dict(
            scope="V781 CPU accounting of retained evidence; no model work"
        ),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=started,
        ended_monotonic_ns=work["ended_ns"],
    )
    atomic_json(raw / "current_work.json", current)
    work["current_work_reference"] = reference(raw / "current_work.json")
    atomic_json(raw / "work.json", work)
    return work


def build(work: Json) -> Json:
    """Administrative readiness cannot promote blocked science or disqualified controls."""
    normalized = dict(work, tasks=[dict(t, gated_on=t.get("gated_on", [])) for t in work["tasks"]])
    with binding(), patch.object(evidence, "reduce", lambda _: reduce(work)):
        value = _build(normalized)
    value.update(
        experiment_id=9179,
        milestone="2026.11.781",
        random_seed=evidence.PROTOCOL["seed"],
        requested_inference_substrate_class="no_model_load",
        generated_tokens=0,
        verifier_is_oracle=False,
        preconditions_checked=work["preconditions_checked"],
        model_invocation_status="not_invoked_empty_requested_list",
        exposure_scope="exposed_development",
        sample_size_budget=dict(
            intended=14,
            attempted=14,
            completed=14,
            failed=0,
            censored=0,
            effective_independent_groups=0,
        ),
        open_gaps={"GAP-ORACLE-DISTINCT": "open", "GAP-DETECTOR-AUROC-4208": "open"},
        staged_authority_status="fourteen complete active YAML objects compared with visible and embedded V781 design",
        full_task_objects=work["tasks"],
        full_task_digest=evidence.canonical_hash(work["tasks"]),
    )
    value["claim_scope"] = dict(
        exposure_scope="Original exposed LPHB900/300 development panel; zero new independent groups",
        independent_generalization_score=0,
        allowed_claims=[
            "fourteen terminal administrative outcomes",
            "retained diagnostic observations without qualified causal correction",
            "offline ARC method controls with failed affected checks",
        ],
        unmeasured_claims=[
            "detector benefit",
            "continuous learning benefit",
            "Brier and ranking changes",
            "full service cost",
            "live ARC generalization",
            "hardware benefit",
        ],
    )
    value["scientific_scope_ledger"] = [
        dict(branch=branch, scope=scope, independent_generalization_score=0)
        for branch, scope in [
            ("validation", "Unqualified causal correction; exact remaining obligation union"),
            ("static", "Exposed development; static predictions and both bounds unmeasured"),
            ("learning", "Exposed development; future and retention bounds unmeasured"),
            ("cost", "Complete scientific service clocks and power unmeasured"),
            ("ARC", "Disqualified offline method controls; no measured live reachability"),
        ]
    ]
    value["model_invocation_counts"].update(
        current_llm_invocations=0,
        loads_attempted=0,
        loads_completed=0,
        loads_failed=0,
        forward_calls=0,
        generated_tokens=0,
    )
    value["task_outcome_rows"][-1].update(
        experiment_id=9179,
        primary_path=f"results/{NAME}.json",
        primary_status="self_slot",
        scheduler_alternative=None,
        quarantine=None,
        gate_operands=[],
        gate_check_summary=[],
        readiness=dict(reconciliation_ready_score=value["reconciliation_ready_score"]),
    )
    value["outcome_rows"] = value["task_outcome_rows"]
    value["publication_gates"] = dict(
        **{k: value[k] for k in ["G1", "G2", "G3", "G4", "paper_ready", "unmet_gates"]},
        publication_scope="historical_FoVer",
        current_science_paper_ready=False,
    )
    for row in value["rows"]:
        row.update(qid=None, sibling=None, seed=evidence.PROTOCOL["seed"])
    value["archive_lag"].update(
        current_milestone="2026.11.781",
        archive_lagged=value["archive_lag"]["observed_archive_end"] != "2026.11.781",
    )
    value["source_references"] = [
        dict(
            r,
            producer_path=r.get("original_path", r["path"]),
            imported_fields=["original_terminal_outcomes_and_per_unit_evidence"],
        )
        for r in value["source_references"]
    ]
    health = work["repository_health_references"]
    if health:
        receipt = json.loads(Path(health[0]["path"]).read_bytes())
        value["repository_health"].update(
            passed=receipt["passed"], timed_out=receipt["timed_out"], exit_code=receipt["exit_code"]
        )
    value["gate_failure_rows"] = [r for r in value["gate_check_summary"] if not r["passed"]]
    value["acceptance_gates"] = {
        k: dict(
            condition=v,
            principle="Frozen operands and all affected checks establish administrative readiness only",
        )
        for k, v in dict(
            full_contract="fourteen complete objects match",
            static="both original static bounds; missing predictions remain unmeasured",
            learning="all seven original learning bounds independent of static",
            validation="100 percent new owned statements, zero exclusions, all affected checks pass",
            scope="no LLM, no independent generalization or publication authority",
        ).items()
    }
    value["methodology_note"] = (
        "Authenticate original V781 sources and full contracts, preserve every scheduler cascade skip and disqualified primary, recompute complete cohort and obligation unions and deterministic ARC controls. No model load, generation, benchmark, scientific prediction, physical-board work or external publication occurs. A complete scientific null differs from unmeasured science."
    )
    value["field_principles"].update(
        {
            k: "Exact retained operands and executed checks support accounting only; absent science remains blocked"
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


_build, _replay = legacy.build, legacy.replay


def replay(path: Path) -> Json:
    """Rebuild from retained original bytes so rehashing cannot bless invented outcomes."""
    with binding():
        return dict(_replay(path))


def main(argv: list[str] | None = None) -> int:
    """Expose only candidate bytes accepted by real replay and terminal checkers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261117"], default="20261117")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--input-root", type=Path, default=ROOT)
    parser.add_argument("--from-work", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.from_work and args.output.absolute().is_relative_to(ROOT):
        parser.error("private replay publication requires output outside the checkout")
    if args.cold_replay:
        report = replay(args.cold_replay)
        print(json.dumps(report), flush=True)
        return 0 if report["passed"] else 1
    raw = args.output.absolute().parent / "raw" / NAME / "invocations" / str(time.time_ns())
    try:
        if args.from_work:
            work = json.loads(args.from_work.read_bytes())
            work["raw"] = str(raw)
        else:
            work = collect(raw, args.date, args.input_root)
            require((time.monotonic_ns() - work["started_ns"]) / 1e9 < 3300, "measurement_deadline")
            progress("owned_validation_before")
            with binding():
                work["validation"] = legacy.validate(work)
            progress("owned_validation_after")
        work["ended_ns"] = time.monotonic_ns()
        atomic_json(raw / "work.json", work)
        atomic_json(raw / "candidate.json", build(work))
        with binding(), legacy.binding():
            report = runner.terminal(
                raw / "candidate.json", raw / "preflight", work["validation_scope_manifest"]
            )
            work["validation"].update(
                terminal_failed=not report["passed"], terminal_receipts=report["receipts"]
            )
            atomic_json(raw / "work.json", work)
            require(report["passed"], "terminal_checks")
            require((time.monotonic_ns() - work["started_ns"]) / 1e9 < 4800, "hard_deadline")
            value = build(work)
            require(
                args.from_work is not None
                or value["required_checks_passed"]
                or any(not r["passed"] for r in work["checks"]),
                "owned_validation",
            )
            published = publish_primary(
                args.output,
                value,
                lambda p: runner.terminal(
                    p, raw / "publication_checks", work["validation_scope_manifest"]
                ),
            )
        progress("complete", 14, 0)
        print(json.dumps(published), flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(passed=False, reason=str(error))), flush=True)
        return 1
