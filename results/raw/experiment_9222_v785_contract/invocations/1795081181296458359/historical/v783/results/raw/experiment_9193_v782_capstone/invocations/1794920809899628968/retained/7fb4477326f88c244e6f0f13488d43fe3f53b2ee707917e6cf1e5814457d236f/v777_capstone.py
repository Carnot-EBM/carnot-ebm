"""REQ-REPORT-9123-V777: reconcile fourteen outcomes with no current inference."""

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

from carnot.reporting import v775_capstone as legacy
from carnot.reporting import scoped_provenance_9014 as runner
from carnot.reporting import v777_capstone_evidence as evidence
from carnot.reporting.current_work_receipt import atomic_json, build_current_work_receipt
from carnot.reporting.encoder_head_diagnosis_8972 import checksum, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v763_capstone_reduction import require
from carnot.reporting.v764_capstone_sources import retain

Json = dict[str, Any]
ROOT = evidence.ROOT
NAME, TASK = "experiment_9123_v777_capstone", "exp9123-capstone"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v777_capstone_9123.py"
OWNED = [
    "python/carnot/reporting/v777_capstone.py",
    "python/carnot/reporting/v777_capstone_evidence.py",
    CLI,
]
DIRECT = legacy.DIRECT
REUSED = list(
    dict.fromkeys(
        [
            *legacy.REUSED,
            *legacy.OWNED,
            "python/carnot/reporting/v776_capstone_evidence.py",
            "python/carnot/reporting/v777_contract.py",
            "python/carnot/reporting/qwen_judge_9107.py",
            "python/carnot/reporting/complete_cost_9122.py",
            "openspec/change-proposals/v777-capstone-inputs.json",
        ]
    )
)
MODEL_SPECS: list[Json] = []
_build, _replay = legacy.build, legacy.replay


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase boundaries while the shared supervisor heartbeats child work."""
    print(f"[exp9123] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def binding() -> Iterator[None]:
    """Reuse validation and atomic publication without modifying historical producers."""
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
        collect=collect,
        build=build,
        replay=replay,
    ):
        yield


def collect(raw: Path, run_date: str, root: Path = ROOT) -> Json:
    """Freeze commands before authenticating all consumed immutable operands."""
    progress("preconditions_before")
    started = time.monotonic_ns()
    raw.mkdir(parents=True, mode=0o700, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix="carnot9123-", dir="/var/tmp"))
    with binding(), legacy.binding():
        manifest = runner.scope_manifest(private)
    work = evidence.authenticate(root, raw)
    work["checks"].extend(
        [
            evidence.check(raw, "run_date", "20261116", run_date, "date"),
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
            for p in [health, Path(receipt["stdout_path"]), Path(receipt["stderr_path"])]
        ]
    with binding(), legacy.binding():
        cmd = CommandSpec(
            "publication_gate",
            (str(ROOT / ".venv/bin/python"), "-u", "scripts/publication_gate.py", "--json"),
            "historical_FoVer",
            60,
        )
        receipt = runner.run_check(cmd, raw / "publication", manifest)
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
        evidence.reduce(work)
        progress("primitive_reduction_after", 13, 1)
    work["ended_ns"] = time.monotonic_ns()
    current = build_current_work_receipt(
        run_id=str(started),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details=dict(
            scope="V777 administrative reconciliation; historical cached features are provenance only"
        ),
        inference_substrate_class="no_model_load",
        execution_venue="local_cpu",
        started_monotonic_ns=started,
        ended_monotonic_ns=work["ended_ns"],
    )
    atomic_json(raw / "current_work.json", current)
    work["current_work_reference"] = reference(raw / "current_work.json")
    atomic_json(raw / "work.json", work)
    return work


def build(work: Json) -> Json:
    """Complete accounting may qualify while every scientific branch remains blocked."""
    with binding():
        value = _build(work)
    value.update(
        experiment_id=9123,
        milestone="2026.11.777",
        random_seed=777123,
        requested_inference_substrate_class="no_model_load",
        generated_tokens=0,
        sample_size_budget=dict(
            administrative_slots=14,
            static_test_qids=100,
            online_future_qids=40,
            repeated_retention_qids=100,
            energy_test_qids=100,
        ),
        verifier_is_oracle=False,
        claim_scope="Fourteen administrative outcomes and conditional exposed-development evidence only; no independent human benefit or generalization.",
        exposure_scope="LPHB remains exposed development. Historical cached representations are provenance, not current inference. V776 judge stays quarantined; graph exact-oracle successes are circular_positive; susceptibility is CPU-only.",
        open_gaps={"GAP-ORACLE-DISTINCT": "open", "GAP-DETECTOR-AUROC-4208": "open"},
        staged_authority_status="V777 full task objects independently compared with authenticated active queue and visible design",
        historical_code_resolutions=work["historical_code_resolutions"],
    )
    value["task_outcome_rows"][-1].update(
        experiment_id=9123,
        primary_path=f"results/{NAME}.json",
        primary_status="self_slot",
        scheduler_alternative=None,
        readiness=dict(reconciliation_ready_score=value["reconciliation_ready_score"]),
    )
    value["archive_lag"].update(
        current_milestone="2026.11.777",
        archive_lagged=value["archive_lag"]["observed_archive_end"] != "2026.11.777",
    )
    for key, rows in evidence.bounds().items():
        value.setdefault(key, rows)
    value.setdefault("closed_scope_rows", [])
    value["gate_failure_rows"] = [r for r in value["gate_check_summary"] if not r["passed"]]
    health = work["repository_health_references"]
    if health:
        receipt = json.loads(Path(health[0]["path"]).read_bytes())
        value["repository_health"].update(
            passed=receipt["passed"],
            timed_out=receipt["timed_out"],
            exit_code=receipt["exit_code"],
            scope="Repository-wide run; failures and timeout are separate from the affected validation scope.",
        )
    value["methodology_note"] = (
        "Authenticate full task objects, declared gates and immutable upstream inputs before dependent reductions. Recompute historical groups, diagnostic judge and CPU rows. Missing current scientific predictions stay terminal blocked; no model, generation, gameplay, board work or external publication occurs. Administrative readiness is independent of scientific benefit. Preserve the five/six/two comparisons without a composite scientific score."
    )
    value["field_principles"].update(
        {
            "full_task_objects": "Complete ordered prompts, cohorts and gates prevent silent contract substitution.",
            "task_outcome_rows": "Scheduler alternatives account for slots without replacing missing declared science.",
            "declared_gate_rows": "Missing files, absent fields and observed zero are distinct operands.",
            "static_bound_rows": "All five matched static controls retain their original denominators and undefined intervals.",
            "online_bound_rows": "All six future/retention bounds remain conjunctive; repeated retention adds no independent qids.",
            "energy_bound_rows": "Both probability and donor contrasts are required; gauge sensitivity limits the claim.",
            "reconciliation_ready_score": "Complete correct accounting and passing owned checks establish administrative readiness only.",
            "current_llm_invocations": "Only current owned model events count; historical calls cannot fill zero.",
            "historical_code_resolutions": "Authenticated source snapshots preserve provenance despite later checkout edits.",
            "independent_generalization_score": "Exposed human development cannot close independent verification gaps.",
            "cost_summary": "Unavailable service clocks and power remain null, separate from administrative time.",
            "duration_s": "Monotonic actual reconciliation time excludes historical inference and padding.",
            "owned_statement_coverage": "Real unit and CLI execution must cover every new statement without exclusions.",
            "validation_receipts": "Frozen argument vectors, actual exits and exact logs establish owned validation.",
            "MODEL_SPECS": "An empty current list prohibits model loads and generated tokens.",
            "retirement_recommendations": "Only authenticated same-verdict comparisons retire their declared unchanged scope.",
            "publication_scope": "G1-G4 retain their historical FoVer scope and authorize no external publication.",
            "rows": "Original identities, arm, seed, support and censor state remain visible.",
            "acceptance_gates": "Administrative completion cannot promote blocked or disqualified scientific evidence.",
            "gate_check_summary": "Exact operands preserve absence, undefined measurements and observed zero without favorable substitution.",
            "requested_inference_substrate_class": "The requested operation class remains separate from upstream historical model provenance.",
            "inference_substrate_class": "CPU reconciliation requires no current model load and no padded runtime floor.",
            "inference_substrate": "Aggregation consumes authenticated retained bytes instead of performing new model inference.",
            "model_invocation_counts": "The empty owned event ledger balances attempts and completions independently of cited calls.",
            "generated_tokens": "No generation may be attributed to this accounting task.",
            "code_config_hashes": "Exact source and configuration bytes bind checks to the implementation measured.",
            "source_references": "Original source identities remain authenticated without changing historical pins.",
            "current_work_receipts": "Owned monotonic intervals and empty invocation events establish current work only.",
            "required_checks_passed": "Every affected frozen check must pass; unrelated repository health remains separately visible.",
            "random_seed": "The fixed protocol identifier does not create another independent scientific unit.",
            "reproducibility_checksum": "All published fields are content bound; replay independently rejects rehashed false aggregates.",
            "verifier_is_oracle": "Human factual labels differ from executable graph oracles; each historical branch retains its own classification.",
            "claim_scope": "Administrative completion makes no claim of useful human verification.",
            "exposure_scope": "Previously exposed development cannot become an independent evaluation through reuse.",
            "sample_size_budget": "Administrative slots, sibling responses and repeated processes have different sampling units.",
            "effective_independent_groups": "Administrative reconciliation adds zero independent scientific groups.",
            "closed_scope_rows": "Closed graph custody and CPU mechanism studies keep their original reopening prerequisites.",
            "open_gaps": "A descriptive development signal cannot close the independent oracle-distinct verifier gaps.",
            "archive_lag": "Scheduling completion and archive state cannot substitute for scientific execution.",
            "external_publication_authorized": "Local artifact validation grants no external publication authority.",
            "activation_authorized": "Accounting cannot activate a roadmap or public default.",
            "field_principles": "Each material field states the separate scientific error it prevents.",
        }
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Cold replay rejects altered retained inputs and rehashed aggregate inventions."""
    with binding():
        return dict(_replay(path))


def main(argv: list[str] | None = None) -> int:
    """Publish only checked bytes within the frozen measurement and validation budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261116"], default="20261116")
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
