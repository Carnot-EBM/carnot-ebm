"""REQ-REPORT-9095-V775: finish reconciliation even when prerequisite science is blocked."""

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

import yaml

from carnot.reporting import scoped_provenance_9014 as runner
from carnot.reporting import v775_capstone_evidence as evidence
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    ZERO_INVOCATION_COUNTS,
)
from carnot.reporting.encoder_head_diagnosis_8972 import checksum, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v756_branch_qualification import measured, verify_measured
from carnot.reporting.v763_capstone_reduction import require
from carnot.reporting.v764_capstone_sources import retain

Json = dict[str, Any]
ROOT = evidence.ROOT
NAME, TASK = "experiment_9095_v775_capstone", "exp9095-capstone"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v775_capstone_9095.py"
OWNED = [
    "python/carnot/reporting/v775_capstone.py",
    "python/carnot/reporting/v775_capstone_evidence.py",
    CLI,
]
DIRECT = [
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_current_work_receipt.py",
    "tests/python/test_conductor_gates.py",
]
REUSED = list(
    dict.fromkeys(
        [
            *runner.REUSED,
            "python/carnot/reporting/service_cost_9093.py",
            "python/carnot/reporting/susceptibility_9094.py",
            "python/carnot/reporting/roadmap_contract.py",
            "scripts/publication_gate.py",
        ]
    )
)
MODEL_SPECS: list[Json] = []
HEALTH = ROOT / "results/raw" / NAME / "development/global_python_suite.receipt.json"
ORIGINAL_PLAN = runner.validation_plan


def validation_plan(private: Path) -> list[CommandSpec]:
    """Allow six hundred seconds for real retained-row replay controls under coverage."""
    return [
        CommandSpec(c.name, c.argv, c.scope, 600 if c.name == "owned_tests" else c.timeout_s)
        for c in ORIGINAL_PLAN(private)
    ]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries while the shared supervisor heartbeats every long child call."""
    print(f"[exp9095] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def binding() -> Iterator[None]:
    """Freeze this capstone's scope while preserving the existing validation policy."""
    with patch.multiple(
        runner,
        NAME=NAME,
        TASK=TASK,
        CLI=CLI,
        OWNED=OWNED,
        TEST=TEST,
        DIRECT=DIRECT,
        REUSED=REUSED,
        progress=progress,
        validation_plan=validation_plan,
    ):
        yield


def collect(raw: Path, run_date: str, root: Path = ROOT) -> Json:
    """Authenticate inputs and freeze commands before reducing available sibling evidence."""
    progress("preconditions_before")
    started = time.monotonic_ns()
    raw.mkdir(parents=True, mode=0o700, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix="carnot9095-", dir="/var/tmp"))
    with binding():
        manifest = runner.scope_manifest(private)
    work = evidence.authenticate(root, raw)
    work["checks"] += [
        evidence.check(raw, "run_date", "20261115", run_date, "date"),
        evidence.check(raw, "private_mode", 0o700, raw.stat().st_mode & 0o777, "private_storage"),
    ]
    code = [retain(Path(r["path"]), raw, r["sha256"]) for r in manifest["code_test_hashes"]]
    work.update(
        raw=str(raw),
        private=str(private),
        run_date=run_date,
        started_ns=started,
        frozen_ns=time.monotonic_ns(),
        validation_scope_manifest=manifest,
        code=code,
        validation=dict(receipts=[], coverage=dict(complete=False)),
        publication_receipts=[],
        publication_sources=[],
    )
    work["repository_health_references"] = []
    if HEALTH.is_file():
        health = json.loads(HEALTH.read_bytes())
        for path in [HEALTH, Path(health["stdout_path"]), Path(health["stderr_path"])]:
            work["repository_health_references"].append(
                retain(path, raw, reference(path)["sha256"])
            )
    progress(
        "preconditions_after", len(work["checks"]), sum(not r["passed"] for r in work["checks"])
    )
    progress("publication_gate_before")
    with binding():
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
                ROOT / "results" / gate["source"]
                for gate in work["publication_gate"]["gates"].values()
                if "source" in gate
            ),
        ]
        work["publication_sources"] = [
            retain(path, raw, reference(path)["sha256"]) for path in dict.fromkeys(sources)
        ]
    progress("publication_gate_after")
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
        inference_substrate_details=dict(scope="V775 reconciliation only"),
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
    """Administrative readiness never turns a missing scientific prerequisite into partial success."""
    failures = [r for r in work["checks"] if not r["passed"]]
    science = (
        evidence.reduce(work)
        if not failures
        else dict(
            seven_bound_rows=[],
            ten_bound_rows=[],
            human_summary={},
            cost_summary={},
            susceptibility_summary={},
            arc_summary={},
            continuation_decisions=[],
            retirement_recommendations=[],
        )
    )
    with binding():
        owned = bool(
            runner.receipts_pass(work)
            and verify_measured(work["validation"]["coverage"], OWNED)
            and not work["validation"].get("terminal_failed")
        )
    ready = owned and not failures
    cls = "blocked" if failures or owned else "disqualified"
    verdict = (
        "complete_blocked_" + (failures[0]["check"] if failures else "declared_gate")
        if cls == "blocked"
        else "complete_disqualified_owned_validation"
    )
    outcomes = evidence.outcomes(work)
    outcomes.append(
        dict(
            task_id=TASK,
            experiment_id=9095,
            disposition="self_slot",
            qualified=ready,
            honest_verdict=verdict,
            verdict_class=cls,
        )
    )
    gates = [
        dict(evidence.edge(g, work["values"]), downstream=t["id"])
        for t in work["tasks"]
        for g in t["gated_on"]
    ]
    declared_paths = {t["id"]: str(ROOT / t["deliverable"]) for t in work["tasks"]}
    for row in gates:
        row["path"] = row["path"] or declared_paths[row["upstream"]]
    upstream = [
        r
        for e in work["values"].values()
        for r in e["value"].get("gate_check_summary", [])
        if isinstance(r, dict)
    ]
    publication = work.get(
        "publication_gate",
        dict(
            gates=dict.fromkeys(["G1", "G2", "G3", "G4"]),
            paper_ready=False,
            unmet_gates=["not_executed"],
        ),
    )
    archive = work["snapshots"].get("research-complete.yaml")
    archived = (
        yaml.safe_load(Path(archive["path"]).read_bytes())["milestones"][-1]["id"]
        if archive
        else "unavailable"
    )
    rows = [
        dict(
            unit_id=t,
            family="administrative_accounting",
            group_id=t,
            arm="reconciliation",
            condition="task_slot",
            seed=None,
            status="completed",
            metric="slot_accounted",
            absolute_metric=1,
            value=1,
            numerator=1,
            denominator=1,
            censor_reason=None,
            exclusion_reason=None,
        )
        for t in evidence.TASKS
    ]
    value: Json = dict(
        experiment_id=9095,
        task_id=TASK,
        milestone="2026.11.775",
        run_date=work["run_date"],
        honest_verdict=verdict,
        verdict_class=cls,
        verifier_is_oracle=True,
        claim_scope="Fourteen administrative outcomes; graph labels are an executable oracle and graph wins are circular_positive. No new detector, online or general verifier benefit.",
        exposure_scope="LPHB is exposed development; no independent human evaluation or generalization. Historical graph access and source authority remain blocked.",
        independent_generalization_score=0,
        current_science_paper_ready=False,
        MODEL_SPECS=[],
        model_specs=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        current_llm_invocations=0,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        historical_model_provenance=[
            dict(
                task_id=t,
                provenance=e["value"].get("historical_model_provenance", []),
                MODEL_SPECS=e["value"].get("MODEL_SPECS", []),
                scope="historical only",
            )
            for t, e in work["values"].items()
        ],
        rows=rows,
        intended_count=14,
        attempted_count=14,
        completed_count=14,
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        effective_independent_groups=0,
        reconciliation_ready_score=int(ready),
        required_checks_passed=ready,
        flagged_adversarial=not ready,
        task_outcome_rows=outcomes,
        full_task_objects=work["tasks"],
        full_task_digest=evidence.canonical_hash(work["tasks"]),
        declared_gate_rows=gates,
        confirmed_branch_results=[r for r in outcomes[:-1] if r["qualified"]],
        scientific_scope_ledger=[
            dict(branch=b, scope=s, independent_generalization_score=0)
            for b, s in [
                ("graph", "executable oracle; unavailable predictions"),
                ("human", "exposed LPHB development; unavailable embeddings"),
                ("cost", "administrative clocks; scientific service unmeasured"),
                ("susceptibility", "32 CPU equation cases; physical speed and joules unmeasured"),
                ("ARC", "unchanged historical context only"),
            ]
        ],
        **science,
        archive_lag=dict(
            observed_archive_end=archived,
            current_milestone="2026.11.775",
            archive_lagged=archived != "2026.11.775",
            archive_modified=False,
            activation_owner="conductor",
        ),
        staged_authority_status="missing_file_after_activation",
        protected_authority_hashes=[
            dict(source_path=k, **r)
            for k, r in sorted(work["snapshots"].items())
            if not k.startswith("results/")
        ],
        gate_check_summary=[*work["checks"], *upstream, *gates],
        acceptance_gates=dict(
            reconciliation_ready_score=int(ready),
            scientific_readiness=0,
            independent_generalization_score=0,
        ),
        source_references=[
            *[r for _, r in sorted(work["snapshots"].items())],
            *work["publication_sources"],
        ],
        raw_shard_hashes=[r for _, r in sorted(work["operands"].items())],
        code_config_hashes=work["code"],
        current_work_receipts=[work["current_work_reference"]],
        validation_scope_manifest=work["validation_scope_manifest"],
        repository_health=dict(
            references=work["repository_health_references"], affects_owned_readiness=False
        ),
        owned_statement_coverage=work["validation"]["coverage"],
        validation_receipts=[
            *work["publication_receipts"],
            *work["validation"]["receipts"],
            *work["validation"].get("terminal_receipts", []),
        ],
        duration_s=(work["ended_ns"] - work["started_ns"]) / 1e9,
        random_seed=775095,
        work_reference=reference(Path(work["raw"]) / "work.json"),
        publication_scope="historical_FoVer",
        publication_gate_receipt=[
            dict(
                r,
                command_sha256=evidence.canonical_hash(r.get("command_argv", [])),
                source_hashes=work["publication_sources"],
            )
            for r in work["publication_receipts"]
        ],
        **{k: publication["gates"][k] for k in ["G1", "G2", "G3", "G4"]},
        paper_ready=publication["paper_ready"],
        unmet_gates=publication["unmet_gates"],
        activation_authorized=False,
        external_publication_authorized=False,
        methodology_note="Authenticate fixed primaries and primitive rows, preserve scheduler skips and missing science, recompute descriptive counts and keep unchanged custody failure terminal. Historical FoVer gates do not qualify current graph or human science.",
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "field_principles", "reproducibility_checksum"],
        "Hashed primitive denominators bind accounting; administrative readiness grants no new scientific claim.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild every headline from pinned retained bytes, including missing science states."""
    try:
        value = json.loads(path.read_bytes())
        require(checksum(value) == value["reproducibility_checksum"], "checksum_drift")
        ref = value["work_reference"]
        require(reference(Path(ref["path"])) == ref, "work_drift")
        work = json.loads(Path(ref["path"]).read_bytes())
        for name, saved in work["snapshots"].items():
            require(saved["sha256"] == evidence.PINS[name], "original_pin")
        refs = [
            *work["snapshots"].values(),
            *work["operands"].values(),
            *work["code"],
            work["current_work_reference"],
            *work["repository_health_references"],
            *work["publication_sources"],
        ]
        for saved in refs:
            require(reference(Path(saved["path"]))["sha256"] == saved["sha256"], "operand_drift")
        for entry in work["values"].values():
            require(
                entry["reference"] == work["snapshots"][entry["selected_name"]]
                and entry["value"]
                == evidence.project(json.loads(Path(entry["reference"]["path"]).read_bytes())),
                "producer_drift",
            )
        for receipt in work["publication_receipts"]:
            require(
                reference(Path(receipt["log_path"]))["sha256"] == receipt["log_sha256"]
                and work.get("publication_gate")
                == json.loads(Path(receipt["log_path"]).read_bytes()),
                "publication_gate_drift",
            )
        require(build(work) == value, "reduction_drift")
        return dict(passed=True, replay_passed=True, slots=14)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(passed=False, replay_passed=False, reason=str(error))


def validate(work: Json) -> Json:
    """Execute the frozen affected scope once and reopen the real coverage database."""
    private, raw = Path(work["private"]), Path(work["raw"])
    with binding(), patch.dict(os.environ, COVERAGE_RCFILE=str(private / "coverage.ini")):
        plan = runner.validation_plan(private)
        require(
            [dict(name=c.name, argv=list(c.argv), timeout_s=c.timeout_s) for c in plan]
            == work["validation_scope_manifest"]["commands"],
            "frozen_commands",
        )
        receipts = [
            runner.run_check(c, raw / "validation", work["validation_scope_manifest"]) for c in plan
        ]
    report = measured([private / ".coverage"], OWNED, raw / "coverage")
    report.pop("rows", None)
    return dict(receipts=receipts, coverage=report)


def main(argv: list[str] | None = None) -> int:
    """Publish only checked bytes; private replay never changes an upstream determination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261115"], default="20261115")
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
            work["validation"] = validate(work)
            progress("owned_validation_after")
        work["ended_ns"] = time.monotonic_ns()
        atomic_json(raw / "work.json", work)
        value = build(work)
        atomic_json(raw / "candidate.json", value)
        with binding():
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
