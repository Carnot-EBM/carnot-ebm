"""REQ-VERIFY-8247: validate private evidence before exposing one terminal primary.

Current checks own execution readiness. Upstream scientific failures remain
finished dispositions, with their original bytes and failure receipts intact.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import v709_execution as x
from carnot.reporting import v711_current_runner as current
from carnot.reporting import v712_capstone_evidence as e
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v710_contract_replay import require_reference, snapshot
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]


def commands(private: Path) -> list[Json]:
    """Measure only added statements while retaining existing consumer and E2E checks."""
    with (
        patch.object(current.q, "OWNED", e.OWNED),
        patch.object(current.q, "TEST", e.TEST),
        patch.object(current.q, "CLI", e.CLI),
    ):
        plan = current.commands(private)
    plan.append(
        dict(
            name="private_E2E021",
            argv=[
                str(e.ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_restricted_decision_audit_8210.py",
                "--basetemp=" + str(private / "e2e021"),
            ],
            deadline=180,
            expected=0,
            scope="owned",
        )
    )
    return list(plan)


def terminal_plan(path: Path) -> list[Json]:
    """Freeze this capstone's own replay operand before measurement begins."""
    with patch.object(current.q, "CLI", e.CLI):
        return list(current.terminal_plan(path))


def replay(path: Path) -> Json:
    """Rebuild reductions so changing an aggregate and its checksum still fails."""
    value = json.loads(path.read_bytes())
    for ref in [
        value["work_reference"],
        *value["source_artifact_hashes"],
        *value["code_config_hashes"],
        *value["raw_shard_hashes"],
    ]:
        require_reference(ref)
    for receipt in value["validation_receipts"]:
        for stream in ["stdout", "stderr"]:
            if sha256_file(Path(receipt[stream + "_path"])) != receipt[stream + "_sha256"]:
                raise ValueError("validation_stream_drift")
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    _, tasks = parse_design(
        Path(work["references"][0]["snapshot_path"]).read_text(), milestone=e.MILESTONE
    )
    if tasks != work["tasks"]:
        raise ValueError("contract_primitive_drift")
    activation = next(
        (
            r
            for r in work["references"]
            if r["path"].endswith("/research-roadmap.yaml") and r.get("exists")
        ),
        None,
    )
    if work["activation_snapshot"]:
        if (
            work["activation_snapshot"] != activation
            or yaml.safe_load(Path(activation["snapshot_path"]).read_bytes())["tasks"] != tasks
        ):
            raise ValueError("activation_primitive_drift")
    elif activation and not any(g["upstream"] == "activation_or_history" for g in work["failures"]):
        raise ValueError("activation_primitive_drift")
    if work["history"]:
        ref = next(r for r in work["references"] if r["path"].endswith("/" + e.HISTORY))
        source = json.loads(Path(ref["snapshot_path"]).read_bytes())
        if any(
            work["history"][k] != source.get(k, [])
            for k in [
                "rows",
                "actual_executed_task_count",
                "conductor_pre_gate_count",
                "required_checks_passed",
                "upstream_validation_failures",
            ]
        ):
            raise ValueError("history_primitive_drift")
    for row in work["dispositions"]:
        if row["task_id"] in work["primaries"]:
            ref = next(r for r in work["references"] if r["path"] == row["path"])
            source = json.loads(Path(ref["snapshot_path"]).read_bytes())
            eligible = (
                source["required_checks_passed"]
                and not source["flagged_adversarial"]
                and source["verdict_class"] in {"positive", "circular_positive", "null"}
            )
            if source != work["primaries"][row["task_id"]] or any(
                row[k] != v
                for k, v in dict(
                    eligible=eligible,
                    excluded=not eligible,
                    honest_verdict=source["honest_verdict"],
                    verdict_class=source["verdict_class"],
                    source_counts={k: source[k] for k in e.COUNTS},
                ).items()
            ):
                raise ValueError("source_primitive_drift")
    reconstructed = deepcopy(work)
    with tempfile.TemporaryDirectory(prefix="carnot8247-replay-") as directory:
        e.science(reconstructed, Path(directory))
    if any(reconstructed[k] != work[k] for k in ["H1", "H2"]):
        raise ValueError("hypothesis_primitive_drift")
    rebuilt = e.reduce(work, value["validation_receipts"])
    for key, observed in rebuilt.items():
        if value[key] != observed:
            raise ValueError("primitive_reduction_drift:" + key)
    return dict(passed=True, rows_checksum=canonical_hash(rebuilt["rows"]))


def write_note(primary: Path, destination: Path) -> None:
    """Derive prose numbers from checked bytes to avoid inventing scientific wins."""
    value = json.loads(primary.read_bytes())
    lines = [
        "# V712 outcomes — 2026-10-07",
        "",
        f"Source: `{primary}`; `{sha256_file(primary)}`.",
        f"Verdict: {value['honest_verdict']}; dispositions {value['completed_count']}/14; executed {value['actual_executed_task_count']}.",
        f"Execution readiness {value['capstone_execution_ready_score']}; science readiness {value['science_ready_score']}; current LLM calls0.",
        "",
        "| Task | Verdict | Next evidence condition |",
        "|---|---|---|",
    ]
    next_conditions = {
        8234: "Changed extraction signal or independent cohort.",
        8235: "Independent later decision and retention gain.",
        8236: "More independent cold-inclusive sweeps.",
        8237: "Changed extraction signal or independent cohort.",
        8238: "Changed extraction signal or independent cohort.",
        8239: "Changed extraction signal or independent cohort; margin-only objective retired.",
        8240: "Decision changes with lower cost under the unchanged delayed protocol.",
        8241: "Later decision gain and independently sealed retention.",
        8242: "More independent sweeps plus actual matched Rust/Python10x.",
        8243: "Authenticated new environment outcomes with cross-game arm overlap.",
        8244: "Authenticated KV260 operations, transfer and whole-request spans.",
        8245: "Passing owned checks and authenticated PolarFire parity.",
        8246: "Dated cable, port and power evidence changing the physical operand.",
        8247: "Close the three PRD gaps with independent benefit evidence.",
    }
    lines += [
        f"| {r['task_id']} | {r['honest_verdict']} | {next_conditions[r['experiment_id']]} |"
        for r in value["rows"]
    ]
    for name in ["H1", "H2"]:
        branch = value[name]
        stats = branch["statistics"] or {}
        lines += [
            "",
            f"{name}: {branch['status']}, alpha=.025; complete {stats.get('completed_count')}/{branch['intended_count']}.",
        ]
        boot = stats.get("bootstrap_diagnostics", stats.get("block_bootstrap_diagnostics", []))
        lines += [f"Primitive bootstrap diagnostics: `{json.dumps(boot, sort_keys=True)}`."]
    lines += ["", "Three PRD gaps remain open:"]
    lines += [f"- {','.join(g['requirements'])}: {g['remaining']}" for g in value["three_prd_gaps"]]
    service = value["request_accounting"]
    lines += [
        "",
        f"Service: {service['observed_measurement_calls']}/96 valid requests; valid completions `{service['valid_completion_counts']}`; cold costs `{json.dumps(service['cold_costs'])}`.",
        f"{service['sweep_support_limit']} NFR-01 remains unmet.",
        f"ARC new outcomes: {value['arc_evidence']['new_outcome_count']}.",
        f"Paper ready: {value['paper_ready']}; unmet gates: {value['unmet_gates']}.",
        "V711 retains12 executed and2 pre-gate dispositions. Validation repairs grant no scientific benefit.",
        "Static margin-only retirement does not rewrite the separate unchanged delayed-protocol null. Both independent generalization scores remain0.",
        "No generator weight updates or external publication occurred. Repository health diagnostics retain their actual exits.",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Keep private parents alive through bounded children and publish checked bytes."""
    e.progress("start_no_model_load", 0, 14)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261007"], default="20261007")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--private-fixture", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        output = (args.output or e.ROOT / "results" / (e.NAME + ".json")).absolute()
        if args.private_fixture and (
            output.is_relative_to(e.ROOT / "results") or args.root == e.ROOT
        ):
            raise ValueError("private_fixture_requires_private_root_and_output")
        raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True, mode=0o700)
        start, wall = time.monotonic_ns(), time.time_ns()
        with tempfile.TemporaryDirectory(prefix="carnot8247-", dir="/tmp") as directory:
            private = Path(directory)
            probe = private / "write_probe"
            probe.write_bytes(b"actual writable private scratch")
            runtime = dict(
                private_path=str(private),
                mode=oct(private.stat().st_mode & 0o777),
                writable=probe.read_bytes() == b"actual writable private scratch",
                executable=sys.executable,
                free_bytes=os.statvfs(private).f_bavail * os.statvfs(private).f_frsize,
            )
            plan = [] if args.private_fixture else commands(private)
            controls = x.pytest_plan(private / "controls")
            preflight = current.qualified.precondition_command()
            terminal = terminal_plan(
                output.parent / "raw" / output.stem / "terminal_candidate.json"
            )
            _, tasks = parse_design((args.root / e.DESIGN).read_text(), milestone=e.MILESTONE)
            branches = [
                dict(
                    name="upstream_audit_" + str(8234 + i),
                    argv=[
                        str(e.ROOT / ".venv/bin/python"),
                        "-u",
                        str(e.ROOT / e.AUDITS[8234 + i]),
                        "--cold-replay",
                        str(args.root / t["deliverable"]),
                    ],
                    deadline=180,
                    expected=0,
                    scope="upstream",
                    task_id=t["id"],
                )
                for i, t in enumerate(tasks[:-1])
                if 8234 + i in e.AUDITS
            ]
            code = [
                snapshot(e.ROOT / p, raw / "code", str(i))
                for i, p in enumerate(
                    [
                        *e.OWNED,
                        e.TEST,
                        "python/carnot/reporting/v709_execution.py",
                        "python/carnot/reporting/primary_publication.py",
                        "python/carnot/reporting/roadmap_contract.py",
                        "python/carnot/verify/margin_decision_audit_8239.py",
                        "python/carnot/verify/delayed_benefit_audit_8241.py",
                        "scripts/publication_gate.py",
                        "scripts/experiment_template.py",
                        "ops/exclusion_manifest.yaml",
                        "AGENTS.md",
                        "CODEX.md",
                        "CLAUDE.md",
                        "ops/e2e-test-plan.md",
                        "openspec/capabilities/research-reporting/spec.md",
                        "openspec/capabilities/verification/spec.md",
                    ]
                )
            ]
            atomic_json(
                raw / "validation_manifest.json",
                dict(
                    commands=plan,
                    controls=controls,
                    preflight=preflight,
                    terminal_commands=terminal,
                    upstream_audits=branches,
                    runtime=runtime,
                    frozen_before_measurement_ns=time.monotonic_ns(),
                ),
            )
            e.progress("preconditions_before")
            receipts = x.execute([preflight], raw / "preconditions")
            measurement_start = time.monotonic_ns()
            work = e.measure(args.root, raw)
            measurement_end = time.monotonic_ns()
            work["branch_audit_receipts"] = []
            if not args.private_fixture:
                for spec in branches:
                    row = next(r for r in work["dispositions"] if r["task_id"] == spec["task_id"])
                    if row["eligible"]:
                        receipt = x.execute([spec], raw / "upstream_audits")[0]
                        work["branch_audit_receipts"].append(receipt)
                        if not receipt["passed"]:
                            work["failures"].append(
                                e.operand(
                                    row["task_id"],
                                    row["path"],
                                    row["sha256"],
                                    "audit_specific_cold_replay_exit",
                                    0,
                                    receipt["actual_exit"],
                                )
                            )
            atomic_json(
                raw / "consumer_manifest.json",
                dict(
                    schema="carnot.v712.consumers.v1",
                    inputs=work["consumer_manifest"],
                    task_objects_sha256=canonical_hash(work["tasks"]),
                ),
            )
            atomic_json(raw / "work.json", work)
            e.progress("preconditions_after", 13, 1)
            receipts += x.execute(controls, raw / "controls")
            receipts += x.execute([p for p in plan if p["scope"] == "owned"], raw / "validation")
            health = x.execute(
                [p for p in plan if p["scope"] == "repository_health"], raw / "health"
            )
            gates = x.execute(
                [
                    dict(
                        name="publication_gate",
                        argv=[
                            str(e.ROOT / ".venv/bin/python"),
                            "scripts/publication_gate.py",
                            "--json",
                        ],
                        expected=0,
                        deadline=90,
                        scope="publication",
                    )
                ],
                raw / "publication",
            )[0]
            publication_gates = json.loads(Path(gates["stdout_path"]).read_bytes())
            coverage_path = private / "coverage.json"
            cov = json.loads(coverage_path.read_bytes()) if coverage_path.is_file() else {}
            atomic_json(raw / "coverage.json", cov)
            end = time.monotonic_ns()
            value = e.reduce(work, receipts)
            value.update(
                experiment_id=8247,
                task_id="exp8247-capstone",
                milestone=e.MILESTONE,
                run_date=args.date,
                schema="carnot.v712.capstone.v1",
                random_seed=7128247,
                duration_s=(end - start) / 1e9,
                inference_substrate="aggregation_from_upstream_artifacts",
                inference_substrate_class="no_model_load",
                MODEL_SPECS=[],
                trained_head_specs=dict(
                    fitted_here=False,
                    source_experiments=[8237, 8240],
                    role="Imported decision heads used only by primitive replay; historical Qwen acquisition supplies no current calls.",
                ),
                model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
                call_ledger=[],
                preconditions_checked=work["consumer_manifest"],
                runtime_preconditions=runtime,
                validation_receipts=receipts,
                upstream_audit_receipts=work["branch_audit_receipts"],
                repository_health=dict(owned=False, receipts=health),
                publication_gate_receipt=gates,
                coverage_totals=cov.get("totals", {}),
                coverage_statement_counts=cov.get("files", {}),
                code_config_hashes=code,
                source_artifact_hashes=work["references"],
                work_reference=dict(
                    path=str(raw / "work.json"), sha256=sha256_file(raw / "work.json")
                ),
                raw_shard_hashes=[
                    dict(path=str(p), sha256=sha256_file(p))
                    for p in sorted(raw.rglob("*"))
                    if p.is_file()
                ],
                terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
                cited_upstream_artifacts=[
                    dict(
                        path=r["path"],
                        sha256=r["sha256"],
                        task_id=r["task_id"],
                        fields_imported=list(work["primaries"].get(r["task_id"], {})),
                    )
                    for r in work["dispositions"]
                ],
                invocation_argv=sys.orig_argv,
                measurement_clocks=dict(
                    started_monotonic_ns=start,
                    ended_monotonic_ns=end,
                    started_wall_ns=wall,
                    ended_wall_ns=time.time_ns(),
                    owner_pid=os.getpid(),
                ),
                phase_spans=[
                    dict(
                        phase="primitive_reduction",
                        started_monotonic_ns=measurement_start,
                        ended_monotonic_ns=measurement_end,
                        duration_s=(measurement_end - measurement_start) / 1e9,
                    ),
                    *[
                        dict(
                            phase=r["name"],
                            started_monotonic_ns=r["started_monotonic_ns"],
                            ended_monotonic_ns=r["ended_monotonic_ns"],
                            duration_s=r["duration_s"],
                        )
                        for r in [*receipts, *health, *work["branch_audit_receipts"], gates]
                    ],
                ],
                paper_ready=publication_gates["paper_ready"],
                unmet_gates=publication_gates["unmet_gates"],
                external_publication_authorized=False,
                **{
                    name.lower(): publication_gates["gates"][name]
                    for name in ["G1", "G2", "G3", "G4"]
                },
                methodology_note="Exact fourteen outcome custody; current H1/H2 primitives are recomputed with alpha=.025 each and equal-information controls. Qualified nulls retire unchanged objectives. Historical Qwen provenance is cached; zero current model calls. Exposure grants no independent generalization. Two service sweeps do not meet NFR-01; board prerequisites remain distinct.",
            )
            value["reproducibility_checksum"] = canonical_hash([value["work_reference"], code])
            value["field_principles"] = {
                k: "Bind actual invocation bytes and denominators; readiness supplies no independent benefit."
                for k in value
            }
            value["field_principles"].update(
                completed_count="Fourteen reconciled dispositions; producer executions are counted separately.",
                task_dispositions="Source row references retain every original arm, metric denominator and missing mask.",
                historical_v711="Immutable history retains12 executed and2 pre-gate outcomes plus original owned failures.",
                validation_repairs="New qualification does not rewrite historical science or establish benefit.",
                H1="Static evaluator primitives and equal-information controls retain alpha=.025 and the original128 slots.",
                H2="Unchanged delayed protocol and sealed retention are distinct from the static margin-only mechanism.",
                request_accounting="Valid requests, cold starts and two-sweep support do not establish matched Rust/Python10x.",
                board_obligations="Separate hardware prerequisites remain visible even when host simulation passes.",
                retirements="Retire the exact failed margin-only objective until extraction changes or independent sources arrive.",
                repository_health="Bounded repository-wide failures retain their actual exits and never become owned passes.",
                trained_head_specs="Imported head provenance is separate from zero current generator loads and calls.",
            )
            value = normalize_artifact_for_template_write(value)
            e.progress("publication_before", 13, 1)

            def validate(candidate: Path) -> Json:
                """Cold and negative replay inspect the same bytes as unchanged consumers."""
                checks = x.execute(terminal, raw / "terminal")
                tampered = deepcopy(value)
                tampered["completed_count"] = 13
                negative = private / "tampered.json"
                atomic_json(negative, tampered)
                checks += x.execute(
                    [
                        dict(
                            name="negative_cold_replay",
                            argv=[
                                str(e.ROOT / ".venv/bin/python"),
                                "-u",
                                str(e.ROOT / e.CLI),
                                "--cold-replay",
                                str(negative),
                            ],
                            expected=1,
                            deadline=180,
                            scope="terminal",
                        )
                    ],
                    raw / "terminal",
                )
                return dict(
                    passed=all(r["passed"] and r["normal_exit"] for r in checks), checks=checks
                )

            publication = publish_primary(output, value, validate)
            atomic_json(
                raw / "terminal_validation.json",
                dict(
                    publication=publication,
                    required_checks_passed=value["required_checks_passed"],
                    normal_process_exit=True,
                ),
            )
            if not args.private_fixture:
                write_note(output, args.root / "docs/research-notes/v712-outcomes.md")
        e.progress("complete", 14, 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration) as error:
        print(json.dumps(dict(passed=False, error=str(error))), flush=True)
        return 1
