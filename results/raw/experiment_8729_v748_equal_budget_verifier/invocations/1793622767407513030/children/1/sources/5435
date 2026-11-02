"""REQ-VERIFY-8261: validate current terminal accounting before atomic publication.

The qualified supervisor supplies bounded children and durable clocks/logs. The
capstone owns only its added statements and never upgrades an upstream failure.
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

from carnot.reporting import v709_execution as x
from carnot.reporting import v712_capstone as qualified
from carnot.reporting import v713_capstone_evidence as e
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v710_contract_replay import require_reference, snapshot
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]


def commands(private: Path) -> list[Json]:
    """Reuse qualified consumer and E2E checks, measuring only current statements."""
    with (
        patch.object(qualified.e, "OWNED", e.OWNED),
        patch.object(qualified.e, "TEST", e.TEST),
        patch.object(qualified.e, "CLI", e.CLI),
    ):
        plan = qualified.commands(private)
    config = private / "coverage.ini"
    config.write_text(config.read_text() + "[report]\nexclude_lines=\n")
    return list(plan)


def terminal_plan(path: Path) -> list[Json]:
    """Freeze exact candidate bytes for the current replay and unchanged auditors."""
    with patch.object(qualified.e, "CLI", e.CLI):
        return list(qualified.terminal_plan(path))


def replay(path: Path) -> Json:
    """Fresh primitive reduction rejects drift even when a summary was rehashed."""
    value = json.loads(path.read_bytes())
    if (value["experiment_id"], value["task_id"], value["milestone"]) != (
        8261,
        "exp8261-capstone",
        e.MILESTONE,
    ):
        raise ValueError("invocation_identity_drift")
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
    if work["references"] != value["source_artifact_hashes"]:
        raise ValueError("source_reference_drift")
    for key, observed in e.reduce(work, value["validation_receipts"]).items():
        if value[key] != observed:
            raise ValueError("primitive_reduction_drift:" + key)
    if value["reproducibility_checksum"] != canonical_hash(
        [value["work_reference"], value["code_config_hashes"]]
    ):
        raise ValueError("checksum_drift")
    if value["MODEL_SPECS"] or value["model_invocation_counts"] != ZERO_INVOCATION_COUNTS:
        raise ValueError("current_model_call_drift")
    publication = json.loads(Path(value["publication_gate_receipt"]["stdout_path"]).read_bytes())
    expected = dict(
        paper_ready=publication["paper_ready"],
        unmet_gates=publication["unmet_gates"],
        **{k.lower(): publication["gates"][k] for k in ["G1", "G2", "G3", "G4"]},
    )
    if any(value[k] != v for k, v in expected.items()):
        raise ValueError("publication_reduction_drift")
    return dict(passed=True, rows_checksum=canonical_hash(value["rows"]))


def write_note(primary: Path, destination: Path) -> None:
    """Link every reported number to checked producer bytes and name reopening evidence."""
    value = json.loads(primary.read_bytes())
    lines = [
        "# V713 outcomes — 2026-10-08",
        "",
        f"Primary: [{primary.name}]({primary}); `{sha256_file(primary)}`.",
        f"Verdict: {value['honest_verdict']}. Reconciled {value['completed_count']}/14; actual producer executions {value['actual_executed_task_count']}; missing outputs {value['missing_output_count']}.",
        f"Execution readiness {value['capstone_execution_ready_score']}; science readiness {value['science_ready_score']}; H1/H2 development scores 0/0, alpha .025 each. Both generalization scores remain zero.",
        "H1 source-information/energy benefit and H2 later constraint use/retention are unmeasured: required current science outputs are absent. Absence is not a scientific null.",
        "",
        "| Task | Actual outcome and primitive source | Next evidence condition |",
        "|---|---|---|",
    ]
    for row, retirement in zip(value["rows"], value["retirements"]):
        source = row.get("path", str(primary))
        lines.append(
            f"| {row['task_id']} | [{row['honest_verdict']}]({source}) | {retirement['reopening_condition']} |"
        )
    lines += [
        "",
        *[f"- {','.join(g['requirements'])}: {g['remaining']}" for g in value["three_prd_gaps"]],
        "",
        f"Publication gate: paper_ready={value['paper_ready']}; unmet={value['unmet_gates']}. Historical V712 paper readiness supplies no V713 benefit.",
        f"Qualified hardware evidence improved={value['evidence_improved']}; learning improved={value['learning_improved']}. This improvement supplies no V713 scientific benefit.",
        "PolarFire parity is oracle-defined hardware validation. GateMate needs changed physical evidence. ARC requires newly authenticated outcomes; unchanged frontier triggers no game rerun.",
        "Acquisition totals are unavailable without current capture ledgers; compatible-kernel bounds do not establish request-scale speedup.",
        "The V712 audit_specific_cold_replay_exit=1 remains in historical_v712.gate_check_summary. Its margin-only retirement remains historical; absent V713 science cannot retire a new mechanism.",
        "No generator loads, weight updates or external publication occurred. Repository health diagnostics retain their actual exits separately.",
    ]
    for board in value["board_obligations"]:
        data = board["evidence"]
        primitive = (
            data.get("board_reference")
            or data.get("primitive_reference")
            or data.get("replay_input_reference")
            or {}
        )
        lines.append(
            f"Device evidence {board['board']}: executions={data.get('current_device_execution_count')}; parity={data.get('host_parity')}; board CPU seconds={data.get('board_cpu_seconds')}; compatible-kernel bound={data.get('ideal_whole_request_bound')}; primitive [{board['board']}]({primitive.get('path', board['path'])})."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Retain private scratch through bounded checks and expose only accepted bytes."""
    e.progress("start_no_model_load", 0, 14)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261008"], default="20261008")
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
        with tempfile.TemporaryDirectory(prefix="carnot8261-", dir="/tmp") as directory:
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
            preflight = qualified.current.qualified.precondition_command()
            terminal = terminal_plan(
                output.parent / "raw" / output.stem / "terminal_candidate.json"
            )
            audits = e.audit_plan(args.root)
            code = [
                snapshot(e.ROOT / p, raw / "code", str(i))
                for i, p in enumerate(
                    [
                        *e.OWNED,
                        e.TEST,
                        "python/carnot/reporting/v709_execution.py",
                        "python/carnot/reporting/v712_capstone.py",
                        "python/carnot/reporting/v712_capstone_evidence.py",
                        "python/carnot/reporting/primary_publication.py",
                        "python/carnot/reporting/roadmap_contract.py",
                        "scripts/experiment_template.py",
                        "scripts/publication_gate.py",
                        "ops/exclusion_manifest.yaml",
                        "openspec/capabilities/research-reporting/spec.md",
                        "openspec/capabilities/verification/spec.md",
                        "openspec/change-proposals/v713-evidence-intervention-protocol.json",
                        "AGENTS.md",
                        "CODEX.md",
                        "CLAUDE.md",
                        "ops/e2e-test-plan.md",
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
                    science_audits=audits,
                    runtime=runtime,
                    frozen_before_measurement_ns=time.monotonic_ns(),
                ),
            )
            e.progress("preconditions_before")
            receipts = x.execute([preflight], raw / "preconditions")
            measurement_start = time.monotonic_ns()
            work = e.measure(args.root, raw)
            measurement_end = time.monotonic_ns()
            atomic_json(raw / "work.json", work)
            e.progress("measurement_after", 13, 1)
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
                experiment_id=8261,
                task_id="exp8261-capstone",
                milestone=e.MILESTONE,
                run_date=args.date,
                schema="carnot.v713.capstone.v1",
                random_seed=7138261,
                duration_s=(end - start) / 1e9,
                inference_substrate="aggregation_from_upstream_artifacts",
                inference_substrate_class="no_model_load",
                MODEL_SPECS=[],
                trained_head_specs=[],
                model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
                call_ledger=[],
                preconditions_checked=work["references"],
                runtime_preconditions=runtime,
                validation_receipts=receipts,
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
                        path=item["reference"]["path"],
                        sha256=item["reference"]["sha256"],
                        task_id=task["id"],
                        fields_imported=[
                            "rows",
                            *e.COUNTS,
                            "honest_verdict",
                            "verdict_class",
                            "required_checks_passed",
                            "gate_check_summary",
                            "model_invocation_counts",
                        ],
                    )
                    for task, item in zip(work["tasks"][:-1], work["inputs"])
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
                        for r in [*receipts, *health, gates]
                    ],
                ],
                paper_ready=publication_gates["paper_ready"],
                unmet_gates=publication_gates["unmet_gates"],
                external_publication_authorized=False,
                **{
                    name.lower(): publication_gates["gates"][name]
                    for name in ["G1", "G2", "G3", "G4"]
                },
                methodology_note="Fourteen terminal dispositions retain original arms, missing masks and hashes. Current science audits are absent; H1/H2 remain unmeasured under alpha=.025 each. Historical Qwen provenance supplies no current calls. PolarFire parity is oracle-defined, not learning benefit. All three PRD gaps remain open; no replacement cohorts or comparators.",
            )
            value["reproducibility_checksum"] = canonical_hash([value["work_reference"], code])
            value["field_principles"] = {
                k: "Bind actual invocation bytes and denominators; execution readiness grants no scientific benefit."
                for k in value
            }
            value["field_principles"].update(
                rows="Fourteen dispositions; source row references preserve every original source, condition, arm and missing slot.",
                historical_v712="Retain original dispositions and failed audit replay exit1; historical publication readiness supplies no V713 benefit.",
                H1="Missing current audit primitives leave source-information and energy-specific gain unavailable at registered alpha=.025.",
                H2="Missing delayed constraint and sealed retention evidence is unavailable, not a measured null; alpha=.025 is not transferred.",
                acquisition_accounting="Absent capture ledgers leave live acquisition totals unknown; current capstone calls are separately measured zero.",
                board_obligations="Real PolarFire parity, compatible-kernel bounds and missing GateMate physical evidence remain separate from request-scale benefit.",
                retirements="No current informative scientific null was measured. Preserve scope-matched predecessors and required anti-churn obligations.",
                repository_health="One bounded full-suite diagnostic retains actual exit, clocks and hashes separately from owned validation.",
            )
            value = normalize_artifact_for_template_write(value)
            e.progress("publication_before", 13, 1)

            def validate(candidate: Path) -> Json:
                """Cold and negative children inspect the same private terminal bytes."""
                if candidate != Path(terminal[0]["argv"][-1]):
                    raise ValueError("terminal_operand_drift")
                checks = x.execute(terminal, raw / "terminal")
                tampered = deepcopy(value)
                tampered["completed_count"] = 13
                tampered["reproducibility_checksum"] = canonical_hash(tampered["rows"])
                negative = private / "rehashed_tampered.json"
                atomic_json(negative, tampered)
                checks += x.execute(
                    [
                        dict(
                            name="negative_rehashed_cold_replay",
                            argv=[
                                str(e.ROOT / ".venv/bin/python"),
                                "-u",
                                str(e.ROOT / e.CLI),
                                "--cold-replay",
                                str(negative),
                            ],
                            expected=1,
                            deadline=120,
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
                write_note(output, args.root / "docs/research-notes/v713-outcomes.md")
        e.progress("complete", 14, 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        print(json.dumps(dict(passed=False, error=str(error))), flush=True)
        return 1
