"""REQ-VERIFY-8275: expose only capstone bytes checked by independent children.

The qualified supervisor records deadlines, clocks and stream hashes. Private
fixtures test execution mechanics; they cannot publish into research results.
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
from carnot.reporting import v713_capstone as prior
from carnot.reporting import v714_capstone_evidence as e
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
MODEL_SPECS: list[Json] = []


def commands(private: Path) -> list[Json]:
    """Reuse existing consumer/E2E commands with coverage restricted to added code."""
    private.mkdir(parents=True, exist_ok=True)
    with (
        patch.object(prior.e, "OWNED", e.OWNED),
        patch.object(prior.e, "TEST", e.TEST),
        patch.object(prior.e, "CLI", e.CLI),
    ):
        return list(prior.commands(private))


def terminal_plan(path: Path) -> list[Json]:
    """Keep unchanged terminal auditors but bind this capstone's replay entrypoint."""
    with patch.object(prior.e, "CLI", e.CLI):
        return list(prior.terminal_plan(path))


def imported_fields(ref: Json) -> list[str]:
    """Name actual JSON fields; document custody remains an explicit byte import."""
    if not ref["exists"]:
        return []
    try:
        value = e.read(ref)
        return list(value)
    except ValueError:
        return ["raw_document_bytes"]


def replay(path: Path) -> Json:
    """Rebuild fields from original primitive bytes even after summary rehashing."""
    value = json.loads(path.read_bytes())
    if (value["experiment_id"], value["task_id"], value["milestone"]) != (
        8275,
        "exp8275-capstone",
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
    for receipt in [*value["validation_receipts"], value["publication_gate_receipt"]]:
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


def branch_plan(work: Json, private: Path) -> list[Json]:
    """Each existing branch replays through its own entrypoint with a tamper control."""
    plan = []
    for task, item in zip(work["tasks"][:-1], work["inputs"]):
        value = e.read(item["reference"]) if item["reference"]["exists"] else {}
        cli = e.ROOT / "scripts/experiments" / (Path(task["deliverable"]).stem + ".py")
        if not value.get("experiment_id") or not cli.is_file():
            continue
        changed = deepcopy(value)
        changed["completed_count"] = value["completed_count"] + 1
        changed["reproducibility_checksum"] = canonical_hash(
            {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
        )
        tamper = private / (str(value["experiment_id"]) + "-rehashed.json")
        atomic_json(tamper, changed)
        for name, path, expected in [
            ("valid", Path(item["reference"]["path"]), 0),
            ("rehashed", tamper, 1),
        ]:
            plan.append(
                dict(
                    name=f"branch_{value['experiment_id']}_{name}",
                    argv=[
                        str(e.ROOT / ".venv/bin/python"),
                        "-u",
                        str(cli),
                        "--cold-replay",
                        str(path),
                    ],
                    expected=expected,
                    deadline=120,
                    scope="upstream",
                )
            )
    return plan


def write_note(primary: Path, destination: Path) -> None:
    """Derive prose from accepted evidence, with one reopening condition per branch."""
    v = json.loads(primary.read_bytes())
    lines = [
        "# V714 outcomes — 2026-10-08",
        "",
        f"Primary: `{primary}`; `{sha256_file(primary)}`.",
        f"Verdict `{v['honest_verdict']}`; dispositions {v['completed_count']}/14; executed {v['actual_executed_task_count']}; pre-gates {v['pre_gate_count']}; absent outputs {v['missing_output_count']}.",
        f"Execution readiness {v['capstone_execution_ready_score']}; science readiness {v['science_ready_score']}; H1/H2 development scores0/0. Both generalization scores0.",
        "H1 information gain and energy-specific advantage, and H2 later distinct-source use and retention, remain blocked_unmeasured at alpha=.025 each. Missing evidence is not a scientific null.",
        f"Evidence mechanics improved={v['evidence_improved']}; learning improved={v['learning_improved']}. {v['evidence_improvement_scope']}.",
        "",
        "| Task | Evidence disposition | Falsifiable next evidence condition |",
        "|---|---|---|",
    ]
    for row, retirement in zip(v["rows"], v["retirements"]):
        lines.append(
            f"| {row['task_id']} | {row['disposition']} | {retirement['reopening_condition']} |"
        )
    lines += [
        "",
        f"PolarFire graduation={v['polarfire_graduation']['graduated']}: authenticated unchanged Exp8259 board-local Linux CPU dispatch and hash parity. No fabric acceleration or scientific benefit.",
        "KV260 retains useful quadratic k<=5 workload, authenticated SSH fabric transcript and transfer-inclusive complete timing. No host storage prerequisite.",
        "GateMate retains its0xffffffff transcript: dated physical change, valid GM1Ax0x20000001, flashed n16 tile and device hash parity. No probe without that change.",
        "ARC carries the unchanged-frontier obligation; new authenticated supervisor outcomes with cross-game arm overlap are required. No retired probe rerun.",
        "Coverage custody and causal typed-action controls improved execution evidence; circular fixture gains do not show natural learning. No current science mechanism is retired for external absence.",
        "Live calls are accounted per canary/fit/tune/reserved task, counting canary once. Missing current acquisition and update spans remain unavailable.",
        "Archive lag is separate: V713 history comes from preserved activated design, authenticated primaries and conductor log; planning archive stopped at V712.",
        f"Publication gate paper_ready={v['paper_ready']}; unmet={v['unmet_gates']}. Readiness establishes no new scientific benefit.",
        "All three PRD gaps remain open. No model loads, generator updates, hardware reruns or external publication occurred. Repository health keeps its actual failures separately.",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Keep private scratch alive until bounded checks accept the final candidate."""
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
        with tempfile.TemporaryDirectory(prefix="carnot8275-", dir="/tmp") as directory:
            private = Path(directory)
            probe = private / "write_probe"
            probe.write_bytes(b"actual private scratch")
            runtime = dict(
                mode=oct(private.stat().st_mode & 0o777),
                writable=probe.read_bytes() == b"actual private scratch",
                executable=sys.executable,
                free_bytes=os.statvfs(private).f_bavail * os.statvfs(private).f_frsize,
            )
            plan = [] if args.private_fixture else commands(private)
            controls = x.pytest_plan(private / "controls")
            preflight = prior.qualified.current.qualified.precondition_command()
            terminal = terminal_plan(
                output.parent / "raw" / output.stem / "terminal_candidate.json"
            )
            code = [
                snapshot(e.ROOT / p, raw / "code", str(i))
                for i, p in enumerate(
                    [
                        *e.OWNED,
                        e.TEST,
                        "python/carnot/reporting/v713_capstone_evidence.py",
                        "python/carnot/reporting/primary_publication.py",
                        "python/carnot/reporting/roadmap_contract.py",
                        "python/carnot/reporting/v709_execution.py",
                        "scripts/experiment_template.py",
                        "scripts/publication_gate.py",
                        "ops/exclusion_manifest.yaml",
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
                    runtime=runtime,
                    frozen_before_measurement_ns=time.monotonic_ns(),
                ),
            )
            e.progress("preconditions_before")
            receipts = x.execute([preflight], raw / "preconditions")
            begin = time.monotonic_ns()
            e.progress("measurement_before", 0, 13)
            work = e.measure(args.root, raw)
            end = time.monotonic_ns()
            atomic_json(raw / "work.json", work)
            e.progress("measurement_after", 13, 1)
            branches = [] if args.private_fixture else branch_plan(work, private)
            atomic_json(raw / "branch_manifest.json", dict(commands=branches))
            receipts += x.execute(branches, raw / "branches")
            receipts += x.execute(controls, raw / "controls")
            receipts += x.execute([p for p in plan if p["scope"] == "owned"], raw / "validation")
            health = x.execute(
                [p for p in plan if p["scope"] == "repository_health"], raw / "health"
            )
            gate = x.execute(
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
            publication = json.loads(Path(gate["stdout_path"]).read_bytes())
            cov = (
                json.loads((private / "coverage.json").read_bytes())
                if (private / "coverage.json").is_file()
                else {}
            )
            atomic_json(raw / "coverage.json", cov)
            value = e.reduce(work, receipts)
            value.update(
                experiment_id=8275,
                task_id="exp8275-capstone",
                milestone=e.MILESTONE,
                run_date=args.date,
                schema="carnot.v714.capstone.v1",
                random_seed=7148275,
                duration_s=(time.monotonic_ns() - start) / 1e9,
                MODEL_SPECS=[],
                trained_head_specs=[],
                inference_substrate="aggregation_from_upstream_artifacts",
                inference_substrate_class="no_model_load",
                model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
                current_model_calls=0,
                call_ledger=[],
                preconditions_checked=work["references"],
                runtime_preconditions=runtime,
                validation_receipts=receipts,
                repository_health=dict(owned=False, receipts=health),
                coverage_totals=cov.get("totals", {}),
                coverage_statement_counts=cov.get("files", {}),
                publication_gate_receipt=gate,
                publication_gate_output_hash=gate["stdout_sha256"],
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
                        fields_imported=imported_fields(r),
                    )
                    for r in work["references"]
                ],
                invocation_argv=sys.orig_argv,
                measurement_clocks=dict(
                    started_monotonic_ns=start,
                    ended_monotonic_ns=time.monotonic_ns(),
                    started_wall_ns=wall,
                    ended_wall_ns=time.time_ns(),
                ),
                phase_spans=[
                    dict(
                        phase="primitive_reduction",
                        started_monotonic_ns=begin,
                        ended_monotonic_ns=end,
                        duration_s=(end - begin) / 1e9,
                    ),
                    *[
                        dict(
                            phase=r["name"],
                            started_monotonic_ns=r["started_monotonic_ns"],
                            ended_monotonic_ns=r["ended_monotonic_ns"],
                            duration_s=r["duration_s"],
                        )
                        for r in [*receipts, *health, gate]
                    ],
                ],
                paper_ready=publication["paper_ready"],
                unmet_gates=publication["unmet_gates"],
                external_publication_authorized=False,
                fixture_mode=args.private_fixture,
                **{k.lower(): publication["gates"][k] for k in ["G1", "G2", "G3", "G4"]},
                methodology_note="Fourteen terminal dispositions preserve producer bytes and intended source/arm denominators. Absent science is blocked_unmeasured, not null. Alpha .025 each, shared fallback and selected comparators remain frozen. Evidence custody and typed-action controls establish mechanics only. PolarFire is historical actual board-local CPU parity, not FPGA acceleration or benefit. No new LLM calls or hardware execution.",
            )
            value["reproducibility_checksum"] = canonical_hash([value["work_reference"], code])
            value["field_principles"] = {
                k: "Bind actual invocation and source bytes; administrative readiness supplies no scientific benefit."
                for k in value
            }
            value["field_principles"].update(
                rows="Fourteen administrative dispositions; source snapshots retain every intended scientific condition and arm.",
                H1="Information gain and energy-specific advantage require independent primitives at alpha .025; absence is unmeasured.",
                H2="Later distinct-source constraint use and retention are separate; missing original slots cannot be replaced.",
                live_call_accounting="Four live tasks; imported canary counted once, unavailable counts and spans remain unavailable.",
                polarfire_graduation="Pinned unchanged primary plus terminal/adversarial bytes and actual board hash parity; CPU-only scope.",
                retirements="External absence cannot retire science; ARC and hardware obligations survive unchanged frontiers.",
            )
            value = normalize_artifact_for_template_write(value)
            e.progress("publication_before", 13, 1)

            def validate(candidate: Path) -> Json:
                """Check the frozen candidate and reject a rehashed false completion count."""
                if candidate != Path(terminal[0]["argv"][-1]):
                    raise ValueError("terminal_operand_drift")
                checks = x.execute(terminal, raw / "terminal")
                changed = dict(value, completed_count=13)
                changed["reproducibility_checksum"] = canonical_hash(changed)
                negative = private / "rehashed_tampered.json"
                atomic_json(negative, changed)
                checks += x.execute(
                    [
                        dict(
                            name="negative_rehashed_replay",
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

            published = publish_primary(output, value, validate)
            atomic_json(
                raw / "terminal_validation.json",
                dict(
                    publication=published,
                    required_checks_passed=value["required_checks_passed"],
                    normal_process_exit=True,
                ),
            )
            if not args.private_fixture:
                write_note(output, args.root / "docs/research-notes/v714-outcomes.md")
        e.progress("complete", 14, 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        print(json.dumps(dict(passed=False, error=str(error))), flush=True)
        return 1
