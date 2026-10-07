"""REQ-VERIFY-8233: measured checks precede atomic current capstone publication.

Only the invocation's checks decide execution readiness. Finished blocked and
null science branches remain visible without retrying their producers.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v709_execution as x
from carnot.reporting import v709_runner as qualified
from carnot.reporting import v711_capstone_evidence as e
from carnot.reporting import v711_current_runner as current
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v710_contract_replay import require_reference
from carnot.reporting.v710_contract_replay import snapshot
from carnot.verify import utility_audit_8224 as audit
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]


def commands(private: Path) -> list[Json]:
    """Freeze qualified child coverage and the four applicable private E2E commands."""
    with (
        patch.object(current.q, "OWNED", e.OWNED),
        patch.object(current.q, "TEST", e.TEST),
        patch.object(current.q, "CLI", e.CLI),
    ):
        plan = current.commands(private)
    for spec in plan:
        if spec["name"] == "changed_module_coverage":
            spec["deadline"] = 600
    plan.append(
        dict(
            name="private_E2E015_019_021",
            argv=[
                str(e.ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_source_boundary_7852.py",
                "tests/python/test_experiment_7942_v689_sentence_labels.py",
                "tests/python/test_restricted_decision_audit_8210.py",
                "--basetemp=" + str(private / "e2e"),
            ],
            deadline=600,
            expected=0,
            scope="owned",
        )
    )
    return list(plan)


def terminal_plan(path: Path) -> list[Json]:
    """Use this audit's own replay CLI instead of a producer's command."""
    with patch.object(current.q, "CLI", e.CLI):
        return list(current.terminal_plan(path))


def replay(path: Path) -> Json:
    """Cold-check immutable input bytes, complete streams and primitive reductions."""
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
    for entry in work["consumer_manifest"]:
        if entry["task_id"] not in work["primaries"]:
            continue
        ref = next(r for r in work["references"] if r["path"] == entry["path"])
        source = json.loads(Path(ref["snapshot_path"]).read_bytes())
        row = next(r for r in work["dispositions"] if r["task_id"] == entry["task_id"])
        eligible = (
            source.get("required_checks_passed") is True
            and source.get("flagged_adversarial") is False
            and source["verdict_class"] in {"positive", "circular_positive", "null"}
        )
        expected = dict(
            honest_verdict=source["honest_verdict"],
            verdict_class=source["verdict_class"],
            eligible=eligible,
            excluded=not eligible,
            producer_executed=True,
            failed=source["verdict_class"] == "disqualified",
            numerator=1,
            source_counts={k: source.get(k) for k in e.COUNTS},
        )
        if source != work["primaries"][entry["task_id"]] or any(
            row.get(k) != v for k, v in expected.items()
        ):
            raise ValueError("source_primitive_drift")
    if work["H1"]:
        ref = work["h1_primitive_reference"]
        primitive = json.loads(Path(ref["snapshot_path"]).read_bytes())
        if audit.reduce(primitive["evidence"]) != work["H1"]:
            raise ValueError("H1_primitive_reduction_drift")
    ref = work["references"][0]
    _, tasks = parse_design(Path(ref["snapshot_path"]).read_text(), milestone=e.MILESTONE)
    if tasks != work["tasks"]:
        raise ValueError("contract_primitive_drift")
    rebuilt = e.reduce(work, value["validation_receipts"])
    for key, observed in rebuilt.items():
        if value[key] != observed:
            raise ValueError("primitive_reduction_drift:" + key)
    return dict(passed=True, rows_checksum=canonical_hash(rebuilt["rows"]))


def main(argv: list[str] | None = None) -> int:
    """Retain private scratch through all children and publish only checked bytes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261007"], default="20261007")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--private-fixture", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    e.progress("start", 0, 14)
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
        with tempfile.TemporaryDirectory(prefix="carnot8233-", dir="/tmp") as directory:
            private = Path(directory)
            probe = private / "write_probe"
            probe.write_bytes(b"actual private scratch")
            runtime = dict(
                private_path=str(private),
                mode=oct(private.stat().st_mode & 0o777),
                writable=probe.read_bytes() == b"actual private scratch",
                executable=sys.executable,
                free_bytes=os.statvfs(private).f_bavail * os.statvfs(private).f_frsize,
            )
            plan = [] if args.private_fixture else commands(private)
            controls = x.pytest_plan(private / "controls")
            preflight = qualified.precondition_command()
            terminal = terminal_plan(
                output.parent / "raw" / output.stem / "terminal_candidate.json"
            )
            code = [
                snapshot(e.ROOT / p, raw / "code", str(i))
                for i, p in enumerate(
                    [
                        *e.OWNED,
                        e.TEST,
                        "python/carnot/verify/utility_audit_8224.py",
                        "python/carnot/reporting/v709_execution.py",
                        "python/carnot/reporting/primary_publication.py",
                        "scripts/publication_gate.py",
                        "scripts/experiment_template.py",
                        "AGENTS.md",
                        "CODEX.md",
                        "CLAUDE.md",
                        "ops/e2e-test-plan.md",
                        "openspec/capabilities/research-reporting/spec.md",
                        "openspec/capabilities/verification/spec.md",
                        "ops/exclusion_manifest.yaml",
                        "python/carnot/reporting/v709_capstone.py",
                        "python/carnot/reporting/v709_capstone_science.py",
                        "python/carnot/reporting/roadmap_contract.py",
                        "openspec/change-proposals/research-roadmap-v710-preserved-20261006.md",
                    ]
                )
            ]
            atomic_json(
                raw / "validation_manifest.json",
                dict(
                    commands=plan,
                    controls=controls,
                    terminal_commands=terminal,
                    preflight=preflight,
                    owned=e.OWNED,
                    runtime=runtime,
                    frozen_before_measurement_ns=time.monotonic_ns(),
                ),
            )
            e.progress("preconditions_before")
            preconditions = x.execute([preflight], raw / "preconditions")
            measurement_start = time.monotonic_ns()
            work = e.measure(args.root, raw)
            measurement_end = time.monotonic_ns()
            atomic_json(
                raw / "consumer_manifest.json",
                dict(
                    schema="carnot.v711.capstone-consumers.v1",
                    inputs=work["consumer_manifest"],
                    contract_sha256=canonical_hash(work["tasks"]),
                ),
            )
            atomic_json(raw / "work.json", work)
            e.progress("preconditions_after", 13, 1)
            receipts = preconditions + x.execute(controls, raw / "controls")
            receipts += x.execute([p for p in plan if p["scope"] == "owned"], raw / "validation")
            health = x.execute(
                [p for p in plan if p["scope"] == "repository_health"], raw / "health"
            )
            value = e.reduce(work, receipts)
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
            value.update(
                experiment_id=8233,
                task_id="exp8233-capstone",
                milestone=e.MILESTONE,
                run_date=args.date,
                schema="carnot.v711.capstone.v1",
                random_seed=7118233,
                duration_s=(end - start) / 1e9,
                inference_substrate="aggregation_from_upstream_artifacts",
                inference_substrate_class="no_model_load",
                MODEL_SPECS=[],
                trained_head_specs=[],
                model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
                call_ledger=[],
                preconditions_checked=work["consumer_manifest"],
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
                        phase="current_evidence_reduction",
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
                publication_gate_results=publication_gates,
                **{
                    name.lower(): publication_gates["gates"][name]
                    for name in ["G1", "G2", "G3", "G4"]
                },
                methodology_note="Exact fourteen outcome accounting and qualified cached H1 reduction. Current H2 and service audits were not executed after upstream disqualification. Historical Qwen provenance supplies zero current model calls; no device operation or independent deployment occurred.",
            )
            value["reproducibility_checksum"] = canonical_hash([value["work_reference"], code])
            value["field_principles"] = {
                k: "Bind actual invocation bytes, clocks and denominators; execution readiness supplies no independent scientific benefit."
                for k in value
            }
            value["field_principles"].update(
                completed_count="All fourteen dispositions are reconciled; this does not count fourteen producer executions.",
                actual_executed_task_count="Only authenticated producer outputs plus this invocation count as executed; conductor pre-gates never execute.",
                H1="Recompute the frozen V711 audit from original measurement rows; keep its negative and missing branches.",
                H2="Missing current eligible audit remains unmeasured; no historical learning result substitutes.",
                request_accounting="Ninety-six planned calls and four cold starts remain obligations; missing measurements stay null.",
                board_obligations="Each attached board retains its own disposition and zero unmeasured deployment benefit.",
                retirements="Repeated exact prior verdicts retire the scheduled scope with falsifiable reopening evidence.",
                repository_health="Bounded existing failures remain separate from owned checks.",
            )
            value = normalize_artifact_for_template_write(value)
            e.progress("publication_before", 13, 1)

            def validate(candidate: Path) -> Json:
                if candidate != Path(terminal[0]["argv"][-1]):
                    raise ValueError("terminal_operand_drift")
                checks = x.execute(terminal, raw / "terminal")
                return dict(passed=all(r["passed"] for r in checks), checks=checks)

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
                write_note(output, args.root / "docs/research-notes/v711-outcomes.md")
        e.progress("complete", 14, 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration) as error:
        print(json.dumps(dict(passed=False, error=str(error))), flush=True)
        return 1


def write_note(primary: Path, destination: Path) -> None:
    """Use checked primary bytes so prose cannot silently invent a new result."""
    value = json.loads(primary.read_bytes())
    h1 = value["H1"]["statistics"]
    boot = h1.get("bootstrap_diagnostics", {})
    lines = [
        "# V711 outcomes — 2026-10-07",
        "",
        f"Source: `{primary}`; `{sha256_file(primary)}`.",
        "",
        f"Verdict: `{value['honest_verdict']}` (`{value['verdict_class']}`).",
        f"Disposition completion: {value['completed_count']}/{value['intended_count']}; actual executed tasks: {value['actual_executed_task_count']} (including this capstone).",
        f"Conductor pre-gates: {value['conductor_pre_gate_count']}; missing outputs: {value['missing_output_count']}; disqualified task rows: {value['failed_count']}.",
        f"Execution readiness: {value['capstone_execution_ready_score']}; science readiness: {value['science_ready_score']}.",
        "",
        "| Task | Disposition | Producer executed |",
        "|---|---|---|",
    ]
    lines += [
        f"| {r['task_id']} | {r['honest_verdict']} | {r['producer_executed']} |"
        for r in value["rows"]
    ]
    lines += [
        "",
        f"H1: {value['H1']['status']}; intended128, complete {h1.get('completed_count', 'unavailable')}; mean cost gain {boot.get('mean_gain', 'unavailable')}; lower97.5% bound {boot.get('lower_one_sided_975', 'unavailable')}.",
        f"H2: {value['H2']['status']};192 later and64 retention slots remain unmeasured by a qualified current audit.",
        f"Development signals H1/H2: {value['h1_development_signal_score']}/{value['h2_development_signal_score']}. Both independent generalization scores remain0. Family alpha stays .05 with .025 per hypothesis.",
        "",
        "The three remaining PRD gaps:",
        "",
    ]
    lines += [
        f"- {', '.join(g['requirements'])}: moved={g['moved']}, closed={g['closed']}. {g['remaining']}"
        for g in value["three_prd_gaps"]
    ]
    lines += [
        "",
        f"Requests: {value['request_accounting']['intended_generation_calls']} planned independent calls and {value['request_accounting']['cold_starts_intended']} cold starts; measured call count is {value['request_accounting']['observed_measurement_calls']} (missing). NFR-01 remains unmet.",
        f"ARC new outcomes: {value['arc_evidence']['new_outcome_count']}; new solve credit:0. Current model calls:0; MODEL_SPECS=[]; aggregation only.",
        "",
        f"Paper ready: {value['paper_ready']}; unmet gates: {', '.join(value['unmet_gates'])}.",
        "G1–G4: " + ", ".join(f"{g.upper()}={value[g]['pass']}" for g in ["g1", "g2", "g3", "g4"]),
        "",
        f"Owned checks passed: {value['required_checks_passed']}; changed statements coverage: {value['coverage_totals'].get('percent_statements_covered', 'unavailable')}%.",
        "Repository health diagnostics retain their actual failures separately. No external publication occurred.",
        "",
        "Boards and mechanism decisions:",
        "",
    ]
    lines += [
        f"- {b['board']}: {b['honest_verdict']}; device benefit score0."
        for b in value["board_obligations"]
    ]
    lines += [
        f"- {r['task_id']}: {r['decision']} for {r['scope']}. Reopen: {r['reopening_condition']}"
        for r in value["retirements"]
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")
