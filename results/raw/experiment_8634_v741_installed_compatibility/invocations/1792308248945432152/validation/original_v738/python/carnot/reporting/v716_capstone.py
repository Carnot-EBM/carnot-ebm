"""REQ-VERIFY-8303: run bounded children before exposing checked primary bytes.

The existing supervisor owns deadlines and process groups. Current validation
qualifies the reader; authenticated upstream failures retain their own scope.
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
from carnot.reporting import v715_capstone as prior
from carnot.reporting import v716_capstone_evidence as e
from carnot.reporting import v713_capstone_evidence as source_evidence
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
    """Measure added statements and retain private consumer and E2E checks."""
    with patch.object(prior, "e", e):
        plan = list(prior.commands(private))
    for spec in plan:
        if spec["name"] == "changed_module_coverage":
            spec["deadline"] = 900
    return plan


def terminal_plan(path: Path) -> list[Json]:
    """Use unchanged auditors against the exact current candidate path."""
    with patch.object(prior, "e", e):
        return list(prior.terminal_plan(path))


def audit_plan(work: Json, private: Path) -> list[Json]:
    """Freeze branch-specific fresh readers and preserve failed-producer exits."""
    plan = []
    for identity, task, item in zip(range(8290, 8303), work["tasks"][:-1], work["inputs"]):
        script = e.ROOT / "scripts/experiments" / (Path(task["deliverable"]).stem + ".py")
        if not item["reference"]["exists"] or not script.is_file():
            continue
        source = source_evidence.read(item["reference"])
        if source.get("schema") == "blocked_gate_check_v1":
            continue
        changed = dict(source, intended_count=-1)
        changed["reproducibility_checksum"] = canonical_hash(changed.get("rows"))
        negative = private / f"upstream_{identity}_rehashed.json"
        atomic_json(negative, changed)
        for name, path, expected in [
            (
                "replay",
                Path(item["reference"]["snapshot_path"]),
                int(source["verdict_class"] == "disqualified"),
            ),
            ("tamper", negative, 1),
        ]:
            plan.append(
                dict(
                    name=f"branch_{identity}_{name}",
                    argv=[
                        str(e.ROOT / ".venv/bin/python"),
                        "-u",
                        str(script),
                        "--cold-replay",
                        str(path),
                    ],
                    expected=expected,
                    deadline=180,
                    scope="upstream",
                )
            )
    return plan


def replay(path: Path) -> Json:
    """Reconstruct primitives so rehashing a false aggregate cannot validate it."""
    value = json.loads(path.read_bytes())
    if (value["experiment_id"], value["task_id"], value["milestone"]) != (
        8303,
        "exp8303-capstone",
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
    gate = json.loads(Path(value["publication_gate_receipt"]["stdout_path"]).read_bytes())
    if any(
        value[k] != v
        for k, v in dict(
            paper_ready=gate["paper_ready"],
            unmet_gates=gate["unmet_gates"],
            **{k.lower(): gate["gates"][k] for k in ["G1", "G2", "G3", "G4"]},
        ).items()
    ):
        raise ValueError("publication_reduction_drift")
    return dict(passed=True, rows_checksum=canonical_hash(value["rows"]))


def write_note(primary: Path, destination: Path) -> None:
    """Give each branch a falsifiable reopening condition and a bounded claim."""
    value = json.loads(primary.read_bytes())
    lines = [
        "# V716 outcomes — 2026-10-08",
        "",
        f"Primary: [{primary.name}]({primary}); `{sha256_file(primary)}`.",
        f"Verdict {value['honest_verdict']}. Reconciled 14/14; executed {value['actual_executed_task_count']}; pre-gate {value['pre_gate_count']}; absent {value['missing_output_count']}.",
        f"Execution readiness {value['capstone_execution_ready_score']}; science readiness {value['science_ready_score']}. H3 fixture soundness/efficiency {value['h3_fixture_soundness_score']}/{value['h3_fixture_efficiency_signal_score']}.",
        "H1/H2 remain unmeasured with denominators 128/96 and retention 32, alpha .025 each. Missing work is unavailable. Both generalization scores remain zero.",
        "H3 qualifies deterministic CPU mechanics only. It does not establish semantic extraction, natural continual learning, independent generalization or energy-specific advantage.",
        "",
        "| Task | Evidence disposition | Next evidence condition |",
        "|---|---|---|",
    ]
    for row, retirement in zip(value["rows"], value["retirements"]):
        lines.append(
            f"| {row['task_id']} | {row['disposition']}; {row['honest_verdict'] or 'producer verdict unavailable'} | {retirement['reopening_condition']} |"
        )
    lines += [
        "",
        *[f"- {b['board']}: {b['terminal_condition']}" for b in value["board_obligations"]],
        "",
        "PolarFire graduation is inherited board-local Linux CPU scope. V715 branch_8286 exit1 and the disqualified ARC primaries remain authenticated failures. No repeated hardware probe occurred.",
        "V715 archive lag retains six executed producer primaries, one separate pre-gate and seven absent outputs. Its planning activation hash and preserved design remain bound sources.",
        f"Publication paper_ready={value['paper_ready']}; unmet={value['unmet_gates']}. This flag establishes no new scientific benefit.",
        "All three PRD gaps remain open. Imported calls count once; current capture/update costs are unavailable. No model, weight update or external publication occurred.",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Keep private scratch alive through bounded checks and atomic publication."""
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
        with tempfile.TemporaryDirectory(prefix="carnot8303-", dir="/tmp") as directory:
            private = Path(directory)
            probe = private / "write_probe"
            probe.write_bytes(b"private scratch")
            runtime = dict(
                private_path=str(private),
                mode=oct(private.stat().st_mode & 0o777),
                writable=probe.read_bytes() == b"private scratch",
                free_bytes=os.statvfs(private).f_bavail * os.statvfs(private).f_frsize,
            )
            plan = [] if args.private_fixture else commands(private)
            preflight = prior.prior.qualified.current.qualified.precondition_command()
            controls = x.pytest_plan(private / "controls")
            candidate = output.parent / "raw" / output.stem / "terminal_candidate.json"
            terminal = terminal_plan(candidate)
            code = [
                snapshot(e.ROOT / p, raw / "code", str(i))
                for i, p in enumerate(
                    [
                        *e.OWNED,
                        e.TEST,
                        "python/carnot/reporting/v715_capstone_evidence.py",
                        "python/carnot/reporting/v713_capstone_evidence.py",
                        "python/carnot/reporting/dependency_admission_execution_8291.py",
                        "python/carnot/reporting/v709_execution.py",
                        "python/carnot/reporting/primary_publication.py",
                        "python/carnot/reporting/roadmap_contract.py",
                        "scripts/publication_gate.py",
                        "scripts/adversarial_verify.py",
                        "scripts/verdict_row_consistency_lint.py",
                        "scripts/experiment_template.py",
                        "openspec/capabilities/research-reporting/spec.md",
                        "openspec/capabilities/verification/spec.md",
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
                    preflight=preflight,
                    controls=controls,
                    terminal_commands=terminal,
                    runtime=runtime,
                    frozen_before_measurement_ns=time.monotonic_ns(),
                ),
            )
            receipts = x.execute([preflight], raw / "preconditions")
            e.progress("measurement_before", 0, 14)
            measured = time.monotonic_ns()
            work = e.measure(args.root, raw)
            atomic_json(raw / "work.json", work)
            e.progress("measurement_after", 13, 1)
            audits = [] if args.private_fixture else audit_plan(work, private)
            atomic_json(raw / "audit_manifest.json", dict(commands=audits))
            receipts += x.execute(controls, raw / "controls")
            receipts += x.execute([p for p in plan if p["scope"] == "owned"], raw / "validation")
            receipts += x.execute(audits, raw / "audits")
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
                        deadline=90,
                        expected=0,
                        scope="publication",
                    )
                ],
                raw / "publication",
            )[0]
            publication = json.loads(Path(gate["stdout_path"]).read_bytes())
            coverage = (
                json.loads((private / "coverage.json").read_bytes())
                if (private / "coverage.json").is_file()
                else {}
            )
            atomic_json(raw / "coverage.json", coverage)
            value = e.reduce(work, receipts)
            end = time.monotonic_ns()
            value.update(
                experiment_id=8303,
                task_id="exp8303-capstone",
                milestone=e.MILESTONE,
                run_date=args.date,
                schema="carnot.v716.capstone.v1",
                random_seed=7168303,
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
                publication_gate_receipt=gate,
                coverage_totals=coverage.get("totals", {}),
                coverage_statement_counts=coverage.get("files", {}),
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
                        path=i["reference"]["path"],
                        sha256=i["reference"]["sha256"],
                        task_id=t["id"],
                        fields_imported=[
                            "rows",
                            *source_evidence.COUNTS,
                            "honest_verdict",
                            "verdict_class",
                            "required_checks_passed",
                            "model_invocation_counts",
                            "gate_check_summary",
                        ],
                    )
                    for t, i in zip(work["tasks"][:-1], work["inputs"])
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
                        phase="primitive_reduction_and_validation",
                        started_monotonic_ns=measured,
                        ended_monotonic_ns=end,
                        duration_s=(end - measured) / 1e9,
                    ),
                    *[
                        dict(
                            phase=r["name"],
                            duration_s=r["duration_s"],
                            started_monotonic_ns=r["started_monotonic_ns"],
                            ended_monotonic_ns=r["ended_monotonic_ns"],
                        )
                        for r in [*receipts, *health, gate]
                    ],
                ],
                paper_ready=publication["paper_ready"],
                unmet_gates=publication["unmet_gates"],
                **{k.lower(): publication["gates"][k] for k in ["G1", "G2", "G3", "G4"]},
                external_publication_authorized=False,
                methodology_note="Fourteen dispositions preserve unavailable science, authenticated upstream failures and H3 deterministic CPU mechanics. Inherited PolarFire CPU scope grants no new scientific benefit. Current model calls zero.",
            )
            value["reproducibility_checksum"] = canonical_hash([value["work_reference"], code])
            value["field_principles"] = {
                k: "Bind real bytes and declared scope; unavailable work is not a measured null; execution readiness grants no scientific benefit."
                for k in value
            }
            value = normalize_artifact_for_template_write(value)
            e.progress("publication_before", 13, 1)

            def validate(path: Path) -> Json:
                """Validate identical candidate bytes and reject a rehashed false aggregate."""
                if path != candidate:
                    raise ValueError("terminal_operand_drift")
                checks = x.execute(terminal, raw / "terminal")
                changed = deepcopy(value)
                changed["completed_count"] = 13
                changed["reproducibility_checksum"] = canonical_hash(changed["rows"])
                negative = private / "rehashed_tampered.json"
                atomic_json(negative, changed)
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

            published = publish_primary(output, value, validate)
            atomic_json(
                raw / "terminal_validation.json",
                dict(publication=published, required_checks_passed=value["required_checks_passed"]),
            )
            if not args.private_fixture:
                write_note(output, args.root / "docs/research-notes/v716-outcomes.md")
            e.progress("publication_after", 14, 0)
            print(json.dumps(published), flush=True)
            return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("capstone_failed:" + str(error), file=sys.stderr, flush=True)
        return 1
