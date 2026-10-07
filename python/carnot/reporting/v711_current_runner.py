"""REQ-VERIFY-8220: freeze owned checks and replay private candidates before exposure."""

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
from carnot.reporting import v711_current_contract as q
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v710_contract_replay import require_reference
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]


def commands(private: Path) -> list[Json]:
    """Reuse qualified ownership planning while measuring only these added statements."""
    with (
        patch.object(qualified, "OWNED", q.OWNED),
        patch.object(qualified, "TEST", q.TEST),
        patch.object(qualified, "CLI", q.CLI),
    ):
        plan = [s for s in qualified.commands(private) if s["name"] != "focused_pytest"]
    config = private / "coverage.ini"
    config.write_text(
        config.read_text().replace("parallel=true", "parallel=true\npatch=subprocess")
    )
    for spec in plan:
        if spec["name"] == "changed_module_mypy":
            spec["argv"] = [
                str(q.ROOT / ".venv/bin/mypy"),
                "--config-file=/dev/null",
                "--strict",
                "--follow-imports=skip",
                "--ignore-missing-imports",
                *q.OWNED,
            ]
    return plan


def terminal_plan(path: Path) -> list[Json]:
    """Literal candidate operands remain fixed before measurement starts."""
    py = str(q.ROOT / ".venv/bin/python")
    return [
        dict(name=name, argv=argv, expected=0, deadline=120, scope="terminal")
        for name, argv in [
            ("cold_replay", [py, "-u", str(q.ROOT / q.CLI), "--cold-replay", str(path)]),
            (
                "adversarial_verify",
                [py, str(q.ROOT / "scripts/adversarial_verify.py"), "--json", str(path)],
            ),
            (
                "strict_rows",
                [
                    py,
                    str(q.ROOT / "scripts/verdict_row_consistency_lint.py"),
                    "--strict",
                    str(path),
                ],
            ),
        ]
    ]


def replay(path: Path) -> Json:
    """Independent reduction rejects rehashed claims as well as altered saved bytes."""
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
                raise ValueError("receipt_hash_drift")
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    rebuilt = q.reduce(work, value["validation_receipts"])
    for key, observed in rebuilt.items():
        if value[key] != observed:
            raise ValueError("primitive_reduction_drift:" + key)
    snapshots = work["contract"]["authority_snapshots"]
    with tempfile.TemporaryDirectory(prefix="carnot8220-replay-") as directory:
        private = Path(directory)
        paths = [
            Path(snapshots[k].get("snapshot_path", private / k))
            for k in ["design", "staged", "active"]
        ]
        result = q.assess(paths[0], paths[1], paths[2], private / "authority")
        for key in ["activated", "contract_rows", "canonical_tasks_sha256"]:
            if result[key] != work["contract"][key]:
                raise ValueError("authority_reduction_drift:" + key)
    outcomes, entries = q.historical(
        [r for r in value["source_artifact_hashes"] if r.get("schema_valid", True)]
    )
    if (
        outcomes != value["historical_dispositions"]
        or entries != value["unexecuted_v710_design_entries"]
    ):
        raise ValueError("historical_reduction_drift")
    return dict(passed=True, rows_checksum=canonical_hash(rebuilt["rows"]))


def main(argv: list[str] | None = None) -> int:
    """A private parent outlives all bounded children and prevents fixture publication."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261007"], default="20261007")
    parser.add_argument("--root", type=Path, default=q.ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--private-fixture", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    print("[exp8220] phase=start completed=0 pending=14", flush=True)
    try:
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        output = (args.output or q.ROOT / "results" / (q.NAME + ".json")).absolute()
        if args.private_fixture and output.is_relative_to(q.ROOT / "results"):
            raise ValueError("private_fixture_requires_private_output")
        raw = output.parent / "raw" / q.NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True, exist_ok=True)
        start, wall = time.monotonic_ns(), time.time_ns()
        with tempfile.TemporaryDirectory(prefix="carnot8220-invocation-") as directory:
            private = Path(directory)
            probe = private / "write_probe"
            probe.write_bytes(b"actual writable private scratch")
            runtime = dict(
                writable=probe.read_bytes() == b"actual writable private scratch",
                mode=oct(private.stat().st_mode & 0o777),
                private_path=str(private),
                executable=sys.executable,
                free_bytes=os.statvfs(private).f_bavail * os.statvfs(private).f_frsize,
            )
            plan = [] if args.private_fixture else commands(private)
            controls = x.pytest_plan(private / "children")
            preflight = qualified.precondition_command()
            terminal = terminal_plan(
                output.parent / "raw" / output.stem / "terminal_candidate.json"
            )
            atomic_json(
                raw / "validation_manifest.json",
                dict(
                    commands=plan,
                    controls=controls,
                    terminal_commands=terminal,
                    preflight=preflight,
                    owned=q.OWNED,
                    runtime=runtime,
                    frozen_before_measurement_ns=time.monotonic_ns(),
                ),
            )
            print("[exp8220] phase=measurement_before completed=0 pending=14", flush=True)
            measurement_start = time.monotonic_ns()
            work = q.measure(args.root, raw)
            measurement_end = time.monotonic_ns()
            print("[exp8220] phase=measurement_after completed=14 pending=0", flush=True)
            preconditions = x.execute([preflight], raw / "preflight_logs")
            validations = preconditions + x.execute(controls, raw / "child_logs")
            validations.extend(
                x.execute([s for s in plan if s["scope"] == "owned"], raw / "validation_logs")
            )
            health = x.execute(
                [s for s in plan if s["scope"] == "repository_health"], raw / "health_logs"
            )
            value = q.reduce(work, validations)
            atomic_json(raw / "work.json", work)
            atomic_json(raw / "primitive_rows.json", value["rows"])
            coverage = private / "coverage.json"
            cov = json.loads(coverage.read_bytes()) if coverage.exists() else {}
            atomic_json(raw / "coverage.json", cov)
            end = time.monotonic_ns()
            value.update(
                duration_s=(end - start) / 1e9,
                random_seed=7118220,
                runtime_preconditions=runtime,
                repository_health=dict(owned=False, receipts=health),
                coverage_totals=cov.get("totals", {}),
                coverage_statement_counts=cov.get("files", {}),
                code_config_hashes=work["immutable_code_snapshots"],
                work_reference=dict(
                    path=str(raw / "work.json"), sha256=sha256_file(raw / "work.json")
                ),
                raw_shard_hashes=[
                    dict(path=str(p), sha256=sha256_file(p))
                    for p in sorted(raw.rglob("*"))
                    if p.is_file()
                ],
                phase_spans=[
                    dict(
                        phase="current_contract_measurement",
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
                        for r in [*validations, *health]
                    ],
                ],
                invocation_argv=sys.orig_argv,
                measurement_clocks=dict(
                    started_monotonic_ns=start,
                    ended_monotonic_ns=end,
                    started_wall_ns=wall,
                    ended_wall_ns=time.time_ns(),
                    owner_pid=os.getpid(),
                ),
                terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
            )
            value["reproducibility_checksum"] = canonical_hash(
                [value["work_reference"], value["code_config_hashes"]]
            )
            value["field_principles"] = {
                k: "Actual invocation custody preserves current readiness separately from historical science."
                for k in value
            }
            value["field_principles"].update(
                current_contract_ready_score="One requires actual active authority and passing owned checks; no benefit credit.",
                historical_dispositions="Imported original verdicts, failures and unmeasured hypotheses retain their original hashes.",
                unexecuted_v710_design_entries="Twelve preserved design promises have no executed outcome.",
                current_code_snapshots="Current copied code predates validation and does not impersonate historical originals.",
                repository_health="Bounded unrelated diagnostics retain failures separately from owned readiness.",
                rows="Every current task retains primitive agreement counts and explicit missingness; no independent science sources.",
            )
            principles = {
                "experiment_id task_id milestone run_date title schema invocation_argv": "Literal invocation identity and documented result shape bind this current administrative task.",
                "honest_verdict verdict_class gate_check_summary acceptance_gates required_checks_passed flagged_adversarial": "Exact failed operands and measured owned exits decide terminal disposition; external absence is blocked.",
                "inference_substrate inference_substrate_class MODEL_SPECS model_invocation_counts trained_head_specs call_ledger": "Only aggregation ran here: no model loads, training or current LLM calls; cached provenance is historical.",
                "intended_count completed_count failed_count censored_count excluded_count independent_count": "Fourteen authority slots retain observed agreement or explicit missingness; no independent science sources.",
                "verifier_is_oracle exposure_scope independent_generalization_score generalized_learning_benefit_score scientific_gate_membership": "Development authority agreement is oracle defined, circular and outside science gates; benefit stays zero.",
                "preconditions_checked runtime_preconditions duration_s random_seed reproducibility_checksum measurement_clocks phase_spans": "Actual storage, input checks and clocks precede publication; terminal check clocks are retained in the sidecar.",
                "source_artifact_hashes cited_upstream_artifacts code_config_hashes raw_shard_hashes authority_snapshots work_reference": "Independently hashable saved bytes name every imported input, code and primitive operand.",
                "validation_receipts terminal_validation_sidecar_path coverage_totals coverage_statement_counts": "Real child argv, exits, full stream hashes and measured owned statement counts remain auditable.",
                "staged_readiness activated_readiness canonical_tasks_sha256 replay_controls receipt_controls": "Complete task digests and private rejection controls distinguish planned authority from real activation.",
                "external_publication_authorized methodology_note": "Administrative local custody grants no external publication permission or scientific benefit.",
            }
            value["field_principles"].update(
                {key: meaning for keys, meaning in principles.items() for key in keys.split()}
            )
            value = normalize_artifact_for_template_write(value)
            atomic_json(raw / "candidate.json", value)
            print("[exp8220] phase=publication_before completed=14 pending=1", flush=True)

            def validate(candidate: Path) -> Json:
                if candidate != Path(terminal[0]["argv"][-1]):
                    raise ValueError("terminal_operand_drift")
                receipts = x.execute(terminal, raw / "terminal_logs")
                return dict(passed=all(r["passed"] for r in receipts), checks=receipts)

            publication = publish_primary(output, value, validate)
            atomic_json(
                raw / "terminal_validation.json",
                dict(
                    publication=publication, required_checks_passed=value["required_checks_passed"]
                ),
            )
        print("[exp8220] phase=complete completed=14 pending=0", flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        print(json.dumps(dict(passed=False, error=str(error))), flush=True)
        return 1
