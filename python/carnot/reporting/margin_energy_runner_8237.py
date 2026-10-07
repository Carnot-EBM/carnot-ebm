"""REQ-REPORT-8237: validate private candidates before exposing primary bytes.

The qualified supervisor keeps child logs and kills only owned process groups.
Fresh replay repeats fitting from original snapshots, so rehashed edits still fail.
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

from carnot.reporting import margin_energy_training_8237 as q
from carnot.reporting import decision_margin_methods_8234 as methods
from carnot.reporting import decision_margin_runner_8234 as qualified
from carnot.reporting import v709_execution as x
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v710_contract_replay import require_reference
from carnot.verify import margin_energy_training_8237 as n
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]


def commands(private: Path) -> list[Json]:
    """Freeze existing private E2E and consumer checks with this producer's coverage."""
    with patch.object(qualified, "q", q):
        plan = list(qualified.commands(private))
    for identity, test in [
        ("015", "test_source_boundary_7852.py"),
        ("019", "test_experiment_7942_v689_sentence_labels.py"),
    ]:
        plan.append(
            dict(
                name="private_E2E" + identity,
                argv=[
                    str(q.ROOT / ".venv/bin/pytest"),
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    "tests/python/" + test,
                    "--basetemp=" + str(private / ("e2e" + identity)),
                ],
                expected=0,
                deadline=240,
                scope="owned",
            )
        )
    return plan


def terminal_plan(candidate: Path) -> list[Json]:
    """Literal terminal operands prevent validation from selecting another artifact."""
    plan = qualified.qualified.terminal_plan(candidate)
    plan[0]["argv"][-3] = str(q.ROOT / q.CLI)
    return list(plan)


def replay(path: Path) -> Json:
    """Rebuild numerical primitives from original operands in a fresh process."""
    value = json.loads(path.read_bytes())
    for receipt in value["validation_receipts"]:
        for stream in ["stdout", "stderr"]:
            if sha256_file(Path(receipt[stream + "_path"])) != receipt[stream + "_sha256"]:
                raise ValueError("receipt_drift")
    for ref in [
        value["work_reference"],
        *value["source_artifact_hashes"],
        *value["code_config_hashes"],
        *value["raw_shard_hashes"],
    ]:
        require_reference(ref)
    if value["reproducibility_checksum"] != canonical_hash(
        [value["work_reference"], value["code_config_hashes"], methods.PIN]
    ):
        raise ValueError("checksum_drift")
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    if work["protocol"] != methods.PROTOCOL_VALUE:
        raise ValueError("protocol_drift")
    refs = {r["path"]: r for r in work["refs"]}
    failures = []
    for expected in [
        dict(path=str(q.ROOT / methods.PROTOCOL), sha256=methods.PIN),
        *work["protocol"]["source_artifact_hashes"],
    ]:
        operand = Path(work["root"]) / Path(expected["path"]).relative_to(q.ROOT)
        ref = refs[str(operand)]
        if ref["sha256"] != expected["sha256"]:
            failures.append(
                methods.failure(operand, "sha256", expected["sha256"], ref["sha256"], ref["sha256"])
            )
    for name in methods.INPUTS:
        if not refs[str(Path(work["root"]) / name)]["exists"] and name != methods.ACTIVE:
            failures.append(methods.failure(Path(work["root"]) / name, "exists", True, None))
    operand = Path(work["root"]) / q.METHODS
    ref = refs[str(operand)]
    if ref["sha256"] != q.METHODS_PIN:
        failures.append(
            methods.failure(operand, "sha256", q.METHODS_PIN, ref["sha256"], ref["sha256"])
        )
    else:
        upstream = json.loads(Path(ref["snapshot_path"]).read_bytes())
        if upstream.get("margin_protocol_ready_score") != 1:
            failures.append(
                methods.failure(
                    operand,
                    "margin_protocol_ready_score",
                    1,
                    upstream.get("margin_protocol_ready_score"),
                    ref["sha256"],
                )
            )
    if failures != work["failures"]:
        raise ValueError("source_gate_drift")
    if not failures:
        read = lambda ref: q.original(work, ref)
        if methods.primitives(read, work["protocol"]) != work["public_rows"]:
            raise ValueError("source_primitive_drift")
        if q.global_rows(work) != (work["global_probabilities"], work["global_head"]):
            raise ValueError("global_head_drift")
        with tempfile.TemporaryDirectory(prefix="carnot8237-replay-") as directory:
            fresh = n.fit(
                [r for r in work["public_rows"] if r["role"] != "reserved"],
                work["protocol"]["role_manifest"],
                Path(directory),
            )
        if n.signature(fresh) != n.signature(work["fitted"]):
            raise ValueError("fit_replay_drift")
    if sha256_file(Path(value["trained_heads_path"])) != value["trained_heads_sha256"]:
        raise ValueError("head_bundle_drift")
    bundle = json.loads(Path(value["trained_heads_path"]).read_bytes())
    if (
        bundle["heads"] != work["fitted"].get("heads", [])
        or bundle["global_head"] != work["global_head"]
    ):
        raise ValueError("head_bundle_drift")
    for key, observed in q.reduce(work, value["validation_receipts"]).items():
        if value[key] != observed:
            raise ValueError("primitive_reduction_drift:" + key)
    return dict(passed=True, rows_checksum=canonical_hash(value["rows"]))


def main(argv: list[str] | None = None) -> int:
    """Keep scratch alive through validation and publish only after normal child exits."""
    q.progress("start", 0, 6)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261007"], default="20261007")
    parser.add_argument("--root", type=Path, default=q.ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--private-fixture", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        output = (args.output or q.ROOT / "results" / (q.NAME + ".json")).absolute()
        if args.private_fixture and (
            output.is_relative_to(q.ROOT / "results") or args.root == q.ROOT
        ):
            raise ValueError("private_fixture_requires_private_root_and_output")
        raw = output.parent / "raw" / q.NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True, mode=0o700)
        start, wall = time.monotonic_ns(), time.time_ns()
        with tempfile.TemporaryDirectory(prefix="carnot8237-", dir="/tmp") as directory:
            private = Path(directory)
            probe = private / "writable"
            probe.write_bytes(b"current private scratch")
            runtime = dict(
                path=str(private),
                writable=probe.read_bytes() == b"current private scratch",
                mode=oct(private.stat().st_mode & 0o777),
                executable=sys.executable,
                free_bytes=os.statvfs(private).f_bavail * os.statvfs(private).f_frsize,
            )
            plan = [] if args.private_fixture else commands(private)
            controls = x.pytest_plan(private / "controls")
            terminal = terminal_plan(output.parent / "raw" / q.NAME / "terminal_candidate.json")
            atomic_json(
                raw / "validation_manifest.json",
                dict(
                    commands=plan,
                    controls=controls,
                    terminal=terminal,
                    owned=q.OWNED,
                    runtime=runtime,
                    frozen_before_measurement_ns=time.monotonic_ns(),
                ),
            )
            q.progress("measurement_before", 0, 6)
            work = q.measure(args.root, raw)
            q.progress("measurement_after", len(work["fitted"].get("heads", [])), 0)
            receipts = x.execute(controls, raw / "control_logs") + x.execute(
                [s for s in plan if s["scope"] == "owned"], raw / "validation_logs"
            )
            health = x.execute(
                [s for s in plan if s["scope"] == "repository_health"], raw / "health_logs"
            )
            value = q.reduce(work, receipts)
            atomic_json(raw / "work.json", work)
            cov_path = private / "coverage.json"
            cov = json.loads(cov_path.read_bytes()) if cov_path.exists() else {}
            atomic_json(raw / "coverage.json", cov)
            end = time.monotonic_ns()
            value.update(
                duration_s=(end - start) / 1e9,
                random_seed=7128237,
                runtime_preconditions=runtime,
                repository_health=dict(owned=False, receipts=health),
                coverage_totals=cov.get("totals", {}),
                coverage_statement_counts=cov.get("files", {}),
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
                        phase="fit_and_validation",
                        started_monotonic_ns=start,
                        ended_monotonic_ns=end,
                        duration_s=(end - start) / 1e9,
                    )
                ]
                + [
                    dict(
                        phase=r["name"],
                        started_monotonic_ns=r["started_monotonic_ns"],
                        ended_monotonic_ns=r["ended_monotonic_ns"],
                        duration_s=r["duration_s"],
                    )
                    for r in receipts + health
                ],
                invocation_argv=sys.orig_argv,
                measurement_clocks=dict(
                    started_monotonic_ns=start,
                    ended_monotonic_ns=end,
                    started_wall_ns=wall,
                    ended_wall_ns=time.time_ns(),
                ),
                terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
            )
            value["reproducibility_checksum"] = canonical_hash(
                [value["work_reference"], value["code_config_hashes"], methods.PIN]
            )
            value["field_principles"] = {
                k: "Actual invocation bytes and clocks separate fit readiness from unmeasured scientific benefit."
                for k in value
            }
            value["field_principles"].update(
                margin_fit_ready_score="Authenticated fitting and selection plus passed owned checks; no positive effect is required.",
                trained_head_specs="Six small seventeen-coefficient selectors; historical Qwen provenance creates no current call.",
                margin_weight_rows="Native probabilities and original permissions set label-independent weights; missing rows stay missing.",
                fit_fold_rows="Disjoint original source folds, exact sample weights and real solver receipts.",
                comparator_name="Separate calibration-role all-slot cost, unweighted Brier and frozen tie order select the primary.",
                equally_weighted_simple_comparisons="Additive and logistic margin objectives have identical weight access to the energy margin head.",
                repository_health="Bounded full-suite failures remain separate from owned readiness.",
            )
            value = normalize_artifact_for_template_write(value)
            atomic_json(raw / "candidate.json", value)
            q.progress("publication_before", 6, 1)

            def validate(candidate: Path) -> Json:
                """Unchanged validators see exactly the locked candidate's bytes."""
                checked = x.execute(terminal, raw / "terminal_logs")
                return dict(passed=all(r["passed"] for r in checked), checks=checked)

            publication = publish_primary(output, value, validate)
            atomic_json(
                raw / "terminal_validation.json",
                dict(
                    publication=publication, required_checks_passed=value["required_checks_passed"]
                ),
            )
        q.progress("complete", len(work["fitted"].get("heads", [])), 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        print(json.dumps(dict(passed=False, error=str(error))), flush=True)
        return 1
