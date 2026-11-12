"""REQ-REPORT-9030-CUSTODY: publish only checks bound to this no-model audit."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

from carnot.reporting import retained_shard_a_9030 as d
from carnot.reporting import invocation_provenance_9000 as original
from carnot.reporting import scoped_provenance_9014 as checks
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v756_branch_qualification import measured, verify_measured

Json = dict[str, Any]
DIRECT = checks.DIRECT


def plan(private: Path) -> list[CommandSpec]:
    """Keep the complete frozen consumer roster, replacing only newly owned paths."""
    substitutions = dict(
        zip([*checks.OWNED, checks.TEST], [d.OWNED[0], d.CLI, d.TEST], strict=True)
    )
    result = []
    for spec in checks.validation_plan(private):
        argv = tuple(substitutions.get(a, a) for a in spec.argv)
        if spec.name in {"ruff_check", "ruff_format", "strict_mypy"}:
            argv += (d.OWNED[1],)
        result.append(
            replace(
                spec, argv=argv, timeout_s=600 if spec.name == "owned_tests" else spec.timeout_s
            )
        )
    (private / "coverage.ini").write_text(
        "[run]\nparallel=true\ndata_file="
        + str(private / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(d.ROOT / p) + "\n" for p in d.OWNED)
        + "[report]\nexclude_lines=\n"
    )
    return result


def replay(path: Path) -> Json:
    """Rebuild the scientific rows from original bytes even if an aggregate was rehashed."""
    try:
        value = json.loads(path.read_bytes())
        if d.checksum(value) != value["reproducibility_checksum"]:
            raise ValueError("checksum_drift")
        for ref in (
            value["source_artifact_hashes"]
            + value["raw_shard_hashes"]
            + value["code_config_hashes"]
        ):
            if d.reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("authority_bytes")
        work = json.loads(Path(value["raw_shard_hashes"][0]["path"]).read_bytes())
        if work["accepted"] and not (
            checks.receipts_pass(
                dict(
                    validation_scope_manifest=value["validation_scope_manifest"],
                    validation=dict(receipts=value["validation_receipts"]),
                    frozen_ns=value["validation_scope_manifest"]["frozen_ns"],
                )
            )
            and verify_measured(value["owned_statement_coverage"], d.OWNED)
        ):
            raise ValueError("validation_drift")
        work["evidence"]["work_references"] = value["raw_shard_hashes"]
        rebuilt = d.build(
            work["evidence"],
            value["validation_receipts"],
            value["owned_statement_coverage"],
            value["validation_scope_manifest"],
            value["duration_s"],
            accepted=work["accepted"],
        )
        if rebuilt != value:
            raise ValueError("reduction_drift")
        current = d.audit(
            Path(work["input_root"]), Path(tempfile.mkdtemp(prefix="carnot-9030-replay-"))
        )
        for field in [
            "rows",
            "role_counts",
            "nonexposure_audit",
            "raw_receipt_audit",
            "original_disqualified_primary",
            "downstream_evaluation_attempts",
        ]:
            # Object paths differ between invocations; their content and source identities do not.
            def normalized(obj: Any) -> Any:
                if isinstance(obj, dict):
                    return {k: normalized(v) for k, v in obj.items() if k != "path"}
                if isinstance(obj, list):
                    return [normalized(v) for v in obj]
                return obj

            if normalized(current[field]) != normalized(work["evidence"][field]):
                raise ValueError("primitive_drift:" + field)

        def failed(e: Json) -> list[Any]:
            return [
                (r["check"], r["artifact_field"], r["observed"])
                for r in e["checks"]
                if not r["passed"]
            ]

        if failed(current) != failed(work["evidence"]):
            raise ValueError("gate_drift")
        return dict(passed=True, reason="replay_passed")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        return dict(passed=False, reason=str(error))


def scanners(path: Path, raw: Path, scope: Json) -> Json:
    """Run unchanged validators on the actual candidate bytes before atomic publication."""
    receipts = [
        checks.run_check(CommandSpec(name, tuple(argv), "current_scoped", 120), raw, scope)
        for name, argv in [
            ("adversarial", [sys.executable, "scripts/adversarial_verify.py", str(path), "--json"]),
            (
                "row_consistency",
                [sys.executable, "scripts/verdict_row_consistency_lint.py", "--strict", str(path)],
            ),
            (
                "cold_replay",
                [
                    "env",
                    "-u",
                    "PYTHONPATH",
                    "-C",
                    tempfile.gettempdir(),
                    sys.executable,
                    "-u",
                    str(d.ROOT / d.CLI),
                    "--cold-replay",
                    str(path),
                ],
            ),
        ]
    ]
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Keep private CLI controls useful without mistaking them for measured acceptance."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", choices=["20261112"], default="20261112")
    parser.add_argument("--input-root", type=Path, default=d.ROOT)
    parser.add_argument("--output", type=Path, default=d.ROOT / "results" / (d.NAME + ".json"))
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        result = replay(args.cold_replay)
        print(json.dumps(result), flush=True)
        return int(not result["passed"])
    started = time.monotonic()
    raw = args.output.absolute().parent / "raw" / d.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    private = Path(tempfile.mkdtemp(prefix="carnot-9030-checks-"))
    commands = plan(private)
    os.environ["COVERAGE_RCFILE"] = str(private / "coverage.ini")
    scope = dict(
        frozen_ns=time.monotonic_ns(),
        owned_paths=d.OWNED,
        direct_consumer_tests=DIRECT,
        commands=[dict(name=c.name, argv=list(c.argv), timeout_s=c.timeout_s) for c in commands],
        code_test_hashes=[
            d.reference(d.ROOT / p)
            for p in dict.fromkeys([*d.OWNED, d.TEST, *DIRECT, *checks.REUSED])
        ],
        scope="New audit statements and all frozen direct consumers; historical repository health stays failed.",
    )
    atomic_json(raw / "validation_scope_manifest.json", scope)
    environment = original.environment(raw)
    evidence = d.audit(args.input_root, raw)
    evidence["environment"] = environment
    evidence["checks"].extend(evidence["environment"]["checks"])
    evidence["phase_spans"] = [
        dict(phase="authentication_and_reduction", duration_s=time.monotonic() - started)
    ]
    receipts, coverage = [], {}
    accepted = False
    if not args.audit_only:
        for index, command in enumerate(commands):
            d.progress("validation", index, len(commands) - index)
            receipts.append(checks.run_check(command, raw / "validation", scope))
        coverage = measured([private / ".coverage"], d.OWNED, raw / "measured")
        coverage.pop("rows", None)
        accepted = checks.receipts_pass(
            dict(
                validation_scope_manifest=scope,
                validation=dict(receipts=receipts),
                frozen_ns=scope["frozen_ns"],
            )
        ) and verify_measured(coverage, d.OWNED)
    work = dict(input_root=str(args.input_root.absolute()), evidence=evidence, accepted=accepted)
    atomic_json(raw / "work.json", work)
    evidence["work_references"] = [d.reference(raw / "work.json")]
    value = d.build(
        evidence, receipts, coverage, scope, time.monotonic() - started, accepted=accepted
    )
    # Work must bind the same reference-free evidence that was written before its own digest.
    candidate = raw / "candidate.json"
    atomic_json(candidate, value)
    if args.audit_only:
        atomic_json(args.output, value)
    else:
        report = scanners(candidate, raw / "terminal", scope)
        atomic_json(raw / "terminal_report.json", report)
        if not report["passed"]:
            return 1
        publish_primary(args.output, value, lambda path: scanners(path, raw / "publication", scope))
    d.progress("terminal_written", len(evidence["rows"]), 0)
    return 0
