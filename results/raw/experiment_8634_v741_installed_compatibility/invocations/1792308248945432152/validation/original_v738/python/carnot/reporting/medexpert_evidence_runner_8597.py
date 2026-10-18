"""REQ-VERIFY-8597: run private replay and unchanged scanners before atomic publication."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, UTC
import json
from pathlib import Path
from typing import Any

from carnot.reporting import medexpert_evidence_8597 as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar, reader_receipt
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def validate(candidate: Path, raw: Path) -> Json:
    """Keep genuine exits and findings because a warning exit cannot qualify publication."""
    checks = []
    for name, args in [
        ("cold", [e.CLI, "--cold-replay", str(candidate)]),
        ("adversarial", ["scripts/adversarial_verify.py", "--json", str(candidate)]),
        ("strict_rows", ["scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)]),
    ]:
        checks.append(
            child(
                name,
                [str(e.ROOT / ".venv/bin/python"), "-u", *args],
                raw,
                deadline=60,
                heartbeat=20,
            )
        )
    return dict(passed=all(row["passed"] for row in checks), checks=checks)


def main(argv: list[str] | None = None) -> int:
    """Expose one CPU audit and keep rejection controls away from other experiment artifacts."""
    print("[exp8597] start no_model_load MODEL_SPECS=[] current_LLM_calls=0", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=datetime.now(UTC).strftime("%Y%m%d"))
    parser.add_argument("--manifest", type=Path, default=e.MANIFEST)
    parser.add_argument("--receipts", type=Path, default=e.RAW / "validation_receipts.json")
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    try:
        datetime.strptime(args.date, "%Y%m%d")
        if len(args.date) != 8:
            raise ValueError("date_format")
        if args.deliberate_error:
            raise ValueError("owned_failure_control")
        if args.cold_replay:
            print(json.dumps(e.replay(args.cold_replay)), flush=True)
            return 0
        print("[exp8597] phase=preconditions_and_inspection", flush=True)
        source = e.measure(args.manifest, args.output, args.date)
        receipts = json.loads(args.receipts.read_bytes()) if args.receipts.is_file() else []
        value = e.build(source, receipts, args.output)
        raw = Path(source["path"]).parent
        candidate = Path(e.read(source)["preconditions_checked"]["scratch"]) / (e.NAME + ".json")
        atomic_json(candidate, value)
        print("[exp8597] phase=initial_validation", flush=True)
        report = validate(candidate, raw / "initial_checks")
        receipts.extend(report["checks"])
        value = e.build(source, receipts, args.output)
        print("[exp8597] phase=terminal_publication", flush=True)
        publication = publish_primary(
            args.output, value, lambda path: validate(path, raw / "terminal_checks")
        )
        bound = read_bound_sidecar(args.output, Path(publication["sidecar_path"]))
        consumer = reader_receipt(
            e.TASK,
            args.output.parent,
            field="release_audit_ready_score",
            expected=value["release_audit_ready_score"],
        )
        atomic_json(
            Path(value["terminal_validation_sidecar_path"]),
            dict(
                publication=publication,
                report=bound["report"],
                downstream_consumer=consumer,
                downstream_acceptance=consumer["passed"],
                interpretation="Blocked evidence must not authorize independent evaluation.",
            ),
        )
        print(
            "[exp8597] published_" + value["verdict_class"] + " completed=1 pending=0", flush=True
        )
        return int(not value["required_checks_passed"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("[exp8597] rejected " + str(error), flush=True)
        return 1
