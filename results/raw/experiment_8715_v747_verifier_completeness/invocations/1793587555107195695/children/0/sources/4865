"""REQ-REPORT-8646: publish immutable compatibility facts without scientific credit."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

from carnot.reporting import installed_compatibility_8646 as e
from carnot.reporting import installed_runner_8634 as prior
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary, reader_receipt, read_bound_sidecar
from carnot.reporting.v709_execution import child, progress

Json = dict[str, Any]
REQUIRED = prior.REQUIRED


def artifact(authority: Json) -> Json:
    """Reuse unchanged numerical checks while binding new identity and both source roots."""
    work = e.load(authority)
    value = prior.artifact(authority)
    value.update(
        experiment_id=8646,
        task_id=e.TASK,
        milestone="2026.10.742",
        random_seed=7428646,
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        methodology="Authenticate original protocol bytes and qualify the existing mount overlay. Execute the unchanged original122 nodes separately from current tests. Build and install a fresh current wheel outside the checkout. Reuse fixed original numerical controls and complete public/rejection contracts. Retain genuine shards before combining copies. No language model or independent factuality claim.",
    )
    for key in [
        "current_node_union",
        "historical_source_reference",
        "resolved_import_rows",
        "isolation_rows",
        "cleanup_rows",
        "wheel_identity",
        "installed_module_hashes",
    ]:
        value[key] = work.get(key, [] if key.endswith("rows") or key.endswith("hashes") else {})
    value["preserved_8634_verdict"] = work.get(
        "preserved_8634_verdict", "complete_disqualified_installed_compatibility"
    )
    value["dependencies"] = work.get("dependencies", [])
    value["honest_verdict"] = "complete_" + value["verdict_class"] + "_v742_installed_compatibility"
    value["field_principles"] = {
        key: "Bind actual invocation evidence; historical and current bytes remain separate and grant no scientific benefit."
        for key in value
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path, authority: Json) -> None:
    """A separately pinned authority rejects repaired summaries and changed primitive bytes."""
    value = json.loads(path.read_bytes())
    if value != artifact(authority):
        raise ValueError("immutable_reduction_drift")
    for ref in (
        value["source_artifact_hashes"]
        + value["code_config_hashes"]
        + value["raw_shard_hashes"]
        + value["child_shard_manifest"]
    ):
        if e.reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
            raise ValueError("replay_input_hash")


def terminal(path: Path, authority: Json, logs: Path) -> Json:
    """Run fresh valid, missing and repaired-tamper controls through unchanged readers."""
    changed = json.loads(path.read_bytes())
    changed["completed_count"] += 1
    changed["reproducibility_checksum"] = canonical_hash(changed)
    bad = logs / "rehashed_tamper.json"
    atomic_json(bad, changed)
    replay_args = ["--authority", authority["path"], authority["sha256"]]
    receipts = []
    for name, argv, expected in [
        ("cold", [e.CLI, "--cold-replay", str(path), *replay_args], 0),
        ("adversarial", ["scripts/adversarial_verify.py", "--json", str(path)], 0),
        ("strict_rows", ["scripts/verdict_row_consistency_lint.py", "--strict", str(path)], 0),
        ("missing", [e.CLI, "--cold-replay", str(logs / "missing.json"), *replay_args], 1),
        ("rehashed_tamper", [e.CLI, "--cold-replay", str(bad), *replay_args], 1),
    ]:
        receipts.append(
            child(name, [sys.executable, "-u", *argv], logs, expected=expected, deadline=90)
        )
    findings = json.loads(Path(receipts[1]["stdout_path"]).read_text())
    return dict(
        passed=all(r["passed"] for r in receipts) and findings["flagged_count"] == 0,
        receipts=receipts,
        actual_findings=findings,
    )


def run(output: Path, run_date: str, root: Path) -> int:
    """Reserve validation time and atomically publish only checked final bytes."""
    from carnot.reporting.installed_validation_8646 import validate

    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    started = time.monotonic_ns()
    work = e.measure(raw, run_date, root)
    before = time.monotonic_ns()
    work["validation_receipts"] = validate(work, raw)
    ended = time.monotonic_ns()
    work["duration_s"] = (ended - started) / 1e9
    work["phase_spans"].append(dict(phase="validation", start_ns=before, end_ns=ended))
    if work["duration_s"] > 4500:
        work["validation_receipts"].append(dict(name="runtime_cap", passed=False))
    work["code_config_hashes"] = [e.reference(e.ROOT / p) for p in e.OWNED + e.TESTS]
    work["raw_shard_hashes"] = [e.reference(p) for p in sorted(raw.rglob("*")) if p.is_file()]
    authority_path = raw / "authority.json"
    atomic_json(authority_path, work)
    authority = e.reference(authority_path)
    value = artifact(authority)
    result = publish_primary(
        output, value, lambda path: terminal(path, authority, raw / "terminal")
    )
    atomic_json(
        raw / "terminal_validation.json", read_bound_sidecar(output, Path(result["sidecar_path"]))
    )
    reader = reader_receipt(
        e.TASK,
        output.parent,
        field="installed_measurement_ready_score",
        expected=value["installed_measurement_ready_score"],
    )
    atomic_json(raw / "reader_receipt.json", reader)
    if not reader["passed"]:
        raise ValueError("publication_reader")
    progress(
        "8646_complete_" + value["verdict_class"], value["completed_count"], value["censored_count"]
    )
    return int(value["verdict_class"] == "disqualified")


def main(argv: list[str] | None = None) -> int:
    """Private outputs support safe replay and failure controls outside the checkout."""
    progress("8646_start_no_model_load_MODEL_SPECS=[]_current_language_model_calls=0")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261018")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--authority", nargs=2)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            if not args.authority:
                raise ValueError("immutable_authority_required")
            replay(args.cold_replay, dict(path=args.authority[0], sha256=args.authority[1]))
            return 0
        if (
            datetime.strptime(args.date, "%Y%m%d").strftime("%Y%m%d") != args.date
            or (args.input_root and not args.private_output)
            or (args.private_output and args.private_output.resolve().is_relative_to(e.ROOT))
        ):
            raise ValueError("date_or_private_boundary")
        return run(
            args.private_output or e.ROOT / "results" / (e.NAME + ".json"),
            args.date,
            args.input_root or e.ROOT,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[8646] error={error}", file=sys.stderr, flush=True)
        return 1
