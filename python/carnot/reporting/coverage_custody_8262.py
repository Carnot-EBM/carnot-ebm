"""REQ-VERIFY-8262: preserve measured reports before their scratch owner exits.

The report path comes from the executed argv, never from a guessed log name.
Statement sets are checked against the actual source so rehashed summaries
cannot invent measured coverage. This hook can serve the next thin runner.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from coverage.parser import PythonParser

from carnot.reporting.current_work_receipt import atomic_json, sha256_file

Json = dict[str, Any]


def operand(spec: Json) -> Path:
    """Require an explicit JSON output so the reader follows the producer contract."""
    argv = spec["argv"]
    if "json" not in argv or "-o" not in argv:
        raise ValueError("coverage_json_operand")
    return Path(argv[argv.index("-o") + 1]).absolute()


def counts(report: Path, owned: list[Json]) -> Json:
    """Recompute every statement count rather than trusting a percentage headline."""
    value = json.loads(report.read_bytes())
    files = value["files"]
    normalized = {str(Path(name).resolve()): entry for name, entry in files.items()}
    if not owned or len(normalized) != len(files) or set(normalized) != {r["path"] for r in owned}:
        raise ValueError("complete_owned_file_identity")
    result = {}
    for ref in owned:
        path = Path(str(ref.get("snapshot_path", ref["path"])))
        if sha256_file(path) != ref["sha256"]:
            raise ValueError("owned_code_hash")
        parser = PythonParser(filename=str(path))
        parser.parse_source()
        entry = normalized[ref["path"]]
        executed, missing, excluded = (
            set(entry[key]) for key in ["executed_lines", "missing_lines", "excluded_lines"]
        )
        summary = entry["summary"]
        total = len(parser.statements)
        if (
            total == 0
            or executed != parser.statements
            or missing
            or excluded
            or summary["num_statements"] != total
            or summary["covered_lines"] != len(executed)
            or summary["missing_lines"] != 0
            or summary["excluded_lines"] != 0
        ):
            raise ValueError("measured_statement_counts")
        result[ref["label"]] = dict(
            num_statements=total, covered_lines=len(executed), missing_lines=0, excluded_lines=0
        )
    if any(
        value["totals"][key] != sum(row[key] for row in result.values())
        for key in ["num_statements", "covered_lines", "missing_lines", "excluded_lines"]
    ):
        raise ValueError("coverage_totals_drift")
    return result


def preserve(root: Path, spec: Json, receipt: Json, owned: list[str], raw: Path) -> Json:
    """Atomically copy primitive bytes and their real command receipt before cleanup."""
    source = operand(spec)
    if (
        receipt["argv"] != spec["argv"]
        or receipt["exit_code"] != 0
        or receipt["actual_exit"] != 0
        or receipt["passed"] is not True
        or receipt["timed_out"]
        or receipt["normal_exit"] is not True
    ):
        raise ValueError("coverage_child_exit")
    stamp = source.stat().st_mtime_ns
    end_wall = (
        receipt["started_wall_ns"] + receipt["ended_monotonic_ns"] - receipt["started_monotonic_ns"]
    )
    generated = int(
        datetime.fromisoformat(json.loads(source.read_bytes())["meta"]["timestamp"]).timestamp()
        * 1e9
    )
    if (
        not receipt["started_wall_ns"] <= stamp <= end_wall
        or not receipt["started_wall_ns"] <= generated <= end_wall
    ):
        raise ValueError("coverage_report_freshness")
    refs = [
        dict(label=name, path=str((root / name).resolve()), sha256=sha256_file(root / name))
        for name in owned
    ]
    for index, ref in enumerate(refs):
        saved_code = raw / "owned" / f"{index}.py"
        saved_code.parent.mkdir(parents=True, exist_ok=True)
        saved_code.write_bytes(Path(ref["path"]).read_bytes())
        ref["snapshot_path"] = str(saved_code)
    measured = counts(source, refs)
    raw.mkdir(parents=True, exist_ok=True)
    target = raw / "coverage.json"
    temporary = raw / ".coverage.json.tmp"
    with temporary.open("wb") as stream:
        stream.write(source.read_bytes())
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)
    binding = dict(
        source_report_path=str(source),
        report_path=str(target),
        report_sha256=sha256_file(target),
        source_mtime_ns=stamp,
        report_generated_wall_ns=generated,
        owned_files=refs,
        owned_statement_counts=measured,
    )
    saved = raw / "coverage_command_receipt.json"
    atomic_json(saved, dict(binding, command_receipt=receipt))
    return dict(binding, receipt_path=str(saved), receipt_sha256=sha256_file(saved))


def replay(binding: Json) -> Json:
    """Read durable primitives in a fresh process without needing removed scratch."""
    report, receipt_path = Path(binding["report_path"]), Path(binding["receipt_path"])
    if (
        sha256_file(report) != binding["report_sha256"]
        or sha256_file(receipt_path) != binding["receipt_sha256"]
    ):
        raise ValueError("coverage_custody_hash")
    saved = json.loads(receipt_path.read_bytes())
    if saved != {
        k: v for k, v in binding.items() if k not in {"receipt_path", "receipt_sha256"}
    } | {"command_receipt": saved["command_receipt"]}:
        raise ValueError("coverage_receipt_binding")
    receipt = saved["command_receipt"]
    if (
        str(operand(receipt)) != binding["source_report_path"]
        or receipt["exit_code"] != 0
        or receipt["actual_exit"] != 0
        or receipt["passed"] is not True
        or receipt["timed_out"]
        or receipt["normal_exit"] is not True
        or not receipt["started_wall_ns"]
        <= binding["report_generated_wall_ns"]
        <= receipt["started_wall_ns"]
        + receipt["ended_monotonic_ns"]
        - receipt["started_monotonic_ns"]
        or not receipt["started_wall_ns"]
        <= binding["source_mtime_ns"]
        <= receipt["started_wall_ns"]
        + receipt["ended_monotonic_ns"]
        - receipt["started_monotonic_ns"]
    ):
        raise ValueError("coverage_command_provenance")
    for prefix in ["stdout", "stderr"]:
        if sha256_file(Path(receipt[prefix + "_path"])) != receipt[prefix + "_sha256"]:
            raise ValueError("coverage_command_stream")
    measured = counts(report, binding["owned_files"])
    if measured != binding["owned_statement_counts"]:
        raise ValueError("coverage_summary_drift")
    return measured
