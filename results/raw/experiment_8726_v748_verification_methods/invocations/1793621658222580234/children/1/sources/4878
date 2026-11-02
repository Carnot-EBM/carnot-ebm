"""REQ-REPORT-8660: owned teardown gates a new packaging qualification.

The old disqualification stays immutable. This invocation grants no timing,
learning or generalization credit, even when every installed check passes.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

from carnot.reporting import installed_checkout_8660 as c
from carnot.reporting import installed_compatibility_8646 as old
from carnot.reporting import installed_report_8646 as report
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar, reader_receipt
from carnot.reporting.v709_execution import child, progress

Json = dict[str, Any]
NAME = "experiment_8660_v743_installed_compatibility"
TASK = "exp8660-installed-compatibility"
CLI = f"scripts/experiments/{NAME}.py"
MODEL_SPECS: list[Json] = []
OWNED = [
    f"python/carnot/reporting/installed_{part}_8660.py"
    for part in ["checkout", "validation", "report"]
] + [CLI]
TESTS = [
    f"tests/python/test_installed_{part}_8660.py" for part in ["checkout", "validation", "report"]
]
HISTORY = "results/experiment_8646_v742_installed_compatibility.json"
HISTORY_PIN = "sha256:1b25fbca7e8a4b313c9ba99eca8378ca09727ce090fb37003a0e2f2794361187"


def measure(raw: Path, run_date: str, root: Path) -> Json:
    """Authenticate the failed teardown rather than inferring success from numerical rows."""
    work = old.measure(raw, run_date, root)
    work.update(
        materialization_manifest={}, alias_rejection_rows=[], original_result_hashes_before_after=[]
    )
    if not work["gate_check_summary"]:
        historical = old.load(dict(path=str(root / HISTORY), sha256=HISTORY_PIN))
        receipt = next(row for row in historical["validation_receipts"] if row["name"] == "nodes-1")
        log = Path(receipt["stdout_path"])
        if c.reference(log)["sha256"] != receipt["stdout_sha256"]:
            raise ValueError("historical_log_hash")
        text = log.read_text()
        if "6 passed, 1 warning, 2 errors" not in text or "TrackedResultWriteError" not in text:
            raise ValueError("historical_teardown_evidence")
        work["preconditions_checked"]["authenticated_tools"] = [
            c.reference(c.ROOT / ".venv/bin" / name)
            for name in ["python", "pytest", "coverage", "ruff", "mypy"]
        ]
        work["historical_8646"] = dict(
            reference=c.reference(root / HISTORY),
            log=c.reference(log),
            verdict=historical["honest_verdict"],
            runner_passed=6,
            teardown_errors=2,
            original_credited=116,
            original_intended=122,
            current_passed=73,
            numerical_rows=690,
        )
        work["source_artifact_hashes"].extend([c.reference(root / HISTORY), c.reference(log)])
        work["original_result_hashes_before_after"] = [
            dict(before=ref) for ref in work["source_artifact_hashes"]
        ]
        guard = c.guard_control(raw / "ownership-control")
        work["alias_rejection_rows"] = [dict(unit_id="guarded_private_alias", **guard)]
        work["cleanup_rows"].append(dict(unit_id="guarded_fixture_teardown", **guard))
    return work


def artifact(authority: Json) -> Json:
    """Reuse the unchanged numerical reducer and bind all additional ownership evidence."""
    work = old.load(authority)
    value = report.artifact(authority)
    value.update(
        experiment_id=8660,
        task_id=TASK,
        milestone="2026.10.743",
        random_seed=7438660,
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
    )
    for key in [
        "historical_8646",
        "materialization_manifest",
        "alias_rejection_rows",
        "original_result_hashes_before_after",
    ]:
        value[key] = work.get(
            key, {} if key in {"historical_8646", "materialization_manifest"} else []
        )
    ownership = dict(
        original122=work.get("original_node_union", {}).get("passed") is True,
        current_union=work.get("current_node_union", {}).get("passed") is True,
        private_alias_rejected=bool(work.get("alias_rejection_rows"))
        and all(row["passed"] for row in work["alias_rejection_rows"]),
        owned_cleanup=bool(work.get("materialization_manifest", {}).get("copies")),
        source_hashes_unchanged=bool(work.get("original_result_hashes_before_after"))
        and all(row.get("passed", False) for row in work["original_result_hashes_before_after"]),
    )
    value["acceptance_gates"].extend(
        dict(name=name, passed=passed) for name, passed in ownership.items()
    )
    if not work["gate_check_summary"] and not all(ownership.values()):
        value.update(
            verdict_class="disqualified",
            installed_measurement_ready_score=0,
            required_checks_passed=False,
        )
        value["branch_gate_values"]["installed_measurement_ready_score"] = 0
    value["honest_verdict"] = "complete_" + value["verdict_class"] + "_v743_installed_compatibility"
    value["methodology"] = (
        "Authenticate Exp8646 teardown and original122 test bytes. Reproduce the unchanged guard in a private git fixture. "
        "Overlay only private materialization, historical protocol and independently sealed V738 authority in child namespaces. "
        "Build a fresh current wheel and exercise installed Python/native complete responses and public/rejection surfaces. "
        "Keep original and current child coverage separate. Qualification supplies no speed or generalization result."
    )
    value["field_principles"] = {
        key: "Bind actual immutable invocation evidence; ownership and packaging grant no scientific benefit."
        for key in value
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path, authority: Json) -> None:
    """An independently pinned work file prevents self-consistent evidence substitution."""
    value = json.loads(path.read_bytes())
    if value != artifact(authority):
        raise ValueError("immutable_reduction_drift")
    for ref in (
        value["source_artifact_hashes"]
        + value["code_config_hashes"]
        + value["raw_shard_hashes"]
        + value["child_shard_manifest"]
    ):
        if c.reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
            raise ValueError("replay_input_hash")


def terminal(path: Path, authority: Json, logs: Path) -> Json:
    """Inspect checked bytes with real readers and fresh missing/tamper processes."""
    changed = json.loads(path.read_bytes())
    changed["completed_count"] += 1
    changed["reproducibility_checksum"] = canonical_hash(changed)
    bad = logs / "rehashed_tamper.json"
    atomic_json(bad, changed)
    replay_args = ["--authority", authority["path"], authority["sha256"]]
    receipts = []
    for name, argv, expected in [
        ("cold", [CLI, "--cold-replay", str(path), *replay_args], 0),
        ("adversarial", ["scripts/adversarial_verify.py", "--json", str(path)], 0),
        ("strict_rows", ["scripts/verdict_row_consistency_lint.py", "--strict", str(path)], 0),
        ("missing", [CLI, "--cold-replay", str(logs / "missing.json"), *replay_args], 1),
        ("rehashed_tamper", [CLI, "--cold-replay", str(bad), *replay_args], 1),
    ]:
        receipts.append(
            child(name, [sys.executable, "-u", *argv], logs, expected=expected, deadline=90)
        )
    findings = json.loads(Path(receipts[1]["stdout_path"]).read_text())
    return dict(
        passed=all(row["passed"] for row in receipts) and findings["flagged_count"] == 0,
        receipts=receipts,
        actual_findings=findings,
    )


def run(output: Path, run_date: str, root: Path) -> int:
    """Measure once, retain bounded validation, and atomically publish checked bytes."""
    from carnot.reporting.installed_validation_8660 import validate

    raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
    started = time.monotonic_ns()
    work = measure(raw, run_date, root)
    before = time.monotonic_ns()
    work["validation_receipts"] = validate(work, raw)
    ended = time.monotonic_ns()
    work["duration_s"] = (ended - started) / 1e9
    work["phase_spans"].append(dict(phase="validation", start_ns=before, end_ns=ended))
    for row in work["original_result_hashes_before_after"]:
        row["after"] = c.reference(Path(row["before"]["path"]))
        row["passed"] = row["before"] == row["after"]
    work["validation_receipts"].append(
        dict(
            name="source_evidence_unchanged",
            passed=all(row["passed"] for row in work["original_result_hashes_before_after"]),
        )
    )
    work["validation_receipts"].append(dict(name="runtime_cap", passed=work["duration_s"] < 4800))
    work["code_config_hashes"] = [c.reference(c.ROOT / p) for p in OWNED + TESTS]
    work["raw_shard_hashes"] = [c.reference(p) for p in sorted(raw.rglob("*")) if p.is_file()]
    authority_path = raw / "authority.json"
    atomic_json(authority_path, work)
    authority = c.reference(authority_path)
    value = artifact(authority)
    result = publish_primary(
        output, value, lambda path: terminal(path, authority, raw / "terminal")
    )
    atomic_json(
        raw / "terminal_validation.json", read_bound_sidecar(output, Path(result["sidecar_path"]))
    )
    reader = reader_receipt(
        TASK,
        output.parent,
        field="installed_measurement_ready_score",
        expected=value["installed_measurement_ready_score"],
    )
    atomic_json(raw / "reader_receipt.json", reader)
    if not reader["passed"]:
        raise ValueError("publication_reader")
    progress(
        "8660_complete_" + value["verdict_class"], value["completed_count"], value["censored_count"]
    )
    return int(value["verdict_class"] == "disqualified")


def main(argv: list[str] | None = None) -> int:
    """Private paths support real CLI rejection and cold replay outside the checkout."""
    progress("8660_start_no_model_load_MODEL_SPECS=[]_current_language_model_calls=0")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261019")
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
            or (args.private_output and args.private_output.resolve().is_relative_to(c.ROOT))
        ):
            raise ValueError("date_or_private_boundary")
        return run(
            args.private_output or c.ROOT / "results" / (NAME + ".json"),
            args.date,
            args.input_root or c.ROOT,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[8660] error={error}", file=sys.stderr, flush=True)
        return 1
