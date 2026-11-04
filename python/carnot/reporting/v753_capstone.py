"""REQ-REPORT-8800: account for exact scheduled identities without promoting history."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import importlib
import json
from pathlib import Path
import resource
import shutil
import sys
import time
from typing import Any

# Retained reporting modules import helpers that live beside the Python package.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from carnot.reporting import v752_capstone as old
from carnot.reporting import v753_branch_qualification as q
from carnot.reporting.current_work_receipt import atomic_json as atomic_json
from carnot.reporting.roadmap_contract import parse_design
from scripts.summarize_artifact import summarize

Json = dict[str, Any]
ROOT, PIN, MILESTONE = q.ROOT, q.PIN, q.MILESTONE
NAME, TASK = "experiment_8800_v753_capstone", "exp8800-capstone"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [f"python/carnot/reporting/v753_capstone{s}.py" for s in ["", "_report", "_runner"]] + [CLI]
TESTS = ["tests/python/test_v753_capstone_8800.py"]
CONSUMERS = old.CONSUMERS
MODEL_SPECS: list[Json] = []
now, reference, digest = old.now, old.reference, old.digest


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Keep bounded CPU reconciliation visible without loading a model."""
    print(f"[exp8800] phase={phase} completed={completed} pending={pending}", flush=True)


def load(ref: Json) -> Json:
    """Reject arrays and changed bytes before any imported field gains meaning."""
    value = old.load(ref)
    if not isinstance(json.loads(Path(ref["path"]).read_bytes()), dict):
        raise ValueError("producer_object")
    return value


def observe(ref: Json, upstream: str, field: str, expected: Any = 1) -> Json:
    """Missing bytes, absent fields and measured zeros have different evidence limits."""
    return old.observe(ref, upstream, field, expected)


def contract(root: Path) -> Json:
    """Reuse the established activation encoding and independent full-object pin."""
    result = q.authority(root)
    if not result["ready"]:
        raise ValueError("v753_full_object_authority")
    tasks = parse_design((root / q.DESIGN).read_text(), milestone=MILESTONE)[1]
    return dict(tasks=tasks, visible_table=result["visible_table"], comparison=result)


def disciplined(path: Path, raw: Path) -> Json:
    """Retain the sanctioned reader's output before interpreting historical numbers."""
    capture = StringIO()
    with redirect_stdout(capture):
        status = summarize(path)
    target = raw / "summaries" / (path.stem + ".txt")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(capture.getvalue())
    return dict(reference(target), summary_status=status)


def slot(task: Json, root: Path, raw: Path) -> Json:
    """Retain alternate scheduler files without accepting them as primary producers."""
    number = int(task["id"].split("-")[0][3:])
    found = sorted((root / "results").glob(f"experiment_{number}_*.json"))
    summaries = [disciplined(p, raw) for p in found]
    row = old.slot(task, root)
    value = load(row["primary_reference"]) if row["primary_reference"]["sha256"] else {}
    row.update(
        excluded=bool(value.get("flagged_adversarial")),
        disposition="quarantined_evidence"
        if value.get("flagged_adversarial")
        else row["verdict_class"],
        summary_references=summaries,
        scientific_measurement_admitted=False,
        exposure_scope=value.get("exposure_scope", "no admitted measurement"),
        missing_reason=None if row["producer_executed"] else row["scheduler_outcome"],
    )
    return row


def preflight(root: Path, raw: Path) -> Json:
    """Measure parsing cost and available memory before collecting scientific dispositions."""
    raw.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    probe = raw / "scratch_probe"
    probe.write_bytes(b"private writable scratch")
    imports = True
    try:
        for name in ["coverage", "pytest", "yaml"]:
            importlib.import_module(name)
    except ImportError:
        imports = False
    tasks = contract(root)["tasks"]
    sources = [reference(ROOT / name) for name in OWNED]
    upstream_paths = {
        p
        for task in tasks
        for prior in task["prior_failures"]
        for p in (root / "results").glob(
            "experiment_" + prior["experiment_id"].split("-")[0][3:] + "_*.json"
        )
    }
    largest = max((p.stat().st_size for p in upstream_paths), default=0)
    elapsed = time.monotonic() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    available = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    checks = dict(
        private_scratch_writable=probe.read_bytes() == b"private writable scratch",
        dependency_imports_passed=imports,
        source_hashes_checked=all(r["sha256"] is not None for r in sources),
        tools_available=all(
            (ROOT / ".venv/bin" / name).is_file() for name in ["python", "pytest", "ruff", "mypy"]
        ),
        memory_available=available > rss * 2 + largest * 16,
        disk_available=shutil.disk_usage(raw).free > 128 * 1024**2,
        time_available=elapsed * 600 + 600 + 900 < 4800,
    )
    return dict(
        checks,
        projection_fits=all(checks.values()),
        parsing_probe_s=elapsed,
        measured_peak_rss_bytes=rss,
        available_memory_bytes=available,
        projected_memory_bytes=rss * 2 + largest * 16,
        largest_input_bytes=largest,
        source_hash_checks=sources,
        projected_validation_s=elapsed * 600 + 600,
        validation_reserve_s=900,
        task_cap_s=4800,
        no_model_load=True,
    )


def freeze(root: Path, raw: Path, run_date: str) -> Json:
    """Seal authority and prior receipts without running historical experiment controllers."""
    start = now()
    preconditions = preflight(root, raw)
    authority_root = q.snapshot(root, raw / "authority")
    authority = contract(authority_root)
    refs = [
        reference(authority_root / name)
        for name in [q.DESIGN, q.ACTIVE, q.STAGED]
        if (authority_root / name).is_file()
    ]
    rows: list[Json] = []
    priors: list[Json] = []
    for i, task in enumerate(authority["tasks"][:-1]):
        rows.append(slot(task, root, raw))
        progress("slot_frozen", i + 1, 11 - i)
    log = root / "ops/conductor-log.md"
    if log.is_file():
        shutil.copyfile(log, raw / "conductor-log.md")
    for row in rows:
        if row["conductor_log_reference"]:
            row["conductor_log_reference"] = reference(raw / "conductor-log.md")
        refs += [
            r
            for r in [
                row["primary_reference"],
                row["conductor_reference"],
                row["conductor_log_reference"],
                *row["evidence_references"],
                *row["summary_references"],
            ]
            if r and r["sha256"]
        ]
    for task in authority["tasks"]:
        for prior in task["prior_failures"]:
            progress(
                "prior_receipt_before",
                len(priors),
                sum(len(t["prior_failures"]) for t in authority["tasks"]) - len(priors),
            )
            n = prior["experiment_id"].split("-")[0][3:]
            paths = sorted((root / "results").glob(f"experiment_{n}_*.json"))
            ref = reference(
                paths[0] if len(paths) == 1 else root / "results" / f"experiment_{n}.json"
            )
            if ref["sha256"]:
                refs.append(disciplined(Path(ref["path"]), raw))
                refs.append(ref)
            value = load(ref) if ref["sha256"] else {}
            priors.append(
                dict(
                    task_id=task["id"],
                    prior_task_id=prior["experiment_id"],
                    primary_reference=ref,
                    actual_task_id=value.get("task_id"),
                    actual_honest_verdict=value.get("honest_verdict"),
                    registered_verdict_matches=value.get("honest_verdict") == prior["verdict"],
                    exact_identity_matches=value.get("task_id") == prior["experiment_id"],
                )
            )
    sources = [
        reference(ROOT / p)
        for p in OWNED
        + TESTS
        + CONSUMERS
        + [
            "scripts/experiment_template.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "scripts/publication_gate.py",
            "scripts/summarize_artifact.py",
            "scripts/recurring_blocker_ledger.py",
            "ops/exclusion_manifest.yaml",
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/reporting/v752_capstone.py",
            "python/carnot/reporting/v753_branch_qualification.py",
        ]
    ]
    work = dict(
        authority,
        slots=rows,
        root=str(root),
        raw=str(raw.absolute()),
        invocation_id=raw.name,
        run_date=run_date,
        started_at_utc=start,
        preconditions=preconditions,
        references=refs,
        sources=sources,
        prior_outcomes=priors,
    )
    atomic_json(raw / "work.json", work)
    return work


from carnot.reporting.v753_capstone_report import build as build, replay as replay
