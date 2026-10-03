"""REQ-REPORT-7953-V690: observe activation without reconstructing consumed staging.

The executable task digest binds prompts. Failure readers decide whether empty
history is legitimate; historical staging custody remains a separate observation.
"""

from copy import deepcopy
from functools import lru_cache
import hashlib
from pathlib import Path
import tempfile
from typing import Any

import yaml

from carnot.reporting import v685_authority_lifecycle as lifecycle
from carnot.reporting import v686_contract_methods as shared
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from scripts import exclusion_manifest_lint as exclusions
from scripts.failure_ledger import FailureLedger, validate_prior_failures

ROOT = Path(__file__).resolve().parents[3]
MILESTONE = "2026.09.690"


def lineage(task: dict[str, Any], ledger: FailureLedger, risks: list[Any]) -> dict[str, Any]:
    """Ask the existing readers before accepting an empty history list."""
    matches = ledger.matching_priors(task)
    relevant = [r for r in risks if r.task_id == task.get("id") and r.severity == "HARD"]
    history = task.get("prior_failures", [])
    valid = isinstance(history, list) and (
        (not matches and not history)
        or (
            validate_prior_failures(task).valid
            and all(item.get("retire_if_same_verdict") is True for item in history)
        )
    )
    return dict(
        task_id=task.get("id"),
        scope_matched=bool(matches),
        required_history=bool(matches or relevant),
        matched_priors=[
            dict(
                experiment_id=m.experiment_id,
                verdict=m.verdict,
                path=str(m.artifact_path),
                sha256=sha256_file(m.artifact_path) if m.artifact_path else None,
            )
            for m in matches
        ],
        exclusion_risks=[r.violation_class for r in relevant],
        reader_receipt=dict(
            lineage_reader="scripts.failure_ledger.FailureLedger.matching_priors",
            exclusion_reader="scripts.exclusion_manifest_lint.lint",
            lineage_sha256=sha256_file(ROOT / "scripts/failure_ledger.py"),
            exclusion_sha256=sha256_file(ROOT / "scripts/exclusion_manifest_lint.py"),
            manifest_sha256=sha256_file(ROOT / "ops/exclusion_manifest.yaml"),
        ),
        passed=valid and not relevant,
    )


@lru_cache(maxsize=16)
def _reader_rows(raw: bytes, corpus_digest: str) -> list[dict[str, Any]]:
    """Reuse unchanged reader results only for identical code, corpus and task bytes."""
    ledger = FailureLedger.load_from_artifacts(ROOT)
    with tempfile.TemporaryDirectory(prefix="exp7953-readers-") as directory:
        source = Path(directory) / "observed-roadmap.yaml"
        source.write_bytes(raw)
        risks = exclusions.lint(source)
    rows = [lineage(t, ledger, risks) for t in yaml.safe_load(raw)["tasks"]]
    for row in rows:
        row["reader_receipt"].update(
            input_sha256="sha256:" + hashlib.sha256(raw).hexdigest(), corpus_sha256=corpus_digest
        )
    return rows


def reader_rows(raw: bytes, corpus_digest: str) -> list[dict[str, Any]]:
    """Copy cached receipts so callers cannot rewrite the frozen reader result."""
    return deepcopy(_reader_rows(raw, corpus_digest))


def assess(
    design: Path,
    staged: Path,
    active: Path,
    snapshots: Path,
    *,
    milestone: str = MILESTONE,
    first_id: int = 7953,
) -> dict[str, Any]:
    """Correct only the task-owned history and staging checks from the old reader."""
    try:
        value = lifecycle.assess_authorities(
            design, staged, active, snapshots, milestone=milestone, first_id=first_id, count=13
        )
    except (OSError, ValueError, IndexError, KeyError, TypeError, yaml.YAMLError):
        value = shared.assess(
            design, staged, active, snapshots, milestone=milestone, first_id=first_id, count=13
        )
    actual = active if active.is_file() else staged
    tasks = yaml.safe_load(actual.read_bytes()).get("tasks", []) if actual.is_file() else []
    rows = []
    if (
        lifecycle.tasks_digest(tasks) == value["canonical_tasks_sha256"]
        and yaml.safe_load(actual.read_bytes()).get("milestone") == milestone
    ):
        corpus = list((ROOT / "results").glob("experiment_*.json")) + [
            ROOT / "scripts/failure_ledger.py",
            ROOT / "scripts/exclusion_manifest_lint.py",
            ROOT / "scripts/in_process_doc_reconcile.py",
            ROOT / "ops/exclusion_manifest.yaml",
        ]
        digest = canonical_hash({str(p): sha256_file(p) for p in sorted(corpus)})
        rows = reader_rows(actual.read_bytes(), digest)
    failures = [
        f for f in value["gate_check_summary"] if f["artifact_field"] != "contract_rows.matched"
    ]
    for index, row in enumerate(value["contract_rows"]):
        row["checks"]["prior"] = index < len(rows) and rows[index]["passed"]
        row["matched"] = all(row["checks"].values())
        row.update(
            absolute_metric=int(row["matched"]),
            raw_numerator=sum(row["checks"].values()),
            raw_denominator=len(row["checks"]),
            excluded=not row["matched"],
        )
    if not all(r["matched"] for r in value["contract_rows"]):
        failures.append(
            shared.operand(
                active,
                "contract_rows.matched",
                True,
                [r["order"] for r in value["contract_rows"] if not r["matched"]],
                "V690_authority",
            )
        )
    expected = value["canonical_tasks_sha256"]
    preserved = []
    for path in snapshots.glob("staged-*.bin"):
        raw = yaml.safe_load(path.read_bytes())
        matching = (
            isinstance(raw, dict)
            and raw.get("milestone") == milestone
            and lifecycle.tasks_digest(raw.get("tasks")) == expected
            and sha256_file(path) == "sha256:" + path.stem.removeprefix("staged-")
        )
        preserved.append(dict(path=str(path), sha256=sha256_file(path), matching=matching))
        if not matching:
            failures.append(
                shared.operand(path, "preserved_staging_matches", True, False, "V690_authority")
            )
    stage = value["authority_snapshots"]["staged"]
    if stage["exists"]:
        planned = yaml.safe_load(staged.read_bytes())
        stage_ok = (
            isinstance(planned, dict)
            and planned.get("milestone") == milestone
            and lifecycle.tasks_digest(planned.get("tasks")) == expected
        )
        if not stage_ok:
            failures.append(
                shared.operand(staged, "staging_matches", True, False, "V690_authority")
            )
        status = "matching_observed" if stage_ok else "contradictory"
    else:
        status = (
            "matching_preserved"
            if preserved and all(p["matching"] for p in preserved)
            else "unknown_consumed"
        )
    if any(not p["matching"] for p in preserved):
        status = "contradictory"
    observed = not failures
    planning = bool(
        value["planning_matched"] and status != "contradictory" and all(r["passed"] for r in rows)
    )
    value.update(
        activated=observed,
        observed_activation=observed,
        staging_custody_status=status,
        planning_ready_score=int(planning),
        lineage_applicability_rows=rows,
        preserved_staging_snapshots=preserved,
        gate_check_summary=failures,
    )
    return value
