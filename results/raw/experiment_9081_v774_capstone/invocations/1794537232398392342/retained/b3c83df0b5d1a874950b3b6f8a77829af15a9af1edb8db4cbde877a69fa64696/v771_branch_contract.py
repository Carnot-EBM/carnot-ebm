"""REQ-REPORT-9027-V771: bind current custody while retaining frozen scientific authority."""

from collections.abc import Iterator
from contextlib import contextmanager
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import v770_branch_contract as old
from carnot.reporting import v771_branch_evidence as e
from carnot.reporting.roadmap_contract import compare_contract, parse_design

Json = dict[str, Any]
ROOT, DESIGN, ACTIVE, STAGED = old.ROOT, old.DESIGN, old.ACTIVE, old.STAGED
NAME, TASK, MILESTONE = (
    "experiment_9027_v771_branch_contract",
    "exp9027-branch-contract",
    "2026.11.771",
)
PIN = "sha256:a353be3a3351cc47a11d5c93a3a36a057653d9851e17da6896e33a7737e9ce1c"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_v771_branch_contract_9027.py"
HEALTH = f"results/raw/{NAME}/repository_health/full_python_suite_once.receipt.json"
OWNED = [
    "python/carnot/reporting/v771_branch_contract.py",
    "python/carnot/reporting/v771_branch_evidence.py",
    CLI,
]
DIRECT, E2E, CHECKS, READINESS = old.DIRECT, old.E2E, old.CHECKS, old.READINESS
REUSED = [*old.REUSED, *old.OWNED]
MODEL_SPECS: list[Json] = []
reference, checksum, seal, verify_measured = (
    old.reference,
    old.checksum,
    old.seal,
    old.verify_measured,
)
atomic_json, canonical_hash = old.atomic_json, old.canonical_hash
_scope, _freeze, _build, _replay = old.scope_manifest, old.freeze, old.build, old.replay
_mutations, _validate, _terminal, _run_check = (
    old.mutations,
    old.validate,
    old.terminal,
    old.run_check,
)
_resources = old.old.prior.resources


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Actual phase counts let an operator distinguish a bounded child from a stall."""
    print(f"[exp9027] phase={phase} completed={completed} pending={pending}", flush=True)


def resources(raw: Path, run_date: str) -> Json:
    """Only the new run date changes; the original CPU and disk floors stay fixed."""
    return dict(
        _resources(raw, run_date),
        run_date_matches=run_date == "20261112",
        network_requests_required=0,
        network_request_limit=0,
    )


@contextmanager
def binding() -> Iterator[None]:
    """Local bindings reuse frozen code without editing historical producers or validators."""
    with (
        patch.multiple(
            old,
            NAME=NAME,
            TASK=TASK,
            MILESTONE=MILESTONE,
            PIN=PIN,
            CLI=CLI,
            TEST=TEST,
            OWNED=OWNED,
            REUSED=REUSED,
            e=e,
            progress=progress,
            scope_manifest=scope_manifest,
            authority=authority,
            mutations=mutations,
            build=build,
            run_check=run_check,
            verify_measured=verify_measured,
        ),
        patch.object(old.old.prior, "resources", resources),
    ):
        yield


def authority(root: Path) -> Json:
    """Compare the selected lifecycle bytes, including every prompt and producer gate field."""
    tasks, rows, checks = [], [], []
    mode, path = "unavailable", root / DESIGN
    try:
        text = path.read_text()
        _, tasks = parse_design(text, milestone=MILESTONE)
        active = yaml.safe_load((root / ACTIVE).read_bytes())
        mode = "activated" if active["milestone"] == MILESTONE else "staged"
        selected = active if mode == "activated" else yaml.safe_load((root / STAGED).read_bytes())
        old.old.Roadmap.model_validate(selected)
        from copy import deepcopy

        selected = deepcopy(selected)
        for actual, expected in zip(selected["tasks"], tasks):
            if actual == dict(expected, prompt=old.old.prior.encode(expected["prompt"])):
                actual["prompt"] = expected["prompt"]
        compared = compare_contract(text, selected, selected, milestone=MILESTONE, first_id=9027)
        rows = compared["rows"]
        producers = {t["id"]: t for t in tasks}
        for i, row in enumerate(rows):
            row["checks"]["full_object"] = (
                i < min(len(tasks), len(selected["tasks"])) and tasks[i] == selected["tasks"][i]
            )
            row["checks"]["declared_gate_fields"] = i < len(tasks) and all(
                g["upstream"] in producers
                and g["artifact_field"]
                in old.old._required_artifact_fields_block(producers[g["upstream"]]["prompt"])
                for g in tasks[i]["gated_on"]
            )
            row["matched"] = all(row["checks"].values())
        for field, expected, observed in [
            ("canonical_tasks_sha256", PIN, canonical_hash(tasks)),
            ("ordered_full_contract", True, compared["passed"] and all(r["matched"] for r in rows)),
            ("end_marker", True, "<!-- V771_TASK_CONTRACT_END -->" in text),
        ]:
            checks.append(e.check(path, field, expected, observed, "contract"))
    except (
        OSError,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
        AttributeError,
        yaml.YAMLError,
    ) as error:
        checks.append(
            e.check(
                path,
                "full_task_objects",
                "14 exact ordered objects",
                "missing_file" if isinstance(error, FileNotFoundError) else str(error),
                "contract",
            )
        )
    ready = all(r["passed"] for r in checks)
    for i in range(14):
        if i >= len(rows):
            rows.append(dict(unit_id=f"exp{9027 + i}-missing", matched=False, checks={}))
        row = rows[i]
        matched = ready and row["matched"]
        row.update(
            group_id=row["unit_id"],
            cluster_id="administrative_contract",
            arm="four_source_contract",
            condition="exact_authority",
            metric="ordered_task_contract_match",
            value=int(matched),
            numerator=int(matched),
            denominator=1,
            absolute_metric=int(matched),
            status="completed" if matched else "censored",
            censored=not matched,
            excluded=False,
            exclusion_reason=None,
            effective_independent_groups=0,
        )
    return dict(ready=ready, tasks=tasks, rows=rows, checks=checks, comparison_mode=mode)


def mutations(source: Path, raw: Path) -> list[Json]:
    """All seven original independent refusal controls retain their assertions and bytes."""
    with binding():
        return _mutations(source, raw)


def scope_manifest() -> Json:
    """Freeze the existing direct consumers, commands and exact current code before reduction."""
    with binding():
        manifest = _scope()
    manifest["repository_health_command"] = [str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"]
    manifest["repository_health_deadline_s"] = 180
    return manifest


def freeze(root: Path, raw: Path, run_date: str) -> Json:
    """Snapshot the V770 primaries and historical failed logs before building any claim."""
    with binding():
        work = _freeze(root, raw, run_date)
    history = e.history(raw / "authority")
    work["checks"] += history["checks"]
    work["preserved_v770_outcomes"] = history
    work["archive_lag"].update(
        plan_time_archive_end="2026.11.769",
        scientific_authority="V768 methods; V770 scheduling closure grants no scientific qualification",
    )
    work["current_repository_health"] = []
    if (ROOT / HEALTH).is_file():
        receipt = json.loads((ROOT / HEALTH).read_bytes())
        for source in [ROOT / HEALTH, *[Path(receipt[k]) for k in ["stdout_path", "stderr_path"]]]:
            ref = seal(source, raw)
            work["snapshots"].append(ref)
            work["current_repository_health"].append(ref)
    work["end_ns"] = time.monotonic_ns()
    work.pop("raw_reference")
    atomic_json(raw / "work.json", work)
    return dict(work, raw_reference=reference(raw / "work.json"))


def build(work: Json, validation: Json) -> Json:
    """Current acceptance cannot clear historical failures or imply scientific generalization."""
    with binding():
        value = _build(work, validation)
    value.update(
        experiment_id=9027,
        random_seed=7719027,
        honest_verdict=value["honest_verdict"].replace("v770", "v771"),
        preserved_v770_outcomes=work["preserved_v770_outcomes"],
        repository_health=dict(
            work["repository_health"], current_observations=work["current_repository_health"]
        ),
        methodology_note="Bind fourteen full V771 tasks to current queue bytes and preserve V770 design, queue and four named primaries. Authenticate original V768 fitted states, four development criteria,1024 graph IDs and five-arm online protocol without refitting. Compare one retained ARC ledger against Exp9013. Current scoped validation remains separate from original failed V769 health, failed Exp9014 consumers and blocked Exp9015 access. No model, generation, training, game or benchmark is run.",
    )
    value["field_principles"]["preserved_v770_outcomes"] = (
        "Original failures stay failed; byte custody grants no science."
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """The existing reducer rebuilds all summaries from retained bytes and measured coverage."""
    try:
        value = json.loads(path.read_bytes())
        work = json.loads(Path(value["raw_shard_hashes"][0]["path"]).read_bytes())
        retained = Path(work["raw"]) / "authority"
        archive = retained / "research-complete.yaml"
        milestones = (
            yaml.load(archive.read_bytes(), Loader=yaml.CSafeLoader)["milestones"]
            if archive.is_file()
            else []
        )
        observed = milestones[-1]["id"] if milestones else "missing_file"
        if (
            work["preserved_v770_outcomes"] != e.history(retained)
            or work["archive_lag"]["observed_archive_end"] != observed
        ):
            return dict(passed=False, reason="historical_reduction_drift")
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return dict(passed=False, reason="invalid_or_changed_evidence")
    with binding():
        return _replay(path)


def run_check(
    name: str, argv: list[str], raw: Path, deadline: float = 120, scope: str = "owned"
) -> Json:
    """Retain every frozen direct-consumer node and the original child supervision."""
    with binding():
        return _run_check(name, argv, raw, deadline, scope)


def validate(raw: Path, tracer: Any) -> Json:
    """Combine real parent, unit and CLI statement data with zero exclusions."""
    with binding():
        return _validate(raw, tracer)


def terminal(candidate: Path, raw: Path) -> Json:
    """Unchanged validators and a real foreign-directory child must reject forged aggregates."""
    with binding():
        return _terminal(candidate, raw)
