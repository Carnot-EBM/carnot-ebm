"""REQ-REPORT-9054-V773: bind current custody while retaining frozen scientific authority."""

from collections.abc import Iterator
from contextlib import contextmanager
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import v770_branch_contract as old
from carnot.reporting import v773_branch_evidence as e
from carnot.reporting.roadmap_contract import compare_contract, parse_design

Json = dict[str, Any]
ROOT, DESIGN, ACTIVE, STAGED = old.ROOT, old.DESIGN, old.ACTIVE, old.STAGED
NAME, TASK, MILESTONE = (
    "experiment_9054_v773_branch_contract",
    "exp9054-branch-contract",
    "2026.11.773",
)
PIN = "sha256:7f419b348f01c64f64ebdf3624f52120990eadf2604b54d778bc73156558a029"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_v773_branch_contract_9054.py"
OWNED = [
    "python/carnot/reporting/v773_branch_contract.py",
    "python/carnot/reporting/v773_branch_evidence.py",
    CLI,
]
DIRECT, E2E, CHECKS = old.DIRECT, old.E2E, old.CHECKS
READINESS = [*old.READINESS, "frozen_methods_ready_score"]
REUSED = [
    *old.REUSED,
    *old.OWNED,
    "python/carnot/reporting/v771_branch_contract.py",
    "python/carnot/reporting/v771_branch_evidence.py",
    "python/carnot/reporting/v772_branch_contract.py",
    "python/carnot/reporting/v772_branch_evidence.py",
]
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
    completed, pending = {
        "preconditions_before": (0, 14),
        "authority_before": (0, 14),
        "measurement_after": (14, 0),
    }.get(phase, (completed, int(phase.startswith("child_")) or pending))
    print(f"[exp9054] phase={phase} completed={completed} pending={pending}", flush=True)


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
            READINESS=READINESS,
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
        compared = compare_contract(
            text, selected, selected, milestone=MILESTONE, first_id=9054, count=14
        )
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
            ("end_marker", True, "<!-- V773_TASK_CONTRACT_END -->" in text),
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
            rows.append(dict(unit_id=f"exp{9054 + i}-missing", matched=False, checks={}))
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
    return manifest


def freeze(root: Path, raw: Path, run_date: str) -> Json:
    """Retain completed V772 evidence and original failed logs before building claims."""
    with binding():
        work = _freeze(root, raw, run_date)
    history = e.history(raw / "authority")
    work["checks"] += history["checks"]
    work["preserved_v772_outcomes"] = history
    work["archive_lag"].update(
        plan_time_archive_end="2026.11.771",
        scientific_authority="V768 fitted states only; V772 closure grants no fresh TEST authority",
    )
    work["current_repository_health"] = []
    work["end_ns"] = time.monotonic_ns()
    work.pop("raw_reference")
    atomic_json(raw / "work.json", work)
    return dict(work, raw_reference=reference(raw / "work.json"))


def history_methods(work: Json) -> bool:
    """Retained provenance and fitted science qualify methods without reusing fresh roles."""
    return bool(
        work["science"]["frozen_graph_authority_ready_score"]
        and work["science"]["online_protocol_ready_score"]
        and work["preserved_v772_outcomes"]["methods_provenance_ready"]
    )


def build(work: Json, validation: Json) -> Json:
    """Current acceptance cannot clear historical failures or imply scientific generalization."""
    with binding():
        value = _build(work, validation)
    value.update(
        experiment_id=9054,
        random_seed=7739054,
        honest_verdict=value["honest_verdict"].replace("v770", "v773"),
        intended_count=14,
        frozen_methods_ready_score=int(value["required_checks_passed"] and history_methods(work)),
        historical_exposure_rows=work["preserved_v772_outcomes"]["exposure_rows"],
        preserved_v772_outcomes=work["preserved_v772_outcomes"],
        repository_health=dict(
            work["repository_health"], current_observations=[validation["global_test_debt"]]
        ),
        frozen_state_references=work["preserved_v772_outcomes"]["frozen_state_references"],
        methodology_note="Bind fourteen full V773 tasks to actual active or staged queue bytes and preserve the original V772 design, queue, independent digest and failed primaries. Authenticate passed Exp9028 provenance, original V768 fitted states, four development criteria and five-arm online protocol without refitting. Preserve Exp9042's 414/1024 roster and 20000-proposal cap, closed Mu-SHROOM identity and participation failures, and separate future MDH authority. Historical graph identities grant no fresh TEST authority. Compare one retained ARC ledger against Exp9041; zero new support grants no scientific benefit. Current scoped validation stays separate from failed historical repository health. No model, generation, training, game or benchmark is run.",
    )
    value["field_principles"].update(
        frozen_methods_ready_score="Original fitted states and protocol only; exposed historical graph identities cannot become fresh TEST evidence.",
        historical_exposure_rows="Every retained A/B unit stays exposed or unknown; roles never certify non-exposure.",
        frozen_state_references="Original trained states and feature hashes only; no new fit or TEST authority.",
    )
    value["field_principles"]["preserved_v772_outcomes"] = (
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
            work["preserved_v772_outcomes"] != e.history(retained)
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
