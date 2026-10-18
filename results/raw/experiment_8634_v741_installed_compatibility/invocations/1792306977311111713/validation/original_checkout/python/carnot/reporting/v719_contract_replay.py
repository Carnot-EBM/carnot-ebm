"""REQ-REPORT-8332: current input custody has no historical-success prerequisite."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import v717_contract_methods as base
from carnot.reporting import v718_contract_replay as legacy
from carnot.reporting import v718_replay_history as history
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
ROOT, DESIGN, ACTIVE, STAGED, PROTOCOL = (
    legacy.ROOT,
    legacy.DESIGN,
    legacy.ACTIVE,
    legacy.STAGED,
    legacy.PROTOCOL,
)
NAME, TASK, MILESTONE = (
    "experiment_8332_v719_contract_replay",
    "exp8332-contract-replay",
    "2026.10.719",
)
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v719_contract_replay_8332.py"
METHODS = "openspec/change-proposals/v719-methods-manifest.json"
OWNED = [
    "python/carnot/reporting/v719_contract_replay.py",
    "python/carnot/reporting/v719_replay_runner.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
failure = base.failure
_authority, _measure, _build = legacy.authority, legacy.measure, legacy.build
_replay, _operands = legacy.replay, legacy.replay_operands


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed boundaries let the supervisor distinguish pending work from stalls."""
    print(f"[exp8332] phase={phase} completed={completed} pending={pending}", flush=True)


def design(root: Path, milestone: str) -> Path:
    """An explicit old milestone always selects its preserved scientific design."""
    paths = {
        "2026.10.716": history.OLD_DESIGN,
        "2026.10.717": history.PRIOR_DESIGN,
        "2026.10.718": "openspec/change-proposals/research-roadmap-v718-preserved-20261009.md",
        MILESTONE: DESIGN,
    }
    return root / str(paths[milestone])


@contextmanager
def bindings() -> Iterator[None]:
    """Small adapters reuse qualified readers without rewriting their algorithms."""
    with ExitStack() as stack:
        for module, name, value in [
            (history, "design", design),
            (legacy, "METHODS", METHODS),
            (legacy, "authority", authority),
            (legacy, "build", build),
            (legacy, "replay_operands", replay_operands),
        ]:
            stack.enter_context(patch.object(module, name, value))
        yield


def authority(root: Path, raw: Path, milestone: str = MILESTONE) -> Json:
    """Full prompts and activated tasks must match before current support is ready."""
    with patch.object(history, "design", design):
        return dict(_authority(root, raw, milestone))


def measure(root: Path, raw: Path) -> Json:
    """Read independent cached operands first; historical failure is retained separately."""
    with bindings():
        work = dict(_measure(root, raw))
    for name in [
        "openspec/change-proposals/research-roadmap-v718-preserved-20261009.md",
        "python/carnot/reporting/v718_contract_replay.py",
        "python/carnot/reporting/v718_replay_history.py",
        "python/carnot/reporting/v718_replay_runner.py",
        "python/carnot/reporting/v717_contract_methods.py",
        "python/carnot/reporting/v717_contract_runner.py",
        "python/carnot/reporting/v709_execution.py",
    ]:
        work["refs"].append(legacy.snapshot(root / name, raw / "dependencies", Path(name).stem))
    health = root / "results/raw" / NAME / "global_health.json"
    work["global_health"] = base.bind(health, raw, work["refs"], []) if health.is_file() else {}
    prior: list[Json] = []
    for name in ["experiment_8318_v718_contract_replay", "experiment_8331_v718_capstone"]:
        problems: list[Json] = []
        value = legacy.authenticate(
            root / "results" / (name + ".json"), raw, work["refs"], problems
        )
        prior.append(
            dict(
                path=str(root / "results" / (name + ".json")),
                verdict_class=value.get("verdict_class"),
                honest_verdict=value.get("honest_verdict"),
                authenticated=bool(value),
                disposition="authenticated_failure"
                if value.get("verdict_class") == "disqualified"
                else "external_operand",
                counts={
                    key: value.get(key)
                    for key in [
                        "actual_executed_task_count",
                        "pre_gate_count",
                        "missing_output_count",
                    ]
                },
                authentication_failures=problems,
            )
        )
    work["prior_dispositions"] = prior
    work["support"]["current_receipt"] = dict(
        status="CURRENT",
        canonical_tasks_sha256=work["contract"]["canonical_tasks_sha256"],
        protocol_sha256=work["protocol_sha256"],
        source_artifact_hashes_sha256=canonical_hash(work["refs"]),
        counts_sha256=canonical_hash(work["support"]["counts"]),
    )
    progress("current_support_bound", 1, 0)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Administrative readiness and historical reader qualification are separate claims."""
    value = dict(_build(work, receipts, raw, output))
    owned = bool(receipts) and all(r["passed"] for r in receipts)
    contract = work["contract"]
    current = (
        owned
        and contract["activated"]
        and len(contract["contract_rows"]) == 14
        and all(r["matched"] for r in contract["contract_rows"])
    )
    cached = current and work["support"]["ready"] and work["protocol_sha256"] == base.PIN
    historical = owned and work["history"].get("deterministic", False)
    value.update(
        experiment_id=8332,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261009",
        random_seed=7198332,
        required_checks_passed=owned,
        current_contract_ready_score=int(current),
        cached_support_ready_score=int(cached),
        history_reader_ready_score=int(historical),
        cached_support=work["support"],
    )
    kind = (
        "disqualified"
        if not owned and not work["failures"] or any(not r["passed"] for r in receipts)
        else "blocked"
        if work["failures"]
        else "circular_positive"
    )
    value.update(
        verdict_class=kind,
        honest_verdict="complete_"
        + kind
        + "_"
        + (work["failures"][0]["artifact_field"] if kind == "blocked" else "contract_replay"),
    )
    invocation = list(value["invocation_argv"])
    if "--date" in invocation:
        invocation[invocation.index("--date") + 1] = "20261009"
    value["invocation_argv"] = invocation
    value["acceptance_gates"].update(
        owned_checks=owned,
        current_authority=current,
        cached_support=cached,
        history_reader=historical,
    )
    value["historical_dispositions"] = dict(
        value["historical_dispositions"], v718_primaries=work.get("prior_dispositions", [])
    )
    value["field_principles"]["cached_support"] = (
        "CURRENT support binds authentic shard counts to exact activated tasks and immutable science; historical failure is separate."
    )
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    return value


def replay_operands(work: Json) -> bool:
    """Recount from snapshots and reject a changed bound input list before reduction."""
    receipt = work["support"]["current_receipt"]
    if receipt != dict(
        status="CURRENT",
        canonical_tasks_sha256=work["contract"]["canonical_tasks_sha256"],
        protocol_sha256=work["protocol_sha256"],
        source_artifact_hashes_sha256=canonical_hash(work["refs"]),
        counts_sha256=canonical_hash(work["support"]["counts"]),
    ):
        return False
    for prior in work["prior_dispositions"]:
        ref = next(r for r in work["refs"] if r["path"] == prior["path"])
        source = (
            json.loads(Path(ref["snapshot_path"]).read_bytes()) if prior["authenticated"] else {}
        )
        expected = dict(
            path=prior["path"],
            verdict_class=source.get("verdict_class"),
            honest_verdict=source.get("honest_verdict"),
            authenticated=bool(source),
            disposition="authenticated_failure"
            if source.get("verdict_class") == "disqualified"
            else "external_operand",
            counts={
                key: source.get(key)
                for key in ["actual_executed_task_count", "pre_gate_count", "missing_output_count"]
            },
            authentication_failures=prior["authentication_failures"],
        )
        if prior != expected:
            return False
    with bindings():
        return bool(_operands(work))


def replay(path: Path) -> bool:
    """The existing cold reader compares every derived field and bound byte operand."""
    with bindings():
        return bool(_replay(path))
