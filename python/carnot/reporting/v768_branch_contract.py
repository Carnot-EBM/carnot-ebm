"""REQ-REPORT-8985: exact V768 authority and preserved V767 evidence qualify custody only."""

from contextlib import contextmanager
from copy import deepcopy
from collections.abc import Iterator
import json
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import v767_branch_contract as prior
from carnot.reporting import v768_branch_evidence as evidence
from carnot.reporting import v755_branch_runner as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.roadmap_contract import compare_contract, parse_design
from scripts.audit_roadmap_gates import _required_artifact_fields_block
from scripts.roadmap_schema import Roadmap

Json = dict[str, Any]
ROOT, DESIGN, ACTIVE, STAGED = prior.ROOT, prior.DESIGN, prior.ACTIVE, prior.STAGED
NAME, TASK, MILESTONE = (
    "experiment_8985_v768_branch_contract",
    "exp8985-branch-contract",
    "2026.11.768",
)
PIN = "sha256:6912dd5161974a8b3bc9b4775934ad1ad50ec017aa178c338a0617fff3d68a83"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/v768_branch_contract.py",
    "python/carnot/reporting/v768_branch_evidence.py",
    CLI,
]
TEST = "tests/python/test_v768_branch_contract_8985.py"
reference, checksum, seal, encode = prior.reference, prior.checksum, prior.seal, prior.encode
_freeze, _build, _replay, _mutations, _resources = (
    prior.freeze,
    prior.build,
    prior.replay,
    prior.mutations,
    prior.resources,
)
resource_failures = prior.resource_failures
_run_check = runner.run_check


def run_check(
    name: str, argv: list[str], raw: Path, deadline: float = 120, scope: str = "owned"
) -> Json:
    """Private tests need no repository-wide mutation inventory or hardware fixtures."""
    if name in {"owned_tests", "private_e2e018_021"}:
        argv = [
            *argv[: argv.index("pytest") + 1],
            "--noconftest",
            *argv[argv.index("pytest") + 1 :],
        ]
    return _run_check(name, argv, raw, deadline, scope)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase counts so a waiting parent can distinguish progress from a stall."""
    print(f"[exp8985] phase={phase} completed={completed} pending={pending}", flush=True)


def failure(path: Path, check: str, field: str, expected: Any, observed: Any) -> Json:
    """Preserve actual failed operands, including absence, for terminal blocked results."""
    return dict(
        reference(path),
        upstream=TASK,
        check=check,
        artifact_field=field,
        op="==",
        operator="==",
        expected=expected,
        observed=observed,
        passed=False,
    )


def authority(root: Path) -> Json:
    """Bind every full object to an independent digest and each declared earlier gate."""
    tasks, table, rows, failures = [], [], [], []
    mode, path = "unavailable", root / DESIGN
    try:
        text = path.read_text()
        table, tasks = parse_design(text, milestone=MILESTONE)
        active = yaml.load((root / ACTIVE).read_bytes(), Loader=yaml.CSafeLoader)
        staged = (
            yaml.load((root / STAGED).read_bytes(), Loader=yaml.CSafeLoader)
            if (root / STAGED).is_file()
            else None
        )
        selected = active if active["milestone"] == MILESTONE else staged
        if selected is None or selected["milestone"] != MILESTONE:
            raise ValueError("no_current_authority")
        mode = "activated" if active["milestone"] == MILESTONE else "staged"
        queues = [deepcopy(staged or selected), deepcopy(selected)]
        for queue in queues:
            Roadmap.model_validate(queue)
            for actual, expected in zip(queue["tasks"], tasks):
                if actual == dict(expected, prompt=encode(expected["prompt"])):
                    actual["prompt"] = expected["prompt"]
        checked = compare_contract(text, *queues, milestone=MILESTONE, first_id=8985, count=14)
        rows = checked["rows"]
        producers = {t["id"]: t for t in tasks}
        for i, row in enumerate(rows):
            row["checks"]["full_object"] = (
                i < min(len(tasks), *(len(q["tasks"]) for q in queues))
                and tasks[i] == queues[0]["tasks"][i] == queues[1]["tasks"][i]
            )
            row["checks"]["declared_gate_fields"] = i < len(tasks) and all(
                g["upstream"] in producers
                and g["artifact_field"]
                in _required_artifact_fields_block(producers[g["upstream"]]["prompt"])
                for g in tasks[i]["gated_on"]
            )
            row["matched"] = all(row["checks"].values())
        if not (
            checked["passed"]
            and all(r["matched"] for r in rows)
            and canonical_hash(tasks) == PIN
            and "<!-- V768_TASK_CONTRACT_END -->" in text
        ):
            raise ValueError("independent_authority_mismatch")
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError, yaml.YAMLError):
        tasks, table = [], []
        failures.append(
            failure(
                path,
                "complete_authority",
                "full_task_objects",
                "14 exact ordered objects, table, digest and queue",
                "missing_file" if not path.is_file() else "rejected_producer",
            )
        )
    for i in range(14):
        if i >= len(rows):
            rows.append(
                dict(
                    unit_id=f"exp{8985 + i}-missing",
                    order=i + 1,
                    checks=dict(independent_authority=False),
                    matched=False,
                )
            )
        row = rows[i]
        matched = row["matched"] and not failures
        row.update(
            matched=matched,
            value=int(matched),
            numerator=int(matched),
            denominator=1,
            absolute_metric=int(matched),
            cluster_id="administrative_contract",
            group_id=row["unit_id"],
            metric="ordered_task_contract_match",
            condition="exact_authority",
            arm="four_source_contract",
            censor_reason=None if matched else "authority_unavailable_or_rejected",
            exclusion_reason=None,
            effective_independent_groups=0,
            censored=not matched,
            excluded=False,
            seed=None,
            status="completed" if matched else "censored",
            missing_reason=None if matched else "authority_unavailable_or_rejected",
        )
    return dict(
        ready=not failures,
        rows=rows,
        tasks=tasks,
        table=table,
        failures=failures,
        comparison_mode=mode,
    )


def resources(raw: Path, run_date: str) -> Json:
    """Check CPU resources without network, hardware or model access."""
    return dict(
        _resources(raw, run_date),
        run_date_matches=run_date == "20261111",
        network_access_required=False,
        network_check="not_required_local_retained_bytes",
    )


@contextmanager
def binding() -> Iterator[None]:
    """Reuse the qualified runner and custody mechanics with explicit V768 bindings."""
    with patch.multiple(
        prior,
        NAME=NAME,
        TASK=TASK,
        MILESTONE=MILESTONE,
        PIN=PIN,
        CLI=CLI,
        OWNED=OWNED,
        TEST=TEST,
        evidence=evidence,
        progress=progress,
        failure=failure,
        authority=authority,
        resources=resources,
        mutations=mutations,
        build=build,
    ):
        yield


def mutations(source: Path, raw: Path) -> list[Json]:
    """Reject missing, changed and coordinated-rehash authorities in private trees."""
    with binding():
        return _mutations(source, raw)


def freeze(root: Path, raw: Path, run_date: str) -> Json:
    """Seal original bytes before reductions and preserve every rejected operand."""
    with binding():
        work = _freeze(root, raw, run_date)
    python = sys.executable
    pytest = [python, "-m", "pytest", "--noconftest", "-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    plan = [
        [
            python,
            "-m",
            "coverage",
            "run",
            "--rcfile",
            str(raw / "coverage.ini"),
            "--data-file",
            str(raw / ".coverage"),
            "--parallel-mode",
            *pytest[1:],
            TEST,
        ],
        [
            *pytest,
            "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
            "tests/python/test_restricted_decision_audit_8210.py",
        ],
        [str(ROOT / ".venv/bin/ruff"), "check", *OWNED, TEST],
        [str(ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED, TEST],
        [str(ROOT / ".venv/bin/mypy"), "--strict", *OWNED],
        [python, "scripts/check_spec_coverage.py", TEST],
    ]
    path = raw / "owned_validation_argv.json"
    atomic_json(path, plan)
    work["raw_operands"].append(reference(path))
    atomic_json(raw / "work.json", {k: v for k, v in work.items() if k != "raw_reference"})
    work["raw_reference"] = reference(raw / "work.json")
    return work


def build(work: Json, validation: Json) -> Json:
    """Administrative readiness is a null result and cannot authorize scientific claims."""
    with binding():
        value = _build(work, validation)
    plan = json.loads((Path(work["raw"]) / "owned_validation_argv.json").read_bytes())
    commands = [r["argv"] for r in validation["receipts"] if "argv" in r]
    if commands and commands != plan:
        value.update(
            verdict_class="disqualified",
            contract_ready_score=0,
            arc_new_outcomes_ready_score=0,
            required_checks_passed=False,
            flagged_adversarial=True,
        )
    verdict = value["verdict_class"]
    value.update(
        experiment_id=8985,
        random_seed=7688985,
        honest_verdict=f"complete_{verdict}_v768_branch_contract",
        claim_scope="Exact administrative task accounting; no scientific benefit claim.",
        trained_head_specs=[],
        historical_slot_rows=work["history"]["rows"],
        methodology_note="Bind fourteen exact V768 full tasks, independent authority digest and declared earlier gate fields. Preserve V767 design and active queue, five terminal primaries, exact scheduler skips and absent primaries. All three Gibbs development checks failed and no fresh graph science ran; AuthenHallu joined zero of 800 pairs. Read the authenticated ARC frontier once; empty optional evidence produces one cheap skip. No model, head fit, game or device probe is run. Administrative readiness is null and does not gate independent scientific roots.",
        reproduction_commands=[
            f"PYTHONUNBUFFERED=1 .venv/bin/python -u {CLI} --date {work['run_date']}",
            f".venv/bin/python -u {CLI} --cold-replay <primary-path>",
        ],
    )
    value["gate_check_summary"] = [
        dict(r, op=r.get("op", r.get("operator", "==")), passed=False)
        for r in value["gate_check_summary"]
    ]
    value["field_principles"].update(
        historical_slot_rows="Pinned V767 primary and scheduler evidence preserves absence without creating missing results.",
        trained_head_specs="No numerical head is fitted by this administrative reduction.",
        claim_scope="Exact administrative authority is a null result, independent of optional science.",
        history="Preserve V767 original design, active queue, five terminal primaries and nine scheduler blocks.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Use the shared cold verifier to reconstruct authority, evidence and all summaries."""
    with binding():
        return _replay(path)
