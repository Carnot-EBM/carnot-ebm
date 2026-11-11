"""REQ-REPORT-8999: qualify administrative custody independently of optional science."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import yaml

from carnot.reporting import v768_branch_contract as prior
from carnot.reporting import v769_branch_evidence as e
from carnot.reporting import v755_branch_runner as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.roadmap_contract import compare_contract, parse_design
from carnot.reporting.v756_branch_qualification import verify_measured
from scripts.audit_roadmap_gates import _required_artifact_fields_block
from scripts.roadmap_schema import Roadmap

Json = dict[str, Any]
ROOT, DESIGN, ACTIVE, STAGED = prior.ROOT, prior.DESIGN, prior.ACTIVE, prior.STAGED
NAME, TASK, MILESTONE = (
    "experiment_8999_v769_branch_contract",
    "exp8999-branch-contract",
    "2026.11.769",
)
PIN = "sha256:40f38027edb61619e46da0480daec1dfcdb560cc946c3e53f5e2b986cb2e7a20"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_v769_branch_contract_8999.py"
OWNED = [
    "python/carnot/reporting/v769_branch_contract.py",
    "python/carnot/reporting/v769_branch_evidence.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
reference, checksum, seal = prior.reference, prior.checksum, prior.seal
_run_check = runner.run_check


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counts so a parent can distinguish progress from a stalled process."""
    print(f"[exp8999] phase={phase} completed={completed} pending={pending}", flush=True)


def run_check(
    name: str, argv: list[str], raw: Path, deadline: float = 120, scope: str = "owned"
) -> Json:
    """Keep private validation independent of the repository-wide artifact guard."""
    if "pytest" in argv:
        i = argv.index("pytest") + 1
        argv = [*argv[:i], "--noconftest", *argv[i:]]
    progress("subprocess_before_" + name, 0, 1)
    result = _run_check(name, argv, raw, deadline, scope)
    progress("subprocess_after_" + name, 1, 0)
    return result


def authority(root: Path) -> Json:
    """Compare full objects as well as visible cells; a digest defeats coordinated edits."""
    tasks: list[Json] = []
    rows: list[Json] = []
    checks: list[Json] = []
    mode = "unavailable"
    path = root / DESIGN
    try:
        text = path.read_text()
        table, tasks = parse_design(text, milestone=MILESTONE)
        active = yaml.safe_load((root / ACTIVE).read_bytes())
        staged = yaml.safe_load((root / STAGED).read_bytes()) if (root / STAGED).is_file() else None
        mode = "activated" if active["milestone"] == MILESTONE else "staged"
        selected = active if mode == "activated" else staged
        queues = [deepcopy(staged or selected), deepcopy(selected)]
        for queue in queues:
            Roadmap.model_validate(queue)
            for actual, expected in zip(queue["tasks"], tasks):
                if actual == dict(expected, prompt=prior.encode(expected["prompt"])):
                    actual["prompt"] = expected["prompt"]
        compared = compare_contract(text, *queues, milestone=MILESTONE, first_id=8999)
        rows = compared["rows"]
        producers = {t["id"]: t for t in tasks}
        for i, row in enumerate(rows):
            row["checks"]["full_object"] = (
                i < min(len(tasks), *(len(q["tasks"]) for q in queues))
                and tasks[i] == queues[0]["tasks"][i] == queues[1]["tasks"][i]
            )
            row["checks"]["declared_gate_fields"] = i < len(tasks) and all(
                g["artifact_field"]
                in _required_artifact_fields_block(producers[g["upstream"]]["prompt"])
                for g in tasks[i]["gated_on"]
            )
            row["matched"] = all(row["checks"].values())
        for field, expected, observed in [
            ("canonical_tasks_sha256", PIN, canonical_hash(tasks)),
            ("ordered_full_contract", True, compared["passed"] and all(r["matched"] for r in rows)),
            ("end_marker", True, "<!-- V769_TASK_CONTRACT_END -->" in text),
        ]:
            checks.append(e.check(path, field, expected, observed, "contract"))
    except (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError) as error:
        checks.append(
            e.check(
                path,
                "full_task_objects",
                "14 exact ordered objects",
                "missing_file" if isinstance(error, FileNotFoundError) else str(error),
                "contract",
            )
        )
    ready = bool(checks) and all(r["passed"] for r in checks)
    for i in range(14):
        row = (
            rows[i]
            if i < len(rows)
            else dict(unit_id=f"exp{8999 + i}-missing", checks={}, matched=False)
        )
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
            exclusion_reason=None,
            censored=not matched,
            excluded=False,
            effective_independent_groups=0,
        )
        if i >= len(rows):
            rows.append(row)
    return dict(ready=ready, tasks=tasks, rows=rows, checks=checks, comparison_mode=mode)


def mutations(source: Path, raw: Path) -> list[Json]:
    """Reuse existing independent mutations without changing historical implementations."""
    from unittest.mock import patch

    with patch.multiple(
        prior.prior,
        DESIGN=DESIGN,
        ACTIVE=ACTIVE,
        STAGED=STAGED,
        MILESTONE=MILESTONE,
        PIN=PIN,
        authority=authority,
        progress=progress,
    ):
        return prior._mutations(source, raw)


def freeze(root: Path, raw: Path, run_date: str) -> Json:
    """Snapshot bytes before reductions, including a single observation of the live ledger."""
    progress("preconditions_before", 0, 14)
    if (raw / "work.json").exists():
        raise FileExistsError("immutable_invocation_already_exists")
    raw.mkdir(parents=True, exist_ok=True)
    start = time.monotonic_ns()
    preconditions = prior.resources(raw, run_date)
    progress("preconditions_after", 1, 0)
    snapshots: list[Json] = []
    names = [DESIGN, ACTIVE, STAGED, *e.PINS, e.LEDGER]
    names += [str(Path(r["path"]).relative_to(ROOT)) for r in e.operands(root)]
    for i, name in enumerate(dict.fromkeys(names)):
        source, target = root / name, raw / "authority" / name
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            snapshots.append(dict(reference(target), source=reference(source)))
        progress("snapshot", i + 1, max(0, len(set(names)) - i - 1))
    retained = raw / "authority"
    progress("authority_before", 0, 14)
    contract = authority(retained)
    progress("authority_after", sum(r["matched"] for r in contract["rows"]), 0)
    progress("frozen_science_before", 0, 2)
    science = e.science(retained)
    progress("frozen_science_after", 2, 0)
    progress("arc_frontier_before", 0, 1)
    arc = e.frontier(retained)
    progress("arc_frontier_after", 1, 0)
    checks = [
        e.check(
            retained / n,
            "sha256",
            pin,
            reference(retained / n)["sha256"] or "missing_file",
            "history",
        )
        for n, pin in e.PINS.items()
        if n not in [e.GRAPH, e.ONLINE, e.PROTOCOL, e.ONLINE_PROTOCOL]
    ]
    controls = mutations(retained, raw / "controls") if contract["ready"] else []
    code = [
        dict(seal(ROOT / p, raw), executed_source=True)
        for p in [
            *OWNED,
            TEST,
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/reporting/roadmap_contract.py",
            "python/carnot/reporting/v755_branch_runner.py",
            "scripts/experiment_template.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "ops/exclusion_manifest.yaml",
        ]
    ]
    work = dict(
        contract=contract,
        science=science,
        arc=arc,
        checks=checks,
        snapshots=snapshots,
        code=code,
        controls=controls,
        raw=str(raw.absolute()),
        run_date=run_date,
        start_ns=start,
        end_ns=time.monotonic_ns(),
        preconditions=preconditions,
    )
    atomic_json(raw / "work.json", work)
    progress("measurement_after", 14, 0)
    return dict(work, raw_reference=reference(raw / "work.json"))


def build(work: Json, validation: Json) -> Json:
    """Readiness describes measured custody; it grants no scientific benefit claim."""
    receipts = validation["receipts"]
    checked = (
        bool(receipts)
        and all(
            r["passed"] and r.get("normal_exit", True) and not r.get("timed_out", False)
            for r in receipts
        )
        and validation["coverage"]["complete"]
    )
    resource_failures = [
        dict(r, op=r["operator"], passed=False)
        for r in prior.resource_failures(work["preconditions"], Path(work["raw"]))
    ]
    gates = work["contract"]["checks"] + work["checks"] + resource_failures
    ready = bool(
        checked
        and work["contract"]["ready"]
        and all(r["passed"] for r in gates)
        and len(work["controls"]) == 7
        and all(r["passed"] for r in work["controls"])
    )
    verdict = "disqualified" if not checked else "null" if ready else "blocked"
    failed = next((r["check"] for r in gates if not r["passed"]), "contract_qualification")
    raw = Path(work["raw"])
    rows = work["contract"]["rows"]
    end = max([work["end_ns"], *[r.get("ended_monotonic_ns", work["end_ns"]) for r in receipts]])
    value: Json = dict(
        experiment_id=8999,
        task_id=TASK,
        milestone=MILESTONE,
        run_date=work["run_date"],
        honest_verdict=f"complete_{verdict}_"
        + (failed if verdict == "blocked" else "v769_branch_contract"),
        verdict_class=verdict,
        verifier_is_oracle=False,
        claim_scope="Administrative custody only; no scientific benefit or independent generalization claim.",
        exposure_scope="retained_exposed_graph_development_and_private_custody_controls",
        independent_generalization_score=0,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        trained_head_specs=[],
        model_invocation_counts=dict(model_loads=0, generation_calls=0, llm=0, numerical_head=0),
        numerical_training_calls=0,
        required_checks_passed=checked,
        flagged_adversarial=not checked,
        validation_receipts=receipts,
        owned_statement_coverage=validation["coverage"],
        global_test_debt=validation["global_test_debt"],
        preconditions_checked=work["preconditions"],
        gate_check_summary=gates + work["science"]["checks"] + work["arc"]["checks"],
        rows=rows,
        full_contract_rows=rows,
        full_task_objects=work["contract"]["tasks"],
        intended_count=14,
        independent_count=0,
        completed_count=sum(r["status"] == "completed" for r in rows),
        excluded_count=0,
        censored_count=sum(r["status"] == "censored" for r in rows),
        failed_count=0,
        contract_ready_score=int(ready),
        frozen_graph_authority_ready_score=int(
            checked and work["science"]["frozen_graph_authority_ready_score"]
        ),
        online_protocol_ready_score=int(checked and work["science"]["online_protocol_ready_score"]),
        arc_new_outcomes_ready_score=int(checked and work["arc"]["ready"]),
        authority_manifest=work["snapshots"],
        authority_comparison_mode=work["contract"]["comparison_mode"],
        authority_mutation_rows=work["controls"],
        frozen_science=work["science"],
        frontier_delta_rows=work["arc"]["new_event_rows"],
        arc_frontier=work["arc"],
        canonical_tasks_sha256=PIN,
        full_task_digest=canonical_hash(work["contract"]["tasks"]),
        duration_s=(end - work["start_ns"]) / 1e9,
        random_seed=7698999,
        phase_spans=[
            dict(
                name="authentication_and_reduction",
                start_ns=work["start_ns"],
                end_ns=work["end_ns"],
                duration_s=(work["end_ns"] - work["start_ns"]) / 1e9,
            )
        ],
        source_artifact_hashes=[r["source"] for r in work["snapshots"]],
        raw_shard_hashes=[work["raw_reference"], *work["snapshots"]],
        code_config_hashes=work["code"],
        cited_upstream_artifacts=[
            dict(
                r,
                experiment_id=int(Path(n).name.split("_")[1]),
                fields_imported=["original_authority_and_frozen_scientific_contract"],
            )
            for n in e.PINS
            if n.startswith("results/")
            for r in [reference(raw / "authority" / n)]
        ],
        acceptance_gates=dict(
            owned_validation=checked,
            exact_contract=ready,
            frozen_graph_authority=bool(work["science"]["frozen_graph_authority_ready_score"]),
            online_protocol=bool(work["science"]["online_protocol_ready_score"]),
        ),
        methodology_note="Bind fourteen full V769 task objects and exact earlier gate fields. Preserve original V768 design, queue, qualified methods and rejected Exp8990 bytes. Authenticate all four passed exposed-development criteria, fitted states, 1024 graph identities and original role counts without fitting, redrawing or rerunning admission. Read a single saved live ARC frontier against Exp8985; require thirty new uncensored episodes, three families and both arms. Administrative readiness is independent of optional scientific gates. No model, game, training or device benchmark is run.",
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "field_principles", "reproducibility_checksum"],
        "Immutable original bytes and measured normal-exit checks bind custody only; missing external operands are terminal blocked, failed owned validation disqualifies readiness, and exposed oracle evidence establishes no independent benefit.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Reconstruct from retained operands so updating an aggregate hash cannot invent readiness."""
    try:
        value = json.loads(path.read_bytes())
        if not value["required_checks_passed"] or not verify_measured(
            value["owned_statement_coverage"], OWNED
        ):
            raise ValueError("unqualified_statements")
        for ref in value["raw_shard_hashes"] + value["code_config_hashes"]:
            if reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("operand_drift")
            if (
                ref.get("executed_source")
                and reference(Path(ref["source"]["path"])) != ref["source"]
            ):
                raise ValueError("executed_code_drift")
        ref = value["raw_shard_hashes"][0]
        work = json.loads(Path(ref["path"]).read_bytes())
        raw = Path(work["raw"])
        retained = raw / "authority"
        if (
            work["contract"] != authority(retained)
            or work["science"] != e.science(retained)
            or work["arc"] != e.frontier(retained)
        ):
            raise ValueError("authority_or_science_drift")
        for control in work["controls"]:
            if not control["passed"] or authority(raw / "controls" / control["mutation"])["ready"]:
                raise ValueError("mutation_control_drift")
        validation_path = raw / "validation.json"
        validation = (
            json.loads(validation_path.read_bytes())
            if validation_path.is_file()
            else dict(
                receipts=value["validation_receipts"],
                coverage=value["owned_statement_coverage"],
                global_test_debt=value["global_test_debt"],
            )
        )
        expected = build(dict(work, raw_reference=ref), validation)
        if validation_path.is_file():
            expected["raw_shard_hashes"].append(reference(validation_path))
        expected["reproducibility_checksum"] = checksum(expected)
        return dict(
            passed=value == expected and checksum(value) == value["reproducibility_checksum"],
            reason="primitive_replay",
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return dict(passed=False, reason="invalid_or_changed_evidence")
