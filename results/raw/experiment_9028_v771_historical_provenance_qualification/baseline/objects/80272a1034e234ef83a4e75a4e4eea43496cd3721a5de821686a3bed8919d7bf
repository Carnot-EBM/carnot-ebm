"""REQ-REPORT-8971: exact administrative authority is independent of optional science."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import yaml

from carnot.reporting import v755_branch_contract as base
from carnot.reporting import v766_branch_contract as prior
from carnot.reporting import v767_branch_evidence as evidence
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.roadmap_contract import compare_contract, parse_design
from carnot.reporting.v756_branch_qualification import verify_measured
from scripts.audit_roadmap_gates import _required_artifact_fields_block
from scripts import exclusion_manifest_lint as exclusions
from scripts.failure_ledger import validate_prior_failures
from scripts.roadmap_schema import Roadmap

Json = dict[str, Any]
ROOT, DESIGN, ACTIVE, STAGED = prior.ROOT, prior.DESIGN, prior.ACTIVE, prior.STAGED
NAME, TASK, MILESTONE = (
    "experiment_8971_v767_branch_contract",
    "exp8971-branch-contract",
    "2026.11.767",
)
PIN = "sha256:e77500d679cfe6488fecb433bda7838a7f2b4ac20ea9f50d399bb408fb995c18"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/v767_branch_contract.py",
    "python/carnot/reporting/v767_branch_evidence.py",
    CLI,
]
TEST = "tests/python/test_v767_branch_contract_8971.py"
reference, checksum, seal, encode = prior.reference, prior.checksum, prior.seal, prior.encode


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual phase counts so a parent can distinguish progress from a stalled child."""
    print(f"[exp8971] phase={phase} completed={completed} pending={pending}", flush=True)


def failure(path: Path, check: str, field: str, expected: Any, observed: Any) -> Json:
    """Keep exact failed fields and hashes so missing bytes cannot masquerade as zero."""
    return dict(
        **reference(path),
        upstream=TASK,
        check=check,
        artifact_field=field,
        operator="==",
        expected=expected,
        observed=observed,
    )


def authority(root: Path) -> Json:
    """An independent digest prevents coordinated edits to all authority copies."""
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
        checked = compare_contract(text, *queues, milestone=MILESTONE, first_id=8971, count=14)
        rows = checked["rows"]
        for i, row in enumerate(rows):
            row["checks"]["full_object"] = (
                i < min(len(tasks), *(len(q["tasks"]) for q in queues))
                and tasks[i] == queues[0]["tasks"][i] == queues[1]["tasks"][i]
            )
            row["matched"] = all(row["checks"].values())
        if not (
            checked["passed"]
            and all(r["matched"] for r in rows)
            and canonical_hash(tasks) == PIN
            and "<!-- V767_TASK_CONTRACT_END -->" in text
        ):
            raise ValueError("independent_authority_mismatch")
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError, yaml.YAMLError):
        tasks, table = [], []
        failures.append(
            failure(
                path,
                "complete_authority",
                "full_task_objects",
                "14 exact ordered full objects, table, independent digest and queue",
                "missing_file" if not path.is_file() else "rejected_producer",
            )
        )
    for i in range(14):
        if i >= len(rows):
            rows.append(
                dict(
                    unit_id=f"exp{8971 + i}-missing",
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
            metric="ordered_task_contract_match",
            condition="exact_authority",
            arm="four_source_contract",
            censor_reason=None if matched else "authority_unavailable_or_rejected",
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


def mutations(source: Path, raw: Path) -> list[Json]:
    """Independent pins reject even an attacker who updates every mutable authority copy."""
    text = (source / DESIGN).read_text()
    queue = yaml.load(
        (source / (STAGED if (source / STAGED).is_file() else ACTIVE)).read_bytes(),
        Loader=yaml.CSafeLoader,
    )
    rows = []
    for mode in ["missing", "title", "order", "prompt", "gate", "path", "rehash"]:
        root, changed, design = raw / mode, deepcopy(queue), text
        if mode == "missing":
            changed["tasks"].pop()
        if mode == "order":
            changed["tasks"].reverse()
        if mode in {"title", "rehash"}:
            changed["tasks"][0]["title"] += " changed"
        if mode == "prompt":
            changed["tasks"][0]["prompt"] += " changed"
        if mode == "gate":
            next(t for t in changed["tasks"] if t["gated_on"])["gated_on"][0]["artifact_field"] = (
                "forged"
            )
        if mode == "path":
            changed["tasks"][0]["deliverable"] = "results/forged.json"
        if mode == "rehash":
            _, original = parse_design(text, milestone=MILESTONE)
            design = text.replace(original[0]["title"], changed["tasks"][0]["title"])
            design = design.replace(
                PIN.removeprefix("sha256:"),
                canonical_hash(changed["tasks"]).removeprefix("sha256:"),
            )
        path = root / DESIGN
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(design)
        (root / ACTIVE).write_text(yaml.dump(changed, Dumper=yaml.CSafeDumper))
        rows.append(
            dict(
                mutation=mode,
                passed=not authority(root)["ready"],
                references=[reference(path), reference(root / ACTIVE)],
            )
        )
        progress("control_" + mode, len(rows), 7 - len(rows))
    return rows


def resources(raw: Path, run_date: str) -> Json:
    """Check only CPU resources this reader needs; optional hardware cannot block it."""
    return dict(
        producer_argv=sys.orig_argv,
        repository_virtualenv=Path(sys.prefix).resolve() == (ROOT / ".venv").resolve(),
        free_bytes=shutil.disk_usage(raw).free,
        writable_disk=os.access(raw, os.W_OK),
        available_memory_bytes=os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"),
        dependencies_available=all(
            __import__(name) is not None for name in ["yaml", "coverage", "pytest"]
        ),
        run_date_matches=run_date == "20261110",
        no_model_loads=True,
        no_game_runs=True,
        no_device_probes=True,
        activation_authorized=False,
    )


def resource_failures(preconditions: Json, raw: Path) -> list[Json]:
    """A failed precondition names its exact value rather than claiming successful work."""
    expected = dict(
        repository_virtualenv=True,
        writable_disk=True,
        dependencies_available=True,
        run_date_matches=True,
        free_bytes=536870912,
        available_memory_bytes=268435456,
    )
    return [
        dict(
            failure(raw, "required_resource", field, limit, preconditions[field]),
            operator=">=" if field.endswith("bytes") else "==",
        )
        for field, limit in expected.items()
        if (
            preconditions[field] < limit
            if field.endswith("bytes")
            else preconditions[field] != limit
        )
    ]


def scope_rows(tasks: list[Json]) -> list[Json]:
    """Reuse manifest rules and prior validation without rescanning every historical primary."""
    retired, patterns = exclusions._load_manifest_exp_ids(), exclusions._load_blocked_patterns()
    rows = []
    for task in tasks:
        prompt = task["prompt"]
        matched = exclusions._blocked_pattern_matches(task["title"] + " " + prompt, patterns)
        wrong = [
            path.pattern
            for board, path, _, _ in exclusions._WRONG_MECHANISM_PATTERNS
            if board.search(prompt)
            and (hit := path.search(prompt))
            and not exclusions._is_negated_context(prompt, hit.start())
        ]
        checks = dict(
            id_not_retired=exclusions._extract_exp_id(task["id"]) not in retired,
            no_retired_requires=not set(exclusions._extract_requires_ids(task)).intersection(
                retired
            ),
            complete_prior_failures=validate_prior_failures(task).valid,
            no_wrong_mechanism=not wrong,
        )
        rows.append(
            dict(
                task_id=task["id"],
                passed=all(checks.values()),
                checks=checks,
                blocked_pattern_matches=matched,
                wrong_mechanism_matches=wrong,
                prior_failures=task["prior_failures"],
            )
        )
    return rows


def freeze(root: Path, raw: Path, run_date: str) -> Json:
    """Seal original bytes before checking them so replay never inherits mutable authority."""
    progress("start", 0, 14)
    if (raw / "work.json").exists():
        raise FileExistsError("immutable_invocation_already_exists")
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    preconditions = resources(raw, run_date)
    snapshots, failures = [], resource_failures(preconditions, raw)
    for name in [DESIGN, ACTIVE, STAGED, *evidence.PINS]:
        source_root = root if name in [DESIGN, ACTIVE, STAGED] else ROOT
        path, target = source_root / name, raw / "authority" / name
        if path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            snapshots.append(dict(source=reference(path), **reference(target)))
        elif name != STAGED and name not in evidence.PINS:
            failures.append(failure(path, "required_input", name, "original bytes", "missing_file"))
    progress("authority_before", 0, 14)
    contract = authority(raw / "authority")
    historical = evidence.history(raw / "authority")
    progress("history_after", len(historical["rows"]), 14 - len(historical["rows"]))
    arc = evidence.arc.arc_evidence()
    for ref in arc["references"]:
        snapshots.append(seal(Path(ref["path"]), raw))
    arc["cheap_skip_count"] = int(not arc["ready"])
    arc["panel_rerun_count"] = 0
    controls = mutations(raw / "authority", raw / "controls") if contract["ready"] else []
    scopes = scope_rows(contract["tasks"])
    code = [
        dict(seal(ROOT / name, raw), executed_source=True)
        for name in [
            *OWNED,
            TEST,
            "python/carnot/reporting/current_work_receipt.py",
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/reporting/roadmap_contract.py",
            "python/carnot/reporting/v755_branch_runner.py",
            "python/carnot/reporting/v755_branch_contract.py",
            "python/carnot/reporting/v766_branch_contract.py",
            "python/carnot/reporting/v766_branch_evidence.py",
            "python/carnot/reporting/arc_supervisor_frontier_8720.py",
            "scripts/exclusion_manifest_lint.py",
            "scripts/experiment_template.py",
            "ops/exclusion_manifest.yaml",
        ]
    ]
    work = dict(
        contract=contract,
        history=historical,
        arc=arc,
        snapshots=snapshots + code,
        input_failures=failures,
        authority_mutation_rows=controls,
        exclusion_scope_rows=scopes,
        run_date=run_date,
        owner_pid=os.getpid(),
        start_ns=start,
        end_ns=time.monotonic_ns(),
        raw=str(raw.absolute()),
        preconditions=preconditions,
        raw_operands=[reference(p) for p in sorted((raw / "controls").rglob("*")) if p.is_file()],
    )
    atomic_json(raw / "work.json", work)
    progress(
        "freeze_after",
        sum(r["matched"] for r in contract["rows"]),
        14 - sum(r["matched"] for r in contract["rows"]),
    )
    return dict(work, raw_reference=reference(raw / "work.json"))


def build(work: Json, validation: Json) -> Json:
    """Administrative qualification earns a null result, never a scientific benefit claim."""
    end = max(
        [
            work["end_ns"],
            *[r.get("ended_monotonic_ns", work["end_ns"]) for r in validation["receipts"]],
        ]
    )
    prepared = dict(
        work,
        end_ns=end,
        manifests=[],
        methods=[],
        history=dict(work["history"], original_child_qualified=True),
        branches=dict(runtime=dict(ready=False), arc=dict(ready=False), failures=[]),
    )
    value = base.build(prepared, validation)
    tasks, controls = work["contract"]["tasks"], work["authority_mutation_rows"]
    producers = {t["id"]: t for t in tasks}
    gates = [
        dict(
            edge,
            future_dependency=True,
            input_file_claimed_present=False,
            required_field_declared=edge["artifact_field"]
            in _required_artifact_fields_block(producers[edge["upstream"]]["prompt"]),
        )
        for edge in base.edges(tasks)
    ]
    independent = all(
        not t.get("requires")
        and all(g["artifact_field"] != "contract_ready_score" for g in t["gated_on"])
        for t in tasks
    )
    checked = bool(
        validation["receipts"]
        and value["required_checks_passed"]
        and not any(r.get("timed_out", False) for r in validation["receipts"])
        and all(r["passed"] for r in controls)
        and value["duration_s"] < 4800
    )
    failures = (
        work["contract"]["failures"]
        + work["input_failures"]
        + work["history"]["failures"]
        + resource_failures(work["preconditions"], Path(work["raw"]))
    )
    ready = bool(
        checked
        and not failures
        and work["contract"]["ready"]
        and len(controls) == 7
        and independent
        and all(r["required_field_declared"] for r in gates)
        and len(work["exclusion_scope_rows"]) == 14
        and all(r["passed"] for r in work["exclusion_scope_rows"])
    )
    if not ready and checked and not failures:
        failures.append(
            failure(
                Path(work["raw"]) / "work.json",
                "contract_qualification",
                "contract_ready_score",
                1,
                0,
            )
        )
    verdict = "disqualified" if not checked else "null" if ready else "blocked"
    for field in [
        "preserved_v754_reference",
        "runtime_repair_ready_score",
        "arc_new_outcomes_available_score",
        "branch_admission_rows",
    ]:
        value.pop(field)
    value.update(
        experiment_id=8971,
        task_id=TASK,
        milestone=MILESTONE,
        random_seed=7678971,
        honest_verdict=f"complete_{verdict}_v767_branch_contract",
        verdict_class=verdict,
        contract_ready_score=int(ready),
        arc_new_outcomes_ready_score=int(checked and work["arc"]["ready"]),
        required_checks_passed=checked,
        flagged_adversarial=not checked,
        model_specs=[],
        execution_venue="host",
        intended_count=14,
        censored_count=14 - value["completed_count"],
        preconditions_checked=work["preconditions"],
        full_task_digest=canonical_hash(tasks),
        canonical_tasks_sha256=PIN,
        prior_scope_rows=[dict(task_id=t["id"], **p) for t in tasks for p in t["prior_failures"]],
        gate_edge_rows=gates,
        authority_mutation_rows=controls,
        exclusion_scope_rows=work["exclusion_scope_rows"],
        history=work["history"],
        authority_comparison_mode=work["contract"]["comparison_mode"],
        arc_frontier=work["arc"],
        arc_support_rows=work["arc"]["support"]["support_rows"],
        gate_check_summary=failures,
        optional_evidence_diagnostics=[
            dict(r, operator=r.get("operator", "==")) for r in work["arc"]["failures"]
        ],
        historical_evidence_diagnostics=work["history"]["failures"],
        code_config_hashes=[s for s in work["snapshots"] if s.get("executed_source")],
        cited_upstream_artifacts=work["history"]["references"],
        acceptance_gates=dict(
            exact_contract=work["contract"]["ready"],
            authenticated_history=work["history"]["authenticated"],
            owned_validation=checked,
            required_resources=not resource_failures(work["preconditions"], Path(work["raw"])),
            independent_science_roots=independent,
            exact_gate_fields=all(r["required_field_declared"] for r in gates),
            exclusion_scopes=all(r["passed"] for r in work["exclusion_scope_rows"]),
            mutation_controls=len(controls) == 7 and all(r["passed"] for r in controls),
        ),
        methodology_note="Bind fourteen exact tasks and declared future gate fields. Authenticate preserved V766 originals and terminal scheduler receipts without inferring success. Read the ARC receipt delta once; unchanged evidence records one cheap skip. No scientific panel, game, model or numerical head is run. Historical graph labels are exposed development evidence, and cheap BFS truth cannot establish the no-cheap-oracle north star.",
        reproduction_commands=[
            f"JAX_PLATFORMS=cpu PYTHONPATH=python:. .venv/bin/python -u {CLI} --date {work['run_date']}",
            f".venv/bin/python -u {CLI} --cold-replay <primary-path>",
        ],
    )
    value["raw_shard_hashes"] += work["raw_operands"]
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Exact immutable bytes, task identity, real clocks and measured owned checks bind administrative readiness. Optional scientific support never establishes benefit or blocks independent authority. Historical calls remain citations; current model and numerical training calls are zero.",
    )
    value["field_principles"].update(
        contract_ready_score="One only after exact ordered authority, declared gate fields, resources, exclusion scopes and owned validation qualify.",
        arc_new_outcomes_ready_score="Independent optional support: at least 30 newly authenticated uncensored episodes across three families with both arms in every held family.",
        rows="One status and contract-match metric per intended task; administrative rows are not independent scientific questions.",
        gate_check_summary="Required failures retain upstream, check, path, byte hash, exact field, operator, expected and observed; absence differs from zero.",
        model_invocation_counts="Only current process model loads and generations are counted; all are zero. Historical Qwen provenance remains in cited originals.",
        numerical_training_calls="Current numerical head fits are counted separately from LLM calls; this reader performs zero fits.",
        history="Every V766 slot retains its original primary or actual terminal scheduler outcome; missing Exp8970 is not a successful capstone.",
        independent_generalization_score="Zero: exposed graph development labels and cheap BFS truth cannot establish the no-cheap-oracle claim.",
        owned_statement_coverage="Unit and real child SQLite measurements cover every newly owned statement with no exclusions; a claimed JSON percentage is insufficient.",
    )
    value["acceptance_gate_principles"] = {
        k: "Require exact authority, declared operands and measured validation before administrative publication; no scientific gain is inferred."
        for k in value["acceptance_gates"]
    }
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Recompute primitives in a fresh process; updated aggregate hashes cannot invent authority."""
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
        if work["contract"] != authority(raw / "authority") or work["history"] != evidence.history(
            raw / "authority"
        ):
            raise ValueError("authority_or_history_drift")
        if work["exclusion_scope_rows"] != scope_rows(work["contract"]["tasks"]):
            raise ValueError("exclusion_scope_drift")
        for control in work["authority_mutation_rows"]:
            if control["passed"] is not (
                not authority(raw / "controls" / control["mutation"])["ready"]
            ):
                raise ValueError("control_drift")
        ledger = next(
            (
                s
                for s in work["snapshots"]
                if s["source"]["path"] == str(ROOT.joinpath(*evidence.arc.arc.DEFAULT_LEDGER_PARTS))
            ),
            None,
        )
        arc = evidence.arc.arc_evidence(ledger_path=Path(ledger["path"]) if ledger else None)
        if ledger:
            for r in arc["references"] + arc["failures"]:
                if r["path"] == ledger["path"]:
                    r["path"] = ledger["source"]["path"]
        arc.update(cheap_skip_count=int(not arc["ready"]), panel_rerun_count=0)
        if work["arc"] != arc:
            raise ValueError("arc_frontier_drift")
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
