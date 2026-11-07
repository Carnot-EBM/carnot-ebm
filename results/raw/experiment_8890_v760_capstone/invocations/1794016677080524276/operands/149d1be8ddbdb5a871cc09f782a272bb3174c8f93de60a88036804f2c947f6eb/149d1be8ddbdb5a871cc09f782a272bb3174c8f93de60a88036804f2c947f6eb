"""REQ-REPORT-8878: complete administrative authority never establishes scientific benefit."""

from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import yaml

from carnot.reporting import v755_branch_contract as base
from carnot.reporting import v759_branch_contract as prior
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.roadmap_contract import compare_contract, parse_design
from carnot.reporting.v749_contract_methods import encode
from carnot.reporting.v756_branch_qualification import verify_measured
from scripts.audit_roadmap_gates import _required_artifact_fields_block
from scripts.experiment_template import normalize_artifact_for_template_write
from scripts.roadmap_schema import Roadmap

Json = dict[str, Any]
ROOT, DESIGN, ACTIVE, STAGED = prior.ROOT, prior.DESIGN, prior.ACTIVE, prior.STAGED
NAME, TASK, MILESTONE = (
    "experiment_8878_v760_branch_contract",
    "exp8878-branch-contract",
    "2026.11.760",
)
PIN = "e650b5b205ad20ac620d4e19adc655b53879c1637b21fc7e0a8dfc302ce97a1a"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = ["python/carnot/reporting/v760_branch_contract.py", CLI]
TEST = "tests/python/test_v760_branch_contract_8878.py"
METHODS = "openspec/change-proposals/research-methods-v760-20261106.json"
PINS = {
    "results/experiment_8866_v759_branch_contract.json": "0909c9352854d04febb5e6a2e5dd9a2d67bed228bc2b916399f8fb692f23635a",
    "results/experiment_8877_v759_capstone.json": "e9f75e768cc1f2290f1123d80b455620da1bc4818fe5cd84d11e288eeacd7594",
    "openspec/change-proposals/research-roadmap-v759-preserved-20261106.md": "c0550b31623ec711c5cb833b1c6fcb8aa986ae77ce9cb03293dbb30ca68d2407",
    "openspec/change-proposals/research-roadmap-v759-queue-preserved-20261106.yaml": "e1eb2c3ca1487fa4efc9acef930c1ecb45ed497d3fc8de87da597290b59b02ad",
}
reference, checksum, seal = prior.reference, prior.checksum, prior.seal
runtime = prior.runtime


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries so supervision keeps its original wall-clock deadline."""
    print(f"[exp8878] phase={phase} completed={completed} pending={pending}", flush=True)


def failure(path: Path, check: str, field: str, expected: Any, observed: str) -> Json:
    """Keep external absence separate from rejected bytes and a measured zero."""
    return dict(
        check=check,
        upstream=TASK,
        artifact_field=field,
        expected=expected,
        observed=observed,
        **reference(path),
    )


def authority(root: Path) -> Json:
    """Read independent JSON first; a stale active queue cannot manufacture a milestone."""
    tasks, table, rows, failures = [], [], [], []
    mode = "unavailable"
    try:
        text = (root / DESIGN).read_text()
        table, tasks = parse_design(text, milestone=MILESTONE)
        active = yaml.safe_load((root / ACTIVE).read_bytes())
        staged = yaml.safe_load((root / STAGED).read_bytes()) if (root / STAGED).is_file() else None
        if active["milestone"] == MILESTONE:
            mode, selected = "activated", active
        elif staged is not None and staged["milestone"] == MILESTONE:
            mode, selected = "staged", staged
        else:
            raise ValueError("no_current_authority")
        queues = [deepcopy(staged or selected), deepcopy(selected)]
        for queue in queues:
            Roadmap.model_validate(queue)
            for actual, expected in zip(queue["tasks"], tasks):
                if actual == dict(expected, prompt=encode(expected["prompt"])):
                    actual["prompt"] = expected["prompt"]
        checked = compare_contract(text, *queues, milestone=MILESTONE, first_id=8878, count=13)
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
            and canonical_hash(tasks) == "sha256:" + PIN
            and f"Canonical task SHA-256: `{PIN}`" in text
            and "<!-- V760_TASK_CONTRACT_END -->" in text
        ):
            raise ValueError("independent_authority_mismatch")
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
        observed = (
            "missing_file"
            if not (root / DESIGN).is_file()
            else "missing_field"
            if isinstance(error, IndexError)
            else "rejected_producer"
        )
        failures.append(
            failure(
                root / DESIGN,
                "complete_authority",
                "full_task_objects",
                "13 full objects, table, digest, closing marker and current queue",
                observed,
            )
        )
    for i in range(13):
        if i >= len(rows):
            rows.append(
                dict(
                    unit_id=tasks[i]["id"] if i < len(tasks) else f"exp{8878 + i}-missing",
                    order=i + 1,
                    checks=dict(independent_authority=False),
                    matched=False,
                )
            )
        row = rows[i]
        matched = row["matched"] and not failures
        row.update(
            matched=matched,
            metric=int(matched),
            absolute_metric=int(matched),
            denominator=1,
            raw_numerator=sum(row["checks"].values()),
            raw_denominator=len(row["checks"]),
            cluster_id="administrative_contract",
            arm="full_object_custody",
            seed=None,
            status="completed" if matched else "censored",
            censored=not matched,
            missing_reason=None if matched else "independent_authority_unavailable_or_rejected",
            excluded=False,
            effective_independent_groups=0,
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
    """Corrupt private copies so every rejection starts with otherwise valid authority."""
    text = (source / DESIGN).read_text()
    tasks = parse_design(text, milestone=MILESTONE)[1]
    queue = yaml.safe_load(
        (
            source / (STAGED if authority(source)["comparison_mode"] == "staged" else ACTIVE)
        ).read_bytes()
    )
    rows = []
    for mode in [
        "missing_last_task",
        "truncation",
        "prompt",
        "gate",
        "model",
        "closing_marker",
        "table",
    ]:
        root = raw / mode
        path = root / DESIGN
        path.parent.mkdir(parents=True, exist_ok=True)
        changed, body = deepcopy(tasks), text
        if mode == "missing_last_task":
            changed.pop()
        if mode == "truncation":
            body = text[: text.index('"id": "exp8890-')]
        if mode in {"prompt", "model"}:
            changed[0][mode] = "forged"
        if mode == "gate":
            changed[3]["gated_on"][0]["artifact_field"] = "wrong_gate"
        if mode == "closing_marker":
            body = text.replace("<!-- V760_TASK_CONTRACT_END -->", "")
        if mode == "table":
            body = text.replace("| 1 | exp8878-branch-contract", "| 2 | exp8878-branch-contract")
        path.write_text(body)
        (root / ACTIVE).write_text(yaml.safe_dump(dict(queue, tasks=changed)))
        rows.append(
            dict(
                mutation=mode,
                passed=not authority(root)["ready"],
                references=[reference(path), reference(root / ACTIVE)],
            )
        )
        progress("authority_control_" + mode, len(rows), 7 - len(rows))
    return rows


def history(raw: Path) -> Json:
    """Authenticate original operands without repairing or relabeling historical failures."""
    refs, failures, verdicts, operands, absences = [], [], {}, {}, []
    for name, pin in PINS.items():
        path = ROOT / name
        if reference(path)["sha256"] != "sha256:" + pin:
            failures.append(
                failure(
                    path,
                    "original_v759",
                    "sha256",
                    pin,
                    "missing_file" if not path.is_file() else "rejected_producer",
                )
            )
            continue
        refs.append(seal(path, raw))
        if path.suffix != ".json":
            continue
        value = json.loads(path.read_bytes())
        verdicts[name] = value["verdict_class"]
        for ref in value["code_config_hashes"] + value["raw_shard_hashes"]:
            operands[(ref["path"], ref["sha256"])] = ref
    for i, ref in enumerate(operands.values()):
        path = Path(ref["path"])
        if ref["sha256"] is None and not path.is_file():
            absences.append(dict(ref, status="historical_asserted_absence"))
        elif not path.is_file() or reference(path)["sha256"] != ref["sha256"]:
            failures.append(
                failure(
                    path,
                    "original_source_raw_operand",
                    "sha256",
                    ref["sha256"],
                    "missing_file" if not path.is_file() else "rejected_producer",
                )
            )
        else:
            refs.append(seal(path, raw))
        if i % 100 == 0:
            progress("historical_operands", i + 1, len(operands) - i - 1)
    return dict(
        authenticated=not failures,
        references=refs,
        failures=failures,
        prior_verdicts=verdicts,
        asserted_absence_rows=absences,
        scientific_results_admitted=False,
        historical_replay_required=False,
    )


def freeze(root: Path, raw: Path, run_date: str) -> Json:
    """Keep original bytes private before validation or publication can start."""
    progress("preconditions_before", 0, 13)
    if (raw / "work.json").exists():
        raise FileExistsError("immutable_invocation_already_exists")
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw.chmod(0o700)
    probe = raw / "write_probe"
    probe.write_text(TASK)
    writable = probe.read_text() == TASK
    probe.unlink()
    resources = dict(
        repository_virtualenv=Path(sys.prefix).resolve() == (ROOT / ".venv").resolve(),
        private_output_writable=writable,
        private_mode=oct(raw.stat().st_mode & 0o777),
        free_bytes=shutil.disk_usage(raw).free,
        required_free_bytes=536870912,
        installed_dependencies={
            p: importlib.metadata.version(p) for p in ["coverage", "pytest", "pydantic", "PyYAML"]
        },
        time_budget_s=4200,
        hard_cap_s=4800,
        no_model_loads=True,
        no_device_probes=True,
        activation_authorized=False,
    )
    historical = history(raw)
    branches = runtime(root, raw)
    for row in branches["gate_check_summary"]:
        row["upstream"] = "supplied_cuda_runtime_repair"
    snapshots = historical["references"] + branches["references"]
    for name in [DESIGN, ACTIVE, STAGED]:
        if (root / name).is_file():
            snapshots.append(seal(root / name, raw))
            target = raw / "authority" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / name).read_bytes())
    for name in [
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "research-references.md",
        "ops/exclusion_manifest.yaml",
    ]:
        snapshots.append(seal(ROOT / name, raw))
    methods, method_failures = [], []
    try:
        saved = seal(root / METHODS, raw)
        snapshots.append(saved)
        methods = json.loads(Path(saved["path"]).read_bytes())
        for row in methods:
            for source in row["sources"]:
                if source.get("path"):
                    if reference(Path(source["path"]))["sha256"] != source["sha256"]:
                        historical["failures"].append(
                            failure(
                                Path(source["path"]),
                                "primary_source",
                                "sha256",
                                source["sha256"],
                                "rejected_producer"
                                if Path(source["path"]).is_file()
                                else "missing_file",
                            )
                        )
                        historical["authenticated"] = False
                    else:
                        snapshots.append(seal(Path(source["path"]), raw))
    except (OSError, ValueError, KeyError, TypeError):
        methods = []
        method_failures.append(
            failure(
                root / METHODS,
                "method_map",
                "source_method_map",
                "eight topics and six secondary channels",
                "rejected_producer" if (root / METHODS).is_file() else "missing_file",
            )
        )
    contract = authority(raw / "authority")
    controls = mutations(raw / "authority", raw / "controls") if contract["ready"] else []
    work = dict(
        contract=contract,
        history=historical,
        branches=branches,
        methods=methods,
        method_failures=method_failures,
        snapshots=snapshots,
        authority_mutation_rows=controls,
        run_date=run_date,
        owner_pid=os.getpid(),
        start_ns=start,
        end_ns=time.monotonic_ns(),
        raw=str(raw.absolute()),
        preconditions=resources,
        raw_operands=[reference(p) for p in sorted((raw / "controls").rglob("*")) if p.is_file()],
    )
    atomic_json(raw / "work.json", work)
    progress(
        "preconditions_after",
        sum(r["matched"] for r in contract["rows"]),
        13 - sum(r["matched"] for r in contract["rows"]),
    )
    return dict(work, raw_reference=reference(raw / "work.json"))


def build(work: Json, validation: Json) -> Json:
    """Owned failures disqualify; independent missing evidence remains terminal blocked."""
    end = max(
        [
            work["end_ns"],
            *[r.get("ended_monotonic_ns", work["end_ns"]) for r in validation["receipts"]],
        ]
    )
    prepared = dict(
        work,
        end_ns=end,
        history=dict(work["history"], original_child_qualified=work["history"]["authenticated"]),
        manifests=[],
        branches=dict(runtime=work["branches"], arc=dict(ready=False), failures=[]),
    )
    value = base.build(prepared, validation)
    tasks = work["contract"]["tasks"]
    producers = {t["id"]: t for t in tasks}
    gates = [
        dict(
            edge,
            required_field_declared=edge["upstream"] in producers
            and edge["artifact_field"]
            in _required_artifact_fields_block(producers[edge["upstream"]]["prompt"]),
        )
        for edge in base.edges(tasks)
    ]
    independent = all(
        g["upstream"] != TASK
        or t["id"] == "exp8887-qwen-source-canary"
        and g["artifact_field"] == "runtime_repair_ready_score"
        for t in tasks
        for g in t.get("gated_on", [])
    )
    resources, controls = work["preconditions"], work["authority_mutation_rows"]
    checked = bool(
        value["required_checks_passed"]
        and resources["private_output_writable"]
        and resources["repository_virtualenv"]
        and resources["free_bytes"] >= resources["required_free_bytes"]
        and value["duration_s"] < 4200
        and all(r["passed"] for r in controls)
    )
    mapped = (
        len(work["methods"]) == 14
        and sum(r["kind"] == "topic" for r in work["methods"]) == 8
        and sum(r["kind"] == "secondary_channel" for r in work["methods"]) == 6
    )
    ready = bool(
        checked
        and work["contract"]["ready"]
        and work["history"]["authenticated"]
        and len(controls) == 7
        and mapped
        and all(r["required_field_declared"] for r in gates)
        and independent
    )
    verdict = "disqualified" if not checked else "null" if ready else "blocked"
    failures = (
        work["contract"]["failures"]
        + work["history"]["failures"]
        + work["branches"]["gate_check_summary"]
        + work["method_failures"]
    )
    value.pop("preserved_v754_reference")
    value.pop("arc_new_outcomes_available_score")
    value.update(
        experiment_id=8878,
        task_id=TASK,
        milestone=MILESTONE,
        random_seed=7608878,
        honest_verdict=f"complete_{verdict}_v760_branch_contract",
        verdict_class=verdict,
        intended_count=13,
        censored_count=13 - value["completed_count"],
        contract_ready_score=int(ready),
        runtime_repair_ready_score=int(work["branches"]["ready"]),
        canonical_tasks_sha256=PIN,
        required_checks_passed=checked,
        execution_venue="host",
        preconditions_checked=resources,
        gate_check_summary=failures,
        gate_edge_rows=gates,
        authority_mutation_rows=controls,
        source_method_map=work["methods"],
        preserved_v759_reference=work["history"],
        authority_comparison_mode=work["contract"]["comparison_mode"],
        external_admission_rows=[
            dict(
                branch="runtime",
                downstream="exp8887-qwen-source-canary",
                optional=True,
                ready=work["branches"]["ready"],
                independent_cpu_roots_gated=False,
                inference_substrate_class="no_model_load"
                if work["branches"]["ready"]
                else "blocked_no_run",
            )
        ],
        acceptance_gates=dict(
            exact_contract=work["contract"]["ready"],
            owned_validation=checked,
            preserved_history=work["history"]["authenticated"],
            mutation_controls=len(controls) == 7 and all(r["passed"] for r in controls),
            method_map=mapped,
            exact_gate_fields=all(r["required_field_declared"] for r in gates),
            independent_cpu_roots=independent,
        ),
        code_config_hashes=[
            reference(ROOT / p)
            for p in OWNED
            + [
                TEST,
                METHODS,
                "python/carnot/reporting/v759_branch_contract.py",
                "python/carnot/reporting/v758_branch_contract.py",
                "python/carnot/reporting/v756_branch_qualification.py",
                "python/carnot/reporting/v749_contract_methods.py",
                "python/carnot/reporting/v685_authority_lifecycle.py",
                "python/carnot/reporting/v709_execution.py",
                *prior.prior.REUSED,
            ]
        ],
        arc_task_count=sum("arc" in t["title"].lower().split() for t in tasks),
        generator_training_task_count=sum(
            "generator training" in t["title"].lower() for t in tasks
        ),
        methodology_note="Administrative authority and source-method ingestion only. Preserved V759 failures remain failures. No activation, science, model load, device probe, ARC or generator training.",
    )
    value["phase_spans"] = [
        dict(phase="operand_custody", start_s=0, end_s=(work["end_ns"] - work["start_ns"]) / 1e9),
        dict(
            phase="owned_validation",
            start_s=(work["end_ns"] - work["start_ns"]) / 1e9,
            end_s=value["duration_s"],
        ),
    ]
    value["raw_shard_hashes"] += work["raw_operands"]
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Administrative custody is not scientific benefit. Independent full authority and private controls establish readiness. Historical failures remain unchanged; external absence blocks, owned failure disqualifies, optional runtime does not gate CPU roots.",
    )
    value = normalize_artifact_for_template_write(value)
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild from sealed operands so recomputing an aggregate hash cannot hide drift."""
    try:
        value = json.loads(path.read_bytes())
        if value["required_checks_passed"] and not verify_measured(
            value["owned_statement_coverage"], OWNED
        ):
            raise ValueError("unqualified_statements")
        for ref in value["raw_shard_hashes"] + value["code_config_hashes"]:
            if reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("operand_drift")
        ref = value["raw_shard_hashes"][0]
        work = json.loads(Path(ref["path"]).read_bytes())
        if work["contract"] != authority(Path(work["raw"]) / "authority"):
            raise ValueError("authority_drift")
        validation_path = Path(work["raw"]) / "validation.json"
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
        passed = value == expected and checksum(value) == value["reproducibility_checksum"]
        return dict(passed=passed, reason="replay_passed" if passed else "semantic_drift")
    except (OSError, ValueError, KeyError, TypeError):
        return dict(passed=False, reason="invalid_or_changed_evidence")
