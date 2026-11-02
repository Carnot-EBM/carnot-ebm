"""REQ-REPORT-8681: bind fresh authority without changing original scientific operands."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import v736_contract_methods as base
from carnot.reporting import v736_historical_inputs as readers
from carnot.reporting import v744_contract_methods as original
from carnot.reporting.v745_private_inputs import binding as original_inputs, ARCHIVE
from carnot.reporting.roadmap_contract import compare_contract, parse_design

Json = dict[str, Any]
ROOT = original.ROOT
NAME, TASK, MILESTONE = (
    "experiment_8681_v745_contract_methods",
    "exp8681-contract-methods",
    "2026.11.745",
)
DESIGN, ACTIVE, STAGED = original.DESIGN, original.ACTIVE, original.STAGED
TASKS_PIN = "05537ca236359e3c7b318c5640e198107a6077521b209fe9d9ad968099b89d43"
PROTOCOL = original.PROTOCOL
CLI = f"scripts/experiments/{NAME}.py"
TESTS = ["tests/python/test_v745_contract_8681.py"]
TEST = TESTS[0]
ORIGINAL_TESTS = original.TESTS
OWNED = [
    f"python/carnot/reporting/v745_contract_{s}.py" for s in ["methods", "report", "runner"]
] + [CLI]
OWNED += [
    "python/carnot/reporting/v745_private_inputs.py",
    "scripts/experiments/experiment_8681_original_private.py",
]
MODEL_SPECS: list[Json] = []
reference, read, checked = original.reference, original.read, original.checked
atomic_json, checksum = original.atomic_json, original.checksum
canonical_hash, sha256_file = original.canonical_hash, original.sha256_file
tasks_digest, gate = original.tasks_digest, original.gate


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase counts so the supervisor can distinguish work from a stalled child."""
    print(f"[exp8681] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Keep the original protocol exact and authenticate the new task map independently."""
    with original_inputs():
        p = dict(original.protocol())
    text = (ROOT / DESIGN).read_text()
    tasks = parse_design(text, milestone=MILESTONE)[1]
    if tasks_digest(tasks) != TASKS_PIN:
        raise ValueError("independent_v745_task_seal")
    p["task_map_v745"] = tasks
    p["future_dependencies_v745"] = [
        dict(task_id=t["id"], path=t["deliverable"], status="future_dependency_not_observation")
        for t in tasks[1:]
    ]
    return p


def authority(root: Path) -> Json:
    """A fixed task digest rejects coordinated changes to every mutable authority."""
    failures, refs = [], []
    try:
        text = (root / DESIGN).read_text()
        stage, active = root / STAGED, root / ACTIVE
        chosen = stage if stage.is_file() else active
        planned = yaml.safe_load(chosen.read_bytes())
        actual = (
            yaml.safe_load(active.read_bytes()) if stage.is_file() and active.is_file() else planned
        )
        if stage.is_file() and actual.get("milestone") == "2026.11.744":
            actual = planned
        result = compare_contract(
            text, planned, actual, milestone=MILESTONE, first_id=8681, count=14
        )
        machine = parse_design(text, milestone=MILESTONE)[1]
        envelope = json.loads(
            text.split("<!-- V745_TASK_CONTRACT_START -->", 1)[1]
            .split("```json", 1)[1]
            .split("```", 1)[0]
        )
        for i, row in enumerate(result["rows"]):
            row["checks"]["full_dictionary"] = (
                i < len(planned["tasks"])
                and i < len(actual["tasks"])
                and i < len(machine)
                and planned["tasks"][i] == actual["tasks"][i] == machine[i]
            )
            row["matched"] = all(row["checks"].values())
            row["excluded"] = False
            row["censored"] = not row["matched"]
            row["status"] = "completed" if row["matched"] else "censored"
            row["effective_independent_groups"] = 0
            row["absolute_metric"] = int(row["matched"])
        ready = bool(
            not set(result["errors"]) - {"row_mismatch"}
            and all(r["matched"] for r in result["rows"])
            and envelope.get("canonical_tasks_sha256") == TASKS_PIN
            and tasks_digest(planned["tasks"]) == TASKS_PIN
            and re.search(r"(?m)^Canonical task SHA256: `" + TASKS_PIN + r"`$", text)
            and "<!-- V745_TASK_CONTRACT_END -->" in text
        )
        rows = result["rows"]
        refs = [reference(root / DESIGN), reference(chosen)]
        if active.is_file() and active != chosen:
            refs.append(reference(active))
        if not ready:
            failures.append(
                gate(root / DESIGN, "fourteen_complete_sealed_objects", True, False, TASK)
            )
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        ready = False
        failures.append(gate(root / DESIGN, "v745_authority", MILESTONE, str(error), TASK))
        rows = [
            dict(
                unit_id=f"exp{8681 + i}",
                order=i + 1,
                arm="four_source_contract",
                matched=False,
                status="censored",
                absolute_metric=0,
                censored=True,
                excluded=False,
                effective_independent_groups=0,
                checks={},
            )
            for i in range(14)
        ]
    return dict(
        ready=ready,
        rows=rows,
        references=refs,
        gate_check_summary=failures,
        staging_present=(root / STAGED).is_file(),
    )


def methods_manifests(p: Json) -> Json:
    """Separate new studies from the unchanged learning, relational and batch methods."""
    refs = [reference(ROOT / PROTOCOL), reference(ROOT / DESIGN)]
    learning = {
        k: p[k]
        for k in [
            "objective_updates",
            "objective_audit",
            "cohort_partition_reference",
            "learning_registration",
            "head_checkpoint_reference",
            "learning_data_reference",
            "learning_label_references",
            "test_inventory_reference",
        ]
    }
    return dict(
        learning=dict(
            learning,
            source_input_manifest=refs,
            operand_hashes={
                k: canonical_hash(p[k]) for k in ["objective_updates", "objective_audit"]
            },
        ),
        extraction=dict(
            protocol=p["relational_evidence_protocol"],
            source_input_manifest=refs + p["operand_references"]["extraction"],
        ),
        deployment=dict(
            protocol=p["installed_batching_protocol"],
            source_input_manifest=refs + p["operand_references"]["deployment"],
        ),
        symbolic_canary=dict(
            task_id="exp8690-qwen-symbolic-canary",
            calls=36,
            cases=12,
            max_output_tokens=128,
            temperature=0,
            MODEL_SPECS=["unsloth/Qwen3.8-27B-GGUF"],
            independent_truth=True,
            execute_generated_code=False,
            controls=["direct_decision", "source_to_record", "record_grounded_decision"],
            source_input_manifest=refs,
        ),
        fixed_point=dict(
            task_id="exp8692-spline-precision-feasibility",
            formats=[[16, 12], [24, 18], [32, 24]],
            primary=[32, 24],
            vectors=256,
            vector_partition=[128, 64, 64],
            rounding="ties_to_even",
            saturation=True,
            checked_accumulator_bits=64,
            probability_error_at_most=0.001,
            update_sequences=4,
            updates_per_sequence=256,
            local_writes_at_most=37,
            physical_fpga_run=False,
            host_operations=["sigmoid", "logit", "persistence"],
            source_input_manifest=refs + [p["head_checkpoint_reference"]],
        ),
    )


def mutation_controls(root: Path) -> list[Json]:
    """Try valid staging and activation, then change each independently sealed operand."""
    text = (ROOT / DESIGN).read_text()
    tasks = protocol()["task_map_v745"]
    modes = [
        "staged",
        "activated",
        "count",
        "order",
        "id",
        "title",
        "phase",
        "deliverable",
        "model",
        "MODEL_SPECS",
        "inference_substrate_class",
        "prompt",
        "gate",
        "prior_failure",
        "marker",
        "start_marker",
        "table",
        "digest",
        "envelope_digest",
        "rehashed",
        "foreign_active",
    ]
    rows = []
    with TemporaryDirectory(prefix="exp8681-authority-", dir="/var/tmp") as directory:
        private = Path(directory)
        design = private / DESIGN
        design.parent.mkdir(parents=True)
        for i, mode in enumerate(modes):
            value, changed = dict(milestone=MILESTONE, tasks=deepcopy(tasks)), text
            if mode == "count":
                value["tasks"].pop()
            if mode == "order":
                value["tasks"].reverse()
            if mode in {
                "id",
                "title",
                "phase",
                "deliverable",
                "model",
                "inference_substrate_class",
                "prompt",
            }:
                value["tasks"][0][mode] = "forged"
            if mode == "MODEL_SPECS":
                value["tasks"][0][mode] = ["forged"]
            if mode == "gate":
                value["tasks"][1]["gated_on"][0]["value"] = 99
            if mode == "prior_failure":
                value["tasks"][0]["prior_failures"][0]["retire_if_same_verdict"] = False
            if mode in {"marker", "start_marker"}:
                changed = changed.replace(
                    "<!-- V745_TASK_CONTRACT_" + ("END" if mode == "marker" else "START") + " -->",
                    "",
                )
            if mode == "table":
                changed = changed.replace("| 1 |", "| 99 |", 1)
            if mode == "digest":
                changed = changed.replace(TASKS_PIN, "0" * 64)
            if mode == "envelope_digest":
                changed = changed.replace(
                    '"canonical_tasks_sha256": "' + TASKS_PIN,
                    '"canonical_tasks_sha256": "' + "0" * 64,
                )
            if mode == "rehashed":
                value["tasks"][0]["prompt"] += "forged"
                envelope = dict(value, canonical_tasks_sha256=tasks_digest(value["tasks"]))
                changed = re.sub(
                    r"(```json\s*)(.*?)(\s*```)",
                    lambda m: m[1] + json.dumps(envelope) + m[3],
                    changed,
                    count=1,
                    flags=re.S,
                )
                changed = changed.replace(TASKS_PIN, envelope["canonical_tasks_sha256"])
            design.write_text(changed)
            (private / ACTIVE).write_text(yaml.safe_dump(value))
            stage = private / STAGED
            if mode in {"staged", "foreign_active"}:
                stage.write_text(yaml.safe_dump(value))
                (private / ACTIVE).write_text(
                    "milestone: "
                    + ("2026.11.744" if mode == "staged" else "2026.10.743")
                    + "\ntasks: []\n"
                )
            else:
                stage.unlink(missing_ok=True)
            observed = authority(private)["ready"]
            expected = mode in {"staged", "activated"}
            rows.append(
                dict(
                    mutation=mode, expected=expected, observed=observed, passed=observed == expected
                )
            )
            progress("authority_control_" + mode, i + 1, len(modes) - i - 1)
    return rows


def measure(root: Path, raw: Path, run_date: str) -> Json:
    """Freeze source and original inputs before deterministic registration measurements."""
    start_wall, start_mono = time.time_ns(), time.monotonic_ns()
    frozen_sources = [reference(ROOT / name) for name in OWNED + TESTS + ORIGINAL_TESTS]
    frozen_sources += [
        reference(root / name) for name in [DESIGN, ACTIVE, STAGED] if (root / name).is_file()
    ]
    execution = raw / "execution_manifest.json"
    execution_ref = reference(execution) if execution.is_file() else None
    if execution_ref:
        config = Path(read(execution_ref)["private_scratch"]) / "coverage.ini"
        frozen_sources += [reference(config)] if config.is_file() else []
    snapshot = raw / "authority"
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in [DESIGN, ACTIVE, STAGED]:
        source, target = root / name, snapshot / name
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    with (
        patch.multiple(
            base,
            NAME=NAME,
            TASK=TASK,
            MILESTONE=MILESTONE,
            PROTOCOL=PROTOCOL,
            OWNED=OWNED,
            TESTS=TESTS,
            protocol=protocol,
            authority=authority,
            progress=progress,
        ),
        patch.object(readers, "historical", historical),
        patch.object(readers, "runtime", lambda ignored, logs: original.runtime(root, logs)),
    ):
        work = dict(base.measure(snapshot, raw))
    work["run_date"] = run_date
    work["wall_ns"], work["start_ns"] = start_wall, start_mono
    work["code_refs"] += frozen_sources
    if execution_ref:
        work["execution_manifest_reference"] = execution_ref
        work["code_refs"].append(execution_ref)
    work["numerical"] = original.numerical(work["protocol"])
    work["authority_mutation_rows"] = mutation_controls(snapshot)
    work["methods_manifest"] = methods_manifests(work["protocol"])
    for name, manifest in work["methods_manifest"].items():
        atomic_json(raw / "methods" / (name + ".json"), manifest)
    work["methods_manifest_references"] = [
        reference(path) for path in sorted((raw / "methods").glob("*.json"))
    ]
    work["source_refs"] += [
        reference(ROOT / DESIGN),
        reference(ARCHIVE),
        original.PRIOR_FAILURE_REFERENCE,
    ] + original.read(original.PRIOR_FAILURE_REFERENCE)["references"]
    work["code_refs"] += [
        reference(ROOT / path)
        for path in ORIGINAL_TESTS
        + original.OWNED
        + [
            "python/carnot/reporting/roadmap_contract.py",
            "python/carnot/reporting/primary_publication.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "python/carnot/verify/cuda_reader_8410.py",
            "ops/exclusion_manifest.yaml",
        ]
    ]
    work["end_ns"] = time.monotonic_ns()
    work["duration_s"] = (work["end_ns"] - work["start_ns"]) / 1e9
    work["phase_spans"][0].update(
        start_monotonic_ns=start_mono,
        end_monotonic_ns=work["end_ns"],
        duration_s=work["duration_s"],
    )
    atomic_json(raw / "measurement.json", work)
    atomic_json(raw / "measurement_seal.json", work)
    progress("measurement_frozen", 14, 0)
    return work


def terminal_custody(ref: Json, field: str) -> Json:
    """Authenticate the declared producer and its terminal publication without rerunning it."""
    refs = [ref]
    try:
        value = read(ref)
        terminal = Path(value["terminal_validation_sidecar_path"])
        refs.append(reference(terminal))
        side = read(refs[-1])
        publication = side.get("publication", side)
        bound_hash = publication.get(
            "primary_sha256", side.get("candidate_reference", {}).get("sha256")
        )
        if publication.get("sidecar_path"):
            bound = Path(publication["sidecar_path"])
            refs.append(reference(bound))
            report = read(refs[-1])["report"]
            terminal_ok = report["passed"] is True
        elif "report" in side:
            terminal_ok = side["report"]["passed"] is True
        else:
            terminal_ok = bool(
                side.get("passed", all(r["passed"] for r in side.get("validation_receipts", [])))
            )
        passed = bool(
            value.get(field) == 1
            and value.get("required_checks_passed") is True
            and value.get("flagged_adversarial") is False
            and bound_hash == ref["sha256"]
            and terminal_ok
        )
        observed = dict(
            value=value.get(field),
            flagged=value.get("flagged_adversarial"),
            bound_hash=bound_hash,
            terminal_passed=terminal_ok,
        )
        failures = (
            []
            if passed
            else [
                gate(
                    Path(ref["path"]),
                    field + "_producer_terminal",
                    True,
                    observed,
                    value.get("task_id", "unknown"),
                )
            ]
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        passed = False
        failures = [
            gate(
                Path(ref["path"]),
                field + "_producer_terminal",
                True,
                str(error),
                "declared_producer",
            )
        ]
    return dict(passed=passed, references=refs, gate_check_summary=failures, field=field)


def historical(p: Json) -> Json:
    """Original operands stay branch-local and gain explicit terminal producer custody."""
    result = dict(original.historical(p))
    requirements = {
        "learning_operands_ready_score": {
            "8431": "head_ready_score",
            "8580": "current_validator_ready_score",
            "8581": "online_adapter_ready_score",
        },
        "extraction_inputs_ready_score": p["branch_requirements"]["extraction_inputs_ready_score"],
        "deployment_inputs_ready_score": dict(
            p["branch_requirements"]["deployment_inputs_ready_score"],
            **{"8633": "installed_prototype_ready_score"},
        ),
    }
    result["terminal_producer_rows"] = []
    for score, producers in requirements.items():
        for number, field in producers.items():
            ref = p["current_producers"].get(number, p["v743_producers"].get(number))
            row = terminal_custody(ref, field)
            result["terminal_producer_rows"].append(dict(row, upstream=number, branch=score))
            result["references"] += row["references"]
            result["gate_check_summary"] += row["gate_check_summary"]
            if not row["passed"]:
                result[score] = 0
    return result
