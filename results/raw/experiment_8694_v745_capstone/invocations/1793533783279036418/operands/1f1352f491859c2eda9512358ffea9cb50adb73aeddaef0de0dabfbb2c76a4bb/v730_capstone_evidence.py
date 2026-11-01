"""REQ-REPORT-8498: independently replay sealed evidence before assigning scientific scope."""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import time
from typing import Any, Iterator
from unittest.mock import patch

import yaml

from carnot.reporting import v729_capstone_evidence as previous
from carnot.reporting import v726_capstone_report as report
from carnot.reporting import v730_contract_methods as methods
from carnot.reporting.current_work_receipt import atomic_json

Json = dict[str, Any]
base, history, workers = previous.base, previous.history, previous.workers
ROOT = base.ROOT
NAME, TASK, MILESTONE = "experiment_8498_v730_capstone", "exp8498-capstone", "2026.10.730"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v730_capstone_8498.py"
OWNED = [f"python/carnot/reporting/v730_capstone{s}.py" for s in ["_evidence", "_report", ""]] + [
    CLI
]
MODEL_SPECS: list[Json] = []
PINS = dict(
    zip(
        range(8485, 8498),
        [
            "552831473fa8df1fd272ae690082cbf23513e559fa9648e9ab0f588f603bb667",
            "f513a56bc388076cb17338cd92f7a256616e2c5fbf2fb8c71dc649872fe4d007",
            "094f758e2046efec7a93d22fcc297a56ac2af388a574fef78a638b15e61dcb4d",
            "64a811a3b880da8446f4bc37bacdef23d00f4558c0d348bb0994e69e583d07cd",
            "f306c910fd8fe95b12ca2505655347c73846c63d7a46700f433f7b67d9384bad",
            "723b8e819301e0c28fabc5768673e503ad38e9dc13aa0c60e36e006e6446bfb3",
            "cbf40c76d250d158881d5ba69164ccc6ed4fc6b3ce92d193272ca543f3e00efa",
            "e4aac512456de843ad91b91f3ed92faedb93c42ebd94f8b7054ba62e346aacf0",
            "55ce337aacfdc707ea068bd1c4a2b38ccf220d16e9154c8354370680b0081e52",
            "e1584c837751142964bb9ce59ba1bfa887df1c6ac331f2c658f2e70b482ab742",
            "eef02a3fd063b87c893b2966901bb346d68b3d3660e6ee14b3bba0068c0447e9",
            "5804a065a668c89df631687b9277a963643848b61b640e59b8efc1ca72a62ceb",
            "97e39bfd14709b241e7f9ea60b6b3c32292d9786ff12d3e7008b40d46d74b5de",
        ],
        strict=True,
    )
)
PREGATE_PINS = {n: PINS[n] for n in [8491, 8495]}
READERS = dict(
    zip(
        [8485, 8486, 8487, 8488, 8489, 8490, 8492, 8493, 8494, 8496, 8497],
        [
            "reporting.v730_contract_methods",
            "reporting.v730_source_control_qualification",
            "verify.v730_source_control_fit",
            "reporting.v730_source_information_audit",
            "reporting.objective_checkpoint_8489",
            "reporting.brier_online_learning_8490",
            "reporting.publication_cost_model_8492",
            "reporting.append_store_qualification_8493",
            "reporting.append_service_cost_8494",
            "reporting.arc_panel_qualification_8496",
            "reporting.board_operation_boundary_8497",
        ],
        strict=True,
    )
)
PREDECESSOR_PINS = previous.PREDECESSOR_PINS | {
    8484: "sha256:d188ab01c8b0ab964350772128ba152df3b8ca60a8e76e1b3566c9e0701b7c00"
}
SELECT = previous.SELECT | {
    "brier_gain_vs_q_only_spline",
    "decision_loss_deltas",
    "contrasts",
    "familywise_intervals",
    "source_information_benefit_score",
    "scientific_measurement_complete",
    "negative_control_degeneracy",
    "fit_tune_losses",
    "head_hashes",
    "scalar_numeric_checks",
    "gradient_checks",
    "historical_qualification",
    "superseded_qualification",
    "changed_reference_control_reference",
    "historical_collection_reference",
    "primary_speedup_ci95",
    "primary_p95_ratio",
    "cell_results",
    "paired_repeat_rows",
    "append_speed_benefit_score",
    "retention_rows",
    "label_updates",
    "admission_count",
    "coefficient_change_count",
    "lost_feedback_count",
    "restart_checks",
    "durability_scope",
    "equivalence_rows",
    "coverage_missing_before_after",
    "new_level_solves",
    "headline_solve_credit",
    "per_episode_rows",
    "refinement_status",
    "frozen_panel",
}
checksum, reference, freeze, load, slot, memory = (
    base.checksum,
    base.reference,
    base.freeze,
    base.load,
    base.slot,
    base.memory,
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counts so supervisors can distinguish bounded work from a stalled child."""
    print(f"[exp8498] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(ref: Json, field: str, expected: Any, observed: Any, upstream: str = TASK) -> Json:
    """Failure operands name this invocation unless an original upstream task is supplied."""
    return dict(base.gate(ref, field, expected, observed, upstream))


@contextmanager
def binding() -> Iterator[None]:
    """Only invocation identity changes; shipped validators and historical authority stay intact."""
    from carnot.reporting.v730_capstone_report import build

    with (
        patch.multiple(
            base,
            NAME=NAME,
            TASK=TASK,
            MILESTONE=MILESTONE,
            CLI=CLI,
            TEST=TEST,
            OWNED=OWNED,
            methods=methods,
            progress=progress,
        ),
        patch.object(report, "build", build),
    ):
        yield


def outcome(task: Json, primary: Json, raw: Path, closure: Any = None) -> Json:
    """A failed producer remains visible without making its unqualified metrics scientific evidence."""
    number = int(task["id"][3:7])
    if number < 8485:
        return dict(history.outcome(task, primary, raw, closure))
    value = load(primary)
    if primary["sha256"] != "sha256:" + PINS[number]:
        raise ValueError("primary_pin")
    if value.get("blocked_at_layer") == "conductor_pre_gate":
        with patch.dict(base.PINS, PINS):
            return dict(base.outcome(task, primary, raw, closure))
    base.validate_primary(value, Path(primary["path"]))
    if value["task_id"] != task["id"]:
        raise ValueError("task_identity")
    terminal = load(reference(Path(value["terminal_validation_sidecar_path"])))
    side = load(reference(Path(terminal["publication"]["sidecar_path"])))
    sealed = (
        terminal["publication"]["primary_sha256"] == primary["sha256"] == side["primary_sha256"]
        and side["report"]["passed"] is True
    )
    reader = importlib.import_module("carnot." + READERS[number])
    replayed = bool(reader.replay(value if number == 8496 else Path(primary["snapshot_path"])))
    qualified = bool(
        sealed
        and replayed
        and value["required_checks_passed"]
        and not value["flagged_adversarial"]
        and value["verdict_class"] != "disqualified"
    )
    return dict(
        producer_executed=True,
        eligible=qualified,
        selected={k: v for k, v in value.items() if k in SELECT or k.endswith("_ready_score")},
        checks=[]
        if qualified
        else [
            gate(
                primary,
                "producer_terminal_replay",
                True,
                dict(
                    sealed=sealed,
                    replayed=replayed,
                    required_checks_passed=value["required_checks_passed"],
                ),
                task["id"],
            )
        ],
        disposition="qualified_producer" if qualified else "disqualified_producer",
        honest_verdict=value["honest_verdict"],
        verdict_class=value["verdict_class"],
        source_gates=value.get("gate_check_summary", []),
        historical_model_provenance=methods.history.model_provenance(value, primary),
        upstream_adversarial_findings=value.get("adversarial_findings", []),
        primitive_reference=value.get("primitive_reference", value.get("work_reference")),
        branch_replay=dict(
            status="replayed" if qualified else "disqualified_upstream_evidence", passed=replayed
        ),
    )


def worker_process(request: Path, output: Path) -> int:
    """Import before the replay baseline, while retaining total peak and growth memory bounds."""
    args = json.loads(request.read_bytes())
    number = int(args["task"]["id"][3:7])
    if number < 8485:
        return int(history.worker_process(request, output))
    if number in READERS:
        progress("reader_dependency_import_before")
        importlib.import_module("carnot." + READERS[number])
        progress("reader_dependency_import_after")
    with patch.multiple(workers, outcome=outcome, memory=memory, progress=progress):
        return int(workers.worker_process(request, output))


def invoke(task: Json, primary: Json, raw: Path) -> tuple[Json, Json]:
    """Use fresh bounded workers rather than accumulating historical replay memory in the parent."""
    with patch.multiple(workers, CLI=CLI, progress=progress):
        result, receipt = workers.invoke(task, primary, raw)
    return dict(result), dict(receipt)


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Only supplied CPU operands enter this invocation; unavailable devices never gate it."""
    start, before = time.monotonic_ns(), memory()
    progress("preconditions", 0, 14)
    authority = methods.authority(root, raw / "authority")
    checks = list(authority["gate_check_summary"])
    resource = dict(
        history.previous.resources(private),
        task_cap_s=4800,
        measurement_cap_s=3600,
        validation_reserve_s=900,
        optional_cuda_required=False,
        optional_boards_required=False,
    )
    if not (
        resource["disk_backed"]
        and resource["memory_available_bytes"] >= 1500 * 1024**2
        and resource["free_disk_bytes"] >= 1024**3
        and resource["tools_available"]
    ):
        checks.append(gate(dict(path=str(private)), "required_resources", True, resource))
    tasks = authority["tasks"] or yaml.safe_load((ROOT / methods.ACTIVE).read_bytes())["tasks"]
    if (
        [t["id"].split("-")[0] for t in tasks] != [f"exp{i}" for i in range(8485, 8499)]
        or tasks[-1].get("gated_on")
        or tasks[-1].get("requires")
    ):
        raise ValueError("exact_unconditional_fourteen_tasks")
    log = freeze(root / "ops/conductor-log.md", raw / "scheduling")
    lines = Path(log["snapshot_path"]).read_text().splitlines() if log["exists"] else []
    activation = next(
        (i for i, line in enumerate(lines) if "Milestone 2026.10.730 activated" in line), len(lines)
    )
    lines = lines[activation:]
    historical_tasks = []
    refs: list[Json] = [log]
    for family, inputs in methods.protocol()["historical_authorities"].items():
        sealed = [freeze(Path(r["path"]), raw / "historical_authorities" / family) for r in inputs]
        refs.extend(sealed)
        if any(
            r["sha256"] != original["sha256"] for r, original in zip(sealed, inputs, strict=True)
        ):
            checks.append(
                gate(
                    sealed[1],
                    "original_authority_pin",
                    inputs[1]["sha256"],
                    sealed[1]["sha256"],
                    family,
                )
            )
            continue
        historical_tasks += [
            t
            for t in yaml.safe_load(Path(sealed[1]["snapshot_path"]).read_bytes())["tasks"]
            if int(t["id"][3:7]) in [8432, 8435, 8436, 8437, 8464, 8465, 8466]
            and (root / t["deliverable"]).is_file()
        ]
    slots: list[Json] = []
    historical: list[Json] = []
    receipts: list[Json] = []
    roster = [*tasks[:-1], *historical_tasks]
    for index, task in enumerate(roster):
        progress("branch_before_" + task["id"], index, len(roster) - index)
        if (time.monotonic_ns() - start) / 1e9 >= 3600:
            raise TimeoutError("measurement_deadline")
        number = int(task["id"][3:7])
        candidates = list((root / "results").glob(f"experiment_{number}_*.json"))
        primary = freeze(
            candidates[0] if len(candidates) == 1 else root / task["deliverable"],
            raw / "primaries" / str(number),
        )
        refs.append(primary)
        summary = None
        if primary["exists"]:
            summary, receipt = invoke(task, primary, raw / "branches" / task["id"])
            receipts.append(receipt)
            refs.append(freeze(raw / "branches" / task["id"] / "summary.json", raw / "summaries"))
        cascade = next(
            (
                line
                for line in lines
                if "GATE_BLOCK" in line and (task["id"] in line or task["title"][:48] in line)
            ),
            None,
        )
        row = slot(task, primary, cascade, summary)
        (slots if number >= 8485 else historical).append(row)
        checks.extend(row["checks"])
        progress("branch_after_" + task["id"], index + 1, len(roster) - index - 1)
    refs.extend(
        freeze(ROOT / name, raw / "code")
        for name in [
            *OWNED,
            TEST,
            "python/carnot/reporting/primary_publication.py",
            "scripts/publication_gate.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "ops/exclusion_manifest.yaml",
            methods.PROTOCOL,
        ]
    )
    predecessors = [
        freeze(next((ROOT / "results").glob(f"experiment_{n}_*.json")), raw / "predecessors")
        for n in PREDECESSOR_PINS
    ]
    after = memory()
    work = dict(
        root=str(root),
        raw=str(raw),
        tasks=tasks,
        authority=authority,
        contract_digest=methods.TASKS_PIN,
        slots=slots,
        historical_slots=historical,
        refs=refs,
        predecessors=predecessors,
        branch_receipts=receipts,
        checks=checks,
        preconditions=resource,
        invocation_argv=list(__import__("sys").argv),
        memory=dict(
            before=before,
            after=after,
            peak_bound_mb=1500,
            growth_bound_mb=500,
            passed=after["peak_rss_mb"] <= 1500
            and after["current_rss_mb"] - before["current_rss_mb"] <= 500,
        ),
        phase_spans=[
            dict(
                name="sealed_branch_reduction",
                started_monotonic_ns=start,
                ended_monotonic_ns=time.monotonic_ns(),
            )
        ],
        started_monotonic_ns=start,
        ended_monotonic_ns=time.monotonic_ns(),
        adversarial_findings=[],
    )
    atomic_json(raw / "measurement.json", work)
    return work


def replay(path: Path) -> bool:
    """Independent task and historical pins reject substitutions even after local rehashing."""
    with patch.multiple(
        previous, methods=methods, PREDECESSOR_PINS=PREDECESSOR_PINS, invoke=invoke, binding=binding
    ):
        return bool(previous.replay(path))
