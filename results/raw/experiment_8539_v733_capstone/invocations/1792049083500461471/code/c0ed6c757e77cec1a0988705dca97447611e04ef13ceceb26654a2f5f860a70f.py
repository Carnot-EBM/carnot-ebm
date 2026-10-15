"""REQ-REPORT-8539: exact seals preserve failed and unavailable science."""

from __future__ import annotations

import importlib
from contextlib import nullcontext
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v726_capstone_evidence as base
from carnot.reporting import v732_capstone_evidence as history
from carnot.reporting import v733_contract_methods as methods
from carnot.reporting.board_operation_boundary_8454 import resources
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import validate_primary, read_bound_sidecar

Json = dict[str, Any]
ROOT = methods.ROOT
NAME, TASK, MILESTONE = "experiment_8539_v733_capstone", "exp8539-capstone", "2026.10.733"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v733_capstone_8539.py"
OWNED = [f"python/carnot/reporting/v733_capstone{s}.py" for s in ["_evidence", "_report", ""]] + [
    CLI
]
MODEL_SPECS: list[Json] = []
_OUTCOME = base.outcome
PINS = dict(
    zip(
        [8526, 8527, 8528, 8529, 8530, 8531, 8533, 8534, 8535, 8536, 8537, 8538],
        [
            "5185ba15b6e23c5a8185553fe4c878921255768bd7d12d88ebf16202d3f5c004",
            "81e5281bc9b115eb32e8327d0c7f3a92ff638903bbc2c6991d70f09d81c2204c",
            "a93b5a605a87f5b125898ffe19a13a59538abf405ec3f6abc67a3e44ae4e28ef",
            "14e26b0268b08c640909378420ee58d9d5c47c1ca91cad5c49a836c53c3de8a8",
            "45c3f74499f49051f4f5c16735a245811c119d432d8aa09057fa26d92138f2b1",
            "fd5d45a53587053187c24976eb41fd214e7eebdd78363915cf2340ec3d6fda92",
            "669dd406e8c381949d6a6fdb0fa25d6253559b5c55d4e7f3f4f35cef58d6d998",
            "061802848d1a8d5be7d92b7d9b5870f732c70effc8cf180e51965592a58f0ec9",
            "485b5b997c4f734a66463f1b48729cbf8920d92accc6e66f141b1c130704cd30",
            "a02db3b46baaed4ff39320bee359a8e905dfd9c809eb299592a4dcc90ce0294f",
            "a6f2f35bd7f96e046eaa2f08c1b141e149cb19c0bb4494da40c959c0ae1f83f3",
            "ab5d6cadea6a98f92746403efb46edc49661d9c556016aba17e715ba856952eb",
        ],
        strict=True,
    )
)
PREGATE_PATHS = {
    n: f"results/experiment_{n}_{s}.json"
    for n, s in [
        (8531, "causal_brier_learning"),
        (8534, "nine_input_inference_cost"),
        (8535, "bounded_qwen_evidence"),
    ]
}
READERS = dict(
    zip(
        [8526, 8527, 8528, 8529, 8530, 8533, 8536, 8537, 8538],
        [
            "reporting.v733_contract_methods",
            "reporting.v733_source_consumer_qualification",
            "verify.v733_identity_brier_fit",
            "reporting.v733_incremental_source_audit",
            "reporting.v733_update_accounting",
            "reporting.nine_input_qualification_8533",
            "reporting.arc_capability_qualification_8536",
            "reporting.arc_capability_panel_8537",
            "reporting.nine_input_board_boundary_8538",
        ],
        strict=True,
    )
)
SELECT = history.SELECT | {
    "primary_contrast_rows",
    "primary_contrasts",
    "scientific_verdict_class",
    "per_seed_metrics",
    "per_stratum_metrics",
    "typed_cost_deltas",
    "negative_control_degeneracy",
    "max_probability_error",
    "max_residual_error",
    "typed_decision_mismatch_count",
    "finite_parity_rows",
    "boundary_fallback_rows",
    "complete_request_rows",
    "numerical_scope",
    "disabled_attempt_counts",
    "resolved_redirects",
    "capability_exclusion_rows",
    "paired_results",
    "action_costs",
    "useful_efficiency",
    "zero_progress_headroom",
    "per_episode_rows",
    "applied_update_events",
    "changed_coordinate_count",
    "metric_reconciliation_rows",
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
    """Flush real work counts so a slow child remains observable."""
    print(f"[exp8539] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(ref: Json, field: str, expected: Any, observed: Any, upstream: str = TASK) -> Json:
    """Missing evidence carries its exact path and differs from a measured zero."""
    return dict(base.gate(ref, field, expected, observed, upstream))


def outcome(task: Json, primary: Json, raw: Path, closure: Any = None) -> Json:
    """Replay never repairs a failed producer or transfers fixture success to science."""
    number = int(task["id"][3:7])
    if primary["sha256"] != "sha256:" + PINS[number]:
        raise ValueError("primary_pin")
    value = load(primary)
    if value.get("blocked_at_layer") == "conductor_pre_gate":
        with patch.dict(base.PINS, PINS):
            result = dict(_OUTCOME(task, primary, raw, closure))
        if value["experiment"] != number:
            raise ValueError("pre_gate_identity")
        for g in value["gates_evaluated"]:
            if sha256_file(Path(g["artifact_path"])) != g["artifact_sha256"]:
                raise ValueError("pre_gate_operand_hash")
    else:
        validate_primary(value, Path(primary["path"]))
        if value["task_id"] != task["id"]:
            raise ValueError("task_identity")
        try:
            terminal = load(reference(Path(value["terminal_validation_sidecar_path"])))
            publication = terminal.get("publication", terminal)
            side = read_bound_sidecar(Path(primary["path"]), Path(publication["sidecar_path"]))
            sealed = (
                publication["primary_sha256"] == primary["sha256"] == side["primary_sha256"]
                and side["report"]["passed"] is True
            )
        except (OSError, ValueError, KeyError):
            sealed = False
        reader = importlib.import_module("carnot." + READERS[number])
        with (
            patch.object(reader, "checked", history.native_json)
            if number == 8537
            else nullcontext()
        ):
            replayed = bool(
                reader.replay(value if number in {8536, 8537} else Path(primary["snapshot_path"]))
            )
        qualified = bool(
            sealed
            and replayed
            and value["required_checks_passed"]
            and not value["flagged_adversarial"]
            and value["verdict_class"] != "disqualified"
        )
        result = dict(
            producer_executed=True,
            eligible=qualified,
            selected={k: v for k, v in value.items() if k in SELECT or k.endswith("_score")},
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
                        flagged_adversarial=value["flagged_adversarial"],
                        verdict_class=value["verdict_class"],
                    ),
                    task["id"],
                )
            ],
            disposition="qualified_producer" if qualified else "disqualified_producer",
            honest_verdict=value["honest_verdict"],
            verdict_class=value["verdict_class"],
            source_gates=value.get("gate_check_summary", []),
            historical_model_provenance=value.get("historical_model_provenance", []),
            upstream_adversarial_findings=value.get("adversarial_findings", []),
            upstream_failed_receipts=[r for r in value["validation_receipts"] if not r["passed"]],
            primitive_reference=value.get("primitive_reference", value.get("work_reference")),
            branch_replay=dict(
                status="replayed" if qualified else "disqualified_upstream_evidence",
                passed=replayed,
            ),
        )
    result.update(
        original_run_date=value.get("run_date"),
        upstream_flagged_adversarial=value.get("flagged_adversarial"),
        original_required_checks_passed=value.get("required_checks_passed"),
    )
    return result


def worker_process(request: Path, output: Path) -> int:
    """Shipped fresh-process workers bound replay memory after dependency imports."""
    with patch.multiple(base, READERS=READERS, outcome=outcome, progress=progress):
        return int(base.worker_process(request, output))


def invoke(task: Json, primary: Json, raw: Path) -> tuple[Json, Json]:
    """Each branch has a bounded child and immutable command and stream receipts."""
    with patch.multiple(base.workers, CLI=CLI, progress=progress):
        result, receipt = base.workers.invoke(task, primary, raw)
    return dict(result), dict(receipt)


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Authenticate exact paths; optional CUDA and boards never gate independent CPU work."""
    start, before = time.monotonic_ns(), memory()
    progress("preconditions", 0, 14)
    authority = methods.authority(root, raw / "authority")
    tasks = authority["tasks"] or methods.protocol()["task_map_v733"]
    if (
        methods.tasks_digest(tasks) != methods.TASKS_PIN
        or tasks[-1].get("requires")
        or tasks[-1].get("gated_on")
    ):
        raise ValueError("exact_unconditional_fourteen_tasks")
    resource = dict(
        resources(private), optional_cuda_required=False, optional_boards_required=False
    )
    checks = list(authority["gate_check_summary"])
    if not (
        resource["disk_backed"]
        and resource["memory_available_bytes"] >= 1500 * 1024**2
        and resource["free_disk_bytes"] >= 1024**3
        and resource["tools_available"]
    ):
        checks.append(gate(dict(path=str(private)), "required_resources", True, resource))
    log = freeze(root / "ops/conductor-log.md", raw / "scheduling")
    lines = Path(log["snapshot_path"]).read_text().splitlines() if log["exists"] else []
    activation = next(
        (i for i, l in enumerate(lines) if "Milestone 2026.10.733 activated" in l), len(lines)
    )
    lines = lines[activation:]
    slots, receipts, refs = [], [], [log]
    for index, task in enumerate(tasks[:-1]):
        progress("branch_before_" + task["id"], index, 13 - index)
        if (time.monotonic_ns() - start) / 1e9 >= 3600:
            raise TimeoutError("measurement_deadline")
        number = int(task["id"][3:7])
        primary = freeze(root / task["deliverable"], raw / "primaries" / str(number))
        refs.append(primary)
        if not primary["exists"] and number in PREGATE_PATHS:
            path = root / PREGATE_PATHS[number]
            if path.is_file():
                primary = freeze(path, raw / "pregates" / str(number))
                refs.append(primary)
        summary = None
        if primary["exists"]:
            summary, receipt = invoke(task, primary, raw / "branches" / task["id"])
            receipts.append(receipt)
            refs.append(freeze(raw / "branches" / task["id"] / "summary.json", raw / "summaries"))
        cascade = next(
            (
                l
                for l in lines
                if "GATE_BLOCK" in l and l.split("|")[2].strip() == task["title"][:50].strip()
            ),
            None,
        )
        row = slot(task, primary, cascade, summary)
        slots.append(row)
        checks.extend(row["checks"])
        progress("branch_after_" + task["id"], index + 1, 12 - index)
    original = methods.protocol()["v732_capstone_reference"]
    predecessor = freeze(Path(original["path"]), raw / "predecessors")
    if predecessor["sha256"] != original["sha256"]:
        raise ValueError("original_v732_pin")
    refs.extend(
        freeze(ROOT / name, raw / "code")
        for name in [
            *OWNED,
            TEST,
            methods.PROTOCOL,
            "python/carnot/reporting/primary_publication.py",
            "scripts/publication_gate.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "ops/exclusion_manifest.yaml",
            "docs/research-notes/v733-capstone.md",
        ]
    )
    after, end = memory(), time.monotonic_ns()
    work = dict(
        root=str(root),
        raw=str(raw),
        tasks=tasks,
        authority=authority,
        contract_digest=methods.TASKS_PIN,
        slots=slots,
        refs=refs,
        predecessors=[predecessor],
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
            dict(name="sealed_branch_reduction", started_monotonic_ns=start, ended_monotonic_ns=end)
        ],
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        adversarial_findings=[],
    )
    atomic_json(raw / "measurement.json", work)
    return work


def replay(path: Path) -> bool:
    """Rebuild authority and primitive rows so repaired local hashes cannot change results."""
    from carnot.reporting.v733_capstone_report import build

    try:
        value = json.loads(path.read_bytes())
        work = load(value["primitive_reference"])
        if (
            value["reproducibility_checksum"] != checksum(value)
            or value["task_count"] != 14
            or methods.tasks_digest(work["tasks"]) != methods.TASKS_PIN
            or work["contract_digest"] != methods.TASKS_PIN
        ):
            return False
        if (
            work["predecessors"][0]["sha256"]
            != methods.protocol()["v732_capstone_reference"]["sha256"]
        ):
            return False
        for ref in [*work["refs"], *work["predecessors"], *work.get("coverage_shards", [])]:
            if (
                ref.get("exists", True)
                and sha256_file(Path(ref.get("snapshot_path", ref["path"]))) != ref["sha256"]
            ):
                return False
        with TemporaryDirectory(prefix="exp8539-authority-", dir="/var/tmp") as directory:
            cold = Path(directory)
            for role, name in [
                ("design", methods.DESIGN),
                ("active", methods.ACTIVE),
                ("staged", methods.STAGED),
            ]:
                ref = work["authority"]["authority_snapshots"][role]
                if ref["exists"]:
                    if sha256_file(Path(ref["snapshot_path"])) != ref["sha256"]:
                        return False
                    target = cold / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ref["snapshot_path"], target)
            actual = methods.authority(cold, cold / "checked")
            if (
                actual["activated"] != work["authority"]["activated"]
                or actual["tasks"] != work["authority"]["tasks"]
            ):
                return False
        for row in work["slots"]:
            if row["task"] != work["tasks"][row["experiment_id"] - 8526]:
                return False
            if row["primary_reference"]["exists"]:
                with TemporaryDirectory(prefix="exp8539-cold-", dir="/var/tmp") as directory:
                    actual, receipt = invoke(row["task"], row["primary_reference"], Path(directory))
                if not receipt["passed"] or any(
                    actual[k] != row[k]
                    for k in actual
                    if k not in {"memory", "reader_initialization"}
                ):
                    return False
            elif row != slot(row["task"], row["primary_reference"], row["cascade_log_line"]):
                return False
        for receipt in [*work["branch_receipts"], *value["validation_receipts"]]:
            for stream in ["stdout", "stderr"]:
                if sha256_file(Path(receipt[stream + "_path"])) != receipt[stream + "_sha256"]:
                    return False
        return bool(
            value
            == build(
                work,
                value["validation_receipts"],
                Path(work["raw"]),
                Path(value["publication_output"]),
            )
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
