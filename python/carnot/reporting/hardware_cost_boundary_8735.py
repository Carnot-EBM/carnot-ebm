"""REQ-REPORT-8735: reuse sealed readers without changing historical controllers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from carnot.reporting import hardware_operation_boundary_8721 as original_report
from carnot.reporting import hardware_operation_inputs_8721 as original_inputs
from carnot.reporting import hardware_operation_runner_8721 as original_runner
from carnot.reporting import hardware_operation_runner_8707 as original_checks
from carnot.reporting import v748_collection_contract as contract
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
e = contract.namespace(original_report, "hardware_cost_report")
i = contract.namespace(original_inputs, "hardware_cost_inputs")
r = contract.namespace(original_runner, "hardware_cost_runner")
r.retained = contract.namespace(original_checks, "hardware_cost_checks")
e.NAME, e.TASK = "experiment_8735_v748_hardware_cost_boundary", "exp8735-hardware-cost-boundary"
e.CLI = f"scripts/experiments/{e.NAME}.py"
e.TESTS = ["tests/python/test_hardware_cost_boundary_8735.py"]
e.OWNED = [
    "python/carnot/reporting/hardware_cost_boundary_8735.py",
    "python/carnot/reporting/hardware_cost_rows_8735.py",
    e.CLI,
]
i.contract, i.MILESTONE = contract.m, contract.m.MILESTONE
i.BRANCHES = [
    (8727, "learning_cost_ready_score"),
    (8732, "update_cost_ready_score"),
    (8733, "installed_attribution_ready_score"),
]
i.PRIOR = "results/experiment_8721_v747_hardware_operation_boundary.json"
i.PINS = dict(
    i.PINS, **{i.PRIOR: "b0ae72a7347ca763362f4cb80158326cf95a855781f04093696e0e3b693c8cd6"}
)
i.CURRENT_PINS = {
    8732: "063218e46ed7d2ad36a3bf3e4282ffde79a9888b7cdbacfcaf3178132157e6e3",
    8733: "955b3ccac700d28094d896f458ca9c8aea44b73f25917f76ad83b2529fe99b44",
}
BASE_AUTHENTICATE, BASE_BUILD = i.authenticate, e.build


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase counts so the parent can distinguish evidence reads from stalls."""
    print(f"[exp8735] phase={phase} completed={completed} pending={pending}", flush=True)


def producer_operands(value: Json, path: Path) -> list[Json]:
    """Producer reconstruction supplies authority; a ready scalar cannot replace it."""
    refs = value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    refs += [
        value["primitive_reference"],
        i.reference(Path(value["terminal_validation_sidecar_path"])),
    ]
    refs = [i.versioned_reference(ref, refs) for ref in refs]
    final = i.load(refs[-1])
    if (
        final.get("passed", True) is not True
        or final["publication"]["primary_sha256"] != i.reference(path)["sha256"]
    ):
        raise ValueError("producer_final_identity")
    progress("producer_replay_before", 0, 1)
    if value["experiment_id"] == 8733:
        from carnot.reporting.installed_cost_attribution_8733 import replay

        if not replay(path, value["primitive_reference"]):
            raise ValueError("original_producer_replay")
    elif value["experiment_id"] == 8732:
        from carnot.reporting.resident_update_runner_8732 import replay as resident_replay

        resident_replay(path)
    else:
        raise ValueError("unregistered_learning_producer_identity")
    progress("producer_replay_after", 1, 0)
    return list(refs)


def authenticate(root: Path) -> Json:
    """Keep current task authority separate from retained scientific method bytes."""
    evidence: Json = BASE_AUTHENTICATE(root)
    for expected in [
        i.reference(e.ROOT / "docs/research-notes/v748-hardware-cost-boundary.md"),
        dict(contract.PRESERVED),
    ]:
        observed = i.reference(Path(expected["path"]))
        evidence["refs"].append(observed)
        if observed != expected:
            evidence["failures"].append(
                contract.gate(
                    Path(expected["path"]),
                    "original_sha256",
                    expected["sha256"],
                    observed["sha256"],
                    e.TASK,
                )
            )
            evidence["board_qualified"] = False
    return evidence


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Retain all independently admitted costs while physical execution stays absent."""
    from carnot.reporting.hardware_cost_rows_8735 import cost_rows

    value: Json = BASE_BUILD(work, receipts, output)
    value.update(
        experiment_id=8735,
        task_id=e.TASK,
        milestone=i.MILESTONE,
        random_seed=7488735,
        honest_verdict=f"complete_{value['verdict_class']}_hardware_cost_or_physical_execution",
        device_commands_executed=[],
        task_contract_rows=work["evidence"]["authority"]["rows"],
        canonical_tasks_sha256=contract.m.PIN,
        original_scientific_method_reference=next(
            (ref for ref in work["evidence"]["refs"] if ref["path"].endswith(i.previous.PROTOCOL)),
            None,
        ),
        model_invocation_counts=dict(
            value["model_invocation_counts"], language_model_calls=0, model_loads=0
        ),
        ideal_required_compatible_fraction={"10x": 0.9, "100x": 0.99},
        acquisition_priority_delta=dict(
            changed=False, reason="No new evidence-backed acquisition prerequisite."
        ),
        preserved_historical_limits=[
            "V747 ran nine producers and skipped five studies; Exp8709 collection failed before qualified learning.",
            "Exp8712 constructed lexical collisions are not natural extraction; Exp8715 used unequal probe counts.",
            "Exp8717 batching gain1.138x; same-batch native/Python1.005x, below10x.",
            "Exp8719 signed32/24 parity is not speed. No CUDA repair, new ARC outcome or physical acceleration.",
        ],
        methodology_note="Read-only aggregation of independently authenticated producer artifacts. Constructed resident updates, inference-only attribution and causal learning use separate denominators. No model load, generation, benchmark or device command. Only compatible disjoint measured operation time permits a conditional ideal accelerator ceiling; host and transfer costs remain in their own measured parents.",
    )
    value.update(cost_rows(work["evidence"]["branches"], value["operation_support_rows"]))
    value.pop("reproducibility_checksum")
    value["field_principles"] = {
        k: "Exact sealed evidence; absent costs and compatibility remain unknown. Mapping and physical readiness are separate."
        for k in [*value, "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


i.producer_operands, i.authenticate = producer_operands, authenticate
e.inputs, e.build, e.progress = i, build, progress
r.e, r.i = e, i
r.BASE_VALIDATE, r.BASE_REPLAY = r.retained.validate, r.retained.replay
r.BASE_TERMINAL, r.BASE_RUN = r.retained.terminal, r.retained.run
main = r.main
