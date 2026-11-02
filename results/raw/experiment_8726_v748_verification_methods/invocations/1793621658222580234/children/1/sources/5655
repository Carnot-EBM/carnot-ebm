"""REQ-REPORT-8723: configure shared custody without rerunning historical producers."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

from carnot.reporting import v747_contract_methods as historical_methods
from carnot.reporting import v747_contract_report as historical_report
from carnot.reporting import v747_contract_runner as historical_runner
from carnot.reporting.current_work_receipt import canonical_hash


def namespace(source: ModuleType, name: str) -> Any:
    """Separate globals keep current bindings from changing historical consumers."""
    spec = spec_from_file_location("carnot.reporting._v748_" + name, source.__file__)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = namespace(historical_methods, "methods")
runtime = namespace(historical_runner, "runner")
report = namespace(historical_report, "report")
m.NAME, m.TASK, m.MILESTONE = (
    "experiment_8723_v748_collection_contract",
    "exp8723-collection-contract",
    "2026.11.748",
)
m.FIRST_ID, m.VERSION, m.PREVIOUS_MILESTONE = 8723, "748", "2026.11.747"
m.REQUIRE_PRIORS = False
m.PIN = "88a6e9a1c4822285e86c9b089047d4383f6e34d63430bdaff72c33beb979a07a"
m.CLI = f"scripts/experiments/{m.NAME}.py"
m.TESTS = ["tests/python/test_v748_collection_8723.py", "tests/python/test_v748_contract_8723.py"]
m.OWNED = ["python/carnot/reporting/collection_contract.py", __file__, m.CLI]
PRESERVED = dict(
    path=str(m.ROOT / "openspec/change-proposals/research-roadmap-v747-preserved-20261102.md"),
    sha256="sha256:4a62f587ca9d03a724945ddbb6e8f4d69995128dec7a015ce4bf738383c82109",
)
base_methods = m.methods
report.m = m
runtime.m = m


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Report actual boundaries so bounded children never appear silently stalled."""
    print(f"[exp8723] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(
    path: Path, field: str, expected: Any, observed: Any, upstream: str = m.TASK
) -> dict[str, Any]:
    """Current ownership must remain distinct from the retained producer's identity."""
    return dict(m.previous.gate(path, field, expected, observed, upstream))


def methods() -> dict[str, Any]:
    """Original numerical objectives and full future prompts are separate operands."""
    value: dict[str, Any] = base_methods()
    if m.original.checked(PRESERVED):
        value["references"].append(dict(PRESERVED))
    else:
        value["learning_authority_inputs_ready_score"] = 0
        value["gate_check_summary"].append(
            gate(
                Path(PRESERVED["path"]),
                "preserved_v747_design",
                PRESERVED["sha256"],
                m.e.sha256_file(Path(PRESERVED["path"]))
                if Path(PRESERVED["path"]).is_file()
                else None,
            )
        )
    tasks = m.authority(m.ROOT)["rows"]
    value["branch_method_bindings"] = [
        dict(
            task_id=row["unit_id"],
            task_object=row.get("task_object"),
            status="execution_eligibility_only",
            measurement_this_invocation=False,
        )
        for row in tasks
    ]
    value["learning_core_execution_owner"] = "exp8724-invocation-clock-core"
    value["v747_qualifiers"] = [
        "Exp8709 collection failed; skipped learning slots are scheduling evidence only.",
        "Exp8712 constructed lexical collisions do not prove natural extraction.",
        "Exp8715 120 versus 130 detections used unequal probe counts.",
        "Exp8717 batching 1.1382x; same-batch native/Python 1.0054x, below 10x.",
        "Exp8719 exact compiled signed32/24 parity establishes no speed.",
        "No causal CUDA repair, new ARC outcomes or physical acceleration established.",
    ]
    return value


def build(
    work: dict[str, Any], validation: dict[str, Any], raw: Path, output: Path
) -> dict[str, Any]:
    """Qualify administrative statements using the sealed current child union."""
    value: dict[str, Any] = report.build(work, validation, raw, output)
    checked = m.e.load(validation)
    nodes = checked["all_child_aggregate"].get("node_ids", [])
    effects = []
    for name in [
        "repeated_collection_and_sentinel",
        "inventory_rejected_without_partial_effects",
        "collection_collision",
        "failed_collection",
        "original_second_collect",
    ]:
        selected = [n for n in nodes if "::test_" + name in n]
        passed = bool(selected and checked["all_child_aggregate"]["passed"])
        effects.append(
            dict(
                check=name,
                node_ids=selected,
                passed=passed,
                status="completed" if passed else "unstarted",
            )
        )
    value.update(
        experiment_id=8723,
        random_seed=7488723,
        honest_verdict=value["honest_verdict"].replace(
            "v747_contract_methods", "v748_collection_contract"
        ),
        collection_effect_rows=effects,
        contract_mutation_rows=value["authority_mutation_rows"],
        preserved_historical_limits=value["preserved_historical_limits"]
        + work["methods_manifest"]["v747_qualifiers"],
        methodology_note="Validate the complete collection before allocating invocation-owned scratch. Bind fourteen complete V748 tasks, visible table and independent digest separately from original V744 scientific methods. Reuse pinned Exp8695 authority without historical producer execution. Retain current disjoint children, SQLite, JUnit, exits and clocks. Readiness grants execution eligibility only; Exp8724 owns the learning-core budget. Zero model loads; no learning benefit, generalization or physical acceleration claim.",
    )
    value["field_principles"].update(
        {
            k: "Bind administrative eligibility to sealed current checks; preserve scientific qualifiers."
            for k in value
        }
    )
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


m.progress, m.gate, m.methods = progress, gate, methods
runtime.report = SimpleNamespace(build=build)
main, replay = runtime.main, runtime.replay
