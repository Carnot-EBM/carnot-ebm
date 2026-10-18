"""REQ-REPORT-8240: bind unchanged causal learning to qualified execution.

Execution readiness describes auditable delayed decisions and recovery. Benefit
on these exposed sources is reserved for the next audit, with retention sealed.
"""

from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import delayed_utility_execution_8225 as legacy
from carnot.verify import learning_validation_8235 as validation

Json = dict[str, Any]
ROOT = legacy.ROOT
NAME = "experiment_8240_v712_qualified_delayed_learning"
TASK = "exp8240-qualified-delayed-learning"
MODULE = "python/carnot/verify/qualified_delayed_learning_8240.py"
RUNNER = MODULE
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_qualified_delayed_learning_8240.py"
OWNED = [MODULE, CLI]
RUN_DATE = "20261007"
MODEL_SPECS: list[Json] = []
QUALIFICATION = "results/" + validation.NAME + ".json"
QUALIFICATION_HASH = "sha256:673280cc8b2c57d29e143a838faee12c5c88d4198c5d30d63e64d2aca7baa727"
BINDING_HASH = "sha256:0893e612444afdf9a2451406ee181575ece0077a7b0559acf99af6c8a5037aca"
BASE_AUTH, BASE_MEASURE = legacy.authenticate, legacy.measure
BASE_BUILD, BASE_REPLAY, BASE_MANIFEST = legacy.build, legacy.replay, legacy.manifest
run_check = legacy.run_check


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts distinguish bounded cached work from a stalled process."""
    print(f"[exp8240] phase={phase} completed={completed} pending={pending}", flush=True)


def bindings() -> ExitStack:
    """Restore upstream names after each use so historical producers retain identity."""
    stack = ExitStack()
    for name in ["NAME", "TASK", "CLI", "TEST", "OWNED", "RUN_DATE", "progress"]:
        stack.enter_context(patch.object(legacy, name, globals()[name]))
    return stack


def qualification(root: Path, raw: Path, work: Json) -> None:
    """Reject unqualified or altered execution before any natural label is released."""
    gate, bind = legacy.k.frozen.gate, legacy.k.frozen.bind
    path = root / QUALIFICATION
    gate(work, path, "exists", True, True if path.is_file() else None)
    value = json.loads(path.read_bytes())
    gate(work, path, "input_schema", True, isinstance(value, dict))
    for field, expected in [
        ("experiment_id", 8235),
        ("task_id", validation.TASK),
        ("learning_execution_ready_score", 1),
        ("required_checks_passed", True),
        ("flagged_adversarial", False),
        ("verdict_class", "null"),
    ]:
        gate(work, path, field, expected, value.get(field))
    bind(work, path, QUALIFICATION_HASH, raw)
    bind(work, root / validation.BINDINGS, BINDING_HASH, raw)
    binding = json.loads((root / validation.BINDINGS).read_bytes())
    gate(
        work,
        path,
        "binding.natural_consumer",
        TASK,
        binding["task_id_mappings"]["natural_consumer"],
    )
    for name, digest in binding["protocol_hashes"].items():
        bind(work, root / name, digest, raw)
    bind(work, root / validation.UPSTREAM, validation.ORIGINAL_SHA256, raw)
    terminal = Path(value["terminal_validation_sidecar_path"])
    bind(work, terminal, sha256_file(terminal), raw)
    side = json.loads(terminal.read_bytes())
    receipt = Path(side["publication"]["sidecar_path"])
    gate(
        work,
        path,
        "terminal.report.passed",
        True,
        read_bound_sidecar(path, receipt)["report"]["passed"],
    )
    bind(work, receipt, sha256_file(receipt), raw)
    for ref in value["code_config_hashes"] + value["raw_shard_hashes"]:
        bind(work, Path(ref["path"]), ref["sha256"], raw)
    for receipt in value["validation_receipts"]:
        gate(work, path, "validation." + receipt["name"], True, receipt["passed"])
        for prefix in ["stdout", "stderr"]:
            bind(work, Path(receipt[prefix + "_path"]), receipt[prefix + "_sha256"], raw)
    work["execution_qualification"] = dict(
        path=str(path),
        sha256=QUALIFICATION_HASH,
        learning_execution_ready_score=1,
        binding_sha256=BINDING_HASH,
        historical_primary_sha256=validation.ORIGINAL_SHA256,
    )


def authenticate(root: Path, raw: Path, work: Json) -> Json:
    """The new execution permission supplements the original scientific permissions."""
    qualification(root, raw, work)
    return dict(BASE_AUTH(root, raw, work))


def coefficient_reads(model: Json) -> int:
    """Count scalar coefficient reads in the portable tree, excluding hardware traffic."""
    if model["kind"] == "input":
        return 0
    if model["kind"] == "global":
        return 2
    if model["kind"] == "patch":
        return coefficient_reads(model["base"]) + len(model["patches"])
    return 1 + coefficient_reads(model["base"]) + coefficient_reads(model["candidate"])


def storage_costs(work: Json, raw: Path) -> list[Json]:
    """Measure durable file bytes and logical reads without claiming accelerator execution."""
    rows = []
    for states in work["states"]:
        for arm in legacy.k.ARMS:
            state = states[arm]
            directory = raw / f"seed-{state['seed']}" / arm
            models = {canonical_hash(dict(kind="input")): dict(kind="input")}
            models.update(
                {
                    e["after_hash"]: e["mixture"]
                    for e in state["events"]
                    if e["kind"] == "admit_once"
                }
            )
            rows.append(
                dict(
                    seed=state["seed"],
                    arm=arm,
                    path=str(directory),
                    state_bytes=(directory / "final.json").stat().st_size,
                    durable_write_bytes=sum(
                        p.stat().st_size for p in directory.iterdir() if p.is_file()
                    ),
                    issue_coefficient_touches=sum(
                        coefficient_reads(models[r["model_sha256"]])
                        for r in state["issued"]
                        if r["p"] is not None
                    ),
                    final_lookup_coefficient_touches=coefficient_reads(state["model"]),
                    update_coefficient_touches=None,
                    update_touch_status="unmeasured_optimizer_internal_reads",
                    substrate="python_cpu",
                    rust_execution_measured=False,
                    fpga_execution_measured=False,
                )
            )
    return rows


def measure(root: Path, raw: Path, **kwargs: Any) -> Json:
    """Run the same learner under current identity and archive actual storage costs."""
    with bindings(), patch.object(legacy, "authenticate", authenticate):
        work = dict(BASE_MEASURE(root, raw, **kwargs))
    work["storage_cost_rows"] = storage_costs(work, raw)
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """A completed trajectory earns readiness while benefit remains reserved for Exp8241."""
    with bindings():
        value = dict(BASE_BUILD(work, raw, receipts, fixture=fixture))
    value.update(
        experiment_id=8240,
        milestone="2026.10.712",
        ops_docs_updated=False,
        reconciliation_note="The conductor owns ops/status/changelog/traceability reconciliation.",
        hardware_cost_scope="Measured Python CPU lookup/update time and durable bytes; logical portable coefficient reads. Rust/FPGA execution and optimizer internal traffic remain unmeasured.",
    )
    value["honest_verdict"] = value["honest_verdict"].replace("8226", "8241")
    value["acceptance_gates"]["benefit"] = "H2 and retention verdicts reserved for Exp8241"
    value["field_principles"]["retention_predictions_path"] = (
        "Final predictions sealed before retention labels become available in Exp8241."
    )
    value["field_principles"].update(
        {
            key: "Describe current measured work separately from unmeasured benefit and hardware costs."
            for key in value
            if key not in value["field_principles"]
        }
    )
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Reexecute primitives and rebuild the current report, including its qualification."""
    try:
        value = json.loads(path.read_bytes())
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        work = json.loads((raw / "measurement.json").read_bytes())
        if storage_costs(work, raw) != work["storage_cost_rows"]:
            return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    with bindings(), patch.object(legacy, "build", build):
        return bool(BASE_REPLAY(path))


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze new code coverage and both required private recovery and audit E2E checks."""
    with bindings():
        specs = dict(BASE_MANIFEST(private, candidate))
    for spec in specs["commands"]:
        if spec["name"] == "consumer_and_E2E015_019":
            spec["argv"].insert(-1, "tests/python/test_restricted_decision_audit_8210.py")
    for spec in specs["terminal_commands"]:
        if spec["name"] == "cold_replay":
            spec["deadline_s"] = 180
    return specs


def main(argv: list[str] | None = None) -> int:
    """Share real seed workers with the unchanged orchestration and atomic publisher."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--seed-input" in args:
        with bindings():
            return int(legacy.main(args))
    execution = legacy.execution
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
        patch.object(execution, "manifest", manifest),
        patch.object(execution, "run_check", run_check),
    ):
        return int(execution.main(args))
