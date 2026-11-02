"""REQ-REPORT-8734: keep qualified reader methods separate from current task authority."""

from pathlib import Path
from typing import Any

from carnot.reporting import arc_supervisor_frontier_8720 as original
from carnot.reporting import arc_supervisor_runner_8720 as original_runtime
from carnot.reporting import v748_collection_contract as contract

Json = dict[str, Any]
e = contract.namespace(original, "arc_outcome_report")
runtime = contract.namespace(original_runtime, "arc_outcome_runtime")
retained = contract.namespace(original_runtime.retained, "arc_outcome_retained")
e.original = contract.namespace(original.original, "arc_outcome_gate")
e.NAME, e.TASK, e.MILESTONE = (
    "experiment_8734_v748_arc_outcome_frontier",
    "exp8734-arc-outcome-frontier",
    "2026.11.748",
)
e.CLI = f"scripts/experiments/{e.NAME}.py"
e.TESTS = ["tests/python/test_arc_outcome_frontier_8734.py"]
e.OWNED = ["python/carnot/reporting/arc_outcome_frontier_8734.py", e.CLI]
FRONTIER = "results/experiment_8720_v747_arc_supervisor_frontier.json"
FRONTIER_PIN = "sha256:bcef7715e5181dd49b89cf019a184465783803a6ff7e51b3d5a79bb182a308f7"
e.UPSTREAM, e.PIN, e.authority = FRONTIER, FRONTIER_PIN, contract.m
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Visible phase boundaries distinguish a short reader from a stalled child."""
    print(f"[exp8734] phase={phase} completed={completed} pending={pending}", flush=True)


def observe(path: Path, upstream: str, field: str, pin: str) -> Json:
    """The current reader requires frontier qualification, independently of other branches."""
    return dict(
        original.original.observe(
            path, "exp8720-arc-supervisor-frontier", "arc_frontier_ready_score", pin
        )
    )


def authenticate() -> tuple[Json, list[Json]]:
    """Reuse original success and rejection checks only at unchanged source identities."""
    original.progress("registry_precheck_before")
    registry = e.ROOT / "ops/arc_solve_registry.yaml"
    registry_ref = e.reference(registry)
    previous, refs = original.qualified(e.ROOT / FRONTIER, FRONTIER_PIN)
    upstream, original_refs = original.authenticate()
    baseline = original.read(dict(path=str(e.ROOT / e.BASELINE), sha256=e.LEDGER_PIN))
    if not (
        previous["arc_frontier_ready_score"] == 1
        and previous["frontier_after"] == original.frontier(upstream, baseline["entries"])
        and previous["registry_precheck"]["sha256"] == registry_ref["sha256"]
        and all(
            e.sha256_file(Path(path)) == digest
            for path, digest in previous["code_config_hashes"].items()
        )
    ):
        raise ValueError("retained_frontier_identity")
    refs += original_refs + [registry_ref, previous["primitive_reference"]]
    refs += [
        dict(path=path, sha256=digest) for path, digest in previous["code_config_hashes"].items()
    ]
    refs += [e.reference(e.ROOT / "docs/research-notes/v748-arc-outcome-frontier.md")]
    progress("registry_and_high_water_authenticated", 14, 0)
    return dict(upstream), refs


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Reader readiness cannot manufacture a solve or a recommendation from an empty delta."""
    value: Json = e_base_build(work, receipts, output)
    value.update(
        experiment_id=8734,
        milestone=e.MILESTONE,
        random_seed=7488734,
        solve_provenance=None,
        solve_provenance_scope="reader_credits_no_live_outcomes",
        policy_decision="no_supported_policy_change",
        standing_generalization_floor_satisfied=not work["failures"],
        retained_frontier_reference=dict(path=str(e.ROOT / FRONTIER), sha256=FRONTIER_PIN),
        evidence_reuse_scope="unchanged_Exp8720_reader_sources_tests_inputs_and_methods",
    )
    value["field_principles"] = {
        key: "Exact sealed inputs and current invocation checks bind this field; empty deltas establish no policy benefit or solve credit."
        for key in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = e.checksum(value)
    return value


e_base_build = e.build
e.build, e.authenticate, e.progress = build, authenticate, progress
e.original.observe = observe
runtime.e, runtime.retained = e, retained
runtime.retained_validate, runtime.retained_terminal = retained.validate, retained.terminal
main, replay = runtime.main, runtime.replay
