"""REQ-REPORT-8746: reuse a sealed reader without reopening the closed panel."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from carnot.reporting import arc_outcome_frontier_8734 as previous
from carnot.reporting import v748_collection_contract as namespace_owner
from carnot.reporting import v749_contract_methods as methods

Json = dict[str, Any]
legacy = previous.original
e = namespace_owner.namespace(legacy, "arc8746_report")
runtime = namespace_owner.namespace(previous.original_runtime, "arc8746_runtime")
retained = namespace_owner.namespace(previous.original_runtime.retained, "arc8746_retained")
e.original = namespace_owner.namespace(legacy.original, "arc8746_gate")
e.NAME, e.TASK, e.MILESTONE = (
    "experiment_8746_v749_arc_outcome_frontier",
    "exp8746-arc-outcome-frontier",
    "2026.11.749",
)
e.CLI = f"scripts/experiments/{e.NAME}.py"
e.TESTS = ["tests/python/test_arc_outcome_frontier_8746.py"]
e.OWNED = ["python/carnot/reporting/arc_outcome_frontier_8746.py", e.CLI]
FRONTIER = "results/experiment_8734_v748_arc_outcome_frontier.json"
FRONTIER_PIN = "sha256:38b02bfcd3421c9f5cc420862fdd035d33fa78a8c671092afb81e3622ed1d331"
e.UPSTREAM, e.PIN = FRONTIER, FRONTIER_PIN
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts so the parent can distinguish a short reader from a stalled child."""
    print(f"[exp8746] phase={phase} completed={completed} pending={pending}", flush=True)


def authority(root: Path) -> Json:
    """Current task authority must not depend on another branch's readiness."""
    value: Json = methods.authority(root)
    value["gate_check_summary"] = (
        []
        if value["ready"]
        else [
            dict(
                upstream="v749_task_contract",
                path=str(root / methods.DESIGN),
                sha256=e.sha256_file(root / methods.DESIGN)
                if (root / methods.DESIGN).is_file()
                else None,
                check="full_task_contract",
                field="ready",
                operator="==",
                expected=True,
                observed=False,
                passed=False,
            )
        ]
    )
    return value


def observe(path: Path, upstream: str, field: str, pin: str) -> Json:
    """A retained reader must pass its own scalar gate before its receipts are interpreted."""
    return dict(
        legacy.original.observe(
            path, "exp8734-arc-outcome-frontier", "arc_frontier_ready_score", pin
        )
    )


def authenticate() -> tuple[Json, list[Json]]:
    """Exact prior bytes bind the high-water mark before any new outcome is opened."""
    progress("registry_precheck_before")
    registry = e.ROOT / "ops/arc_solve_registry.yaml"
    registry_ref = e.reference(registry)
    prior, refs = legacy.qualified(e.ROOT / FRONTIER, FRONTIER_PIN)
    upstream, original_refs = previous.authenticate()
    baseline = e.read(dict(path=str(e.ROOT / e.BASELINE), sha256=e.LEDGER_PIN))
    if not (
        type(prior["arc_frontier_ready_score"]) is int
        and prior["arc_frontier_ready_score"] == 1
        and prior["frontier_after"] == legacy.frontier(upstream, baseline["entries"])
        and prior["registry_precheck"]["sha256"] == registry_ref["sha256"]
        and all(e.sha256_file(Path(p)) == h for p, h in prior["code_config_hashes"].items())
    ):
        raise ValueError("retained_frontier_identity")
    refs += original_refs + [registry_ref, prior["primitive_reference"]]
    refs += [dict(path=p, sha256=h) for p, h in prior["code_config_hashes"].items()]
    refs += [
        e.reference(e.ROOT / "docs/research-notes/v748-arc-outcome-frontier.md"),
        e.reference(e.ROOT / "docs/research-notes/v749-arc-outcome-frontier.md"),
        e.reference(
            e.ROOT / "openspec/change-proposals/research-roadmap-v748-preserved-20261102.md"
        ),
    ]
    precheck = []
    for game in yaml.safe_load(registry.read_bytes())["games"]:
        path = e.ROOT / "results" / f"arc_loop_solve_{game['game']}.json"
        loop = e.read(e.reference(path)) if path.is_file() else {}
        if path.is_file():
            refs.append(e.reference(path))
        precheck.append(
            dict(
                game=game["game"],
                levels_reproduced=game.get("levels_reproduced"),
                arc_loop_solve_reference=e.reference(path) if path.is_file() else None,
                loop_operand_state="observed_value" if path.is_file() else "missing_file",
                loop_solve_provenance=loop.get("solve_provenance"),
                loop_reproduced_levels=loop.get("reproduced_levels"),
                new_level_credit=0,
            )
        )
    upstream = dict(
        upstream,
        registry_precheck=dict(
            upstream["registry_precheck"],
            loop_precheck_rows=precheck,
            checked_before_new_outcomes=True,
            headline_solve_credit=0,
        ),
    )
    progress("registry_and_high_water_authenticated", len(baseline["entries"]), 0)
    return upstream, refs


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Reader qualification cannot create a solve or policy benefit from an empty delta."""
    value: Json = base_build(work, receipts, output)
    value.update(
        experiment_id=8746,
        milestone=e.MILESTONE,
        random_seed=7498746,
        solve_provenance=None,
        solve_provenance_scope="reader_credits_no_live_outcomes",
        policy_decision="no_supported_policy_change",
        policy_decision_rows=value["decision_rows"],
        unresolved_count=sum(r["status"] == "missing" for r in work["new_event_rows"]),
        standing_generalization_floor_satisfied=not work["failures"],
        retained_frontier_reference=dict(path=str(e.ROOT / FRONTIER), sha256=FRONTIER_PIN),
        evidence_reuse_scope="unchanged_Exp8734_reader_methods_and_qualified_Exp8720_controls",
    )
    value["field_principles"] = {
        k: "Exact sealed inputs and current invocation checks bind this field; empty deltas establish no policy benefit or solve credit."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = e.checksum(value)
    return value


base_build = e.build
e.build, e.authenticate, e.progress = build, authenticate, progress
e.original.observe = observe
e.authority = SimpleNamespace(DESIGN=methods.DESIGN, ACTIVE=methods.ACTIVE, authority=authority)
runtime.e, runtime.retained = e, retained
runtime.retained_validate, runtime.retained_terminal = retained.validate, retained.terminal
main, replay = runtime.main, runtime.replay
