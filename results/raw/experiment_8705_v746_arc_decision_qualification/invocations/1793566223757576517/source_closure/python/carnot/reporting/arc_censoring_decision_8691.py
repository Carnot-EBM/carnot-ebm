"""REQ-REPORT-8691: original qualified bytes authorize this bounded reader only."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

from carnot.reporting import arc_observation_8677 as ownership
from carnot.reporting import arc_observer_transfer_8678 as transfer
from carnot.reporting.arc_censoring_reduce_8691 import reduce as reduce
from carnot.reporting.current_work_receipt import (
    atomic_json as atomic_json,
    canonical_hash,
    sha256_file as sha256_file,
)
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_8691_v745_arc_censoring_decision", "exp8691-arc-censoring-decision"
CLI = f"scripts/experiments/{NAME}.py"
MODEL_SPECS: list[Json] = []
TESTS = ["tests/python/test_arc_censoring_decision_8691.py"]
OWNED = [
    f"python/carnot/reporting/arc_censoring{s}_8691.py" for s in ("_decision", "_reduce", "_runner")
] + [CLI]
PINS = {
    "8677": "sha256:6a6cfe0b065cbe0a30e2b45a4dde6683e90bc203025a4ff96c24aad24116aa8a",
    "8678": "sha256:6928301fc9c7443f8bf301a001207196d9b63a3c096655a8921cec122b4ca498",
}
PREREQUISITE = "Distinguish policy-triggered resets from environment termination before a new curated intervention."


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """A flushed boundary lets the conductor distinguish reading from a stalled child."""
    print(f"[exp8691] phase={phase} completed={completed} pending={pending}", flush=True)


def reference(path: Path) -> Json:
    """Bind bytes independently of a mutable candidate envelope."""
    return dict(path=str(path), sha256=sha256_file(path))


def read(ref: Json) -> Json:
    """A missing operand and replaced bytes must never be silently defaulted."""
    if sha256_file(Path(ref["path"])) != ref["sha256"]:
        raise ValueError("changed_bytes:" + ref["path"])
    return dict(json.loads(Path(ref["path"]).read_bytes()))


def checksum(value: Json) -> str:
    """The replay seal, rather than this reproducibility hash, provides authority."""
    return str(canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"}))


def observe(path: Path, upstream: str, field: str, pin: str) -> Json:
    """Keep missing file, field, measured zero and byte drift separately auditable."""
    value = json.loads(path.read_bytes()) if path.is_file() else {}
    digest = sha256_file(path) if path.is_file() else None
    observed = value.get(field)
    state = (
        "missing_file"
        if not path.is_file()
        else "missing_field"
        if field not in value
        else "observed_zero"
        if observed == 0
        else "changed_bytes"
        if digest != pin
        else "observed_value"
    )
    return dict(
        upstream=upstream,
        check="qualified_public_observations",
        path=str(path),
        sha256=digest,
        expected_hash=pin,
        field=field,
        operator="==",
        expected=1,
        observed=observed,
        operand_state=state,
        passed=type(observed) is int
        and observed == 1
        and digest == pin
        and value.get("required_checks_passed") is True
        and value.get("flagged_adversarial") is False,
    )


def authenticate() -> tuple[list[Json], list[Json], list[Json]]:
    """Use declared producer construction and bound terminal receipts, without a controller rerun."""
    episodes, checks, refs = [], [], []
    for producer, field in (
        (ownership, "observer_delivery_ready_score"),
        (transfer, "arc_rows_ready_score"),
    ):
        path = ROOT / "results" / (producer.NAME + ".json")
        row = observe(
            path, producer.TASK, field, PINS[str(8677 if producer is ownership else 8678)]
        )
        checks.append(row)
        if not row["passed"]:
            continue
        value = json.loads(path.read_bytes())
        side_path = Path(value["terminal_validation_sidecar_path"])
        side = json.loads(side_path.read_bytes())
        bound = Path(side["publication"]["sidecar_path"])
        work = read(value["work_reference"])
        if (
            not read_bound_sidecar(path, bound)["report"]["passed"]
            or not side["passed"]
            or producer.build(
                Path(value["work_reference"]["path"]), value["validation_receipts"], path
            )
            != value
        ):
            raise ValueError("producer_terminal_construction")
        for name, digest in value["code_config_hashes"].items():
            read_bytes = Path(name).read_bytes()
            if "sha256:" + hashlib.sha256(read_bytes).hexdigest() != digest:
                raise ValueError("producer_source_changed")
        registry = value["registry_precheck"]
        if not registry["checked_before_execution"] or registry["policy_received_registry"]:
            raise ValueError("registry_precheck")
        if producer is transfer:
            if sha256_file(Path(registry["path"])) != registry["sha256"]:
                raise ValueError("registry_changed")
            episodes = [read(ref) for ref in work["references"]]
            refs.extend(work["references"])
            refs.append(work["numerical_head_reference"])
            if (
                sha256_file(Path(work["numerical_head_reference"]["path"]))
                != work["numerical_head_reference"]["sha256"]
            ):
                raise ValueError("numerical_checkpoint_changed")
            refs.append(dict(path=registry["path"], sha256=registry["sha256"]))
        refs.extend(
            [reference(path), reference(side_path), reference(bound), value["work_reference"]]
        )
    return episodes if all(r["passed"] for r in checks) else [], checks, refs


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Readiness means a qualified decision, while zero outcomes establish no efficacy."""
    result = reduce([read(ref) for ref in work["episode_refs"]])
    checks = bool(receipts) and all(r["passed"] and not r.get("timed_out", False) for r in receipts)
    verdict = "disqualified" if not checks else "blocked" if work["failures"] else "null"
    ready = int(checks and not work["failures"] and len(work["episode_refs"]) == 16)
    indexed = {
        (ep["unit"]["game"], ep["unit"]["seed"], ep["unit"]["reset_provenance"]): ep
        for ep in [read(ref) for ref in work["episode_refs"]]
    }
    rows = [
        dict(
            game=g,
            seed=s,
            arm=mode,
            status="completed" if (g, s, mode) in indexed else "blocked",
            absolute_metric=len(indexed[(g, s, mode)]["observer_rows"])
            if (g, s, mode) in indexed
            else None,
            missing_reason=None if (g, s, mode) in indexed else "qualified_upstream_unavailable",
        )
        for g in ownership.GAMES
        for s in (11, 23)
        for mode in ("off", "on")
    ]
    value: Json = dict(
        experiment_id=8691,
        task_id=TASK,
        milestone="2026.11.745",
        run_date=work["run_date"],
        honest_verdict="complete_"
        + verdict
        + "_"
        + ("upstream_unavailable" if verdict == "blocked" else "arc_censoring_decision"),
        verdict_class=verdict,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(language_model=0, llm=0, numerical_head=0, live_agent_calls=0),
        target_model=None,
        historical_model_provenance=work["historical_model_provenance"],
        rows=rows,
        intended_count=16,
        completed_count=len(indexed),
        failed_count=int(not checks),
        excluded_count=0,
        sample_size_budget=dict(
            games=4, paired_seeds=8, episodes=16, horizon=120, new_live_episodes=0
        ),
        verifier_is_oracle=False,
        exposure_scope="qualified_public_development_ledger_and_private_constructed_reducer_controls",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        promotion_ready_score=0,
        arc_decision_ready_score=ready,
        branch_gate_values=dict(arc_decision_ready_score=ready),
        required_checks_passed=checks,
        flagged_adversarial=not checks,
        acceptance_gates=dict(
            owned_validation=checks,
            authenticated_upstream=not work["failures"],
            unchanged_policy=True,
            no_censored_promotion=True,
            arc_decision_ready_score=ready,
        ),
        validation_receipts=receipts,
        validation_reference=reference(Path(work["raw"]) / "validation.json"),
        gate_check_summary=work["gates"] + work["failures"],
        adversarial_findings=[],
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        publication_output=str(output),
        invocation_id=Path(work["raw"]).name,
        preconditions_checked=work["preconditions"],
        duration_s=work["duration_s"] + sum(r.get("duration_s", 0) for r in receipts),
        started_at_utc=work["started_at_utc"],
        ended_at_utc=work["ended_at_utc"],
        phase_spans=work["phase_spans"]
        + [dict(phase=r["name"], duration_s=r.get("duration_s", 0)) for r in receipts],
        random_seed=8691,
        primitive_reference=reference(Path(work["raw"]) / "measurement.json"),
        source_artifact_hashes={r["path"]: r["sha256"] for r in work["source_refs"]},
        preserved_failed_attempts=[
            r for r in work["source_refs"] if r["path"].endswith("failure_summary.json")
        ],
        code_config_hashes={r["path"]: r["sha256"] for r in work["code_refs"]},
        raw_shard_hashes={r["path"]: r["sha256"] for r in work["episode_refs"]},
        cited_upstream_artifacts=work["upstream_refs"],
        registry_precheck=work["registry_precheck"],
        solve_provenance="live_agent_self_discovery",
        headline_solve_credit=0,
        historical_quarantines=dict(
            exp8664_flagged_adversarial=True,
            exp8667_flagged_adversarial=True,
            later_code_fix_requalifies_historical_artifact=False,
            all_stamped_quarantines_preserved=True,
        ),
        preserved_nulls=[
            "V733 source-information",
            "V735 cold-cost",
            "V737 supervisor",
            "V743 batch-one",
        ],
        diagnostic_closed=bool(ready),
        diagnostic_disposition="closed_once_no_supported_policy_change",
        no_requeue_without_new_mechanism=True,
        generalization_floor_decision="Supervisor-refinement activity: qualified ledger supports no justified policy change.",
        methodology_note="Read the fixed four-game/eight-pair/sixteen-episode panel once. Censored episodes are unresolved exposure, never failures or efficacy evidence. Seeds and recorder twins are dependent controls. Constructed fixtures qualify only the deterministic reducer; no independent generalization or scientific negative is claimed. No new model load, numerical-head evaluation, live solve, adapter, offline BFS or outer-loop RE.",
        policy_source_reference=work["policy_source_reference"],
        **result,
    )
    value["field_principles"] = {
        k: "Exact invocation and original producer bytes bind this field. Games define independent units; censorship and fixture controls grant no efficacy, policy promotion, model-call or solve credit. Readiness requires all owned checks."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = checksum(value)
    return value
