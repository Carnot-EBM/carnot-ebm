"""REQ-REPORT-8720: a closed panel cannot justify repeated policy refinement."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from carnot.agentic.arc_supervisor_refinement import (
    DEFAULT_LEDGER_PARTS as DEFAULT_LEDGER_PARTS,
    LEDGER_SCHEMA,
    extract_rows,
    receipt_id_for_row,
)
from carnot.agentic.arc_trajectory_supervisor import ARM_ORDER
from carnot.reporting import arc_censoring_decision_8691 as _original
from carnot.reporting import v747_contract_methods as _authority
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
original, authority = _original, _authority
ROOT = original.ROOT
NAME, TASK, MILESTONE = (
    "experiment_8720_v747_arc_supervisor_frontier",
    "exp8720-arc-supervisor-frontier",
    "2026.11.747",
)
CLI = f"scripts/experiments/{NAME}.py"
TESTS = ["tests/python/test_arc_supervisor_frontier_8720.py"]
OWNED = [
    "python/carnot/reporting/arc_supervisor_frontier_8720.py",
    "python/carnot/reporting/arc_supervisor_runner_8720.py",
    CLI,
]
UPSTREAM = "results/experiment_8705_v746_arc_decision_qualification.json"
PIN = "sha256:62e8795168a812f13e0e81002f56e7676fe25e275c0e5a5cf9a81827635be091"
BASELINE = f"results/raw/{NAME}/authority/retained_ledger.json"
LEDGER_PIN = "sha256:e83fb0720502e8351792640bf012ded1e96180a04fd05e4d500ae8830609fe34"
MODEL_SPECS: list[Json] = []
reference, read, checksum = original.reference, original.read, original.checksum
atomic_json, sha256_file = original.atomic_json, original.sha256_file


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts so even a short reader has visible phase boundaries."""
    print(f"[exp8720] phase={phase} completed={completed} pending={pending}", flush=True)


def qualified(path: Path, pin: str) -> tuple[Json, list[Json]]:
    """Pinned bytes and bound terminal receipts authenticate execution without rerunning it."""
    value = read(dict(path=str(path), sha256=pin))
    side_path = Path(value["terminal_validation_sidecar_path"])
    side = json.loads(side_path.read_bytes())
    bound = Path(side["publication"]["sidecar_path"])
    report = read_bound_sidecar(path, bound)["report"]
    if not (
        value["required_checks_passed"] is True
        and value["flagged_adversarial"] is False
        and side["passed"] is True
        and report["passed"] is True
        and all(r["passed"] and not r.get("timed_out", False) for r in value["validation_receipts"])
        and any(r["name"] == "cold_replay" and r["passed"] for r in report["receipts"])
    ):
        raise ValueError("unqualified_producer_terminal")
    return value, [reference(path), reference(side_path), reference(bound)]


def authenticate() -> tuple[Json, list[Json]]:
    """Authenticate the closed decision once; its scientific panel stays closed."""
    value, refs = qualified(ROOT / UPSTREAM, PIN)
    if not (
        type(value["arc_decision_ready_score"]) is int
        and value["arc_decision_ready_score"] == 1
        and value["policy_decision"] == "no_supported_policy_change"
        and value["diagnostic_closed"]
        and len(value["censoring_rows"]) == 16
        and value["qualification_scope"] == "production"
    ):
        raise ValueError("closed_decision_contract")
    work = read(value["primitive_reference"])
    registry = value["registry_precheck"]
    if not (
        registry["checked_before_reduction"]
        and not registry["targeted_registered_solve"]
        and not registry["policy_received_registry"]
        and sha256_file(Path(registry["path"])) == registry["sha256"]
    ):
        raise ValueError("registry_precheck")
    seal_path = Path(work["raw"]).parents[1] / "seals" / (Path(work["raw"]).name + ".json")
    seal = json.loads(seal_path.read_bytes())
    if (
        seal["measurement"] != value["primitive_reference"]
        or seal["code_refs"] != work["code_refs"]
    ):
        raise ValueError("upstream_execution_seal")
    refs += [value["primitive_reference"], reference(seal_path), reference(Path(registry["path"]))]
    return value, refs


def frontier(value: Json, entries: Json) -> Json:
    """Use run, episode and redirect identities, since redirect names repeat between seeds."""
    return dict(
        closed_invocation_id=value["invocation_id"],
        closed_at_utc=value["ended_at_utc"],
        closed_event_identities=sorted(
            f"{value['invocation_id']}|{r['episode_id']}|{r['redirect_id']}"
            for r in value["censoring_rows"]
        ),
        ledger_receipt_identities=sorted(entries),
    )


def delta(baseline: Json, ledger: Json, value: Json) -> tuple[list[Json], Json]:
    """Only new native receipts are opened; retained identities cannot silently change."""
    if ledger["schema"] != LEDGER_SCHEMA or not isinstance(ledger["entries"], dict):
        raise ValueError("ledger_authority_schema")
    known, current = baseline["entries"], ledger["entries"]
    if any(current.get(k) != row for k, row in known.items()):
        raise ValueError("changed_retained_identity")
    rows: list[Json] = []
    seen: dict[str, Json] = {}
    for index, identity in enumerate(sorted(set(current) - set(known))):
        progress("new_receipt", index, len(current) - len(known) - index)
        entry = current[identity]
        producer_ref = entry["producer_reference"]
        producer, _ = qualified(Path(producer_ref["path"]), producer_ref["sha256"])
        source = Path(entry["source"])
        if producer["source_artifact_hashes"].get(str(source)) != sha256_file(source):
            raise ValueError("new_source_authority")
        raw = next(
            r
            for r in extract_rows(json.loads(source.read_bytes()))
            if receipt_id_for_row(r) == identity
        )
        if raw["solve_provenance"] != "live_agent_self_discovery" or datetime.fromisoformat(
            raw["event_timestamp"]
        ) <= datetime.fromisoformat(value["ended_at_utc"]):
            raise ValueError("new_event_chronology_or_provenance")
        for event in raw["trajectory_supervisor"]["redirects"]:
            key = f"{raw['invocation_id']}|{raw['game']}|{raw['seed']}|{event['action_index']}"
            row = dict(
                event,
                event_identity=key,
                game=raw["game"],
                seed=raw["seed"],
                invocation_id=raw["invocation_id"],
                source_reference=reference(source),
                producer_reference=producer_ref,
            )
            status = event.get("horizon_status")
            if status == "no_progress_within_horizon" and event.get("observed_horizon") != 120:
                raise ValueError("incomplete_no_progress")
            row.update(
                selected=event.get("selected"),
                applied=event.get("applied"),
                reset_cause=event.get("reset_cause"),
                status="completed"
                if status in {"no_progress_within_horizon", "level_progress_observed"}
                else "censored"
                if status == "censored"
                else "missing",
                outcome=status,
            )
            if row["arm"] not in ARM_ORDER or (key in seen and seen[key] != row):
                raise ValueError("conflicting_or_unknown_event")
            seen[key] = row
    rows = list(seen.values())
    return rows, frontier(value, current)


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """A successful empty reader earns administrative readiness, never a positive result."""
    checks = bool(receipts) and all(r["passed"] and not r.get("timed_out", False) for r in receipts)
    production = all(
        any(r["name"] == name and r["passed"] for r in receipts)
        for name in ("original_disjoint_union", "preserved_raw_child_sqlite", "coverage_report")
    )
    verdict = "disqualified" if not checks else "blocked" if work["failures"] else "null"
    ready = int(checks and production and not work["failures"])
    rows = work["new_event_rows"]
    raw = Path(work["raw"])
    value: Json = dict(
        experiment_id=8720,
        task_id=TASK,
        milestone=MILESTONE,
        run_date=work["run_date"],
        invocation_id=raw.name,
        verdict_class=verdict,
        honest_verdict="complete_null_no_new_supervisor_outcomes"
        if verdict == "null" and not rows
        else f"complete_{verdict}_"
        + ("ledger_authority" if verdict == "blocked" else "supervisor_frontier"),
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(language_model=0, llm=0, numerical_head=0, live_agent_calls=0),
        target_model=None,
        historical_model_provenance=work["historical_model_provenance"],
        rows=rows,
        new_event_rows=rows,
        intended_count=len(rows),
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=int(not checks),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        independent_count=len({r["game"] for r in rows}),
        sample_size_budget=dict(
            new_events=len(rows),
            closed_games=4,
            closed_pairs=8,
            closed_episodes=16,
            closed_horizon=120,
            new_live_episodes=0,
        ),
        verifier_is_oracle=False,
        exposure_scope="retained_public_runtime_evidence_and_private_constructed_controls",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=checks,
        flagged_adversarial=not checks,
        arc_frontier_ready_score=ready,
        branch_gate_values=dict(arc_frontier_ready_score=ready),
        acceptance_gates=dict(
            owned_validation=checks,
            production_receipts=production,
            authenticated_inputs=not work["failures"],
            unchanged_policy=True,
            arc_frontier_ready_score=ready,
        ),
        validation_receipts=receipts,
        validation_reference=reference(raw / "validation.json"),
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        adversarial_findings=[],
        gate_check_summary=work["gates"] + work["failures"],
        preconditions_checked=work["preconditions"],
        duration_s=work["duration_s"],
        started_at_utc=work["started_at_utc"],
        ended_at_utc=work["ended_at_utc"],
        phase_spans=work["phase_spans"],
        random_seed=8720,
        primitive_reference=reference(raw / "measurement.json"),
        publication_output=str(output),
        source_artifact_hashes={r["path"]: r["sha256"] for r in work["source_refs"]},
        code_config_hashes={r["path"]: r["sha256"] for r in work["code_refs"]},
        raw_shard_hashes={r["path"]: r["sha256"] for r in work["upstream_refs"]},
        cited_upstream_artifacts=work["upstream_refs"],
        authority=work["authority"],
        registry_precheck=work["registry_precheck"],
        frontier_before=work["frontier_before"],
        frontier_after=work["frontier_after"],
        no_policy_change=True,
        solve_provenance="live_agent_self_discovery",
        new_level_credit=0,
        solve_provenance_scope="incidental_original_runtime_evidence_only",
        policy_source_reference=work["policy_source_reference"],
        decision_rows=[
            dict(
                arm=arm,
                decision="retain_unchanged",
                selection_order=index,
                usable_cross_game_support=len(
                    {
                        r["game"]
                        for r in rows
                        if r["arm"] == arm
                        and r["status"] == "completed"
                        and r["reset_cause"] is not None
                    }
                ),
                reopening_prerequisites=[
                    "distinct intervention",
                    "separated reset causes",
                    "uncensored cross-game support",
                    "separately authorized future experiment",
                ],
            )
            for index, arm in enumerate(ARM_ORDER)
        ],
        historical_quarantines=dict(
            exp8691_flagged_adversarial=True, all_stamped_quarantines_preserved=True
        ),
        methodology_note="Authenticate Exp8705 and the versioned live ledger once. Compare exact event identities and read only new entries. The closed four-game, eight-pair, sixteen-episode 120-action panel is not replayed or requalified. Empty deltas and censored exposure support no policy change. No live game, model, source RE, offline BFS, adapter, leaderboard submission, learning benefit or physical acceleration is measured.",
    )
    value["field_principles"] = {
        k: "Exact invocation and sealed original bytes bind claims; an empty delta grants no policy benefit or new level credit. Missing evidence differs from observed zero."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = checksum(value)
    return value
