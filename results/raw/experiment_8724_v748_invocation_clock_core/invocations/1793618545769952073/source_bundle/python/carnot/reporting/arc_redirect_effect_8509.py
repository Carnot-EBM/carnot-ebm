"""REQ-REPORT-8509: selected strategies need effect receipts before benefit claims."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from carnot.agentic.arc_trajectory_supervisor import TrajectorySnapshot, TrajectorySupervisor
from carnot.reporting import arc_panel_qualification_8496 as upstream
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.arc_redirect_fixtures_8509 import fixtures as fixtures
from carnot.reporting.v721_capstone_evidence import freeze, frozen_inputs

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_8509_v731_arc_redirect_effect_audit", "exp8509-arc-redirect-effect-audit"
UPSTREAM = "results/experiment_8496_v730_arc_panel_qualification.json"
UPSTREAM_PIN = "sha256:5804a065a668c89df631687b9277a963643848b61b640e59b8efc1ca72a62ceb"
TASK_PIN = "sha256:2f8c4e843f1b5693b3320664b76decf6835022e6d4b855c8186abfe77591a4b7"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_arc_redirect_effect_8509.py"
OWNED = [
    "python/carnot/reporting/arc_redirect_effect_8509.py",
    "python/carnot/reporting/arc_redirect_fixtures_8509.py",
    "python/carnot/reporting/arc_redirect_runner_8509.py",
    CLI,
]
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counts so a bounded audit remains observable."""
    print(f"[exp8509] phase={phase} completed={completed} pending={pending}", flush=True)


def authenticate(root: Path, raw: Path) -> Json:
    """Original publication and frozen authority determine which historical bytes are usable."""
    path = root / UPSTREAM
    if sha256_file(path) != UPSTREAM_PIN:
        raise ValueError("upstream_pin")
    value = json.loads(path.read_bytes())
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    bound = read_bound_sidecar(path, Path(terminal["publication"]["sidecar_path"]))
    if not bound["report"]["passed"] or not terminal["passed"]:
        raise ValueError("upstream_validation")
    ref = value["work_reference"]
    if sha256_file(Path(ref["path"])) != ref["sha256"]:
        raise ValueError("upstream_work_pin")
    work = json.loads(Path(ref["path"]).read_bytes())
    with frozen_inputs(work["source_input_receipts"], raw / "historical_writes"):
        if not upstream.replay(value):
            raise ValueError("original_sealed_replay")
    if value["run_date"] != "20261013" or len(value["rows"]) != 16:
        raise ValueError("upstream_identity")
    operands = []
    for index, row in enumerate(value["rows"]):
        directory = Path(row["primitive_directory"])
        refs = {}
        for kind, name in [("steps", "steps.jsonl"), ("terminal", "episode.json")]:
            path = directory / name
            refs[kind] = freeze(path, raw / "custody", value["raw_shard_hashes"][str(path)])
        operands.append(dict(episode_id=row["episode_id"], **refs))
        progress("primitive_authenticated", index + 1, 15 - index)
    return dict(
        primary=value,
        primary_reference=freeze(root / UPSTREAM, raw / "custody", UPSTREAM_PIN),
        operands=operands,
        historical_date="20261013",
        original_authority=work["preconditions_checked"],
    )


def effect(mode: str, event: Json) -> str:
    """Mode is an execution setting; only recorded mutations establish an applied effect."""
    if mode == "shadow":
        return "shadow_recommendation"
    if str(event.get("application_disposition", "")).startswith("exception:"):
        return "application_exception"
    old, new = event.get("old_state_hash"), event.get("new_state_hash")
    if old is None or new is None:
        return "application_unobserved"
    return "applied_state_change" if old != new else "selected_but_no_state_change"


def support(rows: list[Json]) -> Json:
    """Keep controls separate and enforce the unchanged resolved-outcome floor per game cell."""
    games = sorted({r["game"] for r in rows})
    arms = sorted({r["arm"] for r in rows})
    cells, leave = [], []
    for mode in ["applied", "shadow"]:
        for arm in arms:
            selected = [r for r in rows if r["mode"] == mode and r["arm"] == arm]
            counts = {g: sum(r["resolved"] for r in selected if r["game"] == g) for g in games}
            cells.append(
                dict(
                    mode=mode,
                    arm=arm,
                    selected_count=len(selected),
                    resolved_count=sum(counts.values()),
                    resolved_by_game=counts,
                    censored_count=sum(r["right_censored"] for r in selected),
                )
            )
            for held in games:
                shared = [
                    g
                    for g in games
                    if g != held
                    and all(
                        any(r["game"] == g and r["arm"] == arm and r["mode"] == peer for r in rows)
                        for peer in ["applied", "shadow"]
                    )
                ]
                leave.append(
                    dict(
                        mode=mode,
                        arm=arm,
                        held_out_game=held,
                        overlapping_games=shared,
                        resolved_by_game={g: counts[g] for g in shared},
                        support_gate_passed=len(shared) >= 3
                        and all(counts[g] >= 5 for g in shared),
                    )
                )
    return dict(
        effect_support_by_arm=cells,
        leave_one_game_out=leave,
        arm_refinement_authorized=bool(leave) and all(r["support_gate_passed"] for r in leave),
    )


def reduce(source: Json) -> Json:
    """Join recorded operands without recovering unrecorded policy or game state."""
    rows = []
    for original, operand in zip(source["primary"]["rows"], source["operands"], strict=True):
        steps_path = Path(operand["steps"]["snapshot_path"])
        if sha256_file(steps_path) != operand["steps"]["expected_sha256"]:
            raise ValueError("primitive_pin")
        steps = [json.loads(line) for line in steps_path.read_text().splitlines()]
        selector = TrajectorySupervisor(window=120)
        outcomes = original["supervisor_outcome_rows"]
        for step in steps:
            for event in step["supervisor_events"]:
                snapshot = TrajectorySnapshot(**event["snapshot"])
                derived = selector.arm_eligibility(snapshot)
                redirect = selector.observe(snapshot)
                if event["redirect"] != (asdict(redirect) if redirect else None):
                    raise ValueError("redirect_drift")
                if redirect is None:
                    continue
                outcome = next(o for o in outcomes if o["action_index"] == redirect.action_index)
                mode = "applied" if original["arm"] == "on" else "shadow"
                resolved = bool(outcome["resolved_by_levelup"])
                missing = [
                    k
                    for k in [
                        "eligibility_predicates",
                        "application_disposition",
                        "old_state_hash",
                        "new_state_hash",
                        "induction_attempt",
                    ]
                    if event.get(k) is None
                ]
                trace = [
                    dict(
                        index=s["index"],
                        action=s["action"],
                        action_sha256=s["action_sha256"],
                        observation_sha256=s["observation_sha256"],
                        level=s["observation"]["level"],
                    )
                    for s in steps
                    if s["index"] > step["index"]
                ]
                rows.append(
                    dict(
                        event_id=f"{original['episode_id']}:{redirect.action_index}",
                        episode_id=original["episode_id"],
                        game=original["game"],
                        seed=original["seed"],
                        mode=mode,
                        arm=redirect.arm,
                        snapshot=event["snapshot"],
                        action_at_selection=dict(
                            index=step["index"],
                            action=step["action"],
                            action_sha256=step["action_sha256"],
                        ),
                        induction_tier_reentry_evidence=dict(
                            selection_attempt_count=event["snapshot"]["induction_attempts"],
                            later_attempt_count=max(
                                (
                                    ev["snapshot"]["induction_attempts"]
                                    for s in steps
                                    if s["index"] > step["index"]
                                    for ev in s["supervisor_events"]
                                ),
                                default=None,
                            ),
                            live_skip_reason=None,
                            execution_class="model_disabled_reinduction"
                            if redirect.arm == "allow_reinduction" and mode == "applied"
                            else "no_reinduction_intervention",
                            provenance="later_recorded_snapshot_counters_and_original_model_free_harness_invariant; direct_attempt_receipt_missing",
                        ),
                        eligibility_predicates=event.get("eligibility_predicates"),
                        replayed_predicates=derived,
                        predicate_provenance="recomputed_from_recorded_snapshot_not_recorded_predicate_receipt",
                        application_disposition=event.get("application_disposition"),
                        old_state_hash=event.get("old_state_hash"),
                        new_state_hash=event.get("new_state_hash"),
                        effect_class=effect(mode, event),
                        applied_effect_observed=None if mode == "applied" else False,
                        resolved=resolved,
                        right_censored=not resolved,
                        outcome_class="resolved" if resolved else "right_censored",
                        censoring=original["censoring"],
                        subsequent_action_trace=trace,
                        missing_operands=missing,
                        status="completed",
                        censored=not resolved,
                        absolute_metric=None,
                        missing_reason="outcome_right_censored" if not resolved else None,
                    )
                )
    return dict(
        rows=rows,
        redirect_effect_rows=[r for r in rows if r["mode"] == "applied"],
        shadow_rows=[r for r in rows if r["mode"] == "shadow"],
        censoring_counts=dict(Counter(r["mode"] for r in rows if r["right_censored"])),
        missing_operands=[dict(event_id=r["event_id"], fields=r["missing_operands"]) for r in rows],
        **support(rows),
    )


def build(work_path: Path, receipts: list[Json], output: Path) -> Json:
    """Scientific null and engineering qualification use different gates and counters."""
    work = json.loads(work_path.read_bytes())
    source = work.get("source")
    reduced = (
        reduce(source)
        if source
        else dict(
            rows=[
                dict(
                    event_id=f"missing:{i}",
                    status="unstarted",
                    absolute_metric=None,
                    missing_reason="upstream_unavailable",
                    censored=True,
                )
                for i in range(32)
            ],
            redirect_effect_rows=[],
            shadow_rows=[],
            censoring_counts={},
            missing_operands=[],
            effect_support_by_arm=[],
            leave_one_game_out=[],
            arm_refinement_authorized=False,
        )
    )
    failed = [r for r in receipts if not r["passed"]]
    verdict = "disqualified" if failed else "blocked" if work["failures"] else "null"
    qualified = bool(receipts) and not failed
    ready = int(qualified and verdict == "null" and len(work["fixture_rows"]) == 8)
    primary = source["primary"] if source else {}
    raw = output.parent / "raw" / output.stem
    games = sorted({r["game"] for r in reduced["rows"]}) if source else []
    value = dict(
        experiment_id=8509,
        task_id=TASK,
        milestone="2026.10.731",
        run_date="20261014",
        honest_verdict=f"complete_{verdict}_v731_arc_redirect_effect_audit",
        verdict_class=verdict,
        gate_check_summary=work["failures"],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        llm_invocation_count=0,
        historical_model_provenance=primary.get("historical_model_provenance", {}),
        **reduced,
        intended_count=32,
        completed_count=len(reduced["rows"]) if source else 0,
        failed_count=0,
        censored_count=sum(r.get("censored", False) for r in reduced["rows"]),
        excluded_count=0,
        independent_count=len(games),
        sample_size_budget=dict(
            episodes=16,
            selected_events=32,
            independent_games=4,
            seeds=[11, 22],
            modes=["applied", "shadow"],
            minimum_overlapping_games=3,
            minimum_resolved_per_shared_arm_cell=5,
        ),
        verifier_is_oracle=False,
        exposure_scope="historically_public_adapter_withheld_live_receipts_private_fixtures_separate",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=qualified,
        flagged_adversarial=bool(work.get("adversarial_findings")),
        acceptance_gates=dict(
            owned_validation=qualified,
            original_primitive_custody=bool(source),
            private_policy_mechanisms=len(work["fixture_rows"]) == 8,
            unchanged_refinement_support=reduced["arm_refinement_authorized"],
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7318509,
        reproducibility_checksum=canonical_hash(work),
        work_reference=dict(path=str(work_path), sha256=sha256_file(work_path)),
        source_artifact_hashes=work.get("source_artifact_hashes", {}),
        code_config_hashes=work.get("code_config_hashes", {}),
        raw_shard_hashes={
            r["snapshot_path"]: r["sha256"]
            for o in (source or {}).get("operands", [])
            for r in [o["steps"], o["terminal"]]
        },
        cited_upstream_artifacts=[
            dict(
                producer="exp8496-arc-panel-qualification",
                path=UPSTREAM,
                sha256=UPSTREAM_PIN,
                original_date="20261013",
            )
        ],
        generalization_diagnostic_ready_score=ready,
        per_game_results=[
            dict(
                game=g,
                applied_selected=sum(r["game"] == g for r in reduced["redirect_effect_rows"]),
                shadow_selected=sum(r["game"] == g for r in reduced["shadow_rows"]),
                resolved=sum(r["resolved"] for r in reduced["rows"] if r["game"] == g),
            )
            for g in games
        ],
        fixture_rows=work["fixture_rows"],
        registry_precheck=primary.get("registry_precheck", {}),
        headline_solve_credit=0,
        solve_provenance="live_agent_self_discovery",
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation")
            for n in range(8499, 8512)
            if n != 8509
        ],
        mechanism_map=dict(
            allow_reinduction="Recorded selection latch is true; already-false explanation contradicted. Fixtures reset it then enter induction, which disabled_by_env skips before generation.",
            force_exploration_diversity="Snapshot diversity inactive at selection. Private real policy changes the draw; live effect hashes were not recorded.",
            censoring="All32 events remain right-censored. No failure rate or scientific benefit is estimable.",
            application="All16 applied-mode records lack direct disposition and old/new state hashes. Mode does not prove mutation.",
        ),
        proposed_future_primitive_change=dict(
            scope="Enable the shipped EligibilityReceiptRecorder in the existing passive panel observer and join induction_attempts at episode end.",
            trigger="Exp8496 primitive supervisor_events contain only snapshot and redirect.",
            fields=[
                "eligibility_predicates",
                "application_disposition",
                "old_state_hash",
                "new_state_hash",
                "induction_attempt",
            ],
            live_policy_change=False,
        ),
        methodology_note="Original sealed V730 live-wrapper primitives support a mode-aware diagnostic. Cached receipts are historical evidence. Bounded private policy fixtures qualify mechanisms, not treatment benefit. Zero resolved outcomes cannot refine the unchanged arm table.",
    )
    value["field_principles"] = {
        k: "Bind this field to immutable invocation evidence; qualification does not establish scientific benefit."
        for k in value
    }
    value["field_principles"].update(
        rows="One row per observed selection; retain missing application operands and right censoring.",
        fixture_rows="Private deterministic policy checks remain separate from live evidence.",
        arm_refinement_authorized="Require three overlapping games and five resolved outcomes per shared arm cell, with modes separate.",
        generalization_diagnostic_ready_score="Qualified reduction only; hidden-game benefit and solve credit remain zero.",
        effect_support_by_arm="Count selected, resolved and censored events separately for each mode and arm.",
        missing_operands="Absent fields are unknown, never reconstructed policy state or observed zeros.",
    )
    return value


def replay(value: Json) -> bool:
    """Independent producer seals reject altered evidence even after all local hashes change."""
    try:
        ref = value["work_reference"]
        work_path = Path(ref["path"])
        if sha256_file(work_path) != ref["sha256"]:
            return False
        work = json.loads(work_path.read_bytes())
        source = work.get("source")
        if source is None and not work["failures"]:
            return False
        if source:
            path = Path(source["primary_reference"]["snapshot_path"])
            if (
                sha256_file(path) != UPSTREAM_PIN
                or json.loads(path.read_bytes()) != source["primary"]
            ):
                return False
            for original, operand in zip(
                source["primary"]["rows"], source["operands"], strict=True
            ):
                for kind, name in [("steps", "steps.jsonl"), ("terminal", "episode.json")]:
                    sealed = source["primary"]["raw_shard_hashes"][
                        str(Path(original["primitive_directory"]) / name)
                    ]
                    if (
                        operand[kind]["expected_sha256"] != sealed
                        or sha256_file(Path(operand[kind]["snapshot_path"])) != sealed
                    ):
                        return False
        for path, digest in work.get("code_config_hashes", {}).items():
            if sha256_file(Path(path)) != digest:
                return False
        if work["fixture_rows"]:
            with TemporaryDirectory(prefix="exp8509-replay-", dir="/var/tmp") as directory:
                actual = [
                    r
                    for mode in ["shadow", "applied"]
                    for r in fixtures(Path(directory) / mode, mode)
                ]
            if actual != work["fixture_rows"]:
                return False
        for receipt in value["validation_receipts"]:
            for channel in ["stdout", "stderr"]:
                if (
                    channel + "_path" in receipt
                    and sha256_file(Path(receipt[channel + "_path"]))
                    != receipt[channel + "_sha256"]
                ):
                    return False
        output = Path(value["terminal_validation_sidecar_path"]).parents[2] / (NAME + ".json")
        return build(work_path, value["validation_receipts"], output) == value
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False
