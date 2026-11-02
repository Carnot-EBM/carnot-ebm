"""REQ-REPORT-8599: identify observed support without inventing causal outcomes.

The existing reducer owns application semantics. This diagnostic adds explicit
follow-up windows and game-level transfer gates without executing a game.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

import yaml

from carnot.reporting import arc_capability_panel_8588 as panel
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8599_v738_arc_outcome_identifiability"
TASK = "exp8599-arc-outcome-identifiability"
TASK_PIN = "sha256:a7cd0d49d2c2dd607ecce19c03158579fea62078e10e5bc9ea3a19e64158502d"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = ["tests/python/test_arc_outcome_identifiability_8599.py"]
OWNED = [
    "python/carnot/reporting/arc_outcome_identifiability_8599.py",
    "python/carnot/reporting/arc_outcome_runner_8599.py",
    CLI,
]
PANEL_PATH = "results/experiment_8588_v737_arc_capability_panel.json"
PANEL_PIN = "sha256:2624f20279c06245a393266f9ca793b3ed43ad889e422b64de44fd7688633dda"
PUBLICATION_PATH = panel.PUBLICATION
PUBLICATION_PIN = panel.PUBLICATION_PIN
MODEL_SPECS: list[Json] = []
GAMES = ["ar25-0c556536", "ft09-0d8bbf25", "lf52-271a04aa", "tn36-ef4dde99"]
MODES = panel.panel.MODES
ARMS = [
    "drop_goal_bias",
    "allow_reinduction",
    "tool_loop_reinduction",
    "force_exploration_diversity",
]
REQUIRED_CHECKS = {
    "owned_tests",
    "coverage_report",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
    "collect_0",
    "collect_1",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts so immutable-receipt replay cannot hide a stalled phase."""
    print(f"[exp8599] phase={phase} completed={completed} pending={pending}", flush=True)


def checked(path: Path, digest: str) -> Json:
    """Independent byte pins prevent a rehashed envelope from changing history."""
    if sha256_file(path) != digest:
        raise ValueError("sealed_hash:" + str(path))
    return dict(json.loads(path.read_bytes()))


def authenticate(root: Path) -> tuple[Json, Json]:
    """Original producer snapshots keep old dates separate from current authority."""
    primary = checked(root / PANEL_PATH, PANEL_PIN)
    publication = checked(root / PUBLICATION_PATH, PUBLICATION_PIN)
    for path, value, field in [
        (root / PANEL_PATH, primary, "arc_panel_rows_ready_score"),
        (root / PUBLICATION_PATH, publication, "arc_panel_execution_ready_score"),
    ]:
        if not (
            value[field] == 1
            and value["required_checks_passed"]
            and value["flagged_adversarial"] is False
        ):
            raise ValueError("upstream_readiness")
        side = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
        bound = read_bound_sidecar(path, Path(side["publication"]["sidecar_path"]))
        if not side["passed"] or not bound["report"]["passed"]:
            raise ValueError("upstream_publication_binding")
        for code, digest in value["code_config_hashes"].items():
            if sha256_file(Path(code)) != digest:
                raise ValueError("original_code_hash:" + code)
    ref = primary["work_reference"]
    work = checked(Path(ref["path"]), ref["sha256"])
    authority = work["activation_reference"]
    authority_path = Path(authority["snapshot_path"])
    if sha256_file(authority_path) != authority["sha256"]:
        raise ValueError("original_authority_hash")
    plan = yaml.safe_load(authority_path.read_bytes())
    for task_id, pin in [
        (panel.TASK, panel.TASK_PIN),
        (
            "exp8587-arc-publication-qualification",
            "sha256:aab245ac8b3028b4713e33fcc5a892b20026480cddb644b7da5c85c0582a4205",
        ),
    ]:
        task = next(t for t in plan["tasks"] if t["id"] == task_id)
        if canonical_hash(task) != pin:
            raise ValueError("original_authority_task")
    if len(work["references"]) != 16 or {r["unit"]["game"] for r in work["references"]} != set(
        GAMES
    ):
        raise ValueError("fixed_live_panel")
    return primary, work


def windows(steps: list[Json], events: list[Json]) -> list[Json]:
    """A visible next state is observed evidence; unresolved progress stays censored."""
    rows = []
    for event in events:
        index = event["wrapper_action_index"]
        following = [s for s in steps if s["index"] >= index]
        resolved = next((s["index"] for s in following if s["progress"]), None)
        end = resolved if resolved is not None else len(steps)
        competitors = [r for r in events if index < r["wrapper_action_index"] <= end]
        next_redirect = min((r["wrapper_action_index"] for r in competitors), default=end + 1)
        before, after = event["before_state"], event["after_state"]
        rows.append(
            dict(
                event_id=event["event_id"],
                wrapper_action_index=index,
                selected=event["selected"],
                applied=event["applied"],
                effective_action_policy_change=event["effective_action_policy_change"],
                induction_latch_changed=before["induced"] != after["induced"],
                changed_action=None,
                changed_action_missing_reason="no_same_state_counterfactual",
                observed_action_changed_from_previous=(
                    steps[index - 1]["action_sha256"] != steps[index - 2]["action_sha256"]
                ),
                outcome_observed=True,
                next_observation_sha256=steps[index - 1]["observation_sha256"],
                next_level=steps[index - 1]["observation"]["level"],
                resolved_by_levelup=resolved is not None,
                observable_post_intervention_actions=end - index + 1,
                uncontested_observable_actions=next_redirect - index,
                competing_redirect_ids=[r["event_id"] for r in competitors],
                right_censored=resolved is None,
                missing_reason="action_budget_exhausted" if resolved is None else None,
                attribution="association_only",
            )
        )
    return rows


def transfer(cells: list[Json]) -> list[Json]:
    """Support must survive a held-out game before the table can justify refinement."""
    rows = []
    for held_out in GAMES:
        training = [g for g in GAMES if g != held_out]
        for arm in ARMS:
            shared = [
                g
                for g in training
                if all(
                    any(
                        c["game"] == g
                        and c["mode"] == m
                        and c["arm"] == arm
                        and c["selected_count"] > 0
                        for c in cells
                    )
                    for m in MODES
                )
            ]
            evidence = [c for c in cells if c["game"] in shared and c["arm"] == arm]
            sufficient = len(shared) >= 3 and all(c["resolved_count"] >= 5 for c in evidence)
            rows.append(
                dict(
                    held_out_game=held_out,
                    arm=arm,
                    training_games=training,
                    overlapping_training_games=shared,
                    overlapping_training_game_count=len(shared),
                    shared_cells=evidence,
                    minimum_overlapping_games=3,
                    minimum_resolved_per_shared_cell=5,
                    eligible=sufficient,
                    fitted=False,
                    ranking=None,
                )
            )
    return rows


def audit(root: Path, raw: Path) -> Json:
    """Recompute qualified outcomes and independently count every original action."""
    start = time.monotonic()
    primary, work = authenticate(root)
    rows, actions, outcomes, cells = [], [], [], []
    for number, ref in enumerate(work["references"]):
        progress("replay_episode_before", number, 16 - number)
        if time.monotonic() - start > 3600:
            raise TimeoutError("measurement_deadline")
        row, events, excluded = panel.panel.reduce_episode(ref)
        directory = Path(ref["directory"])
        steps = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
        terminal = json.loads((directory / "episode.json").read_bytes())
        if not terminal["adapter_withheld"] or not terminal["actual_wrapper_path"]:
            raise ValueError("live_adapter_withheld")
        original = next(r for r in primary["rows"] if r["episode_id"] == row["episode_id"])
        if row != original:
            raise ValueError("original_row_mismatch")
        if len(steps) != 512 or sum(s["progress"] for s in steps) != row["peak_level"]:
            raise ValueError("independent_action_progress_count")
        for step in steps:
            actions.append(
                dict(
                    record_id=f"{row['episode_id']}:{step['index']}",
                    episode_id=row["episode_id"],
                    source_path=str(directory / "steps.jsonl"),
                    source_sha256=ref["steps_sha256"],
                    index=step["index"],
                    action=step["action"],
                    action_sha256=step["action_sha256"],
                    observation_sha256=step["observation_sha256"],
                    level=step["observation"]["level"],
                    progress=step["progress"],
                    choose_action_s=step["choose_action_s"],
                    sdk_transition_s=step["sdk_transition_s"],
                )
            )
        local = windows(steps, events)
        for event, outcome in zip(events, local, strict=True):
            if outcome["resolved_by_levelup"] != event["resolved_by_levelup"]:
                raise ValueError("resolution_mismatch")
            outcomes.append(
                dict(
                    outcome,
                    episode_id=row["episode_id"],
                    game=row["game"],
                    seed=row["seed"],
                    mode=row["mode"],
                    arm=event["arm"],
                    source_event_sha256=canonical_hash(event),
                    record_id=f"{row['episode_id']}:{event['event_id']}",
                )
            )
        for arm in ARMS:
            local_arm = [
                r for r in outcomes if r["episode_id"] == row["episode_id"] and r["arm"] == arm
            ]
            cells.append(
                dict(
                    episode_id=row["episode_id"],
                    game=row["game"],
                    seed=row["seed"],
                    mode=row["mode"],
                    arm=arm,
                    selected_count=len(local_arm),
                    applied_count=sum(r["applied"] for r in local_arm),
                    changed_action_count=None,
                    policy_change_count=sum(r["effective_action_policy_change"] for r in local_arm),
                    outcome_observed_count=sum(r["outcome_observed"] for r in local_arm),
                    resolved_count=sum(r["resolved_by_levelup"] for r in local_arm),
                    censored_count=sum(r["right_censored"] for r in local_arm),
                    unavailable_predicate_count=sum(r["arm"] == arm for r in excluded),
                )
            )
        rows.append(row)
        progress("replay_episode_after", number + 1, 15 - number)
    pairing = panel.panel.pair(rows)
    if pairing["paired_results"] != primary["paired_results"]:
        raise ValueError("original_pair_mismatch")
    # Pool seeds within each game; a transfer cell never pools different games.
    pooled = [
        dict(
            game=g,
            mode=m,
            arm=a,
            selected_count=sum(
                c["selected_count"] for c in cells if (c["game"], c["mode"], c["arm"]) == (g, m, a)
            ),
            resolved_count=sum(
                c["resolved_count"] for c in cells if (c["game"], c["mode"], c["arm"]) == (g, m, a)
            ),
        )
        for g in GAMES
        for m in MODES
        for a in ARMS
    ]
    result = dict(
        rows=rows,
        action_receipt_rows=actions,
        censoring_rows=outcomes,
        intervention_support_rows=cells,
        leave_one_game_out_support=transfer(pooled),
        per_game_results=pairing["per_game_results"],
        paired_results=pairing["paired_results"],
        registry_precheck=primary["registry_precheck"],
        historical_model_provenance=primary["historical_model_provenance"],
        next_model_enabled_condition=primary["next_model_enabled_condition"],
        original_references=work["references"],
        original_authority=work["activation_reference"],
    )
    raw.mkdir(parents=True, exist_ok=True)
    for name in ["action_receipt_rows", "censoring_rows", "intervention_support_rows"]:
        atomic_json(raw / (name + ".json"), result[name])
    atomic_json(raw / "original_references.json", work["references"])
    return result


def build(work_path: Path, receipts: list[Json], output: Path) -> Json:
    """Readiness certifies the diagnostic; it does not certify scientific benefit."""
    work = json.loads(work_path.read_bytes())
    data = work["data"]
    failures = work["failures"]
    missing_checks = sorted(REQUIRED_CHECKS - {r["name"] for r in receipts}) if data else []
    passed = bool(receipts) and all(r["passed"] for r in receipts) and not missing_checks
    verdict = "disqualified" if not passed else "blocked" if failures else "null"
    rows = data.get(
        "rows",
        [
            dict(
                episode_id=f"{g}:{s}:{m}",
                game=g,
                seed=s,
                mode=m,
                status="unstarted",
                absolute_metric=None,
                missing_reason="external_precondition_unavailable",
            )
            for g in GAMES
            for s in [11, 22]
            for m in MODES
        ],
    )
    value = dict(
        data,
        experiment_id=8599,
        task_id=TASK,
        milestone="2026.10.738",
        run_date=work["run_date"],
        honest_verdict=f"complete_{verdict}_arc_outcome_identifiability",
        verdict_class=verdict,
        rows=rows,
        intended_count=16,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(bool(r.get("censoring")) for r in rows),
        excluded_count=0,
        independent_count=len({r["game"] for r in rows if r["status"] == "completed"}),
        sample_size_budget=dict(episodes=16, actions=8192, seed_pairs=8, independent_games=4),
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS, llm=0, numerical_head=0),
        historical_model_provenance=data.get("historical_model_provenance", {}),
        verifier_is_oracle=False,
        exposure_scope="exposed_public_adapter_withheld_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        solve_provenance="live_agent_self_discovery",
        headline_solve_credit=0,
        arc_outcome_audit_ready_score=int(passed and not failures),
        refinement_eligible_score=int(
            any(r["eligible"] for r in data.get("leave_one_game_out_support", []))
        ),
        no_supported_update=not any(
            r["eligible"] for r in data.get("leave_one_game_out_support", [])
        ),
        required_checks_passed=passed,
        flagged_adversarial=not passed,
        acceptance_gates={r["name"]: r["passed"] for r in receipts},
        gate_check_summary=failures
        + [
            dict(
                check="required_validation_present",
                upstream=TASK,
                path=str(work_path),
                hash=sha256_file(work_path),
                field=name,
                operator="present",
                expected=True,
                observed=None,
                passed=False,
            )
            for name in missing_checks
        ]
        + [
            dict(
                r,
                check=r["name"],
                upstream=TASK,
                path=r.get("stdout_path"),
                hash=r.get("stdout_sha256"),
                field="exit_code",
                operator="==",
                expected=r.get("expected_exit", 0),
                observed=r.get("exit_code"),
            )
            for r in receipts
            if not r["passed"]
        ],
        validation_receipts=receipts,
        adversarial_findings=[],
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"] + sum(r.get("duration_s", 0) for r in receipts),
        duration_scope="measurement_and_validation_children_before_terminal_publication",
        phase_spans=work["phase_spans"]
        + [
            dict(
                phase=r["name"],
                started_monotonic_ns=r.get("started_monotonic_ns"),
                ended_monotonic_ns=r.get("ended_monotonic_ns"),
                duration_s=r.get("duration_s"),
            )
            for r in receipts
        ],
        random_seed=7388599,
        source_artifact_hashes=work["source_artifact_hashes"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=work["raw_shard_hashes"],
        work_reference=dict(path=str(work_path), sha256=sha256_file(work_path)),
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        cited_upstream_artifacts=[
            dict(path=PANEL_PATH, sha256=PANEL_PIN),
            dict(path=PUBLICATION_PATH, sha256=PUBLICATION_PIN),
        ],
        preserved_historical_outcomes=[
            "V737 twelve scheduled slots; archive ends V736",
            "Exp8580 validator qualified",
            "Exp8581 causal accounting qualified",
            "Exp8582 runner-clock regression quarantined",
            "Exp8584 methodology provenance quarantined",
            "Exp8529 source null",
            "Exp8553 label rejection",
            "Exp8559 cold-cost null",
            "all prior hashes and quarantines retained",
        ],
        methodology_note="Observed states and policy changes are associations. Censored windows provide no failure rate or arm ranking. No new game, solve, hidden-game score or generalization gain.",
    )
    for key in [
        "per_game_results",
        "action_receipt_rows",
        "intervention_support_rows",
        "censoring_rows",
        "leave_one_game_out_support",
    ]:
        value.setdefault(key, [])
    value.setdefault(
        "registry_precheck", dict(available=False, missing_reason="upstream_unavailable")
    )
    value.setdefault(
        "next_model_enabled_condition",
        "Qualify mandated Qwen runtime; separately register a fresh model-enabled study.",
    )
    value["field_principles"] = {
        k: "Bind immutable evidence and actual execution; missing differs from zero. Readiness grants no causal, solve or generalization credit."
        for k in value
    }
    value["field_principles"]["reproducibility_checksum"] = (
        "Hash the complete candidate except this checksum."
    )
    value["field_principles"].update(
        action_receipt_rows="One row per original episode/action index, with original hashes and measured decision/SDK seconds.",
        intervention_support_rows="Count selection, application, policy change, next-state observation and resolution separately in each game/seed/mode/arm cell. Counterfactual action change is missing.",
        censoring_rows="Count observed actions including the intervention action through level-up or terminal, and separately until the next competing redirect. Unresolved is not failure.",
        leave_one_game_out_support="Hold out one game; require three shared training games and five resolved observations per mode/arm/game cell. Pool seeds only within games; fit nothing.",
        per_game_results="Recompute masked-minus-baseline paired metrics and average two seeds within each independent game.",
        arc_outcome_audit_ready_score="Readiness requires authenticated evidence and every required owned and original private E2E check; it grants no scientific benefit.",
        historical_model_provenance="Imported panel used the common frame-change CNN checkpoint with cached per-game memory withheld. This diagnostic performs zero numerical or language-model inference.",
        no_supported_update="Unsupported shared resolved cells cannot rank arms or justify a priority change. Model-disabled induction is not model-enabled capability.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(value: Json) -> bool:
    """Cold replay checks original inputs, logs and the complete derived object."""
    ref = value["work_reference"]
    work = checked(Path(ref["path"]), ref["sha256"])
    for field in ["source_artifact_hashes", "code_config_hashes", "raw_shard_hashes"]:
        for path, digest in work[field].items():
            if sha256_file(Path(path)) != digest:
                raise ValueError("replay_operand_hash:" + path)
    if work["data"]:
        fresh = audit(Path(work["input_root"]), Path(work["replay_scratch"]))
        if fresh != work["data"]:
            raise ValueError("recomputed_primitive_mismatch")
    for receipt in value["validation_receipts"]:
        for stream in ["stdout", "stderr"]:
            if sha256_file(Path(receipt[stream + "_path"])) != receipt[stream + "_sha256"]:
                raise ValueError("validation_log_hash")
    output = Path(value["terminal_validation_sidecar_path"]).parents[2] / (NAME + ".json")
    return value == build(Path(ref["path"]), value["validation_receipts"], output)
