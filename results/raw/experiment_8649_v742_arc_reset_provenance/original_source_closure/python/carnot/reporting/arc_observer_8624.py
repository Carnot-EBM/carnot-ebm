"""REQ-REPORT-8624: bind new horizon accounting to unchanged historical science."""

from __future__ import annotations

import json
import mmap
from pathlib import Path
import re
from typing import Any

import yaml

from carnot.agentic.arc_outcome_observer import ArcOutcomeObserver
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
MODEL_SPECS: list[Json] = []
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8624_v740_arc_outcome_observer"
TASK = "exp8624-arc-outcome-observer"
TASK_PIN = "sha256:b394c385e02ec0a4b9e67733287b8d8f79a8bbc1cb06c8f32c54f36ef6e35d06"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = [
    "tests/python/test_arc_outcome_observer_8624.py",
    "tests/python/test_arc_observer_report_8624.py",
]
OWNED = [
    "python/carnot/agentic/arc_outcome_observer.py",
    "python/carnot/reporting/arc_observer_fixtures_8624.py",
    "python/carnot/reporting/arc_observer_8624.py",
    "python/carnot/reporting/arc_observer_runner_8624.py",
    CLI,
]
LIVE = "python/carnot/agentic/arc_competition_agent.py"
REQUIRED_CHECKS = {
    "owned_tests",
    "coverage_report",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
    "collect_0",
    "collect_1",
    "live_hook_tests",
    "live_statement_coverage",
}
UPSTREAM = {
    "results/experiment_8588_v737_arc_capability_panel.json": "sha256:2624f20279c06245a393266f9ca793b3ed43ad889e422b64de44fd7688633dda",
    "results/experiment_8599_v738_arc_outcome_identifiability.json": "sha256:3864ae15ee590649ef66b6bdd70920717be4226d569fa80981f039371ff90db0",
    "results/experiment_8601_v738_capstone.json": "sha256:fc6b253e172952fb53fbb69a9e96fc490db1e5a7e80d029cc617b8e4cf6b8088",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Use the common bounded validator's flushed phase interface."""
    print(f"[exp8624] phase={phase} completed={completed} pending={pending}", flush=True)


def checked(path: Path, digest: str) -> Json:
    """Pinned original producer bytes cannot be replaced by rehashed envelopes."""
    if sha256_file(path) != digest:
        raise ValueError("original_hash:" + str(path))
    return dict(json.loads(path.read_bytes()))


def authenticate(root: Path) -> tuple[Json, Json, Json]:
    """Use original authority snapshots; the live policy is intentionally newer."""
    paths = list(UPSTREAM)
    common = ["terminal_validation_sidecar_path", "required_checks_passed", "flagged_adversarial"]
    panel = original_fields(
        root / paths[0], UPSTREAM[paths[0]], common + ["work_reference", "honest_verdict"]
    )
    audit = original_fields(
        root / paths[1],
        UPSTREAM[paths[1]],
        common
        + [
            "censoring_rows",
            "historical_model_provenance",
            "refinement_eligible_score",
            "intervention_support_rows",
        ],
    )
    checked_bytes(root / paths[2], UPSTREAM[paths[2]])
    for value, path in [(panel, root / next(iter(UPSTREAM))), (audit, root / list(UPSTREAM)[1])]:
        side = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
        bound = read_bound_sidecar(path, Path(side["publication"]["sidecar_path"]))
        if not (
            value["required_checks_passed"]
            and not value["flagged_adversarial"]
            and side["passed"]
            and bound["report"]["passed"]
        ):
            raise ValueError("original_validation")
    ref = panel["work_reference"]
    work = original_fields(Path(ref["path"]), ref["sha256"], ["activation_reference", "references"])
    authority = work["activation_reference"]
    plan = yaml.safe_load(checked_bytes(Path(authority["snapshot_path"]), authority["sha256"]))
    task = next(t for t in plan["tasks"] if t["id"] == "exp8588-arc-capability-panel")
    if (
        canonical_hash(task)
        != "sha256:d6173cfa57ceaeb6fe9f9a3039d574128e27584bae8ed4a4fbe9dea6efcc1ad2"
    ):
        raise ValueError("original_authority_task")
    return panel, audit, work


def checked_bytes(path: Path, digest: str) -> bytes:
    """Authenticate non-JSON authority and original transition streams too."""
    data = path.read_bytes()
    if sha256_file(path) != digest:
        raise ValueError("original_hash:" + str(path))
    return data


def original_fields(path: Path, digest: str, fields: list[str], *, indent: int = 2) -> Json:
    """Read small fields from pinned pretty JSON without allocating old pixel ledgers."""
    if sha256_file(path) != digest:
        raise ValueError("original_hash:" + str(path))
    result: Json = {}
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        for field in fields:
            pattern = (r"^" + " " * indent + '"' + re.escape(field) + r'":\s*').encode()
            matches = list(re.finditer(pattern, data, re.M))
            if len(matches) != 1:
                raise ValueError("original_field_identity:" + field)
            start, size = matches[0].end(), 4096
            while True:
                try:
                    result[field] = json.JSONDecoder().raw_decode(
                        data[start : start + size].decode()
                    )[0]
                    break
                except json.JSONDecodeError:
                    if start + size >= len(data):
                        raise
                    size *= 2
    return result


def reduce_stream(steps: list[Json], events: list[Json]) -> list[Json]:
    """Replay available actions only; missing history never supplies support."""
    observer = ArcOutcomeObserver()
    handled: set[str] = set()
    for step in steps:
        index = step["index"]
        for event in events:
            if event["wrapper_action_index"] == index:
                observer.redirect(
                    event["event_id"],
                    event["arm"],
                    (event.get("observation") or {}).get("level"),
                    applied=event["applied"],
                    legacy=event.get("legacy_receipt"),
                )
                handled.add(event["event_id"])
        observer.action_index = index - 1
        action = step["action"]
        observer.applied(
            dict(kind=action["name"], data=action.get("data")) if "name" in action else action
        )
        observer.observe(index, step.get("observation"))
    observer.finish("action_cap")
    rows = observer.rows()
    for event in events:
        if event["event_id"] not in handled:
            rows.append(
                dict(
                    redirect_id=event["event_id"],
                    arm=event["arm"],
                    horizon_status="censored",
                    end_reason="missing_application",
                    observed_horizon=0,
                    sequence=[],
                    overlapping_redirect_ids=[],
                    causal_attribution=None,
                    legacy_receipt=event.get("legacy_receipt"),
                )
            )
    return rows


def historical(root: Path) -> Json:
    """Reduce authenticated stored transitions without executing the old panel."""
    panel, audit, work = authenticate(root)
    inputs, windows, hashes = [], [], {}
    for number, ref in enumerate(work["references"]):
        directory = Path(ref["directory"])
        data = checked_bytes(directory / "steps.jsonl", ref["steps_sha256"])
        legacy_rows = original_fields(
            directory / "episode.json", ref["terminal_sha256"], ["redirects"], indent=4
        )["redirects"]
        steps = []
        for line in data.splitlines():
            original = json.loads(line)
            observation = original.get("observation")
            steps.append(
                dict(
                    index=original["index"],
                    action=original["action"],
                    observation=None
                    if observation is None
                    else {k: v for k, v in observation.items() if k != "pixels"},
                    original_observation_sha256=original["observation_sha256"],
                )
            )
        events = []
        for outcome in audit["censoring_rows"]:
            if outcome["episode_id"] != ref["unit"]["episode_id"]:
                continue
            legacy = next(
                r
                for r in legacy_rows
                if r["action_index"] == int(outcome["event_id"].rsplit(":", 1)[1])
            )
            events.append(
                dict(
                    event_id=outcome["event_id"],
                    arm=outcome["arm"],
                    wrapper_action_index=outcome["wrapper_action_index"],
                    applied=outcome["applied"],
                    observation=dict(level=legacy["level"]),
                    legacy_receipt=legacy,
                )
            )
        unit = ref["unit"]
        local = reduce_stream(steps, events)
        inputs.append(dict(unit=unit, steps=steps, events=events))
        windows.extend(
            dict(
                r,
                game=unit["game"],
                mode=unit["mode"],
                seed=unit["seed"],
                episode_id=unit["episode_id"],
            )
            for r in local
        )
        for p, h in [
            (directory / "steps.jsonl", ref["steps_sha256"]),
            (directory / "episode.json", ref["terminal_sha256"]),
        ]:
            hashes[str(p)] = h
        print(
            f"[exp8624] stored completed={number + 1} pending={len(work['references']) - number - 1}",
            flush=True,
        )
    return dict(
        inputs=inputs,
        horizon_status_rows=windows,
        historical_hashes=hashes,
        original_authority=work["activation_reference"],
        historical_model_provenance=audit["historical_model_provenance"],
        arm_support_rows=[
            dict(
                cell,
                missing_reason="no_selected_redirect"
                if cell["selected_count"] == 0
                else "no_resolved_level_outcomes_for_refinement",
            )
            for cell in audit["intervention_support_rows"]
        ],
        missing_historical_fields=[
            "terminal_state_on_each_observation",
            "full_reset_on_each_observation",
            "observation_delivery_id",
        ],
        old_refinement_eligible_score=audit["refinement_eligible_score"],
        original_panel_verdict=panel["honest_verdict"],
    )


def build(work_path: Path, receipts: list[Json], output: Path) -> Json:
    """Readiness requires reachable parity and checks; it grants no benefit."""
    work = json.loads(work_path.read_bytes())
    primitive = checked(
        Path(work["primitive_reference"]["path"]), work["primitive_reference"]["sha256"]
    )
    rows = primitive["fixtures"]
    windows = primitive["history"].get("horizon_status_rows", [])
    failures = list(work["failures"])
    missing_checks = sorted(REQUIRED_CHECKS - {r["name"] for r in receipts}) if rows else []
    passed = bool(receipts) and all(r["passed"] for r in receipts) and not missing_checks
    parity = len(rows) == 27 and all(r["passed"] for r in rows)
    verdict = (
        "disqualified" if not passed or (rows and not parity) else "blocked" if failures else "null"
    )
    ready = int(passed and parity and not failures)
    games = sorted({r["game"] for r in windows})
    per_game = []
    for game in games:
        selected = [r for r in windows if r["game"] == game]
        per_game.append(
            dict(
                game=game,
                absolute_metric=0,
                refinement_eligible_score=0,
                window_count=len(selected),
                no_progress_count=sum(
                    r["horizon_status"] == "no_progress_within_horizon" for r in selected
                ),
                censored_count=sum(r["horizon_status"] == "censored" for r in selected),
                resolved_by_levelup_count=sum(
                    bool((r.get("legacy_receipt") or {}).get("resolved_by_levelup"))
                    for r in selected
                ),
                missing_reason="no_resolved_level_outcomes_for_refinement",
            )
        )
    value = dict(
        experiment_id=8624,
        task_id=TASK,
        milestone="2026.10.740",
        run_date=work["run_date"],
        honest_verdict=f"complete_{verdict}_arc_outcome_observer",
        verdict_class=verdict,
        rows=rows,
        intended_count=27,
        completed_count=len(rows),
        failed_count=sum(not r["passed"] for r in rows),
        censored_count=sum(r["horizon_status"] == "censored" for r in windows),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            mechanical_fixtures=27,
            mechanics_classes=3,
            historical_games=4,
            historical_episodes=16,
            new_independent_science_questions=0,
        ),
        inference_substrate="scripted_live_wrapper_and_stored_transition_reduction",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS, llm=0, numerical_head=0),
        historical_model_provenance=primitive["history"].get("historical_model_provenance", {}),
        target_model=dict(
            name=None,
            checkpoint=None,
            current_numerical_inference=False,
            reason="models_and_small_heads_disabled_in_current_fixtures",
        ),
        verifier_is_oracle=False,
        exposure_scope="mechanical_development_fixtures_and_exposed_public_history",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        arc_observer_ready_score=ready,
        refinement_eligible_score=0,
        promotion_ready_score=0,
        headline_solve_credit=0,
        solve_provenance="development_proxy",
        branch_gate_values=dict(
            arc_observer_ready_score=ready, refinement_eligible_score=0, promotion_ready_score=0
        ),
        observer_reachability=dict(
            reachable=parity,
            default_off=True,
            policy="E3AgentPolicy",
            wrapper="make_carnot_agent.choose_action",
            horizon=120,
        ),
        observer_parity_rows=rows,
        per_game_results=per_game,
        horizon_status_rows=windows,
        arm_window_support_rows=primitive["history"].get("arm_support_rows", []),
        missing_historical_fields=primitive["history"].get("missing_historical_fields", []),
        missing_observation_rows=[
            dict(
                game=r["game"],
                redirect_id=r["redirect_id"],
                reason=r["end_reason"],
                observed_horizon=r["observed_horizon"],
            )
            for r in windows
            if r["horizon_status"] == "censored"
        ],
        legacy_receipt_equivalence=dict(
            passed=parity, rewritten=False, fields=["resolved_by_levelup", "actions_to_levelup"]
        ),
        overlap_rows=[
            dict(
                game=r["game"],
                redirect_id=r["redirect_id"],
                overlapping_redirect_ids=r["overlapping_redirect_ids"],
                causal_attribution=None,
            )
            for r in windows
            if r["overlapping_redirect_ids"]
        ],
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
                check=r["name"],
                upstream=TASK,
                path=r["stdout_path"],
                hash=r["stdout_sha256"],
                field="exit_code",
                operator="==",
                expected=r["expected_exit"],
                observed=r["exit_code"],
                passed=False,
            )
            for r in receipts
            if not r["passed"]
        ],
        validation_receipts=receipts,
        adversarial_findings=[],
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"] + sum(r["duration_s"] for r in receipts),
        phase_spans=work["phase_spans"]
        + [
            dict(
                phase=r["name"],
                started_monotonic_ns=r.get("started_monotonic_ns"),
                ended_monotonic_ns=r.get("ended_monotonic_ns"),
                duration_s=r["duration_s"],
            )
            for r in receipts
        ],
        random_seed=8624,
        source_artifact_hashes=work["source_artifact_hashes"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=work["raw_shard_hashes"],
        work_reference=dict(path=str(work_path), sha256=sha256_file(work_path)),
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        cited_upstream_artifacts=[dict(path=p, sha256=h) for p, h in UPSTREAM.items()],
        registry_precheck=work["registry_precheck"],
        preserved_historical_outcomes=[
            "V733 source null",
            "V735 cold-cost null",
            "V737 ARC null",
            "Exp8603 disqualification retained; continuation line94 maps to executable line92",
            "V738 throughput CI95[1.0740525,1.1057774] does not meet NFR01",
        ],
        methodology_note="Horizon accounting qualifies instrumentation only. No policy, learner, deadline behavior, game solver, arm ranking or solve count changes.",
    )
    value["field_principles"] = {
        key: "Bound to this invocation; fixtures and stored observations do not establish causal benefit."
        for key in value
    }
    value["field_principles"].update(
        arc_observer_ready_score="Numeric0/1: reachable action-preserving instrumentation with all owned checks passed.",
        refinement_eligible_score="Numeric0: no newly resolved shared level outcomes; horizon status never opens the old gate.",
        promotion_ready_score="Numeric0: no policy or solve promotion is measured.",
        censored_count="Historical windows lacking a complete horizon; separate from27 mechanical validation units.",
        horizon_status_rows="120 distinct post-application observations establish no progress only. Censoring and level progress never assign arm credit.",
        missing_observation_rows="Retain short windows and missing support. Historical terminal state was not captured and remains unknown.",
        historical_model_provenance="The old common CNN head and generator pin are imported evidence; current numerical and language-model calls are zero.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(value: Json) -> bool:
    """Fresh replay checks original operands and recomputes every derived window."""
    ref = value["work_reference"]
    work = checked(Path(ref["path"]), ref["sha256"])
    for field in ["source_artifact_hashes", "code_config_hashes", "raw_shard_hashes"]:
        for path, digest in work[field].items():
            checked_bytes(Path(path), digest)
    primitive = checked(
        Path(work["primitive_reference"]["path"]), work["primitive_reference"]["sha256"]
    )
    history = primitive["history"]
    if history:
        fresh = historical(Path(work["input_root"]))
        if fresh != history:
            raise ValueError("historical_reduction_drift")
    for row in primitive["fixtures"]:
        if not row["passed"] or row["absolute_metric"] != int(all(row["equality"].values())):
            raise ValueError("fixture_parity_drift")
    if primitive["fixtures"]:
        from carnot.reporting.arc_observer_fixtures_8624 import qualify

        if qualify() != primitive["fixtures"]:
            raise ValueError("fixture_reduction_drift")
    for receipt in value["validation_receipts"]:
        for stream in ["stdout", "stderr"]:
            checked_bytes(Path(receipt[stream + "_path"]), receipt[stream + "_sha256"])
    output = Path(value["terminal_validation_sidecar_path"]).parents[2] / (NAME + ".json")
    return value == build(Path(ref["path"]), value["validation_receipts"], output)
