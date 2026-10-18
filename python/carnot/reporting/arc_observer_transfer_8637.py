"""REQ-REPORT-8637: real public horizons qualify recording, never arm benefit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting.arc_observer_8624 import checked, checked_bytes, reduce_stream
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8637_v741_arc_observer_transfer"
TASK = "exp8637-arc-observer-transfer"
CLI = f"scripts/experiments/{NAME}.py"
MODEL_SPECS: list[Json] = []
TESTS = ["tests/python/test_arc_observer_transfer_8637.py"]
OWNED = [
    f"python/carnot/reporting/{n}.py"
    for n in ["arc_observer_transfer_8637", "arc_observer_live_8637", "arc_observer_runner_8637"]
] + [CLI]
PINS = {
    "results/experiment_8624_v740_arc_outcome_observer.json": "sha256:54506646569b3456b47b5255a34da33699e2ffadaa07195a8a879f562e0bce3a",
    "results/experiment_8627_v741_contract_methods.json": "sha256:e8968eba7c4bdaee750e51cb0eb1f003c7d842521a97b47eead3dee416983b98",
}
ARMS = ["drop_goal_bias", "allow_reinduction", "force_exploration_diversity"]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Show real phase boundaries while bounded children keep their own clocks."""
    print(f"[exp8637] phase={phase} completed={completed} pending={pending}", flush=True)


def authenticate(root: Path) -> Json:
    """Independent pins prevent rehashed envelopes from replacing original authority."""
    observer, methods = [checked(root / p, h) for p, h in PINS.items()]
    for value, field, name in [
        (observer, "arc_observer_ready_score", next(iter(PINS))),
        (methods, "arc_inputs_ready_score", list(PINS)[1]),
    ]:
        if not (
            value[field] == 1
            and value["required_checks_passed"]
            and value["flagged_adversarial"] is False
        ):
            raise ValueError("upstream_readiness:" + field)
        side = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
        bound = read_bound_sidecar(root / name, Path(side["publication"]["sidecar_path"]))
        if not bound["report"]["passed"]:
            raise ValueError("original_publication")
    for path, digest in observer["code_config_hashes"].items():
        checked_bytes(Path(path), digest)
    ref = methods["methods_protocol_reference"]
    protocol = checked(Path(ref["path"]), ref["sha256"])["arc_observer_protocol"]
    if protocol != methods["arc_observer_protocol"]:
        raise ValueError("registered_protocol")
    ref = protocol["frozen_panel_reference"]
    checked(Path(ref["path"]), ref["sha256"])
    return dict(
        protocol=protocol, historical_model_provenance=observer["historical_model_provenance"]
    )


def units(protocol: Json) -> list[Json]:
    """Keep every original slot; both conditions apply the same masked supervisor."""
    return [
        dict(
            episode_id=f"{g}:{s}:{o}",
            pair_key=f"{g}:{s}",
            game=g,
            seed=s,
            observer=o,
            mode=o,
            arm="on",
            capability_mask=True,
            max_actions=protocol["max_actions"],
            max_seconds=protocol["max_seconds"],
        )
        for g in protocol["games"]
        for s in protocol["seeds"]
        for o in protocol["observer_conditions"]
    ]


def replay_stream(steps: list[Json], events: list[Json], reason: str) -> list[Json]:
    """Rebuild windows using the qualified reducer and retain the actual cutoff."""
    rows = reduce_stream(steps, events)
    for row in rows:
        if row["end_reason"] == "action_cap":
            row["end_reason"] = reason
    return rows


def parity(left: Json, right: Json) -> Json:
    """Compare deterministic behavior fields; wall clocks are recorded separately."""
    equality = {
        k: left[k] == right[k] for k in ["actions", "rng", "termination", "legacy", "observations"]
    }
    return dict(
        equality=equality,
        passed=all(equality.values()),
        absolute_metric=int(all(equality.values())),
    )


def support(rows: list[Json], games: list[str]) -> tuple[list[Json], list[Json]]:
    """Zero firing cells remain denominators and the historical support floor stays fixed."""
    cells = [
        dict(
            game=g,
            arm=a,
            firings=sum(r["game"] == g and r["arm"] == a for r in rows),
            observer_counts={
                o: sum(
                    r["game"] == g and r["arm"] == a and r.get("observer", "on") == o for r in rows
                )
                for o in ["off", "on"]
            },
            episode_denominators=dict(off=2, on=2),
            complete_horizons=sum(
                r["game"] == g
                and r["arm"] == a
                and r["horizon_status"] == "no_progress_within_horizon"
                for r in rows
            ),
        )
        for g in games
        for a in ARMS
    ]
    arms = []
    for arm in ARMS:
        shared = [
            c["game"]
            for c in cells
            if c["arm"] == arm and all(n > 0 for n in c["observer_counts"].values())
        ]
        arms.append(
            dict(
                arm=arm,
                overlapping_games=shared,
                descriptive_support=len(shared) >= 3
                and all(
                    min(c["observer_counts"].values()) >= 5
                    for c in cells
                    if c["arm"] == arm and c["game"] in shared
                ),
                refinement_authorized=False,
            )
        )
    return cells, arms


def build(work_path: Path, receipts: list[Json], output: Path) -> Json:
    """Derive all claims from retained public streams and actual child outcomes."""
    work = json.loads(work_path.read_bytes())
    episodes = [checked(Path(r["path"]), r["sha256"]) for r in work["references"]]
    horizons, pairs, rows = [], [], []
    for episode in episodes:
        unit = episode["unit"]
        reduced = replay_stream(episode["steps"], episode["events"], episode["end_reason"])
        if unit["observer"] == "on" and reduced != episode["observer_rows"]:
            raise ValueError("live_replay_drift")
        horizons.extend(
            dict(
                r,
                game=unit["game"],
                seed=unit["seed"],
                episode_id=unit["episode_id"],
                observer=unit["observer"],
                classification={
                    "level_progress_observed": "progress_associated",
                    "no_progress_within_horizon": "full120_action_no_progress",
                    "censored": "explicitly_censored",
                }[r["horizon_status"]],
            )
            for r in reduced
        )
        rows.append(
            dict(
                episode_id=unit["episode_id"],
                game=unit["game"],
                seed=unit["seed"],
                observer=unit["observer"],
                actions_used=len(episode["actions"]),
                status="completed",
                absolute_metric=0,
                missing_reason=None,
                termination=episode["termination"],
                numerical_head_calls=episode["numerical_head_calls"],
            )
        )
    for key in dict.fromkeys(ep["unit"]["pair_key"] for ep in episodes):
        selected = [ep for ep in episodes if ep["unit"]["pair_key"] == key]
        pairs.append(
            dict(pair_key=key, **parity(selected[0], selected[1]))
            if len(selected) == 2
            else dict(pair_key=key, passed=False, equality={}, absolute_metric=0)
        )
    games = work["games"]
    cells, arms = support(horizons, games)
    passed = bool(receipts) and all(r["passed"] for r in receipts)
    behavior = len(episodes) == 16 and all(r["passed"] for r in pairs)
    owned_bad = any(not r["passed"] for r in receipts) or bool(episodes and not behavior)
    verdict = "disqualified" if owned_bad else "blocked" if work["failures"] else "null"
    ready = int(passed and behavior and not work["failures"])
    measurement_names = {r["name"] for r in work.get("measurement_receipts", [])}
    zeros = dict(
        refinement_eligible_score=0,
        promotion_ready_score=0,
        headline_solve_credit=0,
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
    )
    value = dict(
        experiment_id=8637,
        task_id=TASK,
        milestone="2026.10.741",
        run_date=work["run_date"],
        honest_verdict=f"complete_{verdict}_arc_observer_transfer",
        verdict_class=verdict,
        rows=rows,
        intended_count=16,
        completed_count=len(rows),
        failed_count=int(owned_bad),
        censored_count=sum(r["horizon_status"] == "censored" for r in horizons),
        excluded_count=0,
        independent_count=len({r["game"] for r in rows}),
        sample_size_budget=dict(
            episodes=16,
            games=4,
            seeds=2,
            conditions=2,
            max_actions=512,
            max_seconds=120,
            horizon=120,
            independent_unit="game",
        ),
        inference_substrate="official_SDK_live_E3AgentPolicy_public_observations",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(
            llm=sum(ep.get("llm_calls", 0) for ep in episodes),
            language_model=sum(ep.get("llm_calls", 0) for ep in episodes),
            numerical_head=sum(ep["numerical_head_calls"] for ep in episodes),
        ),
        historical_model_provenance=work["historical_model_provenance"],
        target_model=work.get("numerical_head_reference"),
        verifier_is_oracle=False,
        exposure_scope="historically_exposed_public_games_adapters_withheld",
        **zeros,
        real_horizon_rows_ready_score=ready,
        branch_gate_values=dict(real_horizon_rows_ready_score=ready, **zeros),
        solve_provenance="live_agent_self_discovery",
        development_controls_provenance="development_proxy",
        actual_redirect_rows=[
            dict(
                r,
                original_legacy_receipt=next(
                    ev.get("original_legacy_receipt")
                    for ep in episodes
                    if ep["unit"]["episode_id"] == r["episode_id"]
                    for ev in ep["events"]
                    if ev["event_id"] == r["redirect_id"]
                ),
            )
            for r in horizons
        ],
        horizon_status_rows=horizons,
        observer_action_parity_rows=pairs,
        censored_reason_rows=[r for r in horizons if r["horizon_status"] == "censored"],
        overlap_rows=[r for r in horizons if r["overlapping_redirect_ids"]],
        support_by_game_arm=cells,
        descriptive_arm_support=arms,
        record_completeness_comparison=[
            dict(
                episode_id=ep["unit"]["episode_id"],
                same_actual_episode=True,
                redirects=len(ep["events"]),
                current_public_returns=len(ep["steps"]),
                historical_schema_has_terminal_and_reset=False,
                current_missing_observations=sum(s["observation"] is None for s in ep["steps"]),
            )
            for ep in episodes
        ],
        per_game_results=[
            dict(
                game=g,
                episodes=sum(r["game"] == g for r in rows),
                redirects=sum(r["game"] == g for r in horizons),
                full_horizons=sum(
                    r["game"] == g and r["observed_horizon"] == 120 for r in horizons
                ),
            )
            for g in games
        ],
        registry_precheck=work["registry_precheck"],
        required_checks_passed=passed,
        flagged_adversarial=owned_bad,
        acceptance_gates={r["name"]: r["passed"] for r in receipts},
        gate_check_summary=work["failures"],
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        adversarial_findings=[],
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"]
        + sum(r["duration_s"] for r in receipts if r["name"] not in measurement_names),
        phase_spans=work["phase_spans"]
        + [
            dict(
                phase=r["name"],
                duration_s=r["duration_s"],
                started_monotonic_ns=r.get("started_monotonic_ns"),
                ended_monotonic_ns=r.get("ended_monotonic_ns"),
            )
            for r in receipts
            if r["name"] not in measurement_names
        ],
        random_seed=8637,
        source_artifact_hashes=work["source_artifact_hashes"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=dict(
            {r["path"]: r["sha256"] for r in work["references"]},
            **work.get("additional_raw_hashes", {}),
        ),
        work_reference=dict(path=str(work_path), sha256=sha256_file(work_path)),
        cited_upstream_artifacts=[dict(path=p, sha256=h) for p, h in PINS.items()],
        scope_closed=verdict == "null",
        hidden_game_performance="unmeasured",
        methodology_note="Natural redirects only; overlapping and censored windows assign no causal arm wins. Fixed budgets are never extended. Instrumentation support grants no solve or priority promotion.",
        preserved_historical_outcomes=[
            "V733 source null",
            "V735 cold-cost null",
            "V737 ARC null",
            "Exp8616 lost original shards remains disqualified",
            "Exp8620 55/190 source-window limits",
            "Exp8622 NFR01 unmet",
        ],
    )
    value["field_principles"] = {
        key: "Invocation-bound public evidence; missing differs from zero; instrumentation is not causal benefit."
        for key in value
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(value: Json) -> bool:
    """Check the external measurement seal before rebuilding all candidate claims."""
    ref = value["work_reference"]
    path = Path(ref["path"])
    seal = json.loads(path.with_name("measurement_seal.json").read_bytes())
    if seal["work_sha256"] != ref["sha256"]:
        raise ValueError("measurement_seal")
    work = checked(path, ref["sha256"])
    for field in ["source_artifact_hashes", "code_config_hashes", "additional_raw_hashes"]:
        for name, digest in work.get(field, {}).items():
            checked_bytes(Path(name), digest)
    if "original_measurement_reference" in work:
        original = work["original_measurement_reference"]
        measured = checked(Path(original["path"]), original["sha256"])
        if measured["references"] != work["references"]:
            raise ValueError("original_measurement_substitution")
    for receipt in value["validation_receipts"]:
        for stream in ["stdout", "stderr"]:
            if stream + "_path" in receipt:
                checked_bytes(Path(receipt[stream + "_path"]), receipt[stream + "_sha256"])
    output = Path(value["terminal_validation_sidecar_path"]).parents[2] / (NAME + ".json")
    return value == build(path, value["validation_receipts"], output)


TASK_PIN = "sha256:e8b86e20e713b3d65cf40e3adf84763144af8b6b945f36cca0f72794de3c4831"
