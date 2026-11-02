"""REQ-REPORT-8650: distinguish authentic reset causes from supervisor efficacy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import arc_reset_8649 as original
from carnot.reporting.arc_observer_8624 import checked, checked_bytes
from carnot.reporting.arc_reset_live_8649 import parity
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8650_v742_arc_reset_transfer"
TASK = "exp8650-arc-reset-transfer"
TASK_PIN = "sha256:6d77652fe12d41867844cc59000c75895ef8bf49bcd1289e05cf3609ad85ea5a"
CLI = f"scripts/experiments/{NAME}.py"
MODEL_SPECS: list[Json] = []
TESTS = [
    "tests/python/test_arc_reset_transfer_8650.py",
    "tests/python/test_arc_reset_runner_8650.py",
]
OWNED = [
    "python/carnot/reporting/arc_reset_transfer_8650.py",
    "python/carnot/reporting/arc_reset_runner_8650.py",
    CLI,
]
PINS = {
    "results/experiment_8640_v742_contract_methods.json": "sha256:7abca18566c145b3b197d017fb6a955bf047952ed4325460ba2147c749b33608",
    "results/experiment_8649_v742_arc_reset_provenance.json": "sha256:5ed5072ff6fc398e070f353c7a8190d8113ae4ac43fa4a174ef9f6250f0795e8",
}
GAMES = ["ar25-0c556536", "ft09-0d8bbf25", "lf52-271a04aa", "tn36-ef4dde99"]
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
REQUEST_BRANCHES = {
    "StepwiseExplorer._serve.queued_reset",
    "StepwiseExplorer.next_move.bootstrap",
    "CarnotAgentPolicy.next_move.initial_reset",
    "E3AgentPolicy.induce.plan_needs_reset",
    "CarnotAgent.choose_action.none_fallback",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Expose phase boundaries while the parent supervises potentially slow SDK calls."""
    print(f"[exp8650] phase={phase} completed={completed} pending={pending}", flush=True)


def authenticate(root: Path) -> Json:
    """Original producer pins and sealed closures cannot be replaced by live-path hashes."""
    sources = [checked(root / p, h) for p, h in PINS.items()]
    fields = ["arc_inputs_ready_score", "reset_provenance_ready_score"]
    for value, name, field in zip(sources, PINS, fields, strict=True):
        if value[field] != 1 or not value["required_checks_passed"] or value["flagged_adversarial"]:
            raise ValueError("upstream_readiness:" + field)
        side = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
        if not read_bound_sidecar(root / name, Path(side["publication"]["sidecar_path"]))["report"][
            "passed"
        ]:
            raise ValueError("upstream_publication")
    history = original.authenticate(root)
    ref = sources[1]["work_reference"]
    work = checked(Path(ref["path"]), ref["sha256"])
    for name, digest in work["source_artifact_hashes"].items():
        if "/current_source_closure/python/" in name:
            relative = name.split("/current_source_closure/", 1)[1]
            checked_bytes(Path(name), digest)
            checked_bytes(ROOT / relative, digest)
    return dict(
        arc_inputs_ready_score=1,
        reset_provenance_ready_score=1,
        historical_model_provenance=history["historical_model_provenance"],
    )


def units() -> list[Json]:
    """The original eight game/seed cells each supply one dependent off/on pair."""
    return [
        dict(
            episode_id=f"{g}:{s}:{mode}",
            pair_key=f"{g}:{s}",
            game=g,
            seed=s,
            reset_provenance=mode,
            observer="on",
            arm="on",
            capability_mask=True,
            max_actions=512,
            max_seconds=120,
        )
        for g in GAMES
        for s in (11, 22)
        for mode in ("off", "on")
    ]


def reduce(episodes: list[Json]) -> Json:
    """Join only observed request sites; zero or ambiguous support cannot establish transfer."""
    pairs, requests, joins, missing, censored, causes = [], [], [], [], [], []
    complete = 0
    clean = True
    for game in GAMES:
        for seed in (11, 22):
            selected = [
                ep for ep in episodes if ep["unit"]["game"] == game and ep["unit"]["seed"] == seed
            ]
            modes = {ep["unit"]["reset_provenance"]: ep for ep in selected}
            passed = (
                len(selected) == 2
                and set(modes) == {"off", "on"}
                and parity(modes["off"], modes["on"])["passed"]
            )
            pairs.append(dict(game=game, seed=seed, passed=passed, absolute_metric=int(passed)))
            ep = modes.get("on")
            if ep is None:
                continue
            context = dict(game=game, seed=seed, episode_id=ep["unit"]["episode_id"])
            ledger = ep["reset_provenance"]["reset_request_rows"]
            observed = {r["action_index"]: r for r in ep["reset_observations"]}
            authentic = []
            for row in ledger:
                requests.append(dict(row, **context))
            for index in sorted(set(observed) | {r["action_index"] for r in ledger}):
                candidates = [r for r in ledger if r["action_index"] == index]
                r = candidates[0] if len(candidates) == 1 else {}
                known = (
                    r.get("origin") in {"agent_reset", "harness_reset"}
                    and r.get("initiating_policy_branch") in REQUEST_BRANCHES
                    and index in observed
                    and r.get("sdk_status") is not None
                    and r["sdk_status"].get("error") is None
                    and r.get("post_observation_sha256") == observed[index]["observation_sha256"]
                )
                if known:
                    authentic.append(r)
                else:
                    missing.append(
                        dict(
                            **context,
                            reset_provenance="on",
                            action_index=index,
                            origin="unknown",
                            missing_reason="missing_conflicting_or_unbound_request",
                            candidate_request_ids=[c["reset_request_id"] for c in candidates],
                        )
                    )
            supported = (
                bool(observed) and len(authentic) == len(observed) and len(ledger) == len(observed)
            )
            causes.append(
                dict(
                    game=game,
                    seed=seed,
                    support_count=int(supported),
                    observed_reset_count=len(observed),
                    authentic_origin_count=len(authentic),
                )
            )
            clean &= not (
                ep["recorder_errors"] or ep["reset_provenance"]["issues"] or ep["observer_issues"]
            )
            for row in ep["observer_rows"]:
                complete += int(row["horizon_status"] != "censored")
                if row["horizon_status"] != "censored":
                    continue
                censored.append(dict(row, **context))
                # RESET closes a pending window before that new action is observed.
                cutoff = row["application_index"] + row["observed_horizon"]
                candidates = [r for r in authentic if r["action_index"] == cutoff]
                r = candidates[0] if len(candidates) == 1 and row["end_reason"] == "reset" else {}
                joins.append(
                    dict(
                        **context,
                        reset_provenance="on",
                        redirect_id=row["redirect_id"],
                        reset_request_id=r.get("reset_request_id"),
                        origin=r.get("origin", "unknown"),
                        initiating_policy_branch=r.get("initiating_policy_branch"),
                        observed_horizon=row["observed_horizon"],
                        end_reason=row["end_reason"],
                        remaining_actions=r.get("remaining_actions"),
                        remaining_time_s=r.get("remaining_time_s"),
                        collisions=row["overlapping_redirect_ids"],
                        missing_reason=None if r else "no_unique_reset_at_cutoff",
                    )
                )
            for row in modes.get("off", {}).get("observer_rows", []):
                complete += int(row["horizon_status"] != "censored")
                if row["horizon_status"] == "censored":
                    off_context = dict(
                        context,
                        episode_id=modes["off"]["unit"]["episode_id"],
                        reset_provenance="off",
                    )
                    censored.append(dict(row, **off_context))
                    joins.append(
                        dict(
                            **off_context,
                            redirect_id=row["redirect_id"],
                            reset_request_id=None,
                            origin="unknown",
                            observed_horizon=row["observed_horizon"],
                            end_reason=row["end_reason"],
                            remaining_actions=None,
                            remaining_time_s=None,
                            collisions=row["overlapping_redirect_ids"],
                            missing_reason="reset_provenance_disabled",
                        )
                    )
    clean &= all(
        not ep["termination"]["error"]
        and ep["llm_calls"] == 0
        and ep["numerical_head_calls"] > 0
        and not ep["model_tripwire_attempts"]
        for ep in episodes
    )
    ready = len(episodes) == 16 and all(r["passed"] for r in pairs) and clean
    per_game = [
        dict(
            game=g,
            reset_support_count=sum(r["support_count"] for r in causes if r["game"] == g),
            observed_reset_count=sum(r["observed_reset_count"] for r in causes if r["game"] == g),
            authentic_origin_count=sum(
                r["authentic_origin_count"] for r in causes if r["game"] == g
            ),
            missing_cause_count=sum(r["game"] == g for r in missing),
            cause_coverage={
                branch: sum(
                    r["game"] == g and r["initiating_policy_branch"] == branch for r in requests
                )
                for branch in sorted(REQUEST_BRANCHES)
            },
            missing_reason=None
            if any(r["support_count"] for r in causes if r["game"] == g)
            else "no_authentic_reset_support",
        )
        for g in GAMES
    ]
    return dict(
        reset_transfer_rows_ready_score=int(ready),
        diagnostic_transfer_score=int(
            ready and not missing and all(r["reset_support_count"] > 0 for r in per_game)
        ),
        action_parity_rows=pairs,
        reset_request_rows=requests,
        redirect_reset_joins=joins,
        cause_support_rows=causes,
        per_game_results=per_game,
        missing_cause_rows=missing,
        censored_reason_rows=censored,
        complete_horizon_count=complete,
    )


def build(path: Path, receipts: list[Json], output: Path) -> Json:
    """Derive every published claim from immutable streams and real validation exits."""
    work = json.loads(path.read_bytes())
    episodes = [checked(Path(r["path"]), r["sha256"]) for r in work["references"]]
    result = reduce(episodes)
    checks = bool(receipts) and all(r["passed"] for r in receipts)
    checks &= not episodes or REQUIRED_CHECKS <= {r["name"] for r in receipts}
    bad = not checks or bool(episodes and not result["reset_transfer_rows_ready_score"])
    verdict = "disqualified" if bad else "blocked" if work["failures"] else "null"
    result["reset_transfer_rows_ready_score"] *= int(checks and not work["failures"])
    result["diagnostic_transfer_score"] *= result["reset_transfer_rows_ready_score"]
    zeros = dict(
        policy_refinement_score=0,
        promotion_ready_score=0,
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
    )
    scores = dict(
        **zeros,
        **work["upstream_gates"],
        reset_transfer_rows_ready_score=result["reset_transfer_rows_ready_score"],
        diagnostic_transfer_score=result["diagnostic_transfer_score"],
    )
    present = {ep["unit"]["episode_id"]: ep for ep in episodes}
    rows = [
        dict(
            **u,
            status="completed" if u["episode_id"] in present else "blocked",
            absolute_metric=len(present[u["episode_id"]]["actions"])
            if u["episode_id"] in present
            else None,
            missing_reason=None
            if u["episode_id"] in present
            else "upstream_or_episode_unavailable",
        )
        for u in units()
    ]
    measurement_names = {r["name"] for r in work["measurement_receipts"]}
    validation = [r for r in receipts if r["name"] not in measurement_names]
    value = dict(
        experiment_id=8650,
        task_id=TASK,
        milestone="2026.10.742",
        run_date=work["run_date"],
        honest_verdict=f"complete_{verdict}_arc_reset_transfer",
        verdict_class=verdict,
        inference_substrate="CPU_official_SDK_public_observations_small_frame_change_CNN",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(
            llm=0,
            language_model=0,
            numerical_head=sum(ep["numerical_head_calls"] for ep in episodes),
        ),
        historical_model_provenance=work["historical_model_provenance"],
        target_model=work.get("numerical_head_reference"),
        rows=rows,
        intended_count=16,
        completed_count=len(episodes),
        failed_count=int(bad),
        censored_count=len(result["censored_reason_rows"]),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            episodes=16,
            paired_game_seed_cells=8,
            games=4,
            max_actions=512,
            max_seconds=120,
            horizon=120,
        ),
        verifier_is_oracle=False,
        exposure_scope="public_games_adapter_withheld_development_panel",
        **zeros,
        **result,
        branch_gate_values=scores,
        arc_inputs_ready_score=work["upstream_gates"].get("arc_inputs_ready_score", 0),
        reset_provenance_ready_score=work["upstream_gates"].get("reset_provenance_ready_score", 0),
        registry_precheck=work["registry_precheck"],
        solve_provenance="live_agent_self_discovery",
        headline_solve_credit=0,
        required_checks_passed=checks,
        flagged_adversarial=bad,
        acceptance_gates={r["name"]: r["passed"] for r in receipts},
        gate_check_summary=work["failures"],
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / NAME / "terminal_validation.json"
        ),
        adversarial_findings=[],
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"] + sum(r["duration_s"] for r in validation),
        phase_spans=work["phase_spans"]
        + [dict(phase=r["name"], duration_s=r["duration_s"]) for r in validation],
        random_seed=8650,
        source_artifact_hashes=work["source_artifact_hashes"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes={r["path"]: r["sha256"] for r in work["references"]},
        cited_upstream_artifacts=[dict(path=p, sha256=h) for p, h in PINS.items()],
        work_reference=dict(path=str(path), sha256=sha256_file(path)),
        scope_closed=verdict == "null",
        preserved_history=dict(
            censored_redirects=16,
            complete_horizons=0,
            verdict="complete_null_arc_observer_transfer",
            quarantines_unchanged=True,
        ),
        methodology_note="Reset origin transfer is diagnostic only. Off/on twins are dependent. Censored horizons establish no efficacy; no significance test or level-solve claim.",
    )
    value["field_principles"] = {
        k: "Invocation-bound evidence; unknown causes and missing values remain explicit; diagnostic support gives no policy benefit."
        for k in value
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(value: Json) -> bool:
    """An independently retained seal rejects substituted work before reducing again."""
    ref = value["work_reference"]
    path = Path(ref["path"])
    seal = json.loads(path.with_name("measurement_seal.json").read_bytes())
    if seal["work_sha256"] != ref["sha256"]:
        raise ValueError("measurement_seal")
    work = checked(path, ref["sha256"])
    for field in ("source_artifact_hashes", "code_config_hashes", "raw_file_hashes"):
        for name, digest in work.get(field, {}).items():
            checked_bytes(Path(name), digest)
    for receipt in value["validation_receipts"]:
        for stream in ("stdout", "stderr"):
            if stream + "_path" in receipt:
                checked_bytes(Path(receipt[stream + "_path"]), receipt[stream + "_sha256"])
    output = Path(value["terminal_validation_sidecar_path"]).parents[2] / (NAME + ".json")
    return value == build(path, value["validation_receipts"], output)
