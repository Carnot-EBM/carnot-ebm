"""REQ-REPORT-8678: preserve real horizon censorship after delivery qualification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.agentic.arc_trajectory_supervisor import ARM_ORDER
from carnot.reporting import arc_observation_8677 as qualified
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
ROOT = qualified.ROOT
NAME = "experiment_8678_v744_arc_observer_transfer"
TASK = "exp8678-arc-observer-transfer"
CLI = f"scripts/experiments/{NAME}.py"
MODEL_SPECS: list[Json] = []
TESTS = [
    "tests/python/test_arc_observer_transfer_8678.py",
    "tests/python/test_arc_observer_transfer_runner_8678.py",
]
TEST = TESTS[0]
BASE_BUILD = qualified.build
OWNED = [
    f"python/carnot/reporting/arc_observer_transfer{suffix}_8678.py"
    for suffix in ("", "_live", "_runner")
] + [CLI]
PINS = {
    f"results/{qualified.NAME}.json": "sha256:6a6cfe0b065cbe0a30e2b45a4dde6683e90bc203025a4ff96c24aad24116aa8a",
    qualified.UPSTREAM: qualified.UPSTREAM_PIN,
}
GATE_FIELDS = ("observer_delivery_ready_score", "arc_inputs_ready_score")
GAMES = qualified.GAMES
REQUIRED_CHECKS = qualified.REQUIRED_CHECKS
reference, read, checked = qualified.reference, qualified.read, qualified.checked
empty_work = qualified.empty_work


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries so finite CPU children cannot silently stall."""
    print(f"[exp8678] phase={phase} completed={completed} pending={pending}", flush=True)


def gates(root: Path) -> list[Json]:
    """A changed operand differs from an absent field or a measured zero."""
    rows = []
    for (name, pin), field in zip(PINS.items(), GATE_FIELDS, strict=True):
        path = root / name
        value = json.loads(path.read_bytes()) if path.is_file() else {}
        observed = value.get(field)
        digest = sha256_file(path) if path.is_file() else None
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
        rows.append(
            dict(
                upstream="exp8677-arc-observation-ownership"
                if "8677" in name
                else "exp8667-contract-methods",
                check="declared_arc_gate",
                path=str(path),
                hash=digest,
                expected_hash=pin,
                field=field,
                operator="==",
                expected=1,
                observed=observed,
                observed_state=state,
                passed=type(observed) is int and observed == 1 and digest == pin,
                producer_quarantined=value.get("flagged_adversarial"),
            )
        )
    return rows


def authenticate() -> Json:
    """Only the qualified observer authorizes execution; historical failures stay sealed."""
    upstream = read(dict(path=str(ROOT / next(iter(PINS))), sha256=next(iter(PINS.values()))))
    if (
        not upstream["required_checks_passed"]
        or upstream["flagged_adversarial"]
        or upstream["observer_delivery_ready_score"] != 1
    ):
        raise ValueError("observer_qualification")
    side = json.loads(Path(upstream["terminal_validation_sidecar_path"]).read_bytes())
    if not read_bound_sidecar(ROOT / next(iter(PINS)), Path(side["publication"]["sidecar_path"]))[
        "report"
    ]["passed"]:
        raise ValueError("upstream_publication")
    for name, digest in upstream["code_config_hashes"].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError("qualified_source_changed:" + name)
    work = read(upstream["work_reference"])
    for name, digest in work["source_artifact_hashes"].items():
        if "/source_closure/python/" in name:
            current = ROOT / name.split("/source_closure/", 1)[1]
            if sha256_file(Path(name)) != digest or sha256_file(current) != digest:
                raise ValueError("qualified_source_closure:" + name)
    protocol: Json = dict(qualified.authenticate())
    protocol.update(
        qualified_reference=reference(ROOT / next(iter(PINS))),
        diagnosis_rows=upstream["diagnosis_rows"],
        controls=upstream["horizon_control_rows"],
        qualified_source_hashes=upstream["code_config_hashes"],
    )
    return protocol


def units() -> list[Json]:
    """Eight dependent twins retain four independent games and the registered quotas."""
    return [
        dict(
            game=g,
            seed=s,
            observer="on",
            reset_provenance=mode,
            arm="on",
            capability_mask=True,
            max_actions=512,
            max_seconds=120,
            episode_id=f"{g}:{s}:{mode}",
        )
        for g in GAMES
        for s in (11, 23)
        for mode in ("off", "on")
    ]


def reduce(episodes: list[Json]) -> Json:
    """Use games as support units and retain censored windows without negative credit."""
    pairs: list[Json] = []
    callbacks: list[Json] = []
    causes: list[Json] = []
    horizons: list[Json] = []
    errors: list[Json] = []
    for game in GAMES:
        for seed in (11, 23):
            selected = [
                ep for ep in episodes if ep["unit"]["game"] == game and ep["unit"]["seed"] == seed
            ]
            arms = {ep["unit"]["reset_provenance"]: ep for ep in selected}
            keys = (
                "actions",
                "observations",
                "termination",
                "legacy",
                "rng",
                "torch_rng",
                "sdk_calls",
                "final_state",
                "observer_rows",
            )
            equality = {k: arms.get("off", {}).get(k) == arms.get("on", {}).get(k) for k in keys}
            clean = len(selected) == 2 and set(arms) == {"off", "on"} and all(equality.values())
            for ep in selected:
                canonical = [
                    r["action_index"] for r in ep["delivery_rows"] if r["status"] == "canonical"
                ]
                clean &= bool(canonical) and canonical == list(range(1, len(ep["actions"]) + 1))
                clean &= not (
                    ep["observer_issues"]
                    or ep["recorder_error_metadata"]
                    or ep["reset_provenance"]["issues"]
                    or ep["termination"]["error"]
                    or ep["llm_calls"]
                    or ep["model_tripwire_attempts"]
                )
                clean &= ep["sdk_identity"]["official"]
            pairs.append(
                dict(
                    game=game,
                    seed=seed,
                    equality=equality,
                    passed=bool(clean),
                    absolute_metric=int(clean),
                )
            )
    for ep in episodes:
        context = dict(
            game=ep["unit"]["game"],
            seed=ep["unit"]["seed"],
            recorder=ep["unit"]["reset_provenance"],
            episode_id=ep["unit"]["episode_id"],
        )
        callbacks.extend(dict(r, **context) for r in ep["delivery_rows"])
        causes.extend(dict(r, **context) for r in ep["reset_provenance"]["reset_request_rows"])
        horizons.extend(dict(r, **context) for r in ep["observer_rows"])
        errors.extend(
            dict(r, **context) for r in ep["observer_issues"] + ep["recorder_error_metadata"]
        )
    complete = [
        r
        for r in horizons
        if r["horizon_status"] in ("no_progress_within_horizon", "level_progress_observed")
    ]
    return dict(
        action_parity_rows=pairs,
        callback_sequence_rows=callbacks,
        reset_cause_rows=causes,
        horizon_rows=horizons,
        complete_horizon_count=len(complete),
        observer_error_rows=errors,
        censored_reason_rows=[r for r in horizons if r["horizon_status"] == "censored"],
        per_game_results=[
            dict(
                game=g,
                paired_cells=sum(r["game"] == g and r["passed"] for r in pairs),
                complete_horizons=sum(r["game"] == g for r in complete),
                actions=sum(len(ep["actions"]) for ep in episodes if ep["unit"]["game"] == g),
                official_levels=max(
                    (
                        o["level"]
                        for ep in episodes
                        if ep["unit"]["game"] == g
                        for o in ep["observations"]
                    ),
                    default=None,
                ),
                episode_compute_s=sum(
                    ep["duration_s"] for ep in episodes if ep["unit"]["game"] == g
                ),
            )
            for g in GAMES
        ],
        cross_game_support_rows=[
            dict(
                arm=arm,
                independent_count=len({r["game"] for r in complete if r["arm"] == arm}),
                population_efficacy_claim=False,
                per_game=[
                    dict(
                        game=g,
                        complete_horizons=sum(r["game"] == g and r["arm"] == arm for r in complete),
                        censored_horizons=sum(
                            r["game"] == g and r["arm"] == arm and r["horizon_status"] == "censored"
                            for r in horizons
                        ),
                    )
                    for g in GAMES
                ],
            )
            for arm in ARM_ORDER
        ],
    )


def build(path: Path, receipts: list[Json], output: Path) -> Json:
    """Reuse terminal evidence rules while assigning this invocation its own identity."""
    with patch.multiple(
        qualified, NAME=NAME, TASK=TASK, reduce=reduce, REQUIRED_CHECKS=REQUIRED_CHECKS
    ):
        value: Json = dict(BASE_BUILD(path, receipts, output))
    ready = value["observer_delivery_ready_score"]
    work = json.loads(path.read_bytes())
    value.update(
        experiment_id=8678,
        random_seed=8678,
        honest_verdict=f"complete_{value['verdict_class']}_arc_observer_transfer",
        arc_rows_ready_score=ready,
        branch_gate_values=dict(arc_rows_ready_score=ready, promotion_ready_score=0),
        gate_check_summary=work.get("upstream_gate_checks", []) + work["failures"],
        diagnostic_closed=bool(ready and value["complete_horizon_count"] == 0),
        diagnostic_disposition="closed_censored_under_unchanged_protocol"
        if ready and value["complete_horizon_count"] == 0
        else "qualified_descriptive_support_only"
        if ready
        else "unqualified",
        no_requeue_without_new_mechanism=True,
        cited_upstream_artifacts=[
            dict(path=str(ROOT / name), sha256=pin)
            for name, pin in {
                **PINS,
                qualified.HISTORY: qualified.HISTORY_PIN,
                qualified.PROTOCOL: qualified.PROTOCOL_PIN,
            }.items()
        ],
        methodology_note="Fresh real wrapper episodes compare reset recording off/on with the qualified observer enabled. Seeds and recorder twins remain dependent within four public development games. Censored windows are not no-progress outcomes. No population efficacy, policy refinement, arm promotion or headline solve credit.",
    )
    value["acceptance_gates"].pop("observer_delivery_ready_score")
    value["acceptance_gates"]["arc_rows_ready_score"] = ready
    value["field_principles"] = {
        k: "Invocation-bound immutable live diagnostic; action identity owns delivery; games define independent N; censorship grants no benefit or solve credit."
        for k in value
    }
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    return value


def replay(value: Json) -> bool:
    """An external measurement seal rejects even self-consistently rehashed work."""
    with patch.multiple(qualified, NAME=NAME, build=build):
        return bool(qualified.replay(value))
