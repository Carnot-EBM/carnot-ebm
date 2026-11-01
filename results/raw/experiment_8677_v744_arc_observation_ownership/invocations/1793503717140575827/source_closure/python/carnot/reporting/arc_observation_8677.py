"""REQ-REPORT-8677: bind ownership qualification to immutable callback evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8677_v744_arc_observation_ownership"
TASK = "exp8677-arc-observation-ownership"
CLI = f"scripts/experiments/{NAME}.py"
MODEL_SPECS: list[Json] = []
UPSTREAM = "results/experiment_8667_v744_contract_methods.json"
UPSTREAM_PIN = "sha256:554617dfcb114b398636e333c25b825886478343c3afa4c8fe065a41a25f2ea0"
PROTOCOL = "openspec/change-proposals/v744-methods-protocol.json"
PROTOCOL_PIN = "sha256:842e28d27098b7dadf41290c407e72cfa4019a2cbb09429c492355aa0f55cb67"
HISTORY = "results/experiment_8664_v743_arc_reset_transfer.json"
HISTORY_PIN = "sha256:dcd1ba6f7b62d54d3f832c402420470a123cbe12defb394b3938bdd502fd93e9"
GAMES = ["ar25-0c556536", "ft09-0d8bbf25", "lf52-271a04aa", "tn36-ef4dde99"]
TESTS = [
    "tests/python/test_arc_observation_ownership_8677.py",
    "tests/python/test_arc_observation_runner_8677.py",
]
TEST = TESTS[0]
OWNED = [
    "python/carnot/agentic/arc_observation_delivery.py",
    "python/carnot/reporting/arc_observation_8677.py",
    "python/carnot/reporting/arc_observation_live_8677.py",
    "python/carnot/reporting/arc_observation_runner_8677.py",
    CLI,
]
REQUIRED_CHECKS = {
    "owned_tests",
    "coverage_report",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
    "exact_node_union",
    "original_observer_families",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real phase boundaries so the parent can supervise CPU work."""
    print(f"[exp8677] phase={phase} completed={completed} pending={pending}", flush=True)


def reference(path: Path) -> Json:
    """Readers must bind bytes as well as their location."""
    return dict(path=str(path), sha256=sha256_file(path))


def read(ref: Json) -> Json:
    """Do not accept a caller's replacement bytes under an old reference."""
    path = Path(ref["path"])
    if sha256_file(path) != ref["sha256"]:
        raise ValueError("changed_bytes:" + str(path))
    return dict(json.loads(path.read_bytes()))


def checked(ref: Json) -> bool:
    """Collection logs may be non-JSON while still requiring exact byte identity."""
    path = Path(ref["path"])
    return path.is_file() and sha256_file(path) == ref["sha256"]


def gates(root: Path) -> list[Json]:
    """The declared branch gate is distinct from the producer's global quarantine."""
    path = root / UPSTREAM
    value = json.loads(path.read_bytes()) if path.is_file() else {}
    observed = value.get("arc_inputs_ready_score")
    digest = sha256_file(path) if path.is_file() else None
    state = (
        "missing_file"
        if not path.is_file()
        else "missing_field"
        if observed is None
        else "observed_zero"
        if observed == 0
        else "changed_bytes"
        if digest != UPSTREAM_PIN
        else "observed_value"
    )
    return [
        dict(
            upstream="exp8667-contract-methods",
            check="declared_arc_gate",
            path=str(path),
            hash=digest,
            expected_hash=UPSTREAM_PIN,
            field="arc_inputs_ready_score",
            operator="==",
            expected=1,
            observed=observed,
            observed_state=state,
            passed=observed == 1 and digest == UPSTREAM_PIN,
            producer_quarantined=value.get("flagged_adversarial"),
        )
    ]


def authenticate() -> Json:
    """Sealed historical inputs authorize repair without promoting their failed result."""
    protocol = read(dict(path=str(ROOT / PROTOCOL), sha256=PROTOCOL_PIN))
    p = dict(protocol["observation_ownership_protocol"])
    history = read(dict(path=str(ROOT / HISTORY), sha256=HISTORY_PIN))
    work = read(history["work_reference"])
    for ref in p["actual_transfer_references"]:
        episode = read(ref)
        if len(episode["observer_issues"]) != 1022:
            raise ValueError("diagnosis_gap")
    for path, digest in work["source_artifact_hashes"].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError("historical_source_closure")
    p["original_agent_reference"] = next(
        reference(Path(path))
        for path in work["source_artifact_hashes"]
        if path.endswith("arc_competition_agent.py")
    )
    p["historical_model_provenance"] = history["historical_model_provenance"]
    p["historical_source_hashes"] = work["source_artifact_hashes"]
    return p


def empty_work(date: str, failures: list[Json]) -> Json:
    """Unavailable inputs retain every intended cell rather than inventing samples."""
    return dict(
        run_date=date,
        failures=failures,
        references=[],
        diagnosis_rows=[],
        controls=[],
        registry_precheck={},
        preconditions_checked={},
        source_artifact_hashes={},
        code_config_hashes={},
        raw_file_hashes={},
        measurement_receipts=[],
        duration_s=0.0,
        phase_spans=[],
        historical_model_provenance={},
    )


def reduce(episodes: list[Json]) -> Json:
    """Compare actions and public state separately from diagnostic observer records."""
    pairs: list[Json] = []
    ownership: list[Json] = []
    repeats: list[Json] = []
    for game in GAMES:
        for seed in (11, 23):
            arms = {
                ep["unit"]["observer"]: ep
                for ep in episodes
                if ep["unit"]["game"] == game and ep["unit"]["seed"] == seed
            }
            keys = (
                "actions",
                "observations",
                "termination",
                "legacy",
                "rng",
                "reset_requests",
                "final_state",
            )
            equal = {k: arms.get("off", {}).get(k) == arms.get("on", {}).get(k) for k in keys}
            clean = (
                len(arms) == 2
                and all(equal.values())
                and all(
                    not ep["observer_issues"]
                    and not ep["termination"]["error"]
                    and not ep["model_tripwire_attempts"]
                    and ep["llm_calls"] == 0
                    for ep in arms.values()
                )
            )
            on = arms.get("on", {})
            canonical = [
                r["action_index"] for r in on.get("delivery_rows", []) if r["status"] == "canonical"
            ]
            clean &= canonical == list(range(1, len(on.get("actions", [])) + 1)) and bool(canonical)
            pairs.append(
                dict(game=game, seed=seed, equality=equal, passed=clean, absolute_metric=int(clean))
            )
            for row in arms.get("on", {}).get("delivery_rows", []):
                item = dict(row, game=game, seed=seed)
                (ownership if row["status"] == "canonical" else repeats).append(item)
    return dict(
        action_parity_rows=pairs,
        callback_ownership_rows=ownership,
        repeated_delivery_rows=repeats,
        per_game_results=[
            dict(game=g, paired_cells=sum(r["game"] == g and r["passed"] for r in pairs))
            for g in GAMES
        ],
    )


def build(path: Path, receipts: list[Json], output: Path) -> Json:
    """Readiness requires clean ownership, original assertions and real child validation."""
    work = json.loads(path.read_bytes())
    episodes = [read(ref) for ref in work["references"]]
    result = reduce(episodes)
    checks = bool(receipts) and all(r["passed"] for r in receipts)
    checks &= not episodes or REQUIRED_CHECKS <= {r["name"] for r in receipts}
    diagnosis = len(work["diagnosis_rows"]) == 16 and all(
        r["passed"] for r in work["diagnosis_rows"]
    )
    controls = bool(work["controls"]) and all(r["passed"] for r in work["controls"])
    scientific = (
        len(episodes) == 16
        and all(r["passed"] for r in result["action_parity_rows"])
        and diagnosis
        and controls
    )
    bad = not checks or (not work["failures"] and not scientific)
    verdict = "disqualified" if bad else "blocked" if work["failures"] else "null"
    ready = int(scientific and checks and not work["failures"])
    indexed = {
        (ep["unit"]["game"], ep["unit"]["seed"], ep["unit"]["observer"]): ep for ep in episodes
    }
    measurement_names = {r["name"] for r in work["measurement_receipts"]}
    validation_receipts = [r for r in receipts if r["name"] not in measurement_names]
    rows = []
    for game in GAMES:
        for seed in (11, 23):
            for arm in ("off", "on"):
                ep = indexed.get((game, seed, arm))
                rows.append(
                    dict(
                        game=game,
                        seed=seed,
                        arm=arm,
                        status="completed" if ep else "blocked",
                        absolute_metric=len(ep["actions"]) if ep else None,
                        missing_reason=None if ep else "upstream_or_episode_unavailable",
                    )
                )
    value = dict(
        experiment_id=8677,
        task_id=TASK,
        milestone="2026.11.744",
        run_date=work["run_date"],
        honest_verdict=f"complete_{verdict}_arc_observation_ownership",
        verdict_class=verdict,
        inference_substrate="offline_arcade_live_agent_runtime_self_discovery_no_llm",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(
            language_model=0,
            llm=0,
            numerical_head=sum(ep["numerical_head_calls"] for ep in episodes),
        ),
        historical_model_provenance=work["historical_model_provenance"],
        target_model=work.get("numerical_head_reference"),
        rows=rows,
        intended_count=16,
        completed_count=len(episodes),
        failed_count=int(bad),
        censored_count=sum(
            r["horizon_status"] == "censored" for ep in episodes for r in ep["observer_rows"]
        ),
        excluded_count=0,
        independent_count=len({ep["unit"]["game"] for ep in episodes}),
        sample_size_budget=dict(
            games=4,
            dependent_game_seed_pairs=8,
            episodes=16,
            max_actions=512,
            max_seconds=120,
            horizon=120,
        ),
        verifier_is_oracle=False,
        exposure_scope="public_adapter_withheld_development_panel_and_constructed_controls",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        promotion_ready_score=0,
        observer_delivery_ready_score=ready,
        branch_gate_values=dict(observer_delivery_ready_score=ready),
        required_checks_passed=checks,
        flagged_adversarial=bad,
        acceptance_gates=dict(
            validation=checks,
            diagnosis=diagnosis,
            horizon_controls=controls,
            exact_policy_parity=scientific,
            observer_delivery_ready_score=ready,
        ),
        validation_receipts=receipts,
        gate_check_summary=work["failures"],
        adversarial_findings=[],
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / NAME / "terminal_validation.json"
        ),
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"] + sum(r.get("duration_s", 0) for r in validation_receipts),
        phase_spans=work["phase_spans"]
        + [dict(phase=r["name"], duration_s=r.get("duration_s", 0)) for r in validation_receipts],
        random_seed=8677,
        source_artifact_hashes=work["source_artifact_hashes"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes={r["path"]: r["sha256"] for r in work["references"]},
        cited_upstream_artifacts=[
            dict(path=str(ROOT / n), sha256=h)
            for n, h in [(UPSTREAM, UPSTREAM_PIN), (PROTOCOL, PROTOCOL_PIN), (HISTORY, HISTORY_PIN)]
        ],
        work_reference=reference(path),
        registry_precheck=work["registry_precheck"],
        diagnosis_rows=work["diagnosis_rows"],
        horizon_control_rows=work["controls"],
        conflict_rejection_rows=[r for r in work["controls"] if r["case"] == "conflict"],
        solve_provenance="live_agent_self_discovery",
        headline_solve_credit=0,
        preserved_history=dict(
            exp8664_verdict="complete_disqualified_arc_reset_transfer",
            exp8664_flagged=True,
            exp8667_global_quarantine=True,
            prior_quarantines_unchanged=True,
        ),
        methodology_note="Single delivery ownership is diagnostic readiness only. Constructed horizon controls establish no natural supervisor benefit. Seeds and recorder twins are dependent; no solve, policy promotion or generalization credit.",
        **result,
    )
    value["field_principles"] = {
        k: "Invocation-bound immutable evidence; missing differs from zero; routing readiness gives no benefit or solve credit."
        for k in value
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(value: Json) -> bool:
    """An independently sealed work record rejects self-consistent envelope substitution."""
    ref = value["work_reference"]
    path = Path(ref["path"])
    seal = json.loads(path.with_name("measurement_seal.json").read_bytes())
    if ref["sha256"] != seal["work_sha256"]:
        raise ValueError("measurement_seal")
    work = read(ref)
    validation_seal = path.with_name("validation_seal.json")
    if json.loads(validation_seal.read_bytes())["receipts_sha256"] != canonical_hash(
        value["validation_receipts"]
    ):
        raise ValueError("validation_seal")
    for field in ("source_artifact_hashes", "code_config_hashes", "raw_file_hashes"):
        for name, digest in work[field].items():
            if sha256_file(Path(name)) != digest:
                raise ValueError("operand_changed_bytes")
    for receipt in value["validation_receipts"]:
        for stream in ("stdout", "stderr"):
            if (
                stream + "_path" in receipt
                and sha256_file(Path(receipt[stream + "_path"])) != receipt[stream + "_sha256"]
            ):
                raise ValueError("validation_log_changed_bytes")
    output = Path(value["terminal_validation_sidecar_path"]).parents[2] / (NAME + ".json")
    return value == build(path, value["validation_receipts"], output)
