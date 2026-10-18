"""REQ-REPORT-8649, REQ-VERIFY-8649: qualify recording, preserve scientific nulls."""

from __future__ import annotations

import json
import hashlib
import mmap
from pathlib import Path
import re
from typing import Any

from carnot.reporting.arc_observer_8624 import checked, checked_bytes
from carnot.reporting.arc_observer_transfer_8637 import parity, replay_stream
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8649_v742_arc_reset_provenance"
TASK = "exp8649-arc-reset-provenance"
TASK_PIN = "sha256:ff9d8e487648615af1a68324d0ec2fcf67a5fdf0e794a903e494c75668b1dc2c"
CLI = f"scripts/experiments/{NAME}.py"
UPSTREAM = "results/experiment_8637_v741_arc_observer_transfer.json"
UPSTREAM_PIN = "sha256:80075ff253467526a8dc0eb2167716da8f28967dc8e9c01d1819050139a9a797"
MODEL_SPECS: list[Json] = []
TESTS = [
    "tests/python/test_arc_reset_provenance_8649.py",
    "tests/python/test_arc_reset_reporting_8649.py",
]
OWNED = (
    ["python/carnot/agentic/arc_reset_provenance.py"]
    + [
        f"python/carnot/reporting/{n}.py"
        for n in ["arc_reset_8649", "arc_reset_live_8649", "arc_reset_runner_8649"]
    ]
    + [CLI]
)
CLOSURE = ROOT / "results/raw" / NAME / "original_source_closure.json"
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


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Keep phase transitions visible while children retain their own measured clocks."""
    print(f"[exp8649] phase={phase} completed={completed} pending={pending}", flush=True)


def public_episode(ref: Json) -> Json:
    """Scan top-level fields once; hash the large legacy receipt without allocating it."""
    path = Path(ref["path"])
    if sha256_file(path) != ref["sha256"]:
        raise ValueError("original_hash:" + str(path))
    result: Json = {}
    wanted = {
        "unit",
        "steps",
        "events",
        "observer_rows",
        "actions",
        "rng",
        "termination",
        "observations",
        "end_reason",
    }
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        matches = list(re.finditer(rb'^  "([^"\n]+)":\s*', data, re.M))
        for index, match in enumerate(matches):
            key = match.group(1).decode()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(data)
            if key == "legacy":
                result[key] = hashlib.sha256(memoryview(data)[match.end() : end]).hexdigest()
            elif key in wanted:
                result[key] = json.JSONDecoder().raw_decode(data[match.end() : end].decode())[0]
    return result


def authenticate(root: Path) -> Json:
    """Independent original pins authenticate public evidence after live code changes."""
    value = checked(root / UPSTREAM, UPSTREAM_PIN)
    work = checked(Path(value["work_reference"]["path"]), value["work_reference"]["sha256"])
    episodes = [public_episode(r) for r in work["references"]]
    assert len(episodes) == 16 and value["censored_count"] == 16
    assert len(value["actual_redirect_rows"]) == 16 and value["verdict_class"] == "null"
    assert all(r["horizon_status"] == "censored" for r in value["actual_redirect_rows"])
    assert len(value["observer_action_parity_rows"]) == 8
    assert all(r["passed"] for r in value["observer_action_parity_rows"])
    for ep in episodes:
        if ep["unit"]["observer"] == "on":
            assert replay_stream(ep["steps"], ep["events"], ep["end_reason"]) == ep["observer_rows"]
    for key in {ep["unit"]["pair_key"] for ep in episodes}:
        pair = [ep for ep in episodes if ep["unit"]["pair_key"] == key]
        assert len(pair) == 2 and parity(*pair)["passed"]
    closure = checked(
        CLOSURE, "sha256:cb51a5341c669d08d119ed8e0d8f6ca8e5ce1df3573db8e8b9c43ffb4bd1f04b"
    )
    for ref in closure["files"]:
        checked_bytes(Path(ref["sealed_path"]), ref["sha256"])
    return value


def build(path: Path, receipts: list[Json], output: Path) -> Json:
    """Derive fixture readiness independently from zero scientific benefit."""
    work = json.loads(path.read_bytes())
    pairs = [checked(Path(r["path"]), r["sha256"]) for r in work["references"]]
    rows = [
        dict(
            mechanic=p["mechanic"],
            status="completed",
            absolute_metric=int(p["passed"]),
            passed=p["passed"],
            missing_reason=None,
            public_fixture_only=True,
        )
        for p in pairs
    ]
    from carnot.reporting.arc_reset_live_8649 import parity as current_parity

    invariance = [dict(mechanic=p["mechanic"], **current_parity(p["off"], p["on"])) for p in pairs]
    resets, owners, terminals, missing = [], [], [], []
    for pair in pairs:
        for target, field in [
            (resets, "reset_request_rows"),
            (owners, "requester_ownership_rows"),
            (terminals, "terminal_reason_rows"),
            (missing, "missing_cause_rows"),
        ]:
            target.extend(
                dict(r, mechanic=pair["mechanic"]) for r in pair["on"]["reset_provenance"][field]
            )
    behavior = len(pairs) == 3 and all(r["passed"] for r in invariance)
    ownership = bool(resets) and all(r["status"] == "unique" for r in owners) and not missing
    clean = all(
        not p["on"]["recorder_errors"]
        and not p["on"]["reset_provenance"]["issues"]
        and not p["on"]["termination"]["error"]
        and p["on"]["llm_calls"] == 0
        for p in pairs
    )
    passed = (
        bool(receipts)
        and all(r["passed"] for r in receipts)
        and (not pairs or REQUIRED_CHECKS <= {r["name"] for r in receipts})
    )
    bad = not passed or bool(pairs and not (behavior and ownership and clean))
    verdict = "disqualified" if bad else "blocked" if work["failures"] else "null"
    ready = int(passed and behavior and ownership and clean and not work["failures"])
    zeros = dict(
        policy_refinement_score=0,
        promotion_ready_score=0,
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
    )
    history = work["history"]
    measured = {r["name"] for r in work.get("measurement_receipts", [])}
    validation_receipts = [r for r in receipts if r["name"] not in measured]
    value = dict(
        experiment_id=8649,
        task_id=TASK,
        milestone="2026.10.742",
        run_date=work["run_date"],
        honest_verdict=f"complete_{verdict}_arc_reset_provenance",
        verdict_class=verdict,
        inference_substrate="CPU_scored_wrapper_scripted_public_mechanics_small_numerical_head",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(
            llm=0,
            language_model=0,
            numerical_head=sum(
                p[mode]["numerical_head_calls"] for p in pairs for mode in ("off", "on")
            ),
        ),
        historical_model_provenance=work["historical_model_provenance"],
        target_model=work.get("numerical_head_reference"),
        rows=rows,
        intended_count=3,
        completed_count=len(rows),
        failed_count=int(bad),
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            fixture_pairs=3,
            independent_games=0,
            max_actions=128,
            max_seconds=120,
            observer_horizon=120,
        ),
        verifier_is_oracle=False,
        exposure_scope="scripted_public_mechanics_development_proxy",
        **zeros,
        reset_provenance_ready_score=ready,
        branch_gate_values=dict(reset_provenance_ready_score=ready, **zeros),
        reset_request_rows=resets,
        requester_ownership_rows=owners,
        action_parity_rows=invariance,
        terminal_reason_rows=terminals,
        missing_cause_rows=missing,
        live_path_reference=work.get("live_path_reference", {}),
        registry_precheck=work["registry_precheck"],
        solve_provenance="live_agent_self_discovery",
        headline_solve_credit=0,
        required_checks_passed=passed,
        flagged_adversarial=bad,
        acceptance_gates={r["name"]: r["passed"] for r in receipts},
        gate_check_summary=work["failures"],
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / NAME / "terminal_validation.json"
        ),
        adversarial_findings=[],
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"] + sum(r["duration_s"] for r in validation_receipts),
        phase_spans=work["phase_spans"]
        + [
            dict(
                phase=r["name"],
                duration_s=r["duration_s"],
                started_monotonic_ns=r.get("started_monotonic_ns"),
                ended_monotonic_ns=r.get("ended_monotonic_ns"),
            )
            for r in validation_receipts
        ],
        random_seed=8649,
        source_artifact_hashes=work["source_artifact_hashes"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes={r["path"]: r["sha256"] for r in work["references"]},
        cited_upstream_artifacts=[dict(path=UPSTREAM, sha256=UPSTREAM_PIN)],
        work_reference=dict(path=str(path), sha256=sha256_file(path)),
        preserved_history=dict(
            censored_redirects=16 if history else None,
            complete_horizons=0 if history else None,
            verdict="complete_null_arc_observer_transfer" if history else None,
            missing_reset_cause="unknown",
            historical_quarantines_unchanged=True,
        ),
        methodology_note="Passive recording qualification only. Fixtures are not real outcomes; Exp8650 owns diagnostic transfer. No reset timing, supervisor priority or horizon changes.",
    )
    value["field_principles"] = {
        key: "Invocation-bound recording evidence; missing remains unknown; fixtures supply no scientific benefit."
        for key in value
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(value: Json) -> bool:
    """The independent measurement seal rejects rehashed replacements before reduction."""
    ref = value["work_reference"]
    path = Path(ref["path"])
    seal = json.loads(path.with_name("measurement_seal.json").read_bytes())
    if seal["work_sha256"] != ref["sha256"]:
        raise ValueError("measurement_seal")
    work = checked(path, ref["sha256"])
    for field in ("source_artifact_hashes", "code_config_hashes"):
        for name, digest in work[field].items():
            checked_bytes(Path(name), digest)
    for receipt in value["validation_receipts"]:
        for stream in ("stdout", "stderr"):
            if stream + "_path" in receipt:
                checked_bytes(Path(receipt[stream + "_path"]), receipt[stream + "_sha256"])
    if work["history"] is not None:
        assert checked(ROOT / UPSTREAM, UPSTREAM_PIN) == work["history"]
    output = Path(value["terminal_validation_sidecar_path"]).parents[2] / (NAME + ".json")
    return value == build(path, value["validation_receipts"], output)
