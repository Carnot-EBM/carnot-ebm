"""REQ-REPORT-8663: reduce a bounded engineering witness without solve credit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting.arc_observer_8624 import checked, checked_bytes
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8663_v743_arc_reset_provenance"
TASK = "exp8663-arc-reset-provenance"
CLI = f"scripts/experiments/{NAME}.py"
MODEL_SPECS: list[Json] = []
TESTS = [
    "tests/python/test_arc_reset_provenance_8663.py",
    "tests/python/test_arc_reset_reporting_8663.py",
]
OWNED = (
    ["python/carnot/agentic/arc_reset_provenance.py"]
    + [
        f"python/carnot/reporting/{name}.py"
        for name in ("arc_reset_8663", "arc_reset_witness_8663", "arc_reset_runner_8663")
    ]
    + [CLI]
)
PINS = {
    "results/experiment_8650_v742_arc_reset_transfer.json": "sha256:6351fa5892a02cfd8a724b41854f2755c211edadd6d1abfc84b36bed7d607db2",
    "results/experiment_8649_v742_arc_reset_provenance.json": "sha256:5ed5072ff6fc398e070f353c7a8190d8113ae4ac43fa4a174ef9f6250f0795e8",
}
REQUIRED_CHECKS = {
    "owned_tests",
    "coverage_report",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
    "historical_union",
    "original_semantics",
    "compatible_semantics",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Keep actual phase boundaries visible while bounded children run."""
    print(f"[exp8663] phase={phase} completed={completed} pending={pending}", flush=True)


def reduce(episodes: list[Json]) -> Json:
    """Require real reset joins and exact parity before recording readiness."""
    from carnot.reporting.arc_reset_live_8649 import parity

    pairs = [parity(*episodes)] if len(episodes) == 2 else []
    requests, callbacks, witnesses = [], [], []
    for ep in episodes:
        ledger = ep["reset_provenance"]
        on = ep["unit"]["reset_provenance"] == "on"
        observed = {r["action_index"]: r["observation_sha256"] for r in ep["reset_observations"]}
        joins = [
            r["action_index"] in observed
            and r["reset_request_id"].split(":")[0] == str(r["action_index"])
            and r["post_observation_sha256"] == observed[r["action_index"]]
            and r["sdk_status"] is not None
            and not r["sdk_status"]["error"]
            for r in ledger["reset_request_rows"]
        ]
        clean = not (
            ep["recorder_errors"]
            or ledger["issues"]
            or ledger["missing_cause_rows"]
            or ep["termination"]["error"]
            or ep["model_tripwire_attempts"]
        )
        supported = (
            (len(joins) == len(observed) and bool(joins) and all(joins)) if on else not joins
        )
        witnesses.append(
            dict(
                arm=ep["unit"]["reset_provenance"],
                passed=bool(
                    clean and supported and ep["sdk_identity"]["official"] and ep["llm_calls"] == 0
                ),
                reset_count=len(observed),
                joined_count=sum(joins),
                missing_reason=None if clean and supported else "recorder_or_join_failure",
            )
        )
        requests.extend(ledger["reset_request_rows"])
        callbacks.extend(ledger["callback_sequence_rows"])
    return dict(
        action_parity_rows=pairs,
        request_context_rows=requests,
        callback_sequence_rows=callbacks,
        sdk_witness_rows=witnesses,
        witnessed=bool(pairs and pairs[0]["passed"] and all(r["passed"] for r in witnesses)),
    )


def build(path: Path, receipts: list[Json], output: Path) -> Json:
    """Derive terminal fields from sealed work and actual command exits."""
    work = json.loads(path.read_bytes())
    episodes = [checked(Path(r["path"]), r["sha256"]) for r in work["references"]]
    result = reduce(episodes)
    checks = bool(receipts) and all(r["passed"] for r in receipts)
    checks &= not episodes or REQUIRED_CHECKS <= {r["name"] for r in receipts}
    reproduced = bool(work["reproduction"] and work["reproduction"]["passed"])
    bad = (
        not checks
        or any(f.get("check") == "unreproduced_observed_failure" for f in work["failures"])
        or bool(not work["failures"] and not (result["witnessed"] and reproduced))
    )
    verdict = "disqualified" if bad else "blocked" if work["failures"] else "null"
    ready = int(checks and reproduced and result["witnessed"] and not work["failures"])
    zeros = dict(
        promotion_ready_score=0,
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
    )
    rows = [
        dict(
            unit_id=f"sdk-{arm}",
            arm=arm,
            status="completed" if i < len(episodes) else "blocked",
            absolute_metric=len(episodes[i]["actions"]) if i < len(episodes) else None,
            missing_reason=None if i < len(episodes) else "external_operand_unavailable",
        )
        for i, arm in enumerate(("off", "on"))
    ]
    reproduction = work["reproduction"] or {}
    value = dict(
        experiment_id=8663,
        task_id=TASK,
        milestone="2026.10.743",
        run_date=work["run_date"],
        honest_verdict=f"complete_{verdict}_arc_reset_provenance",
        verdict_class=verdict,
        inference_substrate="CPU_public_official_SDK_small_numerical_frame_change_head",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(
            llm=0,
            language_model=0,
            numerical_head=sum(ep["numerical_head_calls"] for ep in episodes),
        ),
        historical_model_provenance=work["historical_model_provenance"],
        target_model=work["target_model"],
        rows=rows,
        intended_count=2,
        completed_count=len(episodes),
        failed_count=int(bad),
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            episodes=2, pairs=1, independent_scientific_questions=0, max_actions=16, max_seconds=60
        ),
        verifier_is_oracle=False,
        exposure_scope="exposed_public_SDK_engineering_witness",
        **zeros,
        **result,
        reset_provenance_ready_score=ready,
        solve_provenance="live_agent_self_discovery",
        headline_solve_credit=0,
        branch_gate_values=dict(reset_provenance_ready_score=ready, **zeros),
        required_checks_passed=checks,
        flagged_adversarial=bad,
        acceptance_gates=dict(
            **{r["name"]: r["passed"] for r in receipts},
            sdk_witness=result["witnessed"],
            exact_failure_reproduced=reproduced,
            reset_provenance_ready_score=bool(ready),
        ),
        gate_check_summary=work["failures"],
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / NAME / "terminal_validation.json"
        ),
        adversarial_findings=[],
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"] + sum(r["duration_s"] for r in receipts),
        phase_spans=work["phase_spans"]
        + [dict(phase=r["name"], duration_s=r["duration_s"]) for r in receipts],
        random_seed=11,
        source_artifact_hashes=work["source_artifact_hashes"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes={r["path"]: r["sha256"] for r in work["references"]},
        cited_upstream_artifacts=[dict(path=p, sha256=h) for p, h in PINS.items()],
        work_reference=dict(path=str(path), sha256=sha256_file(path)),
        real_frame_fixture_reference=reproduction.get("real_frame_fixture_reference"),
        exception_reproduction_rows=reproduction.get("exception_reproduction_rows", []),
        preserved_history=work["historical"],
        registry_precheck=work["registry_precheck"],
        methodology_note="Bounded engineering witness only. Historical disqualification and quarantines remain unchanged. No solve, policy benefit or transfer-panel qualification. Exp8664 owns the full panel.",
    )
    value["field_principles"] = {
        key: "Sealed invocation evidence; missing values differ from zero; engineering readiness gives no scientific benefit."
        for key in value
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(value: Json) -> bool:
    """An independent retained seal rejects rehashed substituted operands."""
    ref = value["work_reference"]
    path = Path(ref["path"])
    if (
        json.loads(path.with_name("measurement_seal.json").read_bytes())["work_sha256"]
        != ref["sha256"]
    ):
        raise ValueError("measurement_seal")
    work = checked(path, ref["sha256"])
    if work["historical"] is not None:
        from carnot.reporting.arc_reset_witness_8663 import authenticate

        if authenticate(ROOT) != work["historical"]:
            raise ValueError("historical_drift")
    for field in ("source_artifact_hashes", "code_config_hashes", "raw_file_hashes"):
        for name, digest in work[field].items():
            checked_bytes(Path(name), digest)
    for receipt in value["validation_receipts"]:
        for name, digest in receipt.get("artifact_hashes", {}).items():
            checked_bytes(Path(name), digest)
        for stream in ("stdout", "stderr"):
            if stream + "_path" in receipt:
                checked_bytes(Path(receipt[stream + "_path"]), receipt[stream + "_sha256"])
    output = Path(value["terminal_validation_sidecar_path"]).parents[2] / (NAME + ".json")
    return value == build(path, value["validation_receipts"], output)
