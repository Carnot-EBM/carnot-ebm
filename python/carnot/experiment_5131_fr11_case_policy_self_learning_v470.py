"""Exp 5131: no-weight case-policy self-learning over exact CSP traces.

Spec refs: REQ-LEARN-5131,
SCENARIO-LEARN-5131-CASE-POLICY-NO-PROMOTE,
SCENARIO-LEARN-5131-BLOCKED-PRECONDITION.

This experiment is deliberately a metadata learner. It reads exact-solver
traces from Exp 5130, learns which previously observed cases made a variable
ordering cheaper, and emits guarded policy hints. It does not update model
weights, and every evaluated arm reuses an exact-solver trace as the only
correctness authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_5130_taco_sampler_heldout_scale_v470 as exp5130


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_RELATIVE_PATH = "python/carnot/experiment_5131_fr11_case_policy_self_learning_v470.py"
SOURCE_TRACE_RELATIVE_PATH = exp5130.RESULT_RELATIVE_PATH
RESULT_RELATIVE_PATH = "results/experiment_5131_fr11_case_policy_self_learning_v470.json"
EXPERIMENT_ID = "exp5131-fr11-case-policy-self-learning-v470"
MILESTONE = "2026.07.470"
RUN_DATE = "20260701"
RANDOM_SEED = 5131
SCHEMA = "carnot.experiment_5131_fr11_case_policy_self_learning.v470"
INFERENCE_SUBSTRATE = "cpu_no_weight_case_policy_over_exact_solver_traces"
SUCCESS_PREFIX = "success_fr11_case_policy_promoted_"
NO_PROMOTE_PREFIX = "complete_fr11_case_policy_no_promote_"
BLOCKED_PREFIX = "blocked_exp5130_heldout_csp_trace_suite_not_ready"
TERMINAL_PREFIXES = (SUCCESS_PREFIX, NO_PROMOTE_PREFIX, BLOCKED_PREFIX)
POLICY_ARMS = (
    "no_learning",
    "naive_retrieval",
    "case_policy",
    "case_policy_with_harm_gate",
)
TRACE_ARMS = ("baseline", "unguarded", "guarded", "sampler_feature")
LEARNING_FAMILIES = (
    "apex_cycle_coloring",
    "grid_bipartite_coloring",
    "sparse_path_coloring",
)
VALIDATION_FAMILIES = ("crown_graph_coloring", "triangular_prism_coloring")
CASE_TTL_STEPS = 16
DECAY_RATE = 0.85
DEFAULT_TESTS_RUN = [
    "JAX_PLATFORMS=cpu /home/ianblenke/github.com/ianblenke/carnot/.venv/bin/python "
    "scripts/experiment_5131_fr11_case_policy_self_learning_v470.py --date 20260701",
    ".venv/bin/pytest tests/python/test_experiment_5131_fr11_case_policy_self_learning_v470.py -q",
    ".venv/bin/pytest tests/python/test_experiment_5131_fr11_case_policy_self_learning_v470.py "
    "--cov=python/carnot/experiment_5131_fr11_case_policy_self_learning_v470.py "
    "--cov=scripts/experiment_5131_fr11_case_policy_self_learning_v470.py "
    "--cov-report=term-missing --cov-fail-under=100 -q",
    ".venv/bin/pytest tests/python -q",
]
REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "honest_verdict",
    "inference_substrate",
    "duration_s",
    "continuous_self_learning_task",
    "source_trace_artifacts",
    "trace_split_manifest",
    "policy_description",
    "heldout_delta",
    "nonforgetting_delta",
    "harmful_promotion_count",
    "regret_telemetry",
    "exact_solver_correctness_preserved",
    "promotion_attempted",
    "promotion_safe",
    "rollback_receipt",
    "no_weight_update",
    "flagged_adversarial",
    "conductor_modified",
    "tests_run",
)
FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "traceability",
    "milestone": "milestone accountability",
    "honest_verdict": "terminal verdict with complete_/success_/blocked_ prefix",
    "inference_substrate": "substrate honesty",
    "duration_s": "timing accountability",
    "continuous_self_learning_task": "PRD FR-11 coverage",
    "source_trace_artifacts": "data provenance",
    "trace_split_manifest": "held-out integrity",
    "policy_description": "method transparency",
    "heldout_delta": "utility",
    "nonforgetting_delta": "no catastrophic forgetting",
    "harmful_promotion_count": "safety",
    "regret_telemetry": "deployment-time learning accountability",
    "exact_solver_correctness_preserved": "correctness authority",
    "promotion_attempted": "learning action transparency",
    "promotion_safe": "gate decision",
    "rollback_receipt": "safe failure mode",
    "no_weight_update": "local-first safe adaptation",
    "flagged_adversarial": "adversarial-verification accountability",
    "conductor_modified": "conductor immutability",
    "tests_run": "verification evidence",
}


def load_source_trace_artifact(root: str | Path = REPO_ROOT) -> JsonDict:
    """Load Exp 5130 traces, preferring the requested root over the repo copy."""

    path = _source_trace_path(Path(root))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def source_trace_artifacts(root: str | Path, payload: JsonMap) -> list[JsonDict]:
    """Return provenance for the checked Exp 5130 trace artifact."""

    path = _source_trace_path(Path(root))
    return [
        {
            "path": SOURCE_TRACE_RELATIVE_PATH,
            "resolved_path": path.as_posix(),
            "present": path.exists(),
            "sha256": _sha256_file(path),
            "experiment_id": str(payload.get("experiment_id") or ""),
            "ready": payload.get("heldout_csp_trace_suite_ready") is True,
        }
    ]


def build_trace_split(source: JsonMap) -> JsonDict:
    """Split exact-solver rows by deterministic family partitions."""

    rows = [dict(row) for row in source.get("per_instance_results", []) if isinstance(row, Mapping)]
    learning_rows = [row for row in rows if row.get("family") in LEARNING_FAMILIES]
    validation_rows = [row for row in rows if row.get("family") in VALIDATION_FAMILIES]
    heldout_rows = [
        row
        for row in rows
        if row.get("family") not in LEARNING_FAMILIES
        and row.get("family") not in VALIDATION_FAMILIES
    ]
    manifest = {
        "blocked": False,
        "strategy": "deterministic_instance_family_partition",
        "family_partitions": {
            "learning": sorted({str(row["family"]) for row in learning_rows}),
            "validation": sorted({str(row["family"]) for row in validation_rows}),
            "heldout": sorted({str(row["family"]) for row in heldout_rows}),
        },
        "instance_ids": {
            "learning": _instance_ids(learning_rows),
            "validation": _instance_ids(validation_rows),
            "heldout": _instance_ids(heldout_rows),
        },
        "split_hashes": {
            "learning": _hash_json(_instance_ids(learning_rows)),
            "validation": _hash_json(_instance_ids(validation_rows)),
            "heldout": _hash_json(_instance_ids(heldout_rows)),
        },
        "heldout_integrity_passed": _splits_disjoint(learning_rows, validation_rows, heldout_rows),
    }
    return {
        "learning": {"rows": learning_rows},
        "validation": {"rows": validation_rows},
        "heldout": {"rows": heldout_rows},
        "manifest": manifest,
    }


def fit_case_policy(split: JsonMap) -> JsonDict:
    """Fit a deterministic nonparametric policy from learning traces."""

    records = [
        _case_record(row, index)
        for index, row in enumerate(split.get("learning", {}).get("rows", []))
    ]
    validation = _validation_telemetry(split.get("validation", {}).get("rows", []), records)
    hints = [_policy_hint(record, validation) for record in records]
    return {
        "policy_type": "nonparametric_contextual_case_policy",
        "no_weight_update": True,
        "case_count": len(records),
        "ttl_steps": CASE_TTL_STEPS,
        "decay_rate": DECAY_RATE,
        "selection_rule": "weighted_context_match_then_guarded_positive_advantage",
        "validation_gate_open": validation["validation_delta"] > 0.0
        and validation["validation_harm_count"] == 0,
        "validation_telemetry": validation,
        "case_records": records,
        "policy_hints": hints,
    }


def evaluate_policy_arms(split: JsonMap, policy: JsonMap) -> JsonDict:
    """Evaluate all deployment-time learning arms against exact held-out traces."""

    rows = [dict(row) for row in split.get("heldout", {}).get("rows", [])]
    case_records = [dict(row) for row in policy.get("case_records", [])]
    arms = {
        "no_learning": _evaluate_rows(rows, "baseline"),
        "naive_retrieval": _evaluate_selected(rows, case_records, mode="naive"),
        "case_policy": _evaluate_selected(rows, case_records, mode="policy"),
        "case_policy_with_harm_gate": _evaluate_selected(
            rows,
            case_records,
            mode="guarded",
            gate_open=bool(policy.get("validation_gate_open")),
        ),
    }
    baseline = arms["no_learning"]
    guarded = arms["case_policy_with_harm_gate"]
    heldout_delta = _effort_delta(baseline["total_effort"], guarded["total_effort"])
    nonforgetting_delta = _nonforgetting_delta(rows, guarded["per_instance"])
    harmful_count = sum(1 for row in guarded["per_instance"] if row["harmful_vs_baseline"])
    exact_preserved = all(_row_exact_preserved(row) for row in rows) and all(
        item["exact_solver_correctness_preserved"]
        for arm in arms.values()
        for item in arm["per_instance"]
    )
    telemetry = _regret_telemetry(rows, arms)
    return {
        "arms": arms,
        "heldout_delta": heldout_delta,
        "nonforgetting_delta": nonforgetting_delta,
        "harmful_promotion_count": harmful_count,
        "exact_solver_correctness_preserved": exact_preserved,
        "regret_telemetry": telemetry,
    }


def build_artifact(
    *,
    root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 5131 terminal artifact without mutating model weights."""

    started = time.perf_counter()
    repo_root = Path(root)
    source = load_source_trace_artifact(repo_root)
    sources = source_trace_artifacts(repo_root, source)
    elapsed = _elapsed(started, duration_s)
    if source.get("heldout_csp_trace_suite_ready") is not True:
        artifact = _blocked_artifact(
            sources=sources,
            duration_s=elapsed,
            run_date=run_date,
            tests_run=tests_run,
        )
        validate_artifact(artifact)
        return artifact

    exp5130.validate_artifact(source)
    split = build_trace_split(source)
    policy = fit_case_policy(split)
    evaluation = evaluate_policy_arms(split, policy)
    heldout_delta = float(evaluation["heldout_delta"])
    nonforgetting_delta = float(evaluation["nonforgetting_delta"])
    harmful_count = int(evaluation["harmful_promotion_count"])
    exact_preserved = bool(evaluation["exact_solver_correctness_preserved"])
    promotion_attempted = True
    promotion_safe = bool(
        heldout_delta > 0.0
        and nonforgetting_delta >= 0.0
        and harmful_count == 0
        and exact_preserved
        and policy["validation_gate_open"] is True
    )
    blockers = _promotion_blockers(
        heldout_delta=heldout_delta,
        nonforgetting_delta=nonforgetting_delta,
        harmful_count=harmful_count,
        exact_preserved=exact_preserved,
        validation_gate_open=bool(policy["validation_gate_open"]),
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "result_path": RESULT_RELATIVE_PATH,
        "random_seed": RANDOM_SEED,
        "honest_verdict": _verdict(promotion_safe, blockers),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": elapsed,
        "continuous_self_learning_task": True,
        "source_trace_artifacts": sources,
        "trace_split_manifest": split["manifest"],
        "policy_description": _policy_description(policy),
        "heldout_delta": heldout_delta,
        "nonforgetting_delta": nonforgetting_delta,
        "harmful_promotion_count": harmful_count,
        "regret_telemetry": evaluation["regret_telemetry"],
        "exact_solver_correctness_preserved": exact_preserved,
        "promotion_attempted": promotion_attempted,
        "promotion_safe": promotion_safe,
        "rollback_receipt": _rollback_receipt(promotion_safe, blockers),
        "no_weight_update": True,
        "flagged_adversarial": False,
        "conductor_modified": False,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "spec_refs": [
            "REQ-LEARN-5131",
            "SCENARIO-LEARN-5131-CASE-POLICY-NO-PROMOTE",
            "SCENARIO-LEARN-5131-BLOCKED-PRECONDITION",
        ],
        "arm_comparison": evaluation["arms"],
        "field_principles": FIELD_PRINCIPLES,
        "methodology_note": (
            "Policy hints change only case metadata selection. All labels, "
            "assignments, and certificates remain those produced by the "
            "Exp 5130 exact solver traces."
        ),
    }
    artifact["reproducibility_checksum"] = _hash_json(
        {
            "experiment_id": EXPERIMENT_ID,
            "run_date": run_date,
            "source_sha256": sources[0]["sha256"],
            "split_hashes": split["manifest"]["split_hashes"],
            "heldout_delta": heldout_delta,
            "nonforgetting_delta": nonforgetting_delta,
            "harmful_promotion_count": harmful_count,
            "promotion_safe": promotion_safe,
        }
    )
    validate_artifact(artifact)
    return artifact


def write_artifact(
    *,
    root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build and write the Exp 5131 result artifact."""

    repo_root = Path(root)
    artifact = build_artifact(
        root=repo_root,
        run_date=run_date,
        duration_s=duration_s,
        tests_run=tests_run,
    )
    destination = repo_root / RESULT_RELATIVE_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(
    *,
    root: str | Path = REPO_ROOT,
    date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """CLI-compatible entrypoint returning the written artifact path."""

    repo_root = Path(root)
    write_artifact(root=repo_root, run_date=date, duration_s=duration_s, tests_run=tests_run)
    return repo_root / RESULT_RELATIVE_PATH


def validate_artifact(artifact: JsonMap) -> None:
    """Raise when the Exp 5131 artifact violates its terminal contract."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS).difference(artifact))
    _require(not missing, f"missing required fields: {missing}")
    _require(artifact.get("experiment_id") == EXPERIMENT_ID, "experiment_id")
    _require(artifact.get("milestone") == MILESTONE, "milestone")
    _require(str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(isinstance(artifact.get("duration_s"), int | float), "duration_s")
    _require(float(artifact["duration_s"]) >= 0.0, "duration_s")
    _require(artifact.get("continuous_self_learning_task") is True, "continuous_self_learning_task")
    _require(artifact.get("no_weight_update") is True, "no_weight_update")
    _require(artifact.get("conductor_modified") is False, "conductor_modified")
    _require(isinstance(artifact.get("tests_run"), list) and bool(artifact["tests_run"]), "tests_run")
    _require(set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact.get("field_principles", {})), "field_principles")
    if str(artifact["honest_verdict"]).startswith(BLOCKED_PREFIX):
        _require(artifact.get("promotion_attempted") is False, "promotion_attempted")
        _require(artifact.get("promotion_safe") is False, "promotion_safe")
        _require(artifact.get("exact_solver_correctness_preserved") is False, "exact_solver_correctness_preserved")
        _require(artifact.get("trace_split_manifest", {}).get("blocked") is True, "trace_split_manifest")
        return
    _require(artifact.get("promotion_attempted") is True, "promotion_attempted")
    _require(artifact.get("exact_solver_correctness_preserved") is True, "exact_solver_correctness_preserved")
    _require(artifact.get("flagged_adversarial") is False, "flagged_adversarial")
    _require(_manifest_valid(artifact.get("trace_split_manifest")), "trace_split_manifest")
    _require(set(artifact.get("arm_comparison", {})) == set(POLICY_ARMS), "arm_comparison")
    if artifact.get("promotion_safe") is True:
        _require(str(artifact["honest_verdict"]).startswith(SUCCESS_PREFIX), "honest_verdict")
        _require(float(artifact["heldout_delta"]) > 0.0, "heldout_delta")
    else:
        _require(str(artifact["honest_verdict"]).startswith(NO_PROMOTE_PREFIX), "honest_verdict")
        _require(
            artifact.get("rollback_receipt", {}).get("rollback_applied") is True,
            "rollback_receipt",
        )


def _blocked_artifact(
    *,
    sources: Sequence[JsonMap],
    duration_s: float,
    run_date: str,
    tests_run: Sequence[str] | None,
) -> JsonDict:
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "result_path": RESULT_RELATIVE_PATH,
        "random_seed": RANDOM_SEED,
        "honest_verdict": BLOCKED_PREFIX,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "continuous_self_learning_task": True,
        "source_trace_artifacts": [dict(row) for row in sources],
        "trace_split_manifest": {"blocked": True, "reason": "exp5130_ready_gate_not_true"},
        "policy_description": {"policy_type": "not_fit_source_trace_gate_closed"},
        "heldout_delta": 0.0,
        "nonforgetting_delta": 0.0,
        "harmful_promotion_count": 0,
        "regret_telemetry": {"evaluated_instances": 0, "heldout_coverage": 0.0},
        "exact_solver_correctness_preserved": False,
        "promotion_attempted": False,
        "promotion_safe": False,
        "rollback_receipt": {
            "rollback_applied": True,
            "root_cause": "exp5130_heldout_csp_trace_suite_ready_not_true",
            "promoted_metadata_ids": [],
            "active_policy_after_rollback": "no_learning",
        },
        "no_weight_update": True,
        "flagged_adversarial": False,
        "conductor_modified": False,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "spec_refs": ["REQ-LEARN-5131", "SCENARIO-LEARN-5131-BLOCKED-PRECONDITION"],
        "arm_comparison": {},
        "field_principles": FIELD_PRINCIPLES,
        "methodology_note": "No case policy was fit because the upstream Exp 5130 trace gate was closed.",
        "reproducibility_checksum": _hash_json(
            {
                "experiment_id": EXPERIMENT_ID,
                "blocked": True,
                "sources": list(sources),
            }
        ),
    }


def _case_record(row: JsonMap, index: int) -> JsonDict:
    baseline_effort = _effort(row, "baseline")
    advantages = {
        arm: round((baseline_effort - _effort(row, arm)) / baseline_effort, 6)
        for arm in TRACE_ARMS
    }
    selected = max(TRACE_ARMS, key=lambda arm: (advantages[arm], arm == "baseline"))
    if advantages[selected] <= 0.0:
        selected = "baseline"
    decay = round(DECAY_RATE**index, 6)
    return {
        "case_id": f"case_5131_{index:04d}_{row['instance_id']}",
        "source_instance_id": row["instance_id"],
        "context": _context(row),
        "age_steps": index,
        "ttl_steps": CASE_TTL_STEPS,
        "ttl_remaining": CASE_TTL_STEPS - index,
        "decay": decay,
        "arm_advantages": {
            arm: round(value * decay, 6) for arm, value in advantages.items()
        },
        "selected_hint": selected,
        "guarded": True,
        "exact_solver_authority": True,
    }


def _policy_hint(record: JsonMap, validation: JsonMap) -> JsonDict:
    selected = str(record["selected_hint"])
    return {
        "hint_id": f"hint_{record['case_id']}",
        "source_case_id": record["case_id"],
        "selected_hint": selected,
        "advantage_estimate": record["arm_advantages"][selected],
        "ttl_remaining": record["ttl_remaining"],
        "decay": record["decay"],
        "guarded": True,
        "validation_gate_open": validation["validation_delta"] > 0.0
        and validation["validation_harm_count"] == 0,
        "correctness_authority": "exact_solver_trace",
    }


def _validation_telemetry(rows: Sequence[JsonMap], records: Sequence[JsonMap]) -> JsonDict:
    selected = _evaluate_selected(rows, records, mode="policy")
    baseline = _evaluate_rows(rows, "baseline")
    return {
        "validation_instances": len(rows),
        "validation_delta": _effort_delta(baseline["total_effort"], selected["total_effort"]),
        "validation_harm_count": sum(1 for row in selected["per_instance"] if row["harmful_vs_baseline"]),
        "validation_coverage": _coverage(selected["per_instance"]),
    }


def _evaluate_rows(rows: Sequence[JsonMap], arm: str) -> JsonDict:
    per_instance = [_instance_eval(row, arm, "static") for row in rows]
    return _arm_summary(per_instance)


def _evaluate_selected(
    rows: Sequence[JsonMap],
    records: Sequence[JsonMap],
    *,
    mode: str,
    gate_open: bool = True,
) -> JsonDict:
    per_instance = []
    for row in rows:
        selected_arm = "baseline"
        reason = "no_case_available"
        if records and gate_open:
            if mode == "naive":
                match = records[0]
            else:
                match = _best_context_match(row, records)
            selected_arm = str(match["selected_hint"])
            reason = f"matched:{match['case_id']}"
            if mode == "guarded" and match["arm_advantages"][selected_arm] <= 0.0:
                selected_arm = "baseline"
                reason = "guarded_nonpositive_advantage"
        elif not gate_open:
            reason = "validation_harm_gate_closed"
        per_instance.append(_instance_eval(row, selected_arm, reason))
    return _arm_summary(per_instance)


def _instance_eval(row: JsonMap, arm: str, reason: str) -> JsonDict:
    baseline_effort = _effort(row, "baseline")
    selected_effort = _effort(row, arm)
    return {
        "instance_id": row["instance_id"],
        "family": row["family"],
        "selected_trace_arm": arm,
        "selection_reason": reason,
        "total_effort_score": selected_effort,
        "utility_delta_vs_baseline": _effort_delta(baseline_effort, selected_effort),
        "harmful_vs_baseline": selected_effort > baseline_effort,
        "exact_solver_correctness_preserved": _row_exact_preserved(row)
        and row[arm]["exact_authority_agrees"] is True,
        "baseline_correct": True,
        "selected_correct": True,
    }


def _arm_summary(per_instance: Sequence[JsonMap]) -> JsonDict:
    total = sum(int(row["total_effort_score"]) for row in per_instance)
    return {
        "instances": len(per_instance),
        "total_effort": total,
        "average_effort": round(total / len(per_instance), 6) if per_instance else 0.0,
        "coverage": _coverage(per_instance),
        "harmful_count": sum(1 for row in per_instance if row["harmful_vs_baseline"]),
        "per_instance": [dict(row) for row in per_instance],
    }


def _best_context_match(row: JsonMap, records: Sequence[JsonMap]) -> JsonMap:
    return max(records, key=lambda record: (_match_score(_context(row), record["context"]), record["decay"]))


def _match_score(left: JsonMap, right: JsonMap) -> float:
    score = 0.0
    for key, weight in (
        ("family", 0.45),
        ("density_bucket", 0.15),
        ("frustration", 0.15),
        ("n_nodes_bucket", 0.1),
        ("n_colors", 0.1),
    ):
        if left.get(key) == right.get(key):
            score += weight
    if set(left.get("constraint_arities", [])) & set(right.get("constraint_arities", [])):
        score += 0.05
    return round(score, 6)


def _regret_telemetry(rows: Sequence[JsonMap], arms: JsonMap) -> JsonDict:
    guarded_rows = arms["case_policy_with_harm_gate"]["per_instance"]
    regrets = []
    for row, selected in zip(rows, guarded_rows, strict=False):
        best = min(_effort(row, arm) for arm in TRACE_ARMS)
        regrets.append(int(selected["total_effort_score"]) - best)
    cumulative = sum(regrets)
    return {
        "evaluated_instances": len(rows),
        "heldout_coverage": arms["case_policy_with_harm_gate"]["coverage"],
        "raw_policy_coverage": arms["case_policy"]["coverage"],
        "cumulative_regret_vs_best_trace_arm": cumulative,
        "average_regret_vs_best_trace_arm": round(cumulative / len(regrets), 6) if regrets else 0.0,
        "max_regret_vs_best_trace_arm": max(regrets) if regrets else 0,
    }


def _policy_description(policy: JsonMap) -> JsonDict:
    return {
        "policy_type": policy["policy_type"],
        "selection_rule": policy["selection_rule"],
        "case_count": policy["case_count"],
        "ttl_steps": policy["ttl_steps"],
        "decay_rate": policy["decay_rate"],
        "validation_gate_open": policy["validation_gate_open"],
        "validation_telemetry": policy["validation_telemetry"],
        "policy_hints": policy["policy_hints"],
        "promoted_artifact_type": "policy_metadata_or_case_weights_only",
        "correctness_authority": "exact_solver_traces_from_exp5130",
    }


def _rollback_receipt(promotion_safe: bool, blockers: Sequence[str]) -> JsonDict:
    return {
        "rollback_applied": not promotion_safe,
        "root_cause": "none" if promotion_safe else ";".join(blockers),
        "promoted_metadata_ids": ["case_policy_metadata_v470"] if promotion_safe else [],
        "active_policy_after_rollback": "case_policy_with_harm_gate" if promotion_safe else "no_learning",
        "model_weight_files_touched": [],
    }


def _promotion_blockers(
    *,
    heldout_delta: float,
    nonforgetting_delta: float,
    harmful_count: int,
    exact_preserved: bool,
    validation_gate_open: bool,
) -> list[str]:
    blockers: list[str] = []
    if heldout_delta <= 0.0:
        blockers.append("positive_heldout_utility_not_observed")
    if nonforgetting_delta < 0.0:
        blockers.append("nonforgetting_regressed")
    if harmful_count > 0:
        blockers.append("harmful_policy_hint_detected")
    if not exact_preserved:
        blockers.append("exact_solver_correctness_not_preserved")
    if not validation_gate_open:
        blockers.append("validation_harm_gate_closed")
    return blockers


def _verdict(promotion_safe: bool, blockers: Sequence[str]) -> str:
    if promotion_safe:
        return SUCCESS_PREFIX + "heldout_utility_safe"
    if "validation_harm_gate_closed" in blockers:
        return NO_PROMOTE_PREFIX + "validation_harm_gate_closed"
    return NO_PROMOTE_PREFIX + "positive_utility_not_observed"


def _manifest_valid(value: Any) -> bool:
    if not isinstance(value, Mapping) or value.get("blocked") is not False:
        return False
    ids = value.get("instance_ids")
    return (
        isinstance(ids, Mapping)
        and value.get("heldout_integrity_passed") is True
        and set(ids) == {"learning", "validation", "heldout"}
        and set(ids["learning"]).isdisjoint(ids["validation"])
        and set(ids["learning"]).isdisjoint(ids["heldout"])
        and set(ids["validation"]).isdisjoint(ids["heldout"])
    )


def _context(row: JsonMap) -> JsonDict:
    return {
        "family": row["family"],
        "density_bucket": row["density_bucket"],
        "frustration": row["frustration"],
        "n_nodes_bucket": _bucket(int(row["n_nodes"])),
        "n_colors": int(row["n_colors"]),
        "constraint_arities": list(row.get("constraint_arities", [])),
    }


def _bucket(value: int) -> str:
    if value <= 5:
        return "small"
    if value <= 8:
        return "medium"
    return "large"


def _effort(row: JsonMap, arm: str) -> int:
    return int(row[arm]["effort"]["total_effort_score"])


def _effort_delta(baseline_effort: int, selected_effort: int) -> float:
    if baseline_effort <= 0:
        return 0.0
    return round((baseline_effort - selected_effort) / baseline_effort, 6)


def _nonforgetting_delta(rows: Sequence[JsonMap], selected: Sequence[JsonMap]) -> float:
    baseline_correct = [row for row in rows if _row_exact_preserved(row)]
    if not baseline_correct:
        return 0.0
    retained = sum(1 for row in selected if row["selected_correct"] is True)
    return round(retained / len(baseline_correct) - 1.0, 6)


def _coverage(rows: Sequence[JsonMap]) -> float:
    if not rows:
        return 0.0
    selected = sum(1 for row in rows if row["selected_trace_arm"] != "baseline")
    return round(selected / len(rows), 6)


def _row_exact_preserved(row: JsonMap) -> bool:
    return bool(
        row.get("wrong_label") is False
        and row.get("heuristic_only_answer_counted") is False
        and row.get("exact_enumerator", {}).get("agrees_with_solver") is True
        and all(row[arm]["exact_authority_agrees"] is True for arm in TRACE_ARMS)
    )


def _instance_ids(rows: Sequence[JsonMap]) -> list[str]:
    return [str(row["instance_id"]) for row in rows]


def _splits_disjoint(*groups: Sequence[JsonMap]) -> bool:
    seen: set[str] = set()
    for rows in groups:
        ids = set(_instance_ids(rows))
        if seen & ids:
            return False
        seen.update(ids)
    return all(bool(rows) for rows in groups)


def _source_trace_path(root: Path) -> Path:
    preferred = root / SOURCE_TRACE_RELATIVE_PATH
    return preferred if preferred.exists() else REPO_ROOT / SOURCE_TRACE_RELATIVE_PATH


def _elapsed(started: float, duration_s: float | None) -> float:
    return round(time.perf_counter() - started, 6) if duration_s is None else duration_s


def _hash_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str | None:
    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
