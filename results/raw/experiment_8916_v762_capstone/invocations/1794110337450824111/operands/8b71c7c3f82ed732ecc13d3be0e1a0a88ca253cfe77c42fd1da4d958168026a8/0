"""REQ-REPORT-8911: reduce sealed guarded updates without fitting or oracle access.

Only the registered stream clusters are resampled. Operational promotion and
constructed truth cannot establish independent natural-language generalization.
"""

from collections import defaultdict
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np

from carnot.memory import causal_memory_8802 as primitive
from carnot.reporting import guarded_causal_streams_8909 as causal
from carnot.reporting import guarded_update_methods_8905 as methods
from carnot.reporting.causal_learning_audit_8810 import require, score
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.v746_retained_evidence import reference, load

__all__ = ["require", "causal"]
Json = dict[str, Any]
ROOT, SEED = causal.ROOT, causal.SEED
NAME, TASK = "experiment_8911_v762_guarded_benefit_audit", "exp8911-guarded-benefit-audit"
SOURCE, PIN = causal.NAME, "sha256:4ad993f42d183f5b2807a0341e7511ed82b6803b8d49885ce85deb949b7732a3"
CLI = "scripts/experiments/" + NAME + ".py"
OWNED = [
    "python/carnot/reporting/guarded_benefit_audit_8911.py",
    "python/carnot/reporting/guarded_benefit_runner_8911.py",
    CLI,
]
TESTS = ["tests/python/test_guarded_benefit_audit_8911.py"]
GATES: Json = dict(
    family_intervals=9,
    family_alpha=0.01,
    upper_tail_alpha=0.01 / 9,
    bootstrap_draws=10000,
    seed=SEED,
    future_ordinals=[25, 48],
    future_frozen_upper=-0.01,
    future_permuted_upper_strict=0,
    future_ordinary_upper_strict=0,
    retention_upper=0.01,
    retention_checkpoints=[0, 24, 48],
    independent_streams=24,
    scheduled_rows=18432,
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush completed and pending work so a fixed deadline cannot hide a stalled child."""
    print(f"[exp8911] phase={phase} completed={completed} pending={pending}", flush=True)


def preflight(raw: Path, sources: Path) -> Json:
    """Authenticate the actual producers and all operand bytes before interpreting effects."""
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = sources / (SOURCE + ".json")
    d: Json = dict(
        check="missing_file",
        upstream=SOURCE,
        artifact_field="causal_rows_ready_score",
        expected=1,
        observed=None,
        path=str(path),
        hash=None,
    )
    refs: list[Json] = []
    diagnostics: list[Json] = []
    progress("authenticate_before", 0, 2)
    try:
        d["hash"] = sha256_file(path)
        d["check"] = "rejected_producer"
        value = json.loads(path.read_bytes())
        require(isinstance(value, dict), "producer_object")
        d.update(check="missing_field", observed=None)
        gate = value["causal_rows_ready_score"]
        d.update(check="observed_zero" if gate == 0 else "rejected_producer", observed=gate)
        require(type(gate) is int and gate == 1, "upstream_gate")
        d.update(artifact_field="producer_bytes", expected=PIN, observed=d["hash"])
        require(
            d["hash"] == PIN
            and value["experiment_id"] == 8909
            and value["task_id"] == causal.TASK
            and value["milestone"] == "2026.11.762",
            "producer_identity",
        )
        require(
            value["required_checks_passed"] is True
            and value["protocol"] == causal.protocol()
            and value["cluster_manifest"] == causal.protocol()["frozen_evaluation_manifest"]
            and value["owned_statement_coverage"]["complete"]
            and value["owned_statement_coverage"]["percent"] == 100
            and bool(value["control_rows"])
            and all(r["passed"] for r in value["control_rows"]),
            "original_qualification",
        )
        refs.append(reference(path))
        operands = (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        )
        for i, ref in enumerate(operands):
            d.update(artifact_field="original_operand", path=ref["path"], expected=ref["sha256"])
            d["observed"] = d["hash"] = sha256_file(Path(ref["path"]))
            require(d["observed"] == d["expected"], "original_operand_drift")
            refs.append(ref)
            if i % 50 == 0:
                progress("authenticate_operands", i, len(operands) - i)
        method_path = sources / (causal.SOURCE + ".json")
        d = dict(
            check="missing_file",
            upstream=causal.SOURCE,
            artifact_field="guarded_methods_ready_score",
            expected=1,
            observed=None,
            path=str(method_path),
            hash=None,
        )
        d["hash"] = sha256_file(method_path)
        d["check"] = "rejected_producer"
        method = load(reference(method_path))
        for field in ("guarded_methods_ready_score", "candidate_ready_score"):
            d.update(check="missing_field", artifact_field=field, expected=1, observed=None)
            observed = method[field]
            d.update(
                check="observed_zero" if observed == 0 else "rejected_producer", observed=observed
            )
            require(type(observed) is int and observed == 1, "method_gate")
        d.update(
            check="rejected_producer",
            artifact_field="producer_bytes",
            expected=causal.PIN,
            observed=d["hash"],
        )
        require(
            d["hash"] == causal.PIN
            and method["experiment_id"] == 8905
            and method["task_id"] == methods.TASK
            and method["milestone"] == "2026.11.762"
            and method["required_checks_passed"] is True
            and method["protocol"] == value["protocol"]
            and method["frozen_evaluation_manifest"] == value["cluster_manifest"]
            and method["owned_statement_coverage"]["complete"]
            and method["owned_statement_coverage"]["percent"] == 100
            and bool(method["control_rows"])
            and all(r["passed"] for r in method["control_rows"]),
            "method_identity_and_contract",
        )
        causal.disjoint(method["protocol"], method["anchor_manifest"])
        refs.append(reference(method_path))
        operands = (
            method["source_artifact_hashes"]
            + method["code_config_hashes"]
            + method["raw_shard_hashes"]
        )
        for i, ref in enumerate(operands):
            d.update(artifact_field="original_operand", path=ref["path"], expected=ref["sha256"])
            d["observed"] = d["hash"] = sha256_file(Path(ref["path"]))
            require(d["observed"] == d["expected"], "original_method_operand_drift")
            refs.append(ref)
            if i % 50 == 0:
                progress("authenticate_method_operands", i, len(operands) - i)
        shutil.copyfile(method_path, raw / (causal.SOURCE + ".json"))
        shutil.copyfile(path, raw / (SOURCE + ".json"))
    except (OSError, ValueError, KeyError, TypeError) as error:
        diagnostics.append(
            dict(
                d,
                check="missing_file"
                if isinstance(error, OSError)
                else "missing_field"
                if isinstance(error, KeyError)
                else d["check"],
                error=str(error),
            )
        )
    progress("authenticate_after", 2, 0)
    return dict(
        external_ready=not diagnostics,
        references=refs,
        diagnostics=diagnostics,
        resources=dict(
            passed=all(
                (ROOT / ".venv/bin" / n).is_file() for n in ("python", "pytest", "ruff", "mypy")
            )
            and shutil.disk_usage(raw).free > 512 * 1024**2,
            pilot_total_limit_s=4200,
            hard_cap_s=4800,
        ),
    )


def reconstruct(stream: Json, plan: Json) -> list[Json]:
    """Rejoin recorded seals, truth, states and complete identities without new labels."""
    causal.verify(stream, plan)
    expected = {
        (r["unit_id"], r["arm"]): r
        for r in causal.census("")["rows"]
        if r["cluster_id"] == plan["stream_id"]
    }
    rows = stream["rows"]
    require(
        len(rows) == len(expected) == 768
        and {(r["unit_id"], r["arm"]) for r in rows} == set(expected),
        "row_roster",
    )
    online = {u["unit_id"] for u in plan["online"]}
    initial = {r["arm"]: r["parameter_hash"] for r in stream["state_hashes"][:4]}
    for arm, state in stream["states"].items():
        require(
            state["arm"] == arm
            and set(state["splits"]) == online
            and {r["group_id"] for r in state["released"]} == online,
            "retention_leakage",
        )
        for promotion in state["promotions"]:
            require(
                set(promotion["training_groups"] + promotion["guard_groups"]) <= online,
                "retention_leakage",
            )
        if arm == "frozen_verified":
            require(
                state["committed_version"] == 0
                and canonical_hash(state["committed"]) == initial[arm]
                and all(
                    r["parameter_hash"] == initial[arm]
                    for r in stream["state_hashes"]
                    if r["arm"] == arm
                ),
                "frozen_control_changed",
            )
    for row in rows:
        schedule = expected[row["unit_id"], row["arm"]]
        require(
            all(
                row[k] == schedule[k]
                for k in ("cluster_id", "seed", "split", "checkpoint", "ordinal", "denominator")
            )
            and row["status"] == "complete"
            and row["missing_reason"] is None,
            "row_schedule_or_failure",
        )
        sealed = row.get("sealed") or next(
            r
            for r in stream["prediction_rows"]
            if (r["arm"], r["ordinal"]) == (row["arm"], row["ordinal"])
        )
        require(
            row["prediction"] == sealed["prediction"]
            and all(row[k] == v for k, v in score(sealed["prediction"], row["truth"], 0.5).items()),
            "sealed_score",
        )
        require(
            row["error"]
            == int(
                primitive.LABELS[max(range(3), key=sealed["prediction"].__getitem__)]
                != row["truth"]
            ),
            "error_score",
        )
        tx = next(
            r
            for r in stream["state_hashes"]
            if r["arm"] == row["arm"]
            and r["state_hash"] == sealed.get("state_hash", r["state_hash"])
            and r["parameter_hash"] == sealed["parameter_hash"]
            and r["template_hash"] == sealed["template_hash"]
        )
        require(tx["parameter_hash"] == sealed["parameter_hash"], "state_join")
        if row["arm"] == "frozen_verified":
            require(sealed["parameter_hash"] == initial[row["arm"]], "frozen_prediction_changed")
    return list(rows)


def gates(intervals: list[Json]) -> Json:
    """Keep strict attribution tests separate from inclusive benefit and retention margins."""
    rows = []
    for r in intervals:
        strict = r["split"] == "online" and r["comparison"] != "guarded_true_minus_frozen_verified"
        threshold = 0 if strict else -0.01 if r["split"] == "online" else 0.01
        rows.append(
            dict(
                r,
                threshold=threshold,
                operator="<" if strict else "<=",
                passed=bool(r["upper"] < threshold if strict else r["upper"] <= threshold),
            )
        )
    return dict(
        acceptance_gate_rows=rows,
        guarded_benefit_score=int(len(rows) == 9 and all(r["passed"] for r in rows)),
    )


def summarize(rows: list[Json]) -> Json:
    """Reuse paired stream bootstrap draws with the registered one-sided family budget."""
    roster = causal.census("")["rows"]
    fields = (
        "unit_id",
        "cluster_id",
        "arm",
        "seed",
        "split",
        "checkpoint",
        "ordinal",
        "denominator",
    )
    require(
        len(rows) == 18432
        and {tuple(r[k] for k in fields) for r in rows}
        == {tuple(r[k] for k in fields) for r in roster}
        and all(
            r["status"] == "complete" and np.isfinite(r["metric"]) and np.isfinite(r["error"])
            for r in rows
        ),
        "complete_paired_schedule",
    )
    groups: dict[tuple[Any, ...], list[Json]] = defaultdict(list)
    for row in rows:
        if row["split"] == "retention" or row["ordinal"] >= 25:
            groups[row["cluster_id"], row["split"], row["checkpoint"], row["arm"]].append(row)
    streams = sorted({r["cluster_id"] for r in rows})
    # The same paired draws serve all contrasts; query rows never enter the resampling unit.
    draws = np.random.default_rng(SEED).integers(24, size=(10000, 24))
    specs: list[tuple[str, int | None, str, str]] = [
        ("online", None, "metric", arm)
        for arm in ("frozen_verified", "guarded_permuted", "ordinary_true")
    ]
    specs += [
        ("retention", step, metric, "frozen_verified")
        for step in (0, 24, 48)
        for metric in ("metric", "error")
    ]
    paired, intervals = [], []
    progress("bootstrap_before", 0, 9)
    for split, checkpoint, metric, other in specs:
        deltas = []
        for cluster in streams:
            means = {
                arm: float(np.mean([r[metric] for r in groups[cluster, split, checkpoint, arm]]))
                for arm in ("guarded_true", other)
            }
            delta = means["guarded_true"] - means[other]
            deltas.append(delta)
            paired.append(
                dict(
                    unit_id=f"{cluster}/{split}/{checkpoint}/{metric}/{other}",
                    cluster_id=cluster,
                    seed=groups[cluster, split, checkpoint, other][0]["seed"],
                    split=split,
                    checkpoint=checkpoint,
                    endpoint="brier" if metric == "metric" else "error",
                    comparison="guarded_true_minus_" + other,
                    paired_means=means,
                    metric=delta,
                    denominator=len(groups[cluster, split, checkpoint, other]),
                    arm="guarded_true",
                    status="complete",
                    missing_reason=None,
                )
            )
        samples = np.mean(np.asarray(deltas)[draws], axis=1)
        intervals.append(
            dict(
                split=split,
                checkpoint=checkpoint,
                endpoint="brier" if metric == "metric" else "error",
                comparison="guarded_true_minus_" + other,
                mean=float(np.mean(deltas)),
                upper=float(np.quantile(samples, 1 - GATES["upper_tail_alpha"])),
                paired_stream_means=deltas,
                independent_n=24,
            )
        )
    progress("bootstrap_after", 9, 0)
    return dict(
        paired_stream_rows=paired,
        paired_intervals=intervals,
        multiplicity_ledger=dict(
            GATES,
            method="paired stream percentile bootstrap; one-sided simultaneous99-percent family",
            prototype="label_benefit_audit_8886.summarize",
            development_pilots_evaluation_units=0,
            operational_guards_evaluation_units=0,
        ),
        **gates(intervals),
    )


def reduce(source: Json) -> Json:
    """Independently reduce every original shard and retain adverse outcomes and diagnostics."""
    p = causal.protocol()
    require(
        source["protocol"] == p
        and len(source["stream_refs"]) == 24
        and len({r["path"] for r in source["stream_refs"]}) == 24
        and len(source["rows"]) == 18432,
        "frozen_roster",
    )
    imported = {(r["unit_id"], r["arm"]): r for r in source["rows"]}
    rows, checks, promotions, row_hashes = [], [], [], []
    began = time.monotonic()
    for i, (ref, plan) in enumerate(
        zip(source["stream_refs"], p["frozen_evaluation_manifest"], strict=True)
    ):
        require(time.monotonic() - began < 1800, "reduction_deadline")
        progress("stream_before", i, 24 - i)
        stream = load(ref)
        rebuilt = reconstruct(stream, plan)
        require(all(imported[r["unit_id"], r["arm"]] == r for r in rebuilt), "imported_row_drift")
        rows.extend(rebuilt)
        promotions.extend(stream["promotion_rows"])
        row_hashes.append(
            dict(
                cluster_id=plan["stream_id"],
                row_hash=canonical_hash(rebuilt),
                shard=ref,
                row_count=len(rebuilt),
            )
        )
        checks.append(
            dict(
                check="independent_sealed_schedule_states_retention_frozen_control",
                cluster_id=plan["stream_id"],
                passed=True,
            )
        )
        progress("stream_after", i + 1, 23 - i)
    summary = summarize(rows)
    promotion_summary = []
    for arm in causal.g.ARMS:
        selected = [r for r in promotions if r["arm"] == arm]
        promoted = sum(r["promoted"] for r in selected)
        promotion_summary.append(
            dict(
                arm=arm,
                attempts=len(selected),
                promoted=promoted,
                rejected=len(selected) - promoted,
                promotion_frequency=promoted / len(selected),
                rejection_frequency=1 - promoted / len(selected),
                scope="operational diagnostic only",
            )
        )
    retention_summary, calibration_summary = [], []
    for split, checkpoints in (("online", [None]), ("retention", [0, 24, 48])):
        for checkpoint in checkpoints:
            for arm in causal.g.ARMS:
                selected = [
                    r
                    for r in rows
                    if (r["split"], r["checkpoint"], r["arm"]) == (split, checkpoint, arm)
                    and (split == "retention" or r["ordinal"] >= 25)
                ]
                confidence = float(np.mean([max(r["prediction"]) for r in selected]))
                error = float(np.mean([r["error"] for r in selected]))
                diagnostic = dict(
                    split=split,
                    checkpoint=checkpoint,
                    arm=arm,
                    denominator=len(selected),
                    independent_n=24,
                    brier=float(np.mean([r["metric"] for r in selected])),
                    error=error,
                    mean_confidence=confidence,
                    accuracy=1 - error,
                    calibration_gap=confidence - (1 - error),
                    scope="descriptive; no added success criterion",
                )
                calibration_summary.append(diagnostic)
                if split == "retention":
                    retention_summary.append(diagnostic)
    return dict(
        rows=rows,
        **summary,
        control_rows=checks,
        row_hashes=row_hashes,
        promotion_summary=promotion_summary,
        retention_summary=retention_summary,
        calibration_summary=calibration_summary,
        continuation_decision=dict(
            decision="retire_unchanged_guarded_update_mechanism"
            if not summary["guarded_benefit_score"]
            else "retain_constructed_truth_evidence_only",
            unchanged_streams_permitted=False,
            rationale="Same prespecified nine-bound conjunction; thresholds remain frozen.",
        ),
        intended_count=18432,
        completed_count=18432,
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=24,
    )


def empty(reason: str) -> Json:
    """Retain every scheduled slot when original external evidence cannot support an audit."""
    return dict(
        rows=causal.census(reason)["rows"],
        paired_stream_rows=[],
        paired_intervals=[],
        acceptance_gate_rows=[],
        multiplicity_ledger=dict(GATES),
        control_rows=[],
        row_hashes=[],
        promotion_summary=[],
        retention_summary=[],
        calibration_summary=[],
        continuation_decision=dict(
            decision="await_original_external_evidence", missing_reason=reason
        ),
        guarded_benefit_score=0,
        intended_count=18432,
        completed_count=0,
        failed_count=0,
        censored_count=18432,
        excluded_count=0,
        independent_count=0,
    )
