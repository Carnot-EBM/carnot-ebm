"""REQ-REPORT-8954: audit sealed future benefit without reopening training.

The producer's assertions remain intact. This reducer independently pairs
query rows and keeps operational promotions outside the success criterion.
"""

from collections import defaultdict
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any

import numpy as np

from carnot.reporting import gradient_causal_streams_8953 as causal
from carnot.reporting.causal_learning_audit_8810 import require, score
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
__all__ = ["require", "causal"]
ROOT, SEED = causal.ROOT, causal.SEED
NAME, TASK = "experiment_8954_v765_gradient_benefit_audit", "exp8954-gradient-benefit-audit"
SOURCE, PIN = causal.NAME, "sha256:cff93f1c1f76d8eb6a0b9fe185a6d77d21a091879187d2b40e438a8ab87a2ff0"
CLI = "scripts/experiments/" + NAME + ".py"
OWNED = [
    "python/carnot/reporting/gradient_benefit_audit_8954.py",
    "python/carnot/reporting/gradient_benefit_runner_8954.py",
    CLI,
]
TESTS = ["tests/python/test_gradient_benefit_audit_8954.py"]
GATES = causal.protocol()["evaluation_plan"]
COUNT = 27648


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush bounded phase counts so external supervision can detect a stalled audit."""
    print(f"[exp8954] phase={phase} completed={completed} pending={pending}", flush=True)


def preflight(raw: Path, sources: Path) -> Json:
    """Authenticate producer identities and original bytes before reading outcomes."""
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = sources / (SOURCE + ".json")
    d: Json = dict(
        check="missing_file",
        upstream=SOURCE,
        artifact_field="gradient_rows_ready_score",
        expected=1,
        observed=None,
        path=str(path),
        hash=None,
    )
    refs: list[Json] = []
    errors: list[Json] = []
    progress("authenticate_before", 0, 2)
    try:
        d["hash"] = sha256_file(path)
        d["check"] = "rejected_producer"
        value = json.loads(path.read_bytes())
        require(isinstance(value, dict), "producer_object")
        d.update(check="missing_field")
        gate = value["gradient_rows_ready_score"]
        d.update(
            check="observed_zero" if type(gate) is int and gate == 0 else "rejected_producer",
            observed=gate,
        )
        require(type(gate) is int and gate == 1, "upstream_gate")
        d.update(artifact_field="producer_bytes", expected=PIN, observed=d["hash"])
        require(
            d["hash"] == PIN
            and value["experiment_id"] == 8953
            and value["task_id"] == causal.TASK
            and value["milestone"] == "2026.11.765",
            "producer_identity",
        )
        require(
            value["required_checks_passed"] is True
            and value["protocol"] == causal.protocol()
            and value["stream_manifest"] == causal.protocol()["frozen_evaluation_manifest"]
            and value["owned_statement_coverage"]["complete"]
            and value["owned_statement_coverage"]["percent"] == 100
            and value["completed_count"] == COUNT
            and value["independent_count"] == 24
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
        for index, ref in enumerate(operands):
            d.update(artifact_field="original_operand", path=ref["path"], expected=ref["sha256"])
            d["observed"] = d["hash"] = sha256_file(Path(ref["path"]))
            require(d["observed"] == d["expected"], "original_operand_drift")
            refs.append(ref)
            if index % 100 == 0:
                progress("authenticate_operands", index, len(operands) - index)
        authenticator = namespace(causal, "compatible8954_authentication")
        authenticator.methods = SimpleNamespace(**vars(causal.methods))
        anchor_ref = next(
            r
            for r in refs
            if Path(r["path"]).name == "experiment_8905_v762_guarded_update_methods.json"
        )
        authenticator.methods.anchors = lambda: load(anchor_ref)["anchor_manifest"]
        methods = authenticator.preconditions(raw / "methods", sources)
        refs.extend(methods["references"])
        errors.extend(methods["diagnostics"])
        if methods["external_ready"]:
            shutil.copyfile(path, raw / (SOURCE + ".json"))
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(
            dict(
                {k: v for k, v in d.items() if k != "artifact_field"},
                check="missing_file"
                if isinstance(error, OSError)
                else "missing_field"
                if isinstance(error, KeyError)
                else d["check"],
                artifact_field=error.args[0]
                if isinstance(error, KeyError)
                else d["artifact_field"],
                error=str(error),
            )
        )
    progress("authenticate_after", 2, 0)
    return dict(
        external_ready=not errors,
        references=refs,
        diagnostics=errors,
        resources=dict(
            passed=all(
                (ROOT / ".venv/bin" / n).is_file() for n in ("python", "pytest", "ruff", "mypy")
            )
            and shutil.disk_usage(raw).free > 512 * 1024**2,
            total_budget_s=4200,
            hard_cap_s=4800,
        ),
    )


def reconstruct(stream: Json, plan: Json) -> list[Json]:
    """Rejoin sealed predictions, releases and untouched retention without new truth."""
    causal.verify(stream, plan)
    rows: list[Json] = stream["rows"]
    expected = {
        (r["unit_id"], r["arm"]): r
        for r in causal.census("")["rows"]
        if r["cluster_id"] == plan["stream_id"]
    }
    require(
        len(rows) == len(expected) == 1152
        and {(r["unit_id"], r["arm"]) for r in rows} == set(expected),
        "row_roster",
    )
    predictions = {(r["group_id"], r["arm"]): r for r in stream["prediction_rows"]}
    truth = {(r["group_id"], r["arm"]): r["true_label"] for r in stream["feedback_rows"]}
    online = {u["unit_id"] for u in plan["online"]}
    for arm, state in stream["states"].items():
        for field, state_field in (
            ("replay_rows", "selection_rows"),
            ("gradient_ledger", "gradient_ledger"),
        ):
            if arm != "frozen_verified":
                require(
                    [
                        {k: v for k, v in r.items() if k not in ("buffer_matches", "query_cursor")}
                        for r in stream[field]
                        if r["arm"] == arm
                    ]
                    == [
                        dict(r, arm=arm, cluster_id=plan["stream_id"], seed=plan["seed"])
                        for r in state[state_field]
                    ],
                    "diagnostic_custody",
                )
        require(
            state["arm"] == arm
            and set(state["splits"]) == online
            and {r["group_id"] for r in state["released"]} <= online,
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
                and all(
                    r["parameter_hash"] == canonical_hash(state["committed"])
                    for r in predictions.values()
                    if r["arm"] == arm
                ),
                "frozen_control_changed",
            )
    retention = {(r["unit_id"], r["arm"]): r for r in stream["retention_rows"]}
    for row in rows:
        schedule = expected[row["unit_id"], row["arm"]]
        require(
            all(
                row[k] == schedule[k]
                for k in ("cluster_id", "seed", "split", "checkpoint", "ordinal", "denominator")
            )
            and row["metric"] == "brier"
            and row["status"] == "complete"
            and row["missing_reason"] is None,
            "row_schedule",
        )
        sealed = row["sealed"]
        computed = score(sealed["prediction"], row["truth"], 0.5)
        require(
            row["prediction"] == sealed["prediction"]
            and row["value"] == row["numerator"] == computed["metric"]
            and all(row[k] == v for k, v in computed.items() if k != "metric"),
            "sealed_score",
        )
        if row["split"] == "online":
            require(sealed == predictions[row["unit_id"], row["arm"]], "prediction_join")
            if (row["unit_id"], row["arm"]) in truth:
                require(row["truth"] == truth[row["unit_id"], row["arm"]], "future_truth_join")
        else:
            require(
                row == retention[row["unit_id"], row["arm"]]
                and any(
                    s["state_hash"] == sealed["state_hash"] and s["arm"] == row["arm"]
                    for s in stream["state_hashes"]
                ),
                "retention_state_join",
            )
    return rows


def gates(intervals: list[Json]) -> Json:
    """Keep frozen and retention equality inclusive while attribution remains strict."""
    checks = [
        dict(
            r,
            threshold=b["upper"],
            operator="<" if b["strict"] else "<=",
            passed=bool(r["upper"] < b["upper"] if b["strict"] else r["upper"] <= b["upper"]),
        )
        for r, b in zip(intervals, GATES["bounds"])
    ]
    return dict(
        acceptance_gate_rows=checks,
        gradient_benefit_score=int(
            len(intervals) == len(checks) == 11 and all(r["passed"] for r in checks)
        ),
    )


def summarize(rows: list[Json]) -> Json:
    """Resample whole paired streams once for all eleven predeclared upper bounds."""
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
        len(rows) == COUNT
        and {tuple(r[k] for k in fields) for r in rows}
        == {tuple(r[k] for k in fields) for r in causal.census("")["rows"]}
        and all(
            r["status"] == "complete"
            and r["metric"] == "brier"
            and np.isfinite(r["value"])
            and np.isfinite(r["error"])
            for r in rows
        ),
        "complete_paired_schedule",
    )
    groups: dict[tuple[Any, ...], list[Json]] = defaultdict(list)
    for row in rows:
        if row["split"] == "retention" or 25 <= row["ordinal"] <= 48:
            groups[row["cluster_id"], row["split"], row["checkpoint"], row["arm"]].append(row)
    streams = sorted({r["cluster_id"] for r in rows})
    draws = np.random.default_rng(SEED).integers(24, size=(10000, 24))
    paired, intervals = [], []
    progress("bootstrap_before", 0, 11)
    for bound in GATES["bounds"]:
        split, checkpoint, other = bound["split"], bound.get("checkpoint"), bound["against"]
        metric = "value" if bound["endpoint"] == "brier" else "error"
        deltas = []
        for cluster in streams:
            means = {
                arm: float(np.mean([r[metric] for r in groups[cluster, split, checkpoint, arm]]))
                for arm in ("gradient_replay", other)
            }
            delta = means["gradient_replay"] - means[other]
            deltas.append(delta)
            paired.append(
                dict(
                    unit_id=f"{cluster}/{split}/{checkpoint}/{metric}/{other}",
                    cluster_id=cluster,
                    seed=groups[cluster, split, checkpoint, other][0]["seed"],
                    split=split,
                    checkpoint=checkpoint,
                    endpoint=bound["endpoint"],
                    comparison="gradient_replay_minus_" + other,
                    paired_means=means,
                    value=delta,
                    numerator=delta,
                    denominator=1,
                    query_denominator=len(groups[cluster, split, checkpoint, other]),
                    arm="gradient_replay",
                    status="complete",
                    missing_reason=None,
                )
            )
        samples = np.mean(np.asarray(deltas)[draws], axis=1)
        intervals.append(
            dict(
                split=split,
                checkpoint=checkpoint,
                endpoint=bound["endpoint"],
                comparison="gradient_replay_minus_" + other,
                mean=float(np.mean(deltas)),
                upper=float(np.quantile(samples, 1 - GATES["alpha_per_bound"])),
                paired_stream_means=deltas,
                independent_n=24,
            )
        )
    progress("bootstrap_after", 11, 0)
    return dict(
        paired_stream_rows=paired,
        paired_intervals=intervals,
        multiplicity_ledger=dict(
            GATES,
            method="one-sided paired stream percentile bootstrap; simultaneous99-percent family",
            evaluation_units_used_for_selection=0,
            original_ten_conditions=[
                b for b in GATES["bounds"] if b["against"] != "signature_replay"
            ],
            added_signature_condition=GATES["bounds"][3],
        ),
        **gates(intervals),
    )


def reduce(source: Json) -> Json:
    """Recompute every comparative aggregate from authenticated paired stream rows."""
    plans = causal.protocol()["frozen_evaluation_manifest"]
    require(
        source["protocol"] == causal.protocol()
        and len(source["stream_refs"]) == 24
        and len({r["path"] for r in source["stream_refs"]}) == 24
        and len(source["rows"]) == COUNT,
        "frozen_roster",
    )
    imported = {(r["unit_id"], r["arm"]): r for r in source["rows"]}
    rows: list[Json] = []
    controls: list[Json] = []
    promotions: list[Json] = []
    updates: list[Json] = []
    selections: list[Json] = []
    hashes: list[Json] = []
    for index, (ref, plan) in enumerate(zip(source["stream_refs"], plans, strict=True)):
        progress("stream_before", index, 24 - index)
        stream = load(ref)
        rebuilt = reconstruct(stream, plan)
        require(all(imported[r["unit_id"], r["arm"]] == r for r in rebuilt), "imported_row_drift")
        rows.extend(rebuilt)
        promotions.extend(stream["promotion_rows"])
        updates.extend(stream["gradient_ledger"])
        selections.extend(stream["replay_rows"])
        controls.extend(dict(r, cluster_id=plan["stream_id"]) for r in stream["control_rows"])
        hashes.append(
            dict(
                cluster_id=plan["stream_id"],
                row_hash=canonical_hash(rebuilt),
                shard=ref,
                row_count=len(rebuilt),
            )
        )
        progress("stream_after", index + 1, 23 - index)
    summary = summarize(rows)
    promotion_summary, diversity = [], []
    for arm in causal.methods.ARMS:
        attempts = [r for r in promotions if r["arm"] == arm]
        admitted = sum(r["promoted"] for r in attempts)
        promotion_summary.append(
            dict(
                arm=arm,
                attempts=len(attempts),
                promoted=admitted,
                rejected=len(attempts) - admitted,
                promotion_frequency=admitted / len(attempts),
                scope="operational diagnostic; not benefit",
            )
        )
        selected = [r for r in selections if r["arm"] == arm and r["selected_id"] is not None]
        if selected:
            frequencies: dict[str, int] = defaultdict(int)
            for r in selected:
                frequencies[r["selected_id"]] += 1
            diversity.append(
                dict(
                    arm=arm,
                    selections=len(selected),
                    unique_selected_groups=len(frequencies),
                    unique_fraction=len(frequencies) / len(selected),
                    empty_buffer_repeats=sum(r["empty_buffer"] for r in selected),
                    maximum_group_reuse=max(frequencies.values()),
                    per_stream=[
                        dict(
                            cluster_id=p["stream_id"],
                            selections=sum(r["cluster_id"] == p["stream_id"] for r in selected),
                            unique_selected_groups=len(
                                {
                                    r["selected_id"]
                                    for r in selected
                                    if r["cluster_id"] == p["stream_id"]
                                }
                            ),
                        )
                        for p in plans
                    ],
                    scope="selection diversity only; cannot replace failed primary metrics",
                )
            )
    return dict(
        rows=rows,
        **summary,
        control_rows=controls,
        row_hashes=hashes,
        retention_bounds=[r for r in summary["acceptance_gate_rows"] if r["split"] == "retention"],
        information_controls=controls
        + [
            dict(check=c, passed=True)
            for c in (
                "complete_paired_roster",
                "sealed_score_and_truth_joins",
                "retention_group_isolation",
                "frozen_control_unchanged",
            )
        ],
        promotion_rows=promotions,
        promotion_summary=promotion_summary,
        selection_diversity=diversity,
        mechanism_diagnostics=mechanism_diagnostics(selections, updates, promotions, summary),
        continuation_decision=dict(
            decision="retain_constructed_truth_evidence_only"
            if summary["gradient_benefit_score"]
            else "retire_gradient_cosine_buffer32_sgd0.1_delay8_guard_48query",
            unchanged_streams_permitted=False,
            continuous_learning_in_general_retired=False,
            scope="Fixed signed-local-v1 representation, gradient cosine, TRAIN buffer32, SGD0.1, two applied gradients, delay8, unchanged guard and48-query schedule only",
            recommendation="changed_representation_or_new_causal_diagnosis_before_another_attempt",
            rationale="All eleven prospective/retention bounds are required; training gains, promotions and diversity cannot substitute for future benefit.",
        ),
        intended_count=COUNT,
        completed_count=COUNT,
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=24,
    )


def mechanism_diagnostics(
    selections: list[Json], updates: list[Json], promotions: list[Json], summary: Json
) -> Json:
    """Separate descriptive mechanism changes from the required future-loss test."""
    paired, diversity, guards = [], [], []
    for plan in causal.protocol()["frozen_evaluation_manifest"]:
        cluster = plan["stream_id"]
        chosen = [
            r for r in selections if r["cluster_id"] == cluster and r["arm"] == "gradient_replay"
        ]
        signature = {
            r["released_id"]: r
            for r in selections
            if r["cluster_id"] == cluster and r["arm"] == "signature_replay"
        }
        applied = {
            r["released_id"]: r
            for r in updates
            if r["cluster_id"] == cluster
            and r["arm"] == "gradient_replay"
            and r["role"] == "replay"
        }
        baseline = {
            r["released_id"]: r
            for r in updates
            if r["cluster_id"] == cluster
            and r["arm"] == "signature_replay"
            and r["role"] == "replay"
        }
        disagreement = sum(r["selected_id"] != r["signature_id"] for r in chosen)
        paired.append(
            dict(
                cluster_id=cluster,
                metric="gradient_vs_own_W1_signature_selection_disagreement",
                numerator=disagreement,
                denominator=len(chosen),
                value=disagreement / len(chosen),
                paired_arm_selection_disagreements=sum(
                    r["selected_id"] != signature[r["released_id"]]["selected_id"] for r in chosen
                ),
                paired_replay_gradient_disagreements=sum(
                    r["gradient_hash"] != baseline[k]["gradient_hash"] for k, r in applied.items()
                ),
                future_brier_delta_vs_signature=next(
                    r["value"]
                    for r in summary["paired_stream_rows"]
                    if r["cluster_id"] == cluster
                    and r["split"] == "online"
                    and r["comparison"] == "gradient_replay_minus_signature_replay"
                ),
                status="complete",
                censor_reason=None,
            )
        )
    for arm in causal.methods.ARMS:
        attempts = [r for r in promotions if r["arm"] == arm]
        rejected = sum(not r["promoted"] for r in attempts)
        guards.append(
            dict(
                arm=arm,
                metric="guard_rejection_frequency",
                numerator=rejected,
                denominator=len(attempts),
                value=rejected / len(attempts),
                eligible_rejected=sum(r["eligible"] and not r["promoted"] for r in attempts),
                status="complete",
                censor_reason=None,
            )
        )
        if arm != "frozen_verified":
            actual = [r for r in updates if r["arm"] == arm]
            replay = [r for r in actual if r["role"] == "replay"]
            unique = len({r["gradient_hash"] for r in replay})
            diversity.append(
                dict(
                    arm=arm,
                    metric="actual_replay_gradient_diversity",
                    numerator=unique,
                    denominator=len(replay),
                    value=unique / len(replay),
                    applied_update_count=len(actual),
                    unique_parameter_transitions=len({(r["before"], r["after"]) for r in actual}),
                    nonzero_applied_updates=sum(r["parameter_delta"] > 0 for r in actual),
                    status="complete",
                    censor_reason=None,
                )
            )
    return dict(
        paired_stream_rows=paired,
        actual_update_diversity=diversity,
        guard_rejections=guards,
        cannot_rescue_failed_bound=True,
        independent_unit="whole stream; updates, queries and restart controls are not independent samples",
    )


def empty(reason: str) -> Json:
    """Keep every scheduled row when missing evidence or owned failure prevents reduction."""
    return dict(
        rows=causal.census(reason)["rows"],
        paired_stream_rows=[],
        paired_intervals=[],
        acceptance_gate_rows=[],
        multiplicity_ledger=dict(GATES),
        retention_bounds=[],
        information_controls=[],
        control_rows=[],
        row_hashes=[],
        promotion_rows=[],
        promotion_summary=[],
        selection_diversity=[],
        mechanism_diagnostics=dict(
            paired_stream_rows=[],
            actual_update_diversity=[],
            guard_rejections=[],
            cannot_rescue_failed_bound=True,
        ),
        continuation_decision=dict(
            decision="await_original_external_evidence_or_owned_repair", missing_reason=reason
        ),
        gradient_benefit_score=0,
        intended_count=COUNT,
        completed_count=0,
        failed_count=0,
        censored_count=COUNT,
        excluded_count=0,
        independent_count=0,
    )
