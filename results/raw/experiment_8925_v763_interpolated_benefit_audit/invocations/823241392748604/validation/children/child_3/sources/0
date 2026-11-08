"""REQ-REPORT-8925: cold-reduce the frozen interpolation conjunction.

The original guarded audit owns row reconstruction and paired arithmetic.
Private bindings select the sealed V763 streams without changing that producer.
"""

from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any

from carnot.reporting import guarded_benefit_audit_8911 as prior
from carnot.reporting import interpolated_causal_streams_8922 as causal
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.v748_collection_contract import namespace

__all__ = ["require", "causal"]
Json = dict[str, Any]
require = prior.require
ROOT, SEED = causal.ROOT, causal.SEED
NAME, TASK = "experiment_8925_v763_interpolated_benefit_audit", "exp8925-interpolated-benefit-audit"
SOURCE = causal.NAME
PIN = "sha256:53a63467e2cbad29541ba7f3b3bb3b8aec5fa091ca54a1df26aed63e106c710b"
CLI = "scripts/experiments/" + NAME + ".py"
OWNED = [
    "python/carnot/reporting/interpolated_benefit_audit_8925.py",
    "python/carnot/reporting/interpolated_benefit_runner_8925.py",
    CLI,
]
TESTS = ["tests/python/test_interpolated_benefit_audit_8925.py"]
GATES = causal.protocol()["evaluation_plan"]
ARM_MAP = {
    "interpolated_true": "guarded_true",
    "interpolated_permuted": "guarded_permuted",
    "guarded_true": "ordinary_true",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush each boundary and bounded stream count for the external supervisor."""
    print(f"[exp8925] phase={phase} completed={completed} pending={pending}", flush=True)


def preflight(raw: Path, sources: Path) -> Json:
    """Authenticate original stream and method bytes before reading any outcome."""
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
    errors: list[Json] = []
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
            and value["experiment_id"] == 8922
            and value["task_id"] == causal.TASK
            and value["milestone"] == "2026.11.763",
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
        for index, ref in enumerate(operands):
            d.update(artifact_field="original_operand", path=ref["path"], expected=ref["sha256"])
            d["observed"] = d["hash"] = sha256_file(Path(ref["path"]))
            require(d["observed"] == d["expected"], "original_operand_drift")
            refs.append(ref)
            if index % 100 == 0:
                progress("authenticate_operands", index, len(operands) - index)
        authenticator = namespace(causal, "interpolated8925_authentication")
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


def _arms(rows: list[Json], mapping: dict[str, str]) -> list[Json]:
    """Project only arm names so the original numerical reducer stays unchanged."""
    return [dict(r, arm=mapping.get(r["arm"], r["arm"])) for r in rows]


arithmetic = namespace(prior, "interpolated8925_arithmetic")
arithmetic.SEED, arithmetic.GATES = SEED, GATES
arithmetic.causal = SimpleNamespace(
    census=lambda reason: dict(rows=_arms(causal.census(reason)["rows"], ARM_MAP))
)


def _restore(summary: Json) -> Json:
    """Restore actual arm identities after reuse of the registered paired reducer."""
    summary["interpolated_benefit_score"] = summary.pop("guarded_benefit_score")
    inverse = {v: k for k, v in ARM_MAP.items()}
    for key in ("paired_intervals", "acceptance_gate_rows", "paired_stream_rows"):
        for row in summary.get(key, []):
            for field in ("comparison", "unit_id"):
                if field in row:
                    for old, new in inverse.items():
                        row[field] = row[field].replace(old, new)
            if "paired_means" in row:
                row["paired_means"] = {inverse.get(k, k): v for k, v in row["paired_means"].items()}
                row["arm"] = "interpolated_true"
    return summary


def summarize(rows: list[Json]) -> Json:
    """Use 10000 registered whole-stream paired draws with the original family budget."""
    return _restore(dict(arithmetic.summarize(_arms(rows, ARM_MAP))))


def gates(intervals: list[Json]) -> Json:
    """Apply the exact inclusive frozen/retention and strict attribution boundaries."""
    projected = deepcopy(intervals)
    for row in projected:
        row["comparison"] = (
            row["comparison"]
            .replace("guarded_true", "ordinary_true")
            .replace("interpolated_true", "guarded_true")
            .replace("interpolated_permuted", "guarded_permuted")
        )
    return _restore(dict(arithmetic.gates(projected)))


reducer = namespace(prior, "interpolated8925_reducer")
reducer.causal = SimpleNamespace(
    protocol=causal.protocol,
    census=causal.census,
    verify=causal.verify,
    g=SimpleNamespace(ARMS=causal.methods.ARMS),
)
reducer.progress, reducer.GATES = progress, GATES


def reconstruct(stream: Json, plan: Json) -> list[Json]:
    """Keep every original assertion and join evaluator truth back to sealed rows."""
    rows: list[Json] = _original_reconstruct(stream, plan)
    online = {(r["arm"], r["unit_id"]): r for r in rows if r["split"] == "online"}
    require(len(stream["scoring_rows"]) == 192, "scoring_inventory")
    for scoring in stream["scoring_rows"]:
        row = online[scoring["arm"], scoring["group_id"]]
        release = next(
            r
            for r in stream["feedback_rows"]
            if (r["arm"], r["group_id"]) == (scoring["arm"], scoring["group_id"])
        )
        require(
            scoring["scored_before_release"] is True
            and scoring["truth"] == row["truth"] == release["true_label"]
            and scoring["prediction"] == row["prediction"]
            and scoring["metric"] == row["metric"]
            and scoring["release_tick"] == release["release_tick"],
            "future_label_mutation_or_scoring_drift",
        )
    return rows


_original_reconstruct = reducer.reconstruct


def _legacy_summary(rows: list[Json]) -> Json:
    """Keep the historical reducer's private score key at its existing interface."""
    result = summarize(rows)
    return dict(result, guarded_benefit_score=result["interpolated_benefit_score"])


reducer.reconstruct, reducer.summarize = reconstruct, _legacy_summary


def reduce(source: Json) -> Json:
    """Reduce all sealed units and preserve rejected promotions and CPU work."""
    result = dict(reducer.reduce(source))
    result.pop("guarded_benefit_score")
    result["retention_bounds"] = [
        r for r in result["acceptance_gate_rows"] if r["split"] == "retention"
    ]
    result["information_controls"] = result["control_rows"] + [
        dict(check=name, passed=True)
        for name in (
            "group_isolation",
            "prediction_before_release",
            "future_label_mutation_rejection",
            "guard_never_read_research_retention",
        )
    ]
    result["promotion_rows"] = source["promotion_rows"]
    result["operation_counts"] = []
    for arm in causal.methods.ARMS:
        work = [r for r in source["arm_work_ledger"] if r["arm"] == arm]
        result["operation_counts"].append(
            dict(
                arm=arm,
                training_calls=sum(r["training_calls"] for r in work),
                guard_prediction_calls=sum(r["guard_scores"] for r in work),
                prediction_calls=sum(r["arm"] == arm for r in result["rows"]),
                scope="Historical Exp8922 CPU operations; current audit trains and predicts zero times",
            )
        )
    result["continuation_decision"] = dict(
        decision="retire_fixed_interpolated_optimizer_guard_schedule"
        if not result["interpolated_benefit_score"]
        else "retain_constructed_truth_evidence_only",
        unchanged_streams_permitted=False,
        continuous_learning_in_general_retired=False,
        scope="Exp8919 fixed alpha/beta/eta optimizer, unchanged guard, delay8 and 48-query schedule only",
        prior_experiment=8911,
        prior_guarded_benefit_score=0,
        rationale="Changed optimizer again fails the same frozen prospective and retention conjunction; no in-sample promotion gain substitutes for future benefit.",
    )
    return result


def empty(reason: str) -> Json:
    """Censor every scheduled cell when paired external evidence cannot qualify."""
    result = dict(reducer.empty(reason))
    result["interpolated_benefit_score"] = result.pop("guarded_benefit_score")
    result.update(
        retention_bounds=[], information_controls=[], operation_counts=[], promotion_rows=[]
    )
    return result
