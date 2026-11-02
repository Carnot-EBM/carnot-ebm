"""REQ-REPORT-8004: reconstruct costs without producer acceptance functions.

Repeated seeds describe sensitivity. Source means preserve the actual number
of independent observations, and unknown targets never become negative labels.
"""

from collections import defaultdict
from itertools import combinations
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file

Json = dict[str, Any]


def score(rows: list[Json], targets: Json) -> Json:
    """Compute source means before paired contrasts to avoid repeated-seed inflation."""
    cells: Any = defaultdict(list)
    unknown, excluded = 0, 0
    for index, r in enumerate(rows):
        if index % 4096 == 0:
            print(f"[exp8004] phase=primitive_scoring rows={index}/{len(rows)}", flush=True)
        if not isinstance(r, dict):
            raise ValueError("malformed_primitive_row")
        p = r.get("probability", r.get("p"))
        if p is None:
            continue
        if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("invalid_probability")
        if (
            r.get("eligibility", r.get("eligible", True)) is False
            or r.get("failure_status")
            or r.get("censor_status")
        ):
            excluded += 1
            continue
        y = r.get("y", targets.get(r.get("family_id")))
        if type(y) is not int or y not in (0, 1):
            unknown += 1
            continue
        action = r.get("action", r.get("decision", "escalate"))
        cost = {"accept": 5 * y, "reject": 1 - y, "escalate": 0.25}[action]
        if "actual_cost" in r and r["actual_cost"] != cost:
            raise ValueError("primitive_cost_drift")
        source = r.get("source_cluster_id", r.get("source_group", r.get("family_id")))
        if source is None:
            raise ValueError("missing_source_identity")
        key = (
            str(r.get("arm", "default")),
            str(r.get("role", "evaluation")),
            str(source),
            str(r.get("family_id")),
        )
        cells[key].append(
            (float(p), y, cost, int(action == "accept" and y == 1), int(action != "escalate"))
        )
    sources: Any = defaultdict(lambda: defaultdict(list))
    for (arm, role, source, _), values in cells.items():
        if len({v[1] for v in values}) != 1:
            raise ValueError("source_target_drift")
        p = sum(v[0] for v in values) / len(values)
        sources[arm, role][source].append(
            [(p - values[0][1]) ** 2, *[sum(v[i] for v in values) / len(values) for i in (2, 3, 4)]]
        )
    means = {
        key: {s: np.mean(v, axis=0).tolist() for s, v in group.items()}
        for key, group in sources.items()
    }
    metrics = [
        dict(
            arm=k[0],
            role=k[1],
            independent_sources=len(group),
            brier=float(np.mean([v[0] for v in group.values()])),
            cost=float(np.mean([v[1] for v in group.values()])),
            false_accept=float(np.mean([v[2] for v in group.values()])),
            automation=float(np.mean([v[3] for v in group.values()])),
        )
        for k, group in sorted(means.items())
    ]
    contrasts = []
    for left, right in combinations(sorted(means), 2):
        if left[1] != right[1]:
            continue
        paired = sorted(set(means[left]) & set(means[right]))
        diffs = [np.subtract(means[left][s], means[right][s]) for s in paired]
        contrasts.append(
            dict(
                left=left[0],
                right=right[0],
                role=left[1],
                independent_sources=len(paired),
                unmatched_sources=len(set(means[left]) ^ set(means[right])),
                cost_left_minus_right=float(np.mean([d[1] for d in diffs])) if diffs else None,
                brier_left_minus_right=float(np.mean([d[0] for d in diffs])) if diffs else None,
                descriptive_only=True,
            )
        )
    return dict(
        metrics=metrics,
        comparisons=contrasts,
        scored_rows=sum(len(v) for v in cells.values()),
        independent_sources=len({k[2] for k in cells}),
        unknown_labels=unknown,
        excluded_rows=excluded,
    )


def audit(data: Json, targets: Json) -> Json:
    """Recompute issued confidence and exclusive service spans from their raw events."""
    rows = data.get("rows", [])
    result = score(rows, targets)
    service: Any = defaultdict(list)
    confidence: Any = defaultdict(list)
    for r in rows:
        if "wall_ns" in r:
            ticks, spans = r["ticks_ns"], r["exclusive_phase_spans"]
            differences = [b - a for a, b in zip(ticks, ticks[1:])]
            phases = (
                "byte_parsing",
                "public_features",
                "head_prediction",
                "typed_decision",
                "acquisition_selection",
                "feedback_processing",
                "checkpoint_serialization",
                "storage_fsync",
            )
            ordered = (
                [spans[k] for k in phases] if set(spans) == set(phases) else list(spans.values())
            )
            if differences != ordered:
                raise ValueError("exclusive_span_drift")
            if min(differences) < 0 or sum(differences) != r["wall_ns"]:
                raise ValueError("wall_span_drift")
            cached = r["wall_ns"] / 1e9
            acquisition = r.get("acquisition")
            complete = cached + acquisition["duration_s"] if acquisition else None
            if not math.isclose(cached, r["cached_total_s"], abs_tol=1e-12) or (
                complete is not None
                and not math.isclose(complete, r["complete_total_s"], abs_tol=1e-12)
            ):
                raise ValueError("service_cost_drift")
            service[r["arm"], r["case"]].append((cached, complete, spans.get("head_prediction", 0)))
        if "prediction_set" in r and "y" in r:
            if r["release_slot"] < r["issue_slot"] + r["delay"]:
                raise ValueError("feedback_before_release")
            error = int(r["y"] not in r["prediction_set"])
            if r["error"] != error:
                raise ValueError("issued_error_drift")
            confidence[r["arm"], r["delay"]].append(
                (1 - error, len(r["prediction_set"]), int(r["action"] == "escalate"))
            )
    result["service"] = [
        dict(
            arm=k[0],
            case=k[1],
            observations=len(v),
            cached_p50_s=float(np.quantile([x[0] for x in v], 0.5)),
            cached_p95_s=float(np.quantile([x[0] for x in v], 0.95)),
            complete_p50_s=float(np.quantile(known, 0.5))
            if (known := [x[1] for x in v if x[1] is not None])
            else None,
            complete_p95_s=float(np.quantile(known, 0.95)) if known else None,
            missing_acquisition=len(v) - len(known),
            accelerated_fraction=sum(x[2] for x in v) / (sum(x[0] for x in v) * 1e9),
        )
        for k, v in sorted(service.items())
    ]
    result["confidence"] = [
        dict(
            arm=k[0],
            delay=k[1],
            released=len(v),
            coverage=sum(x[0] for x in v) / len(v),
            mean_set_size=sum(x[1] for x in v) / len(v),
            escalation=sum(x[2] for x in v) / len(v),
            population_guarantee=False,
        )
        for k, v in sorted(confidence.items())
    ]
    result["retention"] = score(data.get("retention_rows", []), targets)
    result["causal_updates"] = causal(data)
    return result


def causal(data: Json) -> Json:
    """Check delayed update equations against saved coefficients and unweighted decay."""
    refs, checked, maximum = [], 0, 0.0
    rows = [r for r in data.get("independent_reduction_rows", []) if r.get("gradients")]
    if not rows:
        return dict(updates_checked=0, maximum_gradient_error=None, references=[])

    def load(ref: Json) -> Json:
        path = Path(ref["path"])
        if sha256_file(path) != ref["sha256"]:
            raise ValueError("causal_reference_hash_drift")
        refs.append(dict(path=str(path), sha256=ref["sha256"], role="causal_primitive"))
        return dict(json.loads(path.read_bytes()))

    bundle = load(data["checkpoints"]["bundle"])
    for row in rows:
        saved = bundle["trajectories"][f"{row['arm']}-{row['seed']}"]
        directory = Path(saved["state_directory"])
        for g in load(row["gradients"])["rows"]:
            if g["due_slot"] < g["origin_slot"] + 20:
                raise ValueError("causal_feedback_before_release")
            heads = []
            for slot in (g["due_slot"] - 1, g["due_slot"]):
                path = directory / f"committed-{slot:04d}.json"
                record = json.loads(path.read_bytes())
                if canonical_hash(record["state"]) != record["checksum"]:
                    raise ValueError("causal_state_checksum_drift")
                refs.append(
                    dict(path=str(path), sha256=sha256_file(path), role="causal_committed_state")
                )
                heads.append(record["state"]["head"])
            before, after = heads
            ids = g["coefficient_ids"]
            observed = (
                (np.asarray(before["parameters"])[ids] - np.asarray(after["parameters"])[ids])
                * after["decay_scale"]
                / (0.01 * g["weight"])
            )
            error = float(np.max(np.abs(observed - g["data_gradient"])))
            if error > 1e-10 or not math.isclose(
                after["decay_scale"], before["decay_scale"] * 0.99998, abs_tol=1e-12
            ):
                raise ValueError("causal_gradient_or_decay_drift")
            checked += 1
            maximum = max(maximum, error)
        print(
            f"[exp8004] phase=causal_reconstruction trajectories_checked={row['arm']}-{row['seed']} updates={checked}",
            flush=True,
        )
    return dict(
        updates_checked=checked,
        maximum_gradient_error=maximum,
        references=refs,
        claim_scope="finite_development_equations_only",
    )


def controls() -> Json:
    """Both benefit and no-headroom fixtures check whether the reducer can see change."""
    base = dict(
        source_cluster_id="control",
        family_id="control",
        y=0,
        probability=0.5,
        arm="frozen",
        action="escalate",
    )
    known = score([base, dict(base, arm="learned", probability=0.01, action="accept")], {})
    flat = score(
        [
            dict(base, probability=0.01, action="accept"),
            dict(base, arm="learned", probability=0.01, action="accept"),
        ],
        {},
    )
    return dict(
        passed=known["comparisons"][0]["cost_left_minus_right"] > 0
        and flat["comparisons"][0]["cost_left_minus_right"] == 0,
        claim_scope="circular_positive",
        verifier_is_oracle=True,
        independent_natural_evidence=False,
        known_benefit=known,
        no_headroom=flat,
    )
