"""REQ-REPORT-9111-V777: keep operation evidence and row arithmetic distinct.

Sampling is optional during model scoring. Its absence says nothing about a
completed forward, and equal cardinalities need actual original question rows.
"""

from collections import Counter
from typing import Any

Json = dict[str, Any]


def operation(field: str) -> str:
    """Use typed field identities so prose cannot redefine an operation."""
    if field.startswith("generation"):
        return "generation"
    if field.startswith("forward"):
        return "forward"
    if field.startswith("model_load"):
        return "load"
    return "any"


def conflicting_absences(
    live: list[Json], negative: list[Json], *, requires_generation: bool = False
) -> list[Json]:
    """Keep global and same-operation contradictions while accepting unused sampling."""
    positive_operations = {operation(row["field"]) for row in live}
    has_operation = bool(positive_operations & {"load", "forward", "generation"})
    conflicts = []
    fields = {row["path"]: row for row in [*live, *negative]}
    for path, row in fields.items():
        if row["field"].endswith("_completed") and isinstance(row["value"], (int, float)):
            attempt = fields.get(path.removesuffix("_completed") + "_attempted")
            if attempt is not None and row["value"] > attempt["value"]:
                conflicts.append(dict(row, signal="counter_exceeds_attempts"))
    for row in negative:
        kind = operation(row["field"])
        if (
            has_operation
            and kind in {"generation", "forward"}
            and kind not in positive_operations
            and not (kind == "generation" and requires_generation)
        ):
            continue
        if (
            row["field"] == "model_loads_completed"
            and "load" in positive_operations
            and not positive_operations & {"forward", "generation"}
        ):
            continue
        conflicts.append(row)
    return conflicts


def panel_count_derivations(value: Json) -> Json:
    """Prove only question groups and floor(TEST rows/2), never outcome scores.

    Every identity and row denominator must agree across the original roster,
    predictions and per-question reductions. Siblings do not add independent units.
    """
    if value.get("structural_count_evidence") != "qid-panel-v1":
        return {}
    try:
        rows, predictions, qids = (value[k] for k in ["rows", "test_predictions", "qid_rows"])
        if not all(isinstance(r, list) and 0 < len(r) <= 50000 for r in [rows, predictions, qids]):
            raise ValueError("panel_bounds")
        identities: set[str] = set()
        groups: dict[str, tuple[str, str]] = {}
        test: dict[str, tuple[str, str]] = {}
        sizes: Counter[str] = Counter()
        for row in rows:
            unit, qid, group, role = (row[k] for k in ["unit_id", "qid", "group_id", "role"])
            if not all(type(k) is str and 0 < len(k) <= 256 for k in [unit, qid, group]):
                raise ValueError("panel_identity")
            if unit in identities or role not in {"CAL", "TEST"} or row["status"] != "completed":
                raise ValueError("panel_duplicate_or_status")
            identities.add(unit)
            if group in groups and groups[group] != (qid, role):
                raise ValueError("panel_group_role")
            groups[group] = (qid, role)
            if role == "TEST":
                test[unit] = (qid, group)
                sizes[group] += 1
        if len({qid for qid, _role in groups.values()}) != len(groups):
            raise ValueError("panel_qid_alias")
        actual = {r["unit_id"]: (r["qid"], r["group_id"]) for r in predictions}
        if len(actual) != len(predictions) or actual != test:
            raise ValueError("panel_predictions")
        seen: set[str] = set()
        for row in qids:
            group = row["group_id"]
            if group in seen or group not in sizes or row["qid"] != groups[group][0]:
                raise ValueError("panel_qid_rows")
            seen.add(group)
            for key, expected in [
                ("denominator", sizes[group]),
                ("half_denominator", sizes[group] // 2),
            ]:
                if type(row[key]) is not int or row[key] != expected:
                    raise ValueError("panel_row_denominator")
        metrics = value.get("metrics", value)
        derived = dict(
            effective_independent_groups=len(groups),
            denominator=len(test),
            half_denominator=len(test) // 2,
        )
        if seen != set(sizes):
            raise ValueError("panel_qid_missing")
        for key, expected in derived.items():
            declared = value.get(key, metrics.get(key))
            if type(declared) is not int or declared != expected:
                raise ValueError("panel_count_mismatch:" + key)
        return derived
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("panel_shape") from error


def semantic_control_rows() -> list[Json]:
    """Run frozen administrative controls; their scripted counters are never current work."""
    from scripts import adversarial_verify as av

    loaded = dict(model_loads_attempted=1, model_loads_completed=1)
    no_sampling = dict(generation_calls_attempted=0, generation_calls_completed=0)
    cases = [
        (
            "no_load_refusal",
            dict(
                model_invoked=False,
                inference_substrate="deterministic_no_model_load",
                MODEL_SPECS=[dict(hf_id="requested_not_loaded")],
            ),
            av.CLAIM_STATE_NON_LIVE,
        ),
        (
            "failed_attempted_load",
            dict(
                model_invocation_counts=dict(
                    model_loads_attempted=1,
                    model_loads_completed=0,
                    model_loads_failed=1,
                    **no_sampling,
                )
            ),
            av.CLAIM_STATE_LIVE,
        ),
        (
            "zero_sampling_forward",
            dict(model_invocation_counts=dict(**loaded, forward_calls=900, **no_sampling)),
            av.CLAIM_STATE_LIVE,
        ),
        (
            "bounded_generation",
            dict(
                model_invocation_counts=dict(
                    **loaded, generation_calls_attempted=1, generation_calls_completed=1
                )
            ),
            av.CLAIM_STATE_LIVE,
        ),
        (
            "full_generation",
            dict(
                model_invocation_counts=dict(
                    **loaded, generation_calls_attempted=100, generation_calls_completed=100
                )
            ),
            av.CLAIM_STATE_LIVE,
        ),
        (
            "positive_counter_conflict",
            dict(
                model_invocation_counts=dict(
                    generation_calls_attempted=1, generation_calls_completed=2
                )
            ),
            av.CLAIM_STATE_CONTRADICTORY,
        ),
        (
            "forward_without_load",
            dict(
                model_invocation_counts=dict(
                    model_loads_attempted=0, model_loads_completed=0, forward_calls=1
                )
            ),
            av.CLAIM_STATE_CONTRADICTORY,
        ),
        (
            "failed_load_with_forward",
            dict(
                model_invocation_counts=dict(
                    model_loads_attempted=1, model_loads_completed=0, forward_calls=1
                )
            ),
            av.CLAIM_STATE_CONTRADICTORY,
        ),
        (
            "same_generation_conflict",
            dict(
                model_invocation_counts=dict(
                    **loaded, generation_calls_attempted=0, generation_calls_completed=1
                )
            ),
            av.CLAIM_STATE_CONTRADICTORY,
        ),
        (
            "historical_receipt",
            dict(
                model_invoked=False,
                inference_substrate="deterministic_no_model_load",
                historical_receipts=[dict(model_invoked=True, **loaded)],
            ),
            av.CLAIM_STATE_NON_LIVE,
        ),
        (
            "current_receipt",
            dict(model_invoked=False, current_work_receipts=[dict(model_invoked=True, **loaded)]),
            av.CLAIM_STATE_CONTRADICTORY,
        ),
        (
            "global_false_with_load",
            dict(model_invoked=False, model_invocation_counts=loaded),
            av.CLAIM_STATE_CONTRADICTORY,
        ),
    ]
    rows = []
    for name, kind in [
        ("bounded_without_sampling", "model_bounded_generation"),
        ("full_without_sampling", "model_full_generation"),
    ]:
        cases.append(
            (
                name,
                dict(
                    inference_substrate_class=kind,
                    model_invocation_counts=dict(**loaded, **no_sampling),
                ),
                av.CLAIM_STATE_CONTRADICTORY,
            )
        )
    for name, payload, expected in cases:
        if name in {"bounded_generation", "full_generation"}:
            payload["inference_substrate_class"] = "model_" + name
        observed = av._classify_current_task_inference_claim(payload)["state"]
        passed = observed == expected
        rows.append(
            dict(
                unit_id=name,
                qid=None,
                group_id=name,
                arm="semantic_control",
                condition="scripted_diagnostic",
                seed=7779111,
                absolute_metric=int(passed),
                numerator=int(passed),
                denominator=1,
                status="completed" if passed else "failed",
                exclusion_reason=None,
                censor_reason=None,
                fixture_payload=payload,
                expected=expected,
                observed=observed,
                passed=passed,
            )
        )
    return rows
