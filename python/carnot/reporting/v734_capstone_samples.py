"""REQ-REPORT-8551: a small deterministic diagnostic never repairs failed science."""

from pathlib import Path
from typing import Any

from carnot.reporting import v733_capstone_evidence as e
from carnot.reporting import v734_contract_methods as methods

Json = dict[str, Any]


def samples() -> Json:
    """Recompute original operands and ask real consumers whether the claims qualify."""
    from carnot.reporting import v733_update_accounting as learning
    from carnot.reporting import nine_input_qualification_8545 as native
    from carnot.reporting import hardware_operation_evidence_8550 as cost
    from carnot.reporting import arc_capability_panel_8537 as arc

    prior = e.load(methods.protocol()["v734_capstone_reference"])
    sources = {r["experiment_id"]: r for r in prior["rows"]}
    learn_ref = sources[8530]["primary_reference"]
    learn = e.load(learn_ref)
    history = e.load(e.load(learn["primitive_reference"])["history_reference"])
    row = history["rows"][0]
    recomputed = learning.coordinate_row(
        row["before"],
        row["after"],
        row["assignment_indices"],
        event=row["event_id"],
        namespace=row["namespace"],
    )
    terminal = e.load(e.reference(Path(learn["terminal_validation_sidecar_path"])))
    consumed = learning.stream_consumer(
        Path(learn_ref["path"]), Path(terminal["publication"]["sidecar_path"])
    )
    accepted, reason = True, None
    try:
        native.consume(
            methods.ROOT / "results/experiment_8545_v734_nine_input_native_qualification.json"
        )
    except ValueError as error:
        accepted, reason = False, str(error)
    original = e.load(sources[8537]["primary_reference"])
    work = e.load(original["work_reference"])
    observed, effects, exclusions = arc.reduce_episode(work["references"][0])
    expected = next(
        r for r in original["per_episode_rows"] if r["episode_id"] == observed["episode_id"]
    )
    return dict(
        learning=dict(
            source=learn_ref,
            event_id=row["event_id"],
            applied_update_events=recomputed["applied_update_events"],
            changed_coordinate_count=recomputed["changed_coordinate_count"],
            matches_original=recomputed == row,
            consumer=consumed,
        ),
        native_consumer=dict(accepted=accepted, missing_reason=reason),
        native_cost=cost.costs(methods.ROOT),
        arc=dict(
            source=sources[8537]["primary_reference"],
            episode=work["references"][0],
            observed=observed,
            effect_count=len(effects),
            exclusion_count=len(exclusions),
            matches_original=observed == expected,
        ),
        independent_units_added=0,
        scope="diagnostic original operands; all quarantines preserved",
    )
