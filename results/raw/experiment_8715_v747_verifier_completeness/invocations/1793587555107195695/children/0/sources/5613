"""REQ-REPORT-8680: recompute admitted primitives without rerunning science."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from carnot.reporting import v744_capstone_inputs as e
from carnot.reporting import arc_observation_8677 as ownership
from carnot.reporting import arc_observer_transfer_8678 as transfer
from carnot.reporting import v743_capstone_reduce as historical
from carnot.reporting import hardware_boundary_costs_8679 as hardware

Json = dict[str, Any]


def reduce(values: dict[str, Json], p: Json, raw: Path) -> tuple[Json, list[Json]]:
    """Missing new producers stay unmeasured; sealed historical nulls keep their scope."""
    refs: list[Json] = []

    def load(ref: Json) -> Json:
        """Retain each original primitive before deriving any aggregate from it."""
        e.verify(ref)
        refs.append(ref)
        return dict(e.read(ref))

    old_protocol = load(p["inherited_v743_protocol_reference"])
    old_values = {n: load(ref) for n, ref in p["v743_producers"].items()}
    old_values["8665"] = load(
        next(
            r
            for r in old_values["8666"]["cited_upstream_artifacts"]
            if "experiment_8665_" in r["path"]
        )
    )
    old_summary, old_unions, old_refs = historical.reduce(
        old_values, old_protocol, raw / "historical"
    )
    refs.extend(old_refs)
    result: Json = dict(
        historical=old_summary,
        historical_unions=old_unions,
        historical_stamped_outcomes=[
            dict(
                experiment_id=int(n),
                honest_verdict=old_values[n]["honest_verdict"],
                verdict_class=old_values[n]["verdict_class"],
                flagged_adversarial=old_values[n]["flagged_adversarial"],
                adversarial_findings=old_values[n].get("adversarial_findings", []),
                primary_reference=p["v743_producers"][n],
            )
            for n in ("8654", "8664")
        ],
    )
    result["learning"] = [
        dict(
            measured=False,
            current_union_eligible=False,
            current_causal_updates=None,
            question_cluster_future_brier=None,
            simultaneous_retention_outcome=None,
            missing_reason="Exp8668/8669/8671/8672 did not execute; no admitted current learning primitives",
            original_node_count=64,
            original_runner_count=27,
            missing_current_node_count=64,
        )
    ]
    result["semantics"] = [
        dict(
            measured=False,
            collision_witness_count=None,
            independent_generalization=False,
            missing_reason="Exp8670/8673 did not execute; no extracted feature or numerical scoring witnesses",
        )
    ]
    worlds_ref = p["relational_evidence_protocol"]["worlds_reference"]
    worlds = load(worlds_ref)["worlds"]
    evaluator = load(p["relational_evidence_protocol"]["independent_evaluator_reference"])
    registration = []
    for world in worlds:
        for view in world["views"]:
            document = view["document"].encode()
            for name in ("source", "claim"):
                start, end = view[name + "_span"]
                if (
                    not 0 <= start <= end <= len(document)
                    or document[start:end] != view[name].encode()
                ):
                    raise ValueError("original_relational_span_bytes")
            truth = e.methods.evaluate_world(world, view)
            if truth != view["truth"]:
                raise ValueError("independent_constructed_evaluator")
            registration.append(
                dict(
                    world_id=world["world_id"],
                    arm=view["view"],
                    constructed_truth=truth,
                    status="unstarted",
                    measured=False,
                    missing_reason="no_qualified_extractor_or_feature_audit",
                    source_reference=worlds_ref,
                )
            )
    result["constructed_registration"] = dict(
        rows=registration,
        evaluator_reference=p["relational_evidence_protocol"]["independent_evaluator_reference"],
        evaluator_authority=evaluator,
        collision_claim_established=False,
        independent_count=0,
    )
    for number, reducer in [("8677", ownership.reduce), ("8678", transfer.reduce)]:
        value = values[number]
        work = load(value["work_reference"])
        episodes = [load(ref) for ref in work["references"]]
        rebuilt = reducer(episodes)
        if any(value[k] != v for k, v in rebuilt.items()):
            raise ValueError("arc_original_primitive_reduction:" + number)
        result[number] = dict(rebuilt)
        e.progress("arc_primitive_reduction_" + number, len(episodes), 0)
    arc = result["8678"]
    result["arc"] = dict(
        action_parity_pairs=sum(r["passed"] for r in arc["action_parity_rows"]),
        complete_horizon_count=arc["complete_horizon_count"],
        censored_count=len(arc["censored_reason_rows"]),
        observer_error_count=len(arc["observer_error_rows"]),
        independent_games=len(arc["per_game_results"]),
        cross_game_support_rows=arc["cross_game_support_rows"],
        per_game_results=arc["per_game_results"],
        horizon_rows=arc["horizon_rows"],
        historical_exp8664_flagged=True,
        supervisor_benefit=None,
        decision="close_registered_censored_diagnostic; reopen only with changed causal mechanism and uncensored support",
    )
    cells = hardware.source_cells(old_values["8659"], refs)
    value = values["8679"]
    if [dict(c, producer=8659) for c in cells] != value["complete_boundary_rows"]:
        raise ValueError("hardware_original_cost_reduction")
    result["hardware"] = dict(
        board_status_rows=value["board_status_rows"],
        source_admission_rows=value["source_admission_rows"],
        measured_source_cells=len(cells),
        missing_compatible_cells=sum(c["conditional_ceiling"] is None for c in cells),
        physical_hardware_ready_score=0,
        hardware_100x_requirements=value["hardware_100x_requirements"],
        board_reopen_requirements=value["board_reopen_requirements"],
        conditional_bounds_are_achieved_speed=False,
    )
    result["deployment"] = [
        dict(
            measured=False,
            equal_work_batch_comparison=None,
            same_batch_nfr01=None,
            missing_reason="Exp8674/8675 never executed",
            historical_batch_one_nfr01=old_summary["installed"]["nfr01_score"],
            historical_packaging_noninferiority=old_summary["installed"][
                "packaging_noninferiority_score"
            ],
            reopening="Changed batch mechanism, exact request parity,60 qualified cells/61440 evaluations; independent same-batch native/Python lower95>=10",
        )
    ]
    result["qwen"] = dict(
        current_calls=0,
        token_device_receipts=[],
        measured=False,
        missing_reason="No Exp8676 producer; runtime repair and constructed corpus unavailable",
        reopening="Changed CUDA library/environment identity with actual context/copy receipt, then real tokens and device evidence",
    )
    return result, list({r["path"]: r for r in refs}.values())
