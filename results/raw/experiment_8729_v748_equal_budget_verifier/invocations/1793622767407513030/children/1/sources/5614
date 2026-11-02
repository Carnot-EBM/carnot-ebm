"""REQ-REPORT-8652: numerical replay does not create new independent questions."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from carnot.reporting import v742_capstone_evidence as e
from carnot.reporting import arc_reset_8649 as provenance
from carnot.reporting import arc_reset_transfer_8650 as transfer
from carnot.reporting.lean_resident_8608 import energy_module

Json = dict[str, Any]


def reduce() -> Json:
    """Cold-reduce original losses, windows, clocks and reset causes without new training."""
    refs: list[Json] = []

    def load(ref: Json) -> Json:
        """Every decoded operand also enters the invocation's immutable source closure."""
        refs.append(ref)
        return dict(e.read(ref))

    def producer(number: int) -> Json:
        """One original primary prevents summary copies from becoming evidence."""
        paths = list((e.ROOT / "results").glob(f"experiment_{number}_*.json"))
        if len(paths) != 1:
            raise ValueError("original_producer_identity")
        return load(e.reference(paths[0]))

    p = e.methods.protocol()
    learning = load(p["original_producers"]["8582"])
    predictions = load(learning["prediction_rows_reference"])["rows"][:2]
    losses = [
        dict(
            unit_id=r["unit_id"],
            brier=(r["probability"] - r["evaluator_target"]) ** 2,
            matches_original=(r["probability"] - r["evaluator_target"]) ** 2 == r["brier"],
            qualified=False,
        )
        for r in predictions
    ]
    retention_ref = learning["retention_rows_reference"]
    refs.append(retention_ref)
    if not e.methods.checked(retention_ref):
        raise ValueError("retention_hash")
    retention = json.loads(Path(retention_ref["path"]).read_bytes())
    maximum = max((r["probability"] - r["target"]) ** 2 - r["baseline_brier"] for r in retention)
    retained = dict(
        maximum_brier_increase=maximum,
        matches_original=maximum == max(r["brier"] - r["baseline_brier"] for r in retention),
        qualified=False,
    )
    feedback_ref = learning["released_feedback_reference"]
    refs.append(feedback_ref)
    if not e.methods.checked(feedback_ref):
        raise ValueError("feedback_hash")
    feedback = json.loads(Path(feedback_ref["path"]).read_bytes())
    released = []
    for row in feedback[:2]:
        primitive = row["primitive"]
        update = load(primitive)["updates"][int(primitive["json_pointer"].rsplit("/", 1)[1])]
        probability, target = update["probability_before_update"], update["target"]
        loss = -target * math.log(probability) - (1 - target) * math.log1p(-probability)
        released.append(
            dict(
                feedback_id=row["feedback_id"],
                log_loss=loss,
                matches_original=math.isclose(loss, row["log_loss"], abs_tol=1e-15)
                and row["before_head_hash"] == update["before_head_hash"]
                and row["after_head_hash"] == update["after_head_hash"],
            )
        )
    streaming = producer(8642)
    windows = []
    for row in streaming["window_offset_rows"]:
        ref = row["primitive"]
        refs.append(ref)
        if not e.methods.checked(ref):
            raise ValueError("streaming_window_hash")
        count = len(Path(ref["path"]).read_bytes().splitlines())
        windows.append(
            dict(
                family_id=row["family_id"],
                chunk_size=row["chunk_size"],
                observed_windows=count,
                matches_original=count == row["total_window_count"]
                and ref["sha256"] == "sha256:" + row["complete_window_digest"],
                qualified=False,
                semantic_correctness_established=False,
            )
        )
    installed = producer(8646)
    authority = load(installed["authority_reference"])
    clocks = []
    for session in authority["sessions"]:
        for arm in ["installed", "private"]:
            s = session[arm]
            elapsed = (s["ended_monotonic_ns"] - s["started_monotonic_ns"]) / 1e9
            clocks.append(
                dict(
                    arm=arm,
                    kind=session["kind"],
                    backend=session["arm"],
                    total_s=elapsed,
                    response_count=len(s["responses"]),
                    matches_original=elapsed > 0 and s["exit_code"] == 0,
                    qualified_complete_cost=False,
                )
            )
    current = producer(8650)
    work = load(current["work_reference"])
    episodes = [load(ref) for ref in work["references"]]
    reset = transfer.reduce(episodes)
    reset["matches_original"] = all(reset[k] == current[k] for k in reset)
    reset["qualification_issue_rows"] = [
        dict(
            episode_id=ep["unit"]["episode_id"],
            recorder_errors=sorted(set(ep["recorder_errors"])),
            observer_issue_reasons=sorted({r["reason"] for r in ep["observer_issues"]}),
            reset_issues=ep["reset_provenance"]["issues"],
        )
        for ep in episodes
    ]
    source = producer(8649)
    source_work = load(source["work_reference"])
    for ref in source_work["references"]:
        load(ref)
    rebuilt = provenance.build(
        Path(source["work_reference"]["path"]),
        source["validation_receipts"],
        e.ROOT / "results" / (provenance.NAME + ".json"),
    )
    recording = {
        k: rebuilt[k]
        for k in [
            "action_parity_rows",
            "reset_request_rows",
            "requester_ownership_rows",
            "missing_cause_rows",
        ]
    }
    recording["matches_original"] = all(v == source[k] for k, v in recording.items())
    hardware = producer(8651)
    load(hardware["work_reference"])
    head = load(p["head_checkpoint_reference"])
    data = load(p["learning_data_reference"])["view"]["rows"][0]["x"]
    numeric = energy_module()
    frozen = dict(head["heads"]["spline9"][0], scaler=head["scaler"])
    e.progress("before_small_head_numerical_inference", 12, 1)
    probability = float(numeric.predict(frozen, numeric.np.asarray([data], dtype=float))[0])
    e.progress("after_small_head_numerical_inference", 12, 1)
    return dict(
        causal_losses=losses,
        retention=retained,
        released_feedback=released,
        streaming_windows=windows,
        installed_clocks=clocks,
        arc_reset_transfer=reset,
        reset_provenance=recording,
        hardware={
            k: hardware[k]
            for k in [
                "new_operation_cost_rows",
                "complete_cost_rows",
                "conditional_ceiling_rows",
                "missing_cost_rows",
                "board_status_rows",
                "physical_hardware_ready_score",
            ]
        },
        numerical_head=dict(
            performed=True,
            checkpoint=p["head_checkpoint_reference"],
            checkpoint_head="heads.spline9[0]",
            parameter_count=len(frozen["parameters"]),
            input=data,
            probability=probability,
            current_language_model_calls=0,
        ),
        references=refs,
        independent_units_added=0,
        current_learning=dict(
            observed=None, missing_reason="Exp8643 was pre-gated; Exp8644 cascade skipped"
        ),
        complete_source_coverage=dict(
            observed=None,
            missing_reason="Exp8645 pre-gated; fixture windows do not establish complete roster coverage",
        ),
        complete_installed_cost=dict(
            observed=None,
            missing_reason="Exp8647 pre-gated; compatibility clocks are not the registered complete cost contrast",
        ),
    )
