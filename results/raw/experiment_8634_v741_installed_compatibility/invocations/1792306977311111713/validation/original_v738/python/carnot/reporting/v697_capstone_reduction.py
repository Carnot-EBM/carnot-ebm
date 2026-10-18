"""REQ-REPORT-8056: independently reconstruct available finite scientific operands."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.v696_capstone_reduction import bootstrap, family
from carnot.verify import learning_benefit_8052 as learning
from carnot.verify import guarded_transaction_8053 as transaction

Json = dict[str, Any]


def independent(data: Json, number: int, copies: Json | None = None) -> Json:
    """Use independent audit equations and keep absent evidence explicitly unavailable."""
    copies = copies or {}

    def bound(ref: Json) -> Path:
        return checked(dict(ref, path=copies.get(ref["path"], ref["path"])))

    if number == 8052 and data.get("rows"):
        bundle = json.loads(bound(data["audit_bundle"]).read_bytes())
        labels = json.loads(bound(bundle["target_reference"]).read_bytes())["rows"]
        trajectory = Path(
            copies.get(bundle["trajectory"] + "/inputs.json", bundle["trajectory"] + "/inputs.json")
        ).parent
        replay = learning.replay(trajectory, {r["family_id"]: r["eligible_y"] for r in labels})
        bundle["retention_target"] = dict(
            bundle["retention_target"], path=str(bound(bundle["retention_target"]))
        )
        retained = learning.retention(
            bundle, replay, bound(data["retention_prediction_seal"]), cold=True
        )
        compared = learning.compare(replay["rows"], retained)
        for fields in (replay, dict(retention_rows=retained), compared):
            for key, observed in fields.items():
                learning.equal("producer_drift:" + key, observed, data[key])
        paired = [r for r in compared["later_source_rows"] if r["comparator"] == "unconstrained"]
        diff = [float("nan")] * 256
        for slot in sorted({r["slot"] for r in paired}):
            diff[slot] = float(np.mean([r["paired_gain"] for r in paired if r["slot"] == slot]))
        gates = compared["primary_hypothesis_results"][0]["gates"]
        return dict(
            measurement_available=True,
            primary=bootstrap(diff, 0.02, 32),
            block_sensitivity=[bootstrap(diff, 0.02, n) for n in (16, 64)],
            scientific_qualified=all(gates.values())
            and compared["retention_passed"]
            and all(r["passed"] for r in compared["per_seed_false_accept_rows"]),
            producer_gates=gates,
            rows=replay["rows"],
            retention_rows=retained,
            **compared,
            independent_gradient_rows=replay["independent_reduction_rows"],
            guard_reconstruction_rows=replay["guard_reconstruction_rows"],
            sample_size_budget=replay["sample_size_budget"],
            numerical_agreement=replay["numerical_agreement"],
        )
    if number == 8053 and data.get("rows"):
        path = str(Path(data["raw_directory"]) / "transaction_rows.json")
        rows = json.loads(Path(copies.get(path, path)).read_bytes())["rows"]
        reduced = transaction.summarize(rows, data["config"])
        for key, observed in reduced.items():
            learning.equal("transaction_drift:" + key, observed, data[key])
        return dict(
            measurement_available=True,
            rows=rows,
            costs=reduced,
            missing_service_components=data["missing_service_components"],
            complete_service_measured=False,
        )
    if number == 8055 and data.get("rows"):
        return dict(
            measurement_available=False,
            board_rows=data["board_rows"],
            guard_fallback_counts=data["guard_fallback_counts"],
            guard_fallback_fraction=data["guard_fallback_fraction"],
            missing_cost_components=data["missing_cost_components"],
            missing_reason="Historical custody and empirical numerical guards do not establish useful deployment.",
        )
    return dict(
        measurement_available=False, missing_reason="no_authenticated_measurement_primitives"
    )
