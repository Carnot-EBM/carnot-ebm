"""REQ-VERIFY-8335: public cached features determine every reserved decision.

Independent scalar arithmetic checks the fitted predictor without using truth
labels. Repeated arms describe the same sources, not new independent samples.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np

from carnot.verify import sentence_spline_fit_8334 as fitted
from carnot.verify.cached_sentence_custody_8305 import check_predictor
from carnot.verify.local_update_isolation_8306 import scalar_design
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
ALLOWLIST = [
    "unit_id",
    "source_cluster_id",
    "role",
    "slot",
    "source_sha256",
    "answer_sha256",
    "x",
    "status",
    "exclusion_reason",
]
ARMS = [*fitted.ARMS, "frozen_holistic_probability", "sigmoid34"]
PROTOCOL_HASH = "sha256:853709123024de763e96dd688e819f0430205ae6d97d6561a2b95cca23b81c6f"


def validate_bundle(bundle: Json) -> None:
    """Reject foreign fields and identities before any predictor can see them."""
    if set(bundle) != {"slots", "head_manifest", "head_hash", "protocol_hash", "roster"}:
        raise ValueError("bundle_fields")
    seal = bundle["head_manifest"]
    digest = canonical_hash({k: v for k, v in seal.items() if k != "whole_model_sha256"})
    if (
        bundle["head_hash"] != digest
        or seal["whole_model_sha256"] != digest
        or bundle["protocol_hash"] != PROTOCOL_HASH
        or seal["protocol_reference"]["sha256"] != PROTOCOL_HASH
        or seal["reserved_labels_opened"] is not False
        or [h["arm"] for h in seal["heads"]] != fitted.ARMS
        or seal["selected_comparator"] not in fitted.ARMS[1:]
    ):
        raise ValueError("frozen_policy")
    slots, roster = bundle["slots"], bundle["roster"]
    if len(slots) != 128 or len(roster) != 128:
        raise ValueError("original128")
    seen = set()
    for index, (row, original) in enumerate(zip(slots, roster, strict=True)):
        check_predictor(row)
        if (
            row["role"] != "reserved"
            or row["slot"] != index + 1
            or any(row[k] != original[k] for k in ["unit_id", "source_cluster_id"])
            or row["source_sha256"] != original["original_source_sha256"]
            or row["source_cluster_id"] in seen
        ):
            raise ValueError("original_source")
        seen.add(row["source_cluster_id"])
    if sum(r["x"] is not None for r in slots) != 97:
        raise ValueError("historical97_features")


def score(bundle: Json, issued_at: str, *, independent: bool = False) -> list[Json]:
    """Unavailable slots escalate because absent evidence has no probability."""
    validate_bundle(bundle)
    heads = {h["arm"]: h for h in bundle["head_manifest"]["heads"]}
    rows = []
    for index, slot in enumerate(bundle["slots"]):
        x = None if slot["x"] is None else [slot["x"][0], *slot["x"][12:16]]
        for arm in ARMS:
            p = None
            if x is not None:
                h = (
                    heads["spline34" if arm == "sigmoid34" else arm]
                    if arm != "frozen_holistic_probability"
                    else None
                )
                if h is None:
                    p = float(fitted.expit(x[0]))
                elif independent:
                    if h["arm"] == "spline34":
                        phi = scalar_design(x)
                    elif h["arm"] == "RBF34":
                        g = h["geometry"]
                        phi = [
                            x[0],
                            1.0,
                            *[
                                math.exp(
                                    -sum((a - b) ** 2 for a, b in zip(x[1:], c, strict=True))
                                    / (2 * g["width"] ** 2)
                                )
                                for c in g["centers"]
                            ],
                        ]
                    else:
                        phi = [x[0], 1.0, *x[1:]] if arm == "linear6" else [x[0], 1.0]
                    z = (
                        sum(a * b for a, b in zip(phi, h["coefficients"], strict=True))
                        / h["temperature"]
                    )
                    p = 1.0 / (1.0 + math.exp(-z))
                elif arm == "sigmoid34":
                    z = (
                        fitted.matrix("spline34", np.asarray([x]), h["geometry"])
                        @ np.asarray(h["coefficients"])
                    )[0] / h["temperature"]
                    p = float(fitted.expit(z))
                else:
                    p = fitted.predict(h, {"features": x})
            rows.append(
                dict(
                    unit_id=slot["unit_id"],
                    source_cluster_id=slot["source_cluster_id"],
                    source_sha256=slot["source_sha256"],
                    slot=slot["slot"],
                    arm=arm,
                    role="stream" if index < 96 else "retention",
                    p=p,
                    action=fitted.action(p),
                    status="completed" if p is not None else "unavailable",
                    original_status=slot["status"],
                    feature_hash=canonical_hash(slot["x"]),
                    head_hash=bundle["head_hash"],
                    protocol_hash=bundle["protocol_hash"],
                    issue_timestamp=issued_at,
                    selected_comparator=arm == bundle["head_manifest"]["selected_comparator"],
                )
            )
        if (index + 1) % 32 == 0:
            print(
                f"[exp8335] phase=scoring completed={index + 1} pending={127 - index}", flush=True
            )
    return rows


def parity(bundle: Json, issued_at: str, rows: list[Json]) -> Json:
    """Independent recursion and a changed prediction qualify replay sensitivity."""
    expected = score(bundle, issued_at, independent=True)
    errors = [
        abs(a["p"] - b["p"]) for a, b in zip(rows, expected, strict=True) if a["p"] is not None
    ]
    identity = all(
        {k: v for k, v in a.items() if k != "p"} == {k: v for k, v in b.items() if k != "p"}
        for a, b in zip(rows, expected, strict=True)
    )
    exact = all(rows[i]["p"] == rows[i + 5]["p"] for i in range(0, len(rows), 6))
    return dict(
        independent_max_error=max(errors),
        exact_sigmoid34=exact,
        passed=identity and exact and max(errors) < 1e-12,
    )
