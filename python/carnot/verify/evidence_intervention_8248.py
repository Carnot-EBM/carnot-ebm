"""REQ-VERIFY-8248: freeze label-blind views of complete original text.

Deletion effects measure model sensitivity. They do not certify entailment,
and a nonselected sentence can still contain useful evidence.
"""

from __future__ import annotations

from collections.abc import Callable
import math
import re
from typing import Any

from carnot.verify import sentence_transport_8179 as transport

Json = dict[str, Any]


def validate_roles(protocol: Json) -> dict[str, int]:
    """Original hashes prevent source reuse even when a unit identifier changes."""
    roles = protocol["role_manifest"]
    sizes = dict(fit=128, calibration=32, selection=32, stream=96, retention=32)
    rows = [r for name in sizes for r in roles[name]]
    if any(len(roles[k]) != v for k, v in sizes.items()) or any(
        len({r[k] for r in rows}) != 320
        for k in ["unit_id", "source_cluster_id", "original_source_sha256"]
    ):
        raise ValueError("roles")
    original = protocol["original_roles"]
    tune = sorted(original["tune"], key=lambda r: r["source_cluster_id"])
    evaluation = sorted(original["evaluation"], key=lambda r: r["source_cluster_id"])
    if roles != dict(
        fit=original["fit"],
        calibration=tune[:32],
        selection=tune[32:],
        stream=evaluation[:96],
        retention=evaluation[96:],
    ):
        raise ValueError("partition")
    return sizes


def view(row: Json, cached: list[Json], count: Callable[[str], int]) -> Json:
    """Choose complete sentences without looking at target labels or citations."""
    source, answer = (bytes.fromhex(row[k]) for k in ["source_bytes", "answer_bytes"])
    sources = [s for s in transport.partition(source) if s["complete"]]
    answers = {s["sentence_index"]: s for s in transport.partition(answer) if s["complete"]}
    usable = [
        r
        for r in cached
        if r["sentence_index"] in answers
        and isinstance(r.get("p_unsupported"), (int, float))
        and math.isfinite(r["p_unsupported"])
        and 0 <= r["p_unsupported"] <= 1
    ]
    reason = (
        "missing_focal_cached_probability"
        if not usable
        else ("no_second_complete_source_sentence" if len(sources) < 2 else None)
    )
    if reason:
        return dict(
            status="unavailable",
            unavailable_reason=reason,
            views={},
            added_features=None,
            current_predictions_available=False,
        )
    focal = min(usable, key=lambda r: (-r["p_unsupported"], r["sentence_index"]))
    sentence = answers[focal["sentence_index"]]
    words: Callable[[str], set[str]] = lambda text: set(
        re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
    )
    target = words(answer[sentence["byte_start"] : sentence["byte_end"]].decode())
    diagnostics = []
    for s in sources:
        text = source[s["byte_start"] : s["byte_end"]].decode()
        tokens = words(text)
        overlap = len(tokens & target) / len(tokens | target) if tokens | target else 0.0
        diagnostics.append(dict(s, lexical_overlap=overlap, embedded_tokens=count(text)))
    selected = min(diagnostics, key=lambda s: (-s["lexical_overlap"], s["sentence_index"]))
    control = min(
        (s for s in diagnostics if s != selected),
        key=lambda s: (
            abs(s["embedded_tokens"] - selected["embedded_tokens"]),
            s["sentence_index"],
        ),
    )
    views = {}
    for name, segment in [
        ("original", None),
        ("selected_evidence_deleted", selected),
        ("nonselected_deletion", control),
    ]:
        changed = (
            source
            if segment is None
            else (source[: segment["byte_start"]] + source[segment["byte_end"] :])
        )
        inputs = dict(source_bytes=changed.hex(), answer_bytes=answer.hex())
        views[name] = dict(inputs, **transport.requests(inputs, count))
    original_count = count(source.decode())
    return dict(
        status="completed",
        unavailable_reason=None,
        views=views,
        focal_sentence_index=focal["sentence_index"],
        cached_focal_probability=focal["p_unsupported"],
        cached_relation=focal["relation"] if "relation" in focal else "B",
        selected_sentence_index=selected["sentence_index"],
        control_sentence_index=control["sentence_index"],
        lexical_overlap_strength=selected["lexical_overlap"],
        match_distance=abs(selected["embedded_tokens"] - control["embedded_tokens"]),
        selected_removed_tokens=selected["embedded_tokens"],
        control_removed_tokens=control["embedded_tokens"],
        original_source_tokens=original_count,
        removal_length_difference=(selected["embedded_tokens"] - control["embedded_tokens"])
        / original_count,
        added_features=None,
        current_predictions_available=False,
    )


def added_features(
    probabilities: list[float | None], length_difference: float
) -> list[float] | None:
    """Unavailable current predictions cannot become synthetic treatment evidence."""
    if any(p is None for p in probabilities):
        return None
    values = [float(p) for p in probabilities if p is not None]
    if len(values) != 3 or any(not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError("probability")
    original, selected, control = values
    return [
        original,
        selected - original,
        control - original,
        selected - control,
        length_difference,
    ]
