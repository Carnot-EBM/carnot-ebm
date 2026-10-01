"""Rebuild response-wide human targets without selecting sentences by label.

REQ-VERIFY-7955. Exact original annotations determine the event; public
bytes alone determine predictor inputs and the frozen window admission.
"""

from __future__ import annotations

import json
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.verify import sentence_labels_7942 as custody
from carnot.verify import source_alignment as alignment

Json = dict[str, Any]
public_only = custody.public_only


def freeze(public: list[Json]) -> tuple[list[Json], list[Json]]:
    """Keep complete bytes and apply the existing label-free window limits."""
    custody.index_unique(public, "family_id")
    audit = []
    for row in public:
        if set(row) != custody.PUBLIC_KEYS:
            raise ValueError("public_fields")
        source, answer = (bytes.fromhex(row[k]) for k in ("source_bytes", "answer_bytes"))
        sources, sentences = alignment.sentence_spans(source), alignment.sentence_spans(answer)
        intervals, start = [], 0
        for sentence in sentences:
            intervals.append([start, start + len(sentence)])
            start += len(sentence)
        windows = len(sources) + max(0, len(sources) - 1)
        reason = (
            "answer_units_over_budget"
            if len(sentences) > alignment.MAX_ANSWER_UNITS
            else "source_windows_over_budget"
            if windows > alignment.MAX_WINDOWS
            else "empty_answer"
            if not sentences
            else None
        )
        audit.append(
            dict(
                family_id=row["family_id"],
                intervals=intervals,
                response_sha256=custody.digest(answer),
                reconstructed=b"".join(sentences) == answer,
                public_eligible=reason is None,
                exclusion_reason=reason,
            )
        )
    return [public_only(row) for row in public], audit


def check_roles(public: list[Json], roles: list[Json]) -> dict[str, Json]:
    """Keep source clusters disjoint across the original exposed roles."""
    indexed = custody.index_unique(roles, "family_id")
    if set(indexed) != {r["family_id"] for r in public}:
        raise ValueError("roster")
    clusters: dict[str, str] = {}
    for row in public:
        role = indexed[row["family_id"]]
        cluster = custody.digest(bytes.fromhex(row["source_bytes"]))
        if cluster != role["source_cluster_id"]:
            raise ValueError("source_cluster_hash")
        if cluster in clusters and clusters[cluster] != role["role"]:
            raise ValueError("source_cluster_role")
        clusters[cluster] = role["role"]
    return indexed


def join(public: list[Json], audit: list[Json], data: Json) -> tuple[list[Json], list[Json]]:
    """Authenticate each response independently; unknown custody never gives zero."""
    if freeze(public) != (public, audit):
        raise ValueError("public_drift")
    roles = check_roles(public, data["roles"])
    ev = custody.index_unique(data["evaluators"], "family_id")
    custody.index_unique(data["evaluators"], "response_id")
    responses = custody.index_unique(data["responses"], "id")
    sources = custody.index_unique(data["sources"], "source_id")
    if set(ev) != set(roles):
        raise ValueError("roster")
    rows, annotations = [], []
    for predictor, boundary in zip(public, audit, strict=True):
        fid = predictor["family_id"]
        role, evaluator = roles[fid], ev[fid]
        if evaluator["role"] != role["role"]:
            raise ValueError("role_drift")
        answer = bytes.fromhex(predictor["answer_bytes"])
        response = responses.get(evaluator["response_id"], {})
        spans, error = [], None
        try:
            source = sources[response["source_id"]]["source_info"]
            serialized = (
                source
                if isinstance(source, str)
                else json.dumps(source, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            )
            if serialized.encode() != bytes.fromhex(predictor["source_bytes"]):
                raise ValueError("source_equality")
            spans = custody.checked_spans(response, answer)
        except (KeyError, ValueError) as failure:
            error = str(failure)
        eligible = (
            error is None
            and response.get("quality") == "good"
            and boundary["public_eligible"]
            and role["status"] == "completed"
        )
        reason = (
            "annotation_custody:" + error
            if error is not None
            else response.get("quality")
            if response.get("quality") != "good"
            else boundary["exclusion_reason"] or "inherited_public_budget"
            if not eligible
            else None
        )
        annotations.extend(
            dict(
                s, family_id=fid, response_id=response["id"], response_sha256=custody.digest(answer)
            )
            for s in spans
        )
        y = int(bool(spans)) if eligible else None
        original = evaluator.get("human_label")
        rows.append(
            dict(
                family_id=fid,
                role=role["role"],
                source_cluster_id=role["source_cluster_id"],
                response_id=evaluator["response_id"],
                source_id=response.get("source_id"),
                status="completed" if eligible else "excluded",
                exclusion_reason=reason,
                custody_passed=error is None,
                quality=response.get("quality"),
                completely_annotated=error is None and response.get("quality") == "good",
                y=y,
                implicit_true_excluded_y=int(any(not s["implicit_true"] for s in spans))
                if eligible
                else None,
                original_response_label=original,
                original_label_disagreement=None
                if y is None or original is None
                else y != original,
                annotation_count=len(spans),
                sentence_intervals=boundary["intervals"],
                response_sha256=custody.digest(answer),
                arm="human_response_transport",
                seed=69055,
            )
        )
    return rows, annotations


def reduce_rows(rows: list[Json]) -> Json:
    """Count responses and independent source clusters rather than sentences."""
    custody.index_unique(rows, "family_id")
    evaluation = [r for r in rows if r["role"] == "evaluation"]
    eligible = [r for r in evaluation if r["status"] == "completed"]
    counts = {str(y): sum(r["y"] == y for r in eligible) for y in (0, 1)}
    clusters = len({r["source_cluster_id"] for r in eligible})
    operands = [
        dict(field="intended_evaluation_responses", op="==", expected=64, observed=len(evaluation)),
        dict(field="eligible_source_clusters", op=">=", expected=32, observed=clusters),
        dict(
            field="annotation_custody",
            op="==",
            expected=True,
            observed=all(r["custody_passed"] for r in evaluation),
        ),
        *[dict(field="class_" + y, op=">=", expected=8, observed=counts[y]) for y in counts],
    ]
    failed = [
        o
        for o in operands
        if (o["observed"] != o["expected"] if o["op"] == "==" else o["observed"] < o["expected"])
    ]
    return dict(
        class_counts=counts,
        source_cluster_counts=dict(
            intended=len({r["source_cluster_id"] for r in evaluation}), eligible=clusters
        ),
        sample_size_budget=dict(
            unit="complete_evaluation_response",
            intended=64,
            eligible=len(eligible),
            started=len(evaluation),
            completed=len(eligible),
            failed=0,
            censored=0,
            excluded=len(evaluation) - len(eligible),
            independent=clusters,
        ),
        response_targets_ready_score=int(not failed),
        failed_operands=failed,
        rows_sha256=canonical_hash(rows),
        disagreement_rows=[r for r in rows if r["original_label_disagreement"]],
        exclusion_rows=[r for r in rows if r["status"] == "excluded"],
    )
