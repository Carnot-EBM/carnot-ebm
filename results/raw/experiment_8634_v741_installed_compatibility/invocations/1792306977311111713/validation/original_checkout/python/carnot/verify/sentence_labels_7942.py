"""Join fallible human spans to exact sentences without changing public inputs.

REQ-VERIFY-7942. Public sentence selection happens before evaluator access.
Source support is the annotation target, not a formal certificate of truth.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.verify import source_alignment

Json = dict[str, Any]
PUBLIC_KEYS = frozenset({"family_id", "source_bytes", "answer_bytes"})
PREDICTOR_KEYS = PUBLIC_KEYS | {"sentence_interval"}
SALT = b"v689-sentence-1"


def digest(data: bytes) -> str:
    """Bind source identity to original bytes rather than normalized text."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def public_only(row: Json) -> Json:
    """Remove all evaluator metadata before public sentence selection."""
    return {key: row[key] for key in PUBLIC_KEYS}


def index_unique(rows: list[Json], key: str) -> dict[str, Json]:
    """An ambiguous identity cannot authorize an annotation join."""
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate:" + key + (":response_reused" if key == "response_id" else ""))
    return result


def freeze(public: list[Json]) -> tuple[list[Json], list[Json]]:
    """Retain every byte and choose queries using the registered hash rule."""
    index_unique(public, "family_id")
    predictors, boundaries = [], []
    for row in public:
        if set(row) != PUBLIC_KEYS:
            raise ValueError("public_fields")
        answer = bytes.fromhex(row["answer_bytes"])
        parts = source_alignment.sentence_spans(answer)
        if not parts:
            raise ValueError("empty_answer")
        intervals, start = [], 0
        for part in parts:
            end = start + len(part)
            intervals.append([start, end])
            start = end
        selected = min(
            intervals,
            key=lambda interval: hashlib.sha256(
                row["family_id"].encode()
                + json.dumps(interval, separators=(",", ":")).encode()
                + SALT
            ).digest(),
        )
        predictors.append({**row, "sentence_interval": selected})
        boundaries.append(
            dict(
                family_id=row["family_id"],
                intervals=intervals,
                response_sha256=digest(answer),
                reconstructed=b"".join(parts) == answer,
            )
        )
    return predictors, boundaries


def features(row: Json) -> list[float]:
    """Use only public source bytes and the selected complete answer sentence."""
    start, end = row["sentence_interval"]
    return source_alignment.pair_features(
        bytes.fromhex(row["source_bytes"]), [bytes.fromhex(row["answer_bytes"])[start:end]]
    )


def checked_spans(response: Json, answer: bytes) -> list[Json]:
    """Check original character offsets before deriving UTF-8 byte offsets."""
    text = answer.decode("utf-8", "strict")
    if response.get("response") != text:
        raise ValueError("response_equality")
    if response.get("quality") not in {"good", "incorrect_refusal", "truncated"}:
        raise ValueError("quality")
    if not isinstance(response.get("labels"), list):
        raise ValueError("annotations")
    result = []
    for span in response["labels"]:
        if (
            not isinstance(span, dict)
            or not {
                "start",
                "end",
                "text",
                "label_type",
                "implicit_true",
                "due_to_null",
                "meta",
            }.issubset(span)
            or type(span["implicit_true"]) is not bool
            or type(span["due_to_null"]) is not bool
            or not isinstance(span["label_type"], str)
            or (span["meta"] is not None and not isinstance(span["meta"], str))
        ):
            raise ValueError("span_shape")
        start, end = span["start"], span["end"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            raise ValueError("span_offsets")
        if text[start:end] != span["text"]:
            raise ValueError("span_text")
        result.append(
            dict(
                span,
                start_char=start,
                end_char=end,
                start_byte=len(text[:start].encode()),
                end_byte=len(text[:end].encode()),
                text_equal=True,
            )
        )
    return result


def sentence_label(answer: bytes, interval: list[int], spans: list[Json], sensitivity: bool) -> int:
    """Whitespace contact is not a human unsupported character in a sentence."""
    return int(
        any(
            not (sensitivity and span["implicit_true"])
            and any(
                not char.isspace()
                for char in answer[
                    max(interval[0], span["start_byte"]) : min(interval[1], span["end_byte"])
                ].decode()
            )
            for span in spans
            if max(interval[0], span["start_byte"]) < min(interval[1], span["end_byte"])
        )
    )


def join(
    predictors: list[Json],
    boundaries: list[Json],
    evaluators: list[Json],
    roles: list[Json],
    responses: list[Json],
    sources: list[Json],
) -> tuple[list[Json], list[Json]]:
    """Keep original identity and role custody while deriving local targets."""
    frozen, expected = freeze([public_only(row) for row in predictors])
    if frozen != predictors or boundaries != expected:
        raise ValueError("boundary_drift")
    ev = index_unique(evaluators, "family_id")
    role_map = index_unique(roles, "family_id")
    raw = index_unique(responses, "id")
    source_map = index_unique(sources, "source_id")
    ids = [row["family_id"] for row in predictors]
    if set(ev) != set(ids) or set(role_map) != set(ids):
        raise ValueError("roster")
    index_unique(evaluators, "response_id")
    clusters: dict[str, str] = {}
    rows, offsets = [], []
    for predictor, boundary in zip(predictors, boundaries, strict=True):
        fid = predictor["family_id"]
        evaluator, role = ev[fid], role_map[fid]
        response = raw.get(evaluator["response_id"])
        if response is None or response["source_id"] not in source_map:
            raise ValueError("missing_join")
        source = source_map[response["source_id"]]["source_info"]
        serialized = (
            source
            if isinstance(source, str)
            else json.dumps(source, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        )
        original_source = bytes.fromhex(predictor["source_bytes"])
        if serialized.encode() != original_source:
            raise ValueError("source_equality")
        cluster = digest(original_source)
        if cluster != role["source_cluster_id"]:
            raise ValueError("source_cluster_hash")
        if evaluator["role"] != role["role"]:
            raise ValueError("role_drift")
        if cluster in clusters and clusters[cluster] != role["role"]:
            raise ValueError("source_cluster_role")
        clusters[cluster] = role["role"]
        answer = bytes.fromhex(predictor["answer_bytes"])
        spans = checked_spans(response, answer)
        offsets.extend(
            dict(span, family_id=fid, response_id=response["id"], response_sha256=digest(answer))
            for span in spans
        )
        eligible = response["quality"] == "good" and role["status"] == "completed"
        targets = [
            sentence_label(answer, interval, spans, False) for interval in boundary["intervals"]
        ]
        sensitivity = [
            sentence_label(answer, interval, spans, True) for interval in boundary["intervals"]
        ]
        selected = boundary["intervals"].index(predictor["sentence_interval"])
        rows.append(
            dict(
                family_id=fid,
                role=role["role"],
                source_cluster_id=cluster,
                response_id=response["id"],
                source_id=response["source_id"],
                quality=response["quality"],
                model=response.get("model"),
                status="completed" if eligible else "excluded",
                exclusion_reason=None
                if eligible
                else (
                    "public_byte_budget" if role["status"] != "completed" else response["quality"]
                ),
                sentence_interval=predictor["sentence_interval"],
                sentence_labels=targets if eligible else None,
                implicit_true_excluded_sentence_labels=sensitivity if eligible else None,
                y=targets[selected] if eligible else None,
                implicit_true_excluded_y=sensitivity[selected] if eligible else None,
                completely_annotated=response["quality"] == "good",
                annotation_count=len(spans),
                arm="human_sentence_transport",
                seed=68942,
            )
        )
    return rows, offsets


def reduce_rows(rows: list[Json]) -> Json:
    """Count queried evaluation families and distinct original source clusters."""
    index_unique(rows, "family_id")
    evaluation = [row for row in rows if row["role"] == "evaluation"]
    eligible = [row for row in evaluation if row["status"] == "completed"]
    counts = {str(y): sum(row["y"] == y for row in eligible) for y in (0, 1)}
    clusters = len({row["source_cluster_id"] for row in eligible})
    operands = [
        dict(field="intended_evaluation_families", op="==", expected=64, observed=len(evaluation)),
        dict(field="eligible_source_clusters", op=">=", expected=32, observed=clusters),
        *[dict(field="class_" + y, op=">=", expected=8, observed=counts[y]) for y in counts],
    ]
    failed = [
        operand
        for operand in operands
        if (
            operand["observed"] != operand["expected"]
            if operand["op"] == "=="
            else operand["observed"] < operand["expected"]
        )
    ]
    return dict(
        label_counts=counts,
        source_cluster_counts=dict(
            intended=len({row["source_cluster_id"] for row in evaluation}),
            eligible=clusters,
            all_roles=len({row["source_cluster_id"] for row in rows}),
        ),
        sample_size_budget=dict(
            unit="selected_evaluation_sentence",
            intended=64,
            eligible=len(eligible),
            started=len(evaluation),
            completed=len(eligible),
            failed=0,
            censored=0,
            excluded=len(evaluation) - len(eligible),
            independent=clusters,
        ),
        role_counts=dict(Counter(row["role"] for row in rows)),
        sentence_labels_ready_score=int(not failed),
        failed_operands=failed,
        rows_sha256=canonical_hash(rows),
    )
