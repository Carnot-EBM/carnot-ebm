"""REQ-REPORT-8908: compare dependent annotation criteria without asserting natural truth."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
REVISION = "afaec19e64122b4a58718dc31bcd555c97460233"
ORIGINAL_REVISION = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
NAME = "experiment_8908_v762_natural_criterion_audit"
TASK = "exp8908-natural-criterion-audit"
ROOT = Path(__file__).resolve().parents[3]
CLI = "scripts/experiments/" + NAME + ".py"
OWNED = [
    "python/carnot/reporting/natural_criterion_audit_8908.py",
    "python/carnot/reporting/natural_criterion_runner_8908.py",
    CLI,
]
SEED = 7628908
CROSSWALK = {
    "Evident Conflict": "EvidentConflict",
    "Subtle Conflict": "SubtleConflict",
    "Evident Baseless Info": "EvidentBaselessInfo",
    "Subtle Baseless Info": "SubtleBaselessInfo",
    "Subtle Introduction Of Baseless Info": "SubtleBaselessInfo",
    "Incorrect Refusal": "IncorrectRefusal",
}
COLUMNS = {
    "messages": {"stable_id", "message_thread_stable_id", "position", "role", "text", "meta"},
    "spans": {
        "stable_id",
        "message_stable_id",
        "start_pos",
        "end_pos",
        "span_text",
        "label",
        "reasoning",
    },
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush completed work so the supervisor can distinguish progress from a stall."""
    print(f"[exp8908] phase={phase} completed={completed} pending={pending}", flush=True)


def read_csv(path: Path, kind: str) -> list[Json]:
    """The author files have different schemas and cannot share one dataset viewer."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or []) != COLUMNS[kind]:
            raise ValueError("csv_schema:" + kind)
        rows: list[Json] = list(reader)
    return rows


def normalize_spans(spans: list[Json], text: str, *, revised: bool) -> list[Json]:
    """Exact Unicode slices validate annotation locations, never their semantic correctness."""
    result = []
    for span in spans:
        start, end = (
            int(span["start_pos" if revised else "start"]),
            int(span["end_pos" if revised else "end"]),
        )
        label = span["label" if revised else "label_type"]
        marked = span["span_text" if revised else "text"]
        if label not in CROSSWALK:
            raise ValueError("unsupported_label:" + str(label))
        if not 0 <= start < end <= len(text) or text[start:end] != marked:
            raise ValueError("unicode_offset")
        result.append(dict(start=start, end=end, label=CROSSWALK[label]))
    return sorted(result, key=lambda r: (r["start"], r["end"], r["label"]))


def index(rows: list[Json], field: str) -> dict[str, list[Json]]:
    """Retain duplicate identities so a join cannot silently choose one producer record."""
    result: dict[str, list[Json]] = defaultdict(list)
    for row in rows:
        result[str(row[field])].append(row)
    return result


def select_canary(rows: list[Json], authority: bool) -> list[Json]:
    """Select only by source task and stable ID hashes before opening any label metrics."""
    eligible = [r for r in rows if r["status"] == "completed" and authority]
    rank = lambda r: canonical_hash([REVISION, r["unit_id"]])
    selected = []
    for task in ("QA", "Summary"):
        selected.extend(sorted((r for r in eligible if r["source_task"] == task), key=rank)[:12])
    ids = {r["unit_id"] for r in selected}
    selected.extend(
        sorted((r for r in eligible if r["unit_id"] not in ids), key=rank)[: 24 - len(selected)]
    )
    keys = ["unit_id", "cluster_id", "source_task", "prompt_sha256", "response_sha256"]
    return (
        [{k: row[k] for k in keys} for row in sorted(selected, key=rank)]
        if len(selected) == 24
        else []
    )


def audit(
    messages: list[Json],
    spans: list[Json],
    responses: list[Json],
    sources: list[Json],
    *,
    authority: bool,
) -> Json:
    """Keep every intended slot while comparing the same joined response under two criteria."""
    threads = index(messages, "message_thread_stable_id")
    if len(threads) > 408:
        raise ValueError("roster_overflow")
    message_ids, span_ids = (
        Counter(r["stable_id"] for r in messages),
        Counter(r["stable_id"] for r in spans if "stable_id" in r),
    )
    if any(r["message_stable_id"] not in message_ids for r in spans):
        raise ValueError("orphan_span")
    span_by_message, original, source_index = (
        index(spans, "message_stable_id"),
        index(responses, "id"),
        index(sources, "source_id"),
    )
    original_ids = []
    for message in messages:
        if message["role"] == "assistant":
            try:
                original_ids.append(str(json.loads(message["meta"])["original_id"]))
            except (ValueError, KeyError, TypeError):
                pass
    duplicates = Counter(original_ids)
    joins, private, comparisons = [], [], []
    for thread_id, group in sorted(threads.items()):
        row: Json = dict(
            unit_id="release:" + thread_id,
            cluster_id=thread_id,
            arm="paired_criteria",
            seed=SEED,
            metric="paired_annotation_coverage",
            denominator=1,
            status="excluded",
            missing_reason=None,
        )
        try:
            users, assistants = (
                [r for r in group if r["role"] == "user"],
                [r for r in group if r["role"] == "assistant"],
            )
            if (
                len(group) != 2
                or len(users) != 1
                or len(assistants) != 1
                or users[0]["position"] != "0"
                or assistants[0]["position"] != "1"
                or any(message_ids[r["stable_id"]] != 1 for r in group)
            ):
                raise ValueError("ambiguous_thread")
            user, assistant = users[0], assistants[0]
            meta = json.loads(assistant["meta"])
            ident = str(meta["original_id"])
            row["unit_id"] = ident if duplicates[ident] == 1 else "release:" + thread_id
            if duplicates[ident] != 1 or len(original.get(ident, [])) != 1:
                raise ValueError("ambiguous_original")
            response = original[ident][0]
            candidates = source_index.get(str(response["source_id"]), [])
            if len(candidates) != 1:
                raise ValueError("ambiguous_source")
            source = candidates[0]
            if not source.get("source_info") or not user["text"] or not assistant["text"]:
                raise ValueError("missing_context")
            if source["task_type"] not in {"QA", "Summary"} or response["split"] != "test":
                raise ValueError("population_mismatch")
            if source["prompt"] != user["text"] or response["response"] != assistant["text"]:
                raise ValueError("changed_text")
            if str(json.loads(user["meta"])["original_id"]) != ident:
                raise ValueError("metadata_identity")
            if any(
                meta[k] != response[k] or json.loads(user["meta"])[k] != response[k]
                for k in ("model", "temperature", "quality")
            ):
                raise ValueError("metadata_mismatch")
            revised = span_by_message.get(assistant["stable_id"], [])
            if any(span_ids[s["stable_id"]] != 1 for s in revised) or span_by_message.get(
                user["stable_id"]
            ):
                raise ValueError("ambiguous_span")
            before = normalize_spans(response["labels"], response["response"], revised=False)
            after = normalize_spans(revised, assistant["text"], revised=True)
            row.update(
                status="completed",
                cluster_id=str(source["source_id"]),
                source_task=source["task_type"],
                prompt_sha256="sha256:" + hashlib.sha256(user["text"].encode()).hexdigest(),
                response_sha256="sha256:" + hashlib.sha256(assistant["text"].encode()).hexdigest(),
                original_span_count=len(before),
                revised_span_count=len(after),
                original_evidence_failure_count=sum(
                    s["label"] != "IncorrectRefusal" for s in before
                ),
                revised_evidence_failure_count=sum(s["label"] != "IncorrectRefusal" for s in after),
                original_refusal_count=sum(s["label"] == "IncorrectRefusal" for s in before),
                revised_refusal_count=sum(s["label"] == "IncorrectRefusal" for s in after),
            )
            private.append(
                dict(
                    unit_id=ident,
                    source=source,
                    original_response=response,
                    released_user=user,
                    released_assistant=assistant,
                    original_spans=response["labels"],
                    revised_spans=revised,
                    normalized_original=before,
                    normalized_revised=after,
                )
            )
            comparisons.append((row, before, after))
        except (ValueError, KeyError, TypeError) as error:
            row["missing_reason"] = str(error)
        joins.append(row)
    for slot in range(len(joins), 408):
        joins.append(
            dict(
                unit_id=f"missing:{slot}",
                cluster_id=f"missing:{slot}",
                arm="paired_criteria",
                seed=SEED,
                metric="paired_annotation_coverage",
                denominator=1,
                status="missing",
                missing_reason="unreleased_or_unavailable_case",
            )
        )
    canary = select_canary(joins, authority)
    disagreement = []
    for metric in ("evidence_failure_presence", "category_presence", "exact_spans"):
        numerator = 0
        for row, before, after in comparisons:
            left, right = {s["label"] for s in before}, {s["label"] for s in after}
            changed = (
                bool(left - {"IncorrectRefusal"}) != bool(right - {"IncorrectRefusal"})
                if metric == "evidence_failure_presence"
                else left != right
                if metric == "category_presence"
                else before != after
            )
            row[metric + "_disagreement"] = int(changed)
            numerator += int(changed)
        disagreement.append(
            dict(
                unit_id="all_valid_joins",
                cluster_id="paired_cases",
                arm="original_vs_revised",
                seed=SEED,
                metric=metric,
                numerator=numerator,
                denominator=len(comparisons),
                rate=numerator / len(comparisons) if comparisons else None,
                status="completed" if comparisons else "missing",
                missing_reason=None if comparisons else "no_valid_joins",
            )
        )
    return dict(
        rows=joins,
        join_rows=joins,
        disagreement_rows=disagreement,
        private_pairs=private,
        canary_manifest=canary,
        canary_manifest_hash=canonical_hash(canary),
        intended_count=408,
        completed_count=len(comparisons),
        excluded_count=sum(r["status"] == "excluded" for r in joins),
        censored_count=sum(r["status"] == "missing" for r in joins),
        failed_count=0,
        independent_count=0,
        independent_generalization_score=0,
        corpus_audit_ready_score=1,
        natural_canary_ready_score=int(len(canary) == 24),
    )
