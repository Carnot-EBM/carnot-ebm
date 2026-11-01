"""REQ-REPORT-8033: isolate context lifetime and evaluation shape on fixed tokens.

Repeated scores test numerical plumbing. They do not test factual correctness.
Each completed call owns scalar probabilities before native buffers can change.
"""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

from carnot.inference import fixed_answer_likelihood_8022 as base
from carnot.inference.qwen_sufficiency_7920 import bounded
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]
CONDITIONS = ("reset_full", "fresh_full", "reset_chunks", "fresh_chunks")
PREFERENCE = ("fresh_full", "fresh_chunks", "reset_chunks")
METHODS = dict(
    seed=69633,
    sources=8,
    target_passes=128,
    conditioning_passes=4,
    scored_tokens=80000,
    forward_seconds=120,
    model_work_seconds=900,
    n_batch=256,
    n_ubatch=256,
    chunk_tokens=128,
    duplicate_tolerance=1e-6,
    normalization_tolerance=1e-10,
    labels_available=False,
    preference=list(PREFERENCE),
    selection="eight_evenly_spaced_fit_token_length_ranks",
    context_seconds=120,
    diagnostic_floor_s=2,
    memory_reserve_mb=2048,
    padding=False,
)
likelihood = base.target_likelihood
progress = base.progress


def oracle() -> Json:
    """A small exact distribution catches normalization and preceding-token errors."""
    value = base.target_likelihood(
        [[0.0, math.log(2.0), math.log(3.0)], [math.log(3.0), 0.0, math.log(2.0)], [0.0, 0.0, 0.0]],
        [0, 2, 0],
        1,
    )
    expected = [math.log(0.5), math.log(0.5)]
    if any(abs(a - b) > 1e-12 for a, b in zip(value["target_logprobs"], expected, strict=True)):
        raise ValueError("independent_logit_oracle")
    return dict(
        expected=expected,
        observed=value["target_logprobs"],
        passed=True,
        independent_count=0,
        verifier_is_oracle=True,
    )


def select(rows: list[Json]) -> list[Json]:
    """Length ranks cover the existing range without consulting outcomes."""
    fit = sorted(
        (r for r in rows if r["role"] == "fit"),
        key=lambda r: (len(r["views"]["full"]["tokens"]), r["family_id"]),
    )
    if len(fit) < 8 or len({r["source_normalized_hash"] for r in fit}) != len(fit):
        raise ValueError("eight_distinct_fit_sources_required")
    return [fit[(i * (len(fit) - 1)) // 7] for i in range(8)]


def schedule(panel: list[Json]) -> list[Json]:
    """A separate conditioning call makes the first target follow another source."""
    slots = []
    if not panel:
        return slots
    for condition in CONDITIONS:
        previous = panel[-1]["family_id"]
        sequence = [(panel[-1], -1, "conditioning")]
        sequence += [
            (item, repeat, "target") for repeat in range(2) for item in panel for _ in range(2)
        ]
        for item, repeat, purpose in sequence:
            slots.append(
                dict(
                    id=f"pass-{len(slots):03d}",
                    family_id=item["family_id"],
                    condition=condition,
                    repeat=repeat,
                    purpose=purpose,
                    preceded_by="self" if previous == item["family_id"] else "different",
                    previous_source_id=previous,
                    generated_tokens=0,
                )
            )
            previous = item["family_id"]
    return slots


class NativeController:
    """Reuse weights while closing each temporary native context after its score."""

    def __init__(self, runtime: Any, deadline: float) -> None:
        from llama_cpp import _internals, llama_cpp

        self.model, self.deadline = runtime.model, deadline
        self.original = self.model._ctx
        self.factory, self.native = _internals.LlamaContext, llama_cpp
        self.serial = 0

    def score(self, view: Json, condition: str) -> Json:
        """Observe native memory operations and decode batches, then copy scalars."""
        model, native = self.model, self.native
        fresh = condition.startswith("fresh")
        started = time.monotonic()
        self.serial += 1
        ctx = self.original
        events: list[Json] = []
        shapes: list[Json] = []

        def bound(call: Any) -> Any:
            remaining = min(self.deadline, started + METHODS["forward_seconds"]) - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("total_model_work_cap")
            return bounded(call, min(120, remaining))

        def positions() -> Json:
            return dict(
                min=int(native.llama_memory_seq_pos_min(model._ctx.memory, 0)),
                max=int(native.llama_memory_seq_pos_max(model._ctx.memory, 0)),
            )

        try:
            if fresh:
                progress("8033_before_context_create", started)
                ctx = bound(
                    lambda: self.factory(
                        model=model._model, params=model.context_params, verbose=False
                    )
                )
                model._ctx = ctx
                progress("8033_after_context_create", started)
            remove, decode, clear = ctx.kv_cache_seq_rm, ctx.decode, native.llama_memory_clear

            def rm(seq: int, start: int, stop: int) -> Any:
                before = positions()
                result = remove(seq, start, stop)
                events.append(
                    dict(
                        operation="seq_rm",
                        args=[seq, start, stop],
                        returned=result,
                        before=before,
                        after=positions(),
                    )
                )
                return result

            def wipe(memory: Any, data: bool) -> None:
                before = positions()
                clear(memory, data)
                events.append(
                    dict(operation="memory_clear", data=data, before=before, after=positions())
                )

            def step(batch: Any) -> None:
                shapes.append(dict(n_tokens=int(batch.batch.n_tokens), n_past=int(model.n_tokens)))
                decode(batch)

            events.append(dict(operation="before_reset", positions=positions()))
            with (
                patch.object(ctx, "kv_cache_seq_rm", rm),
                patch.object(ctx, "decode", step),
                patch.object(native, "llama_memory_clear", wipe),
            ):
                model.reset()
                events.append(dict(operation="after_reset", positions=positions()))
                tokens = view["tokens"]
                chunk = 128 if condition.endswith("chunks") else len(tokens)
                for start in range(0, len(tokens), chunk):
                    progress("8033_before_eval", started, start, len(tokens) - start)
                    bound(lambda start=start: model.eval(tokens[start : start + chunk]))
                    progress(
                        "8033_after_eval", started, model.n_tokens, len(tokens) - model.n_tokens
                    )
                if model.n_tokens != len(tokens):
                    raise ValueError("runtime_token_alignment")
                result = likelihood(model.scores, tokens, view["response_start"])
            return dict(
                result,
                context_identity=f"{'fresh-' + str(self.serial) if fresh else 'existing'}",
                native_context_address=int(ctx.ctx),
                kv_lifecycle=events,
                actual_shapes=shapes,
                context_closed=fresh,
                eval_chunk_tokens=chunk,
                n_batch=model.n_batch,
                n_ubatch=int(model.context_params.n_ubatch),
            )
        finally:
            if fresh and ctx is not self.original:
                progress("8033_before_context_close", started)
                ctx.close()
                model._ctx = self.original
                progress("8033_after_context_close", started)


def capture(panel: list[Json], controller: Any, raw: Path, *, deadline: float) -> list[Json]:
    """Durable starts and nonstarts prevent timeout failures from shrinking counts."""
    started = time.monotonic()
    oracle()
    items = {r["family_id"]: r for r in panel}
    rows: list[Json] = []
    stopped, scored = False, 0
    for slot in schedule(panel):
        item = items[slot["family_id"]]
        row = dict(
            slot,
            status="censored",
            numerator=None,
            mean_nll=None,
            denominator=len(item["target_tokens"]),
            token_rows=[],
            token_view_hash=canonical_hash(item["views"]["full"]),
            failure_reason=None,
            exclusion_reason=None,
            censor_reason="model_work_cap_or_failure",
        )
        path = raw / (row["id"] + ".json")
        if not stopped and time.monotonic() < deadline and scored + row["denominator"] <= 80000:
            row.update(
                status="running", censor_reason=None, started_monotonic_ns=time.monotonic_ns()
            )
            atomic_json(path, row)
            progress("8033_before_teacher_forcing", started, len(rows), 132 - len(rows))
            try:
                result = controller.score(item["views"]["full"], row["condition"])
                row.update(result)
                row["token_rows"] = [
                    dict(
                        token_id=t,
                        offset=o,
                        log_probability=float(p),
                        probability=math.exp(float(p)),
                        logit_position=position,
                    )
                    for t, o, p, position in zip(
                        item["target_tokens"],
                        item["response_token_offsets"],
                        result["target_logprobs"],
                        result["conditional_logit_positions"],
                        strict=True,
                    )
                ]
                row["numerator"] = math.fsum(-r["log_probability"] for r in row["token_rows"])
                row["mean_nll"] = row["numerator"] / row["denominator"]
                row["status"] = "completed"
                scored += row["denominator"]
            except (OSError, RuntimeError, TimeoutError, ValueError) as error:
                row.update(status="failed", failure_reason=f"{type(error).__name__}:{error}")
                stopped = True
            row["ended_monotonic_ns"] = time.monotonic_ns()
            progress("8033_after_teacher_forcing", started, len(rows) + 1, 131 - len(rows))
        if row["status"] != "completed":
            row.pop("mean_nll", None)
        atomic_json(path, row)
        rows.append(row)
    return rows


def reduce(panel: list[Json], rows: list[Json]) -> Json:
    """Recompute duplicate ranges from scalar token rows and the exact slot roster."""
    slots = schedule(panel)
    if len(rows) != len(slots) or any(
        any(row[k] != slot[k] for k in slot) for row, slot in zip(rows, slots, strict=True)
    ):
        raise ValueError("slot_roster")
    items = {r["family_id"]: r for r in panel}
    drift, conditions, cross = [], [], []
    scored = 0
    for row in rows:
        item = items[row["family_id"]]
        if row["token_view_hash"] != canonical_hash(item["views"]["full"]):
            raise ValueError("token_alignment_view")
        if row["status"] != "completed":
            continue
        tokens = row["token_rows"]
        view = item["views"]["full"]
        if (
            [r["token_id"] for r in tokens] != item["target_tokens"]
            or [r["offset"] for r in tokens] != item["response_token_offsets"]
            or [r["logit_position"] for r in tokens]
            != list(range(view["response_start"] - 1, len(view["tokens"]) - 1))
        ):
            raise ValueError("token_alignment")
        numerator = math.fsum(-r["log_probability"] for r in tokens)
        if (
            not tokens
            or row["denominator"] != len(tokens)
            or any(
                not math.isfinite(r["log_probability"])
                or r["log_probability"] > 0
                or abs(r["probability"] - math.exp(r["log_probability"])) > 1e-12
                for r in tokens
            )
            or abs(row["numerator"] - numerator) > 1e-10
            or abs(row["mean_nll"] - numerator / len(tokens)) > 1e-10
        ):
            raise ValueError("token_aggregate")
        scored += len(tokens)
    for condition in CONDITIONS:
        owned = [r for r in rows if r["condition"] == condition]
        passed = len(owned) == 33 and all(
            r["status"] == "completed" and r["normalization_max_error"] <= 1e-10 for r in owned
        )
        for item in panel:
            group = [
                r
                for r in owned
                if r["family_id"] == item["family_id"]
                and r["purpose"] == "target"
                and r["status"] == "completed"
            ]
            difference = (
                max(r["mean_nll"] for r in group) - min(r["mean_nll"] for r in group)
                if group
                else None
            )
            valid = len(group) == 4 and difference is not None and difference <= 1e-6
            drift.append(
                dict(
                    condition=condition,
                    family_id=item["family_id"],
                    drift=difference,
                    numerator=difference,
                    denominator=1,
                    captures=len(group),
                    independent_count=1 if len(group) == 4 else 0,
                    passed=valid,
                    mean_nlls=[r["mean_nll"] for r in group],
                )
            )
            passed = passed and valid
        conditions.append(
            dict(
                condition=condition,
                passed=passed,
                intended_count=32,
                completed_count=sum(
                    r["status"] == "completed" and r["purpose"] == "target" for r in owned
                ),
                max_duplicate_drift=max(
                    (
                        d["drift"]
                        for d in drift
                        if d["condition"] == condition and d["drift"] is not None
                    ),
                    default=None,
                ),
            )
        )
    for item in panel:
        means = {}
        for c in CONDITIONS:
            values = [
                r["mean_nll"]
                for r in rows
                if r["condition"] == c
                and r["family_id"] == item["family_id"]
                and r["purpose"] == "target"
                and r["status"] == "completed"
            ]
            means[c] = math.fsum(values) / 4 if len(values) == 4 else None
        cross.append(
            dict(
                family_id=item["family_id"],
                condition_means=means,
                comparison_scope="descriptive_only_not_duplicate_acceptance",
            )
        )
    selected = next(
        (c for c in PREFERENCE if any(r["condition"] == c and r["passed"] for r in conditions)),
        None,
    )
    complete = len(rows) == 132 and all(r["status"] == "completed" for r in rows)
    return dict(
        condition_rows=conditions,
        duplicate_drift_rows=drift,
        selected_condition=selected,
        cross_condition_rows=cross,
        scored_tokens=scored,
        complete=complete,
        forward_pass_counts=sum(r["status"] in {"completed", "failed"} for r in rows),
        token_alignment_checks=dict(passed=complete, complete_answer_coverage=complete),
        causal_attribution="unresolved; associations between context lifetime and shape do not prove a cache cause",
        stale_buffer_policy="probabilities copied by value before any subsequent call",
    )
