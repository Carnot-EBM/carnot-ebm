"""REQ-REPORT-8022: measure a fixed answer without asking a model to produce it.

Each answer token uses the logits from its preceding token. Full-vocabulary
normalization makes this a conditional likelihood, not a top-k confidence proxy.
"""

from __future__ import annotations

import re
import time
import math
from pathlib import Path
from typing import Any

import numpy as np

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify.evidence_features_7980 import normalized
from carnot.verify.source_projection import PUBLIC_KEYS

Json = dict[str, Any]
ROLES = dict(fit=64, tune=32, evaluation=96)
ARMS = ("full", "duplicate", "no_source", "removed_chunk")
QUESTION = b"Write an answer using the supplied source."
METHODS = dict(
    seed=69522,
    input_tokens=6000,
    answer_tokens=384,
    chunk_tokens=256,
    selection="original_role_order_first_admitted_normalized_source_groups",
    removal="unique_casefold_word_intersection_count_then_lowest_chunk_index",
    duplicate_tolerance=1e-6,
    load_preflight_timeout_s=600,
    scoring_child_timeout_s=120,
    heartbeat_s=60,
    scoring_pass_budget=8,
    duration_floor_s=2,
    labels_available=False,
    features=["full_mean_nll", "no_source_minus_full", "removal_minus_full"],
    hypotheses=[
        "H1: paired no-source minus full mean NLL > 0",
        "H2: paired removal minus full mean NLL > 0",
        "H3: independent-label loss improves over full NLL alone",
    ],
    statistical_plan="future source-paired 10000 bootstrap draws; one-sided Holm alpha .05; no hypothesis tested here",
    exposure="all roles retain historical development exposure",
    question_scope="fixed neutral instruction; original generation prompts unavailable",
)


def progress(phase: str, started: float, units: int = 0, pending: int = 0) -> None:
    """Report completed work so an idle child cannot look like ongoing inference."""
    print(
        f"[exp8022] phase={phase} elapsed_s={time.monotonic() - started:.3f} completed={units} pending={pending}",
        flush=True,
    )


def target_likelihood(logits: Any, tokens: list[int], boundary: int) -> Json:
    """Use float64 arithmetic to avoid overflow and expose a wrong token shift."""
    matrix = np.asarray(logits)
    if matrix.ndim != 2 or not 1 <= boundary < len(tokens) or matrix.shape[0] < len(tokens) - 1:
        raise ValueError("conditional_alignment")
    lp, normalizers, error = [], [], 0.0
    for position in range(boundary, len(tokens)):
        values = np.asarray(matrix[position - 1], dtype=np.float64)
        target = tokens[position]
        if not 0 <= target < len(values) or not np.isfinite(values).all():
            raise ValueError("nonfinite_or_invalid_target")
        maximum = float(values.max())
        logz = maximum + float(np.log(np.exp(values - maximum).sum(dtype=np.float64)))
        lp.append(float(values[target] - logz))
        normalizers.append(logz)
        error = max(error, abs(float(np.exp(values - logz).sum(dtype=np.float64)) - 1))
    return dict(
        target_logprobs=lp,
        mean_nll=float(-np.mean(lp, dtype=np.float64)),
        log_normalizers=normalizers,
        vocabulary_size=matrix.shape[1],
        normalization_max_error=error,
        normalization_dtype="float64",
        conditional_logit_positions=list(range(boundary - 1, len(tokens) - 1)),
    )


def prepare(row: Json, runtime: Any) -> Json:
    """Preserve complete bytes; deleting source tokens never changes the target."""
    if set(row) != PUBLIC_KEYS:
        raise ValueError("public_fields")
    source, answer = (bytes.fromhex(row[k]) for k in ("source_bytes", "answer_bytes"))
    source.decode("utf-8")
    answer.decode("utf-8")
    ids = runtime.tokenize(source, add_bos=False, special=False)
    targets = runtime.tokenize(answer, add_bos=False, special=False)
    if not source or not 1 <= len(targets) <= 384:
        raise ValueError("answer_token_budget" if source else "incomplete_context")
    chunks = [ids[i : i + 256] for i in range(0, len(ids), 256)]
    blobs = [runtime.detokenize(c) for c in chunks]
    if b"".join(blobs) != source or runtime.detokenize(targets) != answer:
        raise ValueError("token_byte_roundtrip")
    words = set(re.findall(r"\w+", answer.decode().casefold()))
    overlaps = [
        len(words & set(re.findall(r"\w+", b.decode("utf-8", "replace").casefold()))) for b in blobs
    ]
    chosen = max(range(len(chunks)), key=lambda i: (overlaps[i], -i))
    begin = sum(map(len, blobs[:chosen]))
    end = begin + len(blobs[chosen])
    removed = source[:begin] + source[end:]
    removed.decode("utf-8")
    views = {}
    for arm, blob in zip(ARMS, [source, source, b"", removed], strict=True):
        prompt = runtime.render(blob, QUESTION)
        prefix = runtime.tokenize(prompt, add_bos=True, special=True)
        combined = runtime.tokenize(prompt + answer, add_bos=True, special=True)
        if combined != prefix + targets:
            raise ValueError("prompt_answer_boundary")
        if len(prefix) > 6000:
            raise ValueError("input_token_budget")
        views[arm] = dict(
            source_bytes=blob.hex(),
            prompt_bytes=prompt.hex(),
            tokens=combined,
            response_start=len(prefix),
            input_tokens=len(prefix),
        )
    offsets, start = [], 0
    for i in range(len(targets)):
        stop = len(runtime.detokenize(targets[: i + 1]))
        offsets.append([start, stop])
        start = stop
    return dict(
        row,
        question_bytes=QUESTION.hex(),
        target_tokens=targets,
        response_token_offsets=offsets,
        views=views,
        view_hashes={k: canonical_hash(v) for k, v in views.items()},
        removal_mask=dict(
            selected_chunk_index=chosen,
            token_start=chosen * 256,
            token_end=min(len(ids), (chosen + 1) * 256),
            byte_start=begin,
            byte_end=end,
            chunk_lengths=list(map(len, chunks)),
            lexical_overlaps=overlaps,
        ),
        source_normalized_hash=normalized(source),
    )


def freeze_panel(public: Json, runtime: Any, limits: Json | None = None) -> Json:
    """Select original groups by admission alone before any likelihood or label."""
    limits = ROLES if limits is None else limits
    rows, exclusions, seen, counts = [], [], set(), {}
    started = time.monotonic()
    for role, limit in limits.items():
        counts[role] = 0
        for index, row in enumerate(public[role]):
            if set(row) != PUBLIC_KEYS:
                raise ValueError("public_fields")
            key = normalized(bytes.fromhex(row["source_bytes"]))
            if key in seen:
                raise ValueError("source_role_overlap")
            seen.add(key)
            if counts[role] >= limit:
                continue
            try:
                item = prepare(row, runtime)
            except (ValueError, UnicodeError) as error:
                exclusions.append(
                    dict(
                        family_id=row["family_id"],
                        role=role,
                        original_index=index,
                        source_normalized_hash=key,
                        reason=str(error),
                    )
                )
                continue
            rows.append(dict(item, role=role, original_index=index, exposure=METHODS["exposure"]))
            counts[role] += 1
            progress("panel_admission", started, len(rows), sum(limits.values()) - len(rows))
    return dict(
        rows=rows,
        exclusions=exclusions,
        counts=counts,
        intended=dict(limits),
        complete=counts == limits,
        labels_opened=False,
        methods=METHODS,
        slots=[
            dict(
                id=r["family_id"] + ":" + arm,
                family_id=r["family_id"],
                role=r["role"],
                arm=arm,
                seed=69522,
                mean_nll=None,
                numerator=None,
                denominator=len(r["target_tokens"]),
                status="frozen_unscored",
                scoring_scheduled_here=False,
                exclusion_reason=None,
                failure_reason=None,
                censor_reason=None,
            )
            for r in rows
            for arm in ARMS
        ],
    )


def qualify(runtime: Any, checkpoints: Path | None = None) -> Json:
    """Eight fixed-answer forwards qualify plumbing, without truth labels."""
    started = time.monotonic()
    tiny = target_likelihood([[0.0, math.log(2.0), math.log(3.0)], [0.0, 0.0, 0.0]], [0, 2], 1)
    expected = math.log(3.0 / 6.0)
    if abs(tiny["target_logprobs"][0] - expected) > 1e-12:
        raise ValueError("independent_normalization_fixture")
    rows, differences, scored = [], [], 0
    for i, (source, answer) in enumerate(
        [(b"The seal is blue.", b"Blue."), (b"The box holds seven coins.", b"Seven coins.")]
    ):
        item = prepare(
            dict(family_id=f"private-{i}", source_bytes=source.hex(), answer_bytes=answer.hex()),
            runtime,
        )
        results = {}
        for arm, view in item["views"].items():
            progress("before_teacher_forcing", started, len(rows), 8 - len(rows))
            path = checkpoints / f"pass-{len(rows):02d}.json" if checkpoints is not None else None
            attempt = dict(
                id=f"private-{i}:{arm}",
                status="running",
                generated_tokens=0,
                started_monotonic_ns=time.monotonic_ns(),
                view=view,
            )
            if path is not None:
                atomic_json(path, attempt)
            try:
                runtime.reset()
                runtime.eval(view["tokens"])
                if runtime.n_tokens != len(view["tokens"]):
                    raise ValueError("runtime_token_alignment")
                result = target_likelihood(runtime.scores, view["tokens"], view["response_start"])
            except (OSError, RuntimeError, TimeoutError, ValueError) as error:
                if path is not None:
                    atomic_json(
                        path,
                        dict(
                            attempt,
                            status="failed",
                            error=str(error),
                            ended_monotonic_ns=time.monotonic_ns(),
                        ),
                    )
                progress("after_teacher_forcing_failed", started, len(rows), 8 - len(rows))
                raise
            scored += len(result["target_logprobs"])
            results[arm] = result["mean_nll"]
            rows.append(
                dict(
                    id=f"private-{i}:{arm}",
                    fixture_id=i,
                    arm=arm,
                    metric=result["mean_nll"],
                    numerator=sum(-x for x in result["target_logprobs"]),
                    denominator=len(result["target_logprobs"]),
                    generated_tokens=0,
                    status="completed",
                    **result,
                    view=view,
                    response_token_offsets=item["response_token_offsets"],
                )
            )
            if path is not None:
                atomic_json(
                    path,
                    dict(
                        rows[-1],
                        started_monotonic_ns=attempt["started_monotonic_ns"],
                        ended_monotonic_ns=time.monotonic_ns(),
                    ),
                )
            progress("after_teacher_forcing", started, len(rows), 8 - len(rows))
        differences.append(abs(results["duplicate"] - results["full"]))
    return dict(
        rows=rows,
        forward_pass_counts=len(rows),
        scored_tokens=scored,
        generated_tokens=0,
        duplicate_max_difference=max(differences),
        passed=max(differences) <= 1e-6 and all(r["normalization_max_error"] < 1e-9 for r in rows),
        duration_s=time.monotonic() - started,
        independent_logits_fixture=dict(
            expected=expected, observed=tiny["target_logprobs"][0], passed=True
        ),
    )
