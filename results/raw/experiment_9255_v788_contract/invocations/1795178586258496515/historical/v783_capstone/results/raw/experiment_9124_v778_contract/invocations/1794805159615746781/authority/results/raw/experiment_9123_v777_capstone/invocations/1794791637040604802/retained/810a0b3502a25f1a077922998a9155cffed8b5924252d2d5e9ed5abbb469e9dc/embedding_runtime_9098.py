"""REQ-REPORT-9098-V776: fixed text controls establish capability, never scientific benefit."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np

from carnot.reporting import human_methods_9084 as methods
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.natural_label_eligibility_8975 import ref as ref
from carnot.reporting.spline_basis_methods_8986 import gate as gate

Json = dict[str, Any]
ROOT = methods.ROOT
NAME, TASK = "experiment_9098_v776_embedding_runtime", "exp9098-embedding-runtime"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_embedding_runtime_9098.py"
OWNED = [
    "python/carnot/reporting/embedding_runtime_9098.py",
    "python/carnot/inference/embedding_runtime_9098.py",
    "python/carnot/reporting/embedding_runtime_runner_9098.py",
    CLI,
]
MODEL_SPECS = [dict(hf_id="unsloth/Qwen3.8-27B-GGUF", quantization="Q4_K_M")]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real phase boundaries so the owner can supervise native calls."""
    print(f"[exp9098] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Freeze text bytes and tolerances without reading human labels or runtime outcomes."""
    texts = [
        "",
        "",
        "A short fixture.",
        "Another short fixture.",
        "雪と海。 café naïve 🧪",
        "Καλημέρα κόσμε — مرحبا",
        " echo" * 32,
        " echo" * 64,
        " x" * 256,
        " x" * 768,
        " x" * 257,
        " x" * 769,
    ]
    return dict(
        fixtures=[
            dict(
                unit_id=f"fixture-{i:02d}",
                text=text,
                column="question" if i % 2 == 0 else "answer_text",
                cap=256 if i % 2 == 0 else 768,
                empty=i < 2,
                exact_count=[256, 768, 257, 769][i - 8] if i >= 8 else None,
            )
            for i, text in enumerate(texts)
        ],
        pooling_contract=methods.protocol()["embedding"],
        atol=1e-5,
        rtol=1e-4,
        logits=dict(
            prompt="Text:\n{text}\nDecision:",
            continuations=["Yes", "No"],
            fresh_state_each=True,
            unmasked=True,
            reduction="sum then logsumexp",
        ),
        padding_tokens=7,
        measurement_cap_s=3300,
        validation_reserve_s=1200,
        hard_cap_s=4800,
        generated_tokens=0,
        random_seed=7769098,
        shard_response_counts=dict(A=450, B=450),
    )


def pool(vectors: Any, mask: list[bool], width: int, *, empty: bool = False) -> Any:
    """Average only real last-layer tokens, then normalize this column independently."""
    if not mask:
        if not empty:
            raise ValueError("undeclared_empty")
        return np.zeros(width)
    array = np.asarray(vectors, dtype=np.float64)
    if array.shape != (len(mask), width):
        raise ValueError("hidden_width")
    if not np.isfinite(array).all():
        raise ValueError("nonfinite_hidden")
    mean = array[np.asarray(mask)].mean(axis=0)
    norm = np.linalg.norm(mean)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("zero_vector_or_nonfinite")
    return mean / norm


def exercise(runtime: Any, *, deadline: float = float("inf")) -> list[Json]:
    """Retain every declared fixture even when one capability fails independently."""
    rows = []
    for index, fixture in enumerate(protocol()["fixtures"]):
        if time.monotonic() >= deadline:
            raise TimeoutError("measurement_deadline")
        progress("fixture_before", index, 12 - index)
        row = dict(
            fixture,
            original_qid=None,
            qid=None,
            group_id=fixture["unit_id"],
            arm="frozen_runtime",
            condition=fixture["column"],
            seed=7769098,
            embedding_passed=False,
            logits_passed=False,
            vector=[],
            hidden_state_references=[],
            parity_errors=[],
            embedding_error=None,
            logits_error=None,
            exclusion_reason=None,
            censor_reason=None,
            raw_hidden_width=runtime.width,
        )
        tokens = runtime.tokenize(fixture["text"])
        retained = tokens[: fixture["cap"]]
        row.update(
            full_token_count=len(tokens),
            retained_token_count=len(retained),
            truncated=len(tokens) > len(retained),
            padding_mask=[True] * len(retained),
        )
        try:
            if fixture["exact_count"] is not None and len(tokens) != fixture["exact_count"]:
                raise ValueError("exact_token_count")
            began = time.monotonic()
            hidden = runtime.hidden([retained])[0] if retained else []
            if retained and getattr(runtime, "last_hidden_references", []):
                row["hidden_state_references"].append(
                    dict(
                        condition="serial",
                        reference=runtime.last_hidden_references[0],
                        padding_mask=[True] * len(retained),
                    )
                )
            row["serial_duration_s"] = time.monotonic() - began
            vector = pool(hidden, [True] * len(retained), runtime.width, empty=fixture["empty"])
            if retained:
                # An unrelated sequence exposes sequence-ID and shared-buffer mistakes.
                batch = runtime.hidden([retained, [retained[0]]])[0]
                batch_refs = list(getattr(runtime, "last_hidden_references", []))
                padded = runtime.hidden([retained + [runtime.pad_id] * 7])[0]
                padded_refs = list(getattr(runtime, "last_hidden_references", []))
                for name, other, mask in [
                    ("batch", batch, [True] * len(retained)),
                    ("padding", padded, [True] * len(retained) + [False] * 7),
                ]:
                    candidate = pool(other, mask, runtime.width)
                    references = batch_refs if name == "batch" else padded_refs
                    if references:
                        row["hidden_state_references"].append(
                            dict(condition=name, reference=references[0], padding_mask=mask)
                        )
                    error = float(np.max(np.abs(vector - candidate)))
                    row["parity_errors"].append(dict(condition=name, max_abs_error=error))
                    if not np.allclose(vector, candidate, atol=1e-5, rtol=1e-4):
                        raise ValueError(name + "_parity")
            row.update(vector=vector.tolist(), embedding_passed=True)
        except (ValueError, RuntimeError) as error:
            row["embedding_error"] = str(error)
        try:
            prompt = protocol()["logits"]["prompt"].format(text=fixture["text"])
            checks = {word: runtime.likelihood(prompt, word) for word in ["Yes", "No"]}
            logs = np.array([checks[w]["log_probability"] for w in ["Yes", "No"]])
            if not np.isfinite(logs).all() or not all(
                c["raw_logits_finite"]
                and c["raw_logits_unmasked"]
                and len(c["tokens"]) == len(c["token_logprobs"]) > 0
                and c["log_probability"] == sum(c["token_logprobs"])
                for c in checks.values()
            ):
                raise ValueError("nonfinite_or_masked_logits")
            probabilities = np.exp(logs - np.logaddexp.reduce(logs))
            row.update(
                raw_logit_checks=checks,
                yes_probability=float(probabilities[0]),
                no_probability=float(probabilities[1]),
                logits_passed=True,
            )
        except (ValueError, RuntimeError) as error:
            row["logits_error"] = str(error)
        row.update(
            status="completed",
            absolute_metric=int(row["embedding_passed"]),
            numerator=int(row["embedding_passed"]),
            denominator=1,
        )
        rows.append(row)
        progress("fixture_after", index + 1, 11 - index)
    return rows


def reduce(rows: list[Json], cold: list[Json], *, runtime_ok: bool) -> Json:
    """Recompute readiness and shard estimates from retained primitive measurements."""
    errors: list[Json] = []
    roster = protocol()["fixtures"]
    complete = len(rows) == len(cold) == len(roster)
    for fixture, row, repeated in zip(roster, rows, cold):
        same = row["unit_id"] == repeated["unit_id"] == fixture["unit_id"]
        a, b = np.asarray(row["vector"]), np.asarray(repeated["vector"])
        parity = bool(
            a.shape == b.shape
            and a.size
            and np.isfinite(a).all()
            and np.isfinite(b).all()
            and np.allclose(a, b, atol=1e-5, rtol=1e-4)
        )
        if not same or not parity:
            errors.append(dict(unit_id=fixture["unit_id"], condition="cold_process"))
    embeddings = int(
        runtime_ok and complete and not errors and all(r["embedding_passed"] for r in rows + cold)
    )
    logits = int(runtime_ok and len(rows) == 12 and all(r["logits_passed"] for r in rows))
    times = {
        str(r["cap"]): r.get("serial_duration_s")
        for r in rows
        if r.get("retained_token_count") == r["cap"] and r["embedding_passed"]
    }
    estimate = 450 * (times["256"] + times["768"]) if set(times) == {"256", "768"} else None
    return dict(
        embeddings_ready_score=embeddings,
        logits_ready_score=logits,
        parity_errors=errors
        + [dict(unit_id=r["unit_id"], **p) for r in rows for p in r.get("parity_errors", [])],
        duration_projection=dict(
            shard_a_s=estimate,
            shard_b_s=estimate,
            measured_fixture_times_s=times,
            limit_s=3300,
            feasible=None if estimate is None else estimate <= 3300,
            scope="450 independent question/answer forwards per shard; cold load separate",
        ),
    )


def censored(reason: str) -> list[Json]:
    """An unavailable runtime cannot erase any of the twelve scheduled texts."""
    return [
        dict(
            f,
            original_qid=None,
            qid=None,
            group_id=f["unit_id"],
            seed=7769098,
            arm="frozen_runtime",
            condition=f["column"],
            status="censored",
            embedding_passed=False,
            logits_passed=False,
            vector=[],
            absolute_metric=None,
            numerator=0,
            denominator=1,
            exclusion_reason=None,
            censor_reason=reason,
        )
        for f in protocol()["fixtures"]
    ]
