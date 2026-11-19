"""REQ-REPORT-8988-CAUSAL: durable delayed updates of fixed-basis coefficients."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.special import expit

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]
ARMS = (
    "frozen_spline",
    "adaptive_spline",
    "bias_only_spline",
    "adaptive_linear",
    "permuted_spline",
)
PERMUTATION = np.random.default_rng(768088).permutation(4).tolist()


def probability(theta: Any, phi: Any) -> float:
    """Average three probabilities so encoder seeds never create independent samples."""
    return float(np.mean(expit(np.sum(np.asarray(theta) * phi, axis=-1))))


def step(
    theta: Any, accumulator: Any, initial: Any, phi: Any, label: int, bias: bool
) -> tuple[Any, Any, Json]:
    """Differentiate one ensemble binary loss, anchor, clip, then apply AdaGrad."""
    t, acc, start, x = (np.asarray(v, dtype=float) for v in (theta, accumulator, initial, phi))
    if (
        t.ndim != 2
        or t.shape[0] != 3
        or any(v.shape != t.shape for v in (acc, start, x))
        or type(label) is not int
        or label not in (0, 1)
        or np.any(acc < 0)
        or not all(np.isfinite(v).all() for v in (t, acc, start, x))
    ):
        raise ValueError("update_operands")
    seed_p = expit(np.sum(t * x, axis=1))
    p = float(seed_p.mean())
    # Log-sigmoid ratios keep the ensemble derivative finite at saturated logits.
    z = np.sum(t * x, axis=1)
    log_p = -np.logaddexp(0, -z)
    log_q = -np.logaddexp(0, z)
    from scipy.special import logsumexp

    ratio = (
        -np.exp(log_p + log_q - logsumexp(log_p))
        if label
        else np.exp(log_p + log_q - logsumexp(log_q))
    )
    gradient = ratio[:, None] * x
    anchor = 0.01 * (t - start)
    anchor[:, 0] = 0
    gradient += anchor
    if bias:
        gradient[:, 1:] = 0
    norm = float(np.linalg.norm(gradient))
    gradient /= max(1.0, norm)
    next_acc = acc + gradient * gradient
    candidate = t - 0.05 * gradient / np.sqrt(next_acc + 1e-8)
    return (
        candidate,
        next_acc,
        dict(
            gradient=gradient.tolist(),
            unclipped_norm=norm,
            clipped_norm=float(np.linalg.norm(gradient)),
            p_error=p,
        ),
    )


def guard(candidate: Any, initial: Any, design: Any, labels: Any) -> Json:
    """Every proposal uses all64 original guard graphs and its arm's initial baseline."""
    phi, y = np.asarray(design), np.asarray(labels)
    if phi.shape[:2] != (3, 64) or y.shape != (64,) or not np.isin(y, [0, 1]).all():
        raise ValueError("guard_roster")
    base = expit(np.einsum("snd,sd->sn", phi, np.asarray(initial))).mean(axis=0)
    proposed = expit(np.einsum("snd,sd->sn", phi, np.asarray(candidate))).mean(axis=0)
    brier = float(np.mean((proposed - y) ** 2 - (base - y) ** 2))
    error = float(np.mean(((proposed >= 0.5) != y).astype(int) - ((base >= 0.5) != y).astype(int)))
    return dict(brier_increase=brier, error_increase=error, passed=brier <= 0.005 and error <= 0)


def feedback(state: Json, request: Json, ordinals: list[int], labels: list[int]) -> bool:
    """A consumed block is acknowledged even when its candidate is rolled back."""
    if not ordinals or all(i in state["acknowledgments"] for i in ordinals):
        return False
    q = state["query"]
    if (
        ordinals != list(range(q - 7, q - 3))
        or len(labels) != 4
        or any(i in state["acknowledgments"] for i in ordinals)
    ):
        raise ValueError("feedback_budget")
    for arm in ARMS[1:]:
        kind = "linear" if arm == "adaptive_linear" else "spline"
        before, acc_before = state["coefficients"][arm], state["accumulator"][arm]
        candidate, acc = np.asarray(before), np.asarray(acc_before)
        ys = [labels[i] for i in PERMUTATION] if arm == "permuted_spline" else labels
        norms = []
        for ordinal, y in zip(ordinals, ys, strict=True):
            candidate, acc, info = step(
                candidate,
                acc,
                request["initial"][arm],
                np.asarray(request["online_design"][kind])[:, ordinal - 1],
                y,
                arm == "bias_only_spline",
            )
            norms.append({k: info[k] for k in ("unclipped_norm", "clipped_norm")})
        checked = guard(
            candidate,
            request["initial"][arm],
            request["guard_design"][kind],
            request["guard_labels"],
        )
        if checked["passed"]:
            state["coefficients"][arm], state["accumulator"][arm] = candidate.tolist(), acc.tolist()
        state["promotion_rows"].append(
            dict(
                arm=arm,
                query=q,
                ordinals=ordinals,
                labels=ys,
                before_hash=canonical_hash([before, acc_before]),
                candidate_hash=canonical_hash([candidate.tolist(), acc.tolist()]),
                after_hash=canonical_hash([state["coefficients"][arm], state["accumulator"][arm]]),
                gradient_norms=norms,
                **checked,
            )
        )
    state["acknowledgments"].extend(ordinals)
    state["feedback_rows"].append(
        dict(query=q, ordinals=ordinals, labels=labels, prediction_hash=state["prediction_hash"])
    )
    return True


def release(query: int, seal: Path, label_path: Path, role: str = "online") -> list[int]:
    """Read truth only after all five predictions and the pending queue are durable."""
    if role != "online":
        raise PermissionError("retention_label_access_refused")
    if query not in range(8, 33, 4):
        return []
    state = json.loads(seal.read_text())
    if (
        state["query"] != query
        or len(state["rows"]) != query * 5
        or state["prediction_hash"] != canonical_hash(state["rows"])
        or not set(range(query - 7, query - 3)) <= set(state["pending"])
    ):
        raise ValueError("prediction_seal")
    values = json.loads(label_path.read_text())["labels"]
    if len(values) != 32 or any(type(y) is not int or y not in (0, 1) for y in values):
        raise ValueError("online_label_schema")
    return list(values[query - 8 : query - 4])


def trajectory(request: Json, raw: Path, *, resume: bool = False, stop: int = 32) -> Json:
    """Atomic commits retain coefficients, accumulator, predictions, guard and consumed IDs."""
    if (
        set(request.get("initial", {})) != set(ARMS)
        or stop not in (16, 32)
        or any(
            np.asarray(request["online_design"][k]).shape[:2] != (3, 32)
            for k in ("spline", "linear")
        )
    ):
        raise ValueError("stream_roster")
    journal = raw / "state.json"
    if resume:
        state = json.loads(journal.read_text())
        if (
            state["query"] != 16
            or state["request_hash"] != canonical_hash(request)
            or state["prediction_hash"] != canonical_hash(state["rows"])
        ):
            raise ValueError("resume_state")
    else:
        state = dict(
            query=0,
            request_hash=canonical_hash(request),
            coefficients=json.loads(json.dumps(request["initial"])),
            accumulator={arm: np.zeros_like(request["initial"][arm]).tolist() for arm in ARMS},
            rows=[],
            pending=[],
            acknowledgments=[],
            promotion_rows=[],
            feedback_rows=[],
        )
    for query in range(state["query"] + 1, stop + 1):
        for arm in ARMS:
            kind = "linear" if arm == "adaptive_linear" else "spline"
            p = probability(
                state["coefficients"][arm], np.asarray(request["online_design"][kind])[:, query - 1]
            )
            state["rows"].append(
                dict(
                    unit_id=f"private-{query:02d}",
                    group_id="private-stream",
                    arm=arm,
                    query=query,
                    condition="private_mechanics",
                    metric="p_error",
                    value=p,
                    numerator=p,
                    denominator=1,
                    status="completed",
                    exclusion_reason=None,
                )
            )
        state.update(
            query=query,
            pending=[i for i in range(1, query + 1) if i not in state["acknowledgments"]],
        )
        state["prediction_hash"] = canonical_hash(state["rows"])
        seal = raw / "seals" / f"{query:02d}.json"
        atomic_json(seal, state)
        labels = release(query, seal, Path(request["label_path"]))
        if labels:
            feedback(state, request, list(range(query - 7, query - 3)), labels)
            state["pending"] = [i for i in state["pending"] if i not in state["acknowledgments"]]
        atomic_json(journal, state)
    return dict(state)


def main(argv: list[str] | None = None) -> int:
    """The parent may kill only this child after its durable query16 checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--stop16", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    print("[exp8988-child] before_trajectory completed=0 pending=32", flush=True)
    state = trajectory(
        json.loads(args.request.read_text()),
        args.raw,
        resume=args.resume,
        stop=16 if args.stop16 else 32,
    )
    print(
        f"[exp8988-child] after_trajectory completed={state['query']} pending={32 - state['query']}",
        flush=True,
    )
    if args.stop16:
        print("READY16", flush=True)
        sys.stdin.readline()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
