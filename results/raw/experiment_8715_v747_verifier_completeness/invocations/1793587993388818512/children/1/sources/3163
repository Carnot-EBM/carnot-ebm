"""Fixture trial orchestration for REQ-REPORT-7755.

Private synthetic labels qualify execution only. They are not natural corpus
labels and cannot establish semantic verification or policy benefit.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from carnot.verify import source_set_energy
from carnot.verify import training_runtime as runtime


def progress(start: float, phase: str, event: str, completed: int) -> None:
    """Flush a measured boundary or completed unit."""
    print(
        f"[exp7755] phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} completed={completed}",
        flush=True,
    )


def fixture_records() -> list[dict[str, Any]]:
    """Declare four separable responses and their paired source views."""
    return [
        {
            "id": "alpha_supported",
            "source": "Alpha is 12.",
            "answer": "Alpha is 12.",
            "label": 0,
            "known": [1],
        },
        {
            "id": "alpha_unsupported",
            "source": "Alpha is 12.",
            "answer": "Alpha is 13.",
            "label": 1,
            "known": [0],
        },
        {
            "id": "beta_supported",
            "source": "Beta is 30.",
            "answer": "Beta is 30.",
            "label": 0,
            "known": [1],
        },
        {
            "id": "beta_unsupported",
            "source": "Beta is 30.",
            "answer": "Beta is 31.",
            "label": 1,
            "known": [0],
        },
    ]


def _batch(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for record in records:
        source = record["source"].encode()
        answer = record["answer"].encode()
        rows.append(
            {
                "view_a": source_set_energy.prepare(source, answer),
                "view_b": source_set_energy.prepare(source + b" " + source, answer),
                "label": record["label"],
                "known": record["known"],
            }
        )
    return runtime.prepare_batch(rows)


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def run_fixtures(
    raw: Path,
    *,
    seeds: tuple[int, ...] = runtime.SEEDS,
    rates: tuple[float, ...] = runtime.LEARNING_RATES,
    epochs: int = 12,
    arms: tuple[str, ...] = runtime.ARMS,
    extra_modes: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Train registered trials and save one tune-selected head per condition."""
    start = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    records = fixture_records()
    _jsonl(raw / "fixtures.jsonl", records)
    batch = _batch(records)
    gradient_errors = {
        arm: runtime.gradient_error(runtime.init_params(arm, seeds[0]), batch, arm) for arm in arms
    }
    trial_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    conditions = [(arm, "canonical") for arm in arms] + [
        ("energy_local", mode) for mode in extra_modes
    ]
    progress(start, "fit", "start", 0)
    for arm, mode in conditions:
        best: dict[str, Any] | None = None
        for seed in seeds:
            for rate in rates:
                head = runtime.fit(arm, batch, batch, seed, rate, epochs, mode)
                row = {
                    "arm": arm,
                    "mode": mode,
                    "seed": seed,
                    "learning_rate": rate,
                    "initial_hash": head["initial_hash"],
                    "final_hash": head["final_hash"],
                    "initial_loss": head["curve"][0]["loss"],
                    "final_loss": float(
                        runtime.loss(head["params"], batch, arm, mode, tuple(head["duals"]))
                    ),
                    "tune_nll": head["tune_nll"],
                    "gradient_norm": head["curve"][-1]["gradient_norm"],
                    "duals": head["duals"],
                    "parameter_count": runtime.parameter_count(
                        head["params"], 2 if mode == "constrained" else 0
                    ),
                    "gradient_error": gradient_errors[arm],
                    "raw_path": str(raw / "fixtures.jsonl"),
                    "label_scope": "synthetic_fixture",
                }
                trial_rows.append(row)
                if best is None or head["tune_nll"] < best["tune_nll"]:
                    best = head
                progress(start, "fit", f"trial_{arm}_{mode}_{seed}_{rate}", len(trial_rows))
        assert best is not None
        paired = mode != "canonical"
        best["temperature"] = runtime.calibrate(best["params"], batch, arm, paired)
        name = f"{arm}_{mode}"
        progress(start, "save", f"before_model_save_{name}", len(decisions))
        runtime.save(raw / f"{name}.json", best)
        progress(start, "save", f"after_model_save_{name}", len(decisions))
        for index, record in enumerate(records):
            decision = runtime.decide(best, batch, index, paired)
            decisions.append(
                {
                    "family": record["id"],
                    "arm": arm,
                    "mode": mode,
                    "seed": best["seed"],
                    "label": record["label"],
                    "known": record["known"],
                    **decision,
                }
            )
    _jsonl(raw / "trials.jsonl", trial_rows)
    _jsonl(raw / "decisions.jsonl", decisions)
    progress(start, "fit", "complete", len(trial_rows))
    return {
        "trial_count": len(trial_rows),
        "decision_count": len(decisions),
        "trial_rows": trial_rows,
        "decision_rows": decisions,
        "records": records,
    }


def cold_reduce(raw: Path) -> dict[str, Any]:
    """Reopen fixture bytes and heads, then recompute every decision."""
    start = time.monotonic()
    progress(start, "cold_replay", "start", 0)
    records = [json.loads(line) for line in (raw / "fixtures.jsonl").read_text().splitlines()]
    trials = [json.loads(line) for line in (raw / "trials.jsonl").read_text().splitlines()]
    decisions = [json.loads(line) for line in (raw / "decisions.jsonl").read_text().splitlines()]
    batch = _batch(records)
    for index, row in enumerate(decisions):
        name = f"{row['arm']}_{row['mode']}"
        progress(start, "cold_replay", f"before_model_load_{name}", index)
        head = runtime.load(raw / f"{name}.json")
        progress(start, "cold_replay", f"after_model_load_{name}", index)
        family_index = next(i for i, item in enumerate(records) if item["id"] == row["family"])
        expected = runtime.decide(head, batch, family_index, row["mode"] != "canonical")
        if any(row[key] != value for key, value in expected.items()):
            raise ValueError("decision mismatch")
        progress(start, "cold_replay", "decision_verified", index + 1)
    return {
        "trial_count": len(trials),
        "decision_count": len(decisions),
        "mean_tune_nll": float(np.mean([row["tune_nll"] for row in trials])),
    }
