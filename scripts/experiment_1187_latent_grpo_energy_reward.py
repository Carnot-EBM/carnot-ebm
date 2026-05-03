#!/usr/bin/env python3
"""Exp 1187 — Latent-GRPO invalid energy masking + one-sided reward noise.

Spec: REQ-LEARN-1187, SCENARIO-LEARN-1187, SCENARIO-LEARN-1188,
      SCENARIO-LEARN-1189.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _maybe_reexec_repo_venv_for_cli() -> None:
    """Let the documented ``python scripts/...`` command use the repo venv."""
    if __name__ != "__main__":
        return
    venv_python = _REPO_ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    if os.environ.get("CARNOT_EXP1187_VENV_REEXEC") == "1":
        return
    os.environ["CARNOT_EXP1187_VENV_REEXEC"] = "1"
    os.execv(str(venv_python), [str(venv_python), *sys.argv])


_maybe_reexec_repo_venv_for_cli()

for _d in [str(_REPO_ROOT / "python"), str(_REPO_ROOT / "scripts"), str(_REPO_ROOT)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.training.latent_grpo import (  # noqa: E402
    LatentGRPOTrainer,
    REQUIRED_LATENT_GRPO_ARTIFACT_FIELDS,
    build_latent_grpo_artifact_fields,
)
from carnot.verify.and_composition_verifier import AndCompositionVerifier  # noqa: E402

EXP_ID = 1187
EXP_TITLE = "Latent-GRPO invalid energy masking with one-sided reward noise"
DELIVERABLE = _REPO_ROOT / "results" / "experiment_1187_latent_grpo_energy_reward.json"
FOVER_CORPUS = _REPO_ROOT / "data" / "fover_corpus.jsonl"
REQUESTED_V4_BASELINE = _REPO_ROOT / "results" / "experiment_1164_grpo_v4_structural_warmup.json"
FALLBACK_V4_BASELINE = _REPO_ROOT / "results" / "experiment_1159_grpo_v4_structural_warmup.json"
N_EVAL_QUESTIONS = 100
RANDOM_SEED = 1187


class _GradientRecorder:
    """Tiny base trainer used to exercise the LatentGRPOTrainer wrapper."""

    def __init__(self) -> None:
        self.prepared_rollouts: list[dict[str, Any]] = []

    def gradient_update(self, rollouts: list[dict[str, Any]]) -> dict[str, Any]:
        self.prepared_rollouts = list(rollouts)
        return {"n_update_rollouts": len(self.prepared_rollouts)}


def _utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _run_date() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%d")


def _artifact_base(started_at: str, status: str, body: dict[str, Any]) -> dict[str, Any]:
    finished_at = _utc_now()
    started_dt = _dt.datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    finished_dt = _dt.datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
    artifact: dict[str, Any] = {
        "experiment": EXP_ID,
        "title": EXP_TITLE,
        "run_date": _run_date(),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": round((finished_dt - started_dt).total_seconds(), 3),
        "status": status,
        "random_seed": RANDOM_SEED,
        "cost_usd": 0.0,
        "decision_class": ["training", "verify"],
        "metrics_used": "fover_100q_verifier_accuracy_proxy",
        "schema_version": "latent_grpo_v1",
    }
    artifact.update(body)
    checksum_src = json.dumps(artifact, sort_keys=True, default=str).encode()
    artifact["reproducibility_checksum"] = hashlib.sha256(checksum_src).hexdigest()[:16]
    artifact["schema"] = sorted([*artifact.keys(), "schema"])
    return artifact


def _load_first_100_low_difficulty_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    difficulty_field_seen = False
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            difficulty = row.get("difficulty")
            if difficulty is not None:
                difficulty_field_seen = True
            if difficulty is None or str(difficulty).lower() in {"low", "easy"}:
                rows.append(row)
            if len(rows) >= N_EVAL_QUESTIONS:
                break
    mode = "difficulty_low_or_missing" if difficulty_field_seen else "first_100_no_difficulty_field"
    return rows, mode


def _published_v4_baseline() -> tuple[float | None, str | None, bool]:
    if REQUESTED_V4_BASELINE.exists():
        data = json.loads(REQUESTED_V4_BASELINE.read_text())
        return float(data.get("trained_fraction_correct", 0.0)), str(REQUESTED_V4_BASELINE), True
    if FALLBACK_V4_BASELINE.exists():
        data = json.loads(FALLBACK_V4_BASELINE.read_text())
        return float(data.get("trained_fraction_correct", 0.0)), str(FALLBACK_V4_BASELINE), False
    return None, None, False


def _score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    verifier = AndCompositionVerifier()
    rollouts: list[dict[str, Any]] = []
    energies: list[list[float]] = []
    records: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        text = str(row.get("step_text", ""))
        label = str(row.get("label", "not_verifiable"))
        if not text.strip():
            result_scores = [0.0] * 5
            predicted_label = "incorrect"
        else:
            result = verifier.verify("", text)
            result_scores = [float(v) for v in result.per_verifier_scores.values()]
            predicted_label = "correct" if result.verified else "incorrect"
        standard_correct = predicted_label == label
        reward = 1.0 if label == "correct" else -1.0
        rollouts.append(
            {
                "question_id": str(row.get("question_id", idx)),
                "reward": reward,
                "standard_correct": standard_correct,
                "label": label,
                "predicted_label": predicted_label,
            }
        )
        energies.append(result_scores)
        records.append(
            {
                "question_id": str(row.get("question_id", idx)),
                "label": label,
                "predicted_label": predicted_label,
                "standard_correct": bool(standard_correct),
                "energies": result_scores,
            }
        )

    base = _GradientRecorder()
    trainer = LatentGRPOTrainer(base, noise_scale=0.01)
    rng = random.Random(RANDOM_SEED)
    update_meta = trainer.gradient_update(rollouts, energies, rng=rng.gauss)

    standard_correct_count = sum(1 for rollout in rollouts if rollout["standard_correct"])
    latent_correct_ids = {
        rollout["question_id"] for rollout in base.prepared_rollouts if rollout["standard_correct"]
    }
    latent_correct_count = len(latent_correct_ids)

    positives = [r for r in rollouts if r["reward"] > 0]
    positive_noise_applied = any(
        prepared["reward"] != 1.0
        for prepared in base.prepared_rollouts
        if prepared["label"] == "correct"
    )
    return {
        "records": records,
        "update_meta": update_meta,
        "standard_correct_count": standard_correct_count,
        "latent_correct_count": latent_correct_count,
        "standard_pass_rate": standard_correct_count / len(rows) if rows else 0.0,
        "latent_pass_rate": latent_correct_count / len(rows) if rows else 0.0,
        "mask_rate": trainer.last_mask_rate,
        "n_masked": trainer.last_n_masked,
        "n_positive_reward_rollouts": len(positives),
        "positive_noise_applied": positive_noise_applied,
    }


def _run_experiment() -> dict[str, Any]:
    started_at = _utc_now()
    rows, subset_mode = _load_first_100_low_difficulty_rows(FOVER_CORPUS)
    published_v4, published_v4_source, requested_v4_found = _published_v4_baseline()
    score_meta = _score_rows(rows)

    artifact_fields = build_latent_grpo_artifact_fields(
        mask_rate=round(float(score_meta["mask_rate"]), 6),
        grpo_v4_baseline_pass_rate=round(float(score_meta["standard_pass_rate"]), 6),
        latent_grpo_pass_rate=round(float(score_meta["latent_pass_rate"]), 6),
        n_eval_questions=len(rows),
        one_sided_noise_applied=bool(score_meta["positive_noise_applied"]),
    )
    body: dict[str, Any] = {
        "fover_corpus_path": str(FOVER_CORPUS.relative_to(_REPO_ROOT)),
        "subset_mode": subset_mode,
        "requested_v4_baseline_found": requested_v4_found,
        "requested_v4_baseline_path": str(REQUESTED_V4_BASELINE.relative_to(_REPO_ROOT)),
        "published_v4_baseline_source": (
            str(Path(published_v4_source).relative_to(_REPO_ROOT)) if published_v4_source else None
        ),
        "published_v4_trained_fraction_correct": published_v4,
        "baseline_definition": (
            "standard k=5 verifier classification accuracy on the same first 100 "
            "FoVer rows before Latent-GRPO masking/noise; requested exp1164 "
            "structural-warmup artifact is absent in this checkout"
        ),
        "latent_definition": (
            "same k=5 verifier records after LatentGRPOTrainer masks invalid "
            "energy ensembles and applies one-sided noise before the proxy update"
        ),
        "n_masked": int(score_meta["n_masked"]),
        "n_positive_reward_rollouts": int(score_meta["n_positive_reward_rollouts"]),
        "standard_correct_count": int(score_meta["standard_correct_count"]),
        "latent_correct_count": int(score_meta["latent_correct_count"]),
        "gradient_update_meta": score_meta["update_meta"],
        "eval_records_head": score_meta["records"][:5],
        "paper_refs": [
            "arXiv 2604.27998 (Latent-GRPO invalid-sample masking + one-sided noise)",
            "Exp 1159 GRPO v4 structural warm-up baseline artifact fallback",
        ],
    }
    body.update(artifact_fields)
    return _artifact_base(started_at, "success", body)


def main() -> int:
    artifact = _run_experiment()
    missing = [key for key in REQUIRED_LATENT_GRPO_ARTIFACT_FIELDS if key not in artifact]
    if missing:
        raise AssertionError(f"REQ-LEARN-1187-5 missing fields: {missing}")
    DELIVERABLE.parent.mkdir(parents=True, exist_ok=True)
    DELIVERABLE.write_text(json.dumps(artifact, indent=2, default=str))
    print(f"[exp1187] wrote {DELIVERABLE}", flush=True)
    print(
        f"[exp1187] honest_verdict={artifact.get('honest_verdict')} "
        f"mask_rate={artifact.get('mask_rate')} "
        f"grpo_v4_baseline_pass_rate={artifact.get('grpo_v4_baseline_pass_rate')} "
        f"latent_grpo_pass_rate={artifact.get('latent_grpo_pass_rate')} "
        f"latent_grpo_delta_pp={artifact.get('latent_grpo_delta_pp')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
