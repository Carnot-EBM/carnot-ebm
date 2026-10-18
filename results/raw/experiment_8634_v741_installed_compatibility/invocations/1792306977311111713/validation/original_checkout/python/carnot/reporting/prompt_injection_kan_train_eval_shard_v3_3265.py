"""Build the Exp 3265 prompt-injection KAN train/eval shard v3 artifact.

Spec refs: REQ-REPORT-3265, SCENARIO-REPORT-3265, REQ-KAN-004.

This module turns the Exp 3264 teacher-labeled shard into a bounded viability
check for the 16-knot prompt-injection KAN. It deliberately keeps the claim
small: the artifact reports held-out single-shard AUROC, not replacement-grade
evidence. The full replacement gate still needs the multi-shard 15k corpus,
DeLong non-inferiority, and Garak evidence named in the v4 plan.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

from carnot.models.prompt_injection_kan import InjectionExample, PromptInjectionEnergyCheckerV3


JsonDict = dict[str, Any]
ClockFn = Callable[[], float]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.prompt_injection_kan_train_eval_shard.v3"
EXPERIMENT_ID = "exp3265"
TASK_ID = "exp3265-prompt-injection-kan-train-eval-shard-v3"
ARTIFACT = "experiment_3265_prompt_injection_kan_train_eval_shard_v3"
MILESTONE = "2026.05.302"
RANDOM_SEED = 3265
DEFAULT_EVAL_FRACTION = 0.2
DEFAULT_N_EPOCHS = 100
DEFAULT_LR = 1e-3
ALLOWED_LABELS = ("benign", "injection")

OUTPUT_REL_PATH = Path("results/experiment_3265_prompt_injection_kan_train_eval_shard_v3.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3265_prompt_injection_kan_train_eval_shard_v3.py"
EXP3264_REL_PATH = Path("results/experiment_3264_prompt_injection_teacher_label_shard_v3.json")

NON_HEADLINE_NOTE = (
    "single-shard AUROC is a viability check only; it is not replacement-grade "
    "and cannot replace the full multi-shard 15k corpus plus DeLong "
    "non-inferiority plus Garak acceptance gates."
)


def build_artifact(
    *,
    project_root: str | Path = REPO_ROOT,
    monotonic: ClockFn = time.perf_counter,
    random_seed: int = RANDOM_SEED,
    eval_fraction: float = DEFAULT_EVAL_FRACTION,
    n_epochs: int = DEFAULT_N_EPOCHS,
    lr: float = DEFAULT_LR,
) -> JsonDict:
    """REQ-REPORT-3265: train/eval the 16-knot KAN when Exp 3264 is ready."""

    start = monotonic()
    root = Path(project_root)
    exp3264 = _read_json(root / EXP3264_REL_PATH)
    rows = _extract_labeled_rows(exp3264)
    blocked_reason = ""
    train_rows: list[JsonDict] = []
    eval_rows: list[JsonDict] = []
    train_eval: JsonDict = {
        "shard_auroc": 0.0,
        "loss_curve_count": 0,
        "loss_curve_first": None,
        "loss_curve_last": None,
        "trained_model_checksum": "",
    }

    if exp3264.get("teacher_label_shard_ready") is not True:
        blocked_reason = "gated_exp3264_teacher_label_shard_not_ready"
    elif not _has_both_classes(rows):
        blocked_reason = "labeled_shard_lacks_both_classes"
    else:
        train_rows, eval_rows = _split_rows(
            rows,
            random_seed=int(random_seed),
            eval_fraction=float(eval_fraction),
        )
        if not _has_both_classes(train_rows) or not _has_both_classes(eval_rows):
            blocked_reason = "labeled_shard_split_lacks_both_classes"
        else:
            train_eval = _train_and_eval(
                train_rows=train_rows,
                eval_rows=eval_rows,
                n_epochs=int(n_epochs),
                lr=float(lr),
            )

    shard_ready = blocked_reason == ""
    model_specs = _model_specs(
        random_seed=int(random_seed),
        eval_fraction=float(eval_fraction),
        n_epochs=int(n_epochs),
        lr=float(lr),
    )
    n_train = len(train_rows) if shard_ready else 0
    n_eval = len(eval_rows) if shard_ready else 0
    shard_auroc = float(train_eval["shard_auroc"]) if shard_ready else 0.0
    duration_s = _duration(start, monotonic())
    stable_payload = {
        "blocked_reason": blocked_reason,
        "eval_example_ids": [str(row["example_id"]) for row in eval_rows] if shard_ready else [],
        "eval_label_counts": _label_counts(eval_rows) if shard_ready else {},
        "exp3264_checksum": str(exp3264.get("reproducibility_checksum") or ""),
        "model_specs": model_specs,
        "n_eval": n_eval,
        "n_train": n_train,
        "random_seed": int(random_seed),
        "shard_auroc": shard_auroc,
        "train_example_ids": [str(row["example_id"]) for row in train_rows] if shard_ready else [],
        "train_label_counts": _label_counts(train_rows) if shard_ready else {},
        "trained_model_checksum": str(train_eval["trained_model_checksum"]),
    }
    checksum = _reproducibility_checksum(stable_payload)

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "upstream_exp3264": {
            "path": str(root / EXP3264_REL_PATH),
            "teacher_label_shard_ready": exp3264.get("teacher_label_shard_ready") is True,
            "shard_size": int(exp3264.get("shard_size") or 0),
            "reproducibility_checksum": str(exp3264.get("reproducibility_checksum") or ""),
        },
        "kan_train_eval_shard_v3_ready": shard_ready,
        "kan_train_eval_shard_ready": shard_ready,
        "blocked_reason": blocked_reason,
        "shard_auroc": shard_auroc,
        "non_headline_note": NON_HEADLINE_NOTE,
        "n_train": n_train,
        "n_eval": n_eval,
        "source_label_counts": _label_counts(rows),
        "train_label_counts": _label_counts(train_rows) if shard_ready else {},
        "eval_label_counts": _label_counts(eval_rows) if shard_ready else {},
        "model_specs": model_specs,
        "training_summary": {
            "loss_curve_count": int(train_eval["loss_curve_count"]),
            "loss_curve_first": train_eval["loss_curve_first"],
            "loss_curve_last": train_eval["loss_curve_last"],
            "trained_model_checksum": str(train_eval["trained_model_checksum"]),
        },
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "duration_s": duration_s,
        "honest_verdict": _honest_verdict(
            shard_ready=shard_ready,
            blocked_reason=blocked_reason,
            shard_auroc=shard_auroc,
            n_train=n_train,
            n_eval=n_eval,
        ),
    }


def write_artifact(
    *,
    project_root: str | Path = REPO_ROOT,
    output_path: str | Path = OUTPUT_REL_PATH,
    monotonic: ClockFn = time.perf_counter,
    random_seed: int = RANDOM_SEED,
    eval_fraction: float = DEFAULT_EVAL_FRACTION,
    n_epochs: int = DEFAULT_N_EPOCHS,
    lr: float = DEFAULT_LR,
) -> JsonDict:
    """Build and persist the Exp 3265 KAN train/eval shard JSON."""

    root = Path(project_root)
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = root / destination
    artifact = build_artifact(
        project_root=root,
        monotonic=monotonic,
        random_seed=random_seed,
        eval_fraction=eval_fraction,
        n_epochs=n_epochs,
        lr=lr,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _read_json(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _extract_labeled_rows(exp3264: Mapping[str, Any]) -> list[JsonDict]:
    raw_rows = exp3264.get("per_example_labels")
    if not isinstance(raw_rows, list):
        return []
    rows: list[JsonDict] = []
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            continue
        label = str(raw.get("teacher_label") or "")
        text = str(raw.get("text") or "")
        if raw.get("parse_status") != "parsed" or label not in ALLOWED_LABELS or not text:
            continue
        rows.append(
            {
                "example_id": str(raw.get("example_id") or f"row:{index:06d}"),
                "source": str(raw.get("source") or "exp3264"),
                "source_index": int(raw.get("source_index") or index),
                "source_label": str(raw.get("source_label") or ""),
                "text": text,
                "text_sha256": str(raw.get("text_sha256") or _sha256_text(text)),
                "label": label,
            }
        )
    return rows


def _split_rows(
    rows: list[JsonDict],
    *,
    random_seed: int,
    eval_fraction: float,
) -> tuple[list[JsonDict], list[JsonDict]]:
    train_rows: list[JsonDict] = []
    eval_rows: list[JsonDict] = []
    for offset, label in enumerate(ALLOWED_LABELS):
        class_rows = [row for row in rows if row["label"] == label]
        rng = random.Random(random_seed + offset)
        rng.shuffle(class_rows)
        n_eval = max(1, round(len(class_rows) * eval_fraction))
        n_eval = min(n_eval, max(1, len(class_rows) - 1))
        eval_rows.extend(class_rows[:n_eval])
        train_rows.extend(class_rows[n_eval:])
    random.Random(random_seed + 17).shuffle(train_rows)
    random.Random(random_seed + 31).shuffle(eval_rows)
    return train_rows, eval_rows


def _train_and_eval(
    *,
    train_rows: list[JsonDict],
    eval_rows: list[JsonDict],
    n_epochs: int,
    lr: float,
) -> JsonDict:
    checker = PromptInjectionEnergyCheckerV3()
    train_examples = [_to_example(row) for row in train_rows]
    eval_examples = [_to_example(row) for row in eval_rows]
    loss_curve = checker.train(train_examples, n_epochs=n_epochs, lr=lr)
    shard_auroc = round(float(checker.evaluate_auroc(eval_examples)), 6)
    return {
        "shard_auroc": shard_auroc,
        "loss_curve_count": len(loss_curve),
        "loss_curve_first": round(float(loss_curve[0]), 6) if loss_curve else None,
        "loss_curve_last": round(float(loss_curve[-1]), 6) if loss_curve else None,
        "trained_model_checksum": _model_checksum(checker),
    }


def _to_example(row: Mapping[str, Any]) -> InjectionExample:
    return InjectionExample(
        text=str(row["text"]),
        label=str(row["label"]),  # type: ignore[arg-type]
        source=str(row.get("source") or "exp3264"),
    )


def _model_specs(
    *,
    random_seed: int,
    eval_fraction: float,
    n_epochs: int,
    lr: float,
) -> JsonDict:
    checker = PromptInjectionEnergyCheckerV3()
    return {
        "model_class": "PromptInjectionEnergyCheckerV3",
        "schema": "carnot.prompt_injection_kan.v3",
        "n_features": checker.n_features,
        "n_hidden": checker.n_hidden,
        "n_knots": checker._N_KNOTS,
        "degree": checker._DEGREE,
        "n_params": checker.n_params(),
        "eval_fraction": float(eval_fraction),
        "n_epochs": int(n_epochs),
        "lr": float(lr),
        "random_seed": int(random_seed),
    }


def _has_both_classes(rows: list[JsonDict]) -> bool:
    counts = _label_counts(rows)
    return counts.get("benign", 0) > 0 and counts.get("injection", 0) > 0


def _label_counts(rows: list[JsonDict]) -> JsonDict:
    return dict(sorted(Counter(str(row.get("label") or "") for row in rows).items()))


def _honest_verdict(
    *,
    shard_ready: bool,
    blocked_reason: str,
    shard_auroc: float,
    n_train: int,
    n_eval: int,
) -> str:
    if shard_ready:
        return (
            "complete: kan_train_eval_shard_v3_ready=true; "
            "kan_train_eval_shard_ready=true; "
            f"shard_auroc={shard_auroc:.6f}; n_train={n_train}; n_eval={n_eval}; "
            "non_headline=single_shard_viability_only"
        )
    return (
        "complete: kan_train_eval_shard_v3_ready=false; "
        "kan_train_eval_shard_ready=false; "
        f"blocked_reason={blocked_reason}"
    )


def _duration(start: float, now: float) -> float:
    return round(max(0.0, float(now) - float(start)), 6)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _model_checksum(checker: PromptInjectionEnergyCheckerV3) -> str:
    digest = hashlib.sha256()
    digest.update(str(checker.edge_ctrl.shape).encode("utf-8"))
    digest.update(checker.edge_ctrl.tobytes())
    digest.update(str(checker.output_ctrl.shape).encode("utf-8"))
    digest.update(checker.output_ctrl.tobytes())
    return digest.hexdigest()


def _reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:  # pragma: no cover
    artifact = write_artifact(project_root=REPO_ROOT)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
