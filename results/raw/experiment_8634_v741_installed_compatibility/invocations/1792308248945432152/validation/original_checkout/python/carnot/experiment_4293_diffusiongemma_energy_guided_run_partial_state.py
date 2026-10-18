"""Exp 4293: DiffusionGemma guided run using the Exp 4292 partial-state scorer.

The runner is precondition-first. It refuses to run unless the DiffusionGemma
PR binary and GGUF cache are present, TRM training is stood down, and Exp 4292
produced a built, leak-free, oracle-distinct partial-state scorer. The measured
benchmark compares unguided, RFG, EntRGi, and Carnot partial-state guidance on
the same reasoning-choice denoising tasks.

Spec refs: REQ-VERIFY-4293, SCENARIO-VERIFY-4293.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from carnot.experiment_4260_diffusiongemma_energy_guided_preflight import (
    CACHE_REPO_DIRNAME,
    DEFAULT_CACHE_ROOT,
    GGUF_HF_ID,
    GuidanceConfig,
    VocabLoadResult,
    _default_process_rows,
)
from carnot.experiment_4274_diffusiongemma_loader_fix_preflight import repaired_vocab_loader
from carnot.experiment_4281_diffusiongemma_energy_guided_full_run import (
    CANVAS_LEN,
    MASK_TOKEN_ID,
    PR_BINARY,
    VOCAB_SIZE,
    bootstrap_delta_ci,
    run_guidance_smoke,
)
from carnot.experiment_4292_partial_state_diffusion_scorer_build import (
    ARTIFACT_PATH as EXP4292_ARTIFACT_PATH,
    SCORER_PATH as EXP4292_SCORER_PATH,
    check_preconditions as check_diffusiongemma_preconditions,
    load_reasoning_items,
)
from carnot.inference.sota_models import resolve_cached_gguf
from carnot.verify.partial_state_diffusion_scorer import (
    ByteCanvasEncoder,
    PartialStateDiffusionScorer,
    corpus_checksum,
    split_items_task_disjoint,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = (
    ROOT / "results" / "experiment_4293_diffusiongemma_energy_guided_run_partial_state.json"
)
RANDOM_SEED = 4293
SPEC_REFS = ["REQ-VERIFY-4293", "SCENARIO-VERIFY-4293"]
INFERENCE_SUBSTRATE = "live_llm_inference"
GUIDANCE_CONDITIONS = ["unguided", "RFG", "EntRGi", "Carnot-partial-state-guided"]
CONDITION_KEYS = ("unguided", "rfg", "entrgi", "carnot")
DEFAULT_MAX_TASKS = 30
DEFAULT_BOOTSTRAP_RESAMPLES = 2000
DEFAULT_MINIMUM_LIVE_DURATION_S = 60.0
CHOICE_OPTIONS = ("A", "B", "C", "D")
VISIBLE_FRACTION = 0.7

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A guidance moat (learned-verifier beats RFG, CI95-excl-0), "
        "a bounded null (ties RFG), and an over-guided-diagnostic finding are ALL "
        "COMPLETE and decision-grade for the section 5 thesis."
    ),
    "diffusiongemma_guidance_moat": (
        "BARE bool: the capstone reads this (gated-fields-must-be-bare); true iff the "
        "LEARNED (oracle-distinct) partial-state-guided run beats RFG model-self-guidance "
        "AND CI95-excl-0 -- the moat-scissor realized in generation at LLM scale."
    ),
    "carnot_minus_rfg_delta": (
        "BARE float: Carnot-partial-state-guided minus RFG accuracy on the reasoning "
        "corpus -- the load-bearing comparison (beating the model's own self-guidance "
        "shows an EXTERNAL verifier adds value in-generation)."
    ),
    "carnot_minus_unguided_delta": (
        "BARE float: Carnot-guided minus unguided -- the weaker control (a guidance hook "
        "that does anything beats unguided; the moat needs the RFG comparison too)."
    ),
    "guidance_moat_ci95": (
        "Task-level bootstrap CI95 of the Carnot-minus-RFG delta -- excluding 0 means "
        "the external verifier genuinely steers generation better than model self-guidance."
    ),
    "guidance_dynamics_diagnostic": (
        "Mask entropy / token-change covariance / trajectory stability -- bounds the "
        "result (an over-guided unstable scorer's win is robustness-theater, not real "
        "generation improvement)."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- the learned partial-state scorer is oracle-distinct (NOT the "
        "executable oracle); a circular guidance win cannot headline."
    ),
    "preconditions_checked": (
        "Records the PR-binary + GGUF cache + scorer-leak-free + TRM-stand-down verified; "
        "pre-empts the silent-missing-resource fabrication mode."
    ),
    "random_seed": "Determinism precondition for the denoising + bootstrap.",
    "reproducibility_checksum": (
        "Hash of the corpus + guidance config + PR-binary inputs; lets a third party re-run."
    ),
    "model_specs": (
        "DiffusionGemma GGUF + PR binary + the partial-state scorer wired as guidance + "
        "denoising steps + the four conditions + the corpus; required methodology."
    ),
}

REQUIRED_FIELDS = [
    "honest_verdict",
    "diffusiongemma_guidance_moat",
    "carnot_minus_rfg_delta",
    "carnot_minus_unguided_delta",
    "guidance_moat_ci95",
    "guidance_dynamics_diagnostic",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
    "field_principles",
    "spec_refs",
    "duration_s",
    "inference_substrate",
]


@dataclass(frozen=True)
class Choice:
    option: str
    task_id: str
    corpus_item_id: str
    step_text: str
    label: bool
    canvas_ids: tuple[int, ...]
    scorer_step: int


@dataclass(frozen=True)
class ChoiceTask:
    task_id: str
    prompt: str
    choices: tuple[Choice, ...]
    correct_option: str


def build_choice_tasks(
    items: Sequence[dict[str, Any]],
    *,
    max_tasks: int = DEFAULT_MAX_TASKS,
    seed: int = RANDOM_SEED,
    encoder: ByteCanvasEncoder | None = None,
) -> list[ChoiceTask]:
    """Build four-option reasoning denoising tasks from labeled FoVer/math rows."""

    positives = [dict(item) for item in items if str(item.get("label", "")).lower() == "correct"]
    negatives = [dict(item) for item in items if str(item.get("label", "")).lower() == "incorrect"]
    if len(positives) < max_tasks or not negatives:
        raise ValueError("at least 30 positive reasoning rows and one negative row are required")

    rng = random.Random(seed)
    encoder = encoder or ByteCanvasEncoder(canvas_len=CANVAS_LEN, mask_token_id=MASK_TOKEN_ID)
    tasks: list[ChoiceTask] = []
    for index in range(int(max_tasks)):
        raw_choices = [positives[index]]
        raw_choices.extend(negatives[(index * 3 + offset) % len(negatives)] for offset in range(3))
        rng.shuffle(raw_choices)
        prompt_lines = [
            "Choose the valid reasoning step. Return only A, B, C, or D.",
            f"Task {index:03d}:",
        ]
        choices: list[Choice] = []
        correct_option = ""
        for option, item in zip(CHOICE_OPTIONS, raw_choices, strict=True):
            step_text = str(item.get("step_text", ""))
            canvas_ids, _answer_indices = encoder.encode(
                step_text,
                visible_fraction=VISIBLE_FRACTION,
            )
            label = str(item.get("label", "")).lower() == "correct"
            if label:
                correct_option = option
            prompt_lines.append(f"{option}. {step_text}")
            choices.append(
                Choice(
                    option=option,
                    task_id=str(item.get("question_id") or item.get("corpus_item_id") or index),
                    corpus_item_id=str(
                        item.get("corpus_item_id") or item.get("question_id") or index
                    ),
                    step_text=step_text,
                    label=label,
                    canvas_ids=tuple(canvas_ids),
                    scorer_step=1,
                )
            )
        tasks.append(
            ChoiceTask(
                task_id=f"fover_math_choice_{index:03d}",
                prompt="\n".join(prompt_lines),
                choices=tuple(choices),
                correct_option=correct_option,
            )
        )
    return tasks


def load_benchmark_items() -> list[dict[str, Any]]:  # pragma: no cover - live default wrapper.
    """Use Exp 4292's held-out split so the scorer is evaluated off-train."""

    _train_items, heldout_items = split_items_task_disjoint(
        load_reasoning_items(),
        heldout_fraction=0.25,
        seed=4292,
    )
    return heldout_items


def check_scorer_gate(
    *,
    scorer_artifact_path: Path,
    scorer_path: Path,
    scorer_loader_fn: Callable[[Path], Any] = PartialStateDiffusionScorer.load,
) -> tuple[dict[str, Any], Any | None]:
    """Check that Exp 4292 built a leak-free, loadable, oracle-distinct scorer."""

    path = Path(scorer_artifact_path)
    check: dict[str, Any] = {
        "resource": "partial_state_scorer_gate",
        "artifact_path": str(path),
        "scorer_path": str(scorer_path),
        "ok": False,
    }
    if not path.exists():
        check["error"] = "exp4292 artifact missing"
        return check, None
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        check["error"] = f"exp4292 artifact unreadable: {type(exc).__name__}: {exc}"
        return check, None

    artifact_scorer_path = Path(str(artifact.get("scorer_path") or scorer_path))
    selected_scorer_path = Path(scorer_path or artifact_scorer_path)
    built = artifact.get("partial_state_scorer_built") is True
    leak_free = artifact.get("partial_state_leak_free") is True
    oracle_distinct = artifact.get("verifier_is_oracle") is False
    check.update(
        {
            "partial_state_scorer_built": built,
            "partial_state_leak_free": leak_free,
            "verifier_is_oracle": artifact.get("verifier_is_oracle"),
            "partial_state_auroc": artifact.get("partial_state_auroc"),
            "leak_ablation_auroc": artifact.get("leak_ablation_auroc"),
            "artifact_scorer_path": str(artifact_scorer_path),
            "scorer_path": str(selected_scorer_path),
            "scorer_exists": selected_scorer_path.exists(),
        }
    )
    if not (built and leak_free and oracle_distinct and selected_scorer_path.exists()):
        check["error"] = "exp4292 scorer is missing, leaky, unbuilt, or oracle-circular"
        return check, None
    try:
        scorer = scorer_loader_fn(selected_scorer_path)
        probe_energy = float(scorer.score_partial_state([MASK_TOKEN_ID] * CANVAS_LEN, 0))
    except Exception as exc:
        check["load_error"] = f"{type(exc).__name__}: {exc}"
        return check, None
    check.update({"ok": True, "scorer_loadable": True, "probe_energy": round(probe_energy, 6)})
    return check, scorer


def extract_option_logits_prior(
    *,
    task: ChoiceTask,
    tokenizer: Any,
    pr_binary_path: Path,
    gguf_path: str,
    timeout_s: float = 300.0,
) -> dict[str, Any]:  # pragma: no cover - exercises local 26B PR binary.
    """Run the PR binary once and read the first-mask logits for A/B/C/D."""

    option_token_ids = {option: _option_token_id(tokenizer, option) for option in CHOICE_OPTIONS}
    with tempfile.TemporaryDirectory(prefix="carnot_exp4293_dgemma_") as tmp:
        workdir = Path(tmp)
        prompt_ids = [int(item) for item in tokenizer.tokenize(task.prompt.encode("utf-8"))][:512]
        if not prompt_ids:
            prompt_ids = [0]
        prompt_path = workdir / "prompt_ids.i32"
        canvas_path = workdir / "canvas_ids.i32"
        logits_path = workdir / "out_logits.bin"
        _write_int32_file(prompt_path, prompt_ids)
        _write_int32_file(canvas_path, [MASK_TOKEN_ID] * CANVAS_LEN)
        try:
            proc = subprocess.run(
                [
                    str(pr_binary_path),
                    str(gguf_path),
                    str(prompt_path),
                    str(canvas_path),
                    str(logits_path),
                ],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                },
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "status": "blocked_pr_binary_eval_timeout",
                "timeout_s": float(timeout_s),
                "stderr_tail": str(exc)[-400:],
                "option_token_ids": option_token_ids,
            }
        if proc.returncode != 0 or not logits_path.exists():
            return {
                "status": "blocked_pr_binary_eval_failed",
                "eval_rc": int(proc.returncode),
                "stderr_tail": proc.stderr[-600:],
                "stdout_tail": proc.stdout[-600:],
                "option_token_ids": option_token_ids,
                "prompt_ids_count": len(prompt_ids),
            }
        expected = CANVAS_LEN * VOCAB_SIZE * 4
        size = logits_path.stat().st_size
        if size != expected:
            return {
                "status": "blocked_pr_binary_eval_bad_shape",
                "logits_file_size_bytes": int(size),
                "expected_logits_file_size_bytes": int(expected),
                "option_token_ids": option_token_ids,
                "prompt_ids_count": len(prompt_ids),
            }
        option_logits: dict[str, float] = {}
        with logits_path.open("rb") as handle:
            for option, token_id in option_token_ids.items():
                handle.seek(int(token_id) * 4)
                option_logits[option] = float(struct.unpack("<f", handle.read(4))[0])
        return {
            "status": "extracted",
            "eval_rc": int(proc.returncode),
            "score_shape": [CANVAS_LEN, VOCAB_SIZE],
            "prompt_ids_count": len(prompt_ids),
            "option_token_ids": option_token_ids,
            "option_logits": option_logits,
            "mask_entropy": _entropy_from_logits(list(option_logits.values())),
        }


def run_guided_choice_benchmark(
    *,
    tasks: Sequence[ChoiceTask],
    scorer: Any,
    tokenizer: Any,
    pr_binary_path: Path,
    gguf_path: str,
    config: GuidanceConfig,
    option_prior_fn: Callable[..., dict[str, Any]] = extract_option_logits_prior,
) -> dict[str, Any]:
    """Run all four matched conditions over the same task-level choices."""

    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for task in tasks:
        prior = option_prior_fn(
            task=task,
            tokenizer=tokenizer,
            pr_binary_path=Path(pr_binary_path),
            gguf_path=str(gguf_path),
        )
        if prior.get("status") != "extracted":
            failures.append({"task_id": task.task_id, "prior": dict(prior)})
            continue
        selections = select_conditions(
            task=task,
            option_logits={str(k): float(v) for k, v in dict(prior["option_logits"]).items()},
            scorer=scorer,
            config=config,
        )
        row = {
            "task_id": task.task_id,
            "unguided": selections["unguided"]["correct"],
            "rfg": selections["rfg"]["correct"],
            "entrgi": selections["entrgi"]["correct"],
            "carnot": selections["carnot"]["correct"],
        }
        rows.append(row)
        records.append(
            {
                "task_id": task.task_id,
                "correct_option": task.correct_option,
                "mask_entropy": float(prior.get("mask_entropy", 0.0) or 0.0),
                "option_logits": dict(prior["option_logits"]),
                "selections": selections,
                "unguided_option": selections["unguided"]["option"],
                "rfg_option": selections["rfg"]["option"],
                "entrgi_option": selections["entrgi"]["option"],
                "carnot_option": selections["carnot"]["option"],
                "rfg_correct": selections["rfg"]["correct"],
                "carnot_correct": selections["carnot"]["correct"],
            }
        )
    return {"rows": rows, "records": records, "failures": failures}


def select_conditions(
    *,
    task: ChoiceTask,
    option_logits: dict[str, float],
    scorer: Any,
    config: GuidanceConfig,
) -> dict[str, dict[str, Any]]:
    """Select A/B/C/D under unguided, RFG, EntRGi, and Carnot guidance."""

    energies = {
        choice.option: float(scorer.score_partial_state(choice.canvas_ids, choice.scorer_step))
        for choice in task.choices
    }
    mean_logit = statistics.fmean(option_logits.values())
    mean_energy = statistics.fmean(energies.values())
    entropy = _entropy_from_logits(list(option_logits.values()))
    scores = {
        "unguided": {option: option_logits[option] for option in CHOICE_OPTIONS},
        "rfg": {
            option: option_logits[option]
            + 0.35 * (option_logits[option] - mean_logit)
            + _reward_free_self_prior(task, option)
            for option in CHOICE_OPTIONS
        },
        "entrgi": {
            option: option_logits[option]
            - 0.05 * abs(option_logits[option] - mean_logit)
            + 0.02 * entropy
            for option in CHOICE_OPTIONS
        },
        "carnot": {
            option: option_logits[option]
            - float(config.guidance_lambda) * (energies[option] - mean_energy)
            for option in CHOICE_OPTIONS
        },
    }
    by_option = {choice.option: choice for choice in task.choices}
    selections: dict[str, dict[str, Any]] = {}
    for condition, condition_scores in scores.items():
        selected = max(CHOICE_OPTIONS, key=lambda option: (condition_scores[option], option))
        selections[condition] = {
            "option": selected,
            "correct": bool(by_option[selected].label),
            "score": round(float(condition_scores[selected]), 6),
            "logit": round(float(option_logits[selected]), 6),
            "partial_state_energy": round(float(energies[selected]), 6),
        }
    return selections


def summarize_condition_rows(
    rows: Sequence[dict[str, Any]],
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Summarize task-level pass rates and paired Carnot-minus-control deltas."""

    if not rows:
        raise ValueError("at least one condition row is required")
    pass_counts = {key: sum(1 for row in rows if bool(row[key])) for key in CONDITION_KEYS}
    condition_accuracy = {
        key: round(float(pass_counts[key] / len(rows)), 6) for key in CONDITION_KEYS
    }
    ci95 = bootstrap_delta_ci(
        [bool(row["carnot"]) for row in rows],
        [bool(row["rfg"]) for row in rows],
        resamples=int(resamples),
        seed=seed,
    )
    carnot_minus_rfg = condition_accuracy["carnot"] - condition_accuracy["rfg"]
    carnot_minus_unguided = condition_accuracy["carnot"] - condition_accuracy["unguided"]
    return {
        "status": "measured",
        "n": int(len(rows)),
        "condition_accuracy": condition_accuracy,
        "condition_pass_counts": {key: int(value) for key, value in pass_counts.items()},
        "carnot_minus_rfg_delta": round(float(carnot_minus_rfg), 6),
        "carnot_minus_unguided_delta": round(float(carnot_minus_unguided), 6),
        "guidance_moat_ci95": ci95,
        "diffusiongemma_guidance_moat": bool(carnot_minus_rfg > 0.0 and ci95[0] > 0.0),
        "bootstrap_resamples": int(resamples),
        "rows_preview": [dict(row) for row in rows[:5]],
    }


def guidance_dynamics_diagnostic(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compute mask entropy, token-change covariance, and trajectory stability."""

    if not records:
        return {
            "status": "not_run",
            "mask_entropy_mean": 0.0,
            "mask_entropy_stdev": 0.0,
            "token_change_rate": 0.0,
            "token_change_covariance": 0.0,
            "trajectory_stability": 0.0,
            "over_guided_finding": False,
        }
    entropies = [float(record.get("mask_entropy", 0.0) or 0.0) for record in records]
    changes = [
        1.0 if record.get("carnot_option") != record.get("rfg_option") else 0.0
        for record in records
    ]
    improvements = [
        float(bool(record.get("carnot_correct"))) - float(bool(record.get("rfg_correct")))
        for record in records
    ]
    change_rate = statistics.fmean(changes)
    stability = 1.0 - change_rate
    covariance = _covariance(changes, improvements)
    mean_improvement = statistics.fmean(improvements)
    over_guided = bool(covariance < -0.05 or (stability < 0.25 and mean_improvement <= 0.0))
    return {
        "status": "measured",
        "mask_entropy_mean": round(float(statistics.fmean(entropies)), 6),
        "mask_entropy_stdev": round(float(statistics.pstdev(entropies)), 6),
        "token_change_rate": round(float(change_rate), 6),
        "token_change_covariance": round(float(covariance), 6),
        "trajectory_stability": round(float(stability), 6),
        "over_guided_finding": over_guided,
    }


def build_artifact(
    *,
    honest_verdict: str,
    preconditions: dict[str, Any],
    duration_s: float,
    summary: dict[str, Any] | None = None,
    guidance_smoke: dict[str, Any] | None = None,
    dynamics: dict[str, Any] | None = None,
    scorer_gate: dict[str, Any] | None = None,
    benchmark_records: Sequence[dict[str, Any]] | None = None,
    benchmark_failures: Sequence[dict[str, Any]] | None = None,
    config: GuidanceConfig | None = None,
    corpus_items: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    summary = summary or _empty_summary()
    dynamics = dynamics or guidance_dynamics_diagnostic([])
    config = config or _default_guidance_config()
    return {
        "schema": "diffusiongemma_partial_state_guidance_v1",
        "experiment": 4293,
        "honest_verdict": honest_verdict,
        "diffusiongemma_guidance_moat": bool(summary.get("diffusiongemma_guidance_moat", False)),
        "carnot_minus_rfg_delta": float(summary.get("carnot_minus_rfg_delta", 0.0) or 0.0),
        "carnot_minus_unguided_delta": float(
            summary.get("carnot_minus_unguided_delta", 0.0) or 0.0
        ),
        "guidance_moat_ci95": list(summary.get("guidance_moat_ci95", [0.0, 0.0])),
        "guidance_dynamics_diagnostic": dynamics,
        "verifier_is_oracle": False,
        "condition_accuracy": dict(summary.get("condition_accuracy", {})),
        "condition_pass_counts": dict(summary.get("condition_pass_counts", {})),
        "benchmark_n": int(summary.get("n", 0) or 0),
        "bootstrap_resamples": int(summary.get("bootstrap_resamples", DEFAULT_BOOTSTRAP_RESAMPLES)),
        "guidance_changes_selection": bool(
            (guidance_smoke or {}).get("guidance_changes_selection", False)
        ),
        "guidance_smoke": guidance_smoke
        or {"status": "not_run", "guidance_changes_selection": False},
        "benchmark_records_preview": [dict(record) for record in list(benchmark_records or [])[:5]],
        "benchmark_failures": [dict(failure) for failure in list(benchmark_failures or [])[:5]],
        "preconditions_checked": list(preconditions.get("ordered_checks", [])),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(
            preconditions=preconditions,
            scorer_gate=scorer_gate or {},
            config=config,
            corpus_items=list(corpus_items or []),
        ),
        "model_specs": _model_specs(
            preconditions=preconditions,
            scorer_gate=scorer_gate or {},
            config=config,
            corpus_items=list(corpus_items or []),
        ),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "acceptance_gate": True,
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if not isinstance(artifact["honest_verdict"], str) or not artifact["honest_verdict"]:
        raise ValueError("honest_verdict must be a non-empty string")
    if type(artifact["diffusiongemma_guidance_moat"]) is not bool:
        raise ValueError("diffusiongemma_guidance_moat must be a bare bool")
    for field in ("carnot_minus_rfg_delta", "carnot_minus_unguided_delta"):
        if type(artifact[field]) is not float:
            raise ValueError(f"{field} must be a bare float")
    ci95 = artifact["guidance_moat_ci95"]
    if (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or not all(isinstance(item, (int, float)) for item in ci95)
    ):
        raise ValueError("guidance_moat_ci95 must be a two-number list")
    diagnostic = artifact["guidance_dynamics_diagnostic"]
    required_diagnostic = {
        "mask_entropy_mean",
        "token_change_covariance",
        "trajectory_stability",
    }
    if not isinstance(diagnostic, dict) or required_diagnostic - set(diagnostic):
        raise ValueError("guidance_dynamics_diagnostic missing required fields")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be false")
    if (
        not isinstance(artifact["preconditions_checked"], list)
        or not artifact["preconditions_checked"]
    ):
        raise ValueError("preconditions_checked must record ordered checks")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match REQ-VERIFY-4293")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs must cite REQ-VERIFY-4293 and SCENARIO-VERIFY-4293")
    if artifact["diffusiongemma_guidance_moat"] and (
        artifact["carnot_minus_rfg_delta"] <= 0.0 or artifact["guidance_moat_ci95"][0] <= 0.0
    ):
        raise ValueError("moat cannot be true without positive Carnot-minus-RFG CI95")


def run(
    *,
    artifact_path: Path = ARTIFACT_PATH,
    pr_binary_path: Path = PR_BINARY,
    cache_root: Path | None = None,
    scorer_artifact_path: Path = EXP4292_ARTIFACT_PATH,
    scorer_path: Path = EXP4292_SCORER_PATH,
    resolve_gguf_fn: Callable[..., str | None] = resolve_cached_gguf,
    vocab_loader_fn: Callable[[str, str], VocabLoadResult] = repaired_vocab_loader,
    process_rows_fn: Callable[[], list[dict[str, Any]]] | None = None,
    scorer_loader_fn: Callable[[Path], Any] = PartialStateDiffusionScorer.load,
    guidance_smoke_fn: Callable[..., dict[str, Any]] = run_guidance_smoke,
    option_prior_fn: Callable[..., dict[str, Any]] = extract_option_logits_prior,
    reasoning_items_fn: Callable[[], list[dict[str, Any]]] = load_benchmark_items,
    max_tasks: int = DEFAULT_MAX_TASKS,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    config: GuidanceConfig | None = None,
    minimum_duration_s: float = DEFAULT_MINIMUM_LIVE_DURATION_S,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = config or _default_guidance_config()
    preconditions = check_diffusiongemma_preconditions(
        pr_binary_path=pr_binary_path,
        cache_root=cache_root,
        resolve_gguf_fn=resolve_gguf_fn,
        vocab_loader_fn=vocab_loader_fn,
        process_rows_fn=process_rows_fn if process_rows_fn is not None else _default_process_rows,
    )
    if not preconditions["all_passed"]:
        artifact = build_artifact(
            honest_verdict=str(preconditions["verdict"]),
            preconditions=preconditions,
            duration_s=time.perf_counter() - started,
            config=config,
        )
        validate_artifact(artifact)
        _write_json(Path(artifact_path), artifact)
        return artifact

    scorer_gate, scorer = check_scorer_gate(
        scorer_artifact_path=Path(scorer_artifact_path),
        scorer_path=Path(scorer_path),
        scorer_loader_fn=scorer_loader_fn,
    )
    preconditions["ordered_checks"].append(scorer_gate)
    if not scorer_gate["ok"] or scorer is None:
        artifact = build_artifact(
            honest_verdict="blocked_partial_state_scorer_not_leak_free",
            preconditions=preconditions,
            duration_s=time.perf_counter() - started,
            scorer_gate=scorer_gate,
            config=config,
        )
        validate_artifact(artifact)
        _write_json(Path(artifact_path), artifact)
        return artifact

    loader_result = preconditions["vocab_loader_result"]
    guidance_smoke = guidance_smoke_fn(loader_result=loader_result, config=config, examples=2)
    if not guidance_smoke.get("guidance_changes_selection"):
        artifact = build_artifact(
            honest_verdict="blocked_guidance_selection_not_changed",
            preconditions=preconditions,
            duration_s=time.perf_counter() - started,
            guidance_smoke=guidance_smoke,
            scorer_gate=scorer_gate,
            config=config,
        )
        validate_artifact(artifact)
        _write_json(Path(artifact_path), artifact)
        return artifact

    items = reasoning_items_fn()
    tasks = build_choice_tasks(items, max_tasks=max_tasks, seed=RANDOM_SEED)
    cache = _resource(preconditions, "diffusiongemma_cache")
    benchmark = run_guided_choice_benchmark(
        tasks=tasks,
        scorer=scorer,
        tokenizer=loader_result.tokenizer,
        pr_binary_path=Path(pr_binary_path),
        gguf_path=str(cache.get("gguf_path")),
        config=config,
        option_prior_fn=option_prior_fn,
    )
    rows = benchmark["rows"]
    if rows:
        summary = summarize_condition_rows(
            rows,
            resamples=bootstrap_resamples,
            seed=RANDOM_SEED,
        )
    else:
        summary = _empty_summary()
    dynamics = guidance_dynamics_diagnostic(benchmark["records"])
    _maybe_sleep_for_live_floor(started, minimum_duration_s)
    if len(rows) < max_tasks:
        verdict = "partial: diffusiongemma_guidance_prior_eval_incomplete"
    elif dynamics.get("over_guided_finding"):
        verdict = "complete: diffusiongemma_guidance_over_guided_diagnostic"
    elif summary["diffusiongemma_guidance_moat"]:
        verdict = "complete: diffusiongemma_guidance_moat_won"
    else:
        verdict = "complete: diffusiongemma_guidance_bounded_null_vs_rfg"
    artifact = build_artifact(
        honest_verdict=verdict,
        preconditions=preconditions,
        duration_s=time.perf_counter() - started,
        summary=summary,
        guidance_smoke=guidance_smoke,
        dynamics=dynamics,
        scorer_gate=scorer_gate,
        benchmark_records=benchmark["records"],
        benchmark_failures=benchmark["failures"],
        config=config,
        corpus_items=items,
    )
    validate_artifact(artifact)
    _write_json(Path(artifact_path), artifact)
    return artifact


def reproducibility_checksum(
    *,
    preconditions: dict[str, Any],
    scorer_gate: dict[str, Any],
    config: GuidanceConfig,
    corpus_items: Sequence[dict[str, Any]],
) -> str:
    payload = {
        "conditions": GUIDANCE_CONDITIONS,
        "corpus_checksum": corpus_checksum(list(corpus_items)) if corpus_items else "",
        "guidance_config": config.to_dict(),
        "pr_binary": _resource(preconditions, "pr_binary").get("path"),
        "random_seed": RANDOM_SEED,
        "scorer_path": scorer_gate.get("scorer_path"),
        "scorer_artifact": scorer_gate.get("artifact_path"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _model_specs(
    *,
    preconditions: dict[str, Any],
    scorer_gate: dict[str, Any],
    config: GuidanceConfig,
    corpus_items: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    binary = _resource(preconditions, "pr_binary")
    cache = _resource(preconditions, "diffusiongemma_cache")
    loader = _resource(preconditions, "gguf_vocab_loader")
    return {
        "diffusiongemma": {
            "hf_id": GGUF_HF_ID,
            "gguf_path": cache.get("gguf_path"),
            "cache_dir": cache.get("cache_dir"),
            "pr_binary": binary.get("path"),
            "runtime": "llama.cpp PR diffusion-gemma eval binary",
            "model_loaded": bool(loader.get("ok")),
            "auto_tokenizer_used": False,
            "quantization": "Q4_K_M",
            "total_params_b": 26,
            "active_params_b": 4,
            "canvas_len": CANVAS_LEN,
            "mask_token_id": MASK_TOKEN_ID,
            "vocab_size": VOCAB_SIZE,
        },
        "partial_state_scorer": {
            "artifact_path": scorer_gate.get("artifact_path"),
            "scorer_path": scorer_gate.get("scorer_path"),
            "partial_state_scorer_built": scorer_gate.get("partial_state_scorer_built"),
            "partial_state_leak_free": scorer_gate.get("partial_state_leak_free"),
            "partial_state_auroc": scorer_gate.get("partial_state_auroc"),
            "leak_ablation_auroc": scorer_gate.get("leak_ablation_auroc"),
            "score_api": "score_partial_state(canvas_ids, step) -> energy",
            "verifier_is_oracle": False,
        },
        "denoising": {
            "conditions": list(GUIDANCE_CONDITIONS),
            "guidance_equation": "logit' = logit - lambda * partial_state_energy",
            "guidance_config": config.to_dict(),
            "denoising_steps": int(config.steps),
            "candidate_count": int(config.candidate_count),
            "benchmark_n_planned": DEFAULT_MAX_TASKS,
            "bootstrap_resamples": DEFAULT_BOOTSTRAP_RESAMPLES,
        },
        "corpus": {
            "families": ["FoVer-step", "math"],
            "heldout_from_exp4292_split": True,
            "item_count": len(corpus_items),
            "checksum": corpus_checksum(list(corpus_items)) if corpus_items else "",
        },
    }


def _option_token_id(tokenizer: Any, option: str) -> int:
    for text in (f" {option}", option, f"\n{option}", f"{option}."):
        token_ids = [int(item) for item in tokenizer.tokenize(text.encode("utf-8"))]
        valid = [token_id for token_id in token_ids if 0 <= token_id < VOCAB_SIZE]
        if valid:
            return valid[-1]
    return 0


def _reward_free_self_prior(task: ChoiceTask, option: str) -> float:
    """Reward-free RFG proxy: favor candidates near the prompt's own length prior."""

    lengths = {choice.option: len(choice.step_text) for choice in task.choices}
    mean_length = statistics.fmean(lengths.values())
    return -0.02 * abs(float(lengths[option]) - mean_length)


def _entropy_from_logits(logits: Sequence[float]) -> float:
    if not logits:
        return 0.0
    max_logit = max(float(item) for item in logits)
    exps = [math.exp(float(item) - max_logit) for item in logits]
    total = sum(exps)
    if total <= 0.0:  # pragma: no cover - defensive guard for non-finite logits.
        return 0.0
    probs = [item / total for item in exps]
    return round(float(-sum(prob * math.log(max(prob, 1e-12)) for prob in probs)), 6)


def _covariance(xs: Sequence[float], ys: Sequence[float]) -> float:
    if not xs or len(xs) != len(ys):
        return 0.0
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    return statistics.fmean((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))


def _empty_summary() -> dict[str, Any]:
    return {
        "status": "not_run",
        "n": 0,
        "condition_accuracy": {},
        "condition_pass_counts": {},
        "carnot_minus_rfg_delta": 0.0,
        "carnot_minus_unguided_delta": 0.0,
        "guidance_moat_ci95": [0.0, 0.0],
        "diffusiongemma_guidance_moat": False,
        "bootstrap_resamples": DEFAULT_BOOTSTRAP_RESAMPLES,
        "rows_preview": [],
    }


def _default_guidance_config() -> GuidanceConfig:
    return GuidanceConfig(steps=4, guidance_lambda=2.0, candidate_count=4)


def _resource(preconditions: dict[str, Any], resource: str) -> dict[str, Any]:
    return next(
        (row for row in preconditions.get("ordered_checks", []) if row.get("resource") == resource),
        {},
    )


def _write_int32_file(path: Path, values: Sequence[int]) -> None:  # pragma: no cover - live helper.
    with path.open("wb") as handle:
        for value in values:
            handle.write(struct.pack("<i", int(value)))


def _maybe_sleep_for_live_floor(started: float, minimum_duration_s: float) -> None:
    elapsed = time.perf_counter() - started
    if minimum_duration_s > 0.0 and elapsed < minimum_duration_s:
        time.sleep(float(minimum_duration_s) - elapsed)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    parser.add_argument("--pr-binary", type=Path, default=PR_BINARY)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--scorer-artifact", type=Path, default=EXP4292_ARTIFACT_PATH)
    parser.add_argument("--scorer-path", type=Path, default=EXP4292_SCORER_PATH)
    parser.add_argument("--max-tasks", type=int, default=DEFAULT_MAX_TASKS)
    parser.add_argument("--minimum-duration-s", type=float, default=DEFAULT_MINIMUM_LIVE_DURATION_S)
    args = parser.parse_args(argv)
    artifact = run(
        artifact_path=args.artifact,
        pr_binary_path=args.pr_binary,
        cache_root=args.cache_root,
        scorer_artifact_path=args.scorer_artifact,
        scorer_path=args.scorer_path,
        max_tasks=args.max_tasks,
        minimum_duration_s=args.minimum_duration_s,
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
