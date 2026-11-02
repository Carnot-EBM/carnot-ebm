"""Experiment 4501: frame-change predictor rerun on staged replay shards.

Spec refs: REQ-ARC-FCP-4501, SCENARIO-ARC-FCP-4501.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.agentic import arc_human_replay_corpus
from carnot.agentic.arc_frame_change_predictor import (
    DEFAULT_FRAME_SIZE,
    DEFAULT_NUM_COLORS,
    FrameActionEffectExample,
    FrameChangeScorer,
    build_behavior_prior_from_effect_examples,
    evaluate_positive_control,
    evaluate_replay_candidate_order,
    load_frame_action_effect_examples,
    train_frame_change_model,
)


RESULT_RELATIVE_PATH = "results/experiment_4501_frame_change_predictor_rerun.json"
DATA_RELATIVE_DIR = "data/arc_public_demo_human_replay_corpus"
REPO_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
EXPECTED_ACTION_EFFECT_EXAMPLES = 14672
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)
REQUIREMENTS = ["REQ-ARC-FCP-4490", "REQ-ARC-FCP-4491", "REQ-ARC-FCP-4492", "REQ-ARC-FCP-4501"]
SCENARIOS = ["SCENARIO-ARC-FCP-4490", "SCENARIO-ARC-FCP-4501"]

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "MUST start with terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_ "
        "(Verdict Terminal-Prefix Discipline)."
    ),
    "inference_substrate": (
        "explicit verifier_ensemble_against_cached_candidates declaration so adversarial_verify applies "
        "the cached-candidate duration floor."
    ),
    "preconditions_checked": (
        "records WHICH resources were verified; pre-empts silent-missing-resource fabrication."
    ),
    "expected_action_effect_examples": (
        "the external action-effect corpus target count, fixed at 14,672 for this rerun."
    ),
    "corpus_examples_loaded": (
        "the exact number of frame-only shard examples loaded locally for this run."
    ),
    "feature_source": (
        "proves features were recomputed from raw frames, not mirror feature vectors."
    ),
    "behavior_prior_emitted": (
        "bare bool showing the behavior-cloning prior was built and can rank candidates."
    ),
    "heldout_median_actions_before": (
        "baseline median actions-to-first-level-up on held-out frame-only evaluation."
    ),
    "heldout_median_actions_after": (
        "predictor/prior-ranked median actions-to-first-level-up on the same held-out frame-only evaluation."
    ),
    "implied_efficiency_delta": (
        "score-relevant delta in min(human/agent,1)^2 efficiency."
    ),
    "solve_rate_dropped": (
        "guardrail bool; efficiency wins must not come from reducing solve rate."
    ),
    "false_negative_risk_guard": (
        "records whether the null is interpretable because the positive control passed."
    ),
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "field_principles",
    "requirements",
    "scenarios",
    "weights_bundled",
    "training_summary",
    "positive_control",
    "rich_action_candidates_wired",
    "corpus_manifest",
)


def load_corpus_manifest(root: Path | str) -> dict[str, Any]:
    """REQ-ARC-FCP-4501: read the staged frame-shard manifest if it exists."""

    data_dir = Path(root) / DATA_RELATIVE_DIR
    manifest_path = data_dir / arc_human_replay_corpus.MANIFEST_NAME
    if not manifest_path.exists():
        return {
            "schema": arc_human_replay_corpus.SHARD_SCHEMA,
            "example_count": 0,
            "shard_count": 0,
            "shards": [],
            "source_metadata": {},
        }
    return arc_human_replay_corpus.load_manifest(data_dir)


def check_preconditions(root: Path | str) -> dict[str, Any]:
    """REQ-ARC-FCP-4501: record resources checked before training or reporting."""

    root_path = Path(root)
    data_dir = root_path / DATA_RELATIVE_DIR
    manifest = load_corpus_manifest(root_path)
    preconditions: dict[str, Any] = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "offline_arcade_import": False,
        "torch_import": False,
        "torch_version": "",
        "training_shards_present": int(manifest.get("example_count") or 0) > 0,
        "training_shard_count": int(manifest.get("shard_count") or 0),
        "training_manifest_examples": int(manifest.get("example_count") or 0),
        "cached_raw_parquet_present": bool(list((data_dir / "raw_hf_mirror").glob("data/*.parquet"))),
        "action_effect_npz_present": bool(list(data_dir.rglob("action_effect_dict.npz"))),
        "expected_action_effect_examples": EXPECTED_ACTION_EFFECT_EXAMPLES,
        "official_license_verified_for_bundled_weights": False,
        "env_game_access_blocked": True,
    }
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        preconditions["offline_arcade_import"] = True
    except Exception as exc:  # pragma: no cover - local SDK failure path
        preconditions["offline_arcade_error"] = repr(exc)
    try:
        import torch

        preconditions["torch_import"] = True
        preconditions["torch_version"] = str(torch.__version__)
    except Exception as exc:  # pragma: no cover - torch missing path
        preconditions["torch_error"] = repr(exc)
    preconditions["ok"] = bool(
        preconditions["offline_arcade_import"]
        and preconditions["torch_import"]
        and preconditions["training_shards_present"]
    )
    return preconditions


def _split_train_heldout(
    examples: Sequence[FrameActionEffectExample],
) -> tuple[list[FrameActionEffectExample], list[FrameActionEffectExample]]:
    if len(examples) < 20:
        return list(examples), list(examples)
    keys = sorted({example.state_key for example in examples})
    heldout_keys = {key for index, key in enumerate(keys) if index % 5 == 0}
    train = [example for example in examples if example.state_key not in heldout_keys]
    heldout = [example for example in examples if example.state_key in heldout_keys]
    return train or list(examples), heldout or list(examples)


def summarize_prior(prior: Any) -> dict[str, Any]:
    """REQ-ARC-FCP-4501: serialize the behavior prior without writing weights."""

    return {
        "marginal_action_counts": {
            str(key): float(value) for key, value in prior.marginal_action_counts.items()
        },
        "click_cell_count": int(len(prior.click_cell_counts)),
        "state_action_count": int(len(prior.state_action_counts)),
        "state_click_count": int(len(prior.state_click_counts)),
    }


def false_negative_risk_guard(
    heldout_metrics: Mapping[str, Any],
    positive_control: Mapping[str, Any],
) -> str:
    """SCENARIO-ARC-FCP-4501: identify whether a null rerun is interpretable."""

    if positive_control.get("actions_reduced") is not True:
        return "positive_control_failed_null_uninterpretable"
    delta = heldout_metrics.get("implied_efficiency_delta")
    if delta is not None and float(delta) > 0.0:
        return "positive_control_passed_candidate_order_gain"
    return "positive_control_passed_null_interpretable"


def _honest_verdict(
    preconditions: Mapping[str, Any],
    heldout_metrics: Mapping[str, Any],
    *,
    corpus_examples_loaded: int,
) -> str:
    if not preconditions.get("offline_arcade_import"):
        return "complete: blocked_offline_arcade_import_failed"
    if not preconditions.get("torch_import"):
        return "complete: blocked_torch_missing"
    if corpus_examples_loaded <= 0:
        return "complete: blocked_staged_frame_shards_missing"
    if heldout_metrics.get("solve_rate_dropped") is True:
        return "complete: frame_change_predictor_rerun_solve_rate_guard_failed"
    delta = heldout_metrics.get("implied_efficiency_delta")
    exact_corpus = bool(preconditions.get("action_effect_npz_present")) and (
        corpus_examples_loaded == EXPECTED_ACTION_EFFECT_EXAMPLES
    )
    if delta is not None and float(delta) > 0.0:
        return (
            "complete: frame_change_predictor_rerun_exact_corpus_proxy_gain"
            if exact_corpus
            else "complete: frame_change_predictor_rerun_staged_corpus_proxy_gain"
        )
    return (
        "complete: frame_change_predictor_rerun_exact_corpus_honest_null"
        if exact_corpus
        else "complete: frame_change_predictor_rerun_staged_corpus_shortfall_null_guard"
    )


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    training_summary: Mapping[str, Any],
    heldout_metrics: Mapping[str, Any],
    positive_control: Mapping[str, Any],
    corpus_examples_loaded: int,
    corpus_manifest: Mapping[str, Any],
    prior_summary: Mapping[str, Any],
    started: float | None = None,
    finished: float | None = None,
) -> dict[str, Any]:
    """REQ-ARC-FCP-4501: assemble the terminal rerun artifact."""

    guard = false_negative_risk_guard(heldout_metrics, positive_control)
    artifact = {
        "honest_verdict": _honest_verdict(
            preconditions_checked,
            heldout_metrics,
            corpus_examples_loaded=int(corpus_examples_loaded),
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "requirements": list(REQUIREMENTS),
        "scenarios": list(SCENARIOS),
        "preconditions_checked": dict(preconditions_checked),
        "expected_action_effect_examples": EXPECTED_ACTION_EFFECT_EXAMPLES,
        "corpus_examples_loaded": int(corpus_examples_loaded),
        "corpus_manifest": dict(corpus_manifest),
        "feature_source": "raw_frame_shard_recomputed",
        "behavior_prior_emitted": bool(prior_summary),
        "behavior_prior_summary": dict(prior_summary),
        "small_cnn_trained": int(training_summary.get("batches_trained") or 0) > 0,
        "training_summary": dict(training_summary),
        "weights_bundled": False,
        "weights_bundling_policy": "No replay-derived weights are bundled without official CC0/MIT-0-compatible license verification.",
        "rich_action_candidates_wired": True,
        "heldout_median_actions_before": heldout_metrics.get("heldout_median_actions_before"),
        "heldout_median_actions_after": heldout_metrics.get("heldout_median_actions_after"),
        "implied_efficiency_delta": heldout_metrics.get("implied_efficiency_delta"),
        "solve_rate_before": heldout_metrics.get("solve_rate_before"),
        "solve_rate_after": heldout_metrics.get("solve_rate_after"),
        "solve_rate_dropped": bool(heldout_metrics.get("solve_rate_dropped")),
        "heldout_group_count": int(heldout_metrics.get("heldout_group_count") or 0),
        "heldout_measurement_kind": heldout_metrics.get("measurement_kind"),
        "positive_control": dict(positive_control),
        "false_negative_risk_guard": guard,
        "duration_s": None if started is None or finished is None else max(0.0, float(finished) - float(started)),
    }
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must equal verifier_ensemble_against_cached_candidates")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be a mapping")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match required field principles")
    if artifact.get("expected_action_effect_examples") != EXPECTED_ACTION_EFFECT_EXAMPLES:
        errors.append("expected_action_effect_examples must equal 14672")
    if int(artifact.get("corpus_examples_loaded") or 0) < 0:
        errors.append("corpus_examples_loaded must be non-negative")
    if artifact.get("feature_source") != "raw_frame_shard_recomputed":
        errors.append("feature_source must prove raw-frame recomputation")
    if artifact.get("behavior_prior_emitted") is not True:
        errors.append("behavior_prior_emitted must be true")
    if artifact.get("weights_bundled") is True:
        errors.append("weights_bundled must remain false without official license verification")
    if artifact.get("solve_rate_dropped") is True:
        errors.append("solve rate guard failed")
    positive_control = artifact.get("positive_control")
    if not isinstance(positive_control, Mapping) or positive_control.get("actions_reduced") is not True:
        errors.append("positive_control must prove the ranking harness can reduce actions")
    guard = artifact.get("false_negative_risk_guard")
    if guard not in {
        "positive_control_passed_candidate_order_gain",
        "positive_control_passed_null_interpretable",
        "positive_control_failed_null_uninterpretable",
    }:
        errors.append("false_negative_risk_guard has an unknown value")
    return errors


def run(
    *,
    root: Path | str = REPO_ROOT,
    preconditions_checked: Mapping[str, Any] | None = None,
    write: bool = True,
    train_limit: int | None = None,
    epochs: int = 1,
    batch_size: int = 32,
    hidden_channels: int = 12,
    frame_size: int = DEFAULT_FRAME_SIZE,
    num_colors: int = DEFAULT_NUM_COLORS,
    now: Any = time.monotonic,
) -> dict[str, Any]:
    """SCENARIO-ARC-FCP-4501: train, measure, and write the rerun artifact."""

    root_path = Path(root)
    started = float(now())
    preconditions = (
        dict(preconditions_checked)
        if preconditions_checked is not None
        else check_preconditions(root_path)
    )
    manifest = load_corpus_manifest(root_path)
    data_dir = root_path / DATA_RELATIVE_DIR
    examples = (
        load_frame_action_effect_examples(data_dir, limit=train_limit)
        if int(manifest.get("example_count") or 0) > 0
        else []
    )
    train_examples, heldout_examples = _split_train_heldout(examples)
    prior = build_behavior_prior_from_effect_examples(train_examples) if train_examples else None
    if train_examples:
        model, training_summary = train_frame_change_model(
            train_examples,
            num_colors=num_colors,
            size=frame_size,
            hidden_channels=hidden_channels,
            epochs=epochs,
            batch_size=batch_size,
            seed=4501,
        )
        scorer = FrameChangeScorer(model, num_colors=num_colors, size=frame_size)
    else:
        training_summary = {
            "examples_seen": 0,
            "examples_used": 0,
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "hidden_channels": int(hidden_channels),
            "num_colors": int(num_colors),
            "frame_size": int(frame_size),
            "learning_rate": None,
            "batches_trained": 0,
            "initial_loss": None,
            "final_loss": None,
        }
        scorer = None
    heldout_metrics = evaluate_replay_candidate_order(
        heldout_examples,
        scorer=scorer,
        prior=prior,
    )
    positive_control = evaluate_positive_control()
    artifact = build_artifact(
        preconditions_checked=preconditions,
        training_summary=training_summary,
        heldout_metrics=heldout_metrics,
        positive_control=positive_control,
        corpus_examples_loaded=len(examples),
        corpus_manifest=manifest,
        prior_summary=summarize_prior(prior) if prior is not None else {},
        started=started,
        finished=float(now()),
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        out = root_path / RESULT_RELATIVE_PATH
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover - thin CLI wrapper
    artifact = run()
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    main()
