"""Exp 4223 B1-gated verifier-as-reward 3-arm synchronous finish.

Spec refs: REQ-CODE-4223, SCENARIO-CODE-4223-DEFERRED-HARNESS,
SCENARIO-CODE-4223-SYNC-ACCUMULATE, SCENARIO-CODE-4223-VERDICT-GATES.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from carnot import experiment_4211_verifier_as_reward_finish_synchronous as exp4211


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_FILENAME = "experiment_4223_verifier_as_reward_3arm_synchronous.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_HARNESS_SMOKE = REPO_ROOT / "results" / "experiment_4222_verifier_reward_lora_harness_fix_smoke.json"
DEFAULT_STABLE_CHECKPOINT = exp4211.DEFAULT_STABLE_CHECKPOINT
DEFAULT_LAUNCH_ARTIFACT = exp4211.DEFAULT_LAUNCH_ARTIFACT
RANDOM_SEED = exp4211.RANDOM_SEED
SPEC_REFS = [
    "REQ-CODE-4223",
    "SCENARIO-CODE-4223-DEFERRED-HARNESS",
    "SCENARIO-CODE-4223-SYNC-ACCUMULATE",
    "SCENARIO-CODE-4223-VERDICT-GATES",
]
REQUIRED_ARTIFACT_FIELDS = exp4211.REQUIRED_ARTIFACT_FIELDS
FIELD_PRINCIPLES = {
    **exp4211.FIELD_PRINCIPLES,
    "model_specs": (
        "The NON-Qwen base + the B1-proven LoRA config + the on-policy generator; required "
        "methodology for a live-LLM training artifact."
    ),
}


def load_harness_smoke_artifact(path: str | Path) -> dict[str, Any]:
    payload = exp4211.load_json(path)
    if not isinstance(payload.get("model_specs"), Mapping):
        payload["model_specs"] = {}
    return payload


def b1_harness_smoked(smoke: Mapping[str, Any]) -> bool:
    model_specs = smoke.get("model_specs") if isinstance(smoke.get("model_specs"), Mapping) else {}
    return bool(
        smoke.get("harness_smoke_passed") is True
        and smoke.get("lora_attach_path")
        and isinstance(model_specs.get("lora_config"), Mapping)
    )


def _b1_lora_config(smoke: Mapping[str, Any]) -> dict[str, Any]:
    model_specs = smoke.get("model_specs") if isinstance(smoke.get("model_specs"), Mapping) else {}
    lora_config = model_specs.get("lora_config") if isinstance(model_specs.get("lora_config"), Mapping) else {}
    return exp4211._jsonable(dict(lora_config))


def _b1_lora_attach_path(smoke: Mapping[str, Any]) -> str:
    model_specs = smoke.get("model_specs") if isinstance(smoke.get("model_specs"), Mapping) else {}
    return str(smoke.get("lora_attach_path") or model_specs.get("lora_attach_path") or "")


def reproducibility_checksum(
    *,
    corpus_paths: Mapping[str, Path],
    lora_config: Mapping[str, Any],
    lora_attach_path: str,
    random_seed: int,
    harness_checksum: str | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(exp4211._jsonable(lora_config), sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(str(lora_attach_path).encode("utf-8"))
    digest.update(str(int(random_seed)).encode("utf-8"))
    digest.update(str(harness_checksum or "").encode("utf-8"))
    for arm in sorted(corpus_paths):
        digest.update(str(arm).encode("utf-8"))
        digest.update(Path(corpus_paths[arm]).read_bytes())
    return f"sha256:{digest.hexdigest()}"


def _apply_b1_lora_config(
    manifest: Mapping[str, Any],
    smoke: Mapping[str, Any],
    *,
    corpus_paths: Mapping[str, Path],
    random_seed: int,
) -> dict[str, Any]:
    merged = dict(manifest)
    raw_model_specs = merged.get("model_specs") if isinstance(merged.get("model_specs"), Mapping) else {}
    smoke_model_specs = smoke.get("model_specs") if isinstance(smoke.get("model_specs"), Mapping) else {}
    lora_config = _b1_lora_config(smoke)
    lora_attach_path = _b1_lora_attach_path(smoke)
    model_specs = dict(raw_model_specs)
    for key in ("on_policy_generator", "qwen_train_base_forbidden", "trainable_base", "trainable_base_is_non_qwen"):
        if key in smoke_model_specs:
            model_specs[key] = smoke_model_specs[key]
    model_specs["lora_config"] = lora_config
    model_specs["lora_attach_path"] = lora_attach_path
    model_specs["b1_harness_smoke_artifact"] = str(DEFAULT_HARNESS_SMOKE)
    merged["model_specs"] = model_specs
    merged["lora_config"] = lora_config
    merged["lora_attach_path"] = lora_attach_path
    merged["b1_harness_smoke_passed"] = True
    merged["b1_harness_reproducibility_checksum"] = str(smoke.get("reproducibility_checksum") or "")
    merged["reproducibility_checksum"] = reproducibility_checksum(
        corpus_paths=corpus_paths,
        lora_config=lora_config,
        lora_attach_path=lora_attach_path,
        random_seed=random_seed,
        harness_checksum=smoke.get("reproducibility_checksum") if isinstance(smoke.get("reproducibility_checksum"), str) else None,
    )
    return exp4211._jsonable(merged)


def load_b1_checkpoint_context(
    stable_checkpoint_path: str | Path,
    smoke: Mapping[str, Any],
    *,
    random_seed: int,
    launch_artifact_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Path], dict[str, int]]:
    manifest, corpus_paths, corpus_sizes = exp4211.load_checkpoint_context(stable_checkpoint_path)
    if launch_artifact_path is not None:
        manifest = exp4211._merge_launch_metadata(manifest, launch_artifact_path)
    manifest = _apply_b1_lora_config(
        manifest,
        smoke,
        corpus_paths=corpus_paths,
        random_seed=random_seed,
    )
    return manifest, corpus_paths, corpus_sizes


def _empty_artifact_inputs(stable_checkpoint_path: str | Path) -> tuple[dict[str, Any], dict[str, int]]:
    return {}, {"A": 0, "B": 0, "C": 0, "D": 0}


def build_result_artifact(
    *,
    preconditions: Mapping[str, Any],
    stable_checkpoint_path: Path,
    manifest: Mapping[str, Any],
    corpus_sizes: Mapping[str, int],
    cached_base: exp4211.CachedBase | None,
    training: exp4211.TrainingOutcome,
    evaluation: exp4211.EvaluationOutcome | None,
    adversarial_report: Mapping[str, Any] | None,
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    artifact = exp4211.build_result_artifact(
        preconditions=preconditions,
        stable_checkpoint_path=stable_checkpoint_path,
        manifest=manifest,
        corpus_sizes=corpus_sizes,
        cached_base=cached_base,
        training=training,
        evaluation=evaluation,
        adversarial_report=adversarial_report,
        random_seed=random_seed,
        duration_s=duration_s,
    )
    artifact.update(
        {
            "experiment": "experiment_4223_verifier_as_reward_3arm_synchronous",
            "schema": "carnot.experiment_4223_verifier_as_reward_3arm_synchronous.v1",
            "field_principles": FIELD_PRINCIPLES,
            "spec_refs": SPEC_REFS,
        }
    )
    if manifest.get("reproducibility_checksum"):
        artifact["reproducibility_checksum"] = str(manifest["reproducibility_checksum"])

    if not preconditions.get("b1_harness_smoke_passed", False):
        artifact["honest_verdict"] = "complete_verifier_reward_deferred_harness_not_smoked"
    elif not preconditions.get("nonqwen_base_cached", True):
        artifact["honest_verdict"] = "blocked_no_nonqwen_base_cached"
    elif not preconditions.get("cuda_available", True):
        artifact["honest_verdict"] = "blocked_cuda_unavailable"
    elif not preconditions.get("stable_checkpoint_readable", True):
        artifact["honest_verdict"] = "blocked_stable_checkpoint_unreadable"
    elif not preconditions.get("arms_n_matched", True):
        artifact["honest_verdict"] = "blocked_size_matched_random_label_control_missing"

    artifact["acceptance_gate"] = {
        "condition": (
            "positive_control_confirmed true AND a_vs_b_delta + a_vs_b_ci95 reported "
            "(verifier_label_carries_signal resolved), OR an honest accumulating/invalid/retired/deferred verdict"
        ),
        "satisfied": bool(
            (
                artifact.get("positive_control_confirmed") is True
                and artifact.get("a_vs_b_delta") is not None
                and artifact.get("a_vs_b_ci95") is not None
            )
            or str(artifact.get("honest_verdict", "")).startswith(
                (
                    "blocked_",
                    "progress:",
                    "invalid:",
                    "complete:",
                    "complete_verifier_reward_deferred",
                    "complete_verifier_reward_retired",
                )
            )
        ),
    }
    return exp4211._jsonable(artifact)


def write_artifact(artifact: Mapping[str, Any], path: str | Path = DEFAULT_OUTPUT) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(exp4211._jsonable(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def run(
    *,
    output_path: str | Path = DEFAULT_OUTPUT,
    stable_checkpoint_path: str | Path = DEFAULT_STABLE_CHECKPOINT,
    harness_smoke_path: str | Path = DEFAULT_HARNESS_SMOKE,
    launch_artifact_path: str | Path = DEFAULT_LAUNCH_ARTIFACT,
    random_seed: int = RANDOM_SEED,
    progress_interval_s: float = 30.0,
    cuda_probe: Callable[[], bool] = exp4211._cuda_is_available,
    cached_base_callback: Callable[[], exp4211.CachedBase | None] = exp4211.find_cached_nonqwen_base,
    train_callback: Callable[[exp4211.TrainingContext], exp4211.TrainingOutcome] | None = None,
    eval_callback: Callable[[Path], exp4211.EvaluationOutcome] = exp4211.load_eval_if_available,
) -> dict[str, Any]:
    started = time.time()
    stable = Path(stable_checkpoint_path)
    manifest, corpus_sizes = _empty_artifact_inputs(stable)
    corpus_paths: dict[str, Path] = {}
    cached_base: exp4211.CachedBase | None = None
    training = exp4211._empty_training(stable)
    evaluation = exp4211._empty_evaluation(status="not_run")
    preconditions: dict[str, Any] = {
        "b1_harness_smoke_path": str(harness_smoke_path),
        "stable_checkpoint_path": str(stable),
    }

    smoke: dict[str, Any] = {}
    try:
        smoke = load_harness_smoke_artifact(harness_smoke_path)
        preconditions["b1_harness_smoke_readable"] = True
    except Exception as exc:
        preconditions["b1_harness_smoke_readable"] = False
        preconditions["b1_harness_smoke_error"] = f"{type(exc).__name__}: {exc}"
    preconditions["b1_harness_smoke_passed"] = b1_harness_smoked(smoke)
    preconditions["b1_lora_attach_path"] = _b1_lora_attach_path(smoke)

    if preconditions["b1_harness_smoke_passed"]:
        cached_base = cached_base_callback()
        preconditions["nonqwen_base_cached"] = cached_base is not None
        preconditions["cached_base"] = exp4211._jsonable(cached_base)
        preconditions["cuda_available"] = bool(cuda_probe())
        try:
            manifest, corpus_paths, corpus_sizes = load_b1_checkpoint_context(
                stable,
                smoke,
                random_seed=random_seed,
                launch_artifact_path=launch_artifact_path,
            )
            preconditions["stable_checkpoint_readable"] = True
            preconditions["arm_corpus_sizes"] = dict(corpus_sizes)
            preconditions["arms_n_matched"] = corpus_sizes.get("A", 0) > 0 and corpus_sizes.get("A") == corpus_sizes.get("B")
        except Exception as exc:
            manifest, corpus_sizes = _empty_artifact_inputs(stable)
            preconditions["stable_checkpoint_readable"] = False
            preconditions["stable_checkpoint_error"] = f"{type(exc).__name__}: {exc}"
            preconditions["arms_n_matched"] = False

        if (
            cached_base is not None
            and preconditions.get("cuda_available")
            and preconditions.get("stable_checkpoint_readable")
            and preconditions.get("arms_n_matched")
        ):
            context = exp4211.TrainingContext(
                stable_checkpoint_path=stable,
                manifest=manifest,
                corpus_paths=corpus_paths,
                corpus_sizes=corpus_sizes,
                cached_base=cached_base,
                random_seed=random_seed,
                mode="in_process",
                progress_interval_s=progress_interval_s,
            )
            try:
                training = (train_callback or exp4211.train_in_process)(context)
            except Exception as exc:
                training = exp4211.TrainingOutcome(
                    status="failed",
                    per_arm={},
                    accumulated_train_examples={"A": 0, "B": 0, "C": 0, "D": 0},
                    runner_artifact_path=stable / "runner_artifact.json",
                    progress_events=[],
                    used_detached_process=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            evaluation = eval_callback(stable)

    artifact = build_result_artifact(
        preconditions=preconditions,
        stable_checkpoint_path=stable,
        manifest=manifest,
        corpus_sizes=corpus_sizes,
        cached_base=cached_base,
        training=training,
        evaluation=evaluation,
        adversarial_report=None,
        random_seed=random_seed,
        duration_s=time.time() - started,
    )
    write_artifact(artifact, output_path)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stable-checkpoint", type=Path, default=DEFAULT_STABLE_CHECKPOINT)
    parser.add_argument("--harness-smoke", type=Path, default=DEFAULT_HARNESS_SMOKE)
    parser.add_argument("--launch-artifact", type=Path, default=DEFAULT_LAUNCH_ARTIFACT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--progress-interval-s", type=float, default=30.0)
    args = parser.parse_args(argv)
    artifact = run(
        output_path=args.out,
        stable_checkpoint_path=args.stable_checkpoint,
        harness_smoke_path=args.harness_smoke,
        launch_artifact_path=args.launch_artifact,
        random_seed=args.seed,
        progress_interval_s=args.progress_interval_s,
    )
    print(f"-> {artifact['honest_verdict']}", flush=True)
    print(f"   verifier_label_carries_signal={artifact['verifier_label_carries_signal']}", flush=True)
    print(f"   a_vs_b_delta={artifact['a_vs_b_delta']} ci95={artifact['a_vs_b_ci95']}", flush=True)
    print(f"   accumulated_n={artifact['accumulated_n']}", flush=True)
    return 0 if artifact["acceptance_gate"]["satisfied"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
