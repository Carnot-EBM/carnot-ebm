"""Exp 5030 — ship the reusable moat-trainer module + de-risk D1 with a smoke.

This is the PERMANENT fix for the headline (D1) that bailed twice:

  * `.461` skeleton-first (0 pairs, never trained).
  * `.462` on a 404 — the planner named ``Qwen/Qwen3.5-1.7B``, which does NOT
    exist on HuggingFace, so a single hallucinated repo id blocked training.

The fix is ``python/carnot/moat_trainer.py``: a ``resolve_trainable_base`` that
probes REAL cached bases (so a wrong id can never block again) plus a
``train_energy_head`` / ``score_candidates`` pipeline D1 imports instead of
re-deriving.  This experiment proves that pipeline end-to-end with a 60-second
smoke on a real base (resolve -> train a few steps on FoVer pairs on conductor
GPU 0 -> checkpoint -> reload -> score), so D1 is near-guaranteed to execute.

Spec: REQ-VERIFY-5030, SCENARIO-VERIFY-5030.

Run: ``.venv/bin/python python/carnot/experiment_5030_moat_trainer_module.py``
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from carnot import moat_trainer

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO_ROOT / "results" / "experiment_5030_moat_trainer_module.json"
FOVER_PATH = REPO_ROOT / "data" / "fover_train_v4.json"
CHECKPOINT_DIR = REPO_ROOT / "results" / "checkpoints" / "experiment_5030_moat_smoke"
RANDOM_SEED = 20260630

# Smoke sizing.  We size the smoke so genuine GPU compute (two full base loads
# + this many contrastive steps) clears the 60s anti-skeleton floor WITHOUT any
# sleep-padding — the duration is real training work, not a stall.
SMOKE_MAX_PAIRS = 64
SMOKE_EPOCHS = 2
SMOKE_BATCH_SIZE = 2
SMOKE_MAX_LENGTH = 256

# Two oracle-distinct candidates to score after training (good vs bad reasoning).
SMOKE_CANDIDATES = (
    "Tom had 12 apples and gave away 4, so he has 12 - 4 = 8 apples left.",
    "Tom had 12 apples and gave away 4, so he has 12 + 4 = 16 apples left.",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; success_moat_trainer_module_shipped_smoke_trained_<base>; "
        "a non-training smoke is blocked_smoke_train_did_not_run."
    ),
    "smoke_passed": (
        "true iff the 60s smoke ACTUALLY trained (smoke_train_loss non-null, "
        "smoke_duration_s>60, checkpoint reloads+scores) -- the field D1 gates on; "
        "the pre-flight de-risk that the .462 D1 lacked."
    ),
    "base_used": (
        "the REAL cached base the resolver returned (e.g. Qwen/Qwen3.5-2B) -- "
        "proves the resolver kills the 404 class."
    ),
    "resolver_base_list": (
        "the prioritized list probed [Qwen3.5-2B, Qwen3.5-0.8B, Qwen3-4B, "
        "Qwen2.5-0.5B] + which were present (auditable; a single wrong id can "
        "never block again)."
    ),
    "smoke_train_loss": (
        "the smoke's final contrastive loss (non-null REQUIRED -- proves the "
        "trainer runs the model)."
    ),
    "smoke_duration_s": (
        ">60s REQUIRED (real GPU training takes wall-clock; the anti-skeleton signal)."
    ),
    "checkpoint_path": ("the smoke checkpoint that reloaded + scored (proves resume/score work)."),
    "module_path": "python/carnot/moat_trainer.py -- the shared library D1 imports.",
    "model_specs": ("the resolved base + LoRA + scalar energy head -- the methodology stamp."),
    "inference_substrate": "live_llm_inference (GPU training in the smoke; >=60s floor).",
    "random_seed": "determinism for the smoke train/score.",
    "reproducibility_checksum": "content hash of (base, LoRA config, smoke pairs, seed).",
    "preconditions_checked": (
        "records CUDA/base-cached/fover/peft checks; a missing resource emits "
        "blocked_, never a fabricated smoke."
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "smoke_passed",
    "base_used",
    "resolver_base_list",
    "smoke_train_loss",
    "smoke_duration_s",
    "checkpoint_path",
    "module_path",
    "model_specs",
    "inference_substrate",
    "random_seed",
    "reproducibility_checksum",
    "preconditions_checked",
    "duration_s",
    "field_principles",
    "spec_refs",
)


def _cuda_available() -> tuple[bool, str]:  # pragma: no cover - environment probe
    try:
        import torch

        if torch.cuda.is_available():
            return True, f"torch.cuda.is_available=true, device_count={torch.cuda.device_count()}"
        return False, "torch.cuda.is_available=false"
    except Exception as exc:  # noqa: BLE001
        return False, f"torch import failed: {exc}"


def _peft_importable() -> tuple[bool, str]:  # pragma: no cover - environment probe
    try:
        import peft

        return True, f"peft {getattr(peft, '__version__', '?')}"
    except Exception as exc:  # noqa: BLE001
        return False, f"peft import failed: {exc}"


def _resolver_base_list() -> list[dict[str, Any]]:  # pragma: no cover - filesystem probe
    """Audit which prioritized bases are present (auditable resolver coverage)."""
    out: list[dict[str, Any]] = []
    for repo_id in moat_trainer.PRIORITY_BASES:
        snapshot = moat_trainer.snapshot_with_weights(repo_id)
        out.append(
            {
                "repo_id": repo_id,
                "present": snapshot is not None,
                "path": snapshot.as_posix() if snapshot is not None else None,
            }
        )
    return out


def _build_fover_pairs() -> list[tuple[str, str]]:  # pragma: no cover - data IO
    """Build contrastive (good, bad) pairs from FoVer, reusing the Exp 5003 loader."""
    from carnot.experiment_5003_lora_ebm_scorer_musr import load_fover_pairs

    pairs = load_fover_pairs([FOVER_PATH], max_pairs=SMOKE_MAX_PAIRS)
    return [(p.good_text, p.bad_text) for p in pairs]


def _write(payload: dict[str, Any]) -> None:  # pragma: no cover - file IO
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _base_slug(repo_id: str) -> str:  # pragma: no cover - trivial
    return repo_id.replace("/", "_").replace(".", "")


def _blocked_artifact(
    resource: str, checks: list[dict[str, Any]], duration_s: float
) -> dict[str, Any]:  # pragma: no cover - blocked path
    return {
        "experiment": "experiment_5030_moat_trainer_module",
        "experiment_id": 5030,
        "schema": "carnot.experiment_5030_moat_trainer_module.v1",
        "honest_verdict": f"blocked_{resource}",
        "smoke_passed": False,
        "base_used": None,
        "resolver_base_list": _resolver_base_list(),
        "smoke_train_loss": None,
        "smoke_duration_s": None,
        "checkpoint_path": None,
        "module_path": "python/carnot/moat_trainer.py",
        "model_specs": {
            "resolver_priority": list(moat_trainer.PRIORITY_BASES),
            "adapter": "LoRA",
            "energy_head": "scalar_sequence_regression_head",
        },
        "inference_substrate": "precondition_check_only",
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": None,
        "preconditions_checked": checks,
        "duration_s": round(duration_s, 4),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-VERIFY-5030", "SCENARIO-VERIFY-5030"],
        "result_path": "results/experiment_5030_moat_trainer_module.json",
    }


def main() -> dict[str, Any]:  # pragma: no cover - live GPU experiment entrypoint
    t0 = time.time()
    checks: list[dict[str, Any]] = []

    # --- Step 0: PRECONDITIONS (before any training) -------------------------
    cuda_ok, cuda_detail = _cuda_available()
    checks.append({"resource": "cuda", "available": cuda_ok, "detail": cuda_detail})

    base_repo: str | None = None
    base_path: str | None = None
    try:
        base_repo, base_path = moat_trainer.resolve_trainable_base()
        checks.append(
            {
                "resource": "trainable_base_cached",
                "available": True,
                "detail": f"resolved {base_repo}",
                "path": base_path,
            }
        )
    except RuntimeError as exc:
        checks.append({"resource": "trainable_base_cached", "available": False, "detail": str(exc)})

    fover_ok = FOVER_PATH.exists()
    checks.append(
        {
            "resource": "fover_pairs",
            "available": fover_ok,
            "detail": "data/fover_train_v4.json present" if fover_ok else "missing",
            "path": FOVER_PATH.as_posix(),
        }
    )

    peft_ok, peft_detail = _peft_importable()
    checks.append({"resource": "peft", "available": peft_ok, "detail": peft_detail})

    missing = next((c["resource"] for c in checks if not c["available"]), None)
    if missing is not None:
        artifact = _blocked_artifact(missing, checks, time.time() - t0)
        _write(artifact)
        print(f"[exp5030] BLOCKED on {missing}; wrote {RESULT_PATH}")
        return artifact

    assert base_repo is not None and base_path is not None

    # --- Step 1+2: run the 60s SMOKE on conductor GPU 0 ----------------------
    pair_tuples = _build_fover_pairs()
    print(f"[exp5030] resolved base={base_repo}; built {len(pair_tuples)} FoVer pairs; training...")
    smoke_t0 = time.time()
    train_result = moat_trainer.train_energy_head(
        (base_repo, base_path),
        pair_tuples,
        CHECKPOINT_DIR,
        epochs=SMOKE_EPOCHS,
        batch_size=SMOKE_BATCH_SIZE,
        max_length=SMOKE_MAX_LENGTH,
        device_index=0,
        seed=RANDOM_SEED,
    )
    print(
        f"[exp5030] trained: loss={train_result['train_loss']} "
        f"steps={train_result['n_steps']} ckpt={train_result['checkpoint_dir']}; scoring..."
    )
    energies = moat_trainer.score_candidates(
        train_result["checkpoint_dir"],
        list(SMOKE_CANDIDATES),
        max_length=SMOKE_MAX_LENGTH,
        device_index=0,
    )
    smoke_duration_s = time.time() - smoke_t0

    smoke_train_loss = train_result["train_loss"]
    scored_ok = len(energies) == len(SMOKE_CANDIDATES) and all(
        isinstance(e, float) and e == e for e in energies
    )
    smoke_passed = (smoke_train_loss is not None) and (smoke_duration_s > 60.0) and scored_ok

    if smoke_passed:
        honest_verdict = (
            f"success_moat_trainer_module_shipped_smoke_trained_{_base_slug(base_repo)}"
        )
    else:
        honest_verdict = "blocked_smoke_train_did_not_run"

    artifact: dict[str, Any] = {
        "experiment": "experiment_5030_moat_trainer_module",
        "experiment_id": 5030,
        "schema": "carnot.experiment_5030_moat_trainer_module.v1",
        "honest_verdict": honest_verdict,
        "smoke_passed": smoke_passed,
        "base_used": base_repo,
        "resolver_base_list": _resolver_base_list(),
        "smoke_train_loss": smoke_train_loss,
        "smoke_duration_s": round(smoke_duration_s, 4),
        "smoke_candidate_energies": energies,
        "smoke_n_pairs": train_result["n_pairs"],
        "smoke_n_steps": train_result["n_steps"],
        "checkpoint_path": train_result["checkpoint_dir"],
        "module_path": "python/carnot/moat_trainer.py",
        "model_specs": {
            "base_model": base_repo,
            "base_cache_path": base_path,
            "adapter": "LoRA",
            "energy_head": "scalar_sequence_regression_head",
            "quantization": train_result["model_specs"].get("quantization"),
            "lora_r": train_result["model_specs"].get("lora_r"),
            "lora_alpha": train_result["model_specs"].get("lora_alpha"),
            "resolver_priority": list(moat_trainer.PRIORITY_BASES),
        },
        "inference_substrate": "live_llm_inference",
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": train_result["reproducibility_checksum"],
        "preconditions_checked": checks,
        "duration_s": round(time.time() - t0, 4),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-VERIFY-5030", "SCENARIO-VERIFY-5030"],
        "result_path": "results/experiment_5030_moat_trainer_module.json",
        "verifier_is_oracle": False,
    }
    _write(artifact)
    print(
        f"[exp5030] {'SMOKE PASSED' if smoke_passed else 'SMOKE FAILED'}: "
        f"loss={smoke_train_loss} smoke_duration_s={smoke_duration_s:.1f} "
        f"energies={energies}; wrote {RESULT_PATH}"
    )
    return artifact


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
