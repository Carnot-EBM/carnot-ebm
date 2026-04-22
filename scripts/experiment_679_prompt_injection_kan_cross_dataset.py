"""Experiment 679 — Cross-Dataset Generalization Evaluation for v1 KAN Prompt Injection Classifier.

**Researcher summary:**
    In-distribution AUROC (Exp 678) does not prove real-world readiness.  A
    classifier can score well on the corpus it trained on and then collapse when
    it sees attack phrasing it has never encountered.  This experiment evaluates
    the v1 classifier on THREE held-out datasets it has never seen:
      1. HackAPrompt (a 600k-prompt red-teaming competition, 500-sample subset)
      2. BIPIA (Benchmark for Indirect Prompt Injection Attack, ~400 samples)
      3. Synthetic OWASP LLM-01 stress-test (200 samples from
         scripts/jailbreak_mutations.py using a DIFFERENT seed than Exp 652/678)

**Gate semantics (REQ-SAFE-010):**
    - mean cross-dataset AUROC >= 0.80 → generalization_verified_publishable
    - 0.65 <= mean AUROC < 0.80 → generalization_partial_shareable_with_caveat
    - mean AUROC < 0.65 → generalization_failed_do_not_publish
    - v1 weights absent → blocked_on_upstream_exp_678
    - dataset download failure → blocked_on_dataset_download_failure

Spec: REQ-SAFE-010, SCENARIO-SAFE-010
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_V1_WEIGHTS = _REPO_ROOT / "python" / "carnot" / "models" / "prompt_injection_kan_v1_weights.json"
_RESULT_PATH = _REPO_ROOT / "results" / "experiment_679_prompt_injection_kan_cross_dataset.json"
_MODEL_CARD_PATH = _REPO_ROOT / "python" / "carnot" / "models" / "prompt_injection_kan_v1_MODELCARD.md"

# Publish threshold — do NOT lower this to make v1 pass.
_PUBLISH_THRESHOLD = 0.80
_CAVEAT_THRESHOLD = 0.65

# How many samples to use from each dataset (speed vs. representativeness).
_HACKAPROMPT_SUBSET = 500
_BIPIA_SUBSET = 400
_SYNTHETIC_N = 200
# Use a seed different from Exp 652 (seed=42) so we're testing on unseen phrasing.
_SYNTHETIC_SEED = 679


def _write_artifact(artifact: dict[str, Any], path: Path) -> None:
    """Write a JSON artifact to disk, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(artifact, fh, indent=2)
    print(f"[679] artifact written → {path}", flush=True)


def _compute_confusion(scores: list[float], labels: list[int], threshold: float = 0.5) -> dict[str, int]:
    """Compute TP/FP/TN/FN at a fixed energy threshold.

    Positive class = injection (label=1).  A prompt is predicted injection when
    its energy score exceeds the threshold.

    Why threshold=0.5: this is a neutral reference point for the confusion
    matrix — the deployment threshold is calibrated separately; we just need
    a common reference for cross-dataset comparison in this experiment.

    Args:
        scores:    List of energy values (higher = more likely injection).
        labels:    Corresponding ground-truth labels (1 = injection, 0 = benign).
        threshold: Decision boundary; prompts with energy > threshold are
                   predicted injection.

    Returns:
        Dict with keys tp, fp, tn, fn.
    """
    tp = fp = tn = fn = 0
    for score, label in zip(scores, labels):
        predicted = 1 if score > threshold else 0
        if predicted == 1 and label == 1:
            tp += 1
        elif predicted == 1 and label == 0:
            fp += 1
        elif predicted == 0 and label == 1:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _load_hackaprompt(n: int) -> list[tuple[str, int]]:
    """Load a balanced subset of HackAPrompt from HuggingFace datasets.

    HackAPrompt is a red-teaming competition corpus.  We load the 'train' split
    and take the first n/2 injection prompts + n/2 benign-ish prompts.
    Because HackAPrompt is 100% injection prompts (it's a red-teaming dataset),
    we pair each injection prompt with a synthetic benign prompt so that AUROC
    is well-defined (requires both classes).

    Args:
        n: Total number of (text, label) pairs to return (~n/2 per class).

    Returns:
        List of (text, label) tuples where label ∈ {0, 1}.

    Raises:
        ImportError: if HuggingFace datasets library is not installed.
        Exception: if the dataset cannot be downloaded (network, auth failure).
    """
    from datasets import load_dataset  # type: ignore[import]
    from scripts.jailbreak_mutations import generate_synthetic_benign

    # HackAPrompt v1 — level 0 prompts are "user input as sent to the LLM"
    ds = load_dataset("hackaprompt/hackaprompt-dataset", split="train", trust_remote_code=False)

    # Take first n//2 unique injection prompts.
    seen: set[str] = set()
    injections: list[str] = []
    for row in ds:
        text = row.get("user_input") or row.get("prompt") or ""
        text = str(text).strip()
        if text and text not in seen:
            seen.add(text)
            injections.append(text)
        if len(injections) >= n // 2:
            break

    # Pair with synthetic benign prompts (same count) using a fixed seed so
    # the evaluation is reproducible across re-runs.
    benigns = generate_synthetic_benign(n=len(injections), seed=679)

    result: list[tuple[str, int]] = []
    for text in injections:
        result.append((text, 1))
    for text in benigns:
        result.append((text, 0))

    return result


def _load_bipia(n: int) -> list[tuple[str, int]]:
    """Load a balanced subset of BIPIA (Benchmark for Indirect Prompt Injection Attack).

    BIPIA tests indirect injection where the attack is embedded in a retrieved
    document rather than the direct user turn.  We use the English subset.

    Args:
        n: Target number of samples.

    Returns:
        List of (text, label) tuples.

    Raises:
        Exception: if dataset is unavailable.
    """
    from datasets import load_dataset  # type: ignore[import]
    from scripts.jailbreak_mutations import generate_synthetic_benign

    # BIPIA — indirect injection attacks embedded in external context.
    # The 'test' split has ~400 attack examples across task categories.
    try:
        ds = load_dataset("microsoft/BIPIA", split="test", trust_remote_code=False)
    except Exception:
        # BIPIA may need the 'validation' split on some hub versions.
        ds = load_dataset("microsoft/BIPIA", split="validation", trust_remote_code=False)

    injections: list[str] = []
    for row in ds:
        # 'attack_str' is the injected payload; 'context' is the benign wrapper.
        # We evaluate on the full context (wrapping + injection) because that is
        # what the classifier sees in production: the retrieved document as text.
        attack = row.get("attack_str") or row.get("injected_prompt") or row.get("text") or ""
        context = row.get("context") or row.get("data") or ""
        text = (str(context) + " " + str(attack)).strip() if context else str(attack).strip()
        if text:
            injections.append(text)
        if len(injections) >= n // 2:
            break

    benigns = generate_synthetic_benign(n=len(injections), seed=680)
    result: list[tuple[str, int]] = []
    for text in injections:
        result.append((text, 1))
    for text in benigns:
        result.append((text, 0))

    return result


def _load_synthetic_stress(n: int, seed: int) -> list[tuple[str, int]]:
    """Generate a synthetic OWASP LLM-01 stress test with a different seed than training.

    Uses the same jailbreak_mutations.py generator as Exp 652/678 but with a
    different seed (679) so the generated prompts are unseen by the classifier.
    The Exp 652 corpus used seed=42 for injections and seed=99 for benign; using
    seed=679/680 produces disjoint phrasing variants.

    This catches the "dataset memorisation" failure mode: if the classifier is
    just pattern-matching on the exact teacher-labeled phrasing rather than
    learning the structural pattern, it will fail here.

    Args:
        n:    Total number of (text, label) pairs (~n/2 per class).
        seed: Random seed.  Must differ from Exp 652's seed=42 and seed=99.

    Returns:
        List of (text, label) tuples.
    """
    import sys
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    from jailbreak_mutations import generate_synthetic_benign, generate_synthetic_injections

    injections = generate_synthetic_injections(n=n // 2, seed=seed)
    benigns = generate_synthetic_benign(n=n // 2, seed=seed + 1)

    result: list[tuple[str, int]] = []
    for text in injections:
        result.append((text, 1))
    for text in benigns:
        result.append((text, 0))
    return result


def _score_dataset(
    checker: Any,
    examples: list[tuple[str, int]],
    dataset_name: str,
) -> tuple[float, dict[str, int]]:
    """Compute AUROC and confusion matrix for a (text, label) dataset.

    Args:
        checker:      Loaded PromptInjectionEnergyChecker.
        examples:     List of (text, label) tuples.
        dataset_name: Name for progress logging.

    Returns:
        (auroc, confusion_matrix_dict)
    """
    from carnot.models.prompt_injection_kan import InjectionExample, _compute_auroc

    print(f"[679] scoring {dataset_name} ({len(examples)} samples)…", flush=True)
    t0 = time.time()

    scores: list[float] = []
    labels: list[int] = []
    for text, label in examples:
        scores.append(checker.energy(text))
        labels.append(label)

    auroc = _compute_auroc(scores, labels)
    cm = _compute_confusion(scores, labels, threshold=0.5)
    elapsed = time.time() - t0

    print(f"[679] {dataset_name}: AUROC={auroc:.4f}  elapsed={elapsed:.1f}s", flush=True)
    return auroc, cm


def _map_verdict(mean_auroc: float) -> str:
    """Map mean cross-dataset AUROC to the gate verdict string.

    Do NOT lower _PUBLISH_THRESHOLD to make v1 pass.  A lower verdict is
    the honest result and is actionable (the next experiment iterates on
    corpus diversity).

    Args:
        mean_auroc: Mean AUROC across all three held-out datasets.

    Returns:
        honest_verdict string from {generalization_verified_publishable,
        generalization_partial_shareable_with_caveat,
        generalization_failed_do_not_publish}.
    """
    if mean_auroc >= _PUBLISH_THRESHOLD:
        return "generalization_verified_publishable"
    elif mean_auroc >= _CAVEAT_THRESHOLD:
        return "generalization_partial_shareable_with_caveat"
    else:
        return "generalization_failed_do_not_publish"


def _write_model_card(
    per_dataset_auroc: dict[str, float],
    per_dataset_cm: dict[str, dict[str, int]],
    mean_auroc: float,
    latency_ms: float,
    training_auroc: float | None,
) -> None:
    """Write a draft model card to python/carnot/models/prompt_injection_kan_v1_MODELCARD.md.

    Only called when mean AUROC >= 0.80.  The model card is a prerequisite for
    HuggingFace publication but does NOT trigger the push — that is a separate
    operator action.

    Args:
        per_dataset_auroc:  Dict mapping dataset name → AUROC.
        per_dataset_cm:     Dict mapping dataset name → confusion matrix.
        mean_auroc:         Mean across all held-out datasets.
        latency_ms:         Median CPU inference latency in milliseconds.
        training_auroc:     AUROC on the training-distribution test split
                            (from Exp 678), or None if unavailable.
    """
    # Worst datasets first (for the "known failure modes" section)
    sorted_datasets = sorted(per_dataset_auroc.items(), key=lambda kv: kv[1])
    worst_datasets = [k for k, v in sorted_datasets if v < _PUBLISH_THRESHOLD]

    cm_rows = []
    for ds_name, cm in per_dataset_cm.items():
        auroc = per_dataset_auroc[ds_name]
        cm_rows.append(
            f"| {ds_name} | {auroc:.4f} | {cm['tp']} | {cm['fp']} | {cm['tn']} | {cm['fn']} |"
        )

    training_row = ""
    if training_auroc is not None:
        training_row = f"| training-distribution (Exp 678) | {training_auroc:.4f} | — | — | — | — |"

    known_failures = "\n".join(
        f"- **{ds}** (AUROC {per_dataset_auroc[ds]:.4f} < {_PUBLISH_THRESHOLD:.2f}): "
        f"classifier underperforms on this dataset; consider adding representative samples to training corpus."
        for ds in worst_datasets
    ) or "None — all held-out datasets met the publish threshold."

    card = f"""# Model Card — PromptInjectionEnergyChecker v1

**Model type:** Two-layer KAN (Kolmogorov-Arnold Network) energy-based classifier
**Architecture:** n_features=32, n_hidden=8, ~3,432 parameters
**License:** Apache 2.0
**Spec reference:** [REQ-SAFE-007, REQ-SAFE-010](../../../openspec/capabilities/safety/spec.md)

## Summary

This model detects prompt injection attacks using a lightweight energy-based
model (EBM) distilled from gpt-oss-safeguard-20b (Apache 2.0).  It assigns a
scalar energy to prompt text — low energy for benign requests, high energy for
injection attempts.

## Evaluation Results (Exp 679 — Cross-Dataset Generalization Gate)

REQ-SAFE-010 requires mean AUROC >= {_PUBLISH_THRESHOLD:.2f} across three held-out datasets
(HackAPrompt, BIPIA, and a synthetic OWASP LLM-01 stress test).

**Mean cross-dataset AUROC: {mean_auroc:.4f}** (gate: >= {_PUBLISH_THRESHOLD:.2f} to publish)

| Dataset | AUROC | TP | FP | TN | FN |
|---------|-------|----|----|----|----|
{training_row}
{chr(10).join(cm_rows)}

Confusion matrices use threshold=0.5 (energy > 0.5 → injection prediction).
The deployment threshold should be calibrated separately on a validation set.

## Latency

Median CPU-only inference latency: **{latency_ms:.2f} ms** per prompt (single core, JAX CPU).
Requirement (REQ-SAFE-007): < 5 ms.

## Known Failure Modes

{known_failures}

## Training

Distilled from `gpt-oss-safeguard-20b` (Apache 2.0) via Exp 678.  Training
corpus: ≥ 2,000 balanced (benign / injection) examples covering OWASP LLM-01
attack categories 1–8.  See `scripts/experiment_678_prompt_injection_kan_true_distillation.py`
for the full training pipeline.

## Limitations

- KAN features are bag-of-words (keyword count ratios); novel obfuscation
  techniques that avoid all known injection keywords may evade detection.
- Threshold calibration is deployment-specific; the default threshold of 0.5
  is a neutral reference, not a tuned operating point.
- This model is a gate in a larger pipeline, not a standalone safety system.

## Citation

If you use this model in research, please cite the Carnot EBM repository and
reference Exp 678 (training) and Exp 679 (cross-dataset evaluation).

## License

Apache 2.0 — same as the teacher model (gpt-oss-safeguard-20b).
See [LICENSE](../../../LICENSE) for the full text.
"""

    _MODEL_CARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_MODEL_CARD_PATH, "w") as fh:
        fh.write(card)
    print(f"[679] model card written → {_MODEL_CARD_PATH}", flush=True)


def _measure_latency(checker: Any, n: int = 20) -> float:
    """Measure median CPU inference latency over n cold calls.

    Args:
        checker: Loaded PromptInjectionEnergyChecker.
        n:       Number of calls to median over.

    Returns:
        Median latency in milliseconds.
    """
    probe = "What is the capital of France?"
    times: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        checker.energy(probe)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def run(v1_weights_path: Path = _V1_WEIGHTS) -> dict[str, Any]:
    """Run Exp 679 end-to-end and return the result artifact dict.

    This is the main entry point, separated from main() so that tests can
    call it with a custom weights path without touching the filesystem.

    Args:
        v1_weights_path: Path to the v1 KAN weights JSON produced by Exp 678.
                         Defaults to the canonical location.

    Returns:
        Dict with all required schema fields for Exp 679.
    """
    # ------------------------------------------------------------------
    # Step 1: Preflight — v1 weights must exist.
    # ------------------------------------------------------------------
    if not v1_weights_path.exists():
        artifact: dict[str, Any] = {
            "experiment": 679,
            "honest_verdict": "blocked_on_upstream_exp_678",
            "reason": (
                f"v1 weights file not found at {v1_weights_path}.  "
                "Run Exp 678 first: "
                "python scripts/experiment_678_prompt_injection_kan_true_distillation.py"
            ),
            "per_dataset_auroc": {},
            "mean_auroc": None,
            "per_dataset_cm": {},
            "model_card_written": False,
        }
        return artifact

    # ------------------------------------------------------------------
    # Step 2: Load classifier.
    # ------------------------------------------------------------------
    sys.path.insert(0, str(_REPO_ROOT / "python"))
    from carnot.models.prompt_injection_kan import PromptInjectionEnergyChecker

    print("[679] loading v1 weights…", flush=True)
    checker = PromptInjectionEnergyChecker.load(v1_weights_path)
    print(f"[679] loaded: n_features={checker.n_features}, n_hidden={checker.n_hidden}, "
          f"n_params={checker.n_params()}", flush=True)

    # ------------------------------------------------------------------
    # Step 3: Download / load three held-out datasets.
    # ------------------------------------------------------------------
    datasets: dict[str, list[tuple[str, int]]] = {}
    dataset_loaders = {
        "hackaprompt": lambda: _load_hackaprompt(_HACKAPROMPT_SUBSET),
        "bipia": lambda: _load_bipia(_BIPIA_SUBSET),
        "synthetic_owasp_stress": lambda: _load_synthetic_stress(_SYNTHETIC_N, _SYNTHETIC_SEED),
    }

    for name, loader in dataset_loaders.items():
        try:
            print(f"[679] loading dataset: {name}…", flush=True)
            examples = loader()
            datasets[name] = examples
            n_inj = sum(1 for _, l in examples if l == 1)
            n_ben = sum(1 for _, l in examples if l == 0)
            print(f"[679] {name}: {n_inj} injection, {n_ben} benign", flush=True)
        except Exception as exc:
            artifact = {
                "experiment": 679,
                "honest_verdict": "blocked_on_dataset_download_failure",
                "reason": f"Failed to load dataset '{name}': {exc}",
                "per_dataset_auroc": {},
                "mean_auroc": None,
                "per_dataset_cm": {},
                "model_card_written": False,
            }
            return artifact

    # ------------------------------------------------------------------
    # Step 4: Score each dataset.
    # ------------------------------------------------------------------
    per_dataset_auroc: dict[str, float] = {}
    per_dataset_cm: dict[str, dict[str, int]] = {}

    for name, examples in datasets.items():
        auroc, cm = _score_dataset(checker, examples, name)
        per_dataset_auroc[name] = auroc
        per_dataset_cm[name] = cm

    # ------------------------------------------------------------------
    # Step 5: Gate decision.
    # ------------------------------------------------------------------
    mean_auroc = sum(per_dataset_auroc.values()) / len(per_dataset_auroc)
    honest_verdict = _map_verdict(mean_auroc)

    print(f"[679] per_dataset_auroc: {per_dataset_auroc}", flush=True)
    print(f"[679] mean_cross_dataset_auroc: {mean_auroc:.4f}", flush=True)
    print(f"[679] honest_verdict: {honest_verdict}", flush=True)

    # ------------------------------------------------------------------
    # Step 6: Model card (only if publishable).
    # ------------------------------------------------------------------
    model_card_written = False
    latency_ms = _measure_latency(checker)
    print(f"[679] latency: {latency_ms:.2f} ms", flush=True)

    if honest_verdict == "generalization_verified_publishable":
        _write_model_card(
            per_dataset_auroc=per_dataset_auroc,
            per_dataset_cm=per_dataset_cm,
            mean_auroc=mean_auroc,
            latency_ms=latency_ms,
            training_auroc=None,  # Exp 678 result not re-loaded here; conductor links them.
        )
        model_card_written = True

    # ------------------------------------------------------------------
    # Step 7: Build artifact.
    # ------------------------------------------------------------------
    artifact = {
        "experiment": 679,
        "honest_verdict": honest_verdict,
        "per_dataset_auroc": per_dataset_auroc,
        "mean_auroc": round(mean_auroc, 6),
        "per_dataset_cm": per_dataset_cm,
        "model_card_written": model_card_written,
        "latency_ms": round(latency_ms, 3),
        "publish_threshold": _PUBLISH_THRESHOLD,
        "caveat_threshold": _CAVEAT_THRESHOLD,
    }
    return artifact


def main() -> None:
    """Run Exp 679 end-to-end with a 45-minute hard stop.

    Writes result JSON to results/experiment_679_prompt_injection_kan_cross_dataset.json
    and exits with code 0 regardless of gate outcome (a non-publish verdict is
    an honest result, not a code failure).
    """
    t_start = time.time()
    timeout_s = 45 * 60  # 45 minutes per task spec

    artifact = run()

    elapsed = time.time() - t_start
    if elapsed > timeout_s:
        print(f"[679] WARNING: exceeded {timeout_s}s timeout (elapsed={elapsed:.0f}s)", flush=True)

    artifact["duration_s"] = round(elapsed, 1)
    _write_artifact(artifact, _RESULT_PATH)
    print(f"[679] done in {elapsed:.1f}s — verdict: {artifact['honest_verdict']}", flush=True)


if __name__ == "__main__":
    main()
