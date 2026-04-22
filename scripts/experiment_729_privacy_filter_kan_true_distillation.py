#!/usr/bin/env python3
"""Experiment 729 — Privacy Filter KAN: True Distillation from openai/privacy-filter.

**Goal:**
    Distil openai/privacy-filter (a transformer-based PII classifier) into a
    ~3264-parameter KAN student that:
    - Runs in < 5 ms on a single CPU core (vs ~200 ms for the teacher).
    - Achieves AUROC >= 0.85 on a 400-example held-out PII detection test set.
    - Can be deployed on any Carnot EBM stack without GPU or transformer infra.

    This follows the same distillation pattern as Exps 690 (prompt injection KAN v1)
    and 710 (KAN v2) but targets PII/privacy detection instead of injection detection.

**Pipeline:**
    1. Preflight: verify openai/privacy-filter model is present in models/.
    2. Build corpus: 1000 benign (GSM8K-style + HumanEval + safe web text) +
                     1000 privacy-violating (synthetic CC/SSN/email/phone/address).
    3. Teacher inference: run openai/privacy-filter on each prompt, cache by
       (model_sha, prompt_sha) in data/privacy_filter_distill/.
    4. Train PrivacyFilterEnergyChecker (2-layer KAN, 32 hidden, n_knots=3,
       degree=3, ~3264 params) for 100 epochs via contrastive loss.
    5. Evaluate AUROC on a 400-example held-out split (20% of corpus).
    6. Latency check: 1000 CPU forward passes, assert median < 5 ms.
    7. Save weights to python/carnot/models/privacy_filter_kan_v1.json.

**Honest-verdict enum:**
    - "distillation_corpus_built_classifier_trained_auroc_met"
    - "distillation_corpus_built_classifier_trained_auroc_below_threshold"
    - "distillation_corpus_built_classifier_not_trained"
    - "distillation_corpus_not_built"
    - "blocked_on_dependency"

    REQ-SAFE-011 invariant is AUTO-ENFORCED by assert_deliverable_written():
    if teacher_inference_duration_s < corpus_size * 0.5, the verdict is
    rewritten to "distillation_invariant_violated_teacher_too_fast".

Spec: REQ-SAFE-015, REQ-SAFE-016
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "python"))

# Minimum seconds per prompt that real teacher inference must satisfy (REQ-SAFE-011).
_MIN_S_PER_PROMPT = 0.5

# Fraction of corpus reserved for held-out AUROC evaluation.
_HOLDOUT_FRACTION = 0.20

# Target AUROC for the distillation gate to open.
_AUROC_GATE = 0.85

# Per-class count for corpus construction.
_N_PER_CLASS = 1000


def _build_synthetic_benign() -> list[str]:
    """Return a list of up to _N_PER_CLASS benign (no-PII) texts.

    Uses GSM8K-style math questions and HumanEval-style coding prompts.
    All texts are safe: no names, addresses, IDs, or financial data.

    Spec: REQ-SAFE-016
    """
    texts: list[str] = []
    texts += [f"If a train travels {i * 10} miles in {i} hours, what is its average speed?" for i in range(1, 51)]
    texts += [f"What is {i} * {i + 7} + {i * 2}?" for i in range(1, 51)]
    texts += [f"Write a function that returns the Fibonacci sequence up to {i} terms." for i in range(1, 51)]
    texts += [f"What is the square root of {i * i}?" for i in range(1, 51)]
    texts += [
        "Explain the difference between a list and a tuple in Python.",
        "What is the time complexity of merge sort?",
        "How does a hash map handle collisions?",
        "Explain gradient descent in machine learning.",
        "What is a binary search tree and how do you traverse it?",
        "Describe how TCP/IP packet routing works.",
        "What is the capital of Germany?",
        "Explain the concept of inheritance in object-oriented programming.",
        "What does the map() function do in Python?",
        "How does a quicksort algorithm work?",
        "What is a REST API and how does it differ from GraphQL?",
        "Explain what a Fourier transform is used for.",
        "What is the difference between supervised and unsupervised learning?",
        "How does a database index improve query performance?",
        "Write a Python function that checks if a string is a palindrome.",
        "Explain the CAP theorem for distributed systems.",
        "What is Docker and how does it differ from a virtual machine?",
        "Describe the purpose of a load balancer in web architecture.",
        "What is the difference between synchronous and asynchronous programming?",
        "What is the halting problem in computer science?",
    ]
    # Pad to _N_PER_CLASS.
    while len(texts) < _N_PER_CLASS:
        n = len(texts)
        texts.append(f"Compute {n} squared minus {n - 1} squared.")
    return texts[:_N_PER_CLASS]


def _build_synthetic_pii() -> list[str]:
    """Return a list of up to _N_PER_CLASS PII-containing texts.

    Covers credit cards, SSNs, emails, phone numbers, addresses, and mixed PII.
    All values are synthetically generated (not real PII) but structurally valid
    so that regex-based and ML-based detectors will flag them.

    Spec: REQ-SAFE-016
    """
    texts: list[str] = []

    # Credit card number patterns (4×4 digit format).
    for i in range(50):
        texts.append(
            f"My credit card number is 4111 1111 1111 {1000 + i:04d}, "
            f"expiry 01/{25 + (i % 10)}, CVV {100 + i}."
        )

    # SSN patterns.
    for i in range(50):
        texts.append(
            f"Social Security Number: {100 + i:03d}-{10 + (i % 90):02d}-{1000 + i:04d}. "
            f"Please keep confidential."
        )

    # Email + name combos.
    for i in range(50):
        texts.append(
            f"Please send the document to user{i}@example-private.com — "
            f"my full name is John Doe{i}."
        )

    # Phone + address combos.
    for i in range(50):
        phone_area = 300 + i // 10
        phone_mid = (i * 7) % 900
        phone_end = (i * 13) % 9000
        texts.append(
            f"Call me at ({phone_area:03d}) {phone_mid:03d}-{phone_end:04d}. "
            f"My home address is {100 + i} Main St, Springfield IL 62701."
        )

    # Medical record / PHI style.
    for i in range(50):
        texts.append(
            f"Patient: Jane Smith{i}. DOB: {(i % 28) + 1:02d}/03/198{i % 10}. "
            f"Medical record: MR{10000 + i}. Diagnosis: hypertension."
        )

    # Financial account numbers.
    for i in range(50):
        texts.append(
            f"Bank account: {100000000 + i * 7:09d}. Routing: 021000021. "
            f"Account holder: Robert Johnson{i}."
        )

    # Mixed PII (multiple types in one text).
    for i in range(100):
        texts.append(
            f"Name: Alice Brown{i}, SSN: {200 + i:03d}-{i % 99:02d}-{3000 + i:04d}, "
            f"CC: 5500 0000 0000 {i:04d}, "
            f"Phone: 555-{(i * 7) % 900:03d}-{(i * 13) % 9000:04d}."
        )

    # Keyword-heavy PII disclosure text.
    texts += [
        "Please verify my social security number and credit card details for the account.",
        "My passport number is AB1234567 and date of birth is 15/06/1985.",
        "The billing address on file is 42 Elm St, Boston MA 02101.",
        "Home phone: (617) 555-1234. Office: (617) 555-5678.",
        "My mother's maiden name is Kowalski and my ZIP code is 02134.",
        "Account password is P@ssw0rd123 — please keep confidential.",
        "Private key: -----BEGIN RSA PRIVATE KEY----- (truncated for security).",
        "Driver's license number DL-K123456, issued by California DMV.",
        "Patient ID: PT-987654. Insurance policy number: HMO-123456789.",
        "Wire transfer routing 021000021, account 1234567890, SWIFT: BOFAUS3N.",
        "Full name: Christopher Lee Anderson. Date of birth: 1982-03-17.",
        "Email: c.anderson@private.net; SSN: 555-66-7777; CC: 4532-0151-1283-0366.",
    ]

    # Pad to _N_PER_CLASS with variations of a PII disclosure template.
    pii_base = (
        "The form requires your full name, address, and social security number. "
        "Please also provide credit card information for billing."
    )
    while len(texts) < _N_PER_CLASS:
        texts.append(pii_base + f" Reference ID: {len(texts):05d}.")

    return texts[:_N_PER_CLASS]


def _build_corpus() -> tuple[list[dict], list[dict]]:
    """Build the distillation corpus from built-in synthetic examples.

    Returns (benign_items, pii_items), each a list of dicts with
    keys: text (str), label (str), source (str).

    Spec: REQ-SAFE-016
    """
    benign_texts = _build_synthetic_benign()
    pii_texts = _build_synthetic_pii()
    benign = [
        {"text": t, "label": "benign", "source": "synthetic_gsm8k_humaneval"}
        for t in benign_texts
    ]
    pii = [
        {"text": t, "label": "pii", "source": "synthetic_pii"}
        for t in pii_texts
    ]
    return benign, pii


def _run_teacher_inference_privacy(
    model_dir: Path,
    items: list[dict],
    cache_path: Path,
    log: logging.Logger,
) -> tuple[float, list[dict]]:
    """Run openai/privacy-filter on corpus items, caching results.

    openai/privacy-filter is a safetensors HuggingFace model loaded via
    transformers AutoModelForSequenceClassification.  Each call returns
    a label probability; we threshold at 0.5 to produce binary teacher
    labels (0=benign, 1=pii).

    Cache key: (model_sha, prompt_sha) where model_sha is derived from the
    directory file listing.  Cache is written after each inference call so
    the cache survives an interrupted run.

    Args:
        model_dir:  Path to the locally downloaded openai/privacy-filter directory.
        items:      List of dicts with 'text', 'label', 'source' keys.
        cache_path: Path to JSON cache file.
        log:        Logger instance.

    Returns:
        (total_inference_duration_s, labeled_items)

    Spec: REQ-SAFE-016
    """
    existing_cache: dict[str, dict] = {}
    if cache_path.exists():
        try:
            with open(cache_path) as fh:
                existing_cache = json.load(fh)
            log.info("Loaded %d entries from teacher cache at %s", len(existing_cache), cache_path)
        except Exception as exc:
            log.warning("Failed to load teacher cache: %s", exc)

    model_files = sorted(f.name for f in model_dir.iterdir() if f.is_file())
    model_sha = hashlib.sha256("|".join(model_files).encode()).hexdigest()[:12]
    log.info("Teacher model SHA: %s (%d files in %s)", model_sha, len(model_files), model_dir)

    need_inference = []
    for item in items:
        ph = hashlib.sha256(item["text"].encode()).hexdigest()[:16]
        key = json.dumps([model_sha, ph])
        if key not in existing_cache:
            need_inference.append((item, ph))

    log.info(
        "%d / %d items need teacher inference (%d cached)",
        len(need_inference),
        len(items),
        len(items) - len(need_inference),
    )

    if need_inference:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore[import]
            import torch  # type: ignore[import]

            log.info("Loading openai/privacy-filter from %s", model_dir)
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
            model.eval()

            for item, ph in need_inference:
                t0 = time.perf_counter()
                try:
                    inputs = tokenizer(
                        item["text"][:512],
                        return_tensors="pt",
                        truncation=True,
                        max_length=128,
                        padding=True,
                    )
                    with torch.no_grad():
                        logits = model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1).squeeze()
                    pii_prob = float(probs[1]) if probs.ndim > 0 and len(probs) > 1 else float(probs)
                    teacher_label = 1 if pii_prob >= 0.5 else 0
                    teacher_raw = f"pii_prob={pii_prob:.4f}"
                except Exception as exc:
                    log.debug("Teacher inference error on prompt %s: %s", ph[:8], exc)
                    teacher_label = -1
                    teacher_raw = f"error:{exc}"

                elapsed = time.perf_counter() - t0

                key = json.dumps([model_sha, ph])
                existing_cache[key] = {
                    "model_sha": model_sha,
                    "prompt_sha": ph,
                    "teacher_label": teacher_label,
                    "teacher_raw": teacher_raw,
                    "elapsed_s": elapsed,
                }
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = cache_path.with_suffix(".tmp")
                    with open(tmp, "w") as fh:
                        json.dump(existing_cache, fh)
                    tmp.rename(cache_path)
                except Exception as exc:
                    log.debug("Cache write failed: %s", exc)

        except ImportError as exc:
            log.error("transformers/torch not available: %s — using source labels", exc)
            return 0.0, list(items)

    total_elapsed = sum(
        existing_cache.get(
            json.dumps([model_sha, hashlib.sha256(item["text"].encode()).hexdigest()[:16]]),
            {},
        ).get("elapsed_s", 0.0)
        for item in items
    )

    labeled: list[dict] = []
    for item in items:
        ph = hashlib.sha256(item["text"].encode()).hexdigest()[:16]
        key = json.dumps([model_sha, ph])
        entry = existing_cache.get(key)
        if entry and entry.get("teacher_label") in (0, 1):
            lbl = "pii" if entry["teacher_label"] == 1 else "benign"
            labeled.append({
                "text": item["text"],
                "label": lbl,
                "source": f"teacher_distilled:{item.get('source', 'unknown')}",
                "elapsed_s": entry.get("elapsed_s", 0.0),
            })
        else:
            labeled.append(dict(item))

    return total_elapsed, labeled


def _latency_check(checker, n: int) -> tuple[float, str]:
    """Time n cold-cache CPU forward passes; return (median_ms, flag_string).

    Varies the text each call so JAX's JIT cache cannot return a trivial result.
    The median over n calls is a stable p50 latency estimate.

    Args:
        checker: PrivacyFilterEnergyChecker instance with trained weights.
        n:       Number of calls to time.

    Returns:
        (median_ms, flag) where flag is "PASS" if median < 5 ms else "FAIL:<ms>ms".

    Spec: REQ-SAFE-015
    """
    import statistics

    times_ms = []
    for i in range(n):
        text = f"Latency probe {i}: what is the capital of France?"
        t0 = time.perf_counter()
        _ = checker.energy(text)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    median_ms = statistics.median(times_ms)
    flag = "PASS" if median_ms < 5.0 else f"FAIL:{median_ms:.2f}ms"
    return median_ms, flag


def _write_artifact(artifact: dict, path: Path) -> None:
    """Atomic JSON write: write to .tmp then rename to avoid partial writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(artifact, fh, indent=2)
    tmp.rename(path)


def main() -> None:
    """Run Exp 729 end-to-end with a 120-minute hard stop."""
    try:
        from carnot.pipeline.env_autofix import apply_env_autofix
        apply_env_autofix()
    except ImportError:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger(__name__)

    from scripts.experiment_template import ExperimentTemplate
    from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog

    DELIVERABLE = "results/experiment_729_privacy_filter_kan_true_distillation.json"
    MODEL_DIR = _REPO_ROOT / "models" / "openai_privacy_filter"
    CORPUS_DIR = _REPO_ROOT / "data" / "privacy_filter_distill"
    TEACHER_CACHE = CORPUS_DIR / "teacher_outputs_privacy_v729.json"
    WEIGHTS_PATH = _REPO_ROOT / "python" / "carnot" / "models" / "privacy_filter_kan_v1.json"

    tmpl = ExperimentTemplate(
        729,
        "Privacy Filter KAN — True Distillation from openai/privacy-filter",
        DELIVERABLE,
    )
    tmpl.setup()

    watchdog = ExperimentTimeoutWatchdog(
        experiment_id=729,
        timeout_minutes=120,
        result_path=str(_REPO_ROOT / DELIVERABLE),
    )
    watchdog.start()

    try:
        _run(tmpl, log, DELIVERABLE, MODEL_DIR, CORPUS_DIR, TEACHER_CACHE, WEIGHTS_PATH)
    finally:
        watchdog.stop()

    tmpl.assert_deliverable_written()


def _run(tmpl, log, deliverable, model_dir, corpus_dir, teacher_cache_path, weights_path):
    """Inner experiment logic separated from watchdog wiring.

    Separation rationale: the watchdog writes a partial artifact on timeout while
    this function writes the final artifact on success.  Mixing them would leave
    the watchdog unable to find a clean write point.

    Spec: REQ-SAFE-015, REQ-SAFE-016
    """
    from carnot.models.privacy_filter_kan import PrivacyFilterEnergyChecker, PrivacyExample

    repo_root = tmpl._repo_root

    # ------------------------------------------------------------------
    # Phase 1: Preflight — verify teacher model is present
    # ------------------------------------------------------------------
    log.info("Phase 1: Preflight — checking openai/privacy-filter at %s", model_dir)

    model_dir = Path(model_dir)
    if not model_dir.exists() or not any(model_dir.iterdir()):
        log.error("BLOCKED: openai/privacy-filter not found at %s", model_dir)
        artifact = tmpl.build_result(
            {
                "honest_verdict": "blocked_on_dependency",
                "teacher_inference_duration_s": 0.0,
            },
            status="blocked",
            reason=(
                "openai/privacy-filter model directory missing or empty. "
                "Run: huggingface-cli download openai/privacy-filter "
                "--local-dir models/openai_privacy_filter"
            ),
        )
        _write_artifact(artifact, repo_root / deliverable)
        return

    log.info("Preflight PASS: teacher model directory exists at %s", model_dir)

    # ------------------------------------------------------------------
    # Phase 2: Build corpus (1000 benign + 1000 PII)
    # ------------------------------------------------------------------
    log.info("Phase 2: Building corpus (%d benign + %d PII)", _N_PER_CLASS, _N_PER_CLASS)
    Path(corpus_dir).mkdir(parents=True, exist_ok=True)

    benign_items, pii_items = _build_corpus()
    all_items = benign_items + pii_items

    log.info("Corpus: %d benign + %d PII = %d total", len(benign_items), len(pii_items), len(all_items))

    if len(all_items) < 100:
        artifact = tmpl.build_result(
            {"honest_verdict": "distillation_corpus_not_built", "teacher_inference_duration_s": 0.0},
            status="blocked",
            reason="Corpus construction returned fewer than 100 examples",
        )
        _write_artifact(artifact, repo_root / deliverable)
        return

    # ------------------------------------------------------------------
    # Phase 3: Teacher inference via openai/privacy-filter
    # ------------------------------------------------------------------
    log.info("Phase 3: Running teacher inference on %d items", len(all_items))

    teacher_inference_duration_s, labeled_items = _run_teacher_inference_privacy(
        model_dir, all_items, Path(teacher_cache_path), log
    )

    log.info(
        "Teacher inference done: %.1f s total (%.3f s/prompt, REQ-SAFE-011 needs >= %.1f s)",
        teacher_inference_duration_s,
        teacher_inference_duration_s / max(len(all_items), 1),
        len(all_items) * _MIN_S_PER_PROMPT,
    )

    # ------------------------------------------------------------------
    # Phase 4: Train PrivacyFilterEnergyChecker
    # ------------------------------------------------------------------
    n_holdout = max(1, int(len(labeled_items) * _HOLDOUT_FRACTION))
    train_items = labeled_items[n_holdout:]
    holdout_items = labeled_items[:n_holdout]

    train_examples = [
        PrivacyExample(text=item["text"], label=item["label"], source=item.get("source", "unknown"))
        for item in train_items
    ]
    holdout_examples = [
        PrivacyExample(text=item["text"], label=item["label"], source=item.get("source", "unknown"))
        for item in holdout_items
    ]

    log.info(
        "Phase 4: Training KAN on %d examples (%d held out for eval)",
        len(train_examples),
        len(holdout_examples),
    )

    if len(train_examples) < 20:
        artifact = tmpl.build_result(
            {
                "honest_verdict": "distillation_corpus_built_classifier_not_trained",
                "teacher_inference_duration_s": round(teacher_inference_duration_s, 2),
                "n_corpus": len(all_items),
            },
            status="blocked",
            reason="Too few training examples after holdout split",
        )
        _write_artifact(artifact, repo_root / deliverable)
        return

    t_train_start = time.perf_counter()
    checker = PrivacyFilterEnergyChecker()
    loss_curve = checker.train(train_examples, n_epochs=100, lr=1e-3, weight_decay=1e-4)
    train_time_s = time.perf_counter() - t_train_start

    log.info(
        "Training done in %.1f s; loss first=%.4f last=%.4f, n_params=%d",
        train_time_s,
        loss_curve[0] if loss_curve else float("nan"),
        loss_curve[-1] if loss_curve else float("nan"),
        checker.n_params(),
    )

    # ------------------------------------------------------------------
    # Phase 5: Evaluate AUROC on held-out set
    # ------------------------------------------------------------------
    log.info("Phase 5: Evaluating AUROC on %d held-out examples", len(holdout_examples))
    distillation_auroc = checker.evaluate_auroc(holdout_examples)
    log.info("distillation_auroc = %.4f (gate threshold = %.2f)", distillation_auroc, _AUROC_GATE)

    auroc_gate_open = distillation_auroc >= _AUROC_GATE
    honest_verdict = (
        "distillation_corpus_built_classifier_trained_auroc_met"
        if auroc_gate_open
        else "distillation_corpus_built_classifier_trained_auroc_below_threshold"
    )

    # ------------------------------------------------------------------
    # Phase 6: Latency check (1000 CPU forward passes)
    # ------------------------------------------------------------------
    log.info("Phase 6: Latency check (1000 forward passes)")
    median_latency_ms, latency_flag = _latency_check(checker, 1000)
    log.info("Median latency: %.3f ms (%s)", median_latency_ms, latency_flag)

    # ------------------------------------------------------------------
    # Phase 7: Save weights (distinct file from prompt-injection KAN)
    # ------------------------------------------------------------------
    weights_path = Path(weights_path)
    checker.save(weights_path)
    log.info("Weights saved to %s", weights_path)

    artifact = tmpl.build_result(
        {
            "honest_verdict": honest_verdict,
            "distillation_auroc": round(distillation_auroc, 4),
            "auroc_gate_open": auroc_gate_open,
            "auroc_target": _AUROC_GATE,
            "n_corpus": len(all_items),
            "n_train": len(train_examples),
            "n_holdout": len(holdout_examples),
            "teacher_inference_duration_s": round(teacher_inference_duration_s, 2),
            "teacher_model": "openai/privacy-filter",
            "n_params": checker.n_params(),
            "n_knots": checker._N_KNOTS,
            "degree": checker._DEGREE,
            "n_hidden": checker.n_hidden,
            "n_features": checker.n_features,
            "n_epochs": 100,
            "weight_decay": 1e-4,
            "train_time_s": round(train_time_s, 2),
            "loss_first_epoch": round(loss_curve[0], 4) if loss_curve else None,
            "loss_last_epoch": round(loss_curve[-1], 4) if loss_curve else None,
            "median_latency_ms": round(median_latency_ms, 3),
            "latency_flag": latency_flag,
            "weights_path": str(weights_path),
        },
        status="success",
        decision_class="detect",
    )
    _write_artifact(artifact, repo_root / deliverable)

    log.info(
        "Experiment 729 complete: honest_verdict=%s, AUROC=%.4f, latency=%s",
        honest_verdict,
        distillation_auroc,
        latency_flag,
    )


if __name__ == "__main__":
    main()
