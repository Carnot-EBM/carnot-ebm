#!/usr/bin/env python3
"""Exp 743: Privacy Filter KAN v2 — teacher-free PII detection via regex features.

**Researcher summary:**
    Exps 729 and 730 were blocked for 2 consecutive cycles because the upstream
    dependency `openai/privacy-filter` was unavailable for download.  Two blocked
    cycles met the governance redesign threshold: the teacher-model dependency is
    retired and replaced with direct feature engineering.

    This experiment:
    1. Builds a fully synthetic labeled corpus (no HuggingFace model downloads).
    2. Trains PrivacyFilterKANv2 on Luhn-valid CC + SSN + email + phone + address PII
       features using contrastive loss directly (no teacher soft labels).
    3. Evaluates AUROC on three held-out datasets.
    4. Checks gate: AUROC >= 0.80 AND min_tp >= 1 per dataset.
    5. Writes results/experiment_743_privacy_filter_v2.json.

**Gate (REQ-SAFE-020):**
    - "privacy_filter_v2_gate_passed_high" if AUROC >= 0.85 AND min_tp >= 1
    - "privacy_filter_v2_gate_passed"      if AUROC >= 0.80 AND min_tp >= 1
    - "privacy_filter_v2_minfp_fail"       if AUROC >= 0.80 but some min_tp == 0
    - "privacy_filter_v2_auroc_fail"       if AUROC < 0.80

Spec: REQ-SAFE-019, REQ-SAFE-020
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path

# Ensure repo root is on sys.path so that `scripts.` and `carnot.` imports work.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Force CPU — no GPU required for this experiment.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.models.privacy_filter_kan_v2 import (  # noqa: E402
    PrivacyFilterKANv2,
    PrivacyExampleV2,
    luhn_complete,
)

# ---------------------------------------------------------------------------
# Corpus construction helpers
# ---------------------------------------------------------------------------

_RNG = random.Random(743)


def _gen_luhn_cc() -> str:
    """Generate a Luhn-valid 16-digit credit card number string (XXXX-XXXX-XXXX-XXXX)."""
    prefix = "".join(str(_RNG.randint(0, 9)) for _ in range(15))
    digits = luhn_complete(prefix)
    return f"{digits[0:4]}-{digits[4:8]}-{digits[8:12]}-{digits[12:16]}"


def _gen_ssn() -> str:
    """Generate a fictional SSN in XXX-XX-XXXX format."""
    a = _RNG.randint(100, 899)
    b = _RNG.randint(10, 99)
    c = _RNG.randint(1000, 9999)
    return f"{a:03d}-{b:02d}-{c:04d}"


def _gen_email() -> str:
    """Generate a plausible email address."""
    users = ["alice", "bob", "carol", "dave", "eve", "frank", "grace", "heidi"]
    domains = ["example.com", "mail.org", "testdomain.net", "corp.io", "webmail.co"]
    user = _RNG.choice(users) + str(_RNG.randint(1, 999))
    domain = _RNG.choice(domains)
    return f"{user}@{domain}"


def _gen_phone() -> str:
    """Generate a US phone number in (NXX) NXX-XXXX format."""
    area = _RNG.randint(200, 999)
    exch = _RNG.randint(200, 999)
    subs = _RNG.randint(1000, 9999)
    return f"({area}) {exch}-{subs}"


def _gen_address() -> str:
    """Generate a fictional US mailing address."""
    num = _RNG.randint(1, 9999)
    streets = ["Oak St", "Maple Ave", "Pine Rd", "Elm Blvd", "Cedar Dr", "Birch Ln"]
    cities = ["Springfield", "Shelbyville", "Ogdenville", "North Haverbrook"]
    states = ["CA", "TX", "NY", "FL", "WA"]
    zips = [f"{_RNG.randint(10000, 99999)}"]
    street = _RNG.choice(streets)
    city = _RNG.choice(cities)
    state = _RNG.choice(states)
    zipcode = zips[0]
    return f"{num} {street}, {city}, {state} {zipcode}"


# ---------------------------------------------------------------------------
# Benign corpus builders (no PII)
# ---------------------------------------------------------------------------

_GSM8K_TEMPLATES = [
    "If a store sells {a} apples at ${b:.2f} each, how much does {c} apples cost?",
    "A train travels {a} km in {b} hours. What is its speed in km/h?",
    "There are {a} students in a class. If {b} are girls, how many are boys?",
    "A rectangle has width {a} cm and height {b} cm. What is its area?",
    "John has ${a:.2f}. He spends ${b:.2f}. How much does he have left?",
    "If {a} workers can finish a job in {b} days, how many days for {c} workers?",
    "A tank holds {a} liters. It is {b}% full. How many liters are in it?",
    "Convert {a} miles to kilometers (1 mile = 1.609 km).",
    "What is {a} percent of {b}?",
    "If x + {a} = {b}, what is x?",
]

_HUMANEVAL_TEMPLATES = [
    "def add(a: int, b: int) -> int:\n    return a + b",
    "def factorial(n: int) -> int:\n    if n == 0: return 1\n    return n * factorial(n - 1)",
    "def is_palindrome(s: str) -> bool:\n    return s == s[::-1]",
    "def max_subarray(nums: list) -> int:\n    best = nums[0]; cur = nums[0]\n    for x in nums[1:]:\n        cur = max(x, cur + x); best = max(best, cur)\n    return best",
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: lo = mid + 1\n        else: hi = mid - 1\n    return -1",
    "def fizzbuzz(n: int) -> list:\n    return ['FizzBuzz' if i%15==0 else 'Fizz' if i%3==0 else 'Buzz' if i%5==0 else str(i) for i in range(1, n+1)]",
    "# Count vowels in a string\ndef count_vowels(s: str) -> int:\n    return sum(1 for c in s.lower() if c in 'aeiou')",
    "def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list): result.extend(flatten(item))\n        else: result.append(item)\n    return result",
]

_WIKI_TEMPLATES = [
    "The mitochondria is the powerhouse of the cell and is responsible for producing ATP.",
    "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
    "The French Revolution began in {a} and fundamentally transformed European politics.",
    "Quantum mechanics describes the behavior of particles at the atomic and subatomic level.",
    "The speed of light in a vacuum is approximately {a} kilometers per second.",
    "DNA consists of two strands that form a double helix structure.",
    "The Earth orbits the Sun at an average distance of {a} million kilometers.",
    "Newton's first law states that an object at rest stays at rest unless acted upon by a force.",
    "Calculus was independently developed by Newton and Leibniz in the {a}th century.",
    "The periodic table organizes elements by their atomic number and chemical properties.",
]


def _build_benign_corpus(n: int = 1000) -> list[PrivacyExampleV2]:
    """Build n benign samples from GSM8K-style, HumanEval-style, and Wikipedia-style text."""
    samples: list[PrivacyExampleV2] = []
    templates = _GSM8K_TEMPLATES + _HUMANEVAL_TEMPLATES + _WIKI_TEMPLATES
    for i in range(n):
        tmpl = templates[i % len(templates)]
        try:
            text = tmpl.format(
                a=_RNG.randint(2, 100),
                b=_RNG.uniform(1, 50),
                c=_RNG.randint(1, 20),
            )
        except (KeyError, IndexError):
            text = tmpl
        samples.append(PrivacyExampleV2(text=text, label="benign", source="synthetic_benign"))
    return samples


# ---------------------------------------------------------------------------
# PII corpus builders
# ---------------------------------------------------------------------------

def _build_pii_corpus(n_each: int = 200) -> list[PrivacyExampleV2]:
    """Build PII samples: n_each of each of 5 PII types (CC, SSN, email, phone, address)."""
    samples: list[PrivacyExampleV2] = []

    contexts = [
        "My {} is ",
        "Please charge {}",
        "You can reach me at {}",
        "Send the package to {}",
        "Billing info: {}",
        "Account details: {}",
        "Contact: {}",
    ]

    # Credit card numbers (Luhn-valid).
    for _ in range(n_each):
        cc = _gen_luhn_cc()
        ctx = _RNG.choice(contexts).format(cc)
        samples.append(PrivacyExampleV2(text=ctx, label="pii", source="synthetic_cc"))

    # SSN patterns.
    for _ in range(n_each):
        ssn = _gen_ssn()
        ctx = _RNG.choice(contexts).format(ssn)
        samples.append(PrivacyExampleV2(text=ctx, label="pii", source="synthetic_ssn"))

    # Email addresses.
    for _ in range(n_each):
        email = _gen_email()
        ctx = _RNG.choice(contexts).format(email)
        samples.append(PrivacyExampleV2(text=ctx, label="pii", source="synthetic_email"))

    # Phone numbers.
    for _ in range(n_each):
        phone = _gen_phone()
        ctx = _RNG.choice(contexts).format(phone)
        samples.append(PrivacyExampleV2(text=ctx, label="pii", source="synthetic_phone"))

    # Mailing addresses.
    for _ in range(n_each):
        addr = _gen_address()
        ctx = _RNG.choice(contexts).format(addr)
        samples.append(PrivacyExampleV2(text=ctx, label="pii", source="synthetic_address"))

    return samples


# ---------------------------------------------------------------------------
# Cross-dataset evaluation helpers
# ---------------------------------------------------------------------------

def _build_dataset1_holdout(n: int = 200) -> list[PrivacyExampleV2]:
    """Dataset 1: novel Luhn-valid CC + SSN hold-out (not in training split)."""
    samples: list[PrivacyExampleV2] = []
    for _ in range(n // 2):
        samples.append(PrivacyExampleV2(
            text=f"New card: {_gen_luhn_cc()}",
            label="pii",
            source="holdout_cc",
        ))
        samples.append(PrivacyExampleV2(
            text=f"My SSN: {_gen_ssn()}",
            label="pii",
            source="holdout_ssn",
        ))
    # Add benign counterparts.
    for i in range(n):
        samples.append(PrivacyExampleV2(
            text=_GSM8K_TEMPLATES[i % len(_GSM8K_TEMPLATES)].format(
                a=i + 1, b=float(i + 2), c=i + 3
            ),
            label="benign",
            source="holdout_benign",
        ))
    return samples


def _build_dataset2_mixed(n: int = 200) -> list[PrivacyExampleV2]:
    """Dataset 2: GSM8K-style word problems with embedded PII."""
    samples: list[PrivacyExampleV2] = []
    problem_templates = [
        "Alice has card number {} and needs to pay ${:.2f}. How much remains from ${}?",
        "Bob's SSN is {} and he earns ${:.2f}/hr. What is his weekly pay for {} hours?",
        "Contact {} at {} about the {} kg shipment. What is the cost at ${}?",
    ]
    for i in range(n):
        pii_val = [_gen_luhn_cc(), _gen_ssn(), _gen_email(), _gen_phone()][i % 4]
        tmpl = problem_templates[i % len(problem_templates)]
        try:
            text = tmpl.format(pii_val, float((i + 1) * 3.5), i + 10, i + 1)
        except (IndexError, TypeError):
            text = f"Problem {i}: {pii_val}"
        samples.append(PrivacyExampleV2(text=text, label="pii", source="mixed_gsm8k_pii"))
        samples.append(PrivacyExampleV2(
            text=f"Standard problem {i}: if x + {i+1} = {i*2+3}, solve for x.",
            label="benign",
            source="mixed_benign",
        ))
    return samples


def _build_dataset3_code(n: int = 200) -> list[PrivacyExampleV2]:
    """Dataset 3: code snippets with embedded PII (emails, API key-like SSNs)."""
    samples: list[PrivacyExampleV2] = []
    for i in range(n):
        email = _gen_email()
        # API keys that look like SSN-style strings (common in config leaks).
        ssn_like = _gen_ssn().replace("-", "")
        code_pii = f'EMAIL = "{email}"\nAPI_KEY = "{ssn_like}"'
        samples.append(PrivacyExampleV2(text=code_pii, label="pii", source="code_pii"))
        clean_code = f"def compute_{i}(x):\n    return x * {i + 1} + {i}"
        samples.append(PrivacyExampleV2(text=clean_code, label="benign", source="code_benign"))
    return samples


# ---------------------------------------------------------------------------
# Gate metric helpers
# ---------------------------------------------------------------------------

def _calibrate_threshold(
    model: PrivacyFilterKANv2,
    benign_train: list[PrivacyExampleV2],
    pii_train: list[PrivacyExampleV2],
) -> float:
    """Compute a calibrated classification threshold from training data energies.

    Uses the midpoint between mean benign energy and mean PII energy on the
    training corpus.  This is more robust than a hardcoded 0.0 threshold because
    the EBM energy scale is not normalised to any fixed range — it depends on
    initialization and the contrastive margin.

    Why not threshold=0.0:
        Contrastive training with margin=1.0 pushes E(pii) - E(benign) >= 1.0,
        but does not anchor the absolute scale.  If E(benign) = -2.0 and
        E(pii) = -1.0, the model correctly ranks all samples but threshold=0.0
        would produce zero true positives.  The midpoint threshold (-1.5) works.

    Spec: REQ-SAFE-020
    """
    benign_energies = [model.energy(e.text) for e in benign_train]
    pii_energies = [model.energy(e.text) for e in pii_train]
    mean_benign = sum(benign_energies) / max(len(benign_energies), 1)
    mean_pii = sum(pii_energies) / max(len(pii_energies), 1)
    return (mean_benign + mean_pii) / 2.0


def _confusion_at_threshold(
    model: PrivacyFilterKANv2,
    examples: list[PrivacyExampleV2],
    threshold: float,
) -> dict:
    """Compute confusion matrix metrics at a calibrated energy threshold.

    Returns dict with tp, fp, tn, fn.

    Spec: REQ-SAFE-020
    """
    tp = fp = tn = fn = 0
    for ex in examples:
        e = model.energy(ex.text)
        predicted_pii = e > threshold
        actual_pii = ex.label == "pii"
        if predicted_pii and actual_pii:
            tp += 1
        elif predicted_pii and not actual_pii:
            fp += 1
        elif not predicted_pii and actual_pii:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Exp 743: PrivacyFilterKANv2 training, evaluation, and gate check."""
    tmpl = ExperimentTemplate(
        exp_id=743,
        title="Privacy Filter KAN v2 — Teacher-Free PII Detection",
        deliverable="results/experiment_743_privacy_filter_v2.json",
    )
    tmpl.setup()

    result_path = str(_REPO / "results" / "experiment_743_privacy_filter_v2.json")

    with ExperimentTimeoutWatchdog(743, timeout_minutes=60, result_path=result_path):
        # Step 1: Build corpus.
        benign_all = _build_benign_corpus(1000)
        pii_all = _build_pii_corpus(n_each=200)  # 200 × 5 = 1000 PII samples

        # 80/20 train/eval split (same random seed for reproducibility).
        _RNG.seed(743)
        random.shuffle(benign_all)
        random.shuffle(pii_all)

        n_benign_train = int(len(benign_all) * 0.8)
        n_pii_train = int(len(pii_all) * 0.8)

        benign_train = benign_all[:n_benign_train]
        benign_eval = benign_all[n_benign_train:]
        pii_train = pii_all[:n_pii_train]
        pii_eval = pii_all[n_pii_train:]

        print(f"Corpus: {len(benign_train)} benign train, {len(pii_train)} PII train")
        print(f"Eval:   {len(benign_eval)} benign eval, {len(pii_eval)} PII eval")

        # Step 2: Train model.
        model = PrivacyFilterKANv2(n_features=23, n_hidden=32)
        print(f"Model parameters: {model.n_params}")
        loss_curve = model.train(benign_train, pii_train, n_epochs=100, lr=1e-3)
        print(f"Training complete. Final loss: {loss_curve[-1]:.4f}")

        # Step 3: Evaluate on in-distribution held-out set.
        eval_all = benign_eval + pii_eval
        indist_auroc = model.evaluate_auroc(eval_all)
        print(f"In-distribution AUROC: {indist_auroc:.4f}")

        # Calibrate threshold from training data energies.
        # Using midpoint between mean benign and mean PII energy — see _calibrate_threshold.
        calibrated_threshold = _calibrate_threshold(model, benign_train, pii_train)
        print(f"Calibrated threshold: {calibrated_threshold:.4f}")

        # Step 4: Cross-dataset evaluation.
        ds1 = _build_dataset1_holdout(200)
        ds2 = _build_dataset2_mixed(200)
        ds3 = _build_dataset3_code(200)

        ds_names = ["holdout_synthetic", "mixed_gsm8k_pii", "code_snippet_pii"]
        datasets = [ds1, ds2, ds3]

        per_dataset_auroc = []
        per_dataset_min_tp = []
        per_dataset_confusion = []

        for name, ds in zip(ds_names, datasets):
            auroc = model.evaluate_auroc(ds)
            confusion = _confusion_at_threshold(model, ds, threshold=calibrated_threshold)
            min_tp = confusion["tp"]
            per_dataset_auroc.append(auroc)
            per_dataset_min_tp.append(min_tp)
            per_dataset_confusion.append(confusion)
            print(f"  [{name}] AUROC={auroc:.4f}, tp={min_tp}, fp={confusion['fp']}")

        mean_auroc = sum(per_dataset_auroc) / len(per_dataset_auroc)
        all_min_tp = min(per_dataset_min_tp)

        # Step 5: Gate check.
        gate_passed = mean_auroc >= 0.80 and all_min_tp >= 1

        if mean_auroc >= 0.85 and all_min_tp >= 1:
            honest_verdict = "privacy_filter_v2_gate_passed_high"
        elif mean_auroc >= 0.80 and all_min_tp >= 1:
            honest_verdict = "privacy_filter_v2_gate_passed"
        elif mean_auroc < 0.80:
            honest_verdict = "privacy_filter_v2_auroc_fail"
        else:
            honest_verdict = "privacy_filter_v2_minfp_fail"

        print(f"\nMean cross-dataset AUROC: {mean_auroc:.4f}")
        print(f"Min true positives across datasets: {all_min_tp}")
        print(f"Gate passed: {gate_passed}")
        print(f"Verdict: {honest_verdict}")

        # Step 6: Save weights.
        weights_path = _REPO / "python" / "carnot" / "models" / "privacy_filter_kan_v2.json"
        model.save(weights_path)
        weights_saved = weights_path.exists()
        print(f"Weights saved to: {weights_path}")

        # Step 7: Write gate result.
        gate_result = {
            "experiment": 743,
            "mean_auroc": mean_auroc,
            "per_dataset_auroc": dict(zip(ds_names, per_dataset_auroc)),
            "per_dataset_min_tp": dict(zip(ds_names, per_dataset_min_tp)),
            "per_dataset_confusion": dict(zip(ds_names, per_dataset_confusion)),
            "gate_passed": gate_passed,
            "honest_verdict": honest_verdict,
        }
        gate_path = _REPO / "results" / "privacy_filter_v2_gate.json"
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gate_path, "w") as fh:
            json.dump(gate_result, fh, indent=2)
        print(f"Gate result written to: {gate_path}")

        # Step 8: Build deliverable artifact.
        artifact = tmpl.build_result(
            {
                "auroc": mean_auroc,
                "indist_auroc": indist_auroc,
                "per_dataset_auroc": per_dataset_auroc,
                "per_dataset_min_tp": per_dataset_min_tp,
                "per_dataset_confusion": [dict(c) for c in per_dataset_confusion],
                "dataset_names": ds_names,
                "gate_passed": gate_passed,
                "weights_saved": weights_saved,
                "weights_path": str(weights_path),
                "honest_verdict": honest_verdict,
                "n_params": model.n_params,
                "n_train_benign": len(benign_train),
                "n_train_pii": len(pii_train),
                "final_loss": loss_curve[-1],
                "teacher_used": False,
                "upstream_dependency": "none",
                "calibrated_threshold": calibrated_threshold,
            },
            status="success" if gate_passed else "partial",
        )
        tmpl._output_path.parent.mkdir(parents=True, exist_ok=True)
        tmpl._output_path.write_text(json.dumps(artifact, indent=2))

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
