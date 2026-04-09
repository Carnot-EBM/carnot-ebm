#!/usr/bin/env python3
"""Experiment 72: Autoresearch Self-Verification via Ising Constraint Gate.

**Researcher summary:**
    Dog-foods the Carnot constraint pipeline on the autoresearch loop's OWN
    hypothesis outputs. Extracts verifiable claims from hypothesis code (via
    Exp 48 AST extraction) and output text (via Exp 49 NL extraction plus
    numeric-claim patterns), then verifies them using ComposedEnergy and
    Ising sampling. Evaluates whether an Ising-based "fourth gate"
    (constraint satisfaction) catches bogus hypotheses that the existing
    three gates (energy, time, memory) miss.

**Detailed explanation for engineers:**
    The autoresearch conductor (scripts/research_conductor.py) runs
    experiments autonomously. Currently it uses three gates to decide
    whether to accept a hypothesis output:

    1. **Energy gate** — did the energy metric improve?
    2. **Time gate** — did the experiment finish within the time budget?
    3. **Memory gate** — did the experiment stay within memory limits?

    These gates are necessary but not sufficient. A hypothesis can pass all
    three by gaming the metrics: inflating improvement via numerical hacks,
    measuring the wrong metric, or hard-coding results. This experiment
    tests whether adding an Ising constraint-satisfaction gate (the "fourth
    gate") catches these bogus hypotheses.

    **Pipeline:**
    1. ``extract_hypothesis_claims(code, output)`` — combines Exp 48 AST
       extraction (type constraints, bounds, returns) with Exp 49 NL
       extraction (factual assertions, implications) plus NEW numeric-claim
       patterns (e.g., "energy decreased by 15%", "loss < 0.01").
    2. ``verify_hypothesis(claims, ground_truth)`` — builds a ComposedEnergy
       from extracted constraints, runs Ising verification, returns a
       verification certificate.
    3. Simulates 20 mock hypothesis outputs (10 correct, 10 bogus) and
       measures the false acceptance rate WITH vs WITHOUT the Ising gate.
    4. Prints a confusion matrix: gate combination x (accept/reject) x
       (correct/bogus).

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_72_autoresearch_self_verify.py

Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-002
"""

from __future__ import annotations

import ast
import os
import re
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


# ---------------------------------------------------------------------------
# 1. extract_hypothesis_claims — parse code + output into verifiable claims
# ---------------------------------------------------------------------------

def extract_numeric_claims(text: str) -> list[dict[str, Any]]:
    """Extract numeric claims from hypothesis output text.

    **Detailed explanation for engineers:**
        Scans free text for common patterns that autoresearch outputs produce:
        - "energy decreased by X%"  / "improved by X%"
        - "loss = X" / "final loss: X"
        - "metric < threshold" / "metric > threshold"
        - "X% improvement" / "Xx speedup"
        - "accuracy: X%" / "error rate: X%"

        Each match becomes a claim dict with:
        - claim_type: "numeric_improvement", "numeric_bound", "numeric_value"
        - claim_text: the matched substring
        - constraint: a dict encoding the numeric relationship for Ising
          verification (e.g., {"kind": "bound", "metric": "energy",
          "direction": "decreased", "magnitude": 15.0})

    Returns:
        List of numeric claim dicts.
    """
    claims: list[dict[str, Any]] = []
    text_lower = text.lower()

    # Pattern 1: "energy/loss/metric decreased/increased by X%"
    for m in re.finditer(
        r"(\w+)\s+(?:decreased|reduced|improved|increased)\s+by\s+([\d.]+)\s*%",
        text_lower,
    ):
        metric, magnitude = m.group(1), float(m.group(2))
        claims.append({
            "claim_type": "numeric_improvement",
            "claim_text": m.group(0),
            "constraint": {
                "kind": "improvement",
                "metric": metric,
                "magnitude": magnitude,
            },
        })

    # Pattern 2: "X% improvement/speedup"
    for m in re.finditer(
        r"([\d.]+)\s*%\s*(?:improvement|speedup|faster|reduction)",
        text_lower,
    ):
        magnitude = float(m.group(1))
        claims.append({
            "claim_type": "numeric_improvement",
            "claim_text": m.group(0),
            "constraint": {
                "kind": "improvement",
                "metric": "general",
                "magnitude": magnitude,
            },
        })

    # Pattern 3: "Xx speedup" (e.g., "2.5x speedup")
    for m in re.finditer(r"([\d.]+)\s*x\s*(?:speedup|faster)", text_lower):
        magnitude = float(m.group(1))
        claims.append({
            "claim_type": "numeric_improvement",
            "claim_text": m.group(0),
            "constraint": {
                "kind": "speedup",
                "metric": "time",
                "factor": magnitude,
            },
        })

    # Pattern 4: "loss/energy = X" or "final loss: X"
    for m in re.finditer(
        r"(?:final\s+)?(\w+)\s*[=:]\s*(-?[\d]+(?:\.\d+)?(?:e[+-]?\d+)?)",
        text_lower,
    ):
        metric, value = m.group(1), float(m.group(2))
        if metric in ("loss", "energy", "error", "mse", "rmse", "mae"):
            claims.append({
                "claim_type": "numeric_value",
                "claim_text": m.group(0),
                "constraint": {
                    "kind": "value",
                    "metric": metric,
                    "value": value,
                },
            })

    # Pattern 5: "metric < threshold" or "metric > threshold"
    for m in re.finditer(
        r"(\w+)\s*([<>]=?)\s*([\d]+(?:\.\d+)?(?:e[+-]?\d+)?)",
        text_lower,
    ):
        metric, op, threshold = m.group(1), m.group(2), float(m.group(3))
        if metric in ("loss", "energy", "error", "accuracy", "score"):
            claims.append({
                "claim_type": "numeric_bound",
                "claim_text": m.group(0),
                "constraint": {
                    "kind": "bound",
                    "metric": metric,
                    "operator": op,
                    "threshold": threshold,
                },
            })

    return claims


def extract_hypothesis_claims(
    code: str, output: str
) -> list[dict[str, Any]]:
    """Parse hypothesis code and output text into verifiable claims.

    **Detailed explanation for engineers:**
        Combines three extraction strategies:

        1. **AST extraction (Exp 48)** — parses the hypothesis Python code to
           extract type annotations, loop bounds, return types, and
           initialization constraints. These catch structural bugs like wrong
           return types, uninitialized variables, or impossible loop bounds.

        2. **NL extraction (Exp 49)** — parses the output text for factual
           claims ("X is Y"), implications ("if X then Y"), and logical
           relationships. These catch logical contradictions in the
           hypothesis's reasoning.

        3. **Numeric extraction (new)** — parses the output text for numeric
           claims about improvements, metrics, and bounds. These catch the
           most common autoresearch failure mode: claiming improvement when
           the numbers don't add up.

        Each claim is a dict with at minimum:
        - claim_type: string categorizing the claim
        - claim_text: the raw text or code that produced this claim
        - constraint: a dict encoding the verifiable relationship

    Args:
        code: Python source code of the hypothesis implementation.
        output: Text output produced by running the hypothesis.

    Returns:
        List of claim dicts combining all three extraction strategies.
    """
    claims: list[dict[str, Any]] = []

    # --- Strategy 1: AST extraction (from Exp 48) ---
    from experiment_48_code_constraints import code_to_constraints

    try:
        code_constraints = code_to_constraints(code)
        for c in code_constraints:
            claims.append({
                "claim_type": f"code_{c['kind']}",
                "claim_text": c["description"],
                "constraint": c,
            })
    except SyntaxError:
        # If the hypothesis code doesn't parse, that's itself a violation.
        claims.append({
            "claim_type": "code_syntax_error",
            "claim_text": "Hypothesis code failed to parse",
            "constraint": {"kind": "syntax", "satisfied": False},
        })

    # --- Strategy 2: NL extraction (from Exp 49) ---
    from experiment_49_nl_constraints import extract_claims as nl_extract

    nl_claims = nl_extract(output)
    for c in nl_claims:
        claims.append({
            "claim_type": f"nl_{c['claim_type']}",
            "claim_text": c.get("raw", str(c)),
            "constraint": c,
        })

    # --- Strategy 3: Numeric extraction (new for Exp 72) ---
    numeric_claims = extract_numeric_claims(output)
    claims.extend(numeric_claims)

    return claims


# ---------------------------------------------------------------------------
# 2. verify_hypothesis — build ComposedEnergy and verify via Ising
# ---------------------------------------------------------------------------

class ClaimConstraint:
    """Wraps a single claim as a ConstraintTerm for ComposedEnergy.

    **Detailed explanation for engineers:**
        Each extracted claim becomes a differentiable energy term. The energy
        encoding depends on claim type:

        - **code_type / code_return_type / code_initialization**: Binary
          energy based on the ``satisfied`` field from AST extraction.
          Energy = 0 if satisfied, 1 if violated.

        - **code_loop_bound**: Always satisfied (range() guarantees bounds),
          so energy is always 0.

        - **numeric_improvement**: Energy based on whether the claimed
          improvement magnitude is plausible (0-100% range). Improvements
          claiming >100% or negative values get high energy.

        - **numeric_bound**: Energy based on whether the claimed metric
          value satisfies the stated bound when compared to ground truth.

        - **nl_***: Factual and logical claims get binary energy (0 or 1)
          based on Ising consistency checking of the full claim set.

        The energy is a scalar in [0, 1] so that weights in ComposedEnergy
        are meaningful across different claim types.
    """

    def __init__(self, claim: dict[str, Any], index: int) -> None:
        self._claim = claim
        self._index = index
        self._satisfied: bool | None = None

    @property
    def name(self) -> str:
        return f"claim_{self._index}_{self._claim['claim_type']}"

    @property
    def satisfaction_threshold(self) -> float:
        return 0.5

    def energy(self, x: Any) -> Any:
        """Return energy based on claim satisfaction.

        Energy is 0.0 if the claim is verified, 1.0 if violated. The
        input x is ignored — claim satisfaction is determined during
        the verify_hypothesis() call, not from a continuous state vector.
        This is intentional: autoresearch claims are discrete propositions,
        not continuous configurations.
        """
        import jax.numpy as jnp
        if self._satisfied is True:
            return jnp.float32(0.0)
        elif self._satisfied is False:
            return jnp.float32(1.0)
        # Unknown — treat as mild violation to flag for review.
        return jnp.float32(0.5)

    def grad_energy(self, x: Any) -> Any:
        """Gradient is zero because energy is independent of x.

        Claim verification is discrete (pass/fail from Ising sampling),
        not continuous. The gradient is zero everywhere.
        """
        import jax.numpy as jnp
        return jnp.zeros_like(x)

    def is_satisfied(self, x: Any) -> bool:
        return self._satisfied is True

    def set_satisfied(self, value: bool) -> None:
        """Set the satisfaction status after verification."""
        self._satisfied = value


def _verify_code_claims(claims: list[dict[str, Any]]) -> dict[int, bool]:
    """Verify code-level claims using Exp 48 verification logic.

    **Detailed explanation for engineers:**
        Code claims have their satisfaction determined at extraction time
        (the AST analysis already knows if types match, variables are
        initialized, etc.). We just read off the ``satisfied`` field.
        Claims where ``satisfied`` is None are deferred to Ising.

    Returns:
        Dict mapping claim index → satisfaction boolean.
    """
    results: dict[int, bool] = {}
    for i, claim in enumerate(claims):
        ct = claim["claim_type"]
        if not ct.startswith("code_"):
            continue
        c = claim["constraint"]
        if c.get("satisfied") is True:
            results[i] = True
        elif c.get("satisfied") is False:
            results[i] = False
        # satisfied=None means deferred to Ising — handled later.
    return results


def _verify_numeric_claims(
    claims: list[dict[str, Any]],
    ground_truth: dict[str, Any] | None = None,
) -> dict[int, bool]:
    """Verify numeric claims against ground truth if available.

    **Detailed explanation for engineers:**
        Numeric claims make assertions about metric values and improvements.
        Verification depends on what ground truth is available:

        - If ground_truth provides the actual metric values, we can check
          whether claimed improvements and bounds are correct.
        - If no ground truth, we apply heuristic checks:
          * Improvements > 100% are suspicious (flag as violated).
          * Negative improvements are violated.
          * Speedups > 10x without evidence are suspicious.
          * Claimed loss/energy values must be non-negative.

    Returns:
        Dict mapping claim index → satisfaction boolean.
    """
    gt = ground_truth or {}
    results: dict[int, bool] = {}

    for i, claim in enumerate(claims):
        ct = claim["claim_type"]
        if ct not in ("numeric_improvement", "numeric_bound", "numeric_value"):
            continue
        c = claim["constraint"]
        kind = c["kind"]

        if kind == "improvement":
            magnitude = c["magnitude"]
            # Heuristic: improvements must be in (0, 100] range.
            if magnitude <= 0 or magnitude > 100:
                results[i] = False
                continue
            # If ground truth has before/after values, check the math.
            metric = c["metric"]
            if f"{metric}_before" in gt and f"{metric}_after" in gt:
                before = gt[f"{metric}_before"]
                after = gt[f"{metric}_after"]
                if before > 0:
                    actual_pct = ((before - after) / before) * 100
                    results[i] = abs(actual_pct - magnitude) < 5.0  # 5% tolerance
                else:
                    results[i] = False
            else:
                # No ground truth — trust the claim within heuristic bounds.
                results[i] = True

        elif kind == "speedup":
            factor = c["factor"]
            # Heuristic: speedups must be > 1.0 and < 10.0 without evidence.
            if factor <= 1.0 or factor > 10.0:
                results[i] = False
            else:
                results[i] = True

        elif kind == "value":
            value = c["value"]
            metric = c["metric"]
            # Loss/energy must be non-negative.
            if metric in ("loss", "energy", "error", "mse", "rmse", "mae"):
                results[i] = value >= 0
            else:
                results[i] = True

            # If ground truth provides the actual value, check closeness.
            if metric in gt:
                actual = gt[metric]
                results[i] = abs(value - actual) / max(abs(actual), 1e-10) < 0.1

        elif kind == "bound":
            metric = c["metric"]
            op = c["operator"]
            threshold = c["threshold"]
            if metric in gt:
                actual = gt[metric]
                if op == "<":
                    results[i] = actual < threshold
                elif op == "<=":
                    results[i] = actual <= threshold
                elif op == ">":
                    results[i] = actual > threshold
                elif op == ">=":
                    results[i] = actual >= threshold
                else:
                    results[i] = True
            else:
                results[i] = True

    return results


def _verify_nl_claims_via_ising(claims: list[dict[str, Any]]) -> dict[int, bool]:
    """Verify NL claims for logical consistency via Ising sampling.

    **Detailed explanation for engineers:**
        Natural language claims (implications, factual assertions, etc.)
        are encoded as Ising propositions and sampled for consistency.
        This uses the same pipeline as Exp 49: build Ising biases and
        couplings from the claims, run ParallelIsingSampler, and check
        whether a zero-violation assignment exists.

        If the claims are logically consistent, all get satisfied=True.
        If there are contradictions, the claims involved in the
        contradiction get satisfied=False.

    Returns:
        Dict mapping claim index → satisfaction boolean.
    """
    from experiment_45_logical_consistency import encode_claims_as_ising, count_violations
    from carnot.samplers.parallel_ising import ParallelIsingSampler, AnnealingSchedule
    import jax.numpy as jnp
    import jax.random as jrandom

    nl_indices = [
        i for i, c in enumerate(claims) if c["claim_type"].startswith("nl_")
    ]
    if not nl_indices:
        return {}

    # Build Ising propositions from NL claims.
    prop_map: dict[str, int] = {}

    def get_prop(name: str) -> int:
        if name not in prop_map:
            prop_map[name] = len(prop_map)
        return prop_map[name]

    ising_claims: list[dict] = []
    for idx in nl_indices:
        c = claims[idx]["constraint"]
        ct = c.get("claim_type", "")

        if ct in ("factual", "factual_relation"):
            prop_idx = get_prop(c.get("raw", str(idx)))
            ising_claims.append({"type": "true", "prop": prop_idx})

        elif ct == "implication":
            ante_idx = get_prop(c["antecedent"])
            cons_idx = get_prop(c["consequent"])
            ising_claims.append({"type": "implies", "from": ante_idx, "to": cons_idx})

        elif ct == "conjunction":
            left_idx = get_prop(c["left"])
            right_idx = get_prop(c["right"])
            ising_claims.append({"type": "and", "props": [left_idx, right_idx]})

        elif ct == "negation":
            subj_pred = f"{c['subject']} {c['predicate']}"
            prop_idx = get_prop(subj_pred)
            ising_claims.append({"type": "false", "prop": prop_idx})

        else:
            # Universal, disjunction, exclusion — assert as true.
            prop_idx = get_prop(c.get("raw", str(idx)))
            ising_claims.append({"type": "true", "prop": prop_idx})

    n_props = len(prop_map)
    results: dict[int, bool] = {}

    if n_props < 2 or len(ising_claims) < 2:
        # Too few propositions for meaningful Ising check — pass all.
        for idx in nl_indices:
            results[idx] = True
        return results

    biases_np, edge_pairs, weights_list = encode_claims_as_ising(ising_claims, n_props)

    if len(edge_pairs) == 0:
        # No conflicts — all consistent.
        for idx in nl_indices:
            results[idx] = True
        return results

    J = np.zeros((n_props, n_props), dtype=np.float32)
    for k, (i, j) in enumerate(edge_pairs):
        J[i, j] += weights_list[k]
        J[j, i] += weights_list[k]

    sampler = ParallelIsingSampler(
        n_warmup=500,
        n_samples=30,
        steps_per_sample=10,
        schedule=AnnealingSchedule(0.1, 8.0),
        use_checkerboard=True,
    )

    samples = sampler.sample(
        jrandom.PRNGKey(72),
        jnp.array(biases_np, dtype=jnp.float32),
        jnp.array(J, dtype=jnp.float32),
        beta=8.0,
    )

    # Find the assignment with fewest violations.
    best_violations = len(ising_claims)
    best_assignment: dict[int, bool] = {}
    for s_idx in range(samples.shape[0]):
        assignment = {i: bool(samples[s_idx, i]) for i in range(n_props)}
        v = count_violations(ising_claims, assignment)
        if v < best_violations:
            best_violations = v
            best_assignment = assignment

    # Map back to claim indices.
    # Build reverse map: for each nl_index, find its proposition index.
    prop_to_claim_idx: dict[int, list[int]] = {}
    claim_to_prop: dict[int, int] = {}
    for idx in nl_indices:
        c = claims[idx]["constraint"]
        ct = c.get("claim_type", "")
        if ct in ("factual", "factual_relation"):
            key = c.get("raw", str(idx))
        elif ct == "negation":
            key = f"{c['subject']} {c['predicate']}"
        else:
            key = c.get("raw", str(idx))
        if key in prop_map:
            prop_idx = prop_map[key]
            claim_to_prop[idx] = prop_idx

    for idx in nl_indices:
        if idx in claim_to_prop:
            prop_idx = claim_to_prop[idx]
            results[idx] = best_assignment.get(prop_idx, True)
        else:
            results[idx] = best_violations == 0

    return results


def verify_hypothesis(
    claims: list[dict[str, Any]],
    ground_truth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build ComposedEnergy from extracted claims and run Ising verification.

    **Detailed explanation for engineers:**
        Orchestrates three verification strategies:

        1. Code claims — verified via AST analysis (from extraction).
        2. Numeric claims — verified against ground truth or heuristics.
        3. NL claims — verified for logical consistency via Ising sampling.

        Then builds a ComposedEnergy from all claims, with each claim
        wrapped as a ClaimConstraint. The ComposedEnergy's verify() method
        produces a VerificationResult with per-claim decomposition and
        an overall verdict.

    Args:
        claims: List of claim dicts from extract_hypothesis_claims().
        ground_truth: Optional dict of actual metric values for numeric
            verification. Keys like "energy_before", "energy_after",
            "loss", etc.

    Returns:
        Dict with keys:
        - verified: int count of verified claims
        - violated: int count of violated claims
        - uncovered: int count of claims that couldn't be verified
        - certificate: dict with per-claim details and overall verdict
    """
    import jax.numpy as jnp
    from carnot.verify.constraint import ComposedEnergy

    if not claims:
        return {
            "verified": 0,
            "violated": 0,
            "uncovered": 0,
            "certificate": {"verdict": "EMPTY", "details": []},
        }

    # Step 1: Verify each claim category.
    code_results = _verify_code_claims(claims)
    numeric_results = _verify_numeric_claims(claims, ground_truth)
    nl_results = _verify_nl_claims_via_ising(claims)

    # Step 2: Merge results and build ComposedEnergy.
    all_results = {**code_results, **numeric_results, **nl_results}
    composed = ComposedEnergy(input_dim=1)
    claim_terms: list[ClaimConstraint] = []

    for i, claim in enumerate(claims):
        term = ClaimConstraint(claim, i)
        if i in all_results:
            term.set_satisfied(all_results[i])
        # Claims not in any results remain as "uncovered" (satisfied=None).
        claim_terms.append(term)
        composed.add_constraint(term, weight=1.0)

    # Step 3: Run verification.
    x_dummy = jnp.zeros(1)
    result = composed.verify(x_dummy)

    # Step 4: Count verified / violated / uncovered.
    n_verified = sum(1 for t in claim_terms if t._satisfied is True)
    n_violated = sum(1 for t in claim_terms if t._satisfied is False)
    n_uncovered = sum(1 for t in claim_terms if t._satisfied is None)

    details = []
    for i, t in enumerate(claim_terms):
        details.append({
            "index": i,
            "claim_type": claims[i]["claim_type"],
            "claim_text": claims[i]["claim_text"],
            "satisfied": t._satisfied,
        })

    return {
        "verified": n_verified,
        "violated": n_violated,
        "uncovered": n_uncovered,
        "certificate": {
            "verdict": "PASS" if n_violated == 0 and n_uncovered == 0 else "FAIL",
            "total_energy": result.total_energy,
            "details": details,
        },
    }


# ---------------------------------------------------------------------------
# 3. Mock hypothesis outputs — 10 correct, 10 bogus
# ---------------------------------------------------------------------------

def get_mock_hypotheses() -> list[dict[str, Any]]:
    """Create 20 mock hypothesis outputs for testing the fourth gate.

    **Detailed explanation for engineers:**
        Each hypothesis simulates what the autoresearch loop would produce:
        a Python code snippet (the hypothesis implementation) and text output
        (what the experiment printed). The three existing gates (energy, time,
        memory) are provided as boolean fields so we can compare gate
        combinations.

        The 10 CORRECT hypotheses represent genuine improvements:
        - Valid code with correct types and bounds
        - Honest metric claims backed by ground truth
        - Reasonable improvement magnitudes

        The 10 BOGUS hypotheses represent common autoresearch failure modes:
        - Numerical hacks (hard-coded results, division by zero protection)
        - Wrong metric (measuring training loss instead of test loss)
        - Inflated claims (claiming 200% improvement)
        - Type mismatches in code
        - Logical contradictions in output text
        - Uninitialized variables

        Each bogus hypothesis is designed to pass at least one of the three
        existing gates while being objectively wrong.

    Returns:
        List of 20 hypothesis dicts with keys: name, code, output,
        is_correct, gate_energy, gate_time, gate_memory, ground_truth.
    """
    hypotheses: list[dict[str, Any]] = []

    # ===== 10 CORRECT HYPOTHESES =====

    hypotheses.append({
        "name": "Correct #1: Learning rate tuning",
        "code": '''
def train_step(params: dict, lr: float, data: list) -> float:
    loss = 0.0
    for i in range(len(data)):
        loss += (params.get("w", 0.0) * data[i]) ** 2
    return loss / len(data)
''',
        "output": "Energy decreased by 12%. Final loss: 0.034. loss < 0.05.",
        "is_correct": True,
        "gate_energy": True,
        "gate_time": True,
        "gate_memory": True,
        "ground_truth": {"energy_before": 0.0386, "energy_after": 0.034, "loss": 0.034},
    })

    hypotheses.append({
        "name": "Correct #2: Batch size increase",
        "code": '''
def process_batch(data: list, batch_size: int) -> float:
    total = 0.0
    for i in range(0, len(data), batch_size):
        chunk = data[i:i + batch_size]
        total += sum(chunk)
    return total
''',
        "output": "Energy decreased by 8%. 1.5x speedup. Final loss: 0.041.",
        "is_correct": True,
        "gate_energy": True,
        "gate_time": True,
        "gate_memory": True,
        "ground_truth": {"energy_before": 0.0446, "energy_after": 0.041, "loss": 0.041},
    })

    hypotheses.append({
        "name": "Correct #3: Regularization added",
        "code": '''
def compute_loss(weights: list, data: list, reg_lambda: float) -> float:
    mse = 0.0
    for i in range(len(data)):
        mse += (weights[0] * data[i] - data[i]) ** 2
    reg = reg_lambda * sum(w ** 2 for w in weights)
    return mse + reg
''',
        "output": "Energy decreased by 5%. Regularization stabilized training. loss = 0.028.",
        "is_correct": True,
        "gate_energy": True,
        "gate_time": True,
        "gate_memory": True,
        "ground_truth": {"energy_before": 0.0295, "energy_after": 0.028, "loss": 0.028},
    })

    hypotheses.append({
        "name": "Correct #4: Gradient clipping",
        "code": '''
def clip_gradient(grad: float, max_norm: float) -> float:
    norm = abs(grad)
    if norm > max_norm:
        return grad * (max_norm / norm)
    return grad
''',
        "output": "Energy decreased by 3%. Gradient clipping prevented explosion. loss = 0.052.",
        "is_correct": True,
        "gate_energy": True,
        "gate_time": True,
        "gate_memory": True,
        "ground_truth": {"energy_before": 0.0536, "energy_after": 0.052, "loss": 0.052},
    })

    hypotheses.append({
        "name": "Correct #5: Data augmentation",
        "code": '''
def augment(data: list, noise_scale: float) -> list:
    augmented = []
    for i in range(len(data)):
        augmented.append(data[i] + noise_scale * 0.1)
    return augmented
''',
        "output": "Energy decreased by 7%. Data augmentation improved generalization. loss = 0.039.",
        "is_correct": True,
        "gate_energy": True,
        "gate_time": True,
        "gate_memory": True,
        "ground_truth": {"energy_before": 0.0419, "energy_after": 0.039, "loss": 0.039},
    })

    hypotheses.append({
        "name": "Correct #6: Weight initialization",
        "code": '''
def init_weights(n_dims: int, scale: float) -> list:
    weights = []
    for i in range(n_dims):
        weights.append(scale / n_dims)
    return weights
''',
        "output": "Energy decreased by 15%. Better initialization converges faster. loss = 0.022.",
        "is_correct": True,
        "gate_energy": True,
        "gate_time": True,
        "gate_memory": True,
        "ground_truth": {"energy_before": 0.0259, "energy_after": 0.022, "loss": 0.022},
    })

    hypotheses.append({
        "name": "Correct #7: Early stopping",
        "code": '''
def should_stop(losses: list, patience: int) -> bool:
    if len(losses) < patience:
        return False
    recent = losses[-patience:]
    return all(recent[i] >= recent[i - 1] for i in range(1, len(recent)))
''',
        "output": "Energy decreased by 2%. Early stopping prevented overfitting. loss = 0.045.",
        "is_correct": True,
        "gate_energy": True,
        "gate_time": True,
        "gate_memory": True,
        "ground_truth": {"energy_before": 0.0459, "energy_after": 0.045, "loss": 0.045},
    })

    hypotheses.append({
        "name": "Correct #8: Momentum optimizer",
        "code": '''
def momentum_update(param: float, grad: float, velocity: float, lr: float, beta: float) -> tuple:
    new_velocity = beta * velocity + grad
    new_param = param - lr * new_velocity
    return new_param, new_velocity
''',
        "output": "Energy decreased by 10%. Momentum smoothed convergence. loss = 0.031.",
        "is_correct": True,
        "gate_energy": True,
        "gate_time": True,
        "gate_memory": True,
        "ground_truth": {"energy_before": 0.0344, "energy_after": 0.031, "loss": 0.031},
    })

    hypotheses.append({
        "name": "Correct #9: Feature normalization",
        "code": '''
def normalize(data: list) -> list:
    mean_val = sum(data) / len(data)
    std_val = (sum((x - mean_val) ** 2 for x in data) / len(data)) ** 0.5
    if std_val == 0:
        return [0.0 for _ in data]
    return [(x - mean_val) / std_val for x in data]
''',
        "output": "Energy decreased by 18%. Normalization improved convergence. loss = 0.019.",
        "is_correct": True,
        "gate_energy": True,
        "gate_time": True,
        "gate_memory": True,
        "ground_truth": {"energy_before": 0.0232, "energy_after": 0.019, "loss": 0.019},
    })

    hypotheses.append({
        "name": "Correct #10: Dropout regularization",
        "code": '''
def apply_dropout(activations: list, drop_rate: float, training: bool) -> list:
    if not training:
        return activations
    result = []
    for i in range(len(activations)):
        keep = 1.0 if (hash(i) % 100) >= (drop_rate * 100) else 0.0
        result.append(activations[i] * keep / (1.0 - drop_rate))
    return result
''',
        "output": "Energy decreased by 6%. Dropout reduced overfitting. loss = 0.037.",
        "is_correct": True,
        "gate_energy": True,
        "gate_time": True,
        "gate_memory": True,
        "ground_truth": {"energy_before": 0.0394, "energy_after": 0.037, "loss": 0.037},
    })

    # ===== 10 BOGUS HYPOTHESES =====

    hypotheses.append({
        "name": "Bogus #1: Hard-coded result",
        "code": '''
def train_step(params: dict, lr: float, data: list) -> float:
    return 0.001
''',
        "output": "Energy decreased by 95%. Final loss: 0.001. Massive improvement achieved.",
        "is_correct": False,
        "gate_energy": True,  # Energy metric looks great!
        "gate_time": True,
        "gate_memory": True,
        "ground_truth": {"energy_before": 0.05, "energy_after": 0.05, "loss": 0.05},
    })

    hypotheses.append({
        "name": "Bogus #2: Wrong metric measured",
        "code": '''
def evaluate(model: dict, test_data: list) -> float:
    train_loss = model.get("train_loss", 0.0)
    return train_loss
''',
        "output": "Energy decreased by 40%. loss = 0.012. Test performance improved.",
        "is_correct": False,
        "gate_energy": True,
        "gate_time": True,
        "gate_memory": True,
        "ground_truth": {"energy_before": 0.05, "energy_after": 0.048, "loss": 0.048},
    })

    hypotheses.append({
        "name": "Bogus #3: Inflated improvement claim",
        "code": '''
def compute_loss(data: list, weights: list) -> float:
    total = 0.0
    for i in range(len(data)):
        total += (data[i] - weights[0]) ** 2
    return total / len(data)
''',
        "output": "Energy decreased by 200%. loss = 0.01. Breakthrough result.",
        "is_correct": False,
        "gate_energy": True,
        "gate_time": True,
        "gate_memory": True,
        "ground_truth": {"energy_before": 0.05, "energy_after": 0.045, "loss": 0.045},
    })

    hypotheses.append({
        "name": "Bogus #4: Return type mismatch",
        "code": '''
def compute_energy(state: list) -> int:
    return "not_a_number"
''',
        "output": "Energy decreased by 10%. Energy is now lower. loss = 0.03.",
        "is_correct": False,
        "gate_energy": True,
        "gate_time": True,
        "gate_memory": True,
        "ground_truth": {"loss": 0.05},
    })

    hypotheses.append({
        "name": "Bogus #5: Uninitialized variable",
        "code": '''
def train(data: list, lr: float) -> float:
    loss = sum(d ** 2 for d in data) + regularizer
    return loss
''',
        "output": "Energy decreased by 15%. Regularization helped. loss = 0.025.",
        "is_correct": False,
        "gate_energy": True,
        "gate_time": True,
        "gate_memory": True,
        "ground_truth": {"energy_before": 0.05, "energy_after": 0.045, "loss": 0.045},
    })

    hypotheses.append({
        "name": "Bogus #6: Division by zero hiding",
        "code": '''
def safe_divide(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b

def compute_metric(data: list) -> float:
    total = 0.0
    for i in range(len(data)):
        total += safe_divide(data[i], 0)
    return total
''',
        "output": "Energy decreased by 50%. All metrics zero. loss = 0.0.",
        "is_correct": False,
        "gate_energy": True,  # Zero energy looks like perfection!
        "gate_time": True,
        "gate_memory": True,
        "ground_truth": {"energy_before": 0.05, "energy_after": 0.05, "loss": 0.05},
    })

    hypotheses.append({
        "name": "Bogus #7: Contradictory output claims",
        "code": '''
def train_step(x: float, lr: float) -> float:
    return x - lr * x
''',
        "output": "Energy decreased by 20%. Energy increased slightly. loss = 0.04. The model diverged but converged.",
        "is_correct": False,
        "gate_energy": True,
        "gate_time": True,
        "gate_memory": True,
        "ground_truth": {"loss": 0.05},
    })

    hypotheses.append({
        "name": "Bogus #8: Negative loss claimed",
        "code": '''
def compute_loss(data: list) -> float:
    return -sum(d ** 2 for d in data)
''',
        "output": "Energy decreased by 30%. loss = -0.05. Achieved negative loss.",
        "is_correct": False,
        "gate_energy": True,
        "gate_time": True,
        "gate_memory": True,
        "ground_truth": {"loss": 0.05},
    })

    hypotheses.append({
        "name": "Bogus #9: Impossible speedup",
        "code": '''
def fast_train(data: list) -> float:
    return 0.0
''',
        "output": "Energy decreased by 10%. 100x speedup. loss = 0.01.",
        "is_correct": False,
        "gate_energy": True,
        "gate_time": True,
        "gate_memory": True,
        "ground_truth": {"energy_before": 0.05, "energy_after": 0.048, "loss": 0.048},
    })

    hypotheses.append({
        "name": "Bogus #10: Syntax error in code",
        "code": '''
def train(data list) -> float:
    return sum(data
''',
        "output": "Energy decreased by 5%. Training completed. loss = 0.04.",
        "is_correct": False,
        "gate_energy": True,
        "gate_time": True,
        "gate_memory": True,
        "ground_truth": {"energy_before": 0.05, "energy_after": 0.048, "loss": 0.048},
    })

    return hypotheses


# ---------------------------------------------------------------------------
# 4. Gate evaluation — compare 3-gate vs 4-gate
# ---------------------------------------------------------------------------

def evaluate_gates(hypotheses: list[dict[str, Any]]) -> dict[str, Any]:
    """Run verification pipeline on all hypotheses and evaluate gate combinations.

    **Detailed explanation for engineers:**
        For each hypothesis, computes:
        - 3-gate decision: accept if gate_energy AND gate_time AND gate_memory
        - 4-gate decision: accept if 3-gate AND Ising constraint satisfaction

        Then builds a confusion matrix for each gate combination:
        - Rows: gate combination (3-gate, 4-gate, each individual gate)
        - Columns: true positives, false positives, true negatives, false negatives

        The key metric is **false acceptance rate** (FAR): how often does each
        gate combination accept a bogus hypothesis? The fourth gate should
        reduce FAR by catching bogus hypotheses that pass the three existing
        gates.

    Returns:
        Dict with per-hypothesis results and confusion matrices.
    """
    results: list[dict[str, Any]] = []

    for hyp in hypotheses:
        # Extract claims.
        claims = extract_hypothesis_claims(hyp["code"], hyp["output"])

        # Verify claims.
        verification = verify_hypothesis(claims, hyp.get("ground_truth"))

        # Ising gate: pass if no violations.
        ising_pass = verification["violated"] == 0

        # 3-gate decision.
        three_gate = hyp["gate_energy"] and hyp["gate_time"] and hyp["gate_memory"]

        # 4-gate decision.
        four_gate = three_gate and ising_pass

        results.append({
            "name": hyp["name"],
            "is_correct": hyp["is_correct"],
            "gate_energy": hyp["gate_energy"],
            "gate_time": hyp["gate_time"],
            "gate_memory": hyp["gate_memory"],
            "gate_ising": ising_pass,
            "three_gate_accept": three_gate,
            "four_gate_accept": four_gate,
            "n_claims": len(claims),
            "verified": verification["verified"],
            "violated": verification["violated"],
            "uncovered": verification["uncovered"],
            "certificate": verification["certificate"],
        })

    return {"results": results}


def print_confusion_matrix(results: list[dict[str, Any]]) -> None:
    """Print confusion matrix for gate combinations.

    **Detailed explanation for engineers:**
        Prints a table showing, for each gate combination:
        - TP: correctly accepted (correct hypothesis, gate says accept)
        - FP: falsely accepted (bogus hypothesis, gate says accept) — BAD
        - TN: correctly rejected (bogus hypothesis, gate says reject) — GOOD
        - FN: falsely rejected (correct hypothesis, gate says reject)
        - FAR: false acceptance rate = FP / (FP + TN)
        - FRR: false rejection rate = FN / (FN + TP)

        The goal is to show that 4-gate has lower FAR than 3-gate.
    """
    gate_combos = {
        "Energy only": lambda r: r["gate_energy"],
        "Time only": lambda r: r["gate_time"],
        "Memory only": lambda r: r["gate_memory"],
        "Ising only": lambda r: r["gate_ising"],
        "3-gate (E+T+M)": lambda r: r["three_gate_accept"],
        "4-gate (E+T+M+I)": lambda r: r["four_gate_accept"],
    }

    print("\n" + "=" * 78)
    print("CONFUSION MATRIX: Gate Combination × Accept/Reject × Correct/Bogus")
    print("=" * 78)
    print(f"  {'Gate Combination':<22s}  {'TP':>4s}  {'FP':>4s}  {'TN':>4s}  {'FN':>4s}  {'FAR':>6s}  {'FRR':>6s}")
    print("-" * 78)

    for name, gate_fn in gate_combos.items():
        tp = sum(1 for r in results if r["is_correct"] and gate_fn(r))
        fp = sum(1 for r in results if not r["is_correct"] and gate_fn(r))
        tn = sum(1 for r in results if not r["is_correct"] and not gate_fn(r))
        fn = sum(1 for r in results if r["is_correct"] and not gate_fn(r))

        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        print(f"  {name:<22s}  {tp:>4d}  {fp:>4d}  {tn:>4d}  {fn:>4d}  {far:>5.1%}  {frr:>5.1%}")

    print("=" * 78)
    print("  TP = correct hypothesis accepted (good)")
    print("  FP = bogus hypothesis accepted (BAD — want zero)")
    print("  TN = bogus hypothesis rejected (good)")
    print("  FN = correct hypothesis rejected (acceptable if rare)")
    print("  FAR = false acceptance rate = FP/(FP+TN)")
    print("  FRR = false rejection rate = FN/(FN+TP)")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run the autoresearch self-verification experiment."""
    print("=" * 78)
    print("EXPERIMENT 72: Autoresearch Self-Verification via Ising Constraint Gate")
    print("  Dog-food the constraint pipeline on autoresearch's own outputs")
    print("=" * 78)

    start = time.time()

    # Step 1: Generate mock hypotheses.
    hypotheses = get_mock_hypotheses()
    print(f"\n  Generated {len(hypotheses)} mock hypotheses "
          f"({sum(1 for h in hypotheses if h['is_correct'])} correct, "
          f"{sum(1 for h in hypotheses if not h['is_correct'])} bogus)")

    # Step 2: Run the verification pipeline on each.
    print("\n  Running verification pipeline...")
    eval_result = evaluate_gates(hypotheses)
    results = eval_result["results"]

    # Step 3: Print per-hypothesis results.
    print("\n" + "-" * 78)
    print("PER-HYPOTHESIS RESULTS")
    print("-" * 78)

    for r in results:
        correct_str = "CORRECT" if r["is_correct"] else "BOGUS  "
        ising_str = "✓" if r["gate_ising"] else "✗"
        three_str = "✓" if r["three_gate_accept"] else "✗"
        four_str = "✓" if r["four_gate_accept"] else "✗"
        verdict = r["certificate"]["verdict"]

        # Did the 4-gate make the right call?
        four_gate_correct = (r["four_gate_accept"] == r["is_correct"])
        call_str = "✓" if four_gate_correct else "✗"

        print(f"  [{correct_str}] {r['name']:<45s}")
        print(f"    Claims: {r['n_claims']} "
              f"(verified={r['verified']}, violated={r['violated']}, "
              f"uncovered={r['uncovered']})")
        print(f"    3-gate: {three_str}  Ising: {ising_str}  "
              f"4-gate: {four_str}  Verdict: {verdict}  "
              f"Correct call: {call_str}")

        # Show violated claims for bogus hypotheses.
        if r["violated"] > 0:
            for d in r["certificate"]["details"]:
                if d["satisfied"] is False:
                    print(f"      ✗ {d['claim_type']}: {d['claim_text'][:60]}")

    # Step 4: Print confusion matrix.
    print_confusion_matrix(results)

    # Step 5: Compute summary statistics.
    n_correct = sum(1 for r in results if r["is_correct"])
    n_bogus = sum(1 for r in results if not r["is_correct"])

    three_gate_far = sum(
        1 for r in results if not r["is_correct"] and r["three_gate_accept"]
    ) / max(n_bogus, 1)
    four_gate_far = sum(
        1 for r in results if not r["is_correct"] and r["four_gate_accept"]
    ) / max(n_bogus, 1)

    three_gate_frr = sum(
        1 for r in results if r["is_correct"] and not r["three_gate_accept"]
    ) / max(n_correct, 1)
    four_gate_frr = sum(
        1 for r in results if r["is_correct"] and not r["four_gate_accept"]
    ) / max(n_correct, 1)

    ising_catches = sum(
        1 for r in results
        if not r["is_correct"] and r["three_gate_accept"] and not r["gate_ising"]
    )

    elapsed = time.time() - start

    print(f"\n{'=' * 78}")
    print(f"EXPERIMENT 72 RESULTS ({elapsed:.1f}s)")
    print(f"{'=' * 78}")
    print(f"  Hypotheses: {len(results)} ({n_correct} correct, {n_bogus} bogus)")
    print(f"")
    print(f"  3-gate (Energy+Time+Memory):")
    print(f"    False acceptance rate (FAR): {three_gate_far:.1%}")
    print(f"    False rejection rate (FRR):  {three_gate_frr:.1%}")
    print(f"")
    print(f"  4-gate (Energy+Time+Memory+Ising):")
    print(f"    False acceptance rate (FAR): {four_gate_far:.1%}")
    print(f"    False rejection rate (FRR):  {four_gate_frr:.1%}")
    print(f"")
    print(f"  Ising gate catches {ising_catches}/{n_bogus} bogus hypotheses "
          f"that 3-gate missed")
    print(f"  FAR reduction: {three_gate_far:.1%} → {four_gate_far:.1%}")

    if four_gate_far < three_gate_far:
        print(f"\n  VERDICT: ✅ Fourth gate reduces false acceptance rate!")
    elif four_gate_far == three_gate_far:
        print(f"\n  VERDICT: ⚠️ Fourth gate did not reduce FAR (same as 3-gate)")
    else:
        print(f"\n  VERDICT: ❌ Fourth gate increased FAR (unexpected)")

    if four_gate_frr <= 0.1:
        print(f"  FRR check: ✅ False rejection rate acceptable ({four_gate_frr:.1%})")
    else:
        print(f"  FRR check: ⚠️ False rejection rate high ({four_gate_frr:.1%})")

    print(f"{'=' * 78}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
